import math
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import time
import av
import matplotlib
import numpy as np
import pandas as pd
import pygfx as gfx
from PyQt5.QtCore import QTimer, Qt, QElapsedTimer, QIODevice, QByteArray
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QGridLayout, QVBoxLayout, QHBoxLayout,
    QSlider, QPushButton, QDialog, QAction, QInputDialog,
    QSizePolicy, QShortcut)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pymediainfo import MediaInfo
from PyQt5.QtGui import QKeySequence
matplotlib.use("Qt5Agg")

#***********************************************
 #CONFIG
# MAJOR.MINOR.PATCH
__version__ = "1.1 Fixes"
INSV1="data/VID_20260221_091717_00_050.insv"
INSV2="data/VID_20260221_091717_00_051.insv"
MERGED_DATA = INSV1+".merged_data.csv"
VIDEO1="data/video1.mp4"
VIDEO2="data/video2.mp4"
BOOKMARK_FILE=INSV1+".bookmark.csv"
STL_FILE="data/CAP10.STL"
BOX = 0.007*1.5 # taille box vision en °latitude
DF_FREQ = 100
TRACE = 9000 # taille de la trace 6000=1 minute
TRACE_DEFAULT = TRACE
TRACE_BEFORE = 500 # position précédente, 500 avant soit 5s
TRACE_SLICING_FACTOR = 50
VITESSE_MISE_EN_LIGNE = 80 #km/h
PITCH_MONTAGE_PAR_DEFAUT = 15 #camera verticale au repos par défaut, ecran face à soi, légerement inclinée vers soi
R_recalage_repere=3 # données issues BB
#R_recalage_repere=1 # données issues computed VQF
refcam=[0,0,1] # données issues de BB
#refcam=[0,0,-1] # données issues computed VQF



# ======================================================
# MP4 creation date (UTC)
# ======================================================
def get_mp4_creation_datetime(path):
    media_info = MediaInfo.parse(path)
    for track in media_info.tracks:
        if track.track_type == "General" and track.encoded_date:
            s = track.encoded_date.strip().replace("UTC", "").strip()
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
    raise RuntimeError(f"Date de création MP4 introuvable : {path}")

# ======================================================
# DataFrame
# ======================================================
df = pd.read_csv(MERGED_DATA, low_memory=False)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
frames_df = len(df)
print(MERGED_DATA," START @",df['timestamp'][0])
print(MERGED_DATA," END @",df['timestamp'].iat[-1])
print(MERGED_DATA," FRAMES = ", frames_df)

# ---- numpy caches for fast access inside the realtime loop ----
gps_lat_vals = df["gps_lat"].to_numpy()
gps_lon_vals = df["gps_lon"].to_numpy()
gps_alt_vals = df["gps_alt"].to_numpy()
timestamp_vals = df["timestamp"].to_numpy()

# ======================================================
# VIDEO ANALYSIS
# ======================================================
container_probe = av.open(VIDEO1)
stream_probe = container_probe.streams.video[0]

frames_video = stream_probe.frames
print(VIDEO1," FRAMES = ", frames_video)

N = frames_video
container_probe.close()

# ======================================================
# TOP remarquables
# ======================================================
mask = df['gps_speed'] > VITESSE_MISE_EN_LIGNE
index_enligne_devol= mask.idxmax()
mask = df['gps_alt'] > 3000
index_entree_3000= mask.idxmax()

# ======================================================
# USEFUL FUNCTIONS
# ======================================================
def quat_to_rot(q):
    w, x, y, z = q;x = -x;y = -y;z = -z
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])

perm = np.array([
[[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]],
[[ 1, 0, 0],[ 0, 0, 1],[ 0,-1, 0]],
[[ 1, 0, 0],[ 0,-1, 0],[ 0, 0,-1]],
[[ 1, 0, 0],[ 0, 0,-1],[ 0, 1, 0]],
[[-1, 0, 0],[ 0, 1, 0],[ 0, 0,-1]],
[[-1, 0, 0],[ 0, 0, 1],[ 0, 1, 0]],
[[-1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]],
[[-1, 0, 0],[ 0, 0,-1],[ 0,-1, 0]],
[[ 0, 1, 0],[ 1, 0, 0],[ 0, 0,-1]],
[[ 0, 1, 0],[ 0, 0, 1],[ 1, 0, 0]],
[[ 0, 1, 0],[-1, 0, 0],[ 0, 0, 1]],
[[ 0, 1, 0],[ 0, 0,-1],[-1, 0, 0]],
[[ 0,-1, 0],[ 1, 0, 0],[ 0, 0, 1]],
[[ 0,-1, 0],[ 0, 0, 1],[-1, 0, 0]],
[[ 0,-1, 0],[-1, 0, 0],[ 0, 0,-1]],
[[ 0,-1, 0],[ 0, 0,-1],[ 1, 0, 0]],
[[ 0, 0, 1],[ 1, 0, 0],[ 0, 1, 0]],
[[ 0, 0, 1],[ 0, 1, 0],[-1, 0, 0]],
[[ 0, 0, 1],[-1, 0, 0],[ 0,-1, 0]],
[[ 0, 0, 1],[ 0,-1, 0],[ 1, 0, 0]],
[[ 0, 0,-1],[ 1, 0, 0],[ 0,-1, 0]],
[[ 0, 0,-1],[ 0, 1, 0],[ 1, 0, 0]],
[[ 0, 0,-1],[-1, 0, 0],[ 0, 1, 0]],
[[ 0, 0,-1],[ 0,-1, 0],[-1, 0, 0]]])



def angle_between(u, v):
    u = np.array(u);
    v = np.array(v)
    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u);
    norm_v = np.linalg.norm(v)
    cos_theta = dot / (norm_u * norm_v);
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # sécurité numérique
    return np.degrees(np.arccos(cos_theta))

# ======================================================
# Main Window
class ArtificialHorizon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")
        self.pitch = 0.0
        self.bank = 0.0
        self.heading = 0.0
        # optional aerobatic triangle marker (used for wingtip reference)
        self.show_triangle = False

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QPen, QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width();h = self.height()
        cx = w // 2;cy = h // 2

        painter.fillRect(self.rect(), Qt.transparent)
        painter.translate(cx, cy)
        painter.rotate(-self.bank)

        pitch_scale = 3
        pitch_offset = int(self.pitch * pitch_scale)

        painter.setBrush(QColor(80, 160, 255))
        painter.setPen(Qt.NoPen)

        # Large surfaces so the horizon never leaves the screen even inverted
        size = max(w, h) * 6

        # sky
        painter.drawRect(-size, -size + pitch_offset, size * 2, size)

        # ground
        painter.setBrush(QColor(160, 100, 40))
        painter.drawRect(-size, pitch_offset, size * 2, size)

        pen = QPen(QColor("white"))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(-w, pitch_offset, w, pitch_offset)

        # ---- Pitch reference lines (±10° ±20° ±30°) with EFIS ticks ----
        pen_ref = QPen(QColor("white"))
        pen_ref.setWidth(1)
        painter.setPen(pen_ref)

        for deg in (10, 20, 30,45, 60, 75, 90):
            offset = int(deg * pitch_scale)

            # longer line for 10°, shorter for others
            if deg <= 10:
                half = w // 6
            else:
                half = w // 12

            # +deg line
            y = pitch_offset - offset
            painter.drawLine(-half, y, half, y)

            # EFIS side ticks
            painter.drawLine(-half - 10, y, -half, y)
            painter.drawLine(half, y, half + 10, y)

            # numeric labels
            painter.drawText(-half - 40, y + 5, f"{deg}")
            painter.drawText(half + 15, y + 5, f"{deg}")

            # -deg line
            y = pitch_offset + offset
            painter.drawLine(-half, y, half, y)

            # EFIS side ticks
            painter.drawLine(-half - 10, y, -half, y)
            painter.drawLine(half, y, half + 10, y)

            # numeric labels
            painter.drawText(-half - 45, y + 5, f"-{deg}")
            painter.drawText(half + 15, y + 5, f"-{deg}")

        painter.resetTransform()

        if not self.show_triangle:
            pen = QPen(QColor("yellow"))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(cx - 30, cy, cx + 30, cy)
            painter.drawLine(cx, cy - 10, cx, cy + 10)

        # ---- Aerobatic reference triangle (used for wingtip horizon) ---
        if self.show_triangle:
            from PyQt5.QtGui import QPolygon
            from PyQt5.QtCore import QPoint
            size = 27
            triangle = QPolygon([
                QPoint(cx, cy),
                QPoint(cx, cy - size),
                QPoint(cx - size, cy ),
                QPoint(cx + size, cy ),
                QPoint(cx, cy - size)
            ])
            painter.setBrush(Qt.NoBrush)  # pas de remplissage
            pen = QPen(QColor("yellow"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawPolygon(triangle)

        painter.end()

# ======================================================
# Main Window
# ======================================================
class MainWindow(QMainWindow):
    def trace_plus(self):
        global TRACE
        TRACE += 2000

    def trace_minus(self):
        global TRACE
        if TRACE - 2000 >= 1000:
            TRACE -= 2000

    def reset_trace(self):
        global TRACE
        TRACE = TRACE_DEFAULT
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABView Version "+__version__)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(1600, 1000)

        # ---- état ----
        self.i = 0
        self.idf = 0
        self.playing = True
        self.speed = 1
        self.current_video_time_utc = None
        self.frame_skipped_count = 0
        self.frame_last_delay = 0

        # ---- vidéos ----
        self.video1_path=VIDEO1
        self.video2_path=VIDEO2
        self.container1 = av.open(self.video1_path)
        self.container2 = av.open(self.video2_path)

        # ---- audio (video1) ----
        try:
            self.audio_container = av.open(self.video1_path)
            self.audio_stream = self.audio_container.streams.audio[0]
            # packet based audio demux (more stable than frame iterator)
            self.audio_packets = self.audio_container.demux(self.audio_stream)

            # force audio format compatible with Qt (stereo s16 interleaved)
            self.audio_rate = self.audio_stream.rate
            self.audio_channels = 2

            self.audio_resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="stereo",
                rate=self.audio_rate,
            )

            fmt = QAudioFormat()
            fmt.setSampleRate(self.audio_rate)
            fmt.setChannelCount(self.audio_channels)
            fmt.setSampleSize(16)
            fmt.setCodec("audio/pcm")
            fmt.setByteOrder(QAudioFormat.LittleEndian)
            fmt.setSampleType(QAudioFormat.SignedInt)

            self.audio_output = QAudioOutput(fmt)
            self.audio_device = self.audio_output.start()

            self.audio_buffer = QByteArray()
            # ~2 seconds buffer target (rate * channels * bytes_per_sample * seconds)
            self.audio_buffer_target = self.audio_rate * self.audio_channels * 2 * 2  # ~2 seconds buffer (rate * channels * bytes_per_sample * seconds)
            # start playback only once buffer is sufficiently filled
            self.audio_started = False
            self.audio_clock_sec = 0.0  # audio clock based on decoded PTS
        except Exception:
            self.audio_stream = None

        self.stream1 = self.container1.streams.video[0]
        self.stream2 = self.container2.streams.video[0]
        self.stream1.thread_type = "AUTO"
        self.stream2.thread_type = "AUTO"

        self.decoder1 = self.container1.decode(self.stream1)
        self.decoder2 = self.container2.decode(self.stream2)
        self.video1_start = get_mp4_creation_datetime(self.video1_path)
        self.video2_start = get_mp4_creation_datetime(self.video2_path)
        self.video_df_offset = df.timestamp.iloc[0] - self.video1_start # 🔑 OFFSET TEMPOREL (clé du problème)

        # ---- Init de base pour INU/GFX
        self.g_min = float("inf")
        self.g_max = float("-inf")
        self.montage_pitch_angle = PITCH_MONTAGE_PAR_DEFAUT #camera vericale au repos par défaut, ecran face à soi
        self.gs_max = float("-inf")
        # ---- bookmarks ----
        self.bookmarks = []
        self.bookmarks_df = None
        self.last_bookmark_frame = None
        self.bookmark_overlay = None

        # ---- init for gpsmatplotlib
        self.firstGPS = True
        self.last_azim = 0
        self.fixed_elev = 20  # angle d'inclinaison verrouillé
        self.elev_locked = True
        self.box_zoom = 1.0
        self.update_zoom_menu() # Affichage initial du facteur de zoom dans le menu Settings

        self.init_UI()

        self.init_map_OSM_widget()
        self.map_view.loadFinished.connect(self.on_map_loaded)

        self.init_gps_matplotlib()

        self.init_gfx()
        # self.calibrate_gfx(index_enligne_devol)

        # ---- realtime timer (compensated loop) ----
        self.target_fps = 30
        self.frame_period_ms = 1000 / self.target_fps

        self.clock = QElapsedTimer()
        self.clock.start()

        # absolute frame scheduling to avoid timing drift
        self.next_frame_time = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.main_loop)
        self.timer.start(0)  # controlled manually

        # ---- dedicated audio timer (runs faster than video loop) ----
        if self.audio_stream is not None:
            self.audio_timer = QTimer()
            self.audio_timer.timeout.connect(self.update_audio)
            self.audio_timer.start(10)  # ~100 Hz audio servicing

    def init_UI(self):
        # ---- UI ----
        central = QWidget()
        #central.setStyleSheet("background-color: white;")  # fond gris
        self.setCentralWidget(central)
        self.layout = QVBoxLayout(central)

        # ---- Menu ----
        menubar = self.menuBar()

        menu_fichier = menubar.addMenu("Fichier")
        menu_lecture = menubar.addMenu("Lecture")
        menu_navigation = menubar.addMenu("Navigation")
        self.menu_bookmarks = menubar.addMenu("Bookmarks")
        menu_settings = menubar.addMenu("Settings")

        # Actions
        act_quitter = QAction("Quitter", self)
        act_quitter.setShortcut("Ctrl+Q")
        act_quitter.triggered.connect(self.close)

        act_play_pause = QAction("Lecture / Pause", self)
        act_play_pause.setShortcut("Space")
        act_play_pause.triggered.connect(self.toggle_play)

        # --- Action Start ---
        act_start = QAction("Start", self)
        act_start.setShortcut("Home")
        act_start.triggered.connect(self.goto_start)

        act_mise_en_ligne = QAction("Mise en ligne", self)
        act_mise_en_ligne.setShortcut("Ctrl+M")
        act_mise_en_ligne.triggered.connect(self.goto_mise_en_ligne)

        act_entree_box = QAction("Entrée BOX", self)
        act_entree_box.setShortcut("Ctrl+B")
        act_entree_box.triggered.connect(self.goto_entree_box)


        # Ajout des actions aux menus
        menu_fichier.addAction(act_quitter)
        menu_lecture.addAction(act_play_pause)
        menu_navigation.addAction(act_start)
        menu_navigation.addSeparator()
        menu_navigation.addAction(act_mise_en_ligne)
        menu_navigation.addAction(act_entree_box)

        # ---- Reload bookmarks CSV ----
        self.act_reload_bookmarks = QAction("Recharger CSV", self)
        self.act_reload_bookmarks.triggered.connect(self.reload_bookmarks)

        # ---- Add bookmark ----
        self.act_add_bookmark = QAction("Ajouter Bookmark", self)
        self.act_add_bookmark.setShortcut("Ctrl+D")

        # Force Ctrl+D to work even when QWebEngineView or other widgets capture the keyboard
        self.shortcut_add_bookmark = QShortcut(QKeySequence("Ctrl+D"), self)
        self.shortcut_add_bookmark.activated.connect(self.add_bookmark)

        self.act_add_bookmark.triggered.connect(self.add_bookmark)

        # build base menu structure
        self.menu_bookmarks.addAction(self.act_reload_bookmarks)
        self.menu_bookmarks.addSeparator()
        self.menu_bookmarks.addAction(self.act_add_bookmark)
        self.menu_bookmarks.addSeparator()
        self.load_bookmarks()
        # ---- Ensure keyboard shortcuts work regardless of focused widget ----
        for act in self.findChildren(QAction):
            act.setShortcutContext(Qt.ApplicationShortcut)

        self.grid = QGridLayout()
        self.layout.addLayout(self.grid)
        self.grid.setRowStretch(0, 2)
        self.grid.setRowStretch(1, 2)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(2, 1)
        self.grid.setColumnStretch(3, 1)

        self.video1 = QLabel(alignment=Qt.AlignCenter)
        self.video2 = QLabel(alignment=Qt.AlignCenter)
        # ---- GPS heading overlay on video1 (top center, text sized) ----
        self.video1_heading_label = QLabel("", self.video1)
        self.video1_heading_label.setAlignment(Qt.AlignCenter)
        self.video1_heading_label.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 18px; font-weight: bold;"
        )
        self.video1_heading_label.adjustSize()
        self.video1_heading_label.raise_()
        # avoid expensive per-frame scaling; let Qt scale automatically
        self.video1.setScaledContents(True)
        self.video2.setScaledContents(True)
        # prevent QLabel from expanding to the raw video resolution
        self.video1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # give a reasonable default display size
        self.video1.setMinimumSize(320, 180)
        self.video2.setMinimumSize(320, 180)
        self.grid.addWidget(self.video1, 0, 0, 1, 2)  # colonnes 0-1
        self.grid.addWidget(self.video2, 0, 2, 1, 2)  # colonnes 2-3

        # Blocage élévation action (menu Settings)
        self.act_lock_elev = QAction("Bloquer GPS élévation", self)
        self.act_lock_elev.setCheckable(True)
        self.act_lock_elev.setChecked(self.elev_locked)
        self.act_lock_elev.triggered.connect(self.toggle_elev_lock)
        menu_settings.addAction(self.act_lock_elev)

        # Actions Zoom (menu Settings)
        self.act_zoom_out = QAction("Zoom -", self)
        self.act_zoom_out.triggered.connect(self.zoom_box_out)
        menu_settings.addAction(self.act_zoom_out)

        self.act_zoom_in = QAction("Zoom +", self)
        self.act_zoom_in.triggered.connect(self.zoom_box_in)
        menu_settings.addAction(self.act_zoom_in)

        self.act_zoom_reset = QAction("Reset Zoom", self)
        self.act_zoom_reset.triggered.connect(self.reset_zoom)
        menu_settings.addAction(self.act_zoom_reset)

        # Actions Trace (menu Settings)
        self.act_trace_minus = QAction("Trace -", self)
        self.act_trace_minus.triggered.connect(self.trace_minus)
        menu_settings.addAction(self.act_trace_minus)

        self.act_trace_plus = QAction("Trace +", self)
        self.act_trace_plus.triggered.connect(self.trace_plus)
        menu_settings.addAction(self.act_trace_plus)

        self.act_trace_reset = QAction("Reset Trace", self)
        self.act_trace_reset.triggered.connect(self.reset_trace)
        menu_settings.addAction(self.act_trace_reset)

        # ---- slider + controls ----
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, N - 1)
        self.slider.valueChanged.connect(self.on_slider)

        self.timestamp_label = QLabel(alignment=Qt.AlignCenter)

        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.clicked.connect(self.toggle_play)

        # ---- jump buttons (time navigation) ----
        self.btn_back_10 = QPushButton("⏪ 10s")
        self.btn_back_10.clicked.connect(self.jump_back_10s)

        self.btn_back_2 = QPushButton("◀ 2s")
        self.btn_back_2.clicked.connect(self.jump_back_2s)

        self.btn_fwd_2 = QPushButton("2s ▶")
        self.btn_fwd_2.clicked.connect(self.jump_fwd_2s)

        self.btn_fwd_10 = QPushButton("10s ⏩")
        self.btn_fwd_10.clicked.connect(self.jump_fwd_10s)

        self.btn_recalibrate = QPushButton("Recalibrer")
        self.btn_recalibrate.clicked.connect(self.calibrate_gfx_on_current_frame)

        self.btn_pallier = QPushButton("Palier")
        self.btn_pallier.clicked.connect(self.seek_palier)

        self.btn_add_bookmark = QPushButton("Bookmark")
        self.btn_add_bookmark.clicked.connect(self.add_bookmark)

        self.btn_start = QPushButton("⏮ Start")
        self.btn_start.clicked.connect(self.goto_start)

        self.btn_mise_en_ligne = QPushButton("Mise en ligne")
        self.btn_mise_en_ligne.clicked.connect(self.goto_mise_en_ligne)

        self.btn_entree_box = QPushButton("Entrée BOX")
        self.btn_entree_box.clicked.connect(self.goto_entree_box)

        self.btn_misedos_securite = QPushButton("Mise dos sécurité")
        self.btn_misedos_securite.clicked.connect(self.goto_misedos_securite)
        self.btn_enchainement = QPushButton("Enchaînement")
        self.btn_enchainement.clicked.connect(self.goto_enchainement)


        # self.btn_lock_elev = QPushButton("🔒 Elev")
        # self.btn_lock_elev.clicked.connect(self.toggle_elev_lock)

        self.btn_zoom_out = QPushButton("🔍 Zoom -")
        self.btn_zoom_out.clicked.connect(self.zoom_box_out)

        self.btn_zoom_in = QPushButton("🔍 Zoom +")
        self.btn_zoom_in.clicked.connect(self.zoom_box_in)

        self.btn_zoom_reset = QPushButton("🔄 Reset Zoom")
        self.btn_zoom_reset.clicked.connect(self.reset_zoom)

        self.btn_trace_minus = QPushButton("Trace -")
        self.btn_trace_minus.clicked.connect(self.trace_minus)

        self.btn_trace_plus = QPushButton("Trace +")
        self.btn_trace_plus.clicked.connect(self.trace_plus)

        self.btn_trace_reset = QPushButton("Reset Trace")
        self.btn_trace_reset.clicked.connect(self.reset_trace)

        self.btn_quitter = QPushButton("Quitter")
        self.btn_quitter.clicked.connect(self.close)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.btn_pause)
        buttons_layout.addWidget(self.btn_back_10)
        buttons_layout.addWidget(self.btn_back_2)
        buttons_layout.addWidget(self.btn_fwd_2)
        buttons_layout.addWidget(self.btn_fwd_10)
        buttons_layout.addWidget(self.btn_pallier)
        buttons_layout.addWidget(self.btn_recalibrate)
        buttons_layout.addWidget(self.btn_add_bookmark)
        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_mise_en_ligne)
        buttons_layout.addWidget(self.btn_entree_box)
        buttons_layout.addWidget(self.btn_misedos_securite)
        buttons_layout.addWidget(self.btn_enchainement)
        # buttons_layout.addWidget(self.btn_lock_elev)
        buttons_layout.addWidget(self.btn_zoom_out)
        buttons_layout.addWidget(self.btn_zoom_in)
        buttons_layout.addWidget(self.btn_zoom_reset)
        buttons_layout.addWidget(self.btn_trace_minus)
        buttons_layout.addWidget(self.btn_trace_plus)
        buttons_layout.addWidget(self.btn_trace_reset)
        buttons_layout.addWidget(self.btn_quitter)

        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.timestamp_label)
        self.layout.addLayout(buttons_layout)

        # ---- Bookmark overlay label (top center) ----
        self.bookmark_overlay = QLabel("", self)
        self.bookmark_overlay.setStyleSheet(
            "color: yellow; "
            "background-color: rgba(0,0,0,180); "
            "padding: 12px; "
            "font-family: 'Menlo'; "
            "font-size: 42px; "
            "font-weight: bold;"
        )
        self.bookmark_overlay.setAlignment(Qt.AlignCenter)
        self.bookmark_overlay.hide()

        # ---- matplotlib ----
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.grid.addWidget(self.canvas, 1, 2, 1, 1)
        # double-clic souris sur le graphe -> plein écran
        self.canvas.mpl_connect("button_press_event", self.on_matplotlib_double_click)

    def lock_elev(self, event):
        if not self.elev_locked:
            return
        if event.inaxes != self.ax:
            return
        # force uniquement l'elevation, laisse azim libre
        self.ax.view_init(elev=self.fixed_elev, azim=self.ax.azim)

    def toggle_elev_lock(self):
        self.elev_locked = not self.elev_locked

        if self.elev_locked:
            # réappliquer immédiatement l’élévation verrouillée
            self.ax.view_init(elev=self.fixed_elev, azim=self.ax.azim)

        # synchronisation avec le menu Settings
        if hasattr(self, "act_lock_elev"):
            self.act_lock_elev.setChecked(self.elev_locked)

    def zoom_box_out(self):
        """Zoom  : agrandit la box matplotlib"""
        self.box_zoom *= 1.25
        self.update_zoom_menu()

    def zoom_box_in(self):
        """Zoom  : agrandit la box matplotlib"""
        self.box_zoom /= 1.25
        self.update_zoom_menu()

    def reset_zoom(self):
        """Reset du zoom : remet la box matplotlib à la valeur par défaut"""
        self.box_zoom = 1.0
        self.update_zoom_menu()

    def update_zoom_menu(self):
        """Met à jour l'affichage du facteur de zoom dans le menu Settings"""
        factor = f"×{self.box_zoom:.2f}"
        if hasattr(self, "act_zoom_reset"):
            self.act_zoom_reset.setText(f"Reset Zoom (Actual = {factor})")

    def init_map_OSM_widget(self):
        # ---- OpenStreetMap (OSM) ----
        self.map_ready = False
        self.map_view = QWebEngineView()
        # HTML Leaflet avec OpenStreetMap
        lat0 = df.gps_lat.iloc[0]
        lon0 = df.gps_lon.iloc[0]
        self.map_view.setHtml(f"""
                <!DOCTYPE html>
                <html>
                <head>
                  <meta charset="utf-8" />
                  <title>OpenStreetMap</title>
                  <meta name="viewport" content="width=device-width, initial-scale=1.0">
                  <link
                    rel="stylesheet"
                    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
                  />
                  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
                  <style>
                    html, body, #map {{
                      height: 100%;
                      margin: 0;
                    }}
                  </style>
                </head>
                <body>
                  <div id="map"></div>
                  <script>
                    var map = L.map('map').setView([{lat0}, {lon0}], 11);

                    var bounds = L.latLng([{lat0}, {lon0}]).toBounds(8000);
                    map.fitBounds(bounds);

                    map.options.maxZoom = 11;
                    map.options.minZoom = 11;

                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                      maxZoom: 19,
                      attribution: '&copy; OpenStreetMap contributors'
                    }}).addTo(map);

                    var marker = L.marker([{lat0}, {lon0}]).addTo(map);
                    // ---- Aerobatic axis (extended 1 km each side) ----
                    var p1 = [43.4822, 3.845];
                    var p2 = [43.4689, 3.8209];

                    // direction vector
                    var dx = p2[1] - p1[1];
                    var dy = p2[0] - p1[0];
                    var len = Math.sqrt(dx*dx + dy*dy);

                    // normalize
                    var ux = dx / len;
                    var uy = dy / len;

                    // approx conversion: 1 km in degrees (~0.009° lat)
                    var extend = 0.009;

                    // extend both sides
                    var p1_ext = [p1[0] - uy*extend, p1[1] - ux*extend];
                    var p2_ext = [p2[0] + uy*extend, p2[1] + ux*extend];

                    // central aerobatic axis (purple)
                    var axis = L.polyline([
                      p1,
                      p2
                    ], {{
                      color: 'purple',
                      weight: 4
                    }}).addTo(map);

                    // extension before axis (gray)
                    var axis_ext1 = L.polyline([
                      p1_ext,
                      p1
                    ], {{
                      color: 'gray',
                      weight: 3
                    }}).addTo(map);

                    // extension after axis (gray)
                    var axis_ext2 = L.polyline([
                      p2,
                      p2_ext
                    ], {{
                      color: 'gray',
                      weight: 3
                    }}).addTo(map);

                    function updateMarker(lat, lon) {{
                      marker.setLatLng([lat, lon]);
                      map.panTo([lat, lon], {{ animate: false }});
                    }}

                    window.updateMarker = updateMarker;
                  </script>
                </body>
                </html>
                """)
        self.grid.addWidget(self.map_view, 1, 3, 1, 1)

    def init_gps_matplotlib(self):
        self.ax = self.fig.add_subplot(111, projection="3d")
        # verrouillage de l'elevation
        self.ax.view_init(elev=self.fixed_elev)
        self.canvas.mpl_connect("motion_notify_event", self.lock_elev)
        # réduction des marges autour de la figure 3D
        self.fig.subplots_adjust(
            left=0.02,
            right=0.98,
            bottom=0.02,
            top=0.98
        )

        # persistent artists (created once)
        # trajectoire 3D colorée (créée une seule fois)
        self.traj_collection = Line3DCollection([], cmap="viridis", linewidth=2)
        self.traj_collection.set_norm(Normalize(vmin=0, vmax=6000))  # altitude ft
        self.ax.add_collection(self.traj_collection)
        (self.line_ground,) = self.ax.plot([], [], [], color="gray")
        (self.line_traj_ground,) = self.ax.plot([], [], [], color="black", linestyle="dotted", lw=1)
        (self.line_traj_ground2,) = self.ax.plot([], [], [], color="gray", linestyle="dotted", lw=0.5)
        self.scatter_pt = self.ax.scatter([], [], [])
        # north vector (persistent artists)
        (self.north_line_main,) = self.ax.plot([], [], [], linestyle="dashdot", color="black")
        (self.north_line_left,) = self.ax.plot([], [], [], color="black")
        (self.north_line_right,) = self.ax.plot([], [], [], color="black")
        # HUD texts (screen-space, always horizontal)
        #self.text_alt = self.ax.text2D(0, 1, "", transform=self.ax.transAxes, fontsize=24, color="green", fontfamily="Menlo")
        #self.text_speed = self.ax.text2D(0, 0.90, "", transform=self.ax.transAxes, fontsize=24, color="red", fontfamily="Menlo")
        self.text_taille_box = self.ax.text2D(50, 5, f"Box Width / Trace Length", fontsize=10, color='gray', transform=None)
        self.ax.set_xlabel("Long");
        self.ax.set_ylabel("Lat");
        self.ax.set_zlabel("Z")
        # reduce tick label font size (graduations)
        self.ax.tick_params(axis="both", which="major", labelsize=5)
        self.ax.tick_params(axis="z", which="major", labelsize=10)
        # voltige 6860 = Axe de 2500m orienté au 050%230° centré sur 43°28'30"N,003°49'58"E.
        # lat = 43,475 N
        # long = 3,8328° E
        # 50/230 degrés 2500m
        self.ax.plot([3.845, 3.8209], [43.4822, 43.4689], 3000, linestyle='--', color='purple',linewidth=3)

        # autoscale au démarrage
        self.ax.autoscale()

    def init_gfx(self):
        # ---- pygfx ----
        self.gfx_display = gfx.Display()
        self.gfx_scene = gfx.Scene()



        # axes de repère local (X=rouge, Y=vert, Z=bleu) without arrow tips
        axes_len = 300
        x_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [axes_len, 0, 0]], dtype=np.float32))
        y_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [0, axes_len, 0]], dtype=np.float32))
        z_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [0, 0, axes_len]], dtype=np.float32))

        self.gfx_axes_x = gfx.Line(x_geom, gfx.LineMaterial(color="red", thickness=2, dash_pattern=(10, 10)))
        self.gfx_axes_y = gfx.Line(y_geom, gfx.LineMaterial(color="green", thickness=2, dash_pattern=(10, 10)))
        self.gfx_axes_z = gfx.Line(z_geom, gfx.LineMaterial(color="blue", thickness=2, dash_pattern=(10, 10)))

        self.gfx_scene.add(self.gfx_axes_x)
        self.gfx_scene.add(self.gfx_axes_y)
        self.gfx_scene.add(self.gfx_axes_z)

        light = gfx.DirectionalLight(intensity=3)
        light.local.position = (200, 200, 200)
        self.gfx_scene.add(light)
        ambient = gfx.AmbientLight(intensity=0.4)
        self.gfx_scene.add(ambient)

        # groupe parent (piloté par quaternion)
        self.gfx_object = gfx.Group()

        # vecteurs du repère local
        length = 400  # longueur des vecteurs
        x_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [length, 0, 0]], dtype=np.float32))
        self.gfx_vec_x = gfx.Line(x_geom, gfx.LineMaterial(color="red", thickness=1),)
        y_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [0, length, 0]], dtype=np.float32))
        self.gfx_vec_y = gfx.Line(
            y_geom,
            gfx.LineMaterial(color="green", thickness=8),
        )
        z_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [0, 0, length]], dtype=np.float32))
        self.gfx_vec_z = gfx.Line(z_geom, gfx.LineMaterial(color="blue", thickness=1),)
        self.gfx_object.add(self.gfx_vec_x);self.gfx_object.add(self.gfx_vec_y);self.gfx_object.add(self.gfx_vec_z)

        # ---- Arrow head for Y vector ----
        self.gfx_y_arrow = gfx.Mesh(
            gfx.cone_geometry(radius=25, height=60),
            gfx.MeshPhongMaterial(color="green"),
        )

        # place arrow at end of Y vector
        self.gfx_y_arrow.local.position = (0, length, 0)

        # rotate cone so it points along +Y (cone default axis is +Z)
        theta = -np.pi / 2
        qx = (
            np.sin(theta / 2),
            0.0,
            0.0,
            np.cos(theta / 2),
        )
        self.gfx_y_arrow.local.rotation = qx

        self.gfx_object.add(self.gfx_y_arrow)

        # ---- Acceleration vector (line, updated each frame) ----
        acc_positions = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
        self.acc_geom = gfx.Geometry(positions=acc_positions)
        self.gfx_vec_acc = gfx.Line(self.acc_geom,gfx.LineMaterial(color="green", thickness=8),)
        self.gfx_object.add(self.gfx_vec_acc)

        # ---- Arrow head for acceleration vector ----
        self.gfx_acc_arrow = gfx.Mesh(
            gfx.cone_geometry(radius=25, height=60),
            gfx.MeshPhongMaterial(color="green"),
        )
        self.gfx_acc_arrow.visible = True
        self.gfx_object.add(self.gfx_acc_arrow)

        # ---- G vector trail (history of acceleration direction) ----
        self.g_trail_len = 120  # number of stored points
        self.g_trail = np.zeros((self.g_trail_len, 3), dtype=np.float32)
        trail_geom = gfx.Geometry(positions=self.g_trail)
        self.gfx_g_trail = gfx.Line(
            trail_geom,
            gfx.LineMaterial(color="#ff8800", thickness=3),
        )
        self.gfx_object.add(self.gfx_g_trail)

        # ---- Nose trajectory (trace of aircraft nose) ----
        self.nose_trail_len = 600
        self.nose_trail = np.zeros((self.nose_trail_len, 3), dtype=np.float32)

        nose_geom = gfx.Geometry(positions=self.nose_trail)

        self.gfx_nose_trail = gfx.Line(
            nose_geom,
            gfx.LineMaterial(color="green", thickness=4),
        )

        self.gfx_scene.add(self.gfx_nose_trail)

        # ajout du groupe à la scène
        self.gfx_scene.add(self.gfx_object)
        self.gfx_display.show(self.gfx_scene)

        # ---- Camera configuration : Z axis up ----
        cam = self.gfx_display.camera
        cam.local.position = (600, 600, 400)  # closer camera for stronger initial zoom
        cam.world.reference_up = (refcam[0], refcam[1], refcam[2])
        cam.look_at((0, 0, 0))

        # Camera
        self.gfx_box = gfx.Mesh(
            gfx.box_geometry(240, 120, 600),  # X, Y, Z
            gfx.MeshPhongMaterial(color="Gray"),)
        self.gfx_box.local.position = (0,0,0)
        self.gfx_object.add(self.gfx_box)
        # Hidden by default
        self.gfx_box.visible = False

        # sphère rouge au-dessus du parallélépipède
        sphere_radius = 90
        self.gfx_sphere = gfx.Mesh(
            gfx.sphere_geometry(radius=sphere_radius),
            gfx.MeshPhongMaterial(color="black"),)
        self.gfx_sphere.local.position = (0, 40, 180)
        self.gfx_object.add(self.gfx_sphere)
        # Hidden by default
        self.gfx_sphere.visible = False

        # ---- Load CAP10 STL ----
        try:
            stl_meshes = gfx.load_mesh(STL_FILE)

            # gfx.load_mesh may return a list of meshes
            if not isinstance(stl_meshes, (list, tuple)):
                stl_meshes = [stl_meshes]

            for mesh in stl_meshes:
                mesh.local.scale = (5, 5, 5)
                mesh.local.position = (620, 500, -140) # for cap 10
                # ---- Parametric rotation: 180° around Z + tilt around X ----
                # ---- STL tilt parameter (degrees) ----
                self.stl_tilt_deg = 5.0  # default 5° around X
                theta_x = math.radians(self.stl_tilt_deg)

                # Quaternion for 180° around Z
                qz = np.array([0.0, 0.0, 1.0, 0.0])  # (x, y, z, w)

                # Quaternion for tilt around X
                qx = np.array([
                    math.sin(theta_x / 2.0),0.0,0.0,math.cos(theta_x / 2.0)])

                # Quaternion multiplication q = qz * qx
                x1, y1, z1, w1 = qz
                x2, y2, z2, w2 = qx

                q = np.array([
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                    w1*w2 - x1*x2 - y1*y2 - z1*z2])

                mesh.local.rotation = tuple(q)
                self.gfx_object.add(mesh)

            #print("CAP10.STL loaded successfully")
        except Exception as e:
            print(f"Error loading CAP10.STL: {e}")

        self.gfx_canvas = self.gfx_display.canvas
        self.grid.addWidget(self.gfx_canvas, 1, 0, 1, 2)
        # ---- Force white background on pygfx Qt widget ----

        # ---- DataFrame info overlay (top-left) ----
        self.df_info_label = QLabel("Data:", self.gfx_canvas)
        self.df_info_label.setStyleSheet("color: gray; background-color: transparent; padding: 4px; font-family: 'Menlo';")
        self.df_info_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.df_info_label.adjustSize()
        self.df_info_label.move(10, 10)
        self.df_info_label.raise_()

        # ---- Pitch overlay (custom position) ----
        self.pitch_label = QLabel("Pitch:", self.gfx_canvas)
        self.pitch_label.setStyleSheet("color: green; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 28px; font-weight: bold;")
        self.pitch_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.pitch_label.adjustSize()
        # position under wing artificial horizon (left side)
        self.pitch_label.move(0, 260)
        self.pitch_label.raise_()

        # ---- Inclination overlay (custom position) ----
        self.roll_label = QLabel("Bank:", self.gfx_canvas)
        self.roll_label.setStyleSheet("color: blue; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 28px; font-weight: bold;")
        self.roll_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.roll_label.adjustSize()
        self.roll_label.move(0, 300)
        self.roll_label.raise_()


        # ---- Acceleration magnitude (g) overlay ----
        self.g_label = QLabel("g:", self.gfx_canvas)
        self.g_label.setStyleSheet("color: red; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 44px; font-weight: bold;")
        self.g_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.g_label.adjustSize()
        self.g_label.move(self.gfx_canvas.width() - self.g_label.width() - 10,160)
        self.g_label.raise_()

        # ---- Acceleration magnitude (g) minmax ----
        self.g_label_minmax = QLabel("g:", self.gfx_canvas)
        self.g_label_minmax.setStyleSheet("color: gray; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 14px; font-weight: bold;")
        self.g_label_minmax.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.g_label_minmax.adjustSize()
        self.g_label_minmax.move(self.gfx_canvas.width() - self.g_label.width() - 10,210)
        self.g_label_minmax.raise_()

        # ---- GPS speed & altitude overlay (top-right) ----
        self.gps_label_speed = QLabel("GPS:", self.gfx_canvas)
        self.gps_label_speed.setStyleSheet("color: red; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 44px; font-weight: bold;")
        self.gps_label_speed.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.gps_label_speed.adjustSize()
        # position top‑right of pygfx window
        self.gps_label_speed.move(
            self.gfx_canvas.width() - self.gps_label_speed.width() - 10,
            10)
        self.gps_label_speed.raise_()
        self.gps_label_speed.show()

        # ---- GS max label (just under GS) ----
        self.gsmax_label = QLabel("GSmax:", self.gfx_canvas)
        self.gsmax_label.setStyleSheet(
            "color: gray; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 14px; font-weight: bold;"
        )
        self.gsmax_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.gsmax_label.adjustSize()
        # GSmax just under GS
        self.gsmax_label.move(
            self.gfx_canvas.width() - self.gsmax_label.width() - 10,
            50
        )
        self.gsmax_label.raise_()
        self.gsmax_label.show()

        # ---- GPS speed & altitude overlay (bottom-right) ----
        self.gps_label_alt = QLabel("GPS:", self.gfx_canvas)
        self.gps_label_alt.setStyleSheet("color: blue; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 44px; font-weight: bold;")
        self.gps_label_alt.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.gps_label_alt.adjustSize()
        self.gps_label_alt.move(
            self.gfx_canvas.width() - self.gps_label_alt.width() - 10,
            self.gfx_canvas.height() - self.gps_label_alt.height() - 40
        )
        self.gps_label_alt.raise_()
        self.gps_label_alt.show()

        # ---- GPS speed & altitude overlay (bottom-right) ----
        self.gps_label_vario = QLabel("GPS:", self.gfx_canvas)
        self.gps_label_vario.setStyleSheet(
            "color: gray; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 14px; font-weight: bold;")
        self.gps_label_vario.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.gps_label_vario.adjustSize()
        self.gps_label_vario.move(
            self.gfx_canvas.width() - self.gps_label_vario.width() - 10,
            self.gfx_canvas.height() - self.gps_label_vario.height() - 110
        )
        self.gps_label_vario.raise_()
        self.gps_label_vario.show()



        # ---- Artificial Horizon (top-left) ----
        self.hud_horizon = ArtificialHorizon(self.gfx_canvas)
        self.hud_horizon.setGeometry(10, 10, 130, 125)
        self.hud_horizon.show()

        # ---- Wingtip Artificial Horizon (left wing triangle reference) ----
        self.hud_horizon_wing = ArtificialHorizon(self.gfx_canvas)
        self.hud_horizon_wing.setGeometry(10, 140, 130, 125)
        self.hud_horizon_wing.show_triangle = True
        self.hud_horizon_wing.show()

    def update_gfx_orientation(self):
        #R = quat_to_rot(quats[self.i])
        row = self.row
        R = quat_to_rot([row.x4_quat_w,row.x4_quat_x,row.x4_quat_y,row.x4_quat_z])

        #rotation repère monde > local camera
        #X →  Z    Y → -Y    Z →  X
        #R_recalage_repere=[[0, 0, 1], [0, -1, 0], [1, 0, 0]]
        theta_x = np.deg2rad(self.montage_pitch_angle)  # ---- Rotation supplémentaire : +montage_pitch_angle° autour de X (appliquée en dernier) ----
        R_x_20 = np.array([[1, 0, 0],[0, np.cos(theta_x), -np.sin(theta_x)],[0, np.sin(theta_x),  np.cos(theta_x)]])
        R_final = R_x_20 @ perm[R_recalage_repere] @ R

        # Forward and Up original vectors
        fwd_original = np.array([0.0, 1.0, 0.0])
        up_original = np.array([0.0, 0.0, 1.0])
        down_original = np.array([0.0, 0.0, -1.0])

        # Rotated vectors repère local
        fwd = R_final.T @ fwd_original; up = R_final.T @ up_original; down = R_final.T @ down_original

        # ---- Update acceleration vector ----
        acc = np.array([-row.x4_acc_x,-row.x4_acc_y,-row.x4_acc_z])# rotation
        self.g = np.linalg.norm(acc) / 9.81
        acc = acc / np.linalg.norm(acc) # normer vecteur
        self.acc_vec = R_x_20 @ perm[R_recalage_repere] @ acc
        self.acc_vec = self.acc_vec * 300 +  self.acc_vec * 100 * self.g # scaling up
        self.acc_geom.positions.data[1] = self.acc_vec
        self.acc_geom.positions.update_range(0, 2)

        # ---- Update G vector trail ----
        # If trail is empty (after seek/reset), initialize it with the current vector
        if not np.any(self.g_trail):
            self.g_trail[:] = self.acc_vec
        else:
            self.g_trail[:-1] = self.g_trail[1:]
            self.g_trail[-1] = self.acc_vec

        self.gfx_g_trail.geometry.positions.data[:] = self.g_trail
        self.gfx_g_trail.geometry.positions.update_range(0, len(self.g_trail))

        # ---- Nose trajectory update ----
        nose_vec = fwd * 400

        # If trail is empty (after seek/reset), initialize it with the current nose position
        if not np.any(self.nose_trail):
            self.nose_trail[:] = nose_vec
        else:
            self.nose_trail[:-1] = self.nose_trail[1:]
            self.nose_trail[-1] = nose_vec

        self.gfx_nose_trail.geometry.positions.data[:] = self.nose_trail
        self.gfx_nose_trail.geometry.positions.update_range(0, len(self.nose_trail))

        # ---- Update arrow head position and orientation ----
        direction = self.acc_vec / (np.linalg.norm(self.acc_vec) + 1e-9)
        pos = self.acc_vec

        # compute quaternion rotating +Z to direction
        z_axis = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z_axis, direction)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            quat = (0, 0, 0, 1)
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            s = np.sin(angle / 2.0)
            quat = (
                axis[0] * s,
                axis[1] * s,
                axis[2] * s,
                np.cos(angle / 2.0),
            )

        self.gfx_acc_arrow.local.position = tuple(pos)
        self.gfx_acc_arrow.local.rotation = quat
        # calcul g positif/négatif
        g_sens = angle_between(self.acc_vec, down_original)
        if g_sens > 90:
            self.g=-self.g
        # update min/max encountered G
        self.g_min = min(self.g_min, self.g)
        self.g_max = max(self.g_max, self.g)

        self.g_label.setText(f"G:{self.g:.1f}")
        self.g_label_minmax.setText(f"  Gmin {self.g_min:.2f}\n  Gmax {self.g_max:.2f}")
        self.g_label.adjustSize()
        self.g_label.move(self.gfx_canvas.width() - self.g_label.width(),self.gfx_canvas.height()-100)
        self.g_label_minmax.adjustSize()
        self.g_label_minmax.move(self.gfx_canvas.width() - self.g_label_minmax.width(),self.gfx_canvas.height()-40)

        # update acceleration vector geometry & color
        if self.g > 0.8:
            err = min(abs(self.g - 1.0), 1.0)
            err2G = max(self.g-2.0,0)
            r = err;g = 1.0 - err;b = err2G/2
        else:
            r=0;g=0;b=1

        self.gfx_vec_acc.material.color = (r, g, b, 1)
        self.gfx_acc_arrow.material.color = (r, g, b, 1)
        self.g_label.setStyleSheet(
            f"color: rgb({r * 255},{g*255},{b*255}); "
            "background-color: transparent; padding: 10px; "
            "font-family: 'Menlo'; font-size: 44px; font-weight: bold;")
        #rotation de l'objet
        M = np.eye(4);M[:3, :3] = R_final.T
        self.gfx_object.local.matrix = M

        # ---- Compute Pitch (angle between rotated Y and XY plane) ----
        v_original = np.array([0.0, 1.0, 0.0])
        v_rotated = R_final.T @ v_original

        # XY plane normal = Z axis
        plane_normal = np.array([0.0, 0.0, 1.0])
        dot = np.dot(v_rotated, plane_normal)
        norm_v = np.linalg.norm(v_rotated)

        # angle between vector and plane
        pitch_rad = np.arcsin(np.clip(dot / norm_v, -1.0, 1.0))
        pitch_deg = np.degrees(pitch_rad)

        # ---- Compute Inclinaison (roll) ----
        # Vector v orthogonal to fwd and lying in plane (fwd, world Z)
        world_z = np.array([0.0, 0.0, 1.0])
        v = np.cross(fwd, np.cross(world_z, fwd))
        v = v / np.linalg.norm(v)

        # Inclination computation
        dot = np.dot(up, v)
        cross = np.cross(up, v)
        inclinaison = np.arctan2(np.dot(cross, fwd), dot)
        inclinaison_deg = -np.degrees(inclinaison) # convention aéronautique (droite positive / gauche positive)

        # ---- Compute Inertial Heading ----
        # Projection du vecteur fwd sur le plan XY (Z=0)
        fwd_proj = np.array([fwd[0], fwd[1], 0.0])
        norm_proj = np.linalg.norm(fwd_proj)

        if norm_proj > 1e-8:
            fwd_proj = fwd_proj / norm_proj
            # angle par rapport à l'axe X
            heading_rad = np.arctan2(fwd_proj[1], fwd_proj[0])
            heading_deg = np.degrees(heading_rad)
            if heading_deg < 0:
                heading_deg += 360
            heading_deg=360-heading_deg
        else:
            heading_deg = 0.0

        # ---- Update Artificial Horizon ----
        if hasattr(self, "hud_horizon"):
            self.hud_horizon.pitch = pitch_deg
            self.hud_horizon.bank = inclinaison_deg
            self.hud_horizon.heading = heading_deg
            self.hud_horizon.update()

        # ---- Update Wingtip Artificial Horizon ----
        if hasattr(self, "hud_horizon_wing"):

            # rotate the viewing reference 90° around left wing
            R_view_wing = np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ])

            R_wing = R_view_wing @ R_final

            # forward and up vectors in the wing reference
            fwd_w = R_wing.T @ np.array([0.0, 1.0, 0.0])
            up_w  = R_wing.T @ np.array([0.0, 0.0, 1.0])

            # pitch in wing reference
            pitch_w = np.degrees(np.arcsin(np.clip(fwd_w[2], -1.0, 1.0)))

            # roll in wing reference
            # compute roll around the wing viewing axis using Y/Z plane
            # this remains stable when aircraft pitch approaches vertical
            right_w = np.cross(fwd_w, up_w)
            roll_w = np.degrees(np.arctan2(right_w[2], up_w[2]))

            # invert pitch sign for wingtip reference (viewed from the side)
            self.hud_horizon_wing.pitch = -pitch_w
            self.hud_horizon_wing.bank = roll_w
            self.hud_horizon_wing.heading = heading_deg
            self.hud_horizon_wing.update()

        # ---- Update dataframe info label ----
        # compute elapsed time since start of dataset
        t0 = df.timestamp.iloc[0]
        t_now = df.timestamp.iloc[self.idf]
        elapsed = t_now - t0
        elapsed_s = int(elapsed.total_seconds())
        em = elapsed_s // 60
        es = elapsed_s % 60

        self.df_info_label.setText(
            f"Frame: {self.i}"
            f"\nTime: {t_now.strftime('%H:%M:%S.%f')[:-3]}"
            f"\nElapsed: {em:02d}:{es:02d}"
            f"\nFrames skipped: {self.frame_skipped_count} / {self.frame_last_delay:+04d}ms"
        )
        self.df_info_label.adjustSize()
        self.df_info_label.move(0,self.gfx_canvas.height() - self.df_info_label.height())

        self.pitch_label.setText(f"Pitch {pitch_deg:.1f}°")
        self.pitch_label.adjustSize()
        self.pitch_label.move(0, 260)

        self.roll_label.setText(f"Bank {inclinaison_deg:.1f}°")
        self.roll_label.adjustSize()
        self.roll_label.move(0, 300)

        # ---- Update GPS speed / altitude overlay ----
        # ---- Update heading overlay on video1 ----
        if hasattr(self, "video1_heading_label"):
            self.video1_heading_label.setText(f"{row.gps_heading:.1f}°")
            self.video1_heading_label.adjustSize()
            x = int((self.video1.width() - self.video1_heading_label.width()) / 2)
            self.video1_heading_label.move(x, 5)

        self.gps_label_speed.setText(f"GS {row.gps_speed:.0f} km/h")
        # update GS max
        if row.gps_speed > self.gs_max:
            self.gs_max = row.gps_speed
        self.gps_label_speed.adjustSize()
        self.gps_label_speed.move(
            self.gfx_canvas.width() - self.gps_label_speed.width() - 10,
            0)

        # update GSmax label
        self.gsmax_label.setText(f"GSmax {self.gs_max:.0f}")
        self.gsmax_label.adjustSize()
        self.gsmax_label.move(
            self.gfx_canvas.width() - self.gsmax_label.width() - 10,
            45)

        # update speed vector geometry & color
        r = 0;g = 0;b = 0
        if row.gps_speed < 113:
            r=g=0; b=255
        else:
            if row.gps_speed > 112 and row.gps_speed < 236:
                r=0; g=255; b=0
            else:
                if row.gps_speed > 235 and row.gps_speed < 300:
                    r=int((row.gps_speed-235)/(300-235)*255); g=255-int((row.gps_speed-235)/(300-235)*255); b=0
                else:
                    r=255;g=0;b=0

        self.gps_label_speed.setStyleSheet(
            f"color: rgb({r},{g},{b}); "
            "background-color: transparent; padding: 10px; "
            "font-family: 'Menlo'; font-size: 44px; font-weight: bold;")

        self.gps_label_alt.setText(f"Alt {row.gps_alt:.0f} ft")
        self.gps_label_alt.adjustSize()
        self.gps_label_alt.move(
            self.gfx_canvas.width() - self.gps_label_alt.width(),60)

        self.gps_label_vario.setText(f"{row.gps_fpm:.0f} ft/min")
        self.gps_label_vario.adjustSize()
        self.gps_label_vario.move(
            self.gfx_canvas.width() - self.gps_label_vario.width(),100)

    def calibrate_gfx(self,where):
        row = df.iloc[where]
        grav = [row.x4_acc_x, row.x4_acc_y, row.x4_acc_z]  # utilisation accelero Y au repos
        grav = grav / np.linalg.norm(grav)
        self.montage_pitch_angle = math.degrees(math.acos(-grav[1]))
        print("Angle de montage : ",self.montage_pitch_angle)

    def calibrate_gfx_on_current_frame(self):
        self.calibrate_gfx(self.idf)


    def on_map_loaded(self, ok):
        if ok:
            self.map_ready = True
            #print("Map ready")


    # ==================================================
    # Real-time compensated main loop (absolute scheduling)
    # ==================================================
    def main_loop(self):

        now = self.clock.elapsed()

        # initialize absolute schedule
        if self.next_frame_time == 0:
            self.next_frame_time = now

        if self.playing:
            self.update_all()

        # schedule next frame using absolute timing
        self.next_frame_time += self.frame_period_ms

        # skip frames if we are late (prevents backlog)
        while now > self.next_frame_time + self.frame_period_ms:
            self.next_frame_time += self.frame_period_ms

        delay = int(self.next_frame_time - now)
        self.frame_last_delay = delay
        if delay < 0:
            delay = 0
            self.frame_skipped_count += 1

        self.timer.start(delay)

    # ==================================================
    # BOOKMARK SYSTEM
    # ==================================================
    def load_bookmarks(self):
        try:
            self.bookmarks_df = pd.read_csv(BOOKMARK_FILE)
            # ensure time column exists (backward compatibility)
            if "time" not in self.bookmarks_df.columns:
                self.bookmarks_df["time"] = ""
            # ensure column order: time first
            cols = ["time", "name", "frame"]
            self.bookmarks_df = self.bookmarks_df[[c for c in cols if c in self.bookmarks_df.columns]]
            # sort bookmarks by frame index (ascending)
            if "frame" in self.bookmarks_df.columns:
                self.bookmarks_df = self.bookmarks_df.sort_values("frame").reset_index(drop=True)
        except Exception:
            self.bookmarks_df = pd.DataFrame(columns=["time", "name", "frame"])
        self.refresh_bookmark_menu()

    def save_bookmarks(self):
        self.bookmarks_df.to_csv(BOOKMARK_FILE, index=False)

    def refresh_bookmark_menu(self):
        if not hasattr(self, "menu_bookmarks"):
            return
        # rebuild menu while preserving the fixed actions
        self.menu_bookmarks.clear()
        self.menu_bookmarks.addAction(self.act_reload_bookmarks)
        self.menu_bookmarks.addSeparator()
        self.menu_bookmarks.addAction(self.act_add_bookmark)
        self.menu_bookmarks.addSeparator()

        for _, row in self.bookmarks_df.iterrows():
            name = row["name"]
            frame = int(row["frame"])
            t = row.get("time", "")
            label = f"{t} {name}" if t else name
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, f=frame: self.goto_bookmark(f))
            self.menu_bookmarks.addAction(act)

    def add_bookmark(self):
        name, ok = QInputDialog.getText(self, "Bookmark", "Nom du bookmark:")
        if not ok or name.strip() == "":
            return
        frame = int(self.i)

        # compute time since start of playback
        fps = float(self.stream1.average_rate)
        if fps <= 0:
            fps = 30
        seconds = int(frame / fps)
        minutes = seconds // 60
        sec = seconds % 60
        time_str = f"{minutes:02d}:{sec:02d}"

        new_row = pd.DataFrame([[time_str, name, frame]], columns=["time", "name", "frame"])
        self.bookmarks_df = pd.concat([self.bookmarks_df, new_row], ignore_index=True)

        # sort bookmarks by frame index
        self.bookmarks_df = self.bookmarks_df.sort_values("frame").reset_index(drop=True)

        self.save_bookmarks()
        self.refresh_bookmark_menu()


    def reload_bookmarks(self):
        """Force reload of bookmark CSV file and refresh menu."""
        self.load_bookmarks()

    # ==================================================
    # Centralized video seek (avoid repeating CAP_PROP_POS_FRAMES everywhere)
    # ==================================================
    def seek_video(self, frame):

        self.i = int(frame)
        # ---- Reset trails when seeking (nose + G vector history) ----
        if hasattr(self, "nose_trail"):
            self.nose_trail[:] = 0
            try:
                self.gfx_nose_trail.geometry.positions.data[:] = self.nose_trail
                self.gfx_nose_trail.geometry.positions.update_range(0, len(self.nose_trail))
            except Exception:
                pass

        if hasattr(self, "g_trail"):
            self.g_trail[:] = 0
            try:
                self.gfx_g_trail.geometry.positions.data[:] = self.g_trail
                self.gfx_g_trail.geometry.positions.update_range(0, len(self.g_trail))
            except Exception:
                pass

        fps = float(self.stream1.average_rate)

        ts = int((frame / fps) / float(self.stream1.time_base))

        self.container1.seek(ts, stream=self.stream1)
        self.container2.seek(ts, stream=self.stream2)

        self.decoder1 = self.container1.decode(self.stream1)
        self.decoder2 = self.container2.decode(self.stream2)

        # ---- reset audio when seeking ----
        if self.audio_stream is not None:
            try:
                fps = float(self.stream1.average_rate)
                ts_audio = int((frame / fps) / float(self.audio_stream.time_base))

                # seek audio container
                self.audio_container.seek(ts_audio, stream=self.audio_stream)

                # recreate packet generator
                self.audio_packets = self.audio_container.demux(self.audio_stream)

                # clear buffered audio and restart buffering
                self.audio_buffer.clear()
                self.audio_started = False

            except Exception:
                pass

    def goto_bookmark(self, frame):
        self.seek_video(frame)
        self.slider.setValue(self.i)

        # show bookmark overlay immediately when jumping to it
        if self.bookmarks_df is not None:
            row = self.bookmarks_df[self.bookmarks_df["frame"] == frame]
            if not row.empty:
                name = str(row.iloc[0]["name"])
                self.show_bookmark_overlay(name)

    def show_bookmark_overlay(self, name):

        if self.bookmark_overlay is None:
            return

        self.bookmark_overlay.setText(name)
        self.bookmark_overlay.adjustSize()

        # centrer en haut de la fenêtre
        x = int((self.width() - self.bookmark_overlay.width()) / 2)
        self.bookmark_overlay.move(x, 20)

        self.bookmark_overlay.show()

        # disparaît après 3 secondes
        QTimer.singleShot(3000, self.bookmark_overlay.hide)

    # ==================================================
    def toggle_play(self):
        self.playing = not self.playing
        self.btn_pause.setText("▶ Lecture" if not self.playing else "⏸ Pause")

        # keep audio synchronized with video pause/play
        if hasattr(self, "audio_output") and self.audio_stream is not None:
            try:
                if self.playing:
                    self.audio_output.resume()
                else:
                    self.audio_output.suspend()
            except Exception:
                pass

    # ==================================================
    # Gestion clavier (Espace = Pause / Lecture)
    # ==================================================
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_play()
            event.accept()
        else:
            super().keyPressEvent(event)

    # ==================================================
    # time jump helpers
    # ==================================================
    def jump_back_10s(self):
        fps = float(self.stream1.average_rate) or 30
        frame = max(0, int(self.i - fps * 10))
        self.seek_video(frame)
        self.slider.setValue(self.i)

    def jump_back_2s(self):
        fps = float(self.stream1.average_rate) or 30
        frame = max(0, int(self.i - fps * 2))
        self.seek_video(frame)
        self.slider.setValue(self.i)

    def jump_fwd_2s(self):
        fps = float(self.stream1.average_rate) or 30
        frame = min(N - 1, int(self.i + fps * 2))
        self.seek_video(frame)
        self.slider.setValue(self.i)

    def jump_fwd_10s(self):
        fps = float(self.stream1.average_rate) or 30
        frame = min(N - 1, int(self.i + fps * 10))
        self.seek_video(frame)
        self.slider.setValue(self.i)

    def goto_mise_en_ligne(self):
        self.seek_video(self.get_video_frame_from_df_index(index_enligne_devol))
        self.slider.setValue(self.i)


    def goto_entree_box(self):
        self.seek_video(self.get_video_frame_from_df_index(index_entree_3000))
        self.slider.setValue(self.i)

    def goto_start(self):
        self.seek_video(9900)
        self.slider.setValue(self.i)

    def goto_misedos_securite(self):
        self.seek_video(33700)
        self.slider.setValue(self.i)

    def goto_enchainement(self):
        self.seek_video(61117)
        self.slider.setValue(self.i)

    def seek_palier(self):
        """
        Cherche un pallier :
        |gps_fpm| < 100 ft/min
        gps_speed > 150 km/h
        pendant 10 secondes
        """

        window = int(DF_FREQ * 5)  # 5 secondes

        fpm = df["gps_fpm"].to_numpy()
        speed = df["gps_speed"].to_numpy()

        for i in range(0, len(df) - window):

            seg_fpm = fpm[i:i + window]
            seg_speed = speed[i:i + window]

            if np.all(np.abs(seg_fpm) < 150) and np.all(seg_speed > 150):
                frame = self.get_video_frame_from_df_index(i)
                self.seek_video(frame)
                self.slider.setValue(self.i)

                print("Palier trouvé @ frame", frame)
                return

        print("Aucun palier trouvé")


    # ==================================================
    def read_video_frame(self, decoder):
        try:
            frame = next(decoder)

            # Convert only if needed
            if frame.format.name != "bgr24":
                frame = frame.reformat(format="bgr24")

            # Return the frame directly (no numpy conversion)
            return True, frame, frame

        except StopIteration:
            return False, None, None

    # ==================================================
    # AUDIO
    # ==================================================
    def update_audio(self):

        # fill buffer (decode ahead)
        target = self.audio_buffer_target if not self.audio_started else int(self.audio_buffer_target * 0.75)
        packets_decoded = 0
        while self.audio_buffer.size() < target and packets_decoded < 48:
            try:
                packet = next(self.audio_packets)
            except StopIteration:
                break

            packets_decoded += 1

            if packet.dts is None:
                continue

            frames_decoded = packet.decode()
            if not frames_decoded:
                continue

            for frame in frames_decoded:
                # update audio clock from original frame PTS
                if frame.pts is not None and frame.time_base is not None:
                    try:
                        self.audio_clock_sec = float(frame.pts * frame.time_base)
                    except Exception:
                        pass

                # resample frame to s16 stereo
                frames = self.audio_resampler.resample(frame)

                if not isinstance(frames, (list, tuple)):
                    frames = [frames]

                for f in frames:
                    samples = f.to_ndarray()

                    # ensure interleaved layout
                    if samples.ndim == 2:
                        samples = samples.T.reshape(-1)

                    pcm = samples.astype(np.int16).tobytes()
                    self.audio_buffer.append(pcm)

            # continue filling buffer
            continue

        # feed Qt audio device according to available space
        free = self.audio_output.bytesFree()

        # start audio only when ~1s buffer ready
        if not self.audio_started:
            if self.audio_buffer.size() >= self.audio_buffer_target:
                self.audio_started = True
            else:
                return

        # avoid very small writes which can create micro‑stutters
        MIN_CHUNK = 1024  # smaller chunks reduce dropouts when video frames are skipped

        # continuously feed the audio device while it has space
        while True:
            free = self.audio_output.bytesFree()

            if free < MIN_CHUNK or self.audio_buffer.size() < MIN_CHUNK:
                break

            to_write = min(free, self.audio_buffer.size(), 16384)
            chunk = self.audio_buffer[:to_write]

            written = self.audio_device.write(chunk)
            if written <= 0:
                break

            self.audio_buffer.remove(0, written)

    # 🔑 SYNCHRO VIDEO ← DF
    # ==================================================
    def get_video_frame_from_df_index(self, df_index):
        """
        Calcule l'index frame vidéo à partir d'un index du dataframe.
        Retourne l'index de frame correspondant dans la vidéo 1.
        """
        if df_index < 0 or df_index >= frames_df:
            raise ValueError("df_index hors limites")

        # timestamp dataframe
        ts_df = df.timestamp.iloc[df_index]

        # temps vidéo correspondant (UTC)
        ts_video_utc = ts_df - self.video_df_offset

        # delta par rapport au début vidéo 1
        delta = ts_video_utc - self.video1_start

        # récupération FPS vidéo
        fps = float(self.stream1.average_rate)
        if fps <= 0:
            fps = 30  # fallback sécurité

        # conversion en frame
        frame_index = int(delta.total_seconds() * fps)

        # clamp sécurité
        frame_index = max(0, min(frame_index, N - 1))

        return frame_index

    # ==================================================

    # ==================================================
    def update_all(self):
        #t0 = time.perf_counter()
        self.update_video(self.decoder1, self.video1, self.video1_start, self.stream1)
        self.update_video(self.decoder2, self.video2, self.video2_start, self.stream2)

        if self.current_video_time_utc is not None:
            self.sync_dataframe_on_video()

        self.update_matplotlib_gps()

        # synchronisation orientation pygfx ← DataFrame
        #row = df.iloc[self.idf]
        self.update_gfx_orientation()

        #t1 = time.perf_counter()
        #print(f"UpdateALl duration: {(t1 - t0) * 1000:.2f} ms")

        # ---- Bookmark trigger (1 second before) ----
        if self.bookmarks_df is not None and not self.bookmarks_df.empty:

            fps = float(self.stream1.average_rate)
            if fps <= 0:
                fps = 30
            fps = int(fps)

            for _, row_bm in self.bookmarks_df.iterrows():
                frame = int(row_bm["frame"])
                trigger_frame = frame - fps  # 1 second before

                if self.i == trigger_frame:
                    name = str(row_bm["name"])

                    if frame != self.last_bookmark_frame:
                        self.show_bookmark_overlay(name)
                        self.last_bookmark_frame = frame

    # ==================================================
    def update_video(self, decoder, label, start_dt, stream):

        ret, frame, avframe = self.read_video_frame(decoder)

        if not ret:
            return

        ms = avframe.pts * float(stream.time_base) * 1000

        video_time_utc = start_dt + timedelta(milliseconds=ms)
        self.current_video_time_utc = video_time_utc

        display_time = video_time_utc.astimezone(ZoneInfo("Europe/Paris"))

        plane = frame.planes[0]
        h = frame.height
        w = frame.width

        img = QImage(plane, w, h, plane.line_size, QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(img))

        # advance global frame index once per loop (video1 drives timing)
        if label is self.video1:
            self.i += 1

    # ==================================================
    # 🔑 SYNCHRO DF ← VIDEO
    # ==================================================
    def sync_dataframe_on_video(self):
        ts = self.current_video_time_utc + self.video_df_offset
        idx = df["timestamp"].searchsorted(ts)

        if idx <= 0:
            self.idf = 0
        elif idx >= frames_df:
            self.idf = frames_df - 1
        else:
            before = df.timestamp.iloc[idx - 1]
            after = df.timestamp.iloc[idx]
            self.idf = idx - 1 if abs(ts - before) <= abs(after - ts) else idx

        # cache dataframe row once per frame
        self.row = df.iloc[self.idf]

        self.slider.blockSignals(True)
        self.slider.setValue(self.i)
        self.slider.blockSignals(False)

        self.timestamp_label.setText(f"Video time : {ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    # ==================================================
    # 🔑 SYNCHRO VIDEO ← DF
    # ==================================================
    def compute_video_frame_from_df_index(self, df_index):
        """
        Calcule l'index frame vidéo à partir d'un index du dataframe.
        Retourne l'index de frame correspondant dans la vidéo 1.
        """
        if df_index < 0 or df_index >= frames_df:
            raise ValueError("df_index hors limites")

        # timestamp dataframe
        ts_df = df.timestamp.iloc[df_index]

        # temps vidéo correspondant (UTC)
        ts_video_utc = ts_df - self.video_df_offset

        # delta par rapport au début vidéo 1
        delta = ts_video_utc - self.video1_start

        # récupération FPS vidéo
        fps = float(self.stream1.average_rate)
        if fps <= 0:
            fps = 30  # fallback sécurité

        # conversion en frame
        frame_index = int(delta.total_seconds() * fps)

        # clamp sécurité
        frame_index = max(0, min(frame_index, N - 1))

        return frame_index

    # ==================================================
    def on_slider(self, value):
        # move both videos to the requested frame
        self.seek_video(value)

        # force an immediate refresh when paused
        if not self.playing:
            self.update_all()

    def on_matplotlib_double_click(self, event):
        if event.button != 1 or not event.dblclick:
            return

        # mettre la lecture en pause lors du passage en plein écran
        if self.playing:
            self.toggle_play()

        dlg = QDialog(self)
        dlg.setWindowTitle("Graphe GPS – Plein écran")
        dlg.showFullScreen()

        layout = QVBoxLayout(dlg)

        fig_fs = Figure()
        canvas_fs = FigureCanvas(fig_fs)
        layout.addWidget(canvas_fs)

        # copier le contenu du graphe existant
        ax_fs = fig_fs.add_subplot(111, projection="3d")
        fig_fs.subplots_adjust(
            left=0.02,
            right=0.98,
            bottom=0.02,
            top=0.98
        )

        ax_src = self.ax
        ax_fs.set_xlim(ax_src.get_xlim())
        ax_fs.set_ylim(ax_src.get_ylim())
        ax_fs.set_zlim(ax_src.get_zlim())
        ax_fs.view_init(azim=ax_src.azim, elev=ax_src.elev)

        fs_traj_collection = Line3DCollection([], cmap="viridis", linewidth=2)
        fs_traj_collection.set_norm(Normalize(vmin=0, vmax=6000))  # altitude ft
        ax_fs.add_collection(fs_traj_collection)

        last = self.idf - TRACE
        if last < 0:
            last = 0
        lx = df.loc[last:self.idf, "gps_lon"]
        ly = df.loc[last:self.idf, "gps_lat"]
        lz = df.loc[last:self.idf, "gps_alt"]
        lx = lx[::10]
        ly = ly[::10]
        lz = lz[::10]

        # trajectoire colorée par altitude (Line3DCollection)
        if len(lx) > 1:
            points = np.array([lx.values, ly.values, lz.values]).T
            segments = np.stack([points[:-1], points[1:]], axis=1)
            fs_traj_collection.set_segments(segments)
            fs_traj_collection.set_array(lz.values[:-1])

        # recopier lignes classiques
        for line in ax_src.lines:
            try:
                x, y, z = line.get_data_3d()
            except AttributeError:
                # fallback sécurité (anciennes versions)
                x, y, z = line._verts3d

            ax_fs.plot(
                x, y, z,
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth()
            )


        canvas_fs.draw_idle()

        # ouverture plein écran (bloquant)
        dlg.exec_()

        # reprise automatique de la lecture à la fermeture
        if not self.playing:
            self.toggle_play()

    # ==================================================
    def update_matplotlib_gps(self):
        # Matplotlib is expensive; update only every 4 frames > pas util il y a un test
        #if self.i % 4 != 0:
        #    return

        row = self.row

        #test si MàJ GPS (4hz)
        if  self.firstGPS == True:
            self.gps_lastrow=row
            self.firstGPS = False
        else:
            if self.gps_lastrow.gps_alt == row.gps_alt:
                if self.gps_lastrow.gps_lat == row.gps_lat:
                    if self.gps_lastrow.gps_lon == row.gps_lon:
                        if self.gps_lastrow.gps_speed == row.gps_speed:
                            return

        if self.map_ready:
            lat = row.gps_lat
            lon = row.gps_lon
            self.map_view.page().runJavaScript(f"window.updateMarker({lat}, {lon});")

        if row.gps_alt >= 3000:
            deck=3000
            sky=5000
        else:
            deck=0
            sky=3000

        #self.traj_collection.set_norm(Normalize(deck, sky))


        last = self.idf - TRACE
        if last < 0:
            last = 0
        # Use numpy cached arrays for fast access
        lx = gps_lon_vals[last:self.idf:TRACE_SLICING_FACTOR]
        ly = gps_lat_vals[last:self.idf:TRACE_SLICING_FACTOR]
        lz = gps_alt_vals[last:self.idf:TRACE_SLICING_FACTOR]

        # trajectoire colorée par altitude (Line3DCollection)
        if len(lx) > 1:
            points = np.array([lx, ly, lz]).T
            segments = np.stack([points[:-1], points[1:]], axis=1)
            self.traj_collection.set_segments(segments)
            self.traj_collection.set_array(lz[:-1])
        # ground projection
        self.line_ground.set_data(lx, ly)
        self.line_ground.set_3d_properties([deck] * len(lx))

        # trajectory to ground projection
        tx = [row.gps_lon,row.gps_lon]
        ty = [row.gps_lat,row.gps_lat]
        tz = [row.gps_alt,deck]
        self.line_traj_ground.set_data(tx, ty)
        self.line_traj_ground.set_3d_properties(tz)

        #line_traj_ground2
        if last > 0:
            idx_prev = self.idf - TRACE_BEFORE
            tx = gps_lon_vals[idx_prev]
            ty = gps_lat_vals[idx_prev]
            tz = gps_alt_vals[idx_prev]
            tx = [tx, tx]
            ty = [ty, ty]
            tz = [tz, deck]
            self.line_traj_ground2.set_data(tx, ty)
            self.line_traj_ground2.set_3d_properties(tz)

        # current point
        self.scatter_pt._offsets3d = ([row.gps_lon],[row.gps_lat],[row.gps_alt],)

        # HUD texts (screen-space)
        #self.text_alt.set_text(f"ALT {row.gps_alt:.0f} ft   FPM {row.gps_fpm:.0f}")
        #self.text_speed.set_text(f"SPD {row.gps_speed:.0f} km/h")

        cBOX = BOX / math.cos(math.radians(row.gps_lat))  # corrige en fonction de la latitude
        self.ax.set_xlim(row.gps_lon - cBOX*self.box_zoom, row.gps_lon + cBOX*self.box_zoom)  # latitude en X
        self.ax.set_ylim(row.gps_lat - BOX*self.box_zoom, row.gps_lat + BOX*self.box_zoom)  # longitude en Y
        self.ax.set_zlim(deck, sky)

        # update north vector (incremental update)
        x_main = [row.gps_lon - BOX, row.gps_lon - BOX]
        y_main = [row.gps_lat + BOX / 2, row.gps_lat + BOX]
        x_left = [row.gps_lon - BOX / 8 - BOX, row.gps_lon - BOX]
        y_left = [row.gps_lat - BOX / 8 + BOX, row.gps_lat + BOX]
        x_right = [row.gps_lon + BOX / 8 - BOX, row.gps_lon - BOX]
        y_right = [row.gps_lat - BOX / 8 + BOX, row.gps_lat + BOX]
        self.north_line_main.set_data(x_main, y_main)
        self.north_line_main.set_3d_properties([deck, deck])
        self.north_line_left.set_data(x_left, y_left)
        self.north_line_left.set_3d_properties([deck, deck])
        self.north_line_right.set_data(x_right, y_right)
        self.north_line_right.set_3d_properties([deck, deck])

        az = -int(row.gps_heading / 45) * 45 - 22.5
        if az!= self.last_azim:
            self.last_azim = az
            self.ax.view_init(azim=az)

        self.text_taille_box.set_text(f"Box Width {round((2 * self.box_zoom * BOX * 111320.0)/1000.0,2):.1f} km / Trace Length {TRACE/DF_FREQ:.0f}s")
        self.canvas.draw_idle()
        self.gps_lastrow=row



# ======================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

