import numpy as np
import pyqtgraph.opengl as gl
from stl import mesh

import math
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ---- Enable HiDPI scaling via environment variables BEFORE Qt loads ----
import os
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import av
import time
import numpy as np
import pandas as pd
import pygfx as gfx
from PyQt5.QtCore import QTimer, Qt, QElapsedTimer, QIODevice, QByteArray
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput
from PyQt5.QtWidgets import (
    QShortcut,   QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QFrame,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QAction,
    QSlider,
    QSizePolicy,
    QInputDialog
)
from pymediainfo import MediaInfo

from PyQt5.QtGui import QKeySequence

import pyqtgraph.opengl as gl

# ======================================================
# ScreenCaptureKit sample buffer handler
# ======================================================
import CoreMedia
import AVFoundation
import ScreenCaptureKit
from Cocoa import NSObject
import objc
# silence noisy PyObjC warnings produced when accessing CVPixelBuffer pointers
import warnings
from objc import ObjCPointerWarning


from ver import __version__


#***********************************************
#CONFIG
# MAJOR.MINOR.PATCH
MAINDIR="/Users/drax/Down/ABViewMain/"
BDL="data/Vol_2026_02_21.abv/"
#BDL="data/Vol_2026_03_20.abv/"
#BDL="data/Vol_2026_03_21.abv/"
PDL=MAINDIR+BDL
MERGED_DATA = PDL+"merged_data.csv"
VIDEO1=PDL+"front.mp4"
VIDEO2=PDL+"back.mp4"
BOOKMARK_FILE=PDL+"bookmark.csv"
STL_FILE=MAINDIR+"data/ressources/CAP10.STL"
STL_SIMPLE_PLANE_FILE=MAINDIR+"data/ressources/plane.STL"
BOX = 0.007*1.5 # taille box vision en °latitude
DF_FREQ = 100
TRACE = 6000 # taille de la trace 6000=1 minute
TRACE_DEFAULT = TRACE
TRACE_BEFORE = 500 # position précédente, 500 avant soit 5s
TRACE_SLICING_FACTOR = 50
VITESSE_MISE_EN_LIGNE = 80 #km/h
PITCH_MONTAGE_PAR_DEFAUT = 15 #camera verticale au repos par défaut, écran face à soi, légèrement inclinée vers soi
OFFSET_PITCH_SOL_PALLIER = 2 # différence de pitch entre sol et pallier vers 200kmh
R_recalage_repere=3 # données issues BB
#R_recalage_repere=1 # données issues computed VQF
refcam=[0,0,1] # données issues de BB
#refcam=[0,0,-1] # données issues computed VQF

warnings.filterwarnings("ignore", category=ObjCPointerWarning)

class SCStreamHandler(NSObject, protocols=[objc.protocolNamed("SCStreamOutput")]):

    def init(self):
        self = objc.super(SCStreamHandler, self).init()
        if self is None:
            return None
        self.writer = None
        self.input = None
        self.adaptor = None
        self.started = False
        return self

    def setWriter_input_adaptor_(self, writer, input, adaptor, audio_input=None):
        self.writer = writer
        self.input = input
        self.adaptor = adaptor
        self.audio_input = audio_input

    # called by ScreenCaptureKit for each buffer (video or audio)
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, outputType):
        if self.writer is None:
            return

        try:
            # start writer session on first received buffer
            if not self.started:
                pts = CoreMedia.CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                if self.writer.status() == AVFoundation.AVAssetWriterStatusWriting:
                    self.writer.startSessionAtSourceTime_(pts)
                    self.started = True
                    print("SCStream: first buffer received, session started")

            if not self.started:
                return

            # ---- VIDEO ----
            if outputType == ScreenCaptureKit.SCStreamOutputTypeScreen:
                if self.input is None or not self.input.isReadyForMoreMediaData():
                    return

                if not CoreMedia.CMSampleBufferIsValid(sampleBuffer):
                    return

                pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sampleBuffer)
                if pixel_buffer is None:
                    return

                pts = CoreMedia.CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

                ok = self.adaptor.appendPixelBuffer_withPresentationTime_(pixel_buffer, pts)
                if not ok:
                    status = self.writer.status()
                    print("appendPixelBuffer failed, writer status:", status)

            # ---- AUDIO ----
            elif outputType == ScreenCaptureKit.SCStreamOutputTypeAudio:
                if hasattr(self, "audio_input") and self.audio_input is not None:
                    if self.audio_input.isReadyForMoreMediaData():
                        try:
                            self.audio_input.appendSampleBuffer_(sampleBuffer)
                        except Exception:
                            pass

        except Exception as e:
            print("SCStream handler error:", e)

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
import os

# ---- ensure merged CSV exists before loading ----
if not os.path.exists(MERGED_DATA):
    print("ERROR: merged data file not found:")
    print("  ", MERGED_DATA)
    print("Current working directory:", os.getcwd())
    print("Hint: verify that the merged CSV was generated or that the 'data' folder path is correct.")
    sys.exit(1)

df = pd.read_csv(MERGED_DATA, low_memory=False)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
frames_df = len(df)

# ---- numpy caches for fast access inside the realtime loop ----
gps_lat_vals = df["gps_lat"].to_numpy()
gps_lon_vals = df["gps_lon"].to_numpy()
gps_alt_vals = df["gps_alt"].to_numpy()
timestamp_vals = df["timestamp"].to_numpy()

INPUT_METAR = BDL + "metar.csv"
metar_df = pd.read_csv(INPUT_METAR, encoding="utf-8")
#metar_df["time"] = pd.to_datetime(metar_df["time"])
metar_df["time"] = pd.to_datetime(metar_df["time"], format="mixed", utc=True)

def find_metar_for_time(df, t):

    idx = df["time"].searchsorted(t)

    if idx == 0:
        return df.iloc[0]

    if idx >= len(df):
        return df.iloc[-1]

    before = df.iloc[idx - 1]
    after = df.iloc[idx]

    if abs(t - before.time) < abs(after.time - t):
        return before
    else:
        return after

# ======================================================
# VIDEO ANALYSIS
# ======================================================
container_probe = av.open(VIDEO1)
stream_probe = container_probe.streams.video[0]

frames_video = stream_probe.frames
N = frames_video
container_probe.close()

# ======================================================
# TOP remarquables
# ======================================================
mask = df['gps_speed'] > VITESSE_MISE_EN_LIGNE
index_enligne_devol= mask.idxmax()
mask = df['gps_alt'] > 3000
index_entree_3000= mask.idxmax()
gps_max_alt = round(df['gps_alt'].max())
print("Max Alt : ",gps_max_alt)
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
            pen = QPen(QColor(255, 220, 0))
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
            pen = QPen(QColor(255, 220, 0))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawPolygon(triangle)

        painter.end()


# ======================================================
# Analog Badin (Circular Airspeed Indicator)
# ======================================================
class AnalogBadin(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")
        self.speed = 0.0

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QPen, QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        r = min(w, h) // 2 - 5
        cx = w // 2
        cy = h // 2

        # outer circle
        pen = QPen(QColor("white"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # scale and colored arcs
        max_speed = 360

        from PyQt5.QtCore import QRect

        def angle_from_speed(v):
            return (v / max_speed) * 340 - 135

        rect = QRect(cx - r + 6, cy - r + 6, (r - 6) * 2, (r - 6) * 2)

        # ---- green arc (0‑300 km/h) ----
        pen = QPen(QColor(0, 200, 0))
        pen.setWidth(6)
        painter.setPen(pen)
        start = angle_from_speed(0)
        end = angle_from_speed(300)
        painter.drawArc(rect, int(-end * 16), int((end - start) * 16))

        # ---- yellow arc (300‑340 km/h) ----
        pen = QPen(QColor(255, 200, 0))
        pen.setWidth(6)
        painter.setPen(pen)
        start = angle_from_speed(300)
        end = angle_from_speed(340)
        painter.drawArc(rect, int(-end * 16), int((end - start) * 16))

        # ---- red arc (340‑360 km/h) ----
        pen = QPen(QColor(220, 0, 0))
        pen.setWidth(6)
        painter.setPen(pen)
        start = angle_from_speed(340)
        end = angle_from_speed(360)
        painter.drawArc(rect, int(-end * 16), int((end - start) * 16))

        # ---- tick marks every 20 km/h (major ticks thicker) ----
        for v in range(0, max_speed + 1, 10):
            angle = angle_from_speed(v)
            rad = math.radians(angle)

            x1 = cx + (r - 12) * math.cos(rad)
            y1 = cy + (r - 12) * math.sin(rad)
            x2 = cx + r * math.cos(rad)
            y2 = cy + r * math.sin(rad)

            # major ticks
            if v in (50, 100, 150, 200, 250, 300):
                pen = QPen(QColor("white"))
                pen.setWidth(4)
            else:
                pen = QPen(QColor("white"))
                pen.setWidth(1)

            painter.setPen(pen)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # ---- numeric labels ----
        painter.setPen(QPen(QColor("white")))
        for v in (50, 100, 150, 200, 250, 300):
            angle = angle_from_speed(v)
            rad = math.radians(angle)
            xt = cx + (r - 30) * math.cos(rad)
            yt = cy + (r - 30) * math.sin(rad)
            painter.drawText(int(xt - 10), int(yt + 5), str(v))

        # needle
        v = max(0, min(self.speed, max_speed))
        angle = angle_from_speed(v)
        rad = math.radians(angle)

        pen = QPen(QColor("white"))
        pen.setWidth(3)
        painter.setPen(pen)

        x = cx + (r - 15) * math.cos(rad)
        y = cy + (r - 15) * math.sin(rad)
        painter.drawLine(cx, cy, int(x), int(y))

        painter.end()

# ======================================================
# Analog Altimeter (Circular Altimeter with Two Needles)
# ======================================================
class AnalogAltimeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")
        self.alt = 0.0

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QPen, QColor

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        r = min(w, h) // 2 - 5
        cx = w // 2
        cy = h // 2

        # outer circle
        pen = QPen(QColor("white"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # graduations every 100 ft (0-900)
        for i in range(0, 1000, 100):
            angle = (i / 1000.0) * 360.0
            rad = math.radians(angle - 90)
            x1 = cx + (r - 12) * math.cos(rad)
            y1 = cy + (r - 12) * math.sin(rad)
            x2 = cx + r * math.cos(rad)
            y2 = cy + r * math.sin(rad)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # ---- numeric labels (0‑9 thousands scale) ----
        painter.setPen(QPen(QColor("white")))
        for i in range(10):
            angle = (i / 10.0) * 360.0
            rad = math.radians(angle - 90)

            xt = cx + (r - 28) * math.cos(rad)
            yt = cy + (r - 28) * math.sin(rad)

            painter.drawText(int(xt - 6), int(yt + 6), str(i))

        # ---- Large needle (hundreds of feet) ----
        alt1000 = self.alt % 1000
        angle_big = (alt1000 / 1000.0) * 360.0
        rad = math.radians(angle_big - 90)

        pen = QPen(QColor("white"))
        pen.setWidth(2)
        painter.setPen(pen)

        x = cx + (r - 15) * math.cos(rad)
        y = cy + (r - 15) * math.sin(rad)
        painter.drawLine(cx, cy, int(x), int(y))

        # ---- Small needle (thousands of feet) ----
        alt10000 = (self.alt / 1000.0) % 10
        angle_small = (alt10000 / 10.0) * 360.0
        rad = math.radians(angle_small - 90)

        pen = QPen(QColor("white"))
        pen.setWidth(5)
        painter.setPen(pen)

        x = cx + (r - 30) * math.cos(rad)
        y = cy + (r - 30) * math.sin(rad)
        painter.drawLine(cx, cy, int(x), int(y))
        painter.end()

# ======================================================
# Main Window
# ======================================================
class MainWindow(QMainWindow):
    #def create_cap10_item(self):
    #    scale = 0.2

    #    pts = np.array([
    #        [1.0, 0.0, 0.0],  # nez
    #        [-1.0, 0.0, 0.0],  # queue
    #        [0.0, 1.2, 0.0],  # aile gauche
    #        [0.0, -1.2, 0.0],  # aile droite

    #        [-0.9, 0.5, 0.0],  # empennage G
    #        [-0.9, -0.5, 0.0],  # empennage D

    #        [-0.9, 0.0, 0.5],  # dérive
    #        ], dtype=float) * scale

    #    segments = np.array([
    #        pts[0], pts[1],
    #        pts[2], pts[3],
    #        pts[4], pts[5],
    #        pts[1], pts[6],
    #    ])

    #    return gl.GLLinePlotItem(
    #        pos=segments,
    #        #color=(1, 0.8, 0, 1), # jaune
    #        color=(0, 0, 0, 1), #black
    #        width=30,
    #        antialias=True,
    #        mode='lines')

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

        # screen capture process
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
        self.sync_enabled = False
        self.startup_time_ms = None
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

            self.audio_buffer = bytearray()
            self.audio_started = True
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
        # ---- smoothed values for instruments (visual interpolation) ----
        self.smooth_speed = None
        self.smooth_alt = None
        self.instrument_alpha = 0.2  # smoothing factor (0=slow, 1=no smoothing)

        # ---- filtered acceleration vector for smoother G trail ----
        self.acc_vec_filtered = None
        self.g_filter_alpha = 0.15
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

        self.init_UI()
        # jump to "mise en ligne" at startup
        QTimer.singleShot(0, self.goto_mise_en_ligne)

        self.init_map_OSM_widget()
        self.map_view.loadFinished.connect(self.on_map_loaded)

        self.enable_matplotlib_gps = True
        self.init_gps_pyqtgraph()

        self.init_gfx()
        self.calibrate_gfx(0) # calibration auto en considérant qu'on démarre au sol

        # ---- realtime timer (compensated loop) ----
        self.target_fps = 30
        self.frame_period_ms = 1000 / self.target_fps

        self.clock = QElapsedTimer()
        self.clock.start()

        # absolute frame scheduling to avoid timing drift
        self.next_frame_time = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.main_loop)
        self.timer.start(1)  # controlled manually

        # Metar management
        t_start = df['timestamp'][0]
        metar_row = find_metar_for_time(metar_df, t_start)
        self.last_metar=metar_row.metar
        #print(f"Metar at start: {self.last_metar}")

    def init_UI(self):
        # ---- UI ----
        central = QWidget()
        #central.setStyleSheet("background-color: white;")  # fond gris
        self.setCentralWidget(central)
        self.layout = QVBoxLayout(central)

        # ---- Menu ----
        menubar = self.menuBar()
        menu_fichier = menubar.addMenu("Fichier")
        menu_navigation = menubar.addMenu("Navigation")
        self.menu_bookmarks = menubar.addMenu("Bookmarks")
        menu_settings = menubar.addMenu("Settings")

        # Actions
        act_play_pause = QAction("Lecture / Pause", self)
        act_play_pause.setShortcut("Space")
        act_play_pause.triggered.connect(self.toggle_play)
        menu_navigation.addAction(act_play_pause)  # 👈 en premier
        menu_fichier.addAction(act_play_pause)

        act_quitter = QAction("Quitter", self)
        act_quitter.setShortcut("Ctrl+W")
        act_quitter.setMenuRole(QAction.NoRole)  # 👈 IMPORTANT
        act_quitter.triggered.connect(self.close)
        menu_fichier.addAction(act_quitter)

        # ---- Time navigation actions (with shortcuts shown) ----
        act_fwd_10 = QAction("Avance +10s (→)", self)
        act_fwd_10.setShortcut(QKeySequence("Right"))
        act_fwd_10.triggered.connect(self.jump_fwd_10s)
        menu_navigation.addAction(act_fwd_10)

        act_back_10 = QAction("Recule -10s (←)", self)
        act_back_10.setShortcut(QKeySequence("Left"))
        act_back_10.triggered.connect(self.jump_back_10s)
        menu_navigation.addAction(act_back_10)

        act_fwd_2 = QAction("Avance +2s (Shift+→)", self)
        act_fwd_2.setShortcut(QKeySequence("Shift+Right"))
        act_fwd_2.triggered.connect(self.jump_fwd_2s)
        menu_navigation.addAction(act_fwd_2)

        act_back_2 = QAction("Recule -2s (Shift+←)", self)
        act_back_2.setShortcut(QKeySequence("Shift+Left"))
        act_back_2.triggered.connect(self.jump_back_2s)
        menu_navigation.addAction(act_back_2)

        menu_navigation.addSeparator()
        # Add Next Bookmark action in Navigation menu (with shortcut shown)
        act_next_bm = QAction("Next Bookmark (Ctrl+→)", self)
        act_next_bm.setShortcut(QKeySequence("Ctrl+Right"))
        act_next_bm.triggered.connect(self.goto_next_bookmark)
        menu_navigation.addAction(act_next_bm)

        # Add Previous Bookmark action in Navigation menu (with shortcut shown)
        act_prev_bm = QAction("Previous Bookmark (Ctrl+←)", self)
        act_prev_bm.setShortcut(QKeySequence("Ctrl+Left"))
        act_prev_bm.triggered.connect(self.goto_previous_bookmark)
        menu_navigation.addAction(act_prev_bm)

        act_palier = QAction("Palier", self)
        act_palier.triggered.connect(self.seek_palier)
        menu_navigation.addAction(act_palier)

        act_mise_en_ligne = QAction("Mise en ligne", self)
        act_mise_en_ligne.triggered.connect(self.goto_mise_en_ligne)
        menu_navigation.addAction(act_mise_en_ligne)

        # ---- Shortcuts ----

        shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self.add_bookmark)

        shortcut = QShortcut(QKeySequence("Ctrl+W"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self.close)


        shortcut = QShortcut(QKeySequence("Ctrl+Right"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self.goto_next_bookmark)

        shortcut = QShortcut(QKeySequence("Ctrl+Left"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self.goto_previous_bookmark)

        shortcut = QShortcut(QKeySequence("Right"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self.jump_fwd_10s)

        shortcut = QShortcut(QKeySequence("Shift+Right"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self.jump_fwd_2s)

        shortcut = QShortcut(QKeySequence("Shift+Left"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self.jump_back_2s)

        # ---- Reload bookmarks CSV ----
        self.act_reload_bookmarks = QAction("Recharger CSV", self)
        self.act_reload_bookmarks.triggered.connect(self.reload_bookmarks)

        # ---- Add bookmark ----
        self.act_add_bookmark = QAction("Ajouter Bookmark", self)
        self.act_add_bookmark.setShortcut("Ctrl+B")
        self.act_add_bookmark.triggered.connect(self.add_bookmark)

        # build base menu structure
        self.menu_bookmarks.addAction(self.act_reload_bookmarks)
        self.menu_bookmarks.addSeparator()
        self.menu_bookmarks.addAction(self.act_add_bookmark)
        self.menu_bookmarks.addSeparator()
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

        # ---- Secondary GPS heading overlay (just below main heading) ----
        self.video1_heading_deviation_label = QLabel("", self.video1)
        self.video1_heading_deviation_label.setAlignment(Qt.AlignCenter)
        self.video1_heading_deviation_label.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 16px; font-weight: bold;"
        )
        self.video1_heading_deviation_label.adjustSize()
        self.video1_heading_deviation_label.raise_()

        # ---- Pitch overlay on video1 (bottom center, text sized) ----
        self.video1_pitch_label = QLabel("", self.video1)
        self.video1_pitch_label.setAlignment(Qt.AlignCenter)
        self.video1_pitch_label.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 18px; font-weight: bold;"
        )
        self.video1_pitch_label.adjustSize()
        self.video1_pitch_label.raise_()

        # ---- Bank overlay on video1 (above pitch) ----
        self.video1_bank_label = QLabel("", self.video1)
        self.video1_bank_label.setAlignment(Qt.AlignCenter)
        self.video1_bank_label.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 18px; font-weight: bold;"
        )
        self.video1_bank_label.adjustSize()
        self.video1_bank_label.raise_()

        # ---- Badin (GPS speed) overlay on video1 (bottom-left) ----
        self.video1_speed_label = QLabel("", self.video1)
        self.video1_speed_label.setAlignment(Qt.AlignCenter)
        self.video1_speed_label.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 18px; font-weight: bold;"
        )
        self.video1_speed_label.adjustSize()
        self.video1_speed_label.raise_()

        # ---- Analog badin (circular airspeed indicator) ----
        self.video1_badin = AnalogBadin(self.video1)
        self.video1_badin.setGeometry(10, int(self.video1.height()/2) - 80, 160, 160)
        self.video1_badin.show()

        # ---- Analog altimeter (right side) ----
        self.video1_altimeter = AnalogAltimeter(self.video1)
        self.video1_altimeter.setGeometry(self.video1.width() - 170, int(self.video1.height()/2) - 80, 160, 160)
        self.video1_altimeter.show()

        # ---- GPS altitude overlay on video1 (bottom-right) ----
        self.video1_alt_label = QLabel("", self.video1)
        self.video1_alt_label.setAlignment(Qt.AlignCenter)
        self.video1_alt_label.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 18px; font-weight: bold;"
        )
        self.video1_alt_label.adjustSize()
        self.video1_alt_label.raise_()

        # ---- GPS vario overlay on video1 (above altitude) ----
        self.video1_fpm_label = QLabel("", self.video1)
        self.video1_fpm_label.setAlignment(Qt.AlignCenter)
        self.video1_fpm_label.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 18px; font-weight: bold;"
        )
        self.video1_fpm_label.adjustSize()
        self.video1_fpm_label.raise_()
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

        # ---- Toggle 3D axes visibility ----
        self.act_toggle_axes = QAction("Afficher axes 3D", self)
        self.act_toggle_axes.setCheckable(True)
        self.act_toggle_axes.setChecked(False)
        self.act_toggle_axes.triggered.connect(self.toggle_axes_visibility)
        menu_settings.addAction(self.act_toggle_axes)

        # ---- Recalibrate camera mounting from current frame ----
        self.act_recalibrate_sol = QAction("Recalibrer Sol", self)
        self.act_recalibrate_sol.triggered.connect(self.calibrate_gfx_on_current_frame)
        menu_settings.addAction(self.act_recalibrate_sol)

        # ---- Camera mounting pitch adjustment ----
        self.act_pitch_cam_plus = QAction("Pitch Montage Caméra +1°", self)
        self.act_pitch_cam_plus.triggered.connect(self.pitch_cam_plus)
        menu_settings.addAction(self.act_pitch_cam_plus)

        self.act_pitch_cam_minus = QAction("Pitch Montage Caméra -1°", self)
        self.act_pitch_cam_minus.triggered.connect(self.pitch_cam_minus)
        menu_settings.addAction(self.act_pitch_cam_minus)

        # display current value in menu text
        self.update_pitch_cam_menu()

        # ---- slider + controls ----
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, N - 1)
        self.slider.valueChanged.connect(self.on_slider)

        self.timestamp_label = QLabel(alignment=Qt.AlignCenter)

        # ---- Elapsed time overlay (top center main window) ----
        self.elapsed_time_overlay = QLabel("", self.centralWidget())
        self.elapsed_time_overlay.setAlignment(Qt.AlignCenter)
        self.elapsed_time_overlay.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 18px; font-weight: bold;"
        )
        self.elapsed_time_overlay.adjustSize()
        self.elapsed_time_overlay.raise_()


        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.clicked.connect(self.toggle_play)

        # ---- Screen recording button (ScreenCaptureKit bridge) ----
        self.btn_record = QPushButton("● REC")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_recording)


        # ---- Open pygfx window in separate window (overlay top-center) ----
        # temporary creation without parent (gfx not yet initialized)
        self.btn_detach_gfx = QPushButton("↗Detach")
        self.btn_detach_gfx.clicked.connect(self.detach_gfx_window)

        # ---- Open Video1 in separate window ----
        self.btn_detach_video1 = QPushButton("↗Detach", self.video1)
        self.btn_detach_video1.move(10, 10)
        self.btn_detach_video1.raise_()
        self.btn_detach_video1.clicked.connect(self.detach_video1_window)

        # ---- Open Video2 in separate window (overlay top-right) ----
        self.btn_detach_video2 = QPushButton("↗Detach", self.video2)
        self.btn_detach_video2.adjustSize()
        # initial position (will be corrected on resize)
        self.btn_detach_video2.move(self.video2.width() - self.btn_detach_video2.width() - 10, 10)
        QTimer.singleShot(0, lambda: self.btn_detach_video2.move(
            self.video2.contentsRect().width() - self.btn_detach_video2.width() - 10, 10
        ))
        self.btn_detach_video2.raise_()
        self.btn_detach_video2.clicked.connect(self.detach_video2_window)

        # ---- Open GPS window in separate window ----
        self.btn_detach_pyqtgraph = QPushButton("↗Detach")
        self.btn_detach_pyqtgraph.clicked.connect(self.detach_pyqtgraph_window)

        # ---- jump buttons (time navigation) ----
        self.btn_back_10 = QPushButton("⏪10s")
        self.btn_back_10.clicked.connect(self.jump_back_10s)

        self.btn_back_2 = QPushButton("◀2s")
        self.btn_back_2.clicked.connect(self.jump_back_2s)

        self.btn_fwd_2 = QPushButton("2s▶")
        self.btn_fwd_2.clicked.connect(self.jump_fwd_2s)

        self.btn_fwd_10 = QPushButton("10s⏩")
        self.btn_fwd_10.clicked.connect(self.jump_fwd_10s)


        self.btn_pallier = QPushButton("Palier")
        self.btn_pallier.clicked.connect(self.seek_palier)

        # bouton bookmark précédent
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(self.goto_previous_bookmark)

        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.goto_next_bookmark)

        self.btn_add_bookmark = QPushButton("Bookmark")
        self.btn_add_bookmark.clicked.connect(self.add_bookmark)

        self.btn_mise_en_ligne = QPushButton("En ligne")
        self.btn_mise_en_ligne.clicked.connect(self.goto_mise_en_ligne)

        self.btn_quitter = QPushButton("Quitter")
        self.btn_quitter.clicked.connect(self.close)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.btn_pause)
        buttons_layout.addWidget(self.btn_record)

        # self.btn_detach_gfx is now parented to self.gfx_canvas and positioned manually
        # self.btn_detach_video1 and self.btn_detach_video2 are now parented to their respective video labels and positioned manually
        buttons_layout.addWidget(self.btn_detach_pyqtgraph)
        buttons_layout.addWidget(self.btn_back_10)
        buttons_layout.addWidget(self.btn_back_2)
        buttons_layout.addWidget(self.btn_fwd_2)
        buttons_layout.addWidget(self.btn_fwd_10)
        buttons_layout.addWidget(self.btn_prev)
        buttons_layout.addWidget(self.btn_next)
        buttons_layout.addWidget(self.btn_add_bookmark)
        buttons_layout.addWidget(self.btn_pallier)
        buttons_layout.addWidget(self.btn_mise_en_ligne)
        buttons_layout.addWidget(self.btn_quitter)

        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.timestamp_label)
        self.layout.addLayout(buttons_layout)

        # ---- Now load bookmarks and update ticks, after slider is created ----
        self.load_bookmarks()
        self.update_bookmark_ticks()
        # ensure ticks are drawn after layout is finalized
        QTimer.singleShot(0, self.update_bookmark_ticks)

        self._position_elapsed_time_overlay()

        # ---- Previous bookmark overlay (just below elapsed time) ----
        self.prev_bookmark_overlay = QLabel("", self.centralWidget())
        self.prev_bookmark_overlay.setAlignment(Qt.AlignCenter)
        self.prev_bookmark_overlay.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 30px; font-weight: bold;"
        )
        self.prev_bookmark_overlay.adjustSize()
        self.prev_bookmark_overlay.raise_()

        # ---- Bookmark overlay label (top center) ----
        self.bookmark_overlay = QLabel("", self)
        self.bookmark_overlay.setStyleSheet(
            "color: yellow; "
            "background-color: rgba(0,0,0,180); "
            "padding: 12px; "
            "font-family: 'Menlo'; "
            "font-size: 50px; "
            "font-weight: bold;")
        self.bookmark_overlay.setAlignment(Qt.AlignCenter)
        self.bookmark_overlay.hide()

        self.bookmark_ticks = []

        # ---- PyQtGraph GPS 3D ----
        self.gps_view = gl.GLViewWidget()
        self.gps_view.setBackgroundColor('w')
        self.gps_view.setCameraPosition(distance=4)
        self.grid.addWidget(self.gps_view, 1, 2, 1, 1)

    def update_bookmark_ticks(self):
        # slider may not be initialized yet during early init_UI
        if not hasattr(self, "slider"):
            return

        # supprimer anciens ticks
        for t in getattr(self, "bookmark_ticks", []):
            try:
                t.deleteLater()
            except:
                pass

        self.bookmark_ticks = []

        if self.bookmarks_df is None or self.bookmarks_df.empty:
            return

        # use geometry in parent coordinates (more reliable)
        geom = self.slider.geometry()
        slider_w = geom.width()
        slider_x = geom.x()
        slider_y = geom.y()

        # if width is not ready yet, skip (will be recalled later)
        if slider_w == 0:
            return

        total = max(1, (N - 1))

        for _, row in self.bookmarks_df.iterrows():
            try:
                frame = int(row["frame"])
            except:
                continue

            t = frame / total
            x = int(slider_x + t * slider_w)

            tick = QFrame(self.centralWidget())
            tick.setStyleSheet("background-color: red;")
            tick.setGeometry(x, slider_y - 6, 2, 6)

            # tooltip (optionnel)
            tick.setToolTip(str(row.get("name", "")))

            tick.show()
            self.bookmark_ticks.append(tick)

    def _position_elapsed_time_overlay(self):
        if not hasattr(self, "elapsed_time_overlay"):
            return

        # IMPORTANT : forcer recalcul taille
        self.elapsed_time_overlay.adjustSize()

        # utiliser largeur de la fenêtre principale (plus fiable)
        w = self.width()

        x = (w - self.elapsed_time_overlay.width()) // 2
        y = 17

        self.elapsed_time_overlay.move(x, y)
        self.elapsed_time_overlay.raise_()

    def _position_elapsed_time_overlay(self):
        if not hasattr(self, "elapsed_time_overlay"):
            return

        self.elapsed_time_overlay.adjustSize()
        w = self.width()

        x = (w - self.elapsed_time_overlay.width()) // 2
        y = 17

        self.elapsed_time_overlay.move(x, y)
        self.elapsed_time_overlay.raise_()

        # ---- bookmark juste en dessous ----
        if hasattr(self, "prev_bookmark_overlay"):
            self.prev_bookmark_overlay.adjustSize()
            bx = (w - self.prev_bookmark_overlay.width()) // 2
            by = y + self.elapsed_time_overlay.height() + 5
            self.prev_bookmark_overlay.move(bx, by)
            self.prev_bookmark_overlay.raise_()

    # ==================================================
    # Screen recording using macOS ScreenCaptureKit (PyObjC)
    # ==================================================
    def toggle_recording(self, checked):
        import os
        try:
            import AVFoundation
            import ScreenCaptureKit
            import Foundation
            from Cocoa import NSObject
            import objc
        except Exception as e:
            print("ScreenCaptureKit frameworks not available:", e)
            self.btn_record.setChecked(False)
            return

        # create sequential recording filename: record_001.mp4, record_002.mp4, ...
        base_dir = "data"
        os.makedirs(base_dir, exist_ok=True)

        idx = 1
        while True:
            candidate = os.path.join(base_dir, f"record_{idx:03d}.mp4")
            if not os.path.exists(candidate):
                output_file = candidate
                break
            idx += 1

        print(output_file)

        # Each recording uses a new indexed filename, so no overwrite removal is needed.

        # lazy initialization of recorder
        if not hasattr(self, "sc_stream"):
            try:
                # PyObjC exposes ScreenCaptureKit async API via completion handler
                result_container = {"content": None, "error": None}

                def _handler(content, error):
                    result_container["content"] = content
                    result_container["error"] = error

                ScreenCaptureKit.SCShareableContent.getShareableContentWithCompletionHandler_(_handler)

                # wait briefly until callback fills the result
                import time
                for _ in range(50):
                    if result_container["content"] is not None or result_container["error"] is not None:
                        break
                    time.sleep(0.01)

                if result_container["error"] is not None:
                    raise RuntimeError(f"ScreenCaptureKit error: {result_container['error']}")

                content = result_container["content"]

                # --- Find our Qt window inside shareable windows ---
                target_window = None
                for w in content.windows():
                    try:
                        title = str(w.title())
                    except Exception:
                        title = ""
                    if title and title in self.windowTitle():
                        target_window = w
                        break

                # fallback: use first window if not found
                if target_window is None:
                    target_window = content.windows()[0]

                config = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
                config.setWidth_(target_window.frame().size.width)
                config.setHeight_(target_window.frame().size.height)
                config.setCapturesAudio_(True)

                # capture only this window (not the full display)
                filter = ScreenCaptureKit.SCContentFilter.alloc().initWithDesktopIndependentWindow_(target_window)

                self.sc_stream = ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(
                    filter, config, None
                )

                # create stream handler to receive frames
                self.sc_handler = SCStreamHandler.alloc().init()
                url = Foundation.NSURL.fileURLWithPath_(output_file)

                self.sc_writer = AVFoundation.AVAssetWriter.alloc().initWithURL_fileType_error_(
                    url, AVFoundation.AVFileTypeMPEG4, None
                )[0]

                settings = {
                    AVFoundation.AVVideoCodecKey: AVFoundation.AVVideoCodecTypeH264,
                    AVFoundation.AVVideoWidthKey: config.width(),
                    AVFoundation.AVVideoHeightKey: config.height(),
                }

                self.sc_input = AVFoundation.AVAssetWriterInput.alloc().initWithMediaType_outputSettings_(
                    AVFoundation.AVMediaTypeVideo, settings
                )

                # ---- audio writer input ----
                audio_settings = {
                    AVFoundation.AVFormatIDKey: 1633772320,  # kAudioFormatMPEG4AAC
                    AVFoundation.AVNumberOfChannelsKey: 2,
                    AVFoundation.AVSampleRateKey: 44100,
                    AVFoundation.AVEncoderBitRateKey: 128000,
                }

                self.sc_audio_input = AVFoundation.AVAssetWriterInput.alloc().initWithMediaType_outputSettings_(
                    AVFoundation.AVMediaTypeAudio,
                    audio_settings
                )

                self.sc_audio_input.setExpectsMediaDataInRealTime_(True)

                adaptor_attrs = {
                    "PixelFormatType": 1111970369  # kCVPixelFormatType_32BGRA
                }

                self.sc_adaptor = AVFoundation.AVAssetWriterInputPixelBufferAdaptor.alloc().initWithAssetWriterInput_sourcePixelBufferAttributes_(
                    self.sc_input, adaptor_attrs
                )

                # realtime capture configuration
                self.sc_input.setExpectsMediaDataInRealTime_(True)

                self.sc_writer.addInput_(self.sc_input)
                self.sc_writer.addInput_(self.sc_audio_input)
                self.sc_writer.startWriting()

                # set handler's writer/input/adaptor
                self.sc_handler.setWriter_input_adaptor_(self.sc_writer, self.sc_input, self.sc_adaptor, self.sc_audio_input)

                # ScreenCaptureKit requires a GCD dispatch queue
                import dispatch
                queue = dispatch.dispatch_get_main_queue()

                self.sc_stream.addStreamOutput_type_sampleHandlerQueue_error_(
                    self.sc_handler,
                    ScreenCaptureKit.SCStreamOutputTypeScreen,
                    queue,
                    None
                )
                self.sc_stream.addStreamOutput_type_sampleHandlerQueue_error_(
                    self.sc_handler,
                    ScreenCaptureKit.SCStreamOutputTypeAudio,
                    queue,
                    None
                )

            except Exception as e:
                print("ScreenCaptureKit init failed:", e)
                self.btn_record.setChecked(False)
                return

        if checked:
            try:
                self.sc_stream.startCaptureWithCompletionHandler_(None)
                self.btn_record.setText("■ REC")
                self.btn_record.setStyleSheet("background-color: red; color: white; font-weight: bold;")
                print("Recording start")
            except Exception as e:
                print("Recording start failed:", e)
                self.btn_record.setChecked(False)
        else:
            try:
                if hasattr(self, "sc_stream"):
                    self.sc_stream.stopCaptureWithCompletionHandler_(None)

                    try:
                        self.sc_input.markAsFinished()

                        # keep local reference because attributes will be deleted after stop
                        writer = self.sc_writer

                        # Apple recommends finishing AVAssetWriter off the main thread
                        import dispatch

                        def _finish():
                            try:
                                writer.finishWriting()
                                print("MP4 finalized")
                            except Exception as e:
                                print("finishWriting error:", e)

                        dispatch.dispatch_async(
                            dispatch.dispatch_get_global_queue(0, 0),
                            _finish
                        )

                    except Exception:
                        pass

                    print("Recording stop")
            except Exception:
                pass

            # fully reset capture pipeline so next recording starts cleanly
            try:
                for attr in ("sc_stream", "sc_writer", "sc_input", "sc_adaptor", "sc_handler"):
                    if hasattr(self, attr):
                        delattr(self, attr)
            except Exception:
                pass

            self.btn_record.setText("● REC")
            self.btn_record.setStyleSheet("")
    def update_pitch_cam_menu(self):
        """Update menu text showing current camera mounting pitch."""
        if hasattr(self, "act_pitch_cam_plus"):
            val = f"{self.montage_pitch_angle:.1f}°"
            self.act_pitch_cam_plus.setText(f"Pitch Montage Caméra +1° (actuel {val})")
            self.act_pitch_cam_minus.setText("Pitch Montage Caméra -1°")

    def pitch_cam_plus(self):
        """Increase camera mounting pitch by 1 degree."""
        self.montage_pitch_angle += 1
        self.update_pitch_cam_menu()

        # refresh display immediately even when paused
        if hasattr(self, "row"):
            try:
                self.update_gfx_orientation()
            except Exception:
                pass

    def pitch_cam_minus(self):
        """Decrease camera mounting pitch by 1 degree."""
        self.montage_pitch_angle -= 1
        self.update_pitch_cam_menu()

        # refresh display immediately even when paused
        if hasattr(self, "row"):
            try:
                self.update_gfx_orientation()
            except Exception:
                pass

    def toggle_axes_visibility(self):
        """Show or hide the pygfx world axes."""
        visible = self.act_toggle_axes.isChecked()

        if hasattr(self, "gfx_axes_x"):
            self.gfx_axes_x.visible = visible
            self.gfx_axes_y.visible = visible
            self.gfx_axes_z.visible = visible

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
                    var map = L.map('map').setView([{lat0}, {lon0}], 12);

                    map.options.maxZoom = 18;
                    map.options.minZoom = 8;

                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                      maxZoom: 19,
                      attribution: '&copy; OpenStreetMap contributors'
                    }}).addTo(map);

                    // ---- Plane icon (rotating with heading) ----
                    var planeIcon = L.divIcon({{
                        html: '<div id="plane" style="transform: rotate(0deg); font-size:24px;">✈️</div>',
                        className: '',
                        iconSize: [24, 24],
                        iconAnchor: [12, 12]
                    }});
                    
                    var marker = L.marker([{lat0}, {lon0}], {{ icon: planeIcon }}).addTo(map);
                    
                    
                    // ---- Trajectory (last 1 minute) ----
                    var trajectory = [];
                    var maxPoints = 600; // assuming ~10 Hz → ~1 minute

                    var polyline = L.polyline([], {{
                        color: 'blue',
                        weight: 3
                    }}).addTo(map);
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

                    function updateMarker(lat, lon, heading) {{
                        var point = [lat, lon];

                        // add new point
                        trajectory.push(point);

                        // keep only last minute of data
                        if (trajectory.length > maxPoints) {{
                            trajectory.shift();
                        }}

                        // update polyline
                        polyline.setLatLngs(trajectory);

                        // update marker
                        marker.setLatLng(point);
                        // rotate plane icon
                        var plane = document.getElementById("plane");
                        if (plane && heading !== undefined) {{
                            plane.style.transform = "rotate(" + heading + "deg)";
                        }}
                        map.panTo(point, {{ animate: false }});
                    }}

                    window.updateMarker = updateMarker;

                    // ---- Reset trajectory (used after seek) ----
                    function resetTrajectory() {{
                        trajectory = [];
                        polyline.setLatLngs([]);
                    }}

                    // ---- Reset trajectory with history (used after seek) ----
                    function resetTrajectoryWithData(points) {{
                        trajectory = points || [];
                        polyline.setLatLngs(trajectory);
                    }}

                    window.resetTrajectoryWithData = resetTrajectoryWithData;
                    

                    window.resetTrajectory = resetTrajectory;
                  </script>
                </body>
                </html>
                """)
        self.grid.addWidget(self.map_view, 1, 3, 1, 1)
        # ---- METAR overlay (bottom of OSM map) ----
        # attach overlay to the central widget so it can appear above the grid layout
        parent_widget = self.centralWidget() if self.centralWidget() is not None else self
        self.map_metar_label = QLabel("", parent_widget)
        self.map_metar_label.setAlignment(Qt.AlignCenter)
        self.map_metar_label.setWordWrap(True)
        self.map_metar_label.setStyleSheet(
            "color: black; background-color: white; padding: 6px; font-family: 'Menlo'; font-size: 14px; font-weight: bold;"
        )
        self.map_metar_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.map_metar_label.setText(getattr(self, "last_metar", ""))
        self.map_metar_label.adjustSize()
        self.map_metar_label.raise_()
        # initial position of METAR label (bottom center of map)
        self._position_map_metar_label()

    def _position_map_metar_label(self):
        """Position the METAR label centered at the bottom of the OSM map."""
        if not hasattr(self, "map_metar_label") or not hasattr(self, "map_view"):
            return

        geom = self.map_view.geometry()

        # limit label width to map width so text wraps instead of overflowing
        self.map_metar_label.setMaximumWidth(geom.width() - 20)
        self.map_metar_label.adjustSize()

        # map_view geometry is relative to the central widget
        x = geom.x() + (geom.width() - self.map_metar_label.width()) // 2
        y = geom.y() + geom.height() - self.map_metar_label.height() - 10

        self.map_metar_label.move(x, y)
        self.map_metar_label.raise_()
        self.map_metar_label.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "map_metar_label"):
            self._position_map_metar_label()

        if hasattr(self, "elapsed_time_overlay"):
            self._position_elapsed_time_overlay()

        self.update_bookmark_ticks()

        # Keep btn_detach_video2 pinned to top-right of video2
        if hasattr(self, "btn_detach_video2") and hasattr(self, "video2"):
            self.btn_detach_video2.adjustSize()

            rect = self.video2.contentsRect()
            x = rect.width() - self.btn_detach_video2.width() - 10
            y = 10

            self.btn_detach_video2.move(x, y)
            self.btn_detach_video2.raise_()

        # Keep btn_detach_gfx pinned to bottom-center of pygfx canvas
        if hasattr(self, "btn_detach_gfx") and hasattr(self, "gfx_canvas"):
            self.btn_detach_gfx.adjustSize()
            rect = self.gfx_canvas.contentsRect()
            x = (rect.width() - self.btn_detach_gfx.width()) // 2
            y = rect.height() - self.btn_detach_gfx.height() - 10
            self.btn_detach_gfx.move(x, y)
            self.btn_detach_gfx.raise_()

        # Keep btn_detach_pyqtgraph pinned to bottom-left of gps_view
        if hasattr(self, "btn_detach_pyqtgraph") and hasattr(self, "gps_view"):
            self.btn_detach_pyqtgraph.adjustSize()

            rect = self.gps_view.contentsRect()
            x = 10
            y = rect.height() - self.btn_detach_pyqtgraph.height() - 50

            self.btn_detach_pyqtgraph.move(x, y)
            self.btn_detach_pyqtgraph.raise_()

    def init_gps_pyqtgraph(self):

        # ---- trajectory rendered as a circular bundle (tube-like) ----
        self.gps_lines = []

        tube_radius = 0.006
        tube_segments = 12  # number of lines around the tube

        offsets = []
        for k in range(tube_segments):
            a = 2 * np.pi * k / tube_segments
            offsets.append((
                tube_radius * np.cos(a),
                tube_radius * np.sin(a),
                0.0
            ))

        for off in offsets:
            line = gl.GLLinePlotItem(
                pos=np.zeros((2, 3)),
                color=(1, 1, 1, 1),
                width=8,
                antialias=True
            )
            line._tube_offset = np.array(off)
            self.gps_view.addItem(line)
            self.gps_lines.append(line)

        # aircraft position marker
        self.gps_point = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            size=15,              # screen pixels
            pxMode=True,          # keep constant size on screen
            color=(1.0, 0.0, 0.0, 1.0)
        )

        # ensure the point is rendered normally and not blended like a large sphere
        self.gps_point.setGLOptions('opaque')

        # ground grid (visual reference in meters)
        grid = gl.GLGridItem()
        grid.setSize(2, 2)   # 2 km x 2 km area
        grid.setSpacing(0.25, 0.25)  # grid every 100 m
        grid.translate(0, 0, -1)  # slightly below aircraft
        grid.setColor((150,150,150))
        self.gps_view.addItem(grid)

        # ---- vertical grid (YZ plane) ----
        self.grid_vertical_yz = gl.GLGridItem()
        self.grid_vertical_yz.setSize(2, 2)
        self.grid_vertical_yz.setSpacing(0.25,0.25)
        self.grid_vertical_yz.rotate(90, 1, 0, 0)
        self.grid_vertical_yz.translate(0, -1, 0)
        self.grid_vertical_yz.setColor((135, 206, 235))
        self.gps_view.addItem(self.grid_vertical_yz)

        # ---- vertical grid (XZ plane) ----
        self.grid_vertical_xz = gl.GLGridItem()
        self.grid_vertical_xz.setSize(2, 2)
        self.grid_vertical_xz.setSpacing(0.25, 0.25)
        self.grid_vertical_xz.rotate(90, 0, 1, 0)
        self.grid_vertical_xz.translate(-1,0, 0)
        self.grid_vertical_xz.setColor((135, 206, 235))
        self.gps_view.addItem(self.grid_vertical_xz)

        # self.gps_view.addItem(self.gps_line)  # REMOVED
        #self.gps_view.addItem(self.gps_point)


        #stl_path = os.path.join(os.path.dirname(__file__), "plane.stl")

        try:
            m = mesh.Mesh.from_file(STL_SIMPLE_PLANE_FILE)
            # supprimer triangles dégénérés
            v = m.vectors
            valid = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) > 1e-6
            v = v[valid]
            vertices = v.reshape(-1, 3)

            vertices *= 0.03  # Scale down
            vertices -= vertices.mean(axis=0) # centrage
            # rotation axes
            R_fix = np.array([
                [0, 1, 0],  # X = nez
                [1, 0, 0],  # Y = ailes
                [0, 0, 1]])
            vertices = (R_fix @ vertices.T).T
            # inversion gauche/droite
            vertices[:, 1] *= -1

            faces = np.arange(len(vertices)).reshape(-1, 3)
            meshdata = gl.MeshData(vertexes=vertices, faces=faces)
            self.gps_aircraft = gl.GLMeshItem(
                meshdata=meshdata,
                smooth=False,
                color=(0.8, 0.8, 0.8, 0.2),  # gris clair
                shader='shaded',
                drawEdges=False,
                glOptions='translucent')

        except Exception as e:
            print("STL load failed, fallback to CAP10:", e)
            self.gps_aircraft = self.create_cap10_item()
        self.gps_view.addItem(self.gps_aircraft)

        # base geometry for rotation (handle both LinePlotItem and MeshItem)
        if hasattr(self.gps_aircraft, "pos"):
            self.gps_aircraft_base = self.gps_aircraft.pos.copy().reshape(-1, 3)
        else:
            md = self.gps_aircraft.opts.get("meshdata", None)
            if md is not None:
                self.gps_aircraft_base = md.vertexes().copy()
            else:
                raise RuntimeError("No meshdata found in GLMeshItem")


        # vertical line from aircraft to ground
        self.gps_vertical_line = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            color=(0, 0, 1, 1),
            width=3,
            antialias=True
        )
        self.gps_view.addItem(self.gps_vertical_line)

        # ---- ground projection of trajectory (shadow on ground) ----
        self.gps_shadow = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            #color=(0, 0, 1, 0.6), # gray
            color=(0, 0, 0, 1), #black
            width=2,
            antialias=True
        )
        self.gps_view.addItem(self.gps_shadow)


        # ---- altitude scale overlay (Red Bull style vertical scale) ----
        self.altitude_scale_labels = []
        altitude_scale = list(range(0, max(5500, int(math.ceil(gps_max_alt / 500) * 500)) + 500, 500))

        for z in altitude_scale:
            label = QLabel(f"{z}ft", self.gps_view)
            label.setStyleSheet("color: black; background-color: transparent; padding:2px; font-family:'Menlo'; font-size:10px;")
            label.adjustSize()
            label.show()
            label.raise_()
            self.altitude_scale_labels.append((z, label))

        # small horizontal ticks for altitude graduations
        self.altitude_scale_ticks = []
        for _ in altitude_scale:
            tick = QFrame(self.gps_view)
            tick.setStyleSheet("background-color: black;")
            tick.setGeometry(0, 0, 6, 2)
            tick.show()
            self.altitude_scale_ticks.append(tick)

        # ---- vertical altitude bar ----
        self.altitude_bar = QFrame(self.gps_view)
        self.altitude_bar.setStyleSheet("background-color: rgba(128,128,128,80);")
        self.altitude_bar.setGeometry(0, 0, 4, 200)
        self.altitude_bar.show()

        self.altitude_vario_bar = QFrame(self.gps_view)
        self.altitude_vario_bar.setStyleSheet("background-color: white;")
        self.altitude_vario_bar.setGeometry(0, 0, 3, 0)
        self.altitude_vario_bar.show()

        # ---- altitude colored zones (Red Bull style) ----
        self.altitude_green_zone = QFrame(self.gps_view)
        self.altitude_green_zone.setStyleSheet("background-color: rgba(0,120,0,160);")  # dark green
        self.altitude_green_zone.show()

        self.altitude_orange_zone = QFrame(self.gps_view)
        self.altitude_orange_zone.setStyleSheet("background-color: rgba(128,128,128,80);")  # grey above 5000 ft
        self.altitude_orange_zone.show()

        # moving marker showing current altitude (rectangle cursor like speed bar)
        self.altitude_cursor = QFrame(self.gps_view)
        self.altitude_cursor.setStyleSheet("background-color: black;")
        self.altitude_cursor.setGeometry(0, 0, 18, 8)
        self.altitude_cursor.show()

        # blinking state for fast altitude change
        self.altitude_cursor_visible = True
        self.altitude_last_blink = 0


        # ---- horizontal speed bar (top of GPS view) ----
        self.speed_bar = QFrame(self.gps_view)
        self.speed_bar.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 rgba(0,120,255,200),"      # blue below 100 km/h
            " stop:0.25 rgba(0,120,255,200),"   # 100/400
            " stop:0.25 rgba(0,180,0,180),"
            " stop:0.75 rgba(0,180,0,180),"     # up to 300 km/h
            " stop:0.75 rgba(255,220,0,200),"
            " stop:0.85 rgba(255,220,0,200),"   # 300–340 km/h
            " stop:0.85 rgba(255,0,0,200),"
            " stop:1 rgba(255,0,0,200));"
        )
        self.speed_bar.setGeometry(0, 0, 200, 6)
        self.speed_bar.show()

        # moving cursor for speed
        self.speed_cursor = QFrame(self.gps_view)
        self.speed_cursor.setStyleSheet("background-color: red;")
        self.speed_cursor.setGeometry(0, 0, 10, 16)
        self.speed_cursor.show()

        # ---- speed scale graduations ----
        self.speed_scale_labels = []
        for v in (50, 100, 150, 200, 250, 300, 350, 400):
            text = f"{v} km/h" if v == 50 else str(v)
            label = QLabel(text, self.gps_view)
            label.setStyleSheet(
                "color: black; background-color: transparent; font-family:'Menlo'; font-size:10px;"
            )
            label.adjustSize()
            label.show()
            label.raise_()
            self.speed_scale_labels.append((v, label))

        # small vertical ticks under each speed graduation
        self.speed_scale_ticks = []
        for _ in (50, 100, 150, 200, 250, 300, 350, 400):
            tick = QFrame(self.gps_view)
            tick.setStyleSheet("background-color: black;")
            tick.setGeometry(0, 0, 2, 6)
            tick.show()
            self.speed_scale_ticks.append(tick)

        # ---- G force bar (bottom of GPS view) ----
        self.g_bar = QFrame(self.gps_view)
        self.g_bar.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 rgba(0,120,255,200),"      # -2G
            " stop:0.25 rgba(0,120,255,200),"   # -1G
            " stop:0.40 rgba(0,200,0,200),"     # 0G
            " stop:0.55 rgba(0,200,0,200),"     # 1G
            " stop:0.65 rgba(255,220,0,200),"   # 2G
            " stop:0.85 rgba(255,0,0,220),"     # 3G
            " stop:1 rgba(160,0,255,220));"     # 4G max
        )
        self.g_bar.setGeometry(0, 0, 200, 6)
        self.g_bar.show()

        # moving cursor for G force (same style as speed cursor)
        self.g_cursor = QFrame(self.gps_view)
        self.g_cursor.setStyleSheet("background-color: red;")
        self.g_cursor.setGeometry(0, 0, 10, 16)
        self.g_cursor.show()

        # ---- G scale graduations ----
        self.g_scale_labels = []
        for g in (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6):
            text = "1G" if g == 1 else str(g)
            label = QLabel(text, self.gps_view)
            label.setStyleSheet(
                "color: black; background-color: transparent; font-family:'Menlo'; font-size:10px;"
            )
            label.adjustSize()
            label.show()
            label.raise_()
            self.g_scale_labels.append((g, label))

        # vertical ticks for G scale
        self.g_scale_ticks = []
        for _ in (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6):
            tick = QFrame(self.gps_view)
            tick.setStyleSheet("background-color: black;")
            tick.setGeometry(0, 0, 2, 6)
            tick.show()
            self.g_scale_ticks.append(tick)

        # ---- Attach detach button to gps_view (bottom-left) ----
        if hasattr(self, "btn_detach_pyqtgraph"):
            self.btn_detach_pyqtgraph.setParent(self.gps_view)
            self.btn_detach_pyqtgraph.adjustSize()

            self.btn_detach_pyqtgraph.move(
                10,
                self.gps_view.height() - self.btn_detach_pyqtgraph.height() - 50
            )
            self.btn_detach_pyqtgraph.raise_()

            QTimer.singleShot(0, lambda: self.btn_detach_pyqtgraph.move(
                10,
                self.gps_view.contentsRect().height() - self.btn_detach_pyqtgraph.height() - 50
            ))

    def update_altitude_labels(self):
        """Draw a vertical altitude scale next to the 3D GPS viewer."""
        if not hasattr(self, "altitude_scale_labels"):
            return

        # geometry relative to gps_view itself (works even when detached)
        top = 10
        height = self.gps_view.height() - 20

        max_alt = int(gps_max_alt / 500 + 1) * 500

        # position altitude bar near the right edge
        bar_x = self.gps_view.width() - 60
        bar_top = top
        bar_height = height
        self.altitude_bar.setGeometry(bar_x, bar_top, 4, bar_height)

        # ---- altitude colored zones positioning ----
        alt_green_min = 3000.0
        alt_green_max = 5000.0

        t1 = alt_green_min / max_alt
        t2 = alt_green_max / max_alt

        y_green_top = int(top + height * (1.0 - t2))
        y_green_bottom = int(top + height * (1.0 - t1))

        self.altitude_green_zone.setGeometry(
            bar_x,
            y_green_top,
            4,
            y_green_bottom - y_green_top
        )

        # orange zone above 5000 ft
        t3 = 5000.0 / max_alt
        y_orange_top = top
        y_orange_bottom = int(top + height * (1.0 - t3))

        self.altitude_orange_zone.setGeometry(
            bar_x,
            y_orange_top,
            4,
            y_orange_bottom - y_orange_top
        )

        # labels placed to the RIGHT of the bar
        x_left = bar_x + 8

        for idx, (z, label) in enumerate(self.altitude_scale_labels):
            t = z / max_alt
            y = int(top + height * (1.0 - t))

            label.move(x_left, y - label.height() // 2)

            # horizontal tick aligned with altitude bar (to the right)
            if hasattr(self, "altitude_scale_ticks") and idx < len(self.altitude_scale_ticks):
                tick = self.altitude_scale_ticks[idx]
                tick.setGeometry(bar_x + 4, y - 1, 6, 2)

        # ---- altitude cursor position ----
        try:
            alt = float(self.row.gps_alt)
        except Exception:
            alt = 0

        alt = max(0, min(max_alt, alt))
        t = alt / max_alt
        y_cursor = int(top + height * (1.0 - t))

        # center rectangular cursor on altitude bar
        self.altitude_cursor.move(bar_x - 7, y_cursor - 4)


        # blink triangle if climb/descent rate is high, and color by climb/descent
        try:
            fpm_raw = float(self.row.gps_fpm)
        except Exception:
            fpm_raw = 0.0


        self.altitude_cursor.setVisible(True)
        self.altitude_cursor_visible = True

        # ---- vario vertical bar ----
        try:
            fpm = float(self.row.gps_fpm)
        except Exception:
            fpm = 0.0

        # limite (important pour lisibilité)
        fpm_max = 4000.0
        fpm_clamped = max(-fpm_max, min(fpm_max, fpm))

        # longueur max en pixels
        max_len = 80  # ajuste visuellement

        # conversion en pixels
        #length = int((abs(fpm_clamped) / fpm_max) * max_len)
        length = int((abs(fpm_clamped) / fpm_max) ** 0.6 * max_len) # non linéaire pour amplifier faible valeur


        # position X (à côté du curseur altitude)
        x_vario = bar_x+4  # à gauche du curseur

        # position Y (origine = altitude cursor)
        y0 = y_cursor

        if fpm_clamped >= 0:
            # montée → vers le haut
            y = y0 - length
            h = length
        else:
            # descente → vers le bas
            y = y0
            h = length

        self.altitude_vario_bar.setGeometry(x_vario, y, 6, h)

        if fpm_clamped >= 0:
            color = "lime"
        else:
            color = "#ff0000"  # rouge vif

        self.altitude_vario_bar.setStyleSheet(f"background-color: {color};")

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
        # axes hidden by default (toggle from Settings menu)
        self.gfx_axes_x.visible = False
        self.gfx_axes_y.visible = False
        self.gfx_axes_z.visible = False

        light = gfx.DirectionalLight(intensity=3)
        light.local.position = (200, 200, 200)
        self.gfx_scene.add(light)
        ambient = gfx.AmbientLight(intensity=0.4)
        self.gfx_scene.add(ambient)

        # groupe parent (piloté par quaternion)
        self.gfx_object = gfx.Group()

        # vecteurs du repère local
        length = 400  # longueur des vecteurs
        #x_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [length, 0, 0]], dtype=np.float32))
        #elf.gfx_vec_x = gfx.Line(x_geom, gfx.LineMaterial(color="red", thickness=1),)
        y_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [0, length, 0]], dtype=np.float32))
        self.y_geom = y_geom  # Reference for later updates
        self.gfx_vec_y = gfx.Line(
            y_geom,
            gfx.LineMaterial(color="green", thickness=8, depth_test=False),
        )
        #z_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [0, 0, length]], dtype=np.float32))
        #self.gfx_vec_z = gfx.Line(z_geom, gfx.LineMaterial(color="blue", thickness=1),)
        #self.gfx_object.add(self.gfx_vec_x);
        self.gfx_object.add(self.gfx_vec_y);
        # self.gfx_object.add(self.gfx_vec_z)

        # ---- Arrow head for Y vector ----
        self.gfx_y_arrow = gfx.Mesh(
            gfx.cone_geometry(radius=25, height=60),
            gfx.MeshPhongMaterial(color="green", depth_test=False),
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
        self.gfx_vec_acc = gfx.Line(self.acc_geom, gfx.LineMaterial(color="green", thickness=4, depth_test=False),)
        self.gfx_object.add(self.gfx_vec_acc)

        # ---- Arrow head for acceleration vector ----
        self.gfx_acc_arrow = gfx.Mesh(
            gfx.cone_geometry(radius=25, height=60),
            gfx.MeshPhongMaterial(color="green", depth_test=False),
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
        #self.gfx_scene.background = gfx.Background(None, gfx.BackgroundMaterial((1, 1, 1, 1)))
        self.gfx_display.show(self.gfx_scene)
        #self.gfx_display.renderer.clear_color = (1, 1, 1, 1)

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
        #self.gfx_canvas.setStyleSheet("background-color: white;")
        self.grid.addWidget(self.gfx_canvas, 1, 0, 1, 2)
        # ---- Attach detach button to gfx canvas (bottom-center overlay) ----
        if hasattr(self, "btn_detach_gfx"):
            self.btn_detach_gfx.setParent(self.gfx_canvas)
            self.btn_detach_gfx.adjustSize()
            # position bottom-center
            self.btn_detach_gfx.move(
                (self.gfx_canvas.width() - self.btn_detach_gfx.width()) // 2,
                self.gfx_canvas.height() - self.btn_detach_gfx.height() - 10
            )
            self.btn_detach_gfx.raise_()

            QTimer.singleShot(0, lambda: self.btn_detach_gfx.move(
                (self.gfx_canvas.contentsRect().width() - self.btn_detach_gfx.width()) // 2,
                self.gfx_canvas.contentsRect().height() - self.btn_detach_gfx.height() - 10
            ))
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
        # slightly larger artificial horizon
        self.hud_horizon.setGeometry(10, 10, 160, 150)
        self.hud_horizon.show()

        # ---- Wingtip Artificial Horizon (left wing triangle reference) ----
        self.hud_horizon_wing = ArtificialHorizon(self.gfx_canvas)
        # slightly larger wingtip artificial horizon
        self.hud_horizon_wing.setGeometry(10, 170, 160, 150)
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
        # ---- Scale Y vector with GPS speed ----
        speed_scale = row.gps_speed * 2.0  # visual scale factor
        self.y_geom.positions.data[1] = (0.0, speed_scale, 0.0)
        self.y_geom.positions.update_range(0, 2)

        # move arrow head to the end of the vector
        if hasattr(self, "gfx_y_arrow"):
            self.gfx_y_arrow.local.position = (0.0, speed_scale, 0.0)

        # ---- Update acceleration vector ----
        acc = np.array([-row.x4_acc_x,-row.x4_acc_y,-row.x4_acc_z])# rotation
        self.g = np.linalg.norm(acc) / 9.81
        acc = acc / np.linalg.norm(acc) # normer vecteur
        self.acc_vec = R_x_20 @ perm[R_recalage_repere] @ acc
        self.acc_vec = self.acc_vec * 300 +  self.acc_vec * 100 * self.g # scaling up
        # ---- low-pass filter to smooth G vector trail ----
        if self.acc_vec_filtered is None:
            self.acc_vec_filtered = self.acc_vec.copy()
        else:
            a = self.g_filter_alpha
            self.acc_vec_filtered = (1 - a) * self.acc_vec_filtered + a * self.acc_vec
        # display smoothed G vector
        vec_display = self.acc_vec_filtered if self.acc_vec_filtered is not None else self.acc_vec
        self.acc_geom.positions.data[1] = vec_display
        self.acc_geom.positions.update_range(0, 2)

        # ---- Update G vector trail ----
        # If trail is empty (after seek/reset), initialize it with the current filtered vector
        if not np.any(self.g_trail):
            self.g_trail[:] = self.acc_vec_filtered
        else:
            self.g_trail[:-1] = self.g_trail[1:]
            self.g_trail[-1] = self.acc_vec_filtered

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

        # ---- Update arrow head position and orientation (use smoothed vector) ----
        vec_arrow = self.acc_vec_filtered if self.acc_vec_filtered is not None else self.acc_vec
        direction = vec_arrow / (np.linalg.norm(vec_arrow) + 1e-9)
        pos = vec_arrow

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

        try:
            if self.g_label is not None:
                self.g_label.setText(f"G:{self.g:.1f}")
        except RuntimeError:
            return
        try:
            if self.g_label_minmax is not None:
                self.g_label_minmax.setText(f"  Gmin {self.g_min:.1f}\n  Gmax {self.g_max:.1f}")
        except RuntimeError:
            return
        self.g_label.adjustSize()
        # place G value below Alt and Vario overlays
        self.g_label.move(
            self.gfx_canvas.width() - self.g_label.width(),
            140
        )

        self.g_label_minmax.adjustSize()
        self.g_label_minmax.move(
            self.gfx_canvas.width() - self.g_label_minmax.width(),
            190
        )

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
        self.pitch_deg = pitch_deg

        # ---- Compute Inclinaison (roll) ----
        # Vector v orthogonal to fwd and lying in plane (fwd, world Z)
        world_z = np.array([0.0, 0.0, 1.0])
        v = np.cross(fwd, np.cross(world_z, fwd))
        v = v / np.linalg.norm(v)

        # Inclination computation
        dot = np.dot(up, v)
        cross = np.cross(up, v)
        inclinaison = np.arctan2(np.dot(cross, fwd), dot)
        bank_deg = -np.degrees(inclinaison) # convention aéronautique (droite positive / gauche positive)
        self.bank_deg = bank_deg

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
            self.hud_horizon.bank = bank_deg
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
            #f"\nTime: {t_now.strftime('%H:%M:%S.%f')[:-3]}"
            #f"\nElapsed: {em:02d}:{es:02d}"
            #f"\nFrames skipped: {self.frame_skipped_count} / {self.frame_last_delay:+04d}ms"
        )
        self.df_info_label.adjustSize()
        self.df_info_label.move(
            self.gfx_canvas.width() - self.df_info_label.width() - 10,
            self.gfx_canvas.height() - self.df_info_label.height() - 10
        )

        self.pitch_label.setText(f"Pitch {pitch_deg:.0f}°")
        self.pitch_label.adjustSize()
        # position bottom-left
        self.pitch_label.move(
            10,
            self.gfx_canvas.height() - self.pitch_label.height() - 60
        )

        self.roll_label.setText(f"Bank {bank_deg:.0f}°")
        self.roll_label.adjustSize()
        # place just below pitch label (bottom-left)
        self.roll_label.move(
            10,
            self.gfx_canvas.height() - self.roll_label.height() - 20
        )

        # ---- Update GPS speed / altitude overlay ----
        # ---- Update heading overlay on video1 ----
        if hasattr(self, "video1_heading_label"):
            self.video1_heading_label.setText(f"{row.gps_heading:.0f}°")
            # compute angular deviation from aerobatic axis (50° / 230°)
            h = float(row.gps_heading)

            def ang_diff(a, b):
                d = abs(a - b) % 360.0
                return min(d, 360.0 - d)

            d1 = ang_diff(h, 50.0)
            d2 = ang_diff(h, 230.0)
            deviation = min(d1, d2)

            self.video1_heading_deviation_label.setText(f"Δ {deviation:.0f}°")
            self.video1_heading_deviation_label.adjustSize()
            self.video1_heading_label.adjustSize()
            x = int((self.video1.width() - self.video1_heading_label.width()) / 2)
            self.video1_heading_label.move(x, 5)
            self.video1_heading_deviation_label.move(
                (self.video1.width() - self.video1_heading_deviation_label.width()) // 2,
                40  # 👈 sous le premier
            )


        # ---- Update pitch overlay on video1 (above bank) ----
        if hasattr(self, "video1_pitch_label"):
            self.video1_pitch_label.setText(f"Pitch {pitch_deg:.0f}°")
            self.video1_pitch_label.adjustSize()
            x_pitch = int((self.video1.width() - self.video1_pitch_label.width()) / 2)
            y_pitch = self.video1.height() - self.video1_pitch_label.height() - 45
            self.video1_pitch_label.move(x_pitch, y_pitch)

        # ---- Update bank overlay below pitch ----
        if hasattr(self, "video1_bank_label"):
            self.video1_bank_label.setText(f"Bank {bank_deg:.0f}°")
            self.video1_bank_label.adjustSize()
            x_bank = int((self.video1.width() - self.video1_bank_label.width()) / 2)
            y_bank = y_pitch + self.video1_pitch_label.height() + 5
            self.video1_bank_label.move(x_bank, y_bank)

        # ---- Update badin (GPS speed) overlay on video1 (bottom-left) ----
        if hasattr(self, "video1_speed_label"):
            self.video1_speed_label.setText(f"{row.gps_speed:.0f} km/h")
            self.video1_speed_label.adjustSize()
            x_speed = 10
            y_speed = self.video1.height() - self.video1_speed_label.height() - 10
            self.video1_speed_label.move(x_speed, y_speed)

        # ---- Update analog badin with smoothing ----
        if self.smooth_speed is None:
            self.smooth_speed = row.gps_speed
        else:
            a = self.instrument_alpha
            self.smooth_speed = (1 - a) * self.smooth_speed + a * row.gps_speed

        if hasattr(self, "video1_badin"):
            self.video1_badin.speed = self.smooth_speed
            self.video1_badin.update()
            yb = int(self.video1.height()/2 - self.video1_badin.height()/2) + 80
            self.video1_badin.move(10, yb)

        # ---- Update analog altimeter with smoothing ----
        if self.smooth_alt is None:
            self.smooth_alt = row.gps_alt
        else:
            a = self.instrument_alpha
            self.smooth_alt = (1 - a) * self.smooth_alt + a * row.gps_alt

        if hasattr(self, "video1_altimeter"):
            self.video1_altimeter.alt = self.smooth_alt
            self.video1_altimeter.update()
            ya = int(self.video1.height()/2 - self.video1_altimeter.height()/2) + 80
            xa = self.video1.width() - self.video1_altimeter.width() - 10
            self.video1_altimeter.move(xa, ya)

        # ---- Update vario overlay on video1 (bottom-right) ----
        if hasattr(self, "video1_fpm_label"):
            self.video1_fpm_label.setText(f"{row.gps_fpm:.0f} ft/min")
            self.video1_fpm_label.adjustSize()
            x_fpm = self.video1.width() - self.video1_fpm_label.width() - 10
            y_fpm = self.video1.height() - self.video1_fpm_label.height() - 10
            self.video1_fpm_label.move(x_fpm, y_fpm)

        # ---- Update altitude overlay just above vario ----
        if hasattr(self, "video1_alt_label"):
            self.video1_alt_label.setText(f"Alt {row.gps_alt:.0f} ft")
            self.video1_alt_label.adjustSize()
            x_alt = self.video1.width() - self.video1_alt_label.width() - 10
            y_alt = y_fpm - self.video1_alt_label.height() - 5
            self.video1_alt_label.move(x_alt, y_alt)


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

        # apply same color to velocity vector (gfx_vec_y)
        if hasattr(self, "gfx_vec_y"):
            self.gfx_vec_y.material.color = (r/255.0, g/255.0, b/255.0, 1.0)
        if hasattr(self, "gfx_y_arrow"):
            self.gfx_y_arrow.material.color = (r/255.0, g/255.0, b/255.0, 1.0)

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


    def calibrate_gfx(self, where):
        # average accelerometer over 100 samples to reduce IMU noise
        start = max(0, where - 50)
        end = min(len(df), where + 50)

        acc = df.iloc[start:end][["x4_acc_x", "x4_acc_y", "x4_acc_z"]].to_numpy()
        grav = np.mean(acc, axis=0)
        grav = grav / np.linalg.norm(grav)
        self.montage_pitch_angle = math.degrees(math.acos(grav[1])) - OFFSET_PITCH_SOL_PALLIER
        print("Angle de montage : ", round(self.montage_pitch_angle,1))

        # update menu display of camera pitch
        try:
            self.update_pitch_cam_menu()
        except Exception:
            pass

        # refresh graphics immediately (useful when paused)
        try:
            self.update_gfx_orientation()
        except Exception:
            pass

    def calibrate_gfx_on_current_frame(self):
        self.calibrate_gfx(self.idf)


    def on_map_loaded(self, ok):
        if ok:
            self.map_ready = True

    def update_metar(self):
        if self.i % 30 != 0:
            return
        # ---- Update METAR overlay if needed ----
        try:
            current_time = self.row.timestamp
            metar_row = find_metar_for_time(metar_df, current_time)
            new_metar = metar_row.metar

            if new_metar != self.last_metar:
                self.last_metar = new_metar

                if hasattr(self, "map_metar_label"):
                    self.map_metar_label.setText(self.last_metar)
                    self.map_metar_label.adjustSize()
                    self._position_map_metar_label()

        except Exception:
            pass

    # ==================================================
    # Real-time compensated main loop (absolute scheduling)
    # ==================================================
    def main_loop(self):
        now = self.clock.elapsed()

        # initialize startup time once
        if self.startup_time_ms is None:
            self.startup_time_ms = now

        # initialize absolute schedule
        if self.next_frame_time == 0:
            self.next_frame_time = now

        if self.playing:
            # 🔊 audio ALWAYS runs
            self.update_audio()

            # si audio pas encore prêt → fallback
            if not hasattr(self, "audio_clock_sec") or self.audio_clock_sec <= 0:
                self.update_all()
            else:
                # temps cible dicté par l’audio
                target_time = self.audio_clock_sec

                # temps actuel vidéo
                if self.current_video_time_utc is not None:
                    video_time = (self.current_video_time_utc - self.video1_start).total_seconds()
                else:
                    video_time = 0.0

                # New sync logic with margin
                margin = 0.05  # 50 ms tolerance

                if video_time < target_time - margin:
                    # video late → catch up
                    self.update_all()

                elif video_time > target_time + margin:
                    # video ahead → WAIT only if sync is stable
                    # after seek, allow video to move to avoid freeze
                    if getattr(self, "sync_enabled", False):
                        pass
                    else:
                        self.update_all()

                else:
                    # in sync → normal playback
                    self.update_all()

        # audio-driven → on tick très vite
        self.next_frame_time = now + 5

        #delay = int(self.next_frame_time - now)
        #self.frame_last_delay = delay
        #if delay < 0:
        #    delay = 0
        #    self.frame_skipped_count += 1

        #self.timer.start(delay)

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
        self.update_bookmark_ticks()

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
        self.update_bookmark_ticks()


    def reload_bookmarks(self):
        """Force reload of bookmark CSV file and refresh menu."""
        self.load_bookmarks()
        self.update_bookmark_ticks()

    # ==================================================
    # Centralized video seek (avoid repeating CAP_PROP_POS_FRAMES everywhere)
    # ==================================================
    def seek_video(self, frame):

        self.i = int(frame)

        # reset frame counter to avoid sync gating after seek
        if self.i < 5:
            self.i = int(frame)

        # TO FIX A DEPLACER AU BON ENDROIT
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
                self.audio_buffer = bytearray()
                self.audio_clock_sec = 0.0
                # reset sync warmup after seek
                self.sync_enabled = False
                self.sync_reenable_time = self.clock.elapsed() + 300  # 300 ms
                self.startup_time_ms = self.clock.elapsed()
                self.audio_started = True
                self.audio_prebuffer_done = False

            except Exception:
                pass
        #self.next_frame_time = self.clock.elapsed()

    def goto_bookmark(self, frame):
        self.seek_video(frame)
        self.slider.setValue(self.i)

        # show bookmark overlay immediately when jumping to it
        if self.bookmarks_df is not None:
            row = self.bookmarks_df[self.bookmarks_df["frame"] == frame]
            if not row.empty:
                name = str(row.iloc[0]["name"])
                self.show_bookmark_overlay(name)

    def goto_previous_bookmark(self):
        """Jump to the previous bookmark before the current frame.
        If called again within 2 seconds, jump one bookmark further back."""

        if self.bookmarks_df is None or self.bookmarks_df.empty:
            return

        import time
        now = time.time()

        # bookmarks before current frame
        past = self.bookmarks_df[self.bookmarks_df["frame"] < self.i]

        if past.empty:
            print("No previous bookmark")
            return

        # detect rapid consecutive press
        rapid = False
        if hasattr(self, "_last_prev_time"):
            if now - self._last_prev_time < 2.0:
                rapid = True

        # choose which bookmark to jump to
        if rapid and len(past) >= 2:
            frame = int(past.iloc[-2]["frame"])
        else:
            frame = int(past.iloc[-1]["frame"])

        self._last_prev_time = now
        self.goto_bookmark(frame)

    def goto_next_bookmark(self):
        """Jump to the next bookmark after the current frame."""

        if self.bookmarks_df is None or self.bookmarks_df.empty:
            return

        # chercher les bookmarks après la frame actuelle
        future = self.bookmarks_df[self.bookmarks_df["frame"] > self.i]

        if future.empty:
            print("No next bookmark")
            return

        frame = int(future.iloc[0]["frame"])
        self.goto_bookmark(frame)

    def show_bookmark_overlay(self, name):

        if self.bookmark_overlay is None:
            return

        self.bookmark_overlay.setText(name)
        self.bookmark_overlay.adjustSize()

        # center overlay roughly in the middle of the video area
        x = int((self.width() - self.bookmark_overlay.width()) / 2)
        y = int((self.height() - self.bookmark_overlay.height()) / 6)
        self.bookmark_overlay.move(x, y)

        self.bookmark_overlay.show()

        # restart overlay timer so rapid Next/Previous presses don't shorten display time
        if not hasattr(self, "bookmark_overlay_timer"):
            self.bookmark_overlay_timer = QTimer(self)
            self.bookmark_overlay_timer.setSingleShot(True)
            self.bookmark_overlay_timer.timeout.connect(self.bookmark_overlay.hide)

        self.bookmark_overlay_timer.start(4000)

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

    def seek_palier(self):
        """
        Cherche un pallier :
        |gps_fpm| < 100 ft/min
        gps_speed > 150 km/h
        pendant 2 secondes
        """

        window = int(DF_FREQ * 2)  # 2 secondes

        fpm = df["gps_fpm"].to_numpy()
        speed = df["gps_speed"].to_numpy()

        # start searching AFTER the current dataframe position (next palier behavior)
        start_idx = getattr(self, "idf", 0) + 1000
        for i in range(start_idx, len(df) - window):

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

            if frame.format.name != "rgb24":
                frame = frame.to_rgb()

            #if frame.format.name != "bgr24":
            #    frame = frame.reformat(format="bgr24")

            # Return the frame directly (no numpy conversion)
            return True, frame, frame

        except StopIteration:
            return False, None, None

    # ==================================================
    # AUDIO
    # ==================================================
    def update_audio(self):
        if self.audio_stream is None:
            return

        try:
            # 🔑 combien de place dispo dans le buffer audio OS
            if hasattr(self, "audio_output"):
                bytes_free = max(self.audio_output.bytesFree(), 16384)
            else:
                bytes_free = 16384

            # 🔑 on remplit un peu plus que nécessaire
            target_buffer = max(bytes_free * 4, 65536)

            # 🔑 décodage adaptatif
            while len(self.audio_buffer) < target_buffer:
                packet = next(self.audio_packets)

                for frame in packet.decode():
                    frames = self.audio_resampler.resample(frame)

                    if not isinstance(frames, (list, tuple)):
                        frames = [frames]

                    for f in frames:
                        if f is None:
                            continue

                        if f.pts is not None:
                            self.audio_clock_sec = float(f.pts * f.time_base)

                        self.audio_buffer += f.to_ndarray().tobytes()

        except StopIteration:
            return
        except Exception as e:
            print("audio error:", e)
            return

        # 🔊 écriture CONTINUE (critique)
        chunk_size = 8192

        if not hasattr(self, "audio_output"):
            return

        # ---- PREBUFFER (avoid stutter after start/seek) ----
        if not hasattr(self, "audio_prebuffer_done"):
            self.audio_prebuffer_done = False

        if not self.audio_prebuffer_done:
            if len(self.audio_buffer) < 65536:
                return
            else:
                self.audio_prebuffer_done = True

        # 🔑 on vide autant que possible (et pas 1 seul chunk)
        while len(self.audio_buffer) >= chunk_size:
            written = self.audio_device.write(self.audio_buffer[:chunk_size])

            if written <= 0:
                break

            self.audio_buffer = self.audio_buffer[written:]
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
        self.update_video(self.decoder1, self.video1, self.video1_start, self.stream1)
        self.update_video(self.decoder2, self.video2, self.video2_start, self.stream2)
        self.i += 1
        if self.current_video_time_utc is not None:
            self.sync_dataframe_on_video()
        self.update_gps_pyqtgraph()
        self.update_metar()
        self.update_gfx_orientation()

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

        # ---- Elapsed time overlay update ----
        try:
            if self.current_video_time_utc is not None:
                elapsed = self.current_video_time_utc - df.timestamp.iloc[0]
                total_sec = int(elapsed.total_seconds())

                h = total_sec // 3600
                m = (total_sec % 3600) // 60
                s = total_sec % 60

                if h > 0:
                    txt = f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    txt = f"{m:02d}:{s:02d}"

                self.elapsed_time_overlay.setText(txt)
        except Exception:
            pass

    # ==================================================
    def update_video(self, decoder, label, start_dt, stream):
        ret, frame, avframe = self.read_video_frame(decoder)

        if not ret:
            return

        ms = avframe.pts * float(stream.time_base) * 1000
        video_time_utc = start_dt + timedelta(milliseconds=ms)
        self.current_video_time_utc = video_time_utc
        warmup_elapsed = 0 if self.startup_time_ms is None else (self.clock.elapsed() - self.startup_time_ms)

        if (
                hasattr(self, "audio_clock_sec")
                and self.audio_clock_sec > 0
                and len(self.audio_buffer) > 12000
                and warmup_elapsed > 300
        ):
            self.sync_enabled = True

        if self.sync_enabled:
            video_time_sec = (video_time_utc - start_dt).total_seconds()
            sync_error = video_time_sec - self.audio_clock_sec

            sync_error = max(min(sync_error, 0.5), -0.5)

            # vidéo en avance → ne pas afficher
            if sync_error > 0.02:
                # do not block display → avoid freeze
                pass

            # vidéo en retard → skip
            if sync_error < -0.05:
                try:
                    next(self.decoder1, None)
                    next(self.decoder2, None)
                    self.i += 1
                except:
                    pass


        plane = frame.planes[0]
        h = frame.height
        w = frame.width
        img = QImage(plane, w, h, plane.line_size, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))

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

        self.timestamp_label.setText(f"Video time : {ts.strftime('%Y-%m-%d %H:%M:%S')}")

        # ---- Elapsed time overlay update (compute txt here) ----
        try:
            if self.current_video_time_utc is not None:
                elapsed = self.current_video_time_utc - df.timestamp.iloc[0]
                total_sec = int(elapsed.total_seconds())

                h = total_sec // 3600
                m = (total_sec % 3600) // 60
                s = total_sec % 60

                if h > 0:
                    txt = f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    txt = f"{m:02d}:{s:02d}"

                self.elapsed_time_overlay.setText(txt)
                self.elapsed_time_overlay.adjustSize()
                self._position_elapsed_time_overlay()
                QTimer.singleShot(0, self._position_elapsed_time_overlay)
        except Exception:
            pass

        # ---- Previous bookmark overlay update ----
        try:
            if self.current_video_time_utc is not None and self.bookmarks_df is not None:
                t = self.current_video_time_utc
                df_b = self.bookmarks_df

                # bookmarks passés uniquement
                past = df_b[df_b["frame"] <= self.i]
                if len(past) > 0:
                    last = past.iloc[-1]
                    name = str(last.get("name", last.get("label", "")))

                    self.prev_bookmark_overlay.setText(name)
                    self.prev_bookmark_overlay.show()
                    self.prev_bookmark_overlay.adjustSize()
                else:
                    self.prev_bookmark_overlay.hide()

        except Exception:
            pass


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

        try:
            if self.map_ready:
                self.map_view.page().runJavaScript("resetTrajectory();")
        except Exception:
            pass

    # ==================================================
    def toggle_matplotlib_gps(self):
        """Toggle matplotlib GPS update on/off."""
        self.enable_matplotlib_gps = not self.enable_matplotlib_gps

        # keep menu checkbox synchronized
        if hasattr(self, "act_pause_gps_update"):
            self.act_pause_gps_update.setChecked(not self.enable_matplotlib_gps)

    def update_gps_pyqtgraph(self):
        # skip updates only during playback
        if self.playing and self.i % 8 != 0:
            return

        #t0 = time.perf_counter()

        row = self.row
        if self.map_ready:
            lat = row.gps_lat
            lon = row.gps_lon
            heading = row.gps_heading-45
            self.map_view.page().runJavaScript(
                f"window.updateMarker({lat}, {lon}, {heading});"
            )

        # keep METAR overlay visible and correctly positioned
        if hasattr(self, "map_metar_label"):
            try:
                self.map_metar_label.setText(getattr(self, "last_metar", ""))
                self._position_map_metar_label()
            except Exception:
                pass

        end = self.idf
        start = end - TRACE
        if start < 0:
            start = 0

        # center trajectory on current aircraft position
        lon = gps_lon_vals[start:end]
        lat = gps_lat_vals[start:end]
        alt = gps_alt_vals[start:end]

        lon0 = gps_lon_vals[end]
        lat0 = gps_lat_vals[end]
        alt0 = gps_alt_vals[end]

        # convert degrees to approximate meters
        x = (lon - lon0) * 111320 * np.cos(np.radians(lat0)) / 1000
        y = (lat - lat0) * 111320 / 1000

        if len(alt) > 0 and alt[-1] > 3000:#décalage de 3000 si dans le box
            z = (alt-3000-1000) / 1000
        else:
            z = (alt - 1000) / 1000
        # debug: dernière position calculée
        if len(x) > 0:
            # ---- vertical projection to ground ----
            try:
                p_air = np.array([x[-1], y[-1], z[-1]])
                p_ground = np.array([x[-1], y[-1], -1.0])
                self.gps_vertical_line.setData(pos=np.vstack([p_air, p_ground]))
            except Exception:
                pass

        pts = np.column_stack([x, y, z])

        # ---- update ground projection (shadow) ----
        if len(pts) > 1:
            pts_ground = pts.copy()
            pts_ground[:, 2] = -1.0
            try:
                self.gps_shadow.setData(pos=pts_ground)
            except Exception:
                pass

        if len(pts) > 1:
            # ---- color segments based on altitude (3000 → 5000 ft) ----
            z_abs = alt  # original altitude values in ft

            zmin = 3000.0
            zmax = 5000.0

            # normalize altitude in 0..1 within [3000,5000]
            zn = (z_abs - zmin) / (zmax - zmin)
            zn = np.clip(zn, 0.0, 1.0)

            colors = np.zeros((len(pts), 4))

            # blue -> green -> red gradient
            colors[:, 0] = zn                       # red increases with altitude
            colors[:, 1] = 1.0 - np.abs(zn - 0.5)*2 # green strongest mid-altitude
            colors[:, 2] = 1.0 - zn                 # blue decreases with altitude
            colors[:, 3] = 1.0                      # alpha

            # update the bundle of lines to simulate a tube
            for line in self.gps_lines:
                off = line._tube_offset
                pts_off = pts + off
                line.setData(pos=pts_off, color=colors)


        if end < len(gps_lat_vals):
            # aircraft stays at center

            # ---- rotation using Euler angles (heading, pitch, roll) ----
            heading = float(self.row.gps_heading)
            pitch = float(getattr(self, "pitch_deg", 0.0))
            roll = float(getattr(self, "bank_deg", 0.0))


            self.gps_aircraft.resetTransform()

            # rotations (ordre important)

            # rotations
            self.gps_aircraft.rotate(-roll, 1, 0, 0)  # FIX
            self.gps_aircraft.rotate(pitch, 0, 1, 0)
            self.gps_aircraft.rotate(- heading-90, 0, 0, 1)
            self.gps_aircraft.translate(0, 0, z[-1])

        # ---- update altitude labels for pyqtgraph GPS view ----
        try:
            self.update_altitude_labels()
        except Exception:
            pass

        # ---- update horizontal GPS speed bar ----
        if hasattr(self, "speed_bar") and hasattr(self, "speed_cursor"):
            bar_margin = 20
            # stop the speed bar before the altitude scale area on the right
            bar_width = self.gps_view.width() - bar_margin * 2 - 80
            bar_y = 16  # move speed bar slightly lower from top

            self.speed_bar.setGeometry(bar_margin, bar_y, bar_width, 6)

            # position speed graduations
            if hasattr(self, "speed_scale_labels"):
                max_speed = 400.0
                for idx, (v, label) in enumerate(self.speed_scale_labels):
                    t = v / max_speed
                    x = int(bar_margin + t * bar_width)

                    # place graduations above the speed bar
                    label.move(x - label.width() // 2, bar_y - label.height() - 4)

                    # vertical tick just above the bar
                    if hasattr(self, "speed_scale_ticks") and idx < len(self.speed_scale_ticks):
                        tick = self.speed_scale_ticks[idx]
                        tick.setGeometry(x - 1, bar_y - 2, 2, 6)

            # speed scaling (same logical ranges as badin)
            try:
                speed = float(self.row.gps_speed)
            except Exception:
                speed = 0.0
            max_speed = 400.0
            s = max(0.0, min(speed, max_speed))
            t = s / max_speed

            x_cursor = int(bar_margin + t * bar_width)

            self.speed_cursor.setGeometry(x_cursor - 5, bar_y - 5, 10, 16)

            # same color logic as speed vector
            if speed < 113:
                color = "rgb(0,0,255)"
            elif speed < 236:
                color = "rgb(0,255,0)"
            elif speed < 350:
                # 236-349: green to yellow, then to orange
                # 236-300: green to yellow, 300-350: yellow to orange
                if speed < 300:
                    r = int((speed - 235) / (300 - 235) * 255)
                    g = 255 - r
                else:
                    r = 255
                    g = max(0, 255 - int((speed - 300) / (350 - 300) * 255))
                color = f"rgb({r},{g},0)"
            else:
                color = "rgb(255,0,0)"

            self.speed_cursor.setStyleSheet(f"background-color: {color};")

        # ---- update G force bar (bottom of GPS view) ----
        if hasattr(self, "g_bar") and hasattr(self, "g_cursor"):
            bar_margin = 20
            # stop the G bar before the altitude scale area on the right
            bar_width = self.gps_view.width() - bar_margin * 2 - 80
            bar_y = self.gps_view.height() - 18

            self.g_bar.setGeometry(bar_margin, bar_y, bar_width, 6)

            # position G scale labels and ticks
            if hasattr(self, "g_scale_labels"):
                g_min = -2.0
                g_max = 4.0
                for idx, (gv, label) in enumerate(self.g_scale_labels):
                    t = (gv - g_min) / (g_max - g_min)
                    x = int(bar_margin + t * bar_width)

                    # labels above the bar
                    label.move(x - label.width() // 2, bar_y - label.height() - 4)

                    # small vertical tick
                    if hasattr(self, "g_scale_ticks") and idx < len(self.g_scale_ticks):
                        tick = self.g_scale_ticks[idx]
                        tick.setGeometry(x - 1, bar_y - 2, 2, 6)

            # get current G value
            try:
                g_val = float(self.g)
            except Exception:
                g_val = 0.0

            # clamp between -4 and +6
            g_min = -2.0
            g_max = 4.0
            g_val = max(g_min, min(g_max, g_val))

            # normalize to 0..1
            t = (g_val - g_min) / (g_max - g_min)

            x_cursor = int(bar_margin + t * bar_width)

            self.g_cursor.setGeometry(x_cursor - 5, bar_y - 5, 10, 16)

            # color depending on G magnitude
            if g_val < 0:
                color = "rgb(0,120,255)"
            elif g_val < 3:
                color = "rgb(0,255,0)"
            elif g_val < 5:
                color = "rgb(255,200,0)"
            else:
                color = "rgb(255,0,0)"

            self.g_cursor.setStyleSheet(f"background-color: {color};")


        az = -int(row.gps_heading / 45) * 45 - 22.5
        if az!= self.last_azim:
            self.last_azim = az
            self.gps_view.setCameraPosition(azimuth=az)
            yz=-1
            if az==-67.5 or az==-22.5 or az==-112.5 or az==-157.5:
                yz=1

            self.grid_vertical_yz.resetTransform()
            self.grid_vertical_yz.rotate(90, 1, 0, 0)
            self.grid_vertical_yz.translate(0, yz, 0)

            xz = -1
            if az==-202.5 or az == -247.5 or az == -112.5 or az==-157.5:
                xz = 1

            self.grid_vertical_xz.resetTransform()
            self.grid_vertical_xz.rotate(90, 0, 1, 0)
            self.grid_vertical_xz.translate(xz,0, 0)

    def detach_gfx_window(self):
        """Toggle detach/close for pygfx canvas."""
        if getattr(self, "gfx_detached", False):
            # close popup
            if hasattr(self, "gfx_window"):
                self.gfx_window.close()
            return

        self.gfx_detached = True

        # remove from layout
        self.grid.removeWidget(self.gfx_canvas)

        # create new window
        self.gfx_window = QMainWindow(self)
        self.gfx_window.setWindowTitle("3D View")
        self.gfx_window.setCentralWidget(self.gfx_canvas)
        # hide overlay button while detached window is open
        if hasattr(self, "btn_detach_gfx"):
            self.btn_detach_gfx.hide()
        self.gfx_window.resize(900, 700)

        #self.btn_detach_gfx.setText("Close 3D")

        # detect close event
        self.gfx_window.closeEvent = self._on_gfx_window_closed

        self.gfx_window.show()


    def _on_gfx_window_closed(self, event):
        """Restore pygfx canvas back into main layout when detached window closes."""
        try:
            self.gfx_canvas.setParent(None)
            self.grid.addWidget(self.gfx_canvas, 1, 0, 1, 2)
            self.gfx_detached = False
            #self.btn_detach_gfx.setText("↗Detach")
            # restore overlay button when returning to main window
            if hasattr(self, "btn_detach_gfx"):
                self.btn_detach_gfx.show()
                self.btn_detach_gfx.raise_()
        except Exception:
            pass

        event.accept()


    def detach_video1_window(self):
        """Toggle detach/close for video1."""
        if getattr(self, "video1_detached", False):
            if hasattr(self, "video1_window"):
                self.video1_window.close()
            return

        self.video1_detached = True

        self.grid.removeWidget(self.video1)

        self.video1_window = QMainWindow(self)
        self.video1_window.setWindowTitle("Video 1")
        self.video1_window.setCentralWidget(self.video1)
        self.video1_window.resize(900, 600)

        self.btn_detach_video1.hide()

        self.video1_window.closeEvent = self._on_video1_window_closed

        self.video1_window.show()


    def _on_video1_window_closed(self, event):
        try:
            self.video1.setParent(None)
            self.grid.addWidget(self.video1, 0, 0, 1, 2)
            self.video1_detached = False
            self.btn_detach_video1.show()
            self.btn_detach_video1.raise_()

        except Exception:
            pass

        event.accept()

    def detach_video2_window(self):
        """Toggle detach/close for video2."""
        if getattr(self, "video2_detached", False):
            if hasattr(self, "video2_window"):
                self.video2_window.close()
            return

        self.video2_detached = True

        self.grid.removeWidget(self.video2)

        self.video2_window = QMainWindow(self)
        self.video2_window.setWindowTitle("Video 2")
        self.video2_window.setCentralWidget(self.video2)
        self.video2_window.resize(900, 600)

        self.btn_detach_video2.hide()

        self.video2_window.closeEvent = self._on_video2_window_closed

        self.video2_window.show()


    def _on_video2_window_closed(self, event):
        try:
            self.video2.setParent(None)
            self.grid.addWidget(self.video2, 0, 2, 1, 2)
            self.video2_detached = False
            self.btn_detach_video2.show()
            self.btn_detach_video2.raise_()
        except Exception:
            pass

        event.accept()


    def detach_pyqtgraph_window(self):
        """Toggle detach/close for matplotlib GPS canvas."""
        if getattr(self, "pyqtgraph_detached", False):
            if hasattr(self, "pyqtgraph_window"):
                self.pyqtgraph_window.close()
            return

        self.pyqtgraph_detached = True

        # remove from layout
        self.grid.removeWidget(self.gps_view)

        # create detached window
        self.pyqtgraph_window = QMainWindow(self)
        self.pyqtgraph_window.setWindowTitle("GPS Graph")
        self.pyqtgraph_window.setCentralWidget(self.gps_view)
        self.pyqtgraph_window.resize(900, 700)

        self.btn_detach_pyqtgraph.hide()

        # detect close
        self.pyqtgraph_window.closeEvent = self._on_pyqtgraph_window_closed

        self.pyqtgraph_window.show()


    def _on_pyqtgraph_window_closed(self, event):
        """Restore matplotlib canvas back into the grid when the detached window closes."""
        try:
            self.gps_view.setParent(None)
            self.grid.addWidget(self.gps_view, 1, 2, 1, 1)
            self.pyqtgraph_detached = False
            self.btn_detach_pyqtgraph.show()
            self.btn_detach_pyqtgraph.raise_()
        except Exception:
            pass

        event.accept()


# ======================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    palette = app.palette()
    palette.setColor(palette.Window, Qt.white)
    palette.setColor(palette.Base, Qt.white)
    palette.setColor(palette.AlternateBase, Qt.white)
    palette.setColor(palette.Text, Qt.black)
    palette.setColor(palette.WindowText, Qt.black)

    app.setPalette(palette)
    app.setStyleSheet("""
    QPushButton {
        background-color: #f0f0f0;
        color: black;
        border: 1px solid #888;
        border-radius: 4px;
        padding: 4px 8px;
    }

    QPushButton:hover {
        background-color: #e0e0e0;
    }

    QPushButton:pressed {
        background-color: #d0d0d0;
    }

    QMenuBar {
        background-color: #f5f5f5;
        color: black;
    }

    QMenu {
        background-color: white;
        color: black;
    }

    QMenu::item:selected {
        background-color: #d0d0d0;
    }
    
    QSlider::groove:horizontal {
    height: 6px;
    background: #ddd;
    border-radius: 3px;
    }
    
    QSlider::handle:horizontal {
        background: #666;
        border: 1px solid #444;
        width: 14px;
        height: 14px;
        margin: -5px 0;  /* centre la poignée */
        border-radius: 7px;
    }
    
    QSlider::handle:horizontal:hover {
        background: #444;
    }
    
    QSlider::sub-page:horizontal {
        background: #4a90e2;  /* partie parcourue */
        border-radius: 3px;
    }
    
    QSlider::add-page:horizontal {
        background: #ddd;
        border-radius: 3px;
    }

    """)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

