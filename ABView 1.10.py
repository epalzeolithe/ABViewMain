import sys
import traceback


def excepthook(type_, value, tb):
    print("\n=== FULL TRACEBACK ===")
    traceback.print_exception(type_, value, tb)


sys.excepthook = excepthook

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QOpenGLShaderProgram, QOpenGLShader
import OpenGL.GL as gl
import pyqtgraph.opengl as glpg
from stl import mesh
import math, time, sys, os
import psutil,json
from datetime import datetime, timedelta, timezone

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import av
import sip
import numpy as np
import pandas as pd
import pygfx as gfx
from PyQt5.QtCore import QTimer, Qt, QElapsedTimer
from PyQt5.QtGui import QImage, QKeySequence
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput
from PyQt5.QtWidgets import (QShortcut, QApplication, QMainWindow, QWidget, QLabel, QFrame, QHBoxLayout, QGridLayout,
                             QAction, QSlider, QSizePolicy, QInputDialog)
from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QFileDialog
from pymediainfo import MediaInfo
import CoreMedia
import AVFoundation
import ScreenCaptureKit
from Cocoa import NSObject
import objc
import warnings  # silence noisy PyObjC warnings produced when accessing CVPixelBuffer pointers
from objc import ObjCPointerWarning
from PyQt5.QtWidgets import QVBoxLayout, QPushButton
from PyQt5.QtGui import QPolygon
from PyQt5.QtCore import QPoint

from ver import __version__

from PyQt5 import QtWidgets

# ---- GLOBAL SAFE UPDATE PATCH ----
_orig_update = QtWidgets.QWidget.update


def safe_update(self, *args, **kwargs):
    try:
        if sip.isdeleted(self):
            return
    except Exception:
        return
    return _orig_update(self, *args, **kwargs)


QtWidgets.QWidget.update = safe_update

# ***********************************************
# CONFIG
SKIP_BDL_SELECTION = False
MAINDIR = "/Users/drax/Down/ABViewMain/"
# BDL="data/Vol_2026_02_21.abv/"
# BDL="data/Vol_2026_03_20.abv/"
# BDL="data/Vol_2026_02_15.abv/"
BDL = "data/Vol_2026_03_21.abv/"
PDL = MAINDIR + BDL
STL_FILE = MAINDIR + "ressources/CAP10.STL"
STL_SIMPLE_PLANE_FILE = MAINDIR + "ressources/plane.STL"
BOX = 0.007 * 1.5  # taille box vision en °latitude
DF_FREQ = 100
TRACE = 6000  # taille de la trace 6000=1 minute
TRACE_DEFAULT = TRACE
TRACE_BEFORE = 500  # position précédente, 500 avant soit 5s
TRACE_SLICING_FACTOR = 50
BOX_HEADING = 50
VITESSE_MISE_EN_LIGNE = 80  # km/h
PITCH_MONTAGE_PAR_DEFAUT = 15  # camera verticale au repos par défaut, écran face à soi, légèrement inclinée vers soi
OFFSET_PITCH_SOL_PALLIER = 2  # différence de pitch entre sol et pallier vers 200kmh
R_recalage_repere = 3  # données issues BB
# R_recalage_repere=1 # données issues computed VQF
refcam = [0, 0, 1]  # données issues de BB
# refcam=[0,0,-1] # données issues computed VQF


ROT_BOX = np.array([
    [np.cos(np.radians(BOX_HEADING)), -np.sin(np.radians(BOX_HEADING)), 0.0],
    [np.sin(np.radians(BOX_HEADING)), np.cos(np.radians(BOX_HEADING)), 0.0],
    [0.0, 0.0, 1.0]])


class VideoYUVOpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = None
        self.tex_y = None
        self.tex_u = None
        self.tex_v = None

    def setFrame(self, frame):
        self.frame = frame
        self.update()

    def initializeGL(self):
        self.vertices = [-1, -1, 1, -1, -1, 1, 1, 1]
        self.program = QOpenGLShaderProgram()
        # --- REQUIRED for macOS OpenGL core ---
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        vertices = np.array(self.vertices, dtype=np.float32)

        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            vertices.nbytes,
            vertices,
            gl.GL_STATIC_DRAW
        )

        # --- PBOs for async texture upload ---
        self.pbo_y = gl.glGenBuffers(1)
        self.pbo_u = gl.glGenBuffers(1)
        self.pbo_v = gl.glGenBuffers(1)

        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, """
            #version 330 core
            layout(location = 0) in vec2 position;
            out vec2 texCoord;
            void main() {
                texCoord = vec2((position.x + 1.0) / 2.0, 1.0 - (position.y + 1.0) / 2.0);
                gl_Position = vec4(position, 0.0, 1.0);
            }
        """)

        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, """
            #version 330 core
            in vec2 texCoord;
            out vec4 fragColor;

            uniform sampler2D texY;
            uniform sampler2D texU;
            uniform sampler2D texV;

            void main() {
                float y = texture(texY, texCoord).r;
                float u = texture(texU, texCoord).r - 0.5;
                float v = texture(texV, texCoord).r - 0.5;

                float r = y + 1.402 * v;
                float g = y - 0.344 * u - 0.714 * v;
                float b = y + 1.772 * u;

                fragColor = vec4(r, g, b, 1.0);
            }
        """)

        self.program.link()

        # --- Configure vertex attributes ONCE (avoid per-frame overhead) ---
        pos_attr = 0  # matches layout(location = 0)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(pos_attr)
        gl.glVertexAttribPointer(
            pos_attr,
            2,
            gl.GL_FLOAT,
            False,
            0,
            None)

    def paintGL(self):
        if self.frame is None:
            return
        frame = self.frame
        if "yuv" not in frame.format.name.lower():
            return
        y = frame.planes[0]
        u = frame.planes[1]
        v = frame.planes[2]
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.program.bind()

        def upload(tex_id, plane, w, h, unit, pbo):
            # Activate texture unit BEFORE any glBindTexture
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            # Safety: ensure clean state before texture ops
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
            if tex_id is None:
                tex_id = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
                # Set texture parameters ONCE after creation
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
                # IMPORTANT: ensure no PBO is bound when allocating texture
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D,
                    0,
                    gl.GL_RED,
                    w,
                    h,
                    0,
                    gl.GL_RED,
                    gl.GL_UNSIGNED_BYTE,
                    None
                )
            else:
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)

            # Convert plane
            try:
                data = plane.to_bytes()
            except AttributeError:
                data = np.frombuffer(plane, dtype=np.uint8)

            size = len(data)

            # --- PBO upload (safe path, no mapBuffer) ---
            try:
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)

                # Upload data directly (more stable on macOS than mapBuffer)
                gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, size, data, gl.GL_STREAM_DRAW)

            except Exception as e:
                # Fallback: disable PBO for this frame
                print("PBO fallback:", e)
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
                data_fallback = data
            else:
                data_fallback = None

            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                gl.GL_RED,
                gl.GL_UNSIGNED_BYTE,
                data_fallback
            )

            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

            # Removed redundant per-frame texture parameter setting
            # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

            return tex_id

        self.tex_y = upload(self.tex_y, y, frame.width, frame.height, 0, self.pbo_y)
        self.tex_u = upload(self.tex_u, u, frame.width // 2, frame.height // 2, 1, self.pbo_u)
        self.tex_v = upload(self.tex_v, v, frame.width // 2, frame.height // 2, 2, self.pbo_v)

        self.program.setUniformValue("texY", 0)
        self.program.setUniformValue("texU", 1)
        self.program.setUniformValue("texV", 2)
        self.program.bind()
        gl.glBindVertexArray(self.vao)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_y)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_u)

        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_v)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # ---- draw only on video1 ----
        if self.objectName() != "video1":
            painter.end()
            self.program.release()
            return

        w = self.width()
        h = self.height()

        # position gauche milieu
        cx = 60
        cy = h // 2

        # rotation
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(-self.roll_w)

        pitch = self.pitch_w
        pitch_offset = int(pitch * 3)

        # ---- horizon ----
        painter.setPen(QPen(QColor(20, 20, 139), 2))
        painter.drawLine(-50, pitch_offset, 50, pitch_offset)
        painter.restore()

        # ---- triangle FIXE ----
        size = 40
        painter.setPen(QPen(QColor(0, 255, 120), 3))

        # position gauche milieu (même que horizon)
        cx = 60
        cy = h // 2

        painter.drawLine(cx, cy, cx, cy - size)
        painter.drawLine(cx, cy - size, cx - size, cy)
        painter.drawLine(cx, cy - size, cx + size, cy)
        painter.drawLine(cx - size, cy, cx + size, cy)

        painter.end()

        self.program.release()

    # QLabel compatibility
    def setScaledContents(self, *args):
        pass

    def setPixmap(self, *args):
        pass


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
                        self.audio_input.appendSampleBuffer_(sampleBuffer)


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
# USEFUL FUNCTIONS
# ======================================================
def quat_to_rot(q):
    w, x, y, z = q;
    x = -x;
    y = -y;
    z = -z
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


perm = np.array([
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
    [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
    [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
    [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
    [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]])


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
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.pitch = 0.0
        self.bank = 0.0
        self.heading = 0.0
        # optional aerobatic triangle marker (used for wingtip reference)
        self.show_triangle = False
        self.transparent_mode = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width();
        h = self.height()
        cx = w // 2;
        cy = h // 2
        # ---- gestion transparence ----
        if self.transparent_mode:
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.fillRect(self.rect(), Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        else:
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

        for deg in (10, 20, 30, 45, 60, 75, 90):
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
            size = 27
            triangle = QPolygon([
                QPoint(cx, cy),
                QPoint(cx, cy - size),
                QPoint(cx - size, cy),
                QPoint(cx + size, cy),
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


class AnalogVario(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fpm = 0.0
        self.setMinimumSize(100, 100)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        r = min(w, h) / 2 - 5

        cx = w / 2
        cy = h / 2

        # outer circle with semi-transparent background (same style as other instruments)
        pen = QPen(QColor("white"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))

        # graduations + labels
        painter.setPen(QPen(QColor("white"), 1))

        for f in [-2000, -1000, 0, 1000, 2000]:
            angle = (f / 2000.0) * 120
            theta = math.radians(180 + angle)

            x1 = cx + r * 0.7 * math.cos(theta)
            y1 = cy + r * 0.7 * math.sin(theta)
            x2 = cx + r * 0.9 * math.cos(theta)
            y2 = cy + r * 0.9 * math.sin(theta)

            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            # ---- LABEL (÷1000) ----
            label = str(int(f / 1000))

            xt = cx + r * 0.55 * math.cos(theta)
            yt = cy + r * 0.55 * math.sin(theta)

            painter.drawText(int(xt - 6), int(yt + 5), label)

        # clamp
        fpm = max(-2000, min(2000, self.fpm))

        # aiguille
        angle = (fpm / 2000.0) * 120
        # 0 at left (180°), positive up, negative down
        theta = math.radians(180 + angle)

        x2 = cx + r * 0.8 * math.cos(theta)
        y2 = cy + r * 0.8 * math.sin(theta)

        painter.setPen(QPen(QColor("white"), 3))
        painter.drawLine(int(cx), int(cy), int(x2), int(y2))


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

    def find_metar_for_time(self, t):
        idx = self.metar_df["time"].searchsorted(t)
        if idx == 0:
            return self.metar_df.iloc[0]
        if idx >= len(self.metar_df):
            return self.metar_df.iloc[-1]
        before = self.metar_df.iloc[idx - 1]
        after = self.metar_df.iloc[idx]
        if abs(t - before.time) < abs(after.time - t):
            return before
        else:
            return after

    def load_dataframe(self, file):
        # ---- ensure merged CSV exists before loading ----
        if not os.path.exists(file):
            print("ERROR: merged data file not found:")
            print("  ", file)
            print("Current working directory:", os.getcwd())
            print("Hint: verify that the merged CSV was generated or that the 'data' folder path is correct.")
            sys.exit(1)

        self.df = pd.read_csv(MERGED_DATA, low_memory=False)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], format="mixed", utc=True)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)
        self.frames_df = len(self.df)

        # ---- numpy caches for fast access inside the realtime loop ----
        self.gps_lat_vals = self.df["gps_lat"].to_numpy()
        self.gps_lon_vals = self.df["gps_lon"].to_numpy()
        self.gps_alt_vals = self.df["gps_alt"].to_numpy()
        self.gps_heading_vals = self.df["gps_heading"].to_numpy()
        self.gps_ias_vals = self.df["gps_ias"].to_numpy()
        self.gps_wind_speed_vals = self.df["era5_wind_speed"].to_numpy()
        self.gps_wind_direction_vals = self.df["era5_wind_direction"].to_numpy()
        self.timestamp_vals = self.df["timestamp"].to_numpy()
        self.x4_quat_w_vals = self.df["x4_quat_w"].to_numpy()
        self.x4_quat_x_vals = self.df["x4_quat_x"].to_numpy()
        self.x4_quat_y_vals = self.df["x4_quat_y"].to_numpy()
        self.x4_quat_z_vals = self.df["x4_quat_z"].to_numpy()
        self.x4_acc_x_vals = self.df["x4_acc_x"].to_numpy()
        self.x4_acc_y_vals = self.df["x4_acc_y"].to_numpy()
        self.x4_acc_z_vals = self.df["x4_acc_z"].to_numpy()
        self.gps_speed_vals = self.df["gps_speed"].to_numpy()
        self.gps_fpm_vals = self.df["gps_fpm"].to_numpy()
        self.t0_timestamp = self.df.timestamp.iloc[0]

        # ---- Wind components (vectorized) ----
        heading = np.radians(self.gps_heading_vals)
        wind_dir = np.radians(self.gps_wind_direction_vals)

        # angle relatif (normalisé -pi à pi)
        theta = heading - wind_dir
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        wind_speed_kt = self.gps_wind_speed_vals / 1.852

        self.headwind_vals = - wind_speed_kt * np.cos(theta)
        self.crosswind_vals = wind_speed_kt * np.sin(theta)

        self.metar_df = pd.read_csv(INPUT_METAR, encoding="utf-8")
        self.metar_df["time"] = pd.to_datetime(self.metar_df["time"], format="mixed", utc=True)

        # ======================================================
        # TOP remarquables
        # ======================================================
        mask = self.df['gps_speed'] > VITESSE_MISE_EN_LIGNE
        self.index_enligne_devol = mask.idxmax()
        # mask = self.df['gps_alt'] > 3000
        # self.index_entree_3000 = mask.idxmax()
        self.gps_max_alt = round(self.df['gps_alt'].max())
        print("Max Alt : ", self.gps_max_alt)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABView Version " + __version__)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(2024, 1200)

        self._init_state()
        self._init_video()
        self._init_audio()
        self._init_flight_state()

        self.init_UI()
        QTimer.singleShot(0, self.goto_mise_en_ligne)

        self.init_map_OSM_widget()
        self.map_view.loadFinished.connect(self.on_map_loaded)

        self.enable_matplotlib_gps = True
        self.init_gps_pyqtgraph()

        self.init_gfx()
        self.calibrate_gfx(0)

        self.compute_g_signed()
        self.build_energy_graph()
        self.init_timeline_zoom()
        QTimer.singleShot(0, self._build_all_timelines)

        self._init_timer()
        self._init_metar()

    # ------------------------------------------------------------------
    # Init helpers (called once from __init__)
    # ------------------------------------------------------------------

    def _init_state(self):
        """Charge le dataframe et initialise les variables de lecture."""
        self.load_dataframe(MERGED_DATA)
        self.i = 0
        self.idf = 0
        self.playing = True
        self.speed = 1
        self.current_video_time_utc = None
        self.stutter_count = 0
        self.recording = False

    def _init_video(self):
        """Ouvre les deux containers vidéo et calcule l'offset temporel DF↔vidéo."""
        self.video1_path = VIDEO1
        self.video2_path = VIDEO2

        # Probe pour compter les frames totales
        container_probe = av.open(VIDEO1)
        self.frames_video = container_probe.streams.video[0].frames
        container_probe.close()

        self.container1 = av.open(self.video1_path)
        self.container2 = av.open(self.video2_path)
        self.stream1 = self.container1.streams.video[0]
        self.stream2 = self.container2.streams.video[0]
        self.stream1.thread_type = "AUTO"
        self.stream2.thread_type = "AUTO"
        self.decoder1 = self.container1.decode(self.stream1)
        self.decoder2 = self.container2.decode(self.stream2)
        fps_raw = float(self.stream1.average_rate)
        self.fps_video = fps_raw if fps_raw > 0 else 30.0

        self.video1_start = get_mp4_creation_datetime(self.video1_path)
        self.video2_start = get_mp4_creation_datetime(self.video2_path)
        # 🔑 OFFSET TEMPOREL : décalage entre l'origine du dataframe et le début vidéo
        self.video_df_offset = self.df.timestamp.iloc[0] - self.video1_start

    def _init_audio(self):
        """Initialise le pipeline audio (PyAV → Qt PCM). Silencieux si indisponible."""
        self.sync_enabled = False
        self.startup_time_ms = None
        try:
            self.audio_container = av.open(self.video1_path)
            self.audio_stream = self.audio_container.streams.audio[0]
            # Démux par paquets (plus stable que l'itérateur de frames)
            self.audio_packets = self.audio_container.demux(self.audio_stream)
            # Rééchantillonnage vers stéréo s16 (compatible Qt)
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
            self.audio_clock_sec = 0.0
        except Exception:
            self.audio_stream = None

    def _init_flight_state(self):
        """Initialise les variables d'état vol (INU, instruments, bookmarks, vue GPS)."""
        # Accélérations / G
        self.g_min = float("inf")
        self.g_max = float("-inf")
        self.gs_max = float("-inf")
        self.acc_vec_filtered = None
        self.g_filter_alpha = 0.15

        # Attitude
        self.montage_pitch_angle = PITCH_MONTAGE_PAR_DEFAUT
        theta_x = np.deg2rad(self.montage_pitch_angle)
        self.R_x_20_cached = np.array([[1, 0, 0],
                                       [0, np.cos(theta_x), -np.sin(theta_x)],
                                       [0, np.sin(theta_x),  np.cos(theta_x)]])
        self.pitch_deg = 0
        self.pitch_w = 0
        self.roll_w = 0
        self.bank_deg = 0.0
        self.heading_deg = 0.0

        # Lissage instruments (interpolation visuelle)
        self.smooth_speed = None
        self.smooth_alt = None
        self.instrument_alpha = 0.2  # 0 = très lisse, 1 = pas de lissage

        # Bookmarks
        self.bookmarks = []
        self.bookmarks_df = None
        self.last_bookmark_frame = None
        self.bookmark_overlay = None
        self._bm_frames_cache = np.array([], dtype=int)
        self._bm_names_cache = np.array([], dtype=object)

        # Stylesheet color caches (avoid re-parsing CSS every frame)
        self._last_g_color = None
        self._last_speed_color = None

        # Vue GPS 3-D
        self.firstGPS = True
        self.last_azim = 0

    def _rebuild_R_x_20(self):
        """Recompute the cached pitch-mounting rotation matrix after angle change."""
        theta_x = np.deg2rad(self.montage_pitch_angle)
        self.R_x_20_cached = np.array([[1, 0, 0],
                                       [0, np.cos(theta_x), -np.sin(theta_x)],
                                       [0, np.sin(theta_x),  np.cos(theta_x)]])

    def _rebuild_bookmark_cache(self):
        """Rebuild numpy caches for bookmark hot-loop lookup."""
        if self.bookmarks_df is not None and not self.bookmarks_df.empty:
            self._bm_frames_cache = self.bookmarks_df["frame"].to_numpy(dtype=int)
            self._bm_names_cache = self.bookmarks_df["name"].to_numpy(dtype=object)
        else:
            self._bm_frames_cache = np.array([], dtype=int)
            self._bm_names_cache = np.array([], dtype=object)

    def _init_timer(self):
        """Configure le timer temps-réel à 30 fps avec compensation de dérive."""
        self.target_fps = 30
        self.frame_period_ms = 1000 / self.target_fps

        self.clock = QElapsedTimer()
        self.clock.start()
        self.next_frame_time = 0  # scheduling absolu pour éviter la dérive

        self.timer = QTimer()
        self.timer.timeout.connect(self.main_loop)
        self.timer.start(1)

    def _init_metar(self):
        """Charge le METAR correspondant au début du vol."""
        t_start = self.df['timestamp'][0]
        metar_row = self.find_metar_for_time(t_start)
        self.last_metar = metar_row.metar

    def closeEvent(self, event):
        """Sécurise la fermeture (évite crash wgpu / rendercanvas)"""
        try:

            # ---- stop Qt timer FIRST ----
            if hasattr(self, "timer"):
                self.timer.stop()

            # ---- stop pygfx / rendercanvas cleanly ----
            if hasattr(self, "gfx_canvas") and self.gfx_canvas is not None:
                # ---- CRITICAL: stop rendercanvas loop BEFORE Qt deletes widget ----
                if hasattr(self.gfx_canvas, "_rc_canvas"):
                    loop = getattr(self.gfx_canvas._rc_canvas, "_loop", None)
                    if loop is not None:
                        loop.stop(force=True)
                # ---- detach from Qt ----
                self.gfx_canvas.setParent(None)
                self.gfx_canvas.hide()

            # ---- process remaining Qt events (flush callbacks) ----
            QApplication.processEvents()

            # ---- now safe deletion ----
            if hasattr(self, "gfx_canvas") and self.gfx_canvas is not None:
                self.gfx_canvas.close()
                self.gfx_canvas.deleteLater()

        except Exception:
            pass

        event.accept()

    def init_UI(self):
        # ---- UI ----
        central = QWidget()
        # central.setStyleSheet("background-color: white;")  # fond gris
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

        # ---- Toggle timeline zoom ----
        self.act_toggle_timeline_zoom = QAction("Zoom Timeline", self)
        self.act_toggle_timeline_zoom.setCheckable(True)
        self.act_toggle_timeline_zoom.setChecked(getattr(self, "timeline_zoom", False))
        self.act_toggle_timeline_zoom.setShortcut(QKeySequence("Z"))
        self.act_toggle_timeline_zoom.setShortcutContext(Qt.ApplicationShortcut)
        self.act_toggle_timeline_zoom.triggered.connect(self.toggle_timeline_zoom)
        menu_settings.addAction(self.act_toggle_timeline_zoom)

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

        self.video1 = VideoYUVOpenGLWidget(self)
        self.video1.setObjectName("video1")

        self.video2 = VideoYUVOpenGLWidget(self)
        self.video2.setObjectName("video2")
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

        self.video1_wind_label = QLabel("", self.video1)
        self.video1_wind_label.setAlignment(Qt.AlignCenter)
        self.video1_wind_label.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 14px; font-weight: bold;"
        )
        self.video1_wind_label.adjustSize()
        self.video1_wind_label.raise_()

        # ---- Analog badin (circular airspeed indicator) ----
        self.video1_badin = AnalogBadin(self.video1)
        self.video1_badin.setGeometry(10, int(self.video1.height() / 2) - 80, 160, 160)
        self.video1_badin.show()

        # ---- Analog altimeter (right side) ----
        self.video1_altimeter = AnalogAltimeter(self.video1)
        self.video1_altimeter.setGeometry(self.video1.width() - 170, int(self.video1.height() / 2) - 80, 160, 160)
        self.video1_altimeter.show()

        self.video1_vario = AnalogVario(self.video1)
        self.video1_vario.setFixedSize(120, 120)
        self.video1_vario.show()

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
        # Not needed for VideoYUVWidget (handled in paintEvent)
        #pass
        # prevent QLabel from expanding to the raw video resolution
        self.video1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # give a reasonable default display size
        self.video1.setMinimumSize(0, 0)
        self.video2.setMinimumSize(0, 0)
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

        self.act_toggle_grid_xz = QAction("Grille verticale XZ", self)
        self.act_toggle_grid_xz.setCheckable(True)
        self.act_toggle_grid_xz.setChecked(False)  # caché par défaut
        self.act_toggle_grid_xz.triggered.connect(self.toggle_grid_vertical_xz)
        menu_settings.addAction(self.act_toggle_grid_xz)

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
        self.slider.setRange(0, self.frames_video - 1)
        self.slider.valueChanged.connect(self.on_slider)

        self.g_timeline = QLabel(self)
        self.g_timeline.setFixedHeight(12)
        self.g_timeline.setStyleSheet("background-color: black;")
        self.g_timeline.setScaledContents(False)
        self.g_timeline.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.g_timeline.setMinimumWidth(0)
        self.grid.addWidget(self.g_timeline, self.grid.rowCount(), 0, 1, self.grid.columnCount())
        self.g_timeline.installEventFilter(self)

        self.alt_timeline = QLabel(self)
        self.alt_timeline.setFixedHeight(12)
        self.alt_timeline.setStyleSheet("background-color: black;")
        self.alt_timeline.setScaledContents(False)
        self.alt_timeline.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.alt_timeline.setMinimumWidth(0)
        self.grid.addWidget(self.alt_timeline, self.grid.rowCount(), 0, 1, self.grid.columnCount())
        self.alt_timeline.installEventFilter(self)

        self.fpm_timeline = QLabel(self)
        self.fpm_timeline.setFixedHeight(12)
        self.fpm_timeline.setStyleSheet("background-color: black;")
        self.fpm_timeline.setScaledContents(False)
        self.fpm_timeline.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.fpm_timeline.setMinimumWidth(0)
        self.grid.addWidget(self.fpm_timeline, self.grid.rowCount(), 0, 1, self.grid.columnCount())
        self.fpm_timeline.installEventFilter(self)

        # self.timestamp_label = QLabel(alignment=Qt.AlignCenter)
        self.video2_date_label = QLabel("", self.video2)
        self.video2_date_label.setStyleSheet(
            "color: black; background-color: white; padding: 4px 10px; font-family: 'Menlo'; font-size: 18px; font-weight: bold;"
        )
        self.video2_date_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.video2_date_label.show()

        ts0 = self.df.timestamp.iloc[0]
        mois_fr = [
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ]

        text = f"{ts0.day} {mois_fr[ts0.month - 1]} {ts0.year} {ts0.strftime('%H:%M:%S')}"

        self.video2_date_label.setText(text)
        self.video2_date_label.adjustSize()

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
        # self.layout.addWidget(self.timestamp_label)
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
        self.gps_view = glpg.GLViewWidget()
        self.gps_view.setBackgroundColor('w')
        self.gps_lastzoom = 4
        self.gps_view.setCameraPosition(distance=self.gps_lastzoom)
        self.grid.addWidget(self.gps_view, 1, 2, 1, 1)

    def compute_g_signed(self):
        theta_x = np.deg2rad(self.montage_pitch_angle)
        R_x_20 = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        down_original = np.array([0.0, 0.0, -1.0])

        # ---- Acceleration vector (N x 3) ----
        acc = np.column_stack((
            -self.df["x4_acc_x"].to_numpy(),
            -self.df["x4_acc_y"].to_numpy(),
            -self.df["x4_acc_z"].to_numpy()
        ))

        # norme (G)
        acc_norm = np.linalg.norm(acc, axis=1)
        self.df["g_force"] = acc_norm / 9.81
        # éviter division par 0
        acc_norm_safe = np.where(acc_norm == 0, 1, acc_norm)
        # vecteurs unitaires
        acc_unit = acc / acc_norm_safe[:, None]
        # ---- rotation (attention: vectorisation) ----
        R_total = R_x_20 @ perm[R_recalage_repere]
        # appliquer rotation à tous les vecteurs
        acc_vec = (R_total @ acc_unit.T).T  # (N,3)
        # ---- angle avec verticale ----
        dot = np.einsum("ij,j->i", acc_vec, down_original)
        norm_acc = np.linalg.norm(acc_vec, axis=1)
        cos_theta = dot / norm_acc
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_theta))
        # signe du G
        g_signed = self.df["g_force"].to_numpy().copy()
        g_signed[angles > 90] *= -1
        self.df["g_signed"] = g_signed

    def build_g_timeline(self):
        if "g_signed" not in self.df.columns:
            return
        values = self.df["g_signed"].to_numpy()
        # ---- apply timeline zoom ----
        if getattr(self, "timeline_zoom", False):
            start = max(0, getattr(self, "timeline_start", 0))
            end = min(len(values) - 1, getattr(self, "timeline_end", len(values) - 1))
            if end > start:
                values = values[start:end + 1]

        n = len(values)
        width = max(10, self.g_timeline.width())
        height = 12

        img = QImage(width, height, QImage.Format_RGB888)

        g_min = -2.0
        g_max = 4.0

        def get_color(g):
            # mapping DISCRET sans interpolation
            if g < 0.5:
                return (0, 180, 255)      # bleu clair
            elif g < 1.5:
                return (0, 255, 0)        # vert
            else:
                return (255, 50, 50)      # rouge vif

        if n == 0:
            return

        if width >= n:
            # 🔥 CAS CRITIQUE : peu de data → on étire correctement
            for x in range(width):
                idx = int(x * n / width)
                if idx >= n:
                    idx = n - 1

                fpm = values[idx]

                r, g, b = get_color(fpm)
                color = (r << 16) | (g << 8) | b

                for y in range(height):
                    img.setPixel(x, y, color)

        else:
            # 📊 CAS NORMAL : beaucoup de data → agrégation
            for x in range(width):
                start = int(x * n / width)
                end = int((x + 1) * n / width)

                if end <= start:
                    end = start + 1

                segment = values[start:end]
                fpm = segment[np.argmax(np.abs(segment))]

                r, g, b = get_color(fpm)
                color = (r << 16) | (g << 8) | b

                for y in range(height):
                    img.setPixel(x, y, color)

        pix = QPixmap.fromImage(img)
        self.g_timeline.setPixmap(pix)

        # ---- Legend "G Forces" (top-left INSIDE the timeline) ----
        if not hasattr(self, "g_timeline_legend"):
            # IMPORTANT: parent = g_timeline (not main window)
            self.g_timeline_legend = QLabel("GForce", self.g_timeline)
            self.g_timeline_legend.setStyleSheet(
                "color: black; background-color: white; padding: 2px 6px; font-family: 'Menlo'; font-size: 10px; font-weight: bold;"
            )
            self.g_timeline_legend.setAttribute(Qt.WA_TransparentForMouseEvents)
            self.g_timeline_legend.raise_()

        label = "GForce(Z)" if getattr(self, "timeline_zoom", False) else "GForce"
        self.g_timeline_legend.setText(label)
        self.g_timeline_legend.adjustSize()

        # position INSIDE the bar (top-left)
        self.g_timeline_legend.move(0, 0)
        self.g_timeline_legend.raise_()
        self.g_timeline_legend.show()

    def build_altitude_timeline(self):
        if "gps_alt" not in self.df.columns:
            return

        values = self.df["gps_alt"].to_numpy()
        # ---- apply timeline zoom ----
        if getattr(self, "timeline_zoom", False):
            start = max(0, getattr(self, "timeline_start", 0))
            end = min(len(values) - 1, getattr(self, "timeline_end", len(values) - 1))
            if end > start:
                values = values[start:end + 1]
        n = len(values)

        width = max(10, self.alt_timeline.width())
        height = 12

        img = QImage(width, height, QImage.Format_RGB888)

        # No more alt_min/alt_max-based gradient; use fixed stops
        def get_color(a):
            # points de contrôle (altitude ft, couleur RGB)
            stops = [
                (0, (120, 120, 120)),  # gris (sol)
                (3000, (0, 180, 255)),  # bleu clair (bas)
                (4000, (0, 255, 0)),  # vert (normal)
                (5000, (255, 200, 0)),  # orange (attention)
                (6000, (255, 0, 0)),  # rouge (limite)
                (7000, (180, 0, 0)),  # rouge foncé (extrême)
            ]

            # clamp
            if a <= stops[0][0]:
                return stops[0][1]
            if a >= stops[-1][0]:
                return stops[-1][1]

            # interpolation linéaire entre stops
            for i in range(len(stops) - 1):
                a0, c0 = stops[i]
                a1, c1 = stops[i + 1]

                if a0 <= a <= a1:
                    t = (a - a0) / (a1 - a0 + 1e-6)

                    r = int(c0[0] + (c1[0] - c0[0]) * t)
                    g = int(c0[1] + (c1[1] - c0[1]) * t)
                    b = int(c0[2] + (c1[2] - c0[2]) * t)

                    return (r, g, b)

        for x in range(width):
            start = int(x * n / width)
            end = int((x + 1) * n / width)
            if end <= start:
                end = start + 1
            alt = np.mean(values[start:end])

            r, g, b = get_color(alt)
            color = (r << 16) | (g << 8) | b

            for y in range(height):
                img.setPixel(x, y, color)

        pix = QPixmap.fromImage(img)
        self.alt_timeline.setPixmap(pix)

        # ---- Legend "Altitude" ----
        try:
            if not hasattr(self, "alt_timeline_legend"):
                self.alt_timeline_legend = QLabel("Altitude", self.alt_timeline)
                self.alt_timeline_legend.setStyleSheet(
                    "color: black; background-color: white; padding: 2px 6px; font-family: 'Menlo'; font-size: 10px; font-weight: bold;"
                )
                self.alt_timeline_legend.setAttribute(Qt.WA_TransparentForMouseEvents)
                self.alt_timeline_legend.raise_()

            label = "Alt(Z)" if getattr(self, "timeline_zoom", False) else "Alt"
            self.alt_timeline_legend.setText(label)
            self.alt_timeline_legend.adjustSize()
            self.alt_timeline_legend.move(0, 0)
            self.alt_timeline_legend.show()

        except Exception:
            pass

    def build_fpm_timeline(self):
        if "gps_fpm" not in self.df.columns:
            return

        values = self.df["gps_fpm"].to_numpy()
        # ---- apply timeline zoom ----
        if getattr(self, "timeline_zoom", False):
            start = max(0, getattr(self, "timeline_start", 0))
            end = min(len(values) - 1, getattr(self, "timeline_end", len(values) - 1))
            if end > start:
                values = values[start:end + 1]
        n_original = len(values)

        # ---- downsample: keep peak (abs max) every 100 points ----
        chunk_size = 100
        n = len(values)

        if n > chunk_size:
            trimmed = values[: (n // chunk_size) * chunk_size]
            reshaped = trimmed.reshape(-1, chunk_size)

            idx = np.argmax(np.abs(reshaped), axis=1)
            values = reshaped[np.arange(len(idx)), idx]

        n = len(values)
        # if too few points after downsampling, repeat to fill width better
        if n < self.fpm_timeline.width():
            repeat_factor = int(np.ceil(self.fpm_timeline.width() / max(1, n)))
            values = np.repeat(values, repeat_factor)
            n = len(values)

        width = max(10, self.fpm_timeline.width())
        height = 12

        img = QImage(width, height, QImage.Format_RGB888)

        def get_color(fpm):
            if fpm > 1500:
                return (0, 255, 0)      # montée forte (vert)
            elif fpm < -1500:
                return (255, 0, 0)      # descente forte (rouge)
            else:
                return (180, 180, 180)  # neutre (gris clair)

        for x in range(width):
            start = int(x * n / width)
            end = int((x + 1) * n / width)
            if end <= start:
                end = start + 1
            segment = values[start:end]
            fpm = segment[np.argmax(np.abs(segment))]

            r, g, b = get_color(fpm)
            color = (r << 16) | (g << 8) | b

            for y in range(height):
                img.setPixel(x, y, color)

        pix = QPixmap.fromImage(img)
        self.fpm_timeline.setPixmap(pix)

        # ---- Legend "Vario" ----
        if not hasattr(self, "fpm_timeline_legend"):
            self.fpm_timeline_legend = QLabel("Vario", self.fpm_timeline)
            self.fpm_timeline_legend.setStyleSheet(
                "color: black; background-color: white; padding: 2px 6px; font-family: 'Menlo'; font-size: 10px; font-weight: bold;")
            self.fpm_timeline_legend.setAttribute(Qt.WA_TransparentForMouseEvents)
            self.fpm_timeline_legend.raise_()

        label = "Vario(Z)" if getattr(self, "timeline_zoom", False) else "Vario"
        self.fpm_timeline_legend.setText(label)
        self.fpm_timeline_legend.adjustSize()
        self.fpm_timeline_legend.move(0, 0)
        self.fpm_timeline_legend.show()

    def init_timeline_zoom(self):
        """Initialize timeline zoom bounds based on altitude > 3000 ft."""
        # already initialized → do nothing
        if hasattr(self, "timeline_start") and hasattr(self, "timeline_end"):
            return

        self.timeline_zoom = False

        if "gps_alt" not in self.df.columns or len(self.df) == 0:
            self.timeline_start = 0
            self.timeline_end = len(self.df) - 1
            return

        alt = self.df["gps_alt"].to_numpy()
        mask = alt > 3000

        if not np.any(mask):
            # fallback: no segment above 3000 ft
            self.timeline_start = 0
            self.timeline_end = len(alt) - 1
            return

        # first occurrence
        self.timeline_start = int(np.argmax(mask))

        # last occurrence
        self.timeline_end = int(len(mask) - 1 - np.argmax(mask[::-1]))

    def _build_all_timelines(self):
        self.build_g_timeline()
        self.build_altitude_timeline()
        self.build_fpm_timeline()

    def toggle_timeline_zoom(self, checked):
        """Enable/disable timeline zoom and rebuild timelines."""
        self.timeline_zoom = checked
        self._build_all_timelines()
        self.update_g_timeline_cursor()

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent

        if event.type() == QEvent.Resize:
            if obj in (self.g_timeline, self.alt_timeline, self.fpm_timeline):
                QTimer.singleShot(0, self._build_all_timelines)

        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "_resize_timer"):
            self._resize_timer.stop()
        else:
            self._resize_timer = QTimer()
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._build_all_timelines)

        self._resize_timer.start(50)

    def build_energy_graph(self):
        # ---- Precompute full energy vector once ----
        speed = self.df["gps_speed"].to_numpy()
        alt = self.df["gps_alt"].to_numpy()
        self.energy_full = 0.5 * ((speed/12.96) ** 2) + 9.81 * alt * 0.3048

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

        total = max(1, (self.frames_video - 1))

        for _, row in self.bookmarks_df.iterrows():
            try:
                frame = int(row["frame"])
            except:
                continue

            t = frame / total
            x = int(slider_x + t * slider_w)

            tick = QFrame(self.centralWidget())
            tick.setStyleSheet("background-color: rgb(80,80,80);")  # gris foncé
            slider_h = geom.height()
            tick.setGeometry(x, slider_y + slider_h, 2, 6)

            # tooltip (optionnel)
            tick.setToolTip(str(row.get("name", "")))

            tick.show()
            self.bookmark_ticks.append(tick)

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

    def toggle_recording(self, checked):
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
                # WARNING: higher scale_factor = more CPU/GPU usage
                scale_factor = 2  # increase resolution (e.g. 2 = 2x, 3 = 3x)
                config.setWidth_(int(target_window.frame().size.width * scale_factor))
                config.setHeight_(int(target_window.frame().size.height * scale_factor))
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
                self.sc_handler.setWriter_input_adaptor_(self.sc_writer, self.sc_input, self.sc_adaptor,
                                                         self.sc_audio_input)

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
                self.recording = True
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
                                self.recording = False
                            except Exception as e:
                                print("finishWriting error:", e)

                        dispatch.dispatch_async(
                            dispatch.dispatch_get_global_queue(0, 0),
                            _finish
                        )

                    except Exception:
                        pass

                    print("Recording stop")
                    self.recording = False
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
        self._rebuild_R_x_20()
        self.update_pitch_cam_menu()

        # refresh display immediately even when paused
        if hasattr(self, "row"):
            try:
                self.update_gfx_orientation()
                self.update_video_label()
            except Exception:
                pass

    def pitch_cam_minus(self):
        """Decrease camera mounting pitch by 1 degree."""
        self.montage_pitch_angle -= 1
        self._rebuild_R_x_20()
        self.update_pitch_cam_menu()

        # refresh display immediately even when paused
        if hasattr(self, "row"):
            try:
                self.update_gfx_orientation()
                self.update_video_label()
            except Exception:
                pass

    def toggle_axes_visibility(self):
        """Show or hide the pygfx world axes."""
        visible = self.act_toggle_axes.isChecked()

        if hasattr(self, "gfx_axes_x"):
            self.gfx_axes_x.visible = visible
            self.gfx_axes_y.visible = visible
            self.gfx_axes_z.visible = visible

    def toggle_grid_vertical_xz(self, checked):
        try:
            if hasattr(self, "grid_vertical_xz"):
                self.grid_vertical_xz.setVisible(checked)
        except Exception:
            pass

    def init_map_OSM_widget(self):
        # ---- OpenStreetMap (OSM) ----
        self.map_ready = False
        self.map_view = QWebEngineView()
        # HTML Leaflet avec OpenStreetMap
        lat0 = self.df.gps_lat.iloc[0]
        lon0 = self.df.gps_lon.iloc[0]
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

        # ---- Wind overlay (top-right of OSM map) ----
        self.map_wind_label = QLabel("", parent_widget)
        self.map_wind_label.setAlignment(Qt.AlignCenter)
        self.map_wind_label.setStyleSheet(
            "color: black; background-color: white; padding: 6px; font-family: 'Menlo'; font-size: 14px; font-weight: bold;"
        )
        self.map_wind_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.map_wind_label.adjustSize()
        self.map_wind_label.raise_()

        self._position_map_wind_label()

        # ---- Wind arrow base (pointing UP) ----
        size = 40
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)

        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor("black"))
        pen.setWidth(2)
        p.setPen(pen)

        cx = size // 2

        # ---- tige ----
        p.drawLine(cx, size - 5, cx, 10)

        # ---- tête (2 petits traits) ----
        p.drawLine(cx, 10, cx - 6, 16)
        p.drawLine(cx, 10, cx + 6, 16)

        p.end()

        self.wind_arrow_base = pix

        self.map_wind_arrow = QLabel(parent_widget)
        self.map_wind_arrow.setPixmap(pix)
        self.map_wind_arrow.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.map_wind_arrow.adjustSize()
        self.map_wind_arrow.raise_()

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

    def _position_map_wind_label(self):
        if not hasattr(self, "map_wind_label") or not hasattr(self, "map_view"):
            return

        geom = self.map_view.geometry()

        self.map_wind_label.adjustSize()

        x = geom.x() + geom.width() - self.map_wind_label.width() - 10
        y = geom.y() + 10

        self.map_wind_label.move(x, y)
        self.map_wind_label.raise_()

        if hasattr(self, "map_wind_arrow"):
            self.map_wind_arrow.adjustSize()

            x_arrow = x + (self.map_wind_label.width() - self.map_wind_arrow.width()) // 2
            y_arrow = y + self.map_wind_label.height() + 5

            self.map_wind_arrow.move(x_arrow, y_arrow)
            self.map_wind_arrow.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "map_metar_label"):
            self._position_map_metar_label()

        if hasattr(self, "map_wind_label"):
            self._position_map_wind_label()

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
            line = glpg.GLLinePlotItem(
                pos=np.zeros((2, 3)),
                color=(1, 1, 1, 1),
                width=8,
                antialias=True
            )
            line._tube_offset = np.array(off)
            self.gps_view.addItem(line)
            self.gps_lines.append(line)

        # aircraft position marker
        self.gps_point = glpg.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            size=15,  # screen pixels
            pxMode=True,  # keep constant size on screen
            color=(1.0, 0.0, 0.0, 1.0)
        )

        # ensure the point is rendered normally and not blended like a large sphere
        self.gps_point.setGLOptions('opaque')

        # ground grid (visual reference in meters)
        grid = glpg.GLGridItem()
        grid.setSize(4, 2)  # 2 km x 2 km area
        grid.setSpacing(0.25, 0.25)  # grid every 100 m
        grid.translate(0, 0, -1)  # slightly below aircraft
        grid.rotate(BOX_HEADING, 0, 0, 1)
        grid.setColor((150, 150, 150))
        self.gps_view.addItem(grid)

        # ---- vertical grid (YZ plane) ----
        self.grid_vertical_yz = glpg.GLGridItem()
        self.grid_vertical_yz.setSize(4, 2)
        self.grid_vertical_yz.setSpacing(0.25, 0.25)
        self.grid_vertical_yz.rotate(90, 1, 0, 0)
        self.grid_vertical_yz.translate(0, -1, 0)
        self.grid_vertical_yz.setColor((135, 206, 235))
        self.gps_view.addItem(self.grid_vertical_yz)

        # ---- vertical grid (XZ plane) ----
        self.grid_vertical_xz = glpg.GLGridItem()
        self.grid_vertical_xz.setSize(2, 2)
        self.grid_vertical_xz.setSpacing(0.25, 0.25)
        self.grid_vertical_xz.rotate(90, 0, 1, 0)
        self.grid_vertical_xz.translate(-1, 0, 0)
        self.grid_vertical_xz.setColor((135, 206, 235))
        self.gps_view.addItem(self.grid_vertical_xz)
        self.grid_vertical_xz.setVisible(False)

        # self.gps_view.addItem(self.gps_line)  # REMOVED
        # self.gps_view.addItem(self.gps_point)

        # stl_path = os.path.join(os.path.dirname(__file__), "plane.stl")

        try:
            m = mesh.Mesh.from_file(STL_SIMPLE_PLANE_FILE)
            # supprimer triangles dégénérés
            v = m.vectors
            valid = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) > 1e-6
            v = v[valid]
            vertices = v.reshape(-1, 3)

            vertices *= 0.03  # Scale down
            vertices -= vertices.mean(axis=0)  # centrage
            # rotation axes
            R_fix = np.array([
                [0, 1, 0],  # X = nez
                [1, 0, 0],  # Y = ailes
                [0, 0, 1]])
            vertices = (R_fix @ vertices.T).T
            # inversion gauche/droite
            vertices[:, 1] *= -1

            faces = np.arange(len(vertices)).reshape(-1, 3)
            meshdata = glpg.MeshData(vertexes=vertices, faces=faces)
            self.gps_aircraft = glpg.GLMeshItem(
                meshdata=meshdata,
                smooth=False,
                color=(0.8, 0.8, 0.8, 0.2),  # gris clair
                shader='shaded',
                drawEdges=False,
                glOptions='translucent')

        except Exception as e:
            print("STL load failed, fallback disabled:", e)
            self.gps_aircraft = glpg.GLScatterPlotItem(
                pos=np.array([[0, 0, 0]]),
                size=10,
                color=(1, 0, 0, 1)
            )
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
        self.gps_vertical_line = glpg.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            color=(1, 0, 1, 1),  # violet
            width=3,
            antialias=True
        )
        self.gps_view.addItem(self.gps_vertical_line)

        # ---- ground projection of trajectory (shadow on ground) ----
        self.gps_shadow = glpg.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            # color=(0, 0, 1, 0.6), # gray
            color=(0.6, 0.3, 0.1, 1),  # marron
            width=2,
            antialias=True
        )
        self.gps_view.addItem(self.gps_shadow)

        self.gps_box_projection = glpg.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            # color=(0, 0, 1, 0.6), # gray
            color=(0, 0, 1, 1),  # 🔵 bleu (RGBA)
            width=2, antialias=True
        )
        self.gps_view.addItem(self.gps_box_projection)

        # ---- altitude scale overlay (Red Bull style vertical scale) ----
        self.altitude_scale_labels = []
        altitude_scale = list(range(0, max(5500, int(math.ceil(self.gps_max_alt / 500) * 500)) + 500, 500))

        for z in altitude_scale:
            label = QLabel(f"{z}ft", self.gps_view)
            label.setStyleSheet(
                "color: black; background-color: transparent; padding:2px; font-family:'Menlo'; font-size:10px;")
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
            " stop:0 rgba(0,120,255,200),"  # blue below 100 km/h
            " stop:0.25 rgba(0,120,255,200),"  # 100/400
            " stop:0.25 rgba(0,180,0,180),"
            " stop:0.75 rgba(0,180,0,180),"  # up to 300 km/h
            " stop:0.75 rgba(255,220,0,200),"
            " stop:0.85 rgba(255,220,0,200),"  # 300–340 km/h
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
            " stop:0 rgba(0,120,255,200),"  # -2G
            " stop:0.25 rgba(0,120,255,200),"  # -1G
            " stop:0.40 rgba(0,200,0,200),"  # 0G
            " stop:0.55 rgba(0,200,0,200),"  # 1G
            " stop:0.65 rgba(255,220,0,200),"  # 2G
            " stop:0.85 rgba(255,0,0,220),"  # 3G
            " stop:1 rgba(160,0,255,220));"  # 4G max
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

        max_alt = int(self.gps_max_alt / 500 + 1) * 500

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
        # length = int((abs(fpm_clamped) / fpm_max) * max_len)
        length = int((abs(fpm_clamped) / fpm_max) ** 0.6 * max_len)  # non linéaire pour amplifier faible valeur

        # position X (à côté du curseur altitude)
        x_vario = bar_x + 4  # à gauche du curseur

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
        # x_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [length, 0, 0]], dtype=np.float32))
        # elf.gfx_vec_x = gfx.Line(x_geom, gfx.LineMaterial(color="red", thickness=1),)
        y_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [0, length, 0]], dtype=np.float32))
        self.y_geom = y_geom  # Reference for later updates
        self.gfx_vec_y = gfx.Line(
            y_geom,
            gfx.LineMaterial(color="green", thickness=8, depth_test=False),
        )
        # z_geom = gfx.Geometry(positions=np.array([[0, 0, 0], [0, 0, length]], dtype=np.float32))
        # self.gfx_vec_z = gfx.Line(z_geom, gfx.LineMaterial(color="blue", thickness=1),)
        # self.gfx_object.add(self.gfx_vec_x);
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
        self.gfx_vec_acc = gfx.Line(self.acc_geom, gfx.LineMaterial(color="green", thickness=4, depth_test=False), )
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
        # self.gfx_scene.background = gfx.Background(None, gfx.BackgroundMaterial((1, 1, 1, 1)))
        self.gfx_display.show(self.gfx_scene)
        # self.gfx_display.renderer.clear_color = (1, 1, 1, 1)

        # ---- Camera configuration : Z axis up ----
        cam = self.gfx_display.camera
        cam.local.position = (600, 600, 400)  # closer camera for stronger initial zoom
        cam.world.reference_up = (refcam[0], refcam[1], refcam[2])
        cam.look_at((0, 0, 0))

        # Camera
        self.gfx_box = gfx.Mesh(
            gfx.box_geometry(240, 120, 600),  # X, Y, Z
            gfx.MeshPhongMaterial(color="Gray"), )
        self.gfx_box.local.position = (0, 0, 0)
        self.gfx_object.add(self.gfx_box)
        # Hidden by default
        self.gfx_box.visible = False

        # sphère rouge au-dessus du parallélépipède
        sphere_radius = 90
        self.gfx_sphere = gfx.Mesh(
            gfx.sphere_geometry(radius=sphere_radius),
            gfx.MeshPhongMaterial(color="black"), )
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
                mesh.local.position = (620, 500, -140)  # for cap 10
                # ---- Parametric rotation: 180° around Z + tilt around X ----
                # ---- STL tilt parameter (degrees) ----
                self.stl_tilt_deg = 5.0  # default 5° around X
                theta_x = math.radians(self.stl_tilt_deg)

                # Quaternion for 180° around Z
                qz = np.array([0.0, 0.0, 1.0, 0.0])  # (x, y, z, w)

                # Quaternion for tilt around X
                qx = np.array([
                    math.sin(theta_x / 2.0), 0.0, 0.0, math.cos(theta_x / 2.0)])

                # Quaternion multiplication q = qz * qx
                x1, y1, z1, w1 = qz
                x2, y2, z2, w2 = qx

                q = np.array([
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])

                mesh.local.rotation = tuple(q)
                self.gfx_object.add(mesh)

            # print("CAP10.STL loaded successfully")
        except Exception as e:
            print(f"Error loading CAP10.STL: {e}")

        self.gfx_canvas = self.gfx_display.canvas
        self.gfx_canvas.setAttribute(Qt.WA_DeleteOnClose, False)
        # self.gfx_alive = True
        # self.gfx_render_enabled = True
        self.gfx_canvas.destroyed.connect(self.on_gfx_destroyed)
        # self.gfx_canvas.setStyleSheet("background-color: white;")
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
        self.df_info_label.setStyleSheet(
            "color: gray; background-color: transparent; padding: 4px; font-family: 'Menlo';")
        self.df_info_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.df_info_label.adjustSize()
        self.df_info_label.move(10, 10)
        self.df_info_label.raise_()

        # ---- Timestamp overlay (ensure multi-line support) ----
        # if hasattr(self, "timestamp_label"):
        #    self.timestamp_label.setWordWrap(True)

        # ---- Pitch overlay (custom position) ----
        self.pitch_label = QLabel("Pitch:", self.gfx_canvas)
        self.pitch_label.setStyleSheet(
            "color: green; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 28px; font-weight: bold;")
        self.pitch_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.pitch_label.adjustSize()
        # position top-right of main artificial horizon
        try:
            hx = self.hud_horizon.x() + self.hud_horizon.width() + 10
            hy = self.hud_horizon.y()
        except Exception:
            hx, hy = 180, 10
        self.pitch_label.move(hx, hy)
        self.pitch_label.raise_()

        # ---- Inclination overlay (custom position) ----
        self.roll_label = QLabel("Bank:", self.gfx_canvas)
        self.roll_label.setStyleSheet(
            "color: blue; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 28px; font-weight: bold;")
        self.roll_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.roll_label.adjustSize()
        self.roll_label.move(0, 300)
        self.roll_label.raise_()

        # ---- Acceleration magnitude (g) overlay ----
        self.g_label = QLabel("g:", self.gfx_canvas)
        self.g_label.setStyleSheet(
            "color: red; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 44px; font-weight: bold;")
        self.g_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.g_label.adjustSize()
        self.g_label.move(self.gfx_canvas.width() - self.g_label.width() - 10, 160)
        self.g_label.raise_()

        # ---- Acceleration magnitude (g) minmax ----
        self.g_label_minmax = QLabel("g:", self.gfx_canvas)
        self.g_label_minmax.setStyleSheet(
            "color: gray; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 14px; font-weight: bold;")
        self.g_label_minmax.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.g_label_minmax.adjustSize()
        self.g_label_minmax.move(self.gfx_canvas.width() - self.g_label.width() - 10, 210)
        self.g_label_minmax.raise_()

        # ---- GPS speed & altitude overlay (top-right) ----
        self.gps_label_speed = QLabel("GPS:", self.gfx_canvas)
        self.gps_label_speed.setStyleSheet(
            "color: red; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 44px; font-weight: bold;")
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
        self.gps_label_alt.setStyleSheet(
            "color: blue; background-color: transparent; padding: 10px; font-family: 'Menlo'; font-size: 44px; font-weight: bold;")
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

        # ---- Energy graph (rolling 30s) ----
        import pyqtgraph as pg
        self.energy_plot = pg.PlotWidget(self.gfx_canvas)
        self.energy_plot.setBackground('transparent')
        self.energy_plot.setAttribute(Qt.WA_TranslucentBackground)
        self.energy_plot.setStyleSheet("background: transparent;")
        self.energy_plot.showGrid(x=True, y=True, alpha=0.3)
        # ensure viewbox is also transparent
        self.energy_plot.getPlotItem().getViewBox().setBackgroundColor(None)
        self.energy_plot.enableAutoRange(axis='y')

        # style clean
        self.energy_plot.getAxis('left').setTextPen('gray')
        self.energy_plot.getAxis('bottom').setTextPen('gray')

        self.energy_curve = self.energy_plot.plot(pen=pg.mkPen(color=(0, 120, 255), width=2))

        # position (bottom-left overlay)
        self.energy_plot.setGeometry(
            0,
            self.gfx_canvas.height() - 160,
            300,
            140
        )
        self.energy_plot.raise_()
        self.energy_plot.show()

        # buffers
        self.energy_time = []
        self.energy_values = []

    def compute_orientation(self):

        # computed transform matrix (use pre-cached numpy arrays — no pandas row access)
        idf = self.idf
        R = quat_to_rot([self.x4_quat_w_vals[idf], self.x4_quat_x_vals[idf],
                         self.x4_quat_y_vals[idf], self.x4_quat_z_vals[idf]])
        # use cached pitch-mounting rotation (recomputed only when angle changes)
        R_x_20 = self.R_x_20_cached
        self.R_final = R_x_20 @ perm[R_recalage_repere] @ R
        if not np.isfinite(self.R_final).all():
            return

        # original vectors
        fwd_original = np.array([0.0, 1.0, 0.0])
        up_original = np.array([0.0, 0.0, 1.0])
        self.down_original = np.array([0.0, 0.0, -1.0])

        # computed vectors
        self.fwd = self.R_final.T @ fwd_original;
        self.up = self.R_final.T @ up_original;  # down = self.R_final.T @ self.down_original
        self.nose_vec = self.fwd * 400
        acc = np.array([-self.x4_acc_x_vals[idf], -self.x4_acc_y_vals[idf], -self.x4_acc_z_vals[idf]])
        norm_acc = np.linalg.norm(acc)
        if not np.isfinite(norm_acc) or norm_acc < 1e-6:
            acc = np.array([0.0, 0.0, 0.0])
        else:
            acc = acc / norm_acc
        self.g = norm_acc / 9.81
        self.acc_vec = R_x_20 @ perm[R_recalage_repere] @ acc
        self.acc_vec = self.acc_vec * 300 + self.acc_vec * 100 * self.g  # scaling up
        # ---- low-pass filter to smooth G vector trail ----
        if self.acc_vec_filtered is None:
            self.acc_vec_filtered = self.acc_vec.copy()
        else:
            a = self.g_filter_alpha
            self.acc_vec_filtered = (1 - a) * self.acc_vec_filtered + a * self.acc_vec

        # compute sign of g
        g_sens = angle_between(self.acc_vec, self.down_original)
        if g_sens > 90:
            self.g = -self.g
        self.g_min = min(self.g_min, self.g)
        self.g_max = max(self.g_max, self.g)

        # ---- Compute Pitch (angle between rotated Y and XY plane) ----
        v_original = np.array([0.0, 1.0, 0.0])
        v_rotated = self.R_final.T @ v_original
        plane_normal = np.array([0.0, 0.0, 1.0])
        dot = np.dot(v_rotated, plane_normal)
        norm_v = np.linalg.norm(v_rotated)
        if norm_v < 1e-6 or not np.isfinite(norm_v):
            pitch_rad = 0.0
        else:
            pitch_rad = np.arcsin(np.clip(dot / norm_v, -1.0, 1.0))
        self.pitch_deg = np.degrees(pitch_rad)

        # ---- Compute Inclinaison (roll) ----
        world_z = np.array([0.0, 0.0, 1.0])
        v = np.cross(self.fwd, np.cross(world_z, self.fwd))
        v = v / np.linalg.norm(v)
        dot = np.dot(self.up, v)
        cross = np.cross(self.up, v)
        inclinaison = np.arctan2(np.dot(cross, self.fwd), dot)
        self.bank_deg = -np.degrees(inclinaison)  # convention aéronautique (droite positive / gauche positive)

        # ---- Compute Inertial Heading ----
        fwd_proj = np.array([self.fwd[0], self.fwd[1], 0.0])
        norm_proj = np.linalg.norm(fwd_proj)
        if norm_proj > 1e-8:
            fwd_proj = fwd_proj / norm_proj
            # angle par rapport à l'axe X
            heading_rad = np.arctan2(fwd_proj[1], fwd_proj[0])
            self.heading_deg = np.degrees(heading_rad)
            if self.heading_deg < 0:
                self.heading_deg += 360
            self.heading_deg = 360 - self.heading_deg
        else:
            self.heading_deg = 0.0

        # compute pitch/roll for triaangle
        R_view_wing = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]])
        R_wing = R_view_wing @ self.R_final
        # forward and up vectors in the wing reference
        fwd_w = R_wing.T @ np.array([0.0, 1.0, 0.0])
        up_w = R_wing.T @ np.array([0.0, 0.0, 1.0])
        self.pitch_w = np.degrees(np.arcsin(np.clip(fwd_w[2], -1.0, 1.0)))
        self.right_w = np.cross(fwd_w, up_w)
        self.roll_w = np.degrees(np.arctan2(self.right_w[2], up_w[2]))

        # compute energy (use numpy caches)
        self.energy = 0.5 * self.gps_speed_vals[idf] ** 2 + 9.81 * self.gps_alt_vals[idf] * 0.3048

    def update_energy_graph(self):
        if hasattr(self, "i") and self.i % 30 != 0:
            return

        # fenêtre = 500 derniers points
        end_idx = self.idf
        start_idx = max(0, end_idx - 5000)

        # slicing direct numpy (très rapide)
        values = self.energy_full[start_idx:end_idx]

        # axe temps relatif
        df_freq = getattr(self, "df_freq", 30.0)
        n = len(values)
        t_rel = np.arange(n) / df_freq

        # assignation
        self.energy_values = values.tolist()
        self.energy_time = t_rel.tolist()
        self.energy_curve.setData(t_rel, self.energy_values)

        if len(self.energy_values) > 0:
            ymin = min(self.energy_values)
            ymax = max(self.energy_values)
            if (ymax - ymin) < 2000:
                center = (ymax + ymin) / 2
                ymin = center - 1000
                ymax = center + 1000
            self.energy_plot.setYRange(ymin, ymax, padding=0)

        # keep plot anchored bottom-right (responsive)
        self.energy_plot.setGeometry(
            0,
            self.gfx_canvas.height() - 100,
            300,
            100)


    def update_gfx_orientation(self):

        self.compute_orientation()

        # display smoothed G vector
        self.acc_geom.positions.data[1] = self.acc_vec_filtered if self.acc_vec_filtered is not None else self.acc_vec
        self.acc_geom.positions.update_range(0, 2)

        # ---- Update G vector trail ----
        if not np.any(self.g_trail):
            self.g_trail[:] = self.acc_vec_filtered
        else:
            self.g_trail[:-1] = self.g_trail[1:]
            self.g_trail[-1] = self.acc_vec_filtered

        self.gfx_g_trail.geometry.positions.data[:] = self.g_trail
        self.gfx_g_trail.geometry.positions.update_range(0, len(self.g_trail))

        # If trail is empty (after seek/reset), initialize it with the current nose position
        if not np.any(self.nose_trail):
            self.nose_trail[:] = self.fwd * 400
        else:
            self.nose_trail[:-1] = self.nose_trail[1:]
            self.nose_trail[-1] = self.fwd * 400

        self.gfx_nose_trail.geometry.positions.data[:] = self.nose_trail
        self.gfx_nose_trail.geometry.positions.update_range(0, len(self.nose_trail))

        # ---- Update arrow head position and orientation (use smoothed vector) ----
        vec_arrow = self.acc_vec_filtered if self.acc_vec_filtered is not None else self.acc_vec
        direction = vec_arrow / (np.linalg.norm(vec_arrow) + 1e-9)
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
                np.cos(angle / 2.0),)
        self.gfx_acc_arrow.local.position = tuple(vec_arrow)
        self.gfx_acc_arrow.local.rotation = quat

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
        self.g_label.move(self.gfx_canvas.width() - self.g_label.width(), 140)
        self.g_label_minmax.adjustSize()
        self.g_label_minmax.move(self.gfx_canvas.width() - self.g_label_minmax.width(), 190)

        # update acceleration vector geometry & color
        if self.g > 0.8:
            err = min(abs(self.g - 1.0), 1.0)
            err2G = max(self.g - 2.0, 0)
            r = err;
            g = 1.0 - err;
            b = err2G / 2
        else:
            r = 0;
            g = 0;
            b = 1

        self.gfx_vec_acc.material.color = (r, g, b, 1)
        self.gfx_acc_arrow.material.color = (r, g, b, 1)
        new_g_color = (int(r * 255), int(g * 255), int(b * 255))
        if new_g_color != self._last_g_color:
            self.g_label.setStyleSheet(
                f"color: rgb{new_g_color}; "
                "background-color: transparent; padding: 10px; "
                "font-family: 'Menlo'; font-size: 44px; font-weight: bold;")
            self._last_g_color = new_g_color

        # rotation de l'objet
        M = np.eye(4)
        M[:3, :3] = self.R_final.T

        # ---- ORTHONORMALIZE ROTATION (CRITICAL FIX) ----
        try:
            R = M[:3, :3]

            U, _, Vt = np.linalg.svd(R)
            R_clean = U @ Vt

            M[:3, :3] = R_clean

        except Exception as e:
            print("⚠️ SVD cleanup failed:", e)
            return

        if np.isfinite(M).all():
            self.gfx_object.local.matrix = M

        # ---- Update Artificial Horizon ----
        if hasattr(self, "hud_horizon"):
            self.hud_horizon.pitch = self.pitch_deg
            self.hud_horizon.bank = self.bank_deg
            self.hud_horizon.heading = self.heading_deg
            self.hud_horizon.update()

        # ---- Update Wingtip Artificial Horizon ----
        if hasattr(self, "hud_horizon_wing"):
            # invert pitch sign for wingtip reference (viewed from the side)
            self.hud_horizon_wing.pitch = -self.pitch_w
            self.hud_horizon_wing.bank = self.roll_w
            self.hud_horizon_wing.heading = self.heading_deg
            self.hud_horizon_wing.update()

        # ---- Update dataframe info label ----
        # compute elapsed time since start of dataset (use numpy caches)
        t0 = self.t0_timestamp
        t_now = self.timestamp_vals[self.idf]
        elapsed = t_now - t0
        elapsed_s = int(elapsed.total_seconds())
        em = elapsed_s // 60
        es = elapsed_s % 60

        # --- CPU sampling (1 Hz) ---
        if not hasattr(self, "_last_cpu_update"):
            self._last_cpu_update = 0
            self._cpu_percent = 0
        now_cpu = time.time()
        if now_cpu - self._last_cpu_update >= 1.0:
            try:
                self._cpu_percent = psutil.cpu_percent(interval=None)
            except Exception:
                self._cpu_percent = 0
            self._last_cpu_update = now_cpu

        self.df_info_label.setText(
            f"Frame: {self.i}"
            f"\nStutters: {self.stutter_count}"
            f"\nCPU: {self._cpu_percent:.0f}%"
            # f"\nTime: {t_now.strftime('%H:%M:%S.%f')[:-3]}"
            # f"\nElapsed: {em:02d}:{es:02d}"
            # f"\nFrames skipped: {self.frame_skipped_count} / {self.frame_last_delay:+04d}ms"
        )
        self.df_info_label.adjustSize()
        self.df_info_label.move(
            self.gfx_canvas.width() - self.df_info_label.width() - 10,
            self.gfx_canvas.height() - self.df_info_label.height() - 10)

        self.pitch_label.setText(f"Pitch {self.pitch_deg:.0f}°")
        self.pitch_label.adjustSize()
        # position top-right of artificial horizon (responsive)
        try:
            hx = self.hud_horizon.x() + self.hud_horizon.width() + 10
            hy = self.hud_horizon.y()
        except Exception:
            hx, hy = 180, 10
        self.pitch_label.move(hx, hy)

        self.roll_label.setText(f"Bank {self.bank_deg:.0f}°")
        self.roll_label.adjustSize()
        # place just below pitch label (aligned with pitch)
        try:
            hx = self.pitch_label.x()
            hy = self.pitch_label.y() + self.pitch_label.height() + 5
        except Exception:
            hx, hy = 180, 50
        self.roll_label.move(hx, hy)

        # ---- Update GPS speed / altitude overlay ----
        # use pre-cached numpy arrays for hot-path row access
        gps_speed = self.gps_speed_vals[self.idf]
        gps_alt = self.gps_alt_vals[self.idf]
        gps_fpm = self.gps_fpm_vals[self.idf]

        self.gps_label_speed.setText(f"GS {gps_speed:.0f} km/h")
        # update GS max
        if gps_speed > self.gs_max:
            self.gs_max = gps_speed
        self.gps_label_speed.adjustSize()
        self.gps_label_speed.move(self.gfx_canvas.width() - self.gps_label_speed.width() - 10, 0)

        # update GSmax label
        self.gsmax_label.setText(f"GSmax {self.gs_max:.0f}")
        self.gsmax_label.adjustSize()
        self.gsmax_label.move(self.gfx_canvas.width() - self.gsmax_label.width() - 10, 45)

        # update speed vector geometry & color
        if gps_speed < 113:
            r, g, b = 0, 0, 255
        elif gps_speed < 236:
            r, g, b = 0, 255, 0
        elif gps_speed < 300:
            t = int((gps_speed - 235) / (300 - 235) * 255)
            r, g, b = t, 255 - t, 0
        else:
            r, g, b = 255, 0, 0

        # apply same color to velocity vector (gfx_vec_y)
        self.gfx_vec_y.material.color = (r / 255.0, g / 255.0, b / 255.0, 1.0)
        self.gfx_y_arrow.material.color = (r / 255.0, g / 255.0, b / 255.0, 1.0)

        new_speed_color = (r, g, b)
        if new_speed_color != self._last_speed_color:
            self.gps_label_speed.setStyleSheet(
                f"color: rgb{new_speed_color}; "
                "background-color: transparent; padding: 10px; "
                "font-family: 'Menlo'; font-size: 44px; font-weight: bold;")
            self._last_speed_color = new_speed_color

        self.gps_label_alt.setText(f"Alt {gps_alt:.0f} ft")
        self.gps_label_alt.adjustSize()
        self.gps_label_alt.move(
            self.gfx_canvas.width() - self.gps_label_alt.width(), 60)

        self.gps_label_vario.setText(f"{gps_fpm:.0f} ft/min")
        self.gps_label_vario.adjustSize()
        self.gps_label_vario.move(
            self.gfx_canvas.width() - self.gps_label_vario.width(), 100)

        self.update_energy_graph()

    def update_video_label(self):
        row = self.row
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
            self.video1_pitch_label.setText(f"Pitch {self.pitch_deg:.0f}°")
            self.video1_pitch_label.adjustSize()
            x_pitch = int((self.video1.width() - self.video1_pitch_label.width()) / 2)
            y_pitch = self.video1.height() - self.video1_pitch_label.height() - 45
            self.video1_pitch_label.move(x_pitch, y_pitch)

        # ---- Update bank overlay below pitch ----
        if hasattr(self, "video1_bank_label"):
            self.video1_bank_label.setText(f"Bank {self.bank_deg:.0f}°")
            self.video1_bank_label.adjustSize()
            x_bank = int((self.video1.width() - self.video1_bank_label.width()) / 2)
            y_bank = y_pitch + self.video1_pitch_label.height() + 5
            self.video1_bank_label.move(x_bank, y_bank)

        # ---- Update badin (GPS speed) overlay on video1 (bottom-left) ----
        if hasattr(self, "video1_speed_label"):
            self.video1_speed_label.setText(f"IAS {row.gps_ias:.0f} km/h\nGS {row.gps_speed:.0f} km/h")
            self.video1_speed_label.adjustSize()
            x_speed = 10
            y_speed = self.video1.height() - self.video1_speed_label.height() - 10
            self.video1_speed_label.move(x_speed, y_speed)

        self.video1_wind_label.adjustSize()

        x = self.video1_speed_label.x() + self.video1_speed_label.width() + 10
        y = self.video1.height() - self.video1_wind_label.height() - 10

        self.video1_wind_label.move(x, y)

        try:
            hw = self.headwind_vals[self.idf]
            cw = self.crosswind_vals[self.idf]

            # flèche headwind / tailwind
            if hw >= 0:
                hw_arrow = "↓"  # vent de face
            else:
                hw_arrow = "↑"  # vent arrière

            hw_txt = f"{hw_arrow} {abs(hw):.0f} kt"

            # flèche crosswind
            if cw > 0:
                cw_arrow = "←"  # vent venant de la droite → pousse vers la gauche
            else:
                cw_arrow = "→"  # vent venant de la gauche → pousse vers la droite

            cw_txt = f"{cw_arrow} {abs(cw):.0f} kt"

            self.video1_wind_label.setText(f"{hw_txt}\n{cw_txt}")

        except Exception:
            pass

        # ---- Update analog badin with smoothing ----
        if self.smooth_speed is None:
            self.smooth_speed = row.gps_ias
        else:
            a = self.instrument_alpha
            self.smooth_speed = (1 - a) * self.smooth_speed + a * row.gps_ias

        if hasattr(self, "video1_badin"):
            self.video1_badin.speed = self.smooth_speed
            self.video1_badin.update()
            xb = int(self.video1.width() / 2 - self.video1_badin.width()) - 60
            yb = int(self.video1.height() - self.video1_badin.height())
            self.video1_badin.move(xb, yb)

        # ---- Update analog altimeter with smoothing ----
        if self.smooth_alt is None:
            self.smooth_alt = row.gps_alt
        else:
            a = self.instrument_alpha
            self.smooth_alt = (1 - a) * self.smooth_alt + a * row.gps_alt

        if hasattr(self, "video1_altimeter"):
            self.video1_altimeter.alt = self.smooth_alt
            self.video1_altimeter.update()
            xa = int(self.video1.width() / 2) + 60
            ya = int(self.video1.height() - self.video1_altimeter.height())
            self.video1_altimeter.move(xa, ya)

        # ---- Update analog variometer ----
        if hasattr(self, "video1_vario"):
            if not hasattr(self, "smooth_fpm"):
                self.smooth_fpm = row.gps_fpm
            else:
                a = self.instrument_alpha
                self.smooth_fpm = (1 - a) * self.smooth_fpm + a * row.gps_fpm
            self.video1_vario.fpm = self.smooth_fpm
            self.video1_vario.update()
            xv = int(self.video1.width() / 2) + 60 + self.video1_altimeter.width()
            yv = int(self.video1.height() - self.video1_vario.height())
            self.video1_vario.move(xv, yv)
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

        if hasattr(self, "video2_date_label"):
            self.video2_date_label.move(
                self.video2.width() - self.video2_date_label.width() - 10,
                self.video2.height() - self.video2_date_label.height() - 10
            )

        self.video1.pitch_w = -self.pitch_w
        self.video1.roll_w = self.roll_w
        self.video1.heading = self.heading_deg

    def calibrate_gfx(self, where):
        # average accelerometer over 100 samples to reduce IMU noise
        start = max(0, where - 50)
        end = min(self.frames_df, where + 50)

        acc = self.df.iloc[start:end][["x4_acc_x", "x4_acc_y", "x4_acc_z"]].to_numpy()
        grav = np.mean(acc, axis=0)
        grav = grav / np.linalg.norm(grav)
        self.montage_pitch_angle = math.degrees(math.acos(grav[1])) - OFFSET_PITCH_SOL_PALLIER
        self._rebuild_R_x_20()
        print("Angle de montage : ", round(self.montage_pitch_angle, 1))

        # update menu display of camera pitch
        try:
            self.update_pitch_cam_menu()
        except Exception:
            pass

        # refresh graphics immediately (useful when paused)
        try:
            self.update_gfx_orientation()
            self.update_video_label()
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
        current_time = self.row.timestamp
        metar_row = self.find_metar_for_time(current_time)
        new_metar = metar_row.metar
        if new_metar != self.last_metar:
            self.last_metar = new_metar
            if hasattr(self, "map_metar_label"):
                self.map_metar_label.setText(self.last_metar)
                self.map_metar_label.adjustSize()
                self._position_map_metar_label()

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
                margin = 0.03  # 50 ms tolerance
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
        self._rebuild_bookmark_cache()
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
        self._rebuild_bookmark_cache()
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
        #if self.i < 5:
        #    self.i = int(frame)

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

        # ---- Rebuild OSM trajectory after seek (use numpy caches, no pandas) ----
        if self.map_ready:
            window = 5000
            start = max(0, self.idf - window)

            try:
                import json

                # downsample (1 point every 30)
                lat = self.gps_lat_vals[start:self.idf:30]
                lon = self.gps_lon_vals[start:self.idf:30]

                # stack into [[lat, lon], ...]
                points = np.column_stack((lat, lon)).tolist()

                js = f"resetTrajectoryWithData({json.dumps(points)})"

                if hasattr(self, "map_view"):
                    self.map_view.page().runJavaScript(js)

            except Exception as e:
                print("Trajectory rebuild (numpy) failed:", e)


        fps = float(self.stream1.average_rate)
        ts = int((frame / fps) / float(self.stream1.time_base))
        self.container1.seek(ts, stream=self.stream1)
        self.container2.seek(ts, stream=self.stream2)
        self.decoder1 = self.container1.decode(self.stream1)
        self.decoder2 = self.container2.decode(self.stream2)

        # ---- reset audio when seeking ----
        if self.audio_stream is not None:
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
            if self.playing:
                self.audio_output.resume()
            else:
                self.audio_output.suspend()

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
        frame = min(self.frames_video - 1, int(self.i + fps * 2))
        self.seek_video(frame)
        self.slider.setValue(self.i)

    def jump_fwd_10s(self):
        fps = float(self.stream1.average_rate) or 30
        frame = min(self.frames_video - 1, int(self.i + fps * 10))
        self.seek_video(frame)
        self.slider.setValue(self.i)

    def goto_mise_en_ligne(self):
        self.seek_video(self.get_video_frame_from_df_index(self.index_enligne_devol))
        self.slider.setValue(self.i)

    def seek_palier(self):
        window = int(DF_FREQ * 2)  # 2 secondes
        fpm = self.df["gps_fpm"].to_numpy()
        speed = self.df["gps_speed"].to_numpy()
        # start searching AFTER the current dataframe position (next palier behavior)
        start_idx = getattr(self, "idf", 0) + 1000
        for i in range(start_idx, self.frames_df - window):
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
                bytes_free = self.audio_output.bytesFree()
            else:
                bytes_free = 16384
            # 🔑 on remplit un peu plus que nécessaire
            target_buffer = max(bytes_free * 2, 32768)
            target_buffer = min(target_buffer, 65536)  # hard cap to avoid latency drift
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
                            bytes_per_sample = 2  # assuming s16

                            # fallback safe values if attributes not initialized
                            sample_rate = getattr(self, "audio_sample_rate", None)
                            channels = getattr(self, "audio_channels", None)

                            if sample_rate is None:
                                try:
                                    sample_rate = f.sample_rate
                                    self.audio_sample_rate = sample_rate
                                except Exception:
                                    sample_rate = 48000  # safe default

                            if channels is None:
                                try:
                                    channels = len(f.layout.channels)
                                    self.audio_channels = channels
                                except Exception:
                                    channels = 2  # safe default

                            buffer_duration = len(self.audio_buffer) / (
                                    sample_rate * channels * bytes_per_sample
                            )

                            self.audio_clock_sec = float(f.pts * f.time_base) - buffer_duration
                        self.audio_buffer += f.to_ndarray().tobytes()
        except StopIteration:
            return
        except Exception as e:
            print("audio error:", e)
            return
        # 🔊 écriture CONTINUE (critique)
        chunk_size = 4096  # lower latency, better sync
        if not hasattr(self, "audio_output"):
            return
        bytes_free = self.audio_output.bytesFree()

        if len(self.audio_buffer) < chunk_size:
            self.stutter_count += 1  # underflow risk

        # 🔑 on vide autant que possible (et pas 1 seul chunk)
        while len(self.audio_buffer) >= chunk_size:
            written = self.audio_device.write(self.audio_buffer[:chunk_size])
            if written <= 0:
                break
            self.audio_buffer = self.audio_buffer[written:]

    # 🔑 SYNCHRO VIDEO ← DF
    # ==================================================
    def get_video_frame_from_df_index(self, df_index):
        if df_index < 0 or df_index >= self.frames_df:
            raise ValueError("df_index hors limites")
        # timestamp dataframe
        ts_df = self.df.timestamp.iloc[df_index]
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
        frame_index = max(0, min(frame_index, self.frames_video - 1))
        return frame_index

    def update_wind(self):
        ws = self.gps_wind_speed_vals[self.idf] / 1.852
        wd = self.gps_wind_direction_vals[self.idf]

        self.map_wind_label.setText(f"{ws:.0f} kt\n{wd:.0f}°")
        self.map_wind_label.adjustSize()

        # rotation de la flèche (direction du vent)
        if hasattr(self, "wind_arrow_base"):
            transform = QTransform().rotate(wd)
            rotated = self.wind_arrow_base.transformed(transform, Qt.SmoothTransformation)
            self.map_wind_arrow.setPixmap(rotated)
            self.map_wind_arrow.adjustSize()

        self._position_map_wind_label()


    # ==================================================
    def update_all(self):
        self.update_video(self.decoder1, self.video1, self.video1_start, self.stream1)
        self.update_video(self.decoder2, self.video2, self.video2_start, self.stream2)
        self.i += 1
        # ---- Auto stop recording at end of playback ----
        if self.i >= self.frames_video - 1 and self.recording:
            self.toggle_recording(False)
            self.recording = False

        if self.current_video_time_utc is not None:
            self.sync_dataframe_on_video()
        self.update_gps_pyqtgraph()
        self.update_metar()
        self.update_wind()
        self.update_gfx_orientation()
        self.update_video_label()
        # ---- Update video2 date/time overlay (synced to playback) ----
        if hasattr(self, "video2_date_label") and self.current_video_time_utc is not None:
            ts = self.current_video_time_utc
            try:
                # convert UTC -> local (Europe/Paris)
                if ts.tzinfo is None:
                    import datetime, zoneinfo
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
                    ts = ts.astimezone(zoneinfo.ZoneInfo("Europe/Paris"))
                else:
                    ts = ts.astimezone()
            except Exception:
                pass

            mois_fr = [
                "janvier", "février", "mars", "avril", "mai", "juin",
                "juillet", "août", "septembre", "octobre", "novembre", "décembre"
            ]

            text = f"{ts.day} {mois_fr[ts.month - 1]} {ts.year} {ts.strftime('%H:%M:%S')}"

            self.video2_date_label.setText(text)
            self.video2_date_label.adjustSize()
        self.update_g_timeline_cursor()

        if len(self._bm_frames_cache) > 0:
            fps = int(self.fps_video)
            for j in range(len(self._bm_frames_cache)):
                frame = self._bm_frames_cache[j]
                if self.i == frame - fps:
                    if frame != self.last_bookmark_frame:
                        self.show_bookmark_overlay(str(self._bm_names_cache[j]))
                        self.last_bookmark_frame = frame

        # ---- Elapsed time overlay update ----
        if self.current_video_time_utc is not None:
            elapsed = self.current_video_time_utc - self.t0_timestamp
            total_sec = int(elapsed.total_seconds())
            h = total_sec // 3600
            m = (total_sec % 3600) // 60
            s = total_sec % 60
            if h > 0:
                txt = f"{h:02d}:{m:02d}:{s:02d}"
            else:
                txt = f"{m:02d}:{s:02d}"
            self.elapsed_time_overlay.setText(txt)




    # ==================================================
    def sync_video_to_audio(self, decoder, frame, stream):
        # no audio → no sync
        if not hasattr(self, "audio_clock_sec") or self.audio_clock_sec <= 0:
            return frame

        try:
            fps = float(stream.average_rate)
        except Exception:
            fps = 30.0

        # current video time
        frame_time = frame.pts * float(stream.time_base)

        # audio is master
        delay = self.audio_clock_sec - frame_time

        # only act if more than 1 frame late
        if delay > (1.0 / fps):
            frames_to_skip = int(delay * fps)

            # clamp to avoid brutal jumps
            frames_to_skip = min(frames_to_skip, 10)
            for _ in range(frames_to_skip):
                try:
                    frame = next(decoder)
                except StopIteration:
                    break

        return frame

    def update_video(self, decoder, label, start_dt, stream):
        ret, frame, avframe = self.read_video_frame(decoder)
        # ---- PRO sync: align video to audio ----
        frame = self.sync_video_to_audio(decoder, frame, stream)
        avframe = frame
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
        # legacy sync removed (replaced by direct frame skip logic)

        # Use VideoYUVWidget if available
        if hasattr(label, "setFrame"):
            label.setFrame(frame)
            return
        else:
            print("⚠️ QLabel fallback utilisé")

        # fallback (old QLabel path)
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
        idx = self.df["timestamp"].searchsorted(ts)

        if idx <= 0:
            self.idf = 0
        elif idx >= self.frames_df:
            self.idf = self.frames_df - 1
        else:
            before = self.df.timestamp.iloc[idx - 1]
            after = self.df.timestamp.iloc[idx]
            self.idf = idx - 1 if abs(ts - before) <= abs(after - ts) else idx

        # cache dataframe row once per frame
        self.row = self.df.iloc[self.idf]
        self.slider.blockSignals(True)
        self.slider.setValue(self.i)
        self.slider.blockSignals(False)
        # self.timestamp_label.setText(f"Video time : {ts.strftime('%Y-%m-%d %H:%M:%S')}")

        # ---- Elapsed time overlay update (compute txt here) ----
        if self.current_video_time_utc is not None:
            elapsed = self.current_video_time_utc - self.t0_timestamp
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

        # ---- Previous bookmark overlay update ----
        if self.current_video_time_utc is not None and self.bookmarks_df is not None:
            # t = self.current_video_time_utc
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

    # ==================================================
    # 🔑 SYNCHRO VIDEO ← DF
    # ==================================================
    def compute_video_frame_from_df_index(self, df_index):
        """
        Calcule l'index frame vidéo à partir d'un index du dataframe.
        Retourne l'index de frame correspondant dans la vidéo 1.
        """
        if df_index < 0 or df_index >= self.frames_df:
            raise ValueError("df_index hors limites")
        # timestamp dataframe
        ts_df = self.df.timestamp.iloc[df_index]
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

    def update_g_timeline_cursor(self):
        # ---- compute cursor position (with optional zoom) ----
        if getattr(self, "timeline_zoom", False):
            start = getattr(self, "timeline_start", 0)
            end = getattr(self, "timeline_end", self.frames_df - 1)

            # clamp idf inside zoom window
            idf_clamped = max(start, min(self.idf, end))

            span = max(1, end - start)
            rel = (idf_clamped - start) / span
        else:
            rel = self.idf / self.frames_df

        x_local = int(rel * self.g_timeline.width())

        # compute global position
        x_global = self.g_timeline.x() + x_local

        # compute vertical span across all timelines (G + Alt + Vario)
        timelines = [self.g_timeline]
        if hasattr(self, "alt_timeline"):
            timelines.append(self.alt_timeline)
        if hasattr(self, "fpm_timeline"):
            timelines.append(self.fpm_timeline)

        y_top = min(t.y() for t in timelines)
        y_bottom = max(t.y() + t.height() for t in timelines)
        total_height = y_bottom - y_top

        # create cursor once (IMPORTANT: parent = self, not g_timeline)
        if not hasattr(self, "g_timeline_cursor"):
            self.g_timeline_cursor = QFrame(self)
            self.g_timeline_cursor.setStyleSheet("background-color: black;")
            self.g_timeline_cursor.setGeometry(0, 0, 2, total_height)
            self.g_timeline_cursor.show()

        # ---- triangle indicator (top of cursor) ----
        if not hasattr(self, "g_timeline_cursor_triangle"):
            self.g_timeline_cursor_triangle = QLabel("▼", self)
            self.g_timeline_cursor_triangle.setStyleSheet(
                "color: black; background: transparent; font-size: 12px;"
            )
            self.g_timeline_cursor_triangle.adjustSize()
            self.g_timeline_cursor_triangle.show()

        # update geometry
        self.g_timeline_cursor.setGeometry(x_global, y_top, 2, total_height)
        self.g_timeline_cursor.raise_()

        # position triangle at top of cursor
        if hasattr(self, "g_timeline_cursor_triangle"):
            self.g_timeline_cursor_triangle.move(
                x_global - self.g_timeline_cursor_triangle.width() // 2 + 1,
                y_top - self.g_timeline_cursor_triangle.height()
            )
            self.g_timeline_cursor_triangle.raise_()

    # ==================================================
    def on_slider(self, value):
        if self.map_ready:
            self.map_view.page().runJavaScript("resetTrajectory();")
        # move both videos to the requested frame
        self.seek_video(value)
        # force an immediate refresh when paused
        if not self.playing:
            self.update_all()

    def update_gps_pyqtgraph(self):
        # skip updates only during playback
        if self.playing and self.i % 8 != 0:
            return
        row = self.row
        if self.map_ready:
            lat = row.gps_lat
            lon = row.gps_lon
            heading = row.gps_heading - 45
            self.map_view.page().runJavaScript(
                f"window.updateMarker({lat}, {lon}, {heading});"
            )

        # keep METAR overlay visible and correctly positioned
        self.map_metar_label.setText(getattr(self, "last_metar", ""))
        self._position_map_metar_label()

        end = self.idf
        start = end - TRACE
        if start < 0:
            start = 0

        # center trajectory on current aircraft position
        lon = self.gps_lon_vals[start:end]
        lat = self.gps_lat_vals[start:end]
        alt = self.gps_alt_vals[start:end]

        lon0 = self.gps_lon_vals[end]
        lat0 = self.gps_lat_vals[end]
        # alt0 = self.gps_alt_vals[end]

        # convert degrees to approximate meters
        x = (lon - lon0) * 111320 * np.cos(np.radians(lat0)) / 1000
        y = (lat - lat0) * 111320 / 1000

        if len(alt) > 0 and alt[-1] > 3000:  # décalage de 3000 si dans le box
            z = (alt - 3000 - 1000) / 1000
        else:
            z = (alt - 1000) / 1000
        # debug: dernière position calculée
        if len(x) > 0:
            # ---- vertical projection to ground ----
            p_air = np.array([x[-1], y[-1], z[-1]])
            p_ground = np.array([x[-1], y[-1], -1.0])
            self.gps_vertical_line.setData(pos=np.vstack([p_air, p_ground]))

        pts = np.column_stack([x, y, z])

        az = -int(row.gps_heading / 45) * 45 - 22.5
        yz = -1
        xz = -1
        if az != self.last_azim:
            self.last_azim = az
            self.gps_view.setCameraPosition(azimuth=az)
            if az == -67.5 or az == -22.5 or az == -112.5 or az == -337.5:
                yz = 1
            self.grid_vertical_yz.resetTransform()
            self.grid_vertical_yz.rotate(90, 1, 0, 0)
            self.grid_vertical_yz.translate(0, yz, 0)
            self.grid_vertical_yz.rotate(BOX_HEADING, 0, 0, 1)

            if az == -202.5 or az == -112.5 or az == -157.5 or az == -67.5:
                xz = 1
            self.grid_vertical_xz.resetTransform()
            self.grid_vertical_xz.rotate(90, 0, 1, 0)
            self.grid_vertical_xz.translate(xz, 0, 0)
            self.grid_vertical_xz.rotate(BOX_HEADING, 0, 0, 1)
            # self.grid_vertical_xz.setVisible(False)

        # ---- update ground projection (shadow) ----
        if len(pts) > 1:
            # ---- projection au sol (plan XY, Z constant) ----
            pts_ground = pts.copy()
            pts_ground[:, 2] = -1.0
            self.gps_shadow.setData(pos=pts_ground)

            # ---- projection sur un plan vertical incliné de 50° par rapport à XZ ----
            pts_box = pts.copy()
            # 1) rotation inverse pour aligner le plan avec XZ
            pts_box = pts_box @ ROT_BOX  # 1) rotation inverse pour aligner le plan avec XZ
            px = -1
            if az == -112.5 or az == -337.5 or az == -22.5 or az == -67.5:
                px = 1
            # 2) projection sur XZ
            pts_box[:, 1] = px
            # 3) retour dans le repère initial
            pts_box = pts_box @ ROT_BOX.T
            self.gps_box_projection.setData(pos=pts_box)

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
            colors[:, 0] = zn  # red increases with altitude
            colors[:, 1] = 1.0 - np.abs(zn - 0.5) * 2  # green strongest mid-altitude
            colors[:, 2] = 1.0 - zn  # blue decreases with altitude
            colors[:, 3] = 1.0  # alpha

            # update the bundle of lines to simulate a tube
            for line in self.gps_lines:
                off = line._tube_offset
                pts_off = pts + off
                line.setData(pos=pts_off, color=colors)

        if end < len(self.gps_lat_vals):
            # aircraft stays at center

            # ---- rotation using Euler angles (heading, pitch, roll) ----
            heading = float(self.row.gps_heading)
            pitch = float(getattr(self, "pitch_deg", 0.0))
            roll = float(getattr(self, "bank_deg", 0.0))

            self.gps_aircraft.resetTransform()

            # rotations
            self.gps_aircraft.rotate(-roll, 1, 0, 0)  # FIX
            self.gps_aircraft.rotate(pitch, 0, 1, 0)
            self.gps_aircraft.rotate(- heading - 90, 0, 0, 1)
            # ---- safety: avoid empty data when timeline zoom is active ----
            if z is None or len(z) == 0:
                self.gps_aircraft.translate(0, 0, 0)
            else:
                self.gps_aircraft.translate(0, 0, z[-1])

        # ---- update altitude labels for pyqtgraph GPS view ----
        self.update_altitude_labels()

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

        if alt[-1] > 5500:
            zoom = 8
        elif alt[-1] < 3000:
            zoom = 6
        else:
            zoom = 4

        if zoom != self.gps_lastzoom:
            self.gps_lastzoom = zoom
            self.gps_view.setCameraPosition(distance=zoom)

    def detach_gfx_window(self):
        """Toggle detach/close for pygfx canvas."""
        if getattr(self, "gfx_detached", False):
            # close popup
            if hasattr(self, "gfx_window"):
                self.gfx_window.close()
            return

        self.gfx_detached = True

        # remove from layout
        self.gfx_render_enabled = False
        self.grid.removeWidget(self.gfx_canvas)

        # create new window
        self.gfx_window = QMainWindow(self)
        self.gfx_window.setWindowTitle("3D View")
        self.gfx_window.setCentralWidget(self.gfx_canvas)
        # hide overlay button while detached window is open
        if hasattr(self, "btn_detach_gfx"):
            self.btn_detach_gfx.hide()
        # match main window size
        self.gfx_window.resize(self.size())

        # self.btn_detach_gfx.setText("Close 3D")

        # detect close event
        self.gfx_window.closeEvent = self._on_gfx_window_closed

        self.gfx_window.show()
        self.gfx_render_enabled = True

    def _on_gfx_window_closed(self, event):
        """Restore pygfx canvas back into main layout when detached window closes."""
        self.gfx_render_enabled = False
        self.gfx_canvas.setParent(self)
        self.gfx_canvas.hide()
        self.grid.addWidget(self.gfx_canvas, 1, 0, 1, 2)
        self.gfx_canvas.show()
        self.gfx_render_enabled = True
        self.gfx_detached = False
        # restore overlay button when returning to main window
        if hasattr(self, "btn_detach_gfx"):
            self.btn_detach_gfx.show()
            self.btn_detach_gfx.raise_()
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
        # match main window size
        self.video1_window.resize(self.size())
        self.btn_detach_video1.hide()
        self.video1_window.closeEvent = self._on_video1_window_closed
        self.video1_window.show()

    def _on_video1_window_closed(self, event):
        self.video1.setParent(None)
        self.grid.addWidget(self.video1, 0, 0, 1, 2)
        self.video1_detached = False
        self.btn_detach_video1.show()
        self.btn_detach_video1.raise_()
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
        # match main window size
        self.video2_window.resize(self.size())
        self.btn_detach_video2.hide()
        self.video2_window.closeEvent = self._on_video2_window_closed
        self.video2_window.show()

    def _on_video2_window_closed(self, event):
        self.video2.setParent(None)
        self.grid.addWidget(self.video2, 0, 2, 1, 2)
        self.video2_detached = False
        self.btn_detach_video2.show()
        self.btn_detach_video2.raise_()
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
        # match main window size
        self.pyqtgraph_window.resize(self.size())
        self.btn_detach_pyqtgraph.hide()
        # detect close
        self.pyqtgraph_window.closeEvent = self._on_pyqtgraph_window_closed
        self.pyqtgraph_window.show()

    def _on_pyqtgraph_window_closed(self, event):
        self.gps_view.setParent(None)
        self.grid.addWidget(self.gps_view, 1, 2, 1, 1)
        self.pyqtgraph_detached = False
        self.btn_detach_pyqtgraph.show()
        self.btn_detach_pyqtgraph.raise_()
        event.accept()

    def on_gfx_destroyed(self):
        print("💀 gfx_canvas destroyed")
        if hasattr(self, "gfx_display") and hasattr(self.gfx_display, "canvas"):
            self.gfx_display.canvas = None

STYLE_SHEET = """
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

    /* ---- Force dialogs to light theme even in dark mode ---- */
    QDialog {
        background-color: white;
        color: black;
    }

    /* ---- Fix QFileDialog headers / title bars in dark mode ---- */
    QFileDialog {
        background-color: white;
        color: black;
    }

    QFileDialog QLabel {
        color: black;
    }

    QFileDialog QHeaderView {
        background-color: white;
        color: black;
    }

    QHeaderView::section {
        background-color: #f0f0f0;
        color: black;
        border: 1px solid #ccc;
        padding: 4px;
    }

    QFileDialog QTreeView::item:selected,
    QFileDialog QListView::item:selected {
        background-color: #cce5ff;
        color: black;
    }

    QLabel {
        color: black;
    }

    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: white;
        color: black;
        border: 1px solid #888;
    }

    QComboBox {
        background-color: white;
        color: black;
        border: 1px solid #888;
    }

    QListView, QTreeView {
        background-color: white;
        color: black;
    }

    """


def select_abv_bundle(parent=None):
    dialog = QFileDialog(parent)
    dialog.setWindowTitle("ABView - Select Flight")
    # set initial directory to data folder
    base_path = os.path.join(MAINDIR, "data")
    if os.path.isdir(base_path):
        dialog.setDirectory(base_path)

    # Force Qt dialog (important on macOS)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)

    # Treat .abv as directories to select, not open
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)

    # Filter display to .abv only
    dialog.setNameFilter("ABV bundle (*.abv)")

    # Prevent navigation into folder on double-click
    from PyQt5.QtWidgets import QTreeView, QListView

    def accept_on_double_click(index):
        path = dialog.directory().absoluteFilePath(index.data())

        # if it's an .abv bundle → select and close
        if path.endswith(".abv"):
            dialog.selectFile(path)
            dialog.accept()
        else:
            # otherwise behave like normal navigation
            if os.path.isdir(path):
                dialog.setDirectory(path)

    tree = dialog.findChild(QTreeView)
    if tree:
        tree.doubleClicked.connect(accept_on_double_click)

    listview = dialog.findChild(QListView)
    if listview:
        listview.doubleClicked.connect(accept_on_double_click)

    dialog.show()
    result = dialog.exec_()
    if result:
        files = dialog.selectedFiles()
        if files:
            return files[0] + "/"

    return None


# ======================================================
if __name__ == "__main__":
    # global PDL, MERGED_DATA, INPUT_METAR, VIDEO1, VIDEO2, BOOKMARK_FILE, caffeinate
    app = QApplication(sys.argv)
    palette = app.palette()
    palette.setColor(palette.Window, Qt.white)
    palette.setColor(palette.Base, Qt.white)
    palette.setColor(palette.AlternateBase, Qt.white)
    palette.setColor(palette.Text, Qt.black)
    palette.setColor(palette.WindowText, Qt.black)
    app.setPalette(palette)
    app.setStyleSheet(STYLE_SHEET)

    if not SKIP_BDL_SELECTION:
        selected = select_abv_bundle()
        if selected:
            PDL = selected
            print("PDL sélectionné :", PDL)
        else:
            print("Aucun dossier sélectionné, utilisation valeur par défaut :", PDL)

    MERGED_DATA = PDL + "merged_data.csv"
    INPUT_METAR = PDL + "metar.csv"
    VIDEO1 = PDL + "front.mp4"
    VIDEO2 = PDL + "back.mp4"
    BOOKMARK_FILE = PDL + "bookmark.csv"

    import subprocess

    caffeinate = subprocess.Popen(["caffeinate"])
    win = MainWindow()
    win.show()


    # QTimer.singleShot(0, start_app)

    def cleanup():
        print("🧹 CLEANUP START")
        try:
            if hasattr(win, "gfx_display"):
                win.gfx_display.canvas = None
        except Exception as e:
            print("cleanup gfx error:", e)

        win.playing = False

        print("🧹 CLEANUP DONE")
        caffeinate.terminate()


    app.aboutToQuit.connect(cleanup)

    app.exec_()
