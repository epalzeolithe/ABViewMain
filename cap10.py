import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl


# ======================================================
# CAP10 MODEL
# ======================================================
def create_cap10_item():
    scale = 0.3

    pts = np.array([
        [ 1.0,  0.0,  0.0],   # nez
        [-1.0,  0.0,  0.0],   # queue

        [ 0.0,  1.2,  0.0],   # aile gauche
        [ 0.0, -1.2,  0.0],   # aile droite

        [-0.9,  0.5,  0.0],   # empennage gauche
        [-0.9, -0.5,  0.0],   # empennage droit

        [-0.9,  0.0,  0.5],   # dérive
    ], dtype=float) * scale

    segments = np.array([
        pts[0], pts[1],   # fuselage
        pts[2], pts[3],   # ailes
        pts[4], pts[5],   # empennage
        pts[1], pts[6],   # dérive
    ])

    item = gl.GLLinePlotItem(
        pos=segments,
        color=(1, 0.8, 0, 1),  # jaune ✈️
        width=3,
        antialias=True,
        mode='lines'
    )

    return item


# ======================================================
# ROTATION UTILS
# ======================================================
def rotation_matrix(roll, pitch, yaw):
    r = np.radians(roll)
    p = np.radians(pitch)
    y = np.radians(yaw)

    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [ np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r),  np.cos(r)]
    ])

    return Rz @ Ry @ Rx


# ======================================================
# MAIN
# ======================================================
class Viewer(gl.GLViewWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CAP10 Test")
        self.setCameraPosition(distance=5)

        # grid
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        self.addItem(grid)

        # axes
        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self.addItem(axis)

        # aircraft
        self.aircraft = create_cap10_item()
        self.addItem(self.aircraft)

        self.base = self.aircraft.pos.copy().reshape(-1, 3)

        # animation
        self.t = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_scene)
        self.timer.start(30)

    def update_scene(self):
        self.t += 1

        # animation sympa
        roll = 30 * np.sin(self.t * 0.05)
        pitch = 20 * np.sin(self.t * 0.03)
        yaw = self.t * 1.5

        R = rotation_matrix(roll, pitch, yaw)

        rotated = (R @ self.base.T).T

        self.aircraft.setData(pos=rotated)


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Viewer()
    w.show()
    sys.exit(app.exec_())