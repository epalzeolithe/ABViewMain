import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from datetime import datetime


class ClockWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Horloge")
        self.resize(300, 120)

        layout = QVBoxLayout()

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 40px; font-family: monospace;")

        layout.addWidget(self.label)
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

        self.update_time()

    def update_time(self):
        now = datetime.now().strftime("%H:%M:%S")
        self.label.setText(now)


app = QApplication(sys.argv)
window = ClockWindow()
window.show()

sys.exit(app.exec_())