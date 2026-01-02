import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)

import json
import time
from datetime import datetime

import cv2

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QMessageBox,
)

from utils.logger import setup_logger
from utils.ws_client import WSClient
from utils.work_timer import WorkTimer
from utils.drowsiness import DrowsinessDetector
from utils.attention import AttentionEstimator
from utils.detection import FaceDetector
from utils.posture import PostureEstimator


# ==========================
# CONFIG
# ==========================
def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config, path="config.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ==========================
# USER-FRIENDLY MAPS
# ==========================
POSTURE_MAP = {
    "TUP": "Good Posture",
    "TLF": "Leaning Forward",
    "TLB": "Leaning Backward",
    "TLR": "Leaning Right",
    "TLL": "Leaning Left",
    "NO_PERSON": "No Person Detected",
}


# ============================================================
# MAIN WINDOW
# ============================================================
class CameraWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.logger = setup_logger(config.get("log_level", "INFO"))

        self.primary = config["primary_server"]
        self.backup = config["backup_server"]
        self.device_id = config.get("camera_id", "cam_01")
        self.user_id = config.get("user_id")
        self.user_name = config.get("user_name")

        self.dark_mode = config.get("dark_mode", False)

        self.paired = bool(self.user_id)
        self.monitoring = False
        self.pairing_mode = False

        self.send_interval = config.get("send_interval_sec", 1)
        self.attention_threshold = config.get("attention_threshold", 50)

        self.ws = WSClient(self.primary, self.backup, self.logger)
        self.work_timer = WorkTimer(attention_threshold=self.attention_threshold)
        self.drowsiness_detector = DrowsinessDetector(
            eye_closed_threshold_sec=config.get("eye_closed_threshold_sec", 2.0)
        )
        self.attention_estimator = AttentionEstimator()
        self.detector = FaceDetector()
        self.posture_estimator = PostureEstimator()

        self.qr_detector = cv2.QRCodeDetector()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not found.")

        self.last_send_time = time.time()
        self.last_process_time = 0.0
        self.process_interval = config.get("process_interval_sec", 0.12)

        self.init_ui()
        self.apply_theme()
        self.update_ui_state()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ============================================================
    # UI
    # ============================================================
    def init_ui(self):
        self.setWindowTitle("Posturic Camera Desktop")
        self.setFixedSize(1100, 600)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background:black;border-radius:10px;")
        layout.addWidget(self.camera_label)

        side = QVBoxLayout()

        self.info_box = QFrame()
        info = QVBoxLayout(self.info_box)
        info.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Posturic Camera Desktop")
        title.setFont(QFont("", 18, QFont.Bold))

        self.lbl_user = QLabel(f"Hello, {self.user_name}")
        self.lbl_user.setFont(QFont("", 14, QFont.Bold))

        self.lbl_presence = QLabel("Presence: -")
        self.lbl_focus = QLabel("Focus: -")
        self.lbl_posture = QLabel("Posture: -")
        self.lbl_drowsy = QLabel("Drowsiness: -")
        self.lbl_session = QLabel("Session: 00:00:00")
        self.lbl_connection = QLabel("Connection: -")

        for lbl in [
            title,
            self.lbl_user,
            self.lbl_presence,
            self.lbl_focus,
            self.lbl_posture,
            self.lbl_drowsy,
            self.lbl_session,
            self.lbl_connection,
        ]:
            info.addWidget(lbl)

        side.addWidget(self.info_box)

        self.btn_start = QPushButton("Start Monitoring")
        self.btn_stop = QPushButton("Stop Monitoring")
        self.btn_pair = QPushButton()
        self.btn_change = QPushButton("Change Account")
        self.btn_theme = QPushButton("Theme")

        for b in [self.btn_start, self.btn_stop, self.btn_pair, self.btn_change, self.btn_theme]:
            b.setStyleSheet("padding:10px;font-weight:600;border-radius:8px;")

        self.btn_start.clicked.connect(self.start_monitoring)
        self.btn_stop.clicked.connect(self.stop_monitoring)
        self.btn_pair.clicked.connect(self.enter_pairing)
        self.btn_change.clicked.connect(self.unpair)
        self.btn_theme.clicked.connect(self.toggle_theme)

        side.addWidget(self.btn_start)
        side.addWidget(self.btn_stop)
        side.addWidget(self.btn_pair)
        side.addWidget(self.btn_change)
        side.addWidget(self.btn_theme)
        side.addStretch()

        layout.addLayout(side)

        self.qr_overlay = QLabel(self)
        self.qr_overlay.setAlignment(Qt.AlignCenter)
        self.qr_overlay.setText("Scan QR from Posturic Smart Chair App")
        self.qr_overlay.resize(420, 200)
        self.qr_overlay.hide()

    # ============================================================
    # THEME
    # ============================================================
    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow { background:#0F172A; }
                QLabel { color:#E5E7EB; }
                QFrame { background:#1E293B; border-radius:12px; }
                QPushButton { background:#2563EB; color:white; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background:#EEF2F7; }
                QLabel { color:#1E293B; }
                QFrame { background:white; border-radius:12px; }
                QPushButton { background:#2B4C7E; color:white; }
            """)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.config["dark_mode"] = self.dark_mode
        save_config(self.config)
        self.apply_theme()

    # ============================================================
    # UI STATE
    # ============================================================
    def update_ui_state(self):
        if self.paired:
            self.btn_pair.setText("Paired ✅")
            self.btn_pair.setEnabled(False)
            self.btn_change.setEnabled(True)
        else:
            self.btn_pair.setText("Pair Camera")
            self.btn_pair.setEnabled(True)
            self.btn_change.setEnabled(False)

        self.btn_start.setVisible(self.paired and not self.monitoring)
        self.btn_stop.setVisible(self.monitoring)

    # ============================================================
    # ACTIONS
    # ============================================================
    def enter_pairing(self):
        self.monitoring = False
        self.pairing_mode = True
        self.qr_overlay.show()
        self.update_ui_state()

    def unpair(self):
        if QMessageBox.question(
            self,
            "Change Account",
            "Disconnect camera from this account?",
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes:
            self.user_id = None
            self.config["user_id"] = None
            save_config(self.config)
            self.paired = False
            self.monitoring = False
            self.update_ui_state()

    def start_monitoring(self):
        self.work_timer = WorkTimer(attention_threshold=self.attention_threshold)
        self.monitoring = True
        self.update_ui_state()

    def stop_monitoring(self):
        self.monitoring = False
        self.update_ui_state()

    # ============================================================
    # FRAME LOOP
    # ============================================================
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        now = time.time()

        if self.pairing_mode:
            data, _, _ = self.qr_detector.detectAndDecode(frame)
            if data:
                try:
                    payload = json.loads(data)
                    if payload.get("user_id") and payload.get("user_name"):
                        self.user_id = payload["user_id"]
                        self.user_name = payload["user_name"]

                        self.config["user_id"] = self.user_id
                        self.config["user_name"] = self.user_name
                        save_config(self.config)

                        self.lbl_user.setText(f"Hello, {self.user_name}")

                        self.paired = True
                        self.pairing_mode = False
                        self.qr_overlay.hide()
                        self.update_ui_state()
                except Exception:
                    pass

            self.display_frame(frame)
            return

        if not self.monitoring:
            self.display_frame(frame)
            return

        if now - self.last_process_time < self.process_interval:
            self.display_frame(frame)
            return
        self.last_process_time = now

        posture_label, posture_ok = self.posture_estimator.process_frame(frame)
        face_detected, eyes_open_prob, gaze_centered, head_stable = self.detector.process_frame(frame)

        attention = self.attention_estimator.estimate(
            face_detected, gaze_centered, head_stable, eyes_open_prob
        )

        self.drowsiness_detector.update(eyes_open_prob)
        drowsy = self.drowsiness_detector.is_drowsy

        self.work_timer.update(face_detected, attention)

        self.lbl_presence.setText(f"Presence: {'Present' if face_detected else 'Away'}")

        if attention >= 70:
            self.lbl_focus.setText("Focus: High")
            self.lbl_focus.setStyleSheet("color:#22C55E; font-weight:600;")
        elif attention >= 40:
            self.lbl_focus.setText("Focus: Medium")
            self.lbl_focus.setStyleSheet("color:#F59E0B; font-weight:600;")
        else:
            self.lbl_focus.setText("Focus: Low")
            self.lbl_focus.setStyleSheet("color:#EF4444; font-weight:600;")

        posture_text = POSTURE_MAP.get(posture_label, posture_label)
        self.lbl_posture.setText(f"Posture: {posture_text}")
        self.lbl_posture.setStyleSheet(
            "color:#22C55E;" if posture_ok else "color:#EF4444;"
        )

        self.lbl_drowsy.setText(
            f"Drowsiness: {'Detected' if drowsy else 'Normal'}"
        )
        self.lbl_drowsy.setStyleSheet(
            "color:#EF4444;" if drowsy else "color:#22C55E;"
        )

        sec = self.work_timer.get_current_session_duration()
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        self.lbl_session.setText(f"Session: {h:02d}:{m:02d}:{s:02d}")

        self.lbl_connection.setText(
            f"Connection: {'Online' if self.ws.connected else 'Offline'}"
        )
        self.lbl_connection.setStyleSheet(
            "color:#22C55E;" if self.ws.connected else "color:#EF4444;"
        )

        self.display_frame(frame)

    # ============================================================
    # DISPLAY
    # ============================================================
    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        self.camera_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        self.cap.release()
        self.ws.close()
        event.accept()


# ============================================================
# MAIN
# ============================================================
def main():
    config = load_config()
    app = QApplication([])
    win = CameraWindow(config)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
