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
from pyzbar import pyzbar

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

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not found.")

        self.last_send_time = time.time()
        self.fps_last_time = time.time()
        self.frame_count = 0
        self.fps = 0

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
        self.setWindowTitle("Posturic Desktop — V1")
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

        title = QLabel("Posturic Desktop")
        title.setFont(QFont("", 18, QFont.Bold))

        self.lbl_user = QLabel()
        self.lbl_presence = QLabel("Presence: -")
        self.lbl_attention = QLabel("Attention: -")
        self.lbl_posture = QLabel("Posture: -")
        self.lbl_drowsy = QLabel("Drowsy: -")
        self.lbl_session = QLabel("Session: 00:00")
        self.lbl_fps = QLabel("FPS: 0")
        self.lbl_server = QLabel("Server: Connecting...")

        for lbl in [
            title, self.lbl_user, self.lbl_presence, self.lbl_attention,
            self.lbl_posture, self.lbl_drowsy, self.lbl_session,
            self.lbl_fps, self.lbl_server
        ]:
            info.addWidget(lbl)

        side.addWidget(self.info_box)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
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
        self.qr_overlay.setText("Scan QR from Posturic App")
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
                QFrame { background:#1E293B; }
                QPushButton { background:#2563EB; color:white; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background:#EEF2F7; }
                QLabel { color:#1E293B; }
                QFrame { background:white; }
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
            self.lbl_user.setText(f"User: {self.user_id}")
            self.btn_pair.setText("Paired ✅")
            self.btn_pair.setEnabled(False)
            self.btn_change.setEnabled(True)
        else:
            self.lbl_user.setText("User: Not paired")
            self.btn_pair.setText("Pair Camera")
            self.btn_pair.setEnabled(True)
            self.btn_change.setEnabled(False)

        self.btn_start.setEnabled(self.paired and not self.monitoring)
        self.btn_stop.setEnabled(self.monitoring)

    # ============================================================
    # ACTIONS
    # ============================================================
    def enter_pairing(self):
        self.monitoring = False
        self.pairing_mode = True
        self.qr_overlay.show()
        self.logger.info("🔗 Pairing mode enabled")
        self.update_ui_state()

    def unpair(self):
        if QMessageBox.question(
            self, "Change Account",
            "Disconnect camera from this account?",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self.user_id = None
            self.config["user_id"] = None
            save_config(self.config)
            self.paired = False
            self.monitoring = False
            self.update_ui_state()

    def start_monitoring(self):
        if self.paired:
            self.work_timer = WorkTimer(attention_threshold=self.attention_threshold)
            self.monitoring = True
            self.last_send_time = time.time()
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

        frame_flipped = cv2.flip(frame, 1)

        if not self.ws.camera_enabled:
            self.display_frame(frame_flipped)
            return

        now = time.time()
        self.frame_count += 1
        if now - self.fps_last_time >= 1:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_last_time = now
            self.lbl_fps.setText(f"FPS: {self.fps}")

        if self.pairing_mode:
            decoded = pyzbar.decode(frame)
            for obj in decoded:
                try:
                    data = json.loads(obj.data.decode("utf-8").strip())
                    if data.get("scheme") == "smartchair" and data.get("user_id"):
                        self.user_id = data["user_id"]
                        self.config["user_id"] = self.user_id
                        save_config(self.config)
                        self.paired = True
                        self.pairing_mode = False
                        self.qr_overlay.hide()
                        self.update_ui_state()
                        return
                except Exception:
                    pass

            self.display_frame(frame_flipped)
            return

        if not self.monitoring:
            self.display_frame(frame_flipped)
            return

        if now - self.last_process_time < self.process_interval:
            self.display_frame(frame_flipped)
            return
        self.last_process_time = now

        try:
            posture_label, posture_ok = self.posture_estimator.process_frame(frame_flipped)

            face_detected, eyes_open_prob, gaze_centered, head_stable = (
                self.detector.process_frame(frame_flipped)
            )

            attention = self.attention_estimator.estimate(
                face_detected,
                gaze_centered,
                head_stable,
                eyes_open_prob,
            )

            self.drowsiness_detector.update(eyes_open_prob)
            drowsy = self.drowsiness_detector.is_drowsy

            self.work_timer.update(face_detected, attention)

            self.lbl_presence.setText(
                f"Presence: {'Present' if face_detected else 'Away'}"
            )
            self.lbl_attention.setText(f"Attention: {int(attention)}%")
            self.lbl_posture.setText(f"Posture: {posture_label}")
            self.lbl_drowsy.setText(f"Drowsy: {'Yes' if drowsy else 'No'}")

            sec = self.work_timer.get_current_session_duration()
            self.lbl_session.setText(f"Session: {sec//60:02d}:{sec%60:02d}")

            self.lbl_server.setText(
                f"Server: {'Connected' if self.ws.connected else 'Connecting...'} ({self.ws.active_url})"
            )

            if time.time() - self.last_send_time >= self.send_interval:
                self.last_send_time = time.time()

                payload = {
                    "type": "camera_frame",
                    "device_id": self.device_id,
                    "user_id": self.user_id,
                    "is_present": face_detected,
                    "attention_level": round(attention, 1),
                    "drowsiness": drowsy,
                    "working": self.work_timer.state == "WORK",
                    "working_duration_seconds": self.work_timer.get_total_work_seconds(),
                    "posture_label": posture_label,
                    "posture_correct": posture_ok,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }

                self.send_camera_frame(payload)

        except Exception:
            self.logger.exception("Error during frame processing")

        self.display_frame(frame_flipped)

    # ============================================================
    # NETWORK HELPER
    # ============================================================
    def send_camera_frame(self, payload: dict):
        self.ws.send_json(payload)

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
