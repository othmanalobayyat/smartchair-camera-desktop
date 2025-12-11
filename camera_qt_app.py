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
)

from utils.logger import setup_logger
from utils.ws_client import WSClient
from utils.work_timer import WorkTimer
from utils.drowsiness import DrowsinessDetector
from utils.attention import AttentionEstimator
from utils.detection import FaceDetector
from utils.posture import PostureEstimator


# ==========================
# CONFIG LOADING
# ==========================
def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# MAIN WINDOW — CLEAN, FAST, COMFORT UI
# ============================================================
class CameraWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()

        # CONFIG
        self.config = config
        self.logger = setup_logger(config.get("log_level", "INFO"))

        self.primary = config["primary_server"]
        self.backup = config["backup_server"]
        self.device_id = config.get("camera_id", "cam_01")
        self.user_id = config.get("user_id", "user_01")

        self.send_interval = config.get("send_interval_sec", 1)
        self.attention_threshold = config.get("attention_threshold", 50)

        # COMPONENTS
        self.ws = WSClient(self.primary, self.backup, self.logger)
        self.work_timer = WorkTimer(attention_threshold=self.attention_threshold)
        self.drowsiness_detector = DrowsinessDetector()
        self.attention_estimator = AttentionEstimator()
        self.detector = FaceDetector()
        self.posture_estimator = PostureEstimator()

        # CAMERA
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            raise RuntimeError("Camera not found.")

        # STATE
        self.monitoring = False
        self.pairing_mode = False
        self.last_send_time = time.time()

        self.fps_last_time = time.time()
        self.fps = 0
        self.frame_count = 0

        # UI
        self.init_ui()

        # TIMER
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ============================================================
    # UI SETUP — CLEAN VERSION
    # ============================================================
    def init_ui(self):

        # ===== WINDOW SETTINGS =====
        self.setWindowTitle("Posturic Desktop — Comfort UI")
        self.setFixedSize(1100, 600)  # No maximize
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # ================= CAMERA =================
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background: black; border-radius: 10px;")
        layout.addWidget(self.camera_label)

        # ================= RIGHT PANEL =================
        side_panel = QVBoxLayout()

        # ------ INFO BOX ------
        self.info_box = QFrame()
        self.info_box.setStyleSheet("""
            QFrame {
                background: #F4F6FA;
                border: 1px solid #E0E5EE;
                border-radius: 12px;
            }
            QLabel {
                color: #1E293B;
                font-size: 16px;
                font-weight: 600;
            }
        """)

        info = QVBoxLayout(self.info_box)
        info.setContentsMargins(20, 20, 20, 20)

        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)

        self.lbl_title = QLabel("Posturic Desktop")
        self.lbl_title.setFont(title_font)

        self.lbl_user = QLabel(f"User: {self.user_id}")
        self.lbl_device = QLabel(f"Device: {self.device_id}")

        self.lbl_presence = QLabel("Presence: -")
        self.lbl_attention = QLabel("Attention: -")
        self.lbl_posture = QLabel("Posture: -")
        self.lbl_drowsy = QLabel("Drowsy: -")
        self.lbl_session = QLabel("Session: 00:00")
        self.lbl_fps = QLabel("FPS: 0")
        self.lbl_server = QLabel("Server: Connecting...")

        for lbl in [
            self.lbl_title, self.lbl_user, self.lbl_device,
            self.lbl_presence, self.lbl_attention, self.lbl_posture,
            self.lbl_drowsy, self.lbl_session, self.lbl_fps, self.lbl_server
        ]:
            info.addWidget(lbl)

        side_panel.addWidget(self.info_box)

        # ------ BUTTONS ------
        btn_style = """
            QPushButton {
                background: #2B4C7E;
                border-radius: 8px;
                color: white;
                padding: 10px 20px;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover { background: #4C89C8; }
        """

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_pair = QPushButton("Pair QR")
        self.btn_theme = QPushButton("Theme")

        for b in [self.btn_start, self.btn_stop, self.btn_pair, self.btn_theme]:
            b.setStyleSheet(btn_style)

        self.btn_start.clicked.connect(self.start_monitoring)
        self.btn_stop.clicked.connect(self.stop_monitoring)
        self.btn_pair.clicked.connect(self.enter_qr_mode)
        self.btn_theme.clicked.connect(self.toggle_theme)

        side_panel.addWidget(self.btn_start)
        side_panel.addWidget(self.btn_stop)
        side_panel.addWidget(self.btn_pair)
        side_panel.addWidget(self.btn_theme)

        side_panel.addStretch(1)
        layout.addLayout(side_panel)

        # ------ QR OVERLAY ------
        self.qr_overlay = QLabel(self)
        self.qr_overlay.setText("SCAN QR CODE\n\nWaiting for Posturic App...")
        self.qr_overlay.setAlignment(Qt.AlignCenter)
        self.qr_overlay.setStyleSheet("""
            background: rgba(0,0,0,0.85);
            color: white;
            font-size: 22px;
            font-weight: 700;
            border-radius: 15px;
            padding: 20px;
        """)
        self.qr_overlay.resize(380, 180)
        self.qr_overlay.hide()

    # ============================================================
    # BUTTON ACTIONS
    # ============================================================
    def start_monitoring(self):
        self.monitoring = True
        self.pairing_mode = False
        self.qr_overlay.hide()

    def stop_monitoring(self):
        self.monitoring = False
        self.pairing_mode = False
        self.qr_overlay.hide()

        self.lbl_presence.setText("Presence: -")
        self.lbl_attention.setText("Attention: -")
        self.lbl_posture.setText("Posture: -")
        self.lbl_drowsy.setText("Drowsy: -")
        self.lbl_session.setText("Session: 00:00")

    def enter_qr_mode(self):
        self.monitoring = False
        self.pairing_mode = True
        self.qr_overlay.show()

    def toggle_theme(self):
        # بسيط: فقط يعكس الخلفية
        self.setStyleSheet("background: #0F172A;")


    # ============================================================
    # FRAME UPDATE LOOP
    # ============================================================
    def update_frame(self):

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # FPS
        self.frame_count += 1
        now = time.time()
        if now - self.fps_last_time >= 1:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_last_time = now

        # QR MODE
        if self.pairing_mode:
            self.display_frame(frame)
            return

        # Monitoring Off
        if not self.monitoring:
            self.display_frame(frame)
            return

        # ==== Processing ====
        try:
            posture_label, posture_ok = self.posture_estimator.process_frame(frame)

            detected, eyes, gaze, head = self.detector.process_frame(frame)
            attention = self.attention_estimator.estimate(detected, gaze, head, eyes)

            self.drowsiness_detector.update(eyes)
            drowsy = self.drowsiness_detector.is_drowsy

            self.work_timer.update(detected, attention)

            # Send WebSocket
            if time.time() - self.last_send_time >= self.send_interval:
                self.last_send_time = time.time()
                payload = {
                    "device_id": self.device_id,
                    "user_id": self.user_id,
                    "is_present": detected,
                    "attention_level": round(attention, 1),
                    "drowsiness": drowsy,
                    "working": (self.work_timer.state == "WORK"),
                    "working_duration_seconds": self.work_timer.get_total_work_seconds(),
                    "posture_label": posture_label,
                    "posture_correct": posture_ok,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                self.ws.send_json(payload)

            # Update UI
            self.lbl_presence.setText(f"Presence: {'Present' if detected else 'Away'}")
            self.lbl_attention.setText(f"Attention: {int(attention)}%")
            self.lbl_posture.setText(f"Posture: {posture_label}")
            self.lbl_drowsy.setText(f"Drowsy: {'Yes' if drowsy else 'No'}")
            sec = self.work_timer.get_current_session_duration()
            self.lbl_session.setText(f"Session: {sec//60:02d}:{sec%60:02d}")
            self.lbl_fps.setText(f"FPS: {self.fps}")
            self.lbl_server.setText(f"Server: {'Connected' if self.ws.connected else 'Connecting...'}")

        except Exception as e:
            self.logger.error(f"Error: {e}")

        # DISPLAY
        self.display_frame(frame)

    # ============================================================
    # DISPLAY CAMERA FRAME
    # ============================================================
    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w*c, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(
            pix.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ============================================================
    # EXIT CLEANUP
    # ============================================================
    def closeEvent(self, event):
        try:
            self.cap.release()
            self.ws.close()
        except:
            pass
        cv2.destroyAllWindows()
        event.accept()


# ============================================================
# MAIN
# ============================================================
def main():
    config = load_config()
    app = QApplication([])
    window = CameraWindow(config)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
