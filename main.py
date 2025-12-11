import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # يخفي تحذيرات TensorFlow Lite

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# إخفاء protobuf spam
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# main.py
import json
import time
from datetime import datetime

import cv2

from utils.logger import setup_logger
from utils.ws_client import WSClient
from utils.work_timer import WorkTimer
from utils.drowsiness import DrowsinessDetector
from utils.attention import AttentionEstimator
from utils.detection import FaceDetector
from utils.posture import PostureEstimator


def load_config(path: str = "config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # ===============================
    # Load configuration
    # ===============================
    config = load_config()

    logger = setup_logger(config.get("log_level", "INFO"))

    primary = config["primary_server"]
    backup = config["backup_server"]
    camera_id = config["camera_id"]
    user_id = config["user_id"]
    send_interval = config.get("send_interval_sec", 1)
    attention_threshold = config.get("attention_threshold", 50)

    # ===============================
    # Initialize components
    # ===============================
    ws = WSClient(primary, backup, logger)

    work_timer = WorkTimer(attention_threshold=attention_threshold)
    drowsiness_detector = DrowsinessDetector()
    attention_estimator = AttentionEstimator()
    detector = FaceDetector()
    posture_estimator = PostureEstimator()

    # ===============================
    # Camera setup
    # ===============================
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        logger.error("Could not open camera.")
        return

    logger.info("Camera app started.")
    last_send_time = time.time()

    # ===============================
    # FPS + Session timer setup
    # ===============================
    fps_last_time = time.time()
    fps = 0
    frame_count = 0

    # ===============================
    # Posture friendly labels
    # ===============================
    POSTURE_MAP = {
        "TUP": "Correct ✓",
        "TLF": "Leaning Forward",
        "TLB": "Leaning Backward",
        "TLR": "Leaning Right",
        "TLL": "Leaning Left",
        "NO_PERSON": "No person detected",
    }

    try:
        while True:
            # ===============================
            # Read camera frame
            # ===============================
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera.")
                break

            h, w, _ = frame.shape

            # ===============================
            # FPS calculation
            # ===============================
            frame_count += 1
            now_fps = time.time()
            if now_fps - fps_last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_last_time = now_fps

            # ===============================
            # Posture estimation
            # ===============================
            posture_label, posture_ok = posture_estimator.process_frame(frame)
            friendly_posture = POSTURE_MAP.get(posture_label, posture_label)

            # ===============================
            # Face & attention detection
            # ===============================
            face_detected, eyes_open_prob, gaze_centered, head_stable = detector.process_frame(
                frame
            )
            attention_level = attention_estimator.estimate(
                face_detected=face_detected,
                gaze_centered=gaze_centered,
                head_stable=head_stable,
                eyes_open_prob=eyes_open_prob,
            )

            # ===============================
            # Drowsiness detection
            # ===============================
            drowsiness_detector.update(eyes_open_prob=eyes_open_prob)
            is_drowsy = drowsiness_detector.is_drowsy

            # ===============================
            # Work timer update
            # ===============================
            is_present = face_detected
            work_timer.update(
                is_present=is_present,
                attention_level=attention_level,
            )

            total_work_sec = work_timer.get_total_work_seconds()
            working_flag = work_timer.state == "WORK"

            # ===============================
            # Send data to server
            # ===============================
            now = time.time()
            if now - last_send_time >= send_interval:
                last_send_time = now

                payload = {
                    "device_id": camera_id,
                    "user_id": user_id,
                    "is_present": is_present,
                    "attention_level": round(attention_level, 1),
                    "drowsiness": is_drowsy,
                    "working": working_flag,
                    "working_duration_seconds": total_work_sec,
                    "posture_label": posture_label,
                    "posture_correct": posture_ok,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }

                logger.info(
                    f"Sending: present={is_present}, "
                    f"attention={attention_level:.1f}, "
                    f"work_sec={total_work_sec}, "
                    f"drowsy={is_drowsy}, "
                    f"posture={posture_label}, ok={posture_ok}"
                )

                ws.send_json(payload)

            # ===============================
            # DASHBOARD OVERLAY (Full Panel)
            # ===============================

            # 1) نجهز لوحة جانبية شبه شفافة
            overlay = frame.copy()

            panel_width = int(w * 0.45)  # حوالي نصف العرض
            x1 = w - panel_width - 10
            y1 = 10
            x2 = w - 10
            y2 = h - 10

            # خلفية سوداء شفافة
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                -1,
            )

            alpha = 0.55  # شفافية اللوحة
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # 2) نصوص منظمة داخل اللوحة
            base_x = x1 + 15
            line_y = y1 + 25
            line_step = 22

            # عنوان
            cv2.putText(
                frame,
                "Smart Chair Camera",
                (base_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            line_y += line_step + 5

            cv2.line(
                frame,
                (base_x, line_y),
                (x2 - 15, line_y),
                (80, 80, 80),
                1,
            )
            line_y += line_step

            # Presence / Focus
            presence_text = "Present" if is_present else "Not detected"
            presence_color = (0, 200, 0) if is_present else (0, 0, 255)

            cv2.putText(
                frame,
                f"Presence: {presence_text}",
                (base_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                presence_color,
                1,
            )
            line_y += line_step

            # Focus Level
            if not is_present:
                focus_text = "0% (No user)"
                focus_color = (0, 0, 255)
            else:
                focus_val = int(attention_level)
                if focus_val >= 70:
                    focus_color = (0, 200, 0)
                    focus_status = "High"
                elif focus_val >= 40:
                    focus_color = (0, 215, 255)
                    focus_status = "Medium"
                else:
                    focus_color = (0, 0, 255)
                    focus_status = "Low"

                focus_text = f"{focus_val}% ({focus_status})"

            cv2.putText(
                frame,
                f"Focus: {focus_text}",
                (base_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                focus_color,
                1,
            )
            line_y += line_step

            # Drowsiness
            drowsy_text = "Yes" if is_drowsy else "No"
            drowsy_color = (0, 0, 255) if is_drowsy else (0, 200, 0)

            cv2.putText(
                frame,
                f"Drowsiness: {drowsy_text}",
                (base_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                drowsy_color,
                1,
            )
            line_y += line_step

            # Posture
            cv2.putText(
                frame,
                "Posture:",
                (base_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            line_y += line_step

            cv2.putText(
                frame,
                f"  {friendly_posture}",
                (base_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if posture_ok else (0, 0, 255),
                1,
            )
            line_y += line_step + 5

            # خط فاصل
            cv2.line(
                frame,
                (base_x, line_y),
                (x2 - 15, line_y),
                (80, 80, 80),
                1,
            )
            line_y += line_step

            # Session time
            session_sec = work_timer.get_current_session_duration()
            session_min = session_sec // 60
            session_rem = session_sec % 60

            cv2.putText(
                frame,
                f"Session: {session_min:02d}:{session_rem:02d}",
                (base_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            line_y += line_step

            # Total work
            total_min = total_work_sec // 60
            total_rem = total_work_sec % 60

            cv2.putText(
                frame,
                f"Total Focus Time: {total_min:02d}:{total_rem:02d}",
                (base_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 255),
                1,
            )
            line_y += line_step

            # Camera FPS
            cv2.putText(
                frame,
                f"Camera FPS: {fps}",
                (base_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            line_y += line_step

            # WebSocket status
            ws_status_text = "Connected" if ws.connected else "Connecting..."
            ws_color = (0, 200, 0) if ws.connected else (0, 165, 255)

            cv2.putText(
                frame,
                f"Server: {ws_status_text}",
                (base_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ws_color,
                1,
            )

            # ===============================
            # Show frame
            # ===============================
            cv2.imshow("Smart Chair – Camera Dashboard", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Exiting on user request (q pressed).")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        ws.close()
        logger.info("Camera app stopped.")


if __name__ == "__main__":
    main()
