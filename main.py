import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # يخفي تحذيرات TensorFlow Lite

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# اخفاء protobuf spam
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)



# main.py
import json
import time
from datetime import datetime

import cv2

import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)


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

        try:
            while True:
                # ===============================
                # Read camera frame
                # ===============================
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera.")
                    break

                # ===============================
                # Posture estimation (XGBoost model)
                # ===============================
                posture_label, posture_ok = posture_estimator.process_frame(
                    frame
                )

                # ===============================
                # Face & attention detection
                # ===============================
                (
                    face_detected,
                    eyes_open_prob,
                    gaze_centered,
                    head_stable,
                ) = detector.process_frame(frame)

                attention_level = attention_estimator.estimate(
                    face_detected=face_detected,
                    gaze_centered=gaze_centered,
                    head_stable=head_stable,
                    eyes_open_prob=eyes_open_prob,
                )

                # ===============================
                # Drowsiness detection
                # ===============================
                drowsiness_detector.update(
                    eyes_open_prob=eyes_open_prob
                )
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
                working_flag = (work_timer.state == "WORK")


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
                        f"posture={posture_label}, "
                        f"ok={posture_ok}"
                    )

                    ws.send_json(payload)

                # ===============================
                # UI Overlay
                # ===============================
                cv2.putText(
                    frame,
                    f"Attn: {int(attention_level)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                posture_color = (0, 255, 0) if posture_ok else (0, 0, 255)
                cv2.putText(
                    frame,
                    f"Posture: {posture_label}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    posture_color,
                    2,
                )

                cv2.imshow(
                    "Camera Attention Monitor - Smart Chair",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info(
                        "Exiting on user request (q pressed)."
                    )
                    break

        finally:
            # ===============================
            # Cleanup
            # ===============================
            cap.release()
            cv2.destroyAllWindows()
            ws.close()
            logger.info("Camera app stopped.")


if __name__ == "__main__":
    main()
