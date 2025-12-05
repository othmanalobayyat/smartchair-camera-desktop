# utils/detection.py
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


class FaceDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame):
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False, 0.0, 0.0, 0.0

        face_landmarks = results.multi_face_landmarks[0]

        # ===============================
        # Face center (gaze estimation)
        # ===============================
        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]

        face_center_x = np.mean(xs)
        face_center_y = np.mean(ys)

        dx = abs(face_center_x - 0.5)
        dy = abs(face_center_y - 0.5)
        dist = dx + dy

        gaze_centered = max(
            0.0,
            1.0 - dist * 2,
        )

        # ===============================
        # Eye openness estimation
        # ===============================
        left_eye_indices = [
            33, 160, 159, 158, 133, 153, 144, 145
        ]

        right_eye_indices = [
            362, 385, 386, 387, 263, 373, 380, 374
        ]

        def eye_open_score(indices):
            pts = np.array(
                [
                    [
                        face_landmarks.landmark[i].x * w,
                        face_landmarks.landmark[i].y * h,
                    ]
                    for i in indices
                ]
            )

            min_y = np.min(pts[:, 1])
            max_y = np.max(pts[:, 1])
            min_x = np.min(pts[:, 0])
            max_x = np.max(pts[:, 0])

            vert = max_y - min_y
            horiz = max_x - min_x + 1e-6

            ratio = vert / horiz
            return ratio

        left_ratio = eye_open_score(left_eye_indices)
        right_ratio = eye_open_score(right_eye_indices)

        avg_ratio = (left_ratio + right_ratio) / 2.0

        eyes_open_prob = max(
            0.0,
            min(1.0, (avg_ratio - 0.1) / 0.2),
        )

        # ===============================
        # Head stability (placeholder)
        # ===============================
        head_stable = 1.0

        return (
            True,
            float(eyes_open_prob),
            float(gaze_centered),
            float(head_stable),
        )
