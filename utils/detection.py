# utils/detection.py
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


class FaceDetector:
    """
    يقوم باكتشاف الوجه باستخدام MediaPipe FaceMesh
    ويعيد:
        - face_detected (bool)
        - eyes_open_prob (0..1) مبني على EAR
        - gaze_centered (0..1) مدى تمركز الوجه في وسط الإطار
        - head_stable (0..1) مدى ثبات الرأس بين الإطارات
    """

    def __init__(self):
        # نموذج FaceMesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # مؤشرات العينين (6 نقاط لكل عين) لاستخدامها في EAR
        # مأخوذة من أمثلة EAR مع MediaPipe FaceMesh
        # Right eye: 33, 159, 158, 133, 153, 145
        # Left eye : 362, 385, 386, 263, 373, 380
        self.RIGHT_EYE_IDX = [33, 159, 158, 133, 153, 145]
        self.LEFT_EYE_IDX = [362, 385, 386, 263, 373, 380]

        # لتقدير ثبات الرأس
        self._prev_center = None  # (x, y) normalized in [0,1]
        self._head_stability_ema = 1.0
        self._ema_alpha = 0.5  # كلما كبرت → استجابة أسرع

    # ===============================
    # Utilities
    # ===============================
    @staticmethod
    def _lm_xy(landmark, w, h):
        """إرجاع نقطة (x, y) بالبكسل من Landmark."""
        return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)

    def _eye_ear(self, landmarks, indices, w, h):
        """
        حساب Eye Aspect Ratio لعين واحدة باستخدام 6 نقاط.
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        pts = [self._lm_xy(landmarks[i], w, h) for i in indices]

        p1, p2, p3, p4, p5, p6 = pts

        # المسافات الرأسية
        d1 = np.linalg.norm(p2 - p6)
        d2 = np.linalg.norm(p3 - p5)
        # المسافة الأفقية
        d3 = np.linalg.norm(p1 - p4) + 1e-6  # لتجنب القسمة على صفر

        ear = (d1 + d2) / (2.0 * d3)
        return float(ear)

    def _ear_to_prob(self, ear: float) -> float:
        """
        تحويل EAR إلى احتمال فتح العين.
        قيم EAR النموذجية:
            - عين مفتوحة ~ 0.30 - 0.40
            - عين مغلقة ~ 0.15 - 0.20
        نقوم بعمل mapping خطي تقريبي بين 0.15 و 0.35.
        """
        # نحدّد حدود تقريبية
        closed_ear = 0.15
        open_ear = 0.35

        prob = (ear - closed_ear) / (open_ear - closed_ear)
        prob = max(0.0, min(1.0, prob))
        return prob

    def _update_head_stability(self, center_xy_norm):
        """
        center_xy_norm: np.array([x, y]) between 0 and 1
        نستخدم حركة مركز الوجه بين الإطارات لحساب الثبات.
        """
        if self._prev_center is None:
            self._prev_center = center_xy_norm
            self._head_stability_ema = 1.0
            return 1.0

        # المسافة الإقليدية بين الإطار الحالي والسابق (بوحدات normalized)
        movement = float(np.linalg.norm(center_xy_norm - self._prev_center))
        self._prev_center = center_xy_norm

        # نعتبر أن حركة 0.00 → ثابت جداً، 0.03 أو أكثر → غير ثابت
        # (0.03 تقريباً تعني تحرك واضح للرأس في إطار 320x240)
        max_movement = 0.03
        movement_norm = min(movement / max_movement, 1.0)

        head_stable_instant = 1.0 - movement_norm  # 1 ثابت، 0 غير ثابت

        # فلترة بالإكسپوننشال موفنج أفريج لتخفيف الاهتزاز
        self._head_stability_ema = (
            (1.0 - self._ema_alpha) * self._head_stability_ema
            + self._ema_alpha * head_stable_instant
        )

        # تأكد أنه ضمن [0,1]
        return max(0.0, min(1.0, self._head_stability_ema))

    # ===============================
    # Public API
    # ===============================
    def process_frame(self, frame):
        """
        يعالج إطار BGR من الكاميرا ويعيد:
            face_detected (bool)
            eyes_open_prob (float 0..1)
            gaze_centered (float 0..1)
            head_stable (float 0..1)
        """
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # لا يوجد وجه
            return False, 0.0, 0.0, 0.0

        face_landmarks = results.multi_face_landmarks[0].landmark

        # ===============================
        # 1) Face center & gaze_centered
        # ===============================
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]

        face_center_x = float(np.mean(xs))
        face_center_y = float(np.mean(ys))

        dx = abs(face_center_x - 0.5)
        dy = abs(face_center_y - 0.5)

        # كلما اقترب من منتصف الشاشة (0.5, 0.5) زاد التركيز
        dist = dx + dy  # في حدود تقريبية [0, ~1]
        gaze_centered = max(0.0, 1.0 - dist * 2.0)  # clamp تقريباً إلى [0,1]

        # ===============================
        # 2) Eye openness via EAR
        # ===============================
        right_ear = self._eye_ear(face_landmarks, self.RIGHT_EYE_IDX, w, h)
        left_ear = self._eye_ear(face_landmarks, self.LEFT_EYE_IDX, w, h)
        avg_ear = (right_ear + left_ear) / 2.0

        eyes_open_prob = self._ear_to_prob(avg_ear)

        # ===============================
        # 3) Head stability
        # ===============================
        center_norm = np.array([face_center_x, face_center_y], dtype=np.float32)
        head_stable = self._update_head_stability(center_norm)

        return (
            True,
            float(eyes_open_prob),
            float(gaze_centered),
            float(head_stable),
        )
