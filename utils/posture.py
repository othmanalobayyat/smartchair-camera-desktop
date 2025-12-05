# utils/posture.py
import os
import cv2
import numpy as np
import mediapipe as mp
import joblib

mp_pose = mp.solutions.pose


class PostureEstimator:
    """
    يستخدم MediaPipe Pose لاستخراج 13 مفصل علوي
    (nose, eyes, ears, mouth, shoulders)
    ويطبع الإحداثيات نسبةً إلى hip center (مثل الداتا الأصلية)،
    ثم يمرر 39 feature إلى موديل XGBoost المدرّب.
    """

    def __init__(
        self,
        model_path: str = "models/xgb_posture_model.pkl",
        scaler_path: str = "models/scaler.pkl",
        label_enc_path: str = "models/label_encoder.pkl",
    ):
        # التأكد من وجود الملفات
        if not (os.path.exists(model_path)
                and os.path.exists(scaler_path)
                and os.path.exists(label_enc_path)):
            raise FileNotFoundError(
                "Posture model / scaler / label_encoder not found. "
                "تأكد من وجود الملفات التالية داخل مجلد models/:\n"
                " - xgb_posture_model.pkl\n"
                " - scaler.pkl\n"
                " - label_encoder.pkl"
            )

        # تحميل الموديل، الـ scaler، والـ LabelEncoder
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_enc = joblib.load(label_enc_path)

        # MediaPipe Pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # الـ label الذي نعتبره "جلسة صحيحة"
        self.good_label = "TUP"

        # ترتيب المفاصل المطابق تمامًا لترتيب الأعمدة في التدريب
        self.joints_order = [
            mp_pose.PoseLandmark.NOSE,             # nose_x, nose_y, nose_z
            mp_pose.PoseLandmark.LEFT_EYE_INNER,   # left_eye_inner_*
            mp_pose.PoseLandmark.LEFT_EYE,         # left_eye_*
            mp_pose.PoseLandmark.LEFT_EYE_OUTER,   # left_eye_outer_*
            mp_pose.PoseLandmark.RIGHT_EYE_INNER,  # right_eye_inner_*
            mp_pose.PoseLandmark.RIGHT_EYE,        # right_eye_*
            mp_pose.PoseLandmark.RIGHT_EYE_OUTER,  # right_eye_outer_*
            mp_pose.PoseLandmark.LEFT_EAR,         # left_ear_*
            mp_pose.PoseLandmark.RIGHT_EAR,        # right_ear_*
            mp_pose.PoseLandmark.MOUTH_LEFT,       # mouth_left_*
            mp_pose.PoseLandmark.MOUTH_RIGHT,      # mouth_right_*
            mp_pose.PoseLandmark.LEFT_SHOULDER,    # left_shoulder_*
            mp_pose.PoseLandmark.RIGHT_SHOULDER,   # right_shoulder_*
        ]

        # مفاصل الحوض لحساب hip center (مثل وصف الداتا)
        self.left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        self.right_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP

    def _extract_features(self, pose_landmarks) -> np.ndarray:
        """
        يحول 13 landmark علوي فقط إلى 39 feature بالترتيب:
        ['nose_x', 'nose_y', 'nose_z',
         'left_eye_inner_x', ..., 'right_shoulder_z']
        لكن بعد طرح hip center من كل إحداثية (نفس تطبيع الداتا).
        """

        # 1) حساب hip center
        left_hip = pose_landmarks.landmark[self.left_hip_idx.value]
        right_hip = pose_landmarks.landmark[self.right_hip_idx.value]

        hip_x = (left_hip.x + right_hip.x) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0
        hip_z = (left_hip.z + right_hip.z) / 2.0

        # 2) تجميع الميزات نسبةً إلى hip center
        feats = []
        for lm_enum in self.joints_order:
            lm = pose_landmarks.landmark[lm_enum.value]

            # طرح hip center (نفس فكرة "normalized relative to hip center")
            x_rel = lm.x - hip_x
            y_rel = lm.y - hip_y
            z_rel = lm.z - hip_z

            feats.extend([x_rel, y_rel, z_rel])

        feats = np.array(feats, dtype=np.float32).reshape(1, -1)  # (1, 39)
        return feats

    def process_frame(self, frame):
        """
        يأخذ frame من الكاميرا ويعيد:
          (posture_label: str, posture_ok: bool)

        لو لم يُكتشف شخص → ("NO_PERSON", False)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        # لا يوجد شخص / لا يوجد pose
        if not results.pose_landmarks:
            return "NO_PERSON", False

        # استخراج الـ 39 feature كما في التدريب
        X = self._extract_features(results.pose_landmarks)

        # تطبيق نفس الـ scaler المستخدم أثناء التدريب
        X_scaled = self.scaler.transform(X)

        # توقع الفئة (XGBoost)
        y_prob = self.model.predict_proba(X_scaled)[0]
        class_idx = int(np.argmax(y_prob))

        label = self.label_enc.classes_[class_idx]
        posture_ok = (label == self.good_label)

        return label, bool(posture_ok)
