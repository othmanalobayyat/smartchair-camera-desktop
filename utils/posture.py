# utils/posture.py
"""
Posture Estimation Module

This module extracts 39 upper-body skeletal features using MediaPipe Pose,
normalizes them relative to the hip center, and performs posture classification
using a pre-trained XGBoost model.

The model was trained offline and is used here only for inference.
"""

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
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_EYE_INNER,
            mp_pose.PoseLandmark.LEFT_EYE,
            mp_pose.PoseLandmark.LEFT_EYE_OUTER,
            mp_pose.PoseLandmark.RIGHT_EYE_INNER,
            mp_pose.PoseLandmark.RIGHT_EYE,
            mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
            mp_pose.PoseLandmark.LEFT_EAR,
            mp_pose.PoseLandmark.RIGHT_EAR,
            mp_pose.PoseLandmark.MOUTH_LEFT,
            mp_pose.PoseLandmark.MOUTH_RIGHT,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
        ]

        # مفاصل الحوض لحساب hip center
        self.left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        self.right_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP

    def _extract_features(self, pose_landmarks) -> np.ndarray:
        """
        استخراج الـ 39 feature من 13 مفصل علوي مع طرح hip center.
        """

        # 1) حساب hip center
        left_hip = pose_landmarks.landmark[self.left_hip_idx.value]
        right_hip = pose_landmarks.landmark[self.right_hip_idx.value]

        hip_x = (left_hip.x + right_hip.x) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0
        hip_z = (left_hip.z + right_hip.z) / 2.0

        # 2) تجميع الميزات
        feats = []
        for lm_enum in self.joints_order:
            lm = pose_landmarks.landmark[lm_enum.value]

            x_rel = lm.x - hip_x
            y_rel = lm.y - hip_y
            z_rel = lm.z - hip_z

            feats.extend([x_rel, y_rel, z_rel])

        return np.array(feats, dtype=np.float32).reshape(1, -1)  # (1, 39)

    def process_frame(self, frame):
        """
        يعيد (posture_label, posture_ok)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return "NO_PERSON", False

        # 1) استخراج الميزات
        X = self._extract_features(results.pose_landmarks)
        X_scaled = self.scaler.transform(X)

        # 2) توقع الموديل
        y_prob = self.model.predict_proba(X_scaled)[0]
        class_idx = int(np.argmax(y_prob))
        raw_label = self.label_enc.classes_[class_idx]

        # 3) حساب حدود manual adjustment
        lm = results.pose_landmarks.landmark

        nose_z = lm[self.joints_order[0].value].z
        left_sh = lm[self.joints_order[-2].value]
        right_sh = lm[self.joints_order[-1].value]

        shoulder_diff = abs(left_sh.y - right_sh.y)

        # قواعد التصحيح
        if shoulder_diff < 0.02:
            final_label = "TUP"
        elif nose_z > -0.15:
            final_label = "TUP"
        else:
            final_label = raw_label

        posture_ok = (final_label == self.good_label)

        return final_label, posture_ok
