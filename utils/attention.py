# utils/attention.py
class AttentionEstimator:
    """
    Attention v2
    نظام انتباه يعتمد على:
        - gaze_centered
        - head_stable
        - eyes_open_prob
        - face_detected
    مع فلترة EMA لتقليل التقلبات.
    """

    def __init__(self):
        self.prev_attention = 0.0
        self.alpha = 0.6  # كلما ارتفعت → استجابة أسرع

        # توزيع الأهمية
        self.w_gaze = 0.5
        self.w_head = 0.3
        self.w_eyes = 0.2

    def estimate(
        self,
        face_detected: bool,
        gaze_centered: float,
        head_stable: float,
        eyes_open_prob: float,
    ) -> float:

        # 1) إذا الوجه غير موجود → 0
        if not face_detected:
            raw = 0.0
        else:
            # 2) دمج القيم الثلاث بحسب الأهمية
            raw = (
                self.w_gaze * gaze_centered +
                self.w_head * head_stable +
                self.w_eyes * eyes_open_prob
            )

        # 3) فلترة أقل قفزات (EMA)
        smoothed = (
            self.alpha * raw +
            (1 - self.alpha) * self.prev_attention
        )
        self.prev_attention = smoothed

        # 4) تحويل إلى 0–100
        return max(0.0, min(100.0, smoothed * 100.0))
