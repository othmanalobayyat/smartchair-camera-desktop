class AttentionEstimator:
    def __init__(self):
        pass

    def estimate(self, face_detected: bool, gaze_centered: float, head_stable: float, eyes_open_prob: float) -> float:
        if not face_detected:
            return 0.0

        w_gaze = 0.4
        w_head = 0.3
        w_eyes = 0.3

        score = (w_gaze * gaze_centered +
                 w_head * head_stable +
                 w_eyes * eyes_open_prob)

        return max(0.0, min(100.0, score * 100.0))
