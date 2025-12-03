import time


class DrowsinessDetector:
    def __init__(
        self,
        eye_closed_threshold_sec: float = 2.0,
    ):
        self.eye_closed_threshold_sec = eye_closed_threshold_sec
        self._eye_closed_start = None
        self.is_drowsy = False

    def update(self, eyes_open_prob: float):
        now = time.time()

        if eyes_open_prob < 0.3:
            if self._eye_closed_start is None:
                self._eye_closed_start = now
            elif (
                now - self._eye_closed_start
                >= self.eye_closed_threshold_sec
            ):
                self.is_drowsy = True
        else:
            self._eye_closed_start = None
            self.is_drowsy = False
