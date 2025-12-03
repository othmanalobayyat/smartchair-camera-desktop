import time


class WorkTimer:
    def __init__(self, attention_threshold: int):
        self.attention_threshold = attention_threshold
        self.total_work_seconds = 0.0
        self._last_update_time = time.time()
        self.currently_working = False

    def update(
        self,
        is_present: bool,
        attention_level: float,
    ):
        now = time.time()
        delta = now - self._last_update_time
        self._last_update_time = now

        if is_present and attention_level >= self.attention_threshold:
            self.currently_working = True
            self.total_work_seconds += delta
        else:
            self.currently_working = False

    def get_total_work_seconds(self) -> int:
        return int(self.total_work_seconds)
