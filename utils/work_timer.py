# utils/work_timer.py
import time


class WorkTimer:
    """
    WorkTimer v3 — نظام جلسات عمل صحيح هندسيًا
    ✔️ يعد فقط وقت العمل الفعلي
    ✔️ يتوقف عند الغياب أو الشرود
    ✔️ يكمل بعد العودة
    ✔️ لا يعد وقت الاستراحة
    """

    def __init__(self, attention_threshold: int):
        self.attention_threshold = attention_threshold

        # ==========================
        # TOTAL TIME
        # ==========================
        self.total_work_seconds = 0.0
        self.total_break_seconds = 0.0

        # ==========================
        # INTERNAL TIMING
        # ==========================
        self._last_update_time = time.time()

        # ==========================
        # STATE
        # ==========================
        self.state = "IDLE"  # WORK / BREAK / IDLE

    # ============================================================
    # UPDATE LOOP
    # ============================================================
    def update(self, is_present: bool, attention_level: float):
        now = time.time()
        delta = now - self._last_update_time
        self._last_update_time = now

        # ==========================
        # 1) غير موجود → BREAK
        # ==========================
        if not is_present:
            self.state = "BREAK"
            self.total_break_seconds += delta
            return

        # ==========================
        # 2) موجود لكن غير مركز → BREAK
        # ==========================
        if attention_level < self.attention_threshold:
            self.state = "BREAK"
            self.total_break_seconds += delta
            return

        # ==========================
        # 3) مركز → WORK
        # ==========================
        self.state = "WORK"
        self.total_work_seconds += delta

    # ============================================================
    # GETTERS
    # ============================================================
    def get_current_session_duration(self) -> int:
        """
        مدة الجلسة = وقت العمل الفعلي فقط
        """
        return int(self.total_work_seconds)

    def get_total_work_seconds(self) -> int:
        return int(self.total_work_seconds)

    def get_total_break_seconds(self) -> int:
        return int(self.total_break_seconds)

    def get_state(self) -> str:
        return self.state
