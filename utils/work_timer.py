# utils/work_timer.py
import time


class WorkTimer:
    """
    WorkTimer v2 — نظام جلسات عمل احترافي
    يتتبع:
        - بداية الجلسة
        - نهاية الجلسة
        - المدة الحالية
        - مجموع وقت العمل
        - مجموع وقت الاستراحات
        - حالة النظام: WORK / BREAK / IDLE
    """

    def __init__(self, attention_threshold: int):
        self.attention_threshold = attention_threshold

        # الزمن
        self.total_work_seconds = 0.0
        self.total_break_seconds = 0.0

        # الجلسة الحالية
        self.session_start = None
        self.session_active = False

        # تتبع الوقت بين التحديثات
        self._last_update_time = time.time()

        # حالة النظام
        self.state = "IDLE"  # WORK / BREAK / IDLE

    def update(self, is_present: bool, attention_level: float):
        now = time.time()
        delta = now - self._last_update_time
        self._last_update_time = now

        # ==========================
        # 1) لا يوجد مستخدم → BREAK
        # ==========================
        if not is_present:
            self.state = "BREAK"
            self.session_active = False
            self.session_start = None
            self.total_break_seconds += delta
            return

        # ==========================
        # 2) المستخدم موجود لكن غير مركز
        # ==========================
        if attention_level < self.attention_threshold:
            self.state = "BREAK"
            self.session_active = False
            self.session_start = None
            self.total_break_seconds += delta
            return

        # ==========================
        # 3) المستخدم مركز → WORK
        # ==========================
        self.state = "WORK"

        # بداية جلسة جديدة
        if not self.session_active:
            self.session_start = now
            self.session_active = True

        # تراكم وقت العمل
        self.total_work_seconds += delta

    def get_current_session_duration(self) -> int:
        if not self.session_active or self.session_start is None:
            return 0

        return int(time.time() - self.session_start)

    def get_total_work_seconds(self) -> int:
        return int(self.total_work_seconds)

    def get_total_break_seconds(self) -> int:
        return int(self.total_break_seconds)
