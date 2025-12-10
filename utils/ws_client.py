import json
import threading
import time
import queue

from websocket import (
    create_connection,
    WebSocketConnectionClosedException,
)


class WSClient:
    """
    WSClient v2
    - الاتصال وإعادة الاتصال يتمان في Thread منفصل
    - إرسال البيانات من خلال Queue غير حاجبة للـ main thread
    - دعم primary / backup server
    - keep-alive + إعادة اتصال تلقائية
    """

    def __init__(self, primary_url, backup_url, logger):
        self.primary_url = primary_url
        self.backup_url = backup_url
        self.active_url = primary_url

        self.logger = logger

        self.ws = None
        self.connected = False

        self._lock = threading.Lock()

        # Queue للرسائل (Strings جاهزة للإرسال)
        self._queue: "queue.Queue[str]" = queue.Queue()

        # Thread التحكم في الاتصال والإرسال
        self._worker_thread = None
        self._stop_event = threading.Event()

        # keep-alive
        self.last_ping_time = 0.0
        self.ping_interval = 20  # ثواني

        # إعادة الاتصال (backoff بسيط)
        self._base_reconnect_delay = 1.0
        self._max_reconnect_delay = 10.0

        # تشغيل worker من البداية
        self._start_worker()

    # ============================
    # Worker management
    # ============================
    def _start_worker(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="WSClientWorker",
        )
        self._worker_thread.start()

    def _worker_loop(self):
        reconnect_delay = self._base_reconnect_delay

        while not self._stop_event.is_set():
            # 1) إذا لسنا متصلين → حاول الاتصال
            if not self.connected:
                if not self._connect_sequence():
                    # فشل الاتصال بكل السيرفرات → انتظر ثم حاول ثانية
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(
                        self._max_reconnect_delay,
                        reconnect_delay * 2,
                    )
                    continue
                else:
                    # نجاح الاتصال → إعادة تعيين التأخير
                    reconnect_delay = self._base_reconnect_delay

            # 2) متصلين → حاول إرسال رسالة من الـ Queue أو عمل keep-alive
            try:
                # نستخدم timeout صغير حتى نتمكن من تنفيذ keep-alive
                try:
                    msg = self._queue.get(timeout=0.1)
                except queue.Empty:
                    msg = None

                # keep-alive
                self._keep_alive()

                # لا توجد رسالة لإرسالها حاليا
                if msg is None:
                    continue

                # إذا وصلنا هنا، هناك رسالة جاهزة للإرسال
                self.ws.send(msg)

            except (
                WebSocketConnectionClosedException,
                BrokenPipeError,
                ConnectionResetError,
            ) as e:
                self.logger.warning(f"🔄 Connection lost: {e}")
                self._safe_close()
                self.connected = False

            except Exception as e:
                self.logger.error(f"WS worker error: {e}")
                self._safe_close()
                self.connected = False

    # ============================
    # Connection helpers
    # ============================
    def _try_connect(self, url: str) -> bool:
        try:
            self.logger.info(f"🌐 Trying WebSocket connect: {url}")
            ws = create_connection(
                url,
                timeout=5,
                enable_multithread=True,
            )
            with self._lock:
                # في حال أُغلِق الاتصال السابق داخل worker
                if self.ws is not None:
                    try:
                        self.ws.close()
                    except Exception:
                        pass

                self.ws = ws
                self.connected = True
                self.active_url = url
                self.last_ping_time = time.time()

            self.logger.info(f"✅ Connected to {url}")
            return True

        except Exception as e:
            self.logger.warning(f"❌ Failed to connect to {url}: {e}")
            return False

    def _connect_sequence(self) -> bool:
        """
        يحاول الاتصال أولاً بالـ primary
        ثم بالـ backup إذا فشل.
        """
        # primary
        if self._try_connect(self.primary_url):
            return True

        # backup
        self.logger.warning("⬇️ Switching to BACKUP server")
        if self._try_connect(self.backup_url):
            return True

        return False

    def _keep_alive(self):
        """
        إرسال ping كل ping_interval ثانية تقريباً،
        حتى لا يغلق السيرفر الاتصال لعدم النشاط.
        """
        if not self.connected or self.ws is None:
            return

        now = time.time()
        if now - self.last_ping_time >= self.ping_interval:
            try:
                self.ws.ping()
                self.last_ping_time = now
            except Exception as e:
                self.logger.warning(f"Ping failed: {e}")
                self._safe_close()
                self.connected = False

    def _safe_close(self):
        with self._lock:
            if self.ws is not None:
                try:
                    self.ws.close()
                except Exception:
                    pass
                self.ws = None

    # ============================
    # Public API
    # ============================
    def connect(self):
        """
        للإبقاء على التوافق مع الكود القديم.
        الآن الـ worker هو الذي يتحكم في الاتصال،
        وهذه الدالة فقط تتأكد أن الـ worker يعمل.
        """
        self._start_worker()

    def send_json(self, data: dict):
        """
        لا ترسل مباشرة عبر الـ socket،
        فقط تضيف الرسالة إلى الـ Queue وترجع فوراً
        حتى لا تحجب الخيط الرئيسي.
        """
        try:
            msg = json.dumps(data)
        except Exception as e:
            self.logger.error(f"JSON encode error: {e}")
            return

        # تأكد أن الـ worker يعمل
        self._start_worker()

        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            # Queue غير محدودة افتراضياً، لكن نضع هذا للحماية
            self.logger.warning("WS send queue is full, dropping message")

    def close(self):
        """
        إيقاف الـ worker وإغلاق الـ WebSocket.
        """
        self._stop_event.set()

        # إضافة رسالة فارغة لإخراج الـ worker من الـ get(timeout)
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        self._safe_close()
        self.connected = False
        self.logger.info("WebSocket closed")
