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
    - استقبال أوامر camera_control من الموبايل (start / stop)
    """

    def __init__(self, primary_url, backup_url, logger):
        self.primary_url = primary_url
        self.backup_url = backup_url
        self.active_url = primary_url

        self.logger = logger

        # =========================
        # Camera control from mobile
        # =========================
        self.camera_enabled = True

        self.ws = None
        self.connected = False

        self._lock = threading.Lock()

        # Queue للرسائل (Strings جاهزة للإرسال)
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=10)

        # Thread التحكم في الاتصال والإرسال
        self._worker_thread = None

        # Thread استقبال الرسائل
        self._receiver_thread = None

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
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(
                        self._max_reconnect_delay,
                        reconnect_delay * 2,
                    )
                    continue
                else:
                    reconnect_delay = self._base_reconnect_delay

            # 2) متصلين → إرسال / keep-alive
            try:
                try:
                    msg = self._queue.get(timeout=0.1)
                except queue.Empty:
                    msg = None

                self._keep_alive()

                if msg is None:
                    continue

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
                self.logger.exception("WS worker error")
                self._safe_close()
                self.connected = False

    # ============================
    # Receiver loop (NEW)
    # ============================
    def _start_receiver(self):
        if self._receiver_thread is not None and self._receiver_thread.is_alive():
            return

        self._receiver_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
            name="WSClientReceiver",
        )
        self._receiver_thread.start()

    def _receive_loop(self):
        while not self._stop_event.is_set():
            if not self.connected or self.ws is None:
                time.sleep(0.2)
                continue

            try:
                msg = self.ws.recv()
                if not msg:
                    continue

                data = json.loads(msg)

                # ==========================
                # CAMERA CONTROL FROM MOBILE
                # ==========================
                if data.get("type") == "camera_control":
                    action = data.get("action")

                    if action == "stop":
                        self.camera_enabled = False
                        self.logger.info("⏹ Camera disabled from mobile")

                    elif action == "start":
                        self.camera_enabled = True
                        self.logger.info("▶️ Camera enabled from mobile")

            except WebSocketConnectionClosedException:
                self.connected = False

            except Exception as e:
                self.logger.warning(f"WS recv error: {e}")

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

            # تشغيل receiver بعد الاتصال
            self._start_receiver()

            return True

        except Exception as e:
            self.logger.warning(f"❌ Failed to connect to {url}: {e}")
            return False

    def _connect_sequence(self) -> bool:
        if self._try_connect(self.primary_url):
            return True

        self.logger.warning("⬇️ Switching to BACKUP server")
        if self._try_connect(self.backup_url):
            return True

        return False

    def _keep_alive(self):
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
        self._start_worker()

    def send_json(self, data: dict):
        try:
            msg = json.dumps(data)
        except Exception as e:
            self.logger.error(f"JSON encode error: {e}")
            return

        self._start_worker()

        try:
            if not self._queue.full():
                self._queue.put_nowait(msg)
        except queue.Full:
            self.logger.warning("WS send queue is full, dropping message")

    def close(self):
        self._stop_event.set()

        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

        self._safe_close()
        self.connected = False
        self.logger.info("WebSocket closed")
