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
    WSClient (STABLE VERSION)
    - إرسال فقط (no recv)
    - Thread واحد فقط
    - keep-alive آمن
    - بدون camera_control من السيرفر
    """

    def __init__(self, primary_url, backup_url, logger):
        self.primary_url = primary_url
        self.backup_url = backup_url
        self.active_url = primary_url

        self.logger = logger

        self.ws = None
        self.connected = False

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Queue للإرسال فقط
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=50)

        self._worker_thread = None

        # keep-alive
        self.last_ping_time = 0.0
        self.ping_interval = 20  # seconds

        # reconnect backoff
        self._base_reconnect_delay = 1.0
        self._max_reconnect_delay = 10.0

        self._start_worker()

    # ============================
    # Worker
    # ============================
    def _start_worker(self):
        if self._worker_thread and self._worker_thread.is_alive():
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
            if not self.connected:
                if not self._connect_sequence():
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(self._max_reconnect_delay, reconnect_delay * 2)
                    continue
                reconnect_delay = self._base_reconnect_delay

            try:
                # send queued messages
                try:
                    msg = self._queue.get(timeout=0.1)
                except queue.Empty:
                    msg = None

                self._keep_alive()

                if msg:
                    self.ws.send(msg)

            except (
                WebSocketConnectionClosedException,
                BrokenPipeError,
                ConnectionResetError,
            ) as e:
                self.logger.warning(f"🔄 Connection lost: {e}")
                self._safe_close()
                self.connected = False

            except Exception:
                self.logger.exception("WS worker error")
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
                enable_multithread=False,  # 🔴 مهم
            )

            with self._lock:
                if self.ws:
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
        if self._try_connect(self.primary_url):
            return True

        self.logger.warning("⬇️ Switching to BACKUP server")
        return self._try_connect(self.backup_url)

    def _keep_alive(self):
        if not self.connected or not self.ws:
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
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass
                self.ws = None

    # ============================
    # Public API
    # ============================
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
            self.logger.warning("WS queue full, dropping message")

    def close(self):
        self._stop_event.set()
        self._safe_close()
        self.connected = False
        self.logger.info("WebSocket closed")
