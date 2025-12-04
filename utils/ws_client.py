import json
import threading
import time
from websocket import (
    create_connection,
    WebSocketConnectionClosedException,
)


class WSClient:
    def __init__(self, primary_url, backup_url, logger):
        self.primary_url = primary_url
        self.backup_url = backup_url
        self.active_url = primary_url

        self.logger = logger
        self.ws = None
        self.connected = False
        self._lock = threading.Lock()

        # ✅ keep-alive
        self.last_ping_time = 0
        self.ping_interval = 20  # seconds

    # ============================
    # Try connecting to server
    # ============================
    def _try_connect(self, url):
        try:
            self.ws = create_connection(
                url,
                timeout=5,
                enable_multithread=True,
            )
            self.connected = True
            self.active_url = url
            self.last_ping_time = time.time()

            self.logger.info(f"✅ Connected to {url}")
            return True

        except Exception as e:
            self.logger.warning(
                f"❌ Failed to connect to {url}: {e}"
            )
            return False

    # ============================
    # Connect logic (Primary -> Backup)
    # ============================
    def connect(self):
        with self._lock:
            self.connected = False

            # 1️⃣ Try LOCAL first
            if self._try_connect(self.primary_url):
                return

            # 2️⃣ Fallback to Railway
            self.logger.warning("⬇️ Switching to BACKUP server")
            self._try_connect(self.backup_url)

    # ============================
    # Keep WebSocket alive (Railway needs this)
    # ============================
    def _keep_alive(self):
        try:
            now = time.time()
            if now - self.last_ping_time >= self.ping_interval:
                self.ws.ping()
                self.last_ping_time = now
        except Exception:
            self.connected = False

    # ============================
    # Send JSON data
    # ============================
    def send_json(self, data: dict):
        if not self.connected:
            self.connect()

        if not self.connected:
            return

        try:
            self._keep_alive()
            self.ws.send(json.dumps(data))

        except (
            WebSocketConnectionClosedException,
            BrokenPipeError,
            ConnectionResetError,
        ) as e:
            self.logger.warning(f"🔄 Connection lost: {e}")
            self.connected = False
            time.sleep(1)
            self.connect()

        except Exception as e:
            self.logger.error(f"Send error: {e}")
            self.connected = False

    # ============================
    # Close connection
    # ============================
    def close(self):
        with self._lock:
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass

        self.connected = False
        self.logger.info("WebSocket closed")
