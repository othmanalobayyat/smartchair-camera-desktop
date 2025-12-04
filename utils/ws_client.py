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
        self.last_fail_time = 0

    def _try_connect(self, url):
        try:
            self.ws = create_connection(url, timeout=5)
            self.connected = True
            self.active_url = url
            self.logger.info(f"✅ Connected to {url}")
            return True
        except Exception as e:
            self.logger.warning(
                f"❌ Failed to connect to {url}: {e}"
            )
            return False

    def connect(self):
        with self._lock:
            self.connected = False

            # 1️⃣ حاول السيرفر المحلي
            if self._try_connect(self.primary_url):
                return

            # 2️⃣ لو فشل → حوّل تلقائياً للـ Railway
            self.logger.warning(
                "⬇️ Switching to BACKUP server"
            )
            self._try_connect(self.backup_url)

    def send_json(self, data: dict):
        if not self.connected:
            self.connect()

        if not self.connected:
            return

        try:
            self.ws.send(json.dumps(data))

        except (
            WebSocketConnectionClosedException,
            BrokenPipeError,
            ConnectionResetError,
        ):
            self.logger.warning(
                "🔄 Connection lost, retrying..."
            )
            self.connected = False
            time.sleep(1)
            self.connect()

        except Exception as e:
            self.logger.error(
                f"Send error: {e}"
            )
            self.connected = False

    def close(self):
        with self._lock:
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass

        self.connected = False
        self.logger.info("WebSocket closed")
