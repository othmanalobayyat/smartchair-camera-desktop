import json
import threading
import time

from websocket import (
    create_connection,
    WebSocketConnectionClosedException,
)


class WSClient:
    def __init__(self, url: str, logger):
        self.url = url
        self.logger = logger
        self.ws = None
        self.connected = False
        self._lock = threading.Lock()

    def connect(self):
        with self._lock:
            try:
                self.ws = create_connection(self.url)
                self.connected = True
                self.logger.info(
                    f"Connected to WebSocket: {self.url}"
                )
            except Exception as e:
                self.logger.error(
                    f"WebSocket connection failed: {e}"
                )
                self.connected = False

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
        ):
            self.logger.warning(
                "WebSocket closed, reconnecting..."
            )
            self.connected = False
            time.sleep(1)
            self.connect()

        except Exception as e:
            self.logger.error(
                f"Error sending data: {e}"
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
            self.logger.info(
                "WebSocket connection closed"
            )
