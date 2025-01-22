# data_fetcher_polygon.py
import websocket
import json
import os
from dotenv import load_dotenv

load_dotenv()

class PolygonWS:
    def __init__(self, symbol="C:F_NQ"):
        self.ws = websocket.WebSocketApp(
            "wss://socket.polygon.io/stocks",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.symbol = symbol

    def on_open(self, ws):
        print("Connection opened")
        auth = {"action": "auth", "params": os.getenv("COQcptRYuyqttD2HsE5ZCENk0gv4gvwf")}
        self.ws.send(json.dumps(auth))
        subscribe = {"action": "subscribe", "params": f"C.F.{self.symbol}"}
        self.ws.send(json.dumps(subscribe))

    def on_message(self, ws, message):
        data = json.loads(message)
        for result in data:
            if result.get("status") == "success":
                continue
            print(f"Trade update: {result}")

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("Connection closed")

    def run(self):
        self.ws.run_forever()

if __name__ == "__main__":
    ws = PolygonWS()
    ws.run()