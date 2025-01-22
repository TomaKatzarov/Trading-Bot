# core/realtime_handler.py
import os
import asyncio
from alpaca_trade_api.stream import Stream
from dotenv import load_dotenv

load_dotenv("Credential.env")

class RealTimeDataHandler:
    def __init__(self, symbol='NDAQ'):
        self.symbol = symbol
        self.stream = Stream(
            key_id=os.getenv('APCA_API_KEY_ID'),
            secret_key=os.getenv('APCA_API_SECRET_KEY'),
            base_url='https://paper-api.alpaca.markets',
            data_feed='iex'  # Required for paper trading
        )
        self.data_window = []

    async def _on_bar(self, bar):
        """Handle incoming 1-minute bars"""
        processed = {
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        self.data_window = (self.data_window + [processed])[-100:]  # Keep 100 bars
        return processed

    async def start_stream(self, callback):
        """Start real-time data stream"""
        self.stream.subscribe_bars(self._on_bar, self.symbol)
        await self.stream.run()  # Changed to proper async run

    async def stop(self):
        """Stop streaming"""
        await self.stream.stop()