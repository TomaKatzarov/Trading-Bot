import asyncio
import signal
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

class AlpacaStream:
    def __init__(self):
        self.client = StockDataStream(
            api_key="PKC2IAM5P53NSEKM3T3F",
            secret_key="saQAAafnt3dtuuJQtEmZ8NpIUohrJ6YTLgP6KXSB",
            feed=DataFeed.IEX
        )
        self.running = False

    async def _handler(self, bar):
        print(f"\nReceived {bar.symbol}:")
        print(f"Time: {bar.timestamp}")
        print(f"Close: {bar.close}")
        print(f"Volume: {bar.volume}")

    async def run(self):
        self.client.subscribe_bars(self._handler, 'IWM')
        self.running = True
        await self.client._run_forever()  # Directly call internal runner

    async def stop(self):
        if self.running:
            await self.client.stop()
            self.running = False

async def main():
    stream = AlpacaStream()
    try:
        await stream.run()
    except KeyboardInterrupt:
        await stream.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStream stopped successfully")