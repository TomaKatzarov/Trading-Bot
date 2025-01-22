# test_realtime.py
import asyncio
from core.realtime_handler import RealTimeDataHandler

async def main():
    print("Testing real-time data feed...")
    handler = RealTimeDataHandler()
    
    
    async def print_data(data):
        print(f"\nNew Bar @ {data['timestamp']}")
        print(f"Close: {data['close']} Volume: {data['volume']}")
    
    # Subscribe to bars and start the stream within the same event loop
    handler.stream.subscribe_bars(print_data, 'NDAQ')
    stream_task = asyncio.create_task(handler.stream._run_forever())
    await asyncio.gather(stream_task)  # Directly await the coroutine

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped real-time feed")