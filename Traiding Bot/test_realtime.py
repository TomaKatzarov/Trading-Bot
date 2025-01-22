import asyncio
import signal
import logging
import argparse
from datetime import datetime
from core.realtime_handler import RealTimeDataHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GracefulExiter:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.info("Received exit signal")
        self.should_exit = True

async def test_data_format(handler):
    """Test if received data matches expected format"""
    data = handler.get_recent_data(1)
    if not data:
        return False
    
    required_fields = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'vwap', 'Returns']
    return all(field in data[0] for field in required_fields)

async def live_test(symbol='SPY', test_duration=60):
    exiter = GracefulExiter()
    handler = RealTimeDataHandler(symbol=symbol)
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting {symbol} real-time feed...")
        await handler.start()
        
        while not exiter.should_exit:
            await asyncio.sleep(1)
            
            # Test data format
            if not await test_data_format(handler):
                logger.error("Data format validation failed")
                break
            
            # Print recent data sample
            recent = handler.get_recent_data(1)
            if recent:
                logger.info(
                    f"Latest: {recent[0]['timestamp']} | "
                    f"Close: ${recent[0]['Close']:.2f} | "
                    f"Volume: {recent[0]['Volume']}"
                )
            
            # Check test duration
            if (datetime.now() - start_time).seconds > test_duration:
                logger.info("Test duration complete")
                break
                
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False
    finally:
        logger.info("Shutting down stream...")
        await handler.stop()
        
        # Print test summary
        df = handler.get_dataframe()
        if df is not None:
            logger.info(f"Test Summary:")
            logger.info(f"Total data points: {len(df)}")
            logger.info(f"First timestamp: {df.index[0]}")
            logger.info(f"Last timestamp: {df.index[-1]}")
            return True
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Test real-time data streaming')
    parser.add_argument('--symbol', type=str, default='SPY', help='Stock symbol to test')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        success = asyncio.run(live_test(args.symbol, args.duration))
        exit_code = 0 if success else 1
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test terminated by user")
        exit(1)