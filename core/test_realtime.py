import os
import sys
import asyncio
from datetime import datetime
import logging
from dotenv import load_dotenv
from pathlib import Path

# Load .env with explicit path
env_path = Path(__file__).resolve().parent.parent / "Credential.env"
load_dotenv(dotenv_path=env_path, override=True)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from core package
from core.realtime_handler import RealTimeDataHandler

logger = logging.getLogger(__name__)

async def live_test(symbol='AAPL', test_duration=60):
    """Test real-time data streaming"""
    handler = None
    try:
        handler = RealTimeDataHandler(symbol=symbol)
        logger.info(f"Starting {symbol} stream...")
        
        start_time = datetime.now()
        await handler.start()
        
        while (datetime.now() - start_time).total_seconds() < test_duration:
            await asyncio.sleep(1)
            recent = handler.get_recent_data(1)
            if recent:
                logger.info(f"Latest: {recent[0]['timestamp']} | Close: ${recent[0]['close']:.2f}")
                
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return False
    finally:
        if handler:
            await handler.stop()
    return True

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(live_test())