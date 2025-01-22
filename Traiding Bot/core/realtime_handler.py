import os
import asyncio
import logging
import pandas as pd
from datetime import datetime, time
from collections import deque
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from dotenv import load_dotenv
import pytz

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class RealTimeDataHandler:
    def __init__(self, symbol: str = 'IWM', buffer_size: int = 100):
        """Initialize real-time data handler with configurable buffer"""
        self._load_credentials()
        self.symbol = symbol.upper()
        self.data_buffer = deque(maxlen=buffer_size)
        self.ws_client = None
        self.running = False
        self.valid_symbols = []
        
        self._init_clients()
        self._validate_symbol()

    def _load_credentials(self):
        """Load credentials from .env file"""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(root_dir, 'Credential.env')
        if not os.path.exists(env_path):
            raise FileNotFoundError(f"Credential file not found at {env_path}")
        load_dotenv(env_path)

    def _init_clients(self):
        """Initialize Alpaca clients with proper error handling"""
        try:
            # Initialize trading client for symbol validation
            trading_client = TradingClient(
                api_key=os.getenv('APCA_API_KEY_ID'),
                secret_key=os.getenv('APCA_API_SECRET_KEY'),
                paper=True
            )
            
            # Get available symbols
            request = GetAssetsRequest(
                status='active',
                asset_class=AssetClass.US_EQUITY
            )
            assets = trading_client.get_all_assets(request)
            self.valid_symbols = [a.symbol for a in assets if a.tradable]
            
            # Initialize streaming client
            self.ws_client = StockDataStream(
                api_key=os.getenv('APCA_API_KEY_ID'),
                secret_key=os.getenv('APCA_API_SECRET_KEY'),
                feed=DataFeed.IEX
            )
            
            logger.info(f"Found {len(self.valid_symbols)} tradable symbols")

        except Exception as e:
            logger.error(f"Client initialization failed: {str(e)}")
            raise

    def _validate_symbol(self):
        """Validate symbol against available symbols"""
        if self.symbol not in self.valid_symbols:
            logger.error(f"Symbol {self.symbol} not available. Valid symbols: {self.valid_symbols[:10]}...")
            raise ValueError(f"Invalid symbol: {self.symbol}")

    def _market_is_open(self) -> bool:
        """Check if market is open (IEX operational hours) with proper timezone handling"""
        try:
            ny_tz = pytz.timezone('America/New_York')
            ny_time = datetime.now(ny_tz)
            return (
                ny_time.weekday() < 5 and          # Monday-Friday
                time(8, 0) <= ny_time.time() < time(17, 0)  # 8 AM - 5 PM ET
            )
        except Exception as e:
            logger.error(f"Market hours check failed: {str(e)}")
            return False

    async def _process_bar(self, bar) -> dict:
        """Process incoming bar data with enhanced validation"""
        processed = {
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'returns': 0.0
        }
        
        if len(self.data_buffer) > 0:
            prev_close = self.data_buffer[-1]['close']
            processed['returns'] = (bar.close - prev_close) / prev_close
        
        self.data_buffer.append(processed)
        return processed

    async def _handler(self, bar):
        """Async bar handler with error protection"""
        try:
            if self._market_is_open():
                processed = await self._process_bar(bar)
                logger.info(f"New bar: {processed['timestamp']} | Close: {processed['close']}")
            else:
                logger.warning("Received data outside market hours")
        except Exception as e:
            logger.error(f"Bar processing error: {str(e)}")

    async def start(self):
        """Start real-time streaming with proper async management"""
        if not self.running:
            try:
                self.running = True
                self.ws_client.subscribe_bars(self._handler, self.symbol)
                logger.info(f"Started {self.symbol} stream")
                await self.ws_client._run_forever()  # Direct internal method
            except Exception as e:
                self.running = False
                logger.error(f"Stream error: {str(e)}")
                raise

    async def stop(self):
        """Stop streaming gracefully"""
        if self.running:
            try:
                await self.ws_client.stop()
                self.running = False
                logger.info("Stream stopped successfully")
            except Exception as e:
                logger.error(f"Stop error: {str(e)}")

    def get_recent_data(self, n: int = 10) -> list:
        """Get most recent n data points safely"""
        return list(self.data_buffer)[-n:] if self.data_buffer else []

    def get_dataframe(self) -> pd.DataFrame:
        """Convert buffer to DataFrame with proper indexing"""
        try:
            df = pd.DataFrame(list(self.data_buffer))
            return df.set_index('timestamp') if not df.empty else df
        except Exception as e:
            logger.error(f"DataFrame error: {str(e)}")
            return pd.DataFrame()