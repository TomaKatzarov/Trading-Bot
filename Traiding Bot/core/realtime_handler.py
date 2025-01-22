import os
import asyncio
import pandas as pd
from datetime import datetime
from collections import deque
from alpaca.data.live import StockDataStream
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RealTimeDataHandler:
    def __init__(self, symbol: str = 'NDAQ', buffer_size: int = 100):
        """Initialize real-time data handler with configurable buffer"""
        self.symbol = symbol
        self.data_buffer = deque(maxlen=buffer_size)
        self.ws_client = None
        self.running = False
        
        # Initialize connection
        self._init_client()
        
        # Get valid symbols for validation
        self.valid_symbols = self._get_available_symbols()

    def _init_client(self):
        """Initialize Alpaca websocket client"""
        try:
            self.ws_client = StockDataStream(
                api_key=os.getenv('PKC2IAM5P53NSEKM3T3F'),
                secret_key=os.getenv('saQAAafnt3dtuuJQtEmZ8NpIUohrJ6YTLgP6KXSB'),
                feed='iex'  # Using IEX feed for free tier
            )
            logger.info(f"Initialized websocket client for {self.symbol}")
        except Exception as e:
            logger.error(f"Failed to initialize client: {str(e)}")
            raise

    def _get_available_symbols(self) -> List[str]:
        """Get list of available symbols from Alpaca"""
        try:
            # This would need to be implemented using Alpaca REST API
            # For now returning a test list
            return ['NDAQ', 'SPY', 'IWM', 'QQQ']
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {str(e)}")
            return []

    async def _process_bar(self, bar) -> Dict:
        """Process incoming bar data to match historical format"""
        processed = {
            'timestamp': bar.timestamp,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume,
            'vwap': (bar.high + bar.low + bar.close) / 3,  # Simplified VWAP
            'VWAP': None,  # Will be calculated in batches
            'Returns': None  # Will be calculated in batches
        }
        
        self.data_buffer.append(processed)
        
        # Calculate rolling metrics when we have enough data
        if len(self.data_buffer) > 1:
            prev_close = self.data_buffer[-2]['Close']
            processed['Returns'] = (bar.close - prev_close) / prev_close
        
        return processed

    async def _handler(self, bar):
        """Handle incoming bar data"""
        try:
            processed = await self._process_bar(bar)
            logger.debug(f"Processed bar: {processed['timestamp']} - ${processed['Close']}")
        except Exception as e:
            logger.error(f"Error processing bar: {str(e)}")

    async def start(self):
        """Start the data stream"""
        if not self.running:
            try:
                self.running = True
                self.ws_client.subscribe_bars(self._handler, self.symbol)
                logger.info(f"Started streaming {self.symbol}")
                await self.ws_client.run()
            except Exception as e:
                self.running = False
                logger.error(f"Stream error: {str(e)}")
                raise

    async def stop(self):
        """Stop the data stream"""
        if self.running:
            try:
                await self.ws_client.close()
                self.running = False
                logger.info("Stream closed successfully")
            except Exception as e:
                logger.error(f"Error closing stream: {str(e)}")

    def get_recent_data(self, n: int = 10) -> Optional[List[Dict]]:
        """Get most recent n data points"""
        try:
            return list(self.data_buffer)[-n:]
        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            return None

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Convert buffer to pandas DataFrame"""
        try:
            df = pd.DataFrame(list(self.data_buffer))
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                return df
            return None
        except Exception as e:
            logger.error(f"Error converting to DataFrame: {str(e)}")
            return None