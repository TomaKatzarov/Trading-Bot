# hist_data_loader.py
import os
import logging
import argparse
from datetime import datetime, UTC, timedelta
from typing import Optional
import pandas as pd
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("Credential.env")

class HistoricalDataLoader:
    def __init__(self):
        self.api = self._init_alpaca_client()
        self.valid_timeframes = {
            'minute': '1Min', '5min': '5Min', '15min': '15Min',
            'hour': '1Hour', 'day': '1Day', 'week': '1Week'
        }

    def _init_alpaca_client(self) -> tradeapi.REST:
        """Initialize and validate Alpaca API client"""
        api_key = os.getenv('APCA_API_KEY_ID')
        secret_key = os.getenv('APCA_API_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError("Missing Alpaca API credentials in .env file")
            
        return tradeapi.REST(
            api_key,
            secret_key,
            base_url='https://paper-api.alpaca.markets',
            api_version='v2'
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((tradeapi.rest.APIError, ConnectionError)),
        reraise=True
    )
    def _fetch_alpaca_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Fetch data from Alpaca API with proper time formatting"""
        try:
            # Convert to Alpaca-compatible format
            start_str = start.isoformat(timespec='seconds').replace('+00:00', 'Z')
            end_str = end.isoformat(timespec='seconds').replace('+00:00', 'Z')
            
            logger.info(f"Fetching {timeframe} data for {symbol} from {start_str} to {end_str}")
            
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start_str,
                end=end_str,
                adjustment='all',
                feed='iex'
            ).df
            
            if bars.empty:
                logger.warning(f"No data returned for {symbol} in {timeframe} timeframe")
                
            return bars

        except tradeapi.rest.APIError as e:
            logger.error(f"Alpaca API Error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data fetch: {str(e)}")
            raise

    def _validate_dates(self, start: datetime, end: datetime):
        """Ensure dates meet Alpaca requirements"""
        now = datetime.now(UTC)
        
        # Free tier data delay check
        if end > now - timedelta(minutes=15):
            logger.warning("Adjusting end date for free tier data delay")
            end = now - timedelta(minutes=15)
            
        if start > end:
            raise ValueError(f"Start date {start} cannot be after end date {end}")
            
        max_lookback = timedelta(days=365*2)
        if (end - start) > max_lookback:
            raise ValueError(f"Lookback period exceeds {max_lookback.days} days limit")
            
        return start, end

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate the integrity of fetched data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not isinstance(df, pd.DataFrame):
            logger.error("Returned data is not a DataFrame")
            return False
            
        if df.empty:
            logger.warning(f"No data available for {symbol}")
            return False
            
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in data: {df.columns}")
            return False
            
        if df.isnull().values.any():
            logger.warning("Data contains missing values - filling forward")
            df.ffill(inplace=True)
            
        return True

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and enhance the raw data"""
        try:
            # Convert timezone and rename columns
            df = df.tz_convert('America/New_York')
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Add additional features
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            df['Returns'] = df['Close'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise

    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save data to Parquet with partitioning"""
        try:
            filename = f"data/{symbol.replace('=', '_')}_{timeframe}_historical.parquet"
            
            #Create directory if it doesn't exist # type: ignore
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Check if file exists to append
            if os.path.exists(filename):
                existing_df = pd.read_parquet(filename)
                combined_df = pd.concat([existing_df, df]).drop_duplicates()
            else:
                combined_df = df
                
            combined_df.to_parquet(
                filename,
                engine='pyarrow',
                compression='snappy',
                index=True
            )
            logger.info(f"Successfully saved data to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            raise

    def load_historical_data(
        self,
        symbol: str = 'NDAQ',
        timeframe: str = '1Hour',
        lookback_days: int = 30,
        verbose: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Main method to load and process historical data
        """
        try:
            # Validate inputs
            if timeframe not in self.valid_timeframes:
                raise ValueError(f"Invalid timeframe. Use one of {list(self.valid_timeframes.keys())}")
                
            alpaca_timeframe = self.valid_timeframes[timeframe]
            
            # Create timezone-aware datetimes
            end_date = datetime.now(UTC).replace(microsecond=0)
            start_date = end_date - timedelta(days=lookback_days)
            
            # Adjust dates for API requirements
            start_date, end_date = self._validate_dates(start_date, end_date)
            
            # Fetch data
            raw_data = self._fetch_alpaca_data(symbol, alpaca_timeframe, start_date, end_date)
            
            # Validate and process
            if not self._validate_data(raw_data, symbol):
                return None
                
            processed_data = self._process_data(raw_data)
            
            # Save and return
            self._save_data(processed_data, symbol, alpaca_timeframe)
            
            if verbose:
                logger.info(f"\nData Summary:\n{processed_data.describe()}")
                logger.info(f"\nLatest 5 records:\n{processed_data.tail()}")
                
            return processed_data

        except Exception as e:
            logger.error(f"Failed to load historical data: {str(e)}")
            return None

if __name__ == "__main__":
    loader = HistoricalDataLoader()
    
    parser = argparse.ArgumentParser(description='Historical Data Loader')
    parser.add_argument('--symbol', type=str, default='NDAQ', help='Stock symbol')
    parser.add_argument('--timeframe', type=str, default='hour', help='Timeframe (minute, 5min, 15min, hour, day, week)')
    parser.add_argument('--lookback', type=int, default=3, help='Lookback period in days')
    
    args = parser.parse_args()
    
    data = loader.load_historical_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback,
        verbose=True
    )
    
    if data is not None:
        print("\nFirst 3 rows of loaded data:")
        print(data.head(3))