# hist_data_loader.py
import os
import logging
import argparse
import sys
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
import pandas as pd
import alpaca_trade_api as tradeapi
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from tqdm import tqdm  # For progress bars

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load .env with explicit path
env_path = Path(__file__).resolve().parent.parent / "Credential.env"
load_dotenv(dotenv_path=env_path, override=True)

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

# --- Constants ---
# Alpaca free tier limit: 200 requests per minute
ALPACA_RATE_LIMIT_PER_MINUTE = 200
# Max symbols per single API call for get_bars
MAX_SYMBOLS_PER_BATCH = 100
# Calculate sleep time between batches to stay under the limit
# Add a small buffer (e.g., process 180 per minute instead of 200)
SAFE_RATE_LIMIT = ALPACA_RATE_LIMIT_PER_MINUTE * 0.9
CALLS_PER_SECOND = SAFE_RATE_LIMIT / 60.0
SLEEP_BETWEEN_CALLS = 1.0 / CALLS_PER_SECOND
# Add market hours constants
MARKET_OPEN_HOUR = 9  # 9:30 AM ET is standard market open, use 9 for hourly data
MARKET_CLOSE_HOUR = 16  # 4:00 PM ET is standard market close

class HistoricalDataLoader:
    def __init__(self):
        self.api = self._init_alpaca_client()
        self.valid_timeframes = {
            'minute': '1Min', '5min': '5Min', '15min': '15Min',
            'hour': '1Hour', 'day': '1Day', 'week': '1Week'
        }
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_root = self.project_root / "data" / "historical"
        self.last_api_call_time = 0  # For rate limiting

    def _init_alpaca_client(self) -> tradeapi.REST:
        """Initialize and validate Alpaca API client."""
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

    def load_symbols_config(self):
        """Load and validate symbols configuration."""
        config_path = os.path.join(os.path.dirname(__file__), '../config/symbols.json')
        try:
            with open(config_path) as f:
                data = json.load(f)
            # Convert each symbol to uppercase, stripping spaces
            for sector, syms in data.get('sectors', {}).items():
                data['sectors'][sector] = [s.strip().upper() for s in syms]
            # Also normalize ETF, index, and crypto symbols
            for category, syms in data.get('etfs', {}).items():
                data['etfs'][category] = [s.strip().upper() for s in syms]
            for index_name, syms in data.get('indices', {}).items():
                data['indices'][index_name] = [s.strip().upper() for s in syms]
            for category, syms in data.get('crypto', {}).items():
                data['crypto'][category] = [s.strip().upper() for s in syms]
            return data
        except FileNotFoundError:
            raise ValueError(f"Symbols config file not found at {config_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in symbols config file")

    def _rate_limit_control(self):
        """Ensures calls do not exceed the rate limit."""
        now = time.monotonic()
        elapsed = now - self.last_api_call_time
        if elapsed < SLEEP_BETWEEN_CALLS:
            sleep_time = SLEEP_BETWEEN_CALLS - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        self.last_api_call_time = time.monotonic()

    def _get_latest_timestamp(self, data_path: str) -> Optional[datetime]:
        """Get the timestamp of the last record in the most recent Parquet file."""
        try:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            # Assuming only one file 'data.parquet' per symbol/timeframe now
            latest_file = os.path.join(data_path, 'data.parquet')
            if not os.path.exists(latest_file):
                return None

            df = pd.read_parquet(latest_file)
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                # Ensure the timestamp is timezone-aware (UTC) for comparison
                last_ts = df.index[-1].to_pydatetime()
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)  # Assume UTC if naive
                else:
                    last_ts = last_ts.astimezone(timezone.utc)
                return last_ts
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get latest timestamp from {data_path}: {str(e)}")
            return None

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((tradeapi.rest.APIError, ConnectionError)),
        reraise=True
    )
    def _fetch_alpaca_data_batch(self, symbols: List[str], timeframe: str, start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for a batch of symbols from Alpaca API.
        Handles rate limiting before making the call.
        Returns a dictionary {symbol: DataFrame}.
        """
        self._rate_limit_control()  # Apply rate limiting before the API call

        try:
            start_str = start.isoformat(timespec='seconds').replace('+00:00', 'Z')
            end_str = end.isoformat(timespec='seconds').replace('+00:00', 'Z')
            logger.info(f"Fetching {timeframe} data for {len(symbols)} symbols ({symbols[0]}...) from {start_str} to {end_str}")

            # Use the multi-symbol capability
            bars_response = self.api.get_bars(
                symbols,
                timeframe,
                start=start_str,
                end=end_str,
                adjustment='all'
            )
            
            # Initialize results dictionary
            processed_bars = {}
            
            # For the newer Alpaca API version, directly use the bars_response object
            # which has proper parsing methods
            if hasattr(bars_response, 'df'):
                logger.info(f"Using bars_response.df attribute for data extraction")
                
                # Check if we have any data
                if bars_response.df.empty:
                    logger.warning(f"API returned empty DataFrame. Likely no data available for requested period.")
                    for symbol in symbols:
                        processed_bars[symbol] = pd.DataFrame()
                    return processed_bars
                
                # Check the type of index we have and process accordingly
                if isinstance(bars_response.df.index, pd.MultiIndex):
                    logger.info(f"Processing MultiIndex DataFrame with {len(bars_response.df)} rows")
                    
                    # Expected structure where symbol is in the index
                    if 'symbol' in bars_response.df.index.names:
                        # Get data for each requested symbol from the MultiIndex
                        for symbol in symbols:
                            try:
                                # Use .xs() to extract data for this symbol
                                symbol_data = bars_response.df.xs(symbol, level='symbol')
                                if not symbol_data.empty:
                                    processed_bars[symbol] = symbol_data
                                    logger.info(f"Extracted {len(symbol_data)} bars for {symbol}")
                                else:
                                    logger.warning(f"No data found for {symbol} in MultiIndex")
                                    processed_bars[symbol] = pd.DataFrame()
                            except KeyError:
                                logger.warning(f"Symbol {symbol} not found in MultiIndex")
                                processed_bars[symbol] = pd.DataFrame()
                    else:
                        logger.warning(f"MultiIndex does not contain 'symbol' level. Index names: {bars_response.df.index.names}")
                        # Fall back to empty DataFrames
                        for symbol in symbols:
                            processed_bars[symbol] = pd.DataFrame()
                
                # Handle DatetimeIndex case - this is what we're seeing in the logs
                elif isinstance(bars_response.df.index, pd.DatetimeIndex):
                    logger.info(f"Processing DatetimeIndex DataFrame with {len(bars_response.df)} rows")
                    
                    # Check if 'symbol' is a column
                    if 'symbol' in bars_response.df.columns:
                        # Group by symbol and process each group
                        symbol_groups = bars_response.df.groupby('symbol')
                        
                        # Extract data for each requested symbol
                        for symbol in symbols:
                            if symbol in symbol_groups.groups:
                                symbol_data = symbol_groups.get_group(symbol)
                                if not symbol_data.empty:
                                    # Remove the 'symbol' column as it's redundant
                                    if 'symbol' in symbol_data.columns:
                                        symbol_data = symbol_data.drop('symbol', axis=1)
                                    processed_bars[symbol] = symbol_data
                                    logger.info(f"Extracted {len(symbol_data)} bars for {symbol} from grouped data")
                                else:
                                    logger.warning(f"Empty group data for {symbol}")
                                    processed_bars[symbol] = pd.DataFrame()
                            else:
                                logger.warning(f"No data group found for {symbol}")
                                processed_bars[symbol] = pd.DataFrame()
                    else:
                        # If only one symbol was requested and symbol isn't a column, assume all data is for that symbol
                        if len(symbols) == 1:
                            processed_bars[symbols[0]] = bars_response.df
                            logger.info(f"Assuming all {len(bars_response.df)} rows are for single symbol {symbols[0]}")
                        else:
                            logger.warning(f"DataFrame has DatetimeIndex but no 'symbol' column with {len(symbols)} symbols requested")
                            for symbol in symbols:
                                processed_bars[symbol] = pd.DataFrame()
                else:
                    logger.warning(f"Unexpected index type: {type(bars_response.df.index)}")
                    # Fallback to empty DataFrames
                    for symbol in symbols:
                        processed_bars[symbol] = pd.DataFrame()
            
            # Use direct API for older versions or as fallback
            else:
                logger.warning(f"No 'df' attribute found in bars_response, using bars directly")
                # Older Alpaca API or different format - handle it by accessing bars directly
                for symbol in symbols:
                    try:
                        symbol_bars = bars_response.get(symbol, [])
                        if symbol_bars:
                            # Convert list of bars to DataFrame
                            symbol_df = pd.DataFrame([bar._asdict() for bar in symbol_bars])
                            if not symbol_df.empty and 't' in symbol_df.columns:
                                # Set the timestamp as index
                                symbol_df.set_index(pd.to_datetime(symbol_df['t']), inplace=True)
                                processed_bars[symbol] = symbol_df
                                logger.info(f"Processed {len(symbol_df)} bars for {symbol} using direct bar access")
                            else:
                                logger.warning(f"Invalid DataFrame structure for {symbol}")
                                processed_bars[symbol] = pd.DataFrame()
                        else:
                            logger.warning(f"No bars found for {symbol}")
                            processed_bars[symbol] = pd.DataFrame()
                    except Exception as bar_err:
                        logger.error(f"Error processing bars for {symbol}: {str(bar_err)}")
                        processed_bars[symbol] = pd.DataFrame()
            
            # Final check to ensure all requested symbols have an entry
            for symbol in symbols:
                if symbol not in processed_bars:
                    logger.warning(f"Symbol {symbol} missing from processed results, adding empty DataFrame")
                    processed_bars[symbol] = pd.DataFrame()
                elif not isinstance(processed_bars[symbol], pd.DataFrame):
                    logger.warning(f"Result for {symbol} is not a DataFrame ({type(processed_bars[symbol])}), replacing with empty DataFrame")
                    processed_bars[symbol] = pd.DataFrame()

            return processed_bars

        except tradeapi.rest.APIError as e:
            # Handle specific rate limit error (429)
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                 logger.warning(f"Rate limit exceeded (429). Retrying after delay...")
                 raise  # Re-raise for tenacity to catch
            else:
                 logger.error(f"Alpaca API Error fetching batch ({symbols[0]}...): {str(e)}")
                 raise  # Re-raise other API errors
        except Exception as e:
            logger.error(f"Unexpected error during batch data fetch ({symbols[0]}...): {str(e)}")
            raise  # Re-raise other exceptions

    def _validate_dates(self, start: datetime, end: datetime):
        """Ensure dates meet Alpaca requirements."""
        now = datetime.now(timezone.utc)
        
        # Adjust for free tier delay (15 minutes)
        if end > now - timedelta(minutes=15):
            logger.warning("Adjusting end date for free tier data delay")
            end = now - timedelta(minutes=15)
        
        # Ensure start is before end
        if start > end:
            logger.warning(f"Start date {start} is after end date {end}. Adjusting start date.")
            start = end - timedelta(hours=24)  # Default to last 24 hours in this case
        
        # Check for max lookback period
        max_lookback = timedelta(days=365*2)
        if (end - start) > max_lookback:
            logger.warning(f"Lookback period exceeds {max_lookback.days} days limit. Adjusting start date.")
            start = end - max_lookback
        
        # For hourly data, if we're spanning a weekend or holiday, log this information
        if (end - start).days >= 1:
            logger.info(f"Date range spans {(end - start).days} days. May include non-trading periods.")
        
        return start, end

    def _debug_check_data_files(self, symbol: str, timeframe_str: str):
        """Debug function to check what data exists in the parquet files."""
        data_path = self.data_root / symbol / timeframe_str
        filename = data_path / "data.parquet"
        
        if not os.path.exists(filename):
            logger.info(f"DEBUG: No existing file found for {symbol}")
            return
            
        try:
            df = pd.read_parquet(filename)
            if df.empty:
                logger.info(f"DEBUG: File exists for {symbol} but is empty")
                return
                
            # Check timestamps
            first_ts = df.index[0].to_pydatetime()
            last_ts = df.index[-1].to_pydatetime()
            row_count = len(df)
            
            logger.info(f"DEBUG: {symbol} data file contains {row_count} rows")
            logger.info(f"DEBUG: {symbol} first timestamp: {first_ts}")
            logger.info(f"DEBUG: {symbol} last timestamp: {last_ts}")
            
            # Check current system time vs last timestamp
            now = datetime.now(timezone.utc)
            time_diff = now - last_ts.astimezone(timezone.utc)
            logger.info(f"DEBUG: {symbol} last data is {time_diff.total_seconds()/3600:.2f} hours old")
            
        except Exception as e:
            logger.error(f"DEBUG: Error reading file for {symbol}: {e}")

    def _process_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Processes a DataFrame for a single symbol."""
        try:
            if df.empty:
                return df  # Return empty if nothing to process

            # Add symbol column before processing
            df.insert(0, "Symbol", symbol)

            # Ensure the DataFrame index is datetime and timezone-aware.
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                logger.warning(f"Index for {symbol} is not datetime; attempting to convert.")
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                # Alpaca usually returns tz-aware, but handle if not
                logger.warning(f"Index for {symbol} is not timezone-aware; localizing to UTC.")
                df.index = df.index.tz_localize('UTC')
            else:
                # Ensure it's UTC before converting
                df.index = df.index.tz_convert('UTC')

            # Convert index to Eastern Time.
            df = df.tz_convert('America/New_York')

            # Rename columns
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Fill NaNs before calculating features that depend on previous rows
            if df.isnull().values.any():
                logger.warning(f"Filling NaNs using ffill for {symbol} before feature calculation.")
                df.ffill(inplace=True)

            # Add additional features
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            df['Returns'] = df['Close'].pct_change()
        
            # New feature additions:
            df['HL_diff'] = df['High'] - df['Low']
            df['OHLC_avg'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

            # Drop rows that have any NaN (e.g., from initial pct_change)
            df = df.dropna()

            # Select only the desired 9 columns + Symbol.
            desired_columns = ["Symbol", "Open", "High", "Low", "Close", "Volume", "VWAP", "Returns", "HL_diff", "OHLC_avg"]
            # Filter only columns that exist
            cols_to_use = [col for col in desired_columns if col in df.columns]
            df = df[cols_to_use]

            return df

        except Exception as e:
            logger.error(f"Data processing failed for {symbol}: {str(e)}")
            # Return an empty DataFrame or raise, depending on desired behavior
            return pd.DataFrame()  # Return empty DF on processing error

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate the integrity of fetched data for a single symbol."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Data for {symbol} is not a DataFrame")
            return False
        if df.empty:
            # This is now handled upstream, but keep check for safety
            return True  # Empty DataFrame is considered "valid" in the sense that it was processed
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in data for {symbol}: {df.columns}")
            return False
        if df.isnull().values.any():
            logger.warning(f"Data for {symbol} contains missing values - filling forward during processing")
            # Filling is done in _process_data now
        # Check timestamp monotonicity
        if not df.index.is_monotonic_increasing:
            logger.warning(f"Timestamps for {symbol} are not monotonically increasing. Sorting index.")
            df.sort_index(inplace=True)
        return True

    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe_str: str):
        """
        Save data for a single symbol to its Parquet file, merging with existing data.
        """
        if df.empty:
            logger.info(f"No processed data to save for {symbol} ({timeframe_str}).")
            return

        data_path = self.data_root / symbol / timeframe_str
        filename = data_path / "data.parquet"
        os.makedirs(data_path, exist_ok=True)

        try:
            # Define the desired columns (excluding Symbol, as it's implicit in the path)
            desired_columns = ["Open", "High", "Low", "Close", "Volume",
                               "VWAP", "Returns", "HL_diff", "OHLC_avg"]

            # Ensure df has the correct columns before saving
            cols_to_use = [col for col in desired_columns if col in df.columns]
            save_df = df[cols_to_use].copy()  # Create copy to avoid SettingWithCopyWarning

            if not set(desired_columns).issubset(save_df.columns):
                 logger.warning(f"Processed data for {symbol} is missing some desired columns. Available: {save_df.columns}")

            # If file exists, load, append new data, drop duplicates, sort.
            if os.path.exists(filename):
                try:
                    existing_df = pd.read_parquet(filename)
                    
                    # Handle both DatetimeIndex and timestamp column formats
                    if 'timestamp' in existing_df.columns and not isinstance(existing_df.index, pd.DatetimeIndex):
                        # Has timestamp column but not as index - convert to DatetimeIndex
                        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], utc=True)
                        existing_df = existing_df.set_index('timestamp')
                    elif not isinstance(existing_df.index, pd.DatetimeIndex):
                        # Try to convert index to DatetimeIndex
                        existing_df.index = pd.to_datetime(existing_df.index, utc=True)
                    else:
                        # Already has DatetimeIndex, ensure timezone-aware
                        if existing_df.index.tz is None:
                            existing_df.index = existing_df.index.tz_localize('UTC')
                    
                    # Ensure existing data also has the desired columns
                    existing_cols_to_use = [col for col in desired_columns if col in existing_df.columns]
                    existing_df = existing_df[existing_cols_to_use]

                    combined_df = pd.concat([existing_df, save_df])
                    # Drop duplicates based on index (timestamp)
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    # Sort by index to ensure chronological order
                    combined_df.sort_index(inplace=True)
                except Exception as read_err:
                     logger.error(f"Error reading existing file {filename}, overwriting: {read_err}")
                     logger.exception(read_err)  # Log full traceback for debugging
                     combined_df = save_df.sort_index()  # Use only new data if read fails
            else:
                combined_df = save_df.sort_index()

            # Final check for desired columns before saving
            final_cols = [col for col in desired_columns if col in combined_df.columns]
            combined_df = combined_df[final_cols]

            if combined_df.empty:
                 logger.warning(f"Combined DataFrame for {symbol} is empty after processing/merging. Not saving.")
                 return

            combined_df.to_parquet(
                str(filename),  # Ensure filename is string
                engine='pyarrow',
                compression='snappy',
                index=True
            )
            logger.info(f"Successfully saved/updated data for {symbol} to {filename} ({len(combined_df)} rows)")

        except Exception as e:
            logger.error(f"Failed to save data for {symbol} to {filename}: {str(e)}")
            # Do not raise here to allow other symbols in batch to proceed
    
    def _fetch_process_save_batch(self, symbols_batch: List[str], alpaca_timeframe: str, start_dates: Dict[str, datetime], end_date: datetime, verbose: bool):
        """Fetches, processes, and saves data for a batch of symbols."""
        batch_results = {}
        try:
            # Determine the earliest start date needed for the batch API call
            batch_start_date = min(start_dates.values())
            batch_start_date, batch_end_date = self._validate_dates(batch_start_date, end_date)

            # Fetch data for the entire batch
            raw_data_dict = self._fetch_alpaca_data_batch(symbols_batch, alpaca_timeframe, batch_start_date, batch_end_date)

            for symbol in symbols_batch:
                raw_df = raw_data_dict.get(symbol, pd.DataFrame())  # Get df or empty df

                # Validate raw data for this symbol
                if not self._validate_data(raw_df, symbol):
                    logger.warning(f"Raw data validation failed for {symbol}. Skipping processing and saving.")
                    batch_results[symbol] = None  # Indicate failure/skip
                    continue

                if raw_df.empty:
                    logger.info(f"No new data fetched from API for {symbol}. Will rely on existing file if present.")
                    # Check if we have existing data on disk for this symbol
                    data_path = self.data_root / symbol / alpaca_timeframe
                    filename = data_path / "data.parquet"
                    if os.path.exists(filename):
                        try:
                            # Mark as up-to-date rather than failed
                            batch_results[symbol] = "up-to-date"
                            logger.debug(f"Existing data file found for {symbol}")
                        except Exception as e:
                            logger.warning(f"Error checking existing data for {symbol}: {e}")
                            batch_results[symbol] = None
                    else:
                        logger.warning(f"No existing data file found for {symbol}")
                        batch_results[symbol] = None
                    continue

                # Process the newly fetched data
                processed_df = self._process_data(raw_df, symbol)

                if processed_df.empty:
                     logger.warning(f"Processing resulted in empty DataFrame for {symbol}. Skipping save.")
                     batch_results[symbol] = None
                     continue

                # Save data (merges with existing within the function)
                self._save_data(processed_df, symbol, alpaca_timeframe)

                # Optionally load the final combined data for returning (if needed)
                batch_results[symbol] = processed_df  # Store processed data for potential return

                if verbose:
                    logger.info(f"Data processing and saving completed for {symbol}.")

        except Exception as e:
            logger.error(f"Error processing batch ({symbols_batch[0]}...): {e}", exc_info=True)
            # Mark all symbols in this batch as failed
            for symbol in symbols_batch:
                batch_results[symbol] = None
        finally:
            return batch_results  # Return results (or None for failures) for each symbol

    def load_all_symbols(self, timeframe='hour', years=2, start_date=None, end_date=None, verbose=True, max_workers=4, append=False, force_refresh=False):
        """
        Iterate over all symbols from symbols.json, fetch, process, and save
        historical data in parallel batches.
        """
        try:
            symbols_config = self.load_symbols_config()
            # Collect symbols from all categories: sectors, etfs, and crypto
            all_symbols = set()
            # Add sector symbols
            for sector_symbols in symbols_config.get('sectors', {}).values():
                all_symbols.update(sector_symbols)
            # Add ETF symbols
            for etf_symbols in symbols_config.get('etfs', {}).values():
                all_symbols.update(etf_symbols)
            # Add index symbols
            for index_symbols in symbols_config.get('indices', {}).values():
                all_symbols.update(index_symbols)
            # Add crypto symbols
            for crypto_symbols in symbols_config.get('crypto', {}).values():
                all_symbols.update(crypto_symbols)
            
            all_symbols = sorted(list(all_symbols))  # Convert to sorted list
            logger.info(f"Starting parallel historical data load for {len(all_symbols)} symbols...")

            if timeframe not in self.valid_timeframes:
                raise ValueError(f"Invalid timeframe. Valid options: {list(self.valid_timeframes.keys())}")
            alpaca_timeframe = self.valid_timeframes[timeframe]

            # Use provided end_date or default to now
            end_date = end_date or datetime.now(timezone.utc)
            
            # If start_date provided, use it for all symbols (append mode)
            if start_date:
                default_start_date = start_date
                logger.info(f"Using provided start date {start_date.date()} for all symbols (append mode)")
            else:
                default_start_date = end_date - timedelta(days=365 * years)

            # Determine start date for each symbol
            start_dates = {}
            logger.info("Determining start dates based on existing data...")
            for symbol in tqdm(all_symbols, desc="Checking existing data"):
                data_path = str(self.data_root / symbol / alpaca_timeframe)  # Convert Path to string
                latest_ts = self._get_latest_timestamp(data_path)
                
                if append and latest_ts:
                    # Append mode: start from after the latest existing timestamp
                    start_dates[symbol] = latest_ts + timedelta(seconds=1)
                    logger.debug(f"Append mode for {symbol}: starting from {start_dates[symbol]}")
                elif start_date:
                    # Custom start date provided, use it
                    start_dates[symbol] = start_date
                elif latest_ts and not force_refresh:
                    # Check how old the latest data is
                    hours_old = (end_date - latest_ts).total_seconds() / 3600
                    
                    if hours_old < 1:
                        # Data is very recent (less than an hour old)
                        logger.debug(f"Very recent data for {symbol} ({hours_old:.2f} hours old). Using existing.")
                        start_dates[symbol] = latest_ts + timedelta(seconds=1)
                    elif hours_old > 24 and force_refresh:
                        # If data is more than a day old and force_refresh is True, refetch last 24 hours
                        logger.info(f"Data for {symbol} is {hours_old:.2f} hours old. Force refreshing last 24 hours.")
                        start_dates[symbol] = end_date - timedelta(hours=24)
                    else:
                        # Fetch from the next interval after the last timestamp
                        start_dates[symbol] = latest_ts + timedelta(seconds=1)
                        logger.debug(f"Existing data for {symbol} is {hours_old:.2f} hours old. Fetching from {start_dates[symbol]}")
                else:
                    # No data exists or force refresh is enabled
                    if force_refresh and latest_ts:
                        # Force refresh the last day if data exists
                        start_dates[symbol] = end_date - timedelta(days=1)
                        logger.debug(f"Force refreshing {symbol} data for the last day.")
                    else:
                        # No existing data or complete refresh needed
                        start_dates[symbol] = default_start_date
                        logger.debug(f"No existing data for {symbol}. Fetching from {start_dates[symbol]}")

            # After creating start_dates, run debug check on a few symbols
            debug_symbols = all_symbols[:5]  # Check first 5 symbols
            logger.info(f"DEBUG: Checking existing data files for: {debug_symbols}")
            for symbol in debug_symbols:
                self._debug_check_data_files(symbol, alpaca_timeframe)
                
            # Log system time
            logger.info(f"DEBUG: Current system time: {datetime.now(timezone.utc)}")

            # Create batches of symbols
            symbol_batches = [all_symbols[i:i + MAX_SYMBOLS_PER_BATCH]
                             for i in range(0, len(all_symbols), MAX_SYMBOLS_PER_BATCH)]
            logger.info(f"Created {len(symbol_batches)} batches for parallel processing.")

            num_workers = min(max_workers, len(symbol_batches))  # Adjust workers based on batch count
            logger.info(f"Using {num_workers} worker threads.")

            all_results = {}
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self._fetch_process_save_batch,
                                         batch,
                                         alpaca_timeframe,
                                         {s: start_dates[s] for s in batch},  # Pass only relevant start dates
                                         end_date,
                                         verbose): batch for batch in symbol_batches}

                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Batches"):
                    batch = futures[future]
                    try:
                        batch_result = future.result()
                        all_results.update(batch_result)  # Collect results (or None for failures)
                    except Exception as exc:
                        logger.error(f"Batch ({batch[0]}...) generated an exception: {exc}", exc_info=True)
                        for symbol in batch:
                            all_results[symbol] = None  # Mark all symbols in failed batch

            # Fix the result categorization to avoid DataFrame comparison issues
            successful_symbols = []
            up_to_date_symbols = []
            failed_symbols = []

            # Properly categorize results by checking types first
            for symbol, result in all_results.items():
                if result is None:
                    failed_symbols.append(symbol)
                elif isinstance(result, str) and result == "up-to-date":
                    up_to_date_symbols.append(symbol)
                else:
                    # DataFrame or other non-None, non-"up-to-date" result
                    successful_symbols.append(symbol)
            
            logger.info(f"Finished loading all symbols. Success: {len(successful_symbols)}, Up-to-date: {len(up_to_date_symbols)}, Failed: {len(failed_symbols)}")
            
            if failed_symbols:
                logger.warning(f"Failed symbols: {failed_symbols}")
            
            if up_to_date_symbols and verbose:
                logger.info(f"Up-to-date symbols (no new data needed): {len(up_to_date_symbols)}")
                if len(up_to_date_symbols) <= 10 or verbose:
                    logger.debug(f"Up-to-date symbols list: {up_to_date_symbols}")

        except Exception as e:
            logger.error(f"Failed to load historical data for all symbols: {str(e)}", exc_info=True)

    def load_historical_data(
        self,
        symbol: str = 'AAPL',
        timeframe: str = 'hour',
        years: int = 2,
        start_date=None,
        end_date=None,
        verbose: bool = True,
        append=False
    ) -> Optional[pd.DataFrame]:
        """
        Main method to load historical data for a SINGLE symbol.
        Uses the batch processing logic internally.
        Returns the complete, updated DataFrame for the symbol, loaded from the saved file.
        """
        symbol = symbol.strip().upper()
        try:
            symbols_config = self.load_symbols_config()
            # Collect symbols from all categories: sectors, etfs, and crypto
            all_symbols_flat = []
            for sector_symbols in symbols_config.get('sectors', {}).values():
                all_symbols_flat.extend(sector_symbols)
            for etf_symbols in symbols_config.get('etfs', {}).values():
                all_symbols_flat.extend(etf_symbols)
            for index_symbols in symbols_config.get('indices', {}).values():
                all_symbols_flat.extend(index_symbols)
            for crypto_symbols in symbols_config.get('crypto', {}).values():
                all_symbols_flat.extend(crypto_symbols)
            
            if symbol not in all_symbols_flat:
                raise ValueError(f"Invalid symbol '{symbol}'. See config/symbols.json")

            if timeframe not in self.valid_timeframes:
                raise ValueError(f"Invalid timeframe. Valid options: {list(self.valid_timeframes.keys())}")

            alpaca_timeframe = self.valid_timeframes[timeframe]
            data_path = self.data_root / symbol / alpaca_timeframe
            filename = data_path / "data.parquet"

            # Use provided end_date or default to now
            end_date = end_date or datetime.now(timezone.utc)
            
            # Determine start date
            latest_ts = self._get_latest_timestamp(str(data_path))
            if append and latest_ts:
                start_date = latest_ts + timedelta(seconds=1)
                logger.info(f"Append mode for {symbol}: Fetching updates from {start_date}.")
            elif start_date:
                start_date = start_date
                logger.info(f"Using provided start date {start_date.date()} for {symbol}.")
            elif latest_ts:
                start_date = latest_ts + timedelta(seconds=1)
                logger.info(f"Found existing data for {symbol}. Last timestamp: {latest_ts}. Fetching updates from {start_date}.")
            else:
                start_date = end_date - timedelta(days=365 * years)
                logger.info(f"No existing data found for {symbol}. Fetching full history from {start_date}.")

            # Use the batch fetch/process/save logic for a single symbol
            results = self._fetch_process_save_batch(
                symbols_batch=[symbol],
                alpaca_timeframe=alpaca_timeframe,
                start_dates={symbol: start_date},
                end_date=end_date,
                verbose=verbose
            )

            # After fetching/saving, load the complete data from the file
            if os.path.exists(filename):
                try:
                    final_df = pd.read_parquet(filename)
                    logger.info(f"Successfully loaded final data for {symbol} from {filename} ({len(final_df)} rows)")
                    if verbose:
                        logger.info(f"\nData Summary for {symbol}:\n{final_df.describe()}")
                        logger.info(f"\nLatest 5 records for {symbol}:\n{final_df.tail()}")
                    return final_df
                except Exception as e:
                    logger.error(f"Failed to load final data file {filename} after update: {e}")
                    return None
            else:
                # This might happen if fetching/processing failed or resulted in empty data
                logger.warning(f"Data file {filename} not found after update attempt for {symbol}.")
                return None

        except Exception as e:
            logger.error(f"Failed to load historical data for {symbol}: {str(e)}", exc_info=True)
            return None

if __name__ == "__main__":
    loader = HistoricalDataLoader()

    parser = argparse.ArgumentParser(description='Historical Data Loader')
    parser.add_argument('--symbol', type=str, default=None, help='Stock symbol (optional, default loads all)')
    parser.add_argument('--all', action='store_true', help='Load data for all symbols in config (default if --symbol is not provided)')
    parser.add_argument('--timeframe', type=str, default='hour', help='Timeframe (minute, 5min, 15min, hour, day, week)')
    parser.add_argument('--years', type=int, default=2, help='Number of years historical data to maintain')
    parser.add_argument('--workers', type=int, default=4, help='Max number of parallel workers for --all')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed logging')
    parser.add_argument('--start-date', type=str, default=None, help='Start date in YYYY-MM-DD format (overrides years)')
    parser.add_argument('--end-date', type=str, default=None, help='End date in YYYY-MM-DD format (defaults to now)')
    parser.add_argument('--append', action='store_true', help='Append mode: fetch only new data after existing')

    args = parser.parse_args()

    # Determine verbosity
    is_verbose = args.verbose

    # Parse dates if provided
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)

    # Default to loading all if no specific symbol is given
    load_all = args.all or (args.symbol is None)

    if load_all:
        print(f"Loading data for ALL symbols. Timeframe: {args.timeframe}, Years: {args.years}, Workers: {args.workers}")
        if start_date:
            print(f"Using custom date range: {start_date.date()} to {end_date.date()}")
        loader.load_all_symbols(
            timeframe=args.timeframe,
            years=args.years,
            start_date=start_date,
            end_date=end_date,
            verbose=is_verbose,
            max_workers=args.workers,
            append=args.append
        )
        print("Finished loading data for all symbols.")
    elif args.symbol:
        print(f"Loading data for symbol: {args.symbol}. Timeframe: {args.timeframe}, Years: {args.years}")
        if start_date:
            print(f"Using custom date range: {start_date.date()} to {end_date.date()}")
        data = loader.load_historical_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            years=args.years,
            start_date=start_date,
            end_date=end_date,
            verbose=is_verbose,
            append=args.append
        )
        if data is not None:
            print(f"\nSuccessfully loaded data for {args.symbol}.")
            print("\nFirst 3 rows of loaded data:")
            print(data.head(3))
            print("\nLast 3 rows of loaded data:")
            print(data.tail(3))
        else:
            print(f"\nFailed to load data for {args.symbol}.")
    else:
         # Should not happen due to default logic, but handle anyway
         print("Please specify a symbol using --symbol or use --all.")