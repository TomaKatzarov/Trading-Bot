# news_sentiment.py
import logging
import os
import json
import requests
from datetime import datetime, timedelta
from transformers import pipeline
from dotenv import load_dotenv
import time
from pathlib import Path
from core.db_models import SentimentAnalysis
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from utils.db_setup import Base

# Your DB base + session
from utils.db_setup import Base, SessionLocal

# Import GPU utilities
from utils.gpu_utils import setup_gpu, get_optimal_batch_size

logger = logging.getLogger(__name__)
env_path = Path(__file__).resolve().parent.parent / "Credential.env"
load_dotenv(dotenv_path=env_path, override=True)

class NewsSentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert", device=None, max_workers=4, batch_size=None):
        self.logger = logging.getLogger(__name__)
        self.config_path = os.path.join(os.path.dirname(__file__), '../config/symbols.json')
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.api_secret = os.getenv('APCA_API_SECRET_KEY')
        self.max_workers = max_workers
        self.model_name = model_name
        
        # Create data directories
        self.data_dir = Path(__file__).parent.parent / "data"
        self.sentiment_dir = self.data_dir / "sentiment"
        self.sentiment_dir.mkdir(parents=True, exist_ok=True)

        # Determine device: Use GPU if available, otherwise CPU
        try:
            import torch
            if torch.cuda.is_available():
                device_id = 0
                self.logger.info("CUDA is available. Setting sentiment pipeline device to GPU (0).")
            else:
                device_id = -1
                self.logger.info("CUDA not available. Setting sentiment pipeline device to CPU (-1).")
        except ImportError:
            device_id = -1
            self.logger.warning("PyTorch not found. Setting sentiment pipeline device to CPU (-1).")
        except Exception as e:
            device_id = -1
            self.logger.error(f"Error checking CUDA availability: {e}. Defaulting to CPU (-1).")

        # Setup GPU optimization
        self.gpu_info = setup_gpu(memory_fraction=0.9)
        self.device = self.gpu_info["device"]
        
        # Determine optimal batch size
        if batch_size is None:
            self.batch_size = get_optimal_batch_size(
                model_name="finbert",
                gpu_info=self.gpu_info
            )
        else:
            self.batch_size = batch_size
        
        # Initialize the pipeline with GPU device
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=self.device
        )
        
        # Log GPU status
        if self.gpu_info["available"]:
            logger.info(f"Sentiment analysis using GPU: {self.gpu_info['name']}, batch_size: {self.batch_size}")
        else:
            logger.warning(f"GPU not available. Using CPU for sentiment analysis, batch_size: {self.batch_size}")
        
        # Rate limiting for API calls
        self.api_call_delay = 0.1  # 100ms between API calls to respect rate limits
        self.last_api_call = 0

    def load_symbols_config(self):
        """Load symbols configuration from symbols.json."""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Symbols config file not found at {self.config_path}")
            raise ValueError(f"Symbols config file not found at {self.config_path}")
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON format in symbols.json")
            raise ValueError("Invalid JSON format in symbols.json")

    def fetch_news(self, symbol: str, target_date: datetime = None) -> list:
        """Fetch news for a specific symbol and date."""
        if target_date:
            # Set time range to cover the specific date (24h window)
            from_date = target_date.replace(hour=0, minute=0, second=0).isoformat() + "Z"
            to_date = target_date.replace(hour=23, minute=59, second=59).isoformat() + "Z"
        else:
            # Default to last 24h if no date specified
            from_date = (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z"
            to_date = datetime.utcnow().isoformat() + "Z"

        url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        params = {
            "symbols": symbol,
            "start": from_date,
            "end": to_date,
            "limit": 50, # Consider if pagination is needed for more than 50 articles
            "include_content": False # Fetch only headlines/summaries initially if content isn't always needed
        }
        max_retries = 3
        base_delay = 1 # seconds

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                news_items = response.json().get("news", [])
                articles = [
                    f"{item.get('headline', '')} {item.get('summary', '')}".strip()
                    for item in news_items if item.get('headline') or item.get('summary')
                ]
                if not articles:
                    # Distinguish between no news and API returning empty list
                    self.logger.info(f"No news articles returned by API for {symbol} on {target_date.strftime('%Y-%m-%d') if target_date else 'last 24h'}.")
                return articles # Success

            except requests.exceptions.Timeout as e:
                self.logger.warning(f"Timeout fetching news for {symbol} (Attempt {attempt + 1}/{max_retries}): {e}")
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Connection error fetching news for {symbol} (Attempt {attempt + 1}/{max_retries}): {e}")
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"HTTP error fetching news for {symbol} (Attempt {attempt + 1}/{max_retries}): {e.response.status_code} - {e.response.text}")
                # Don't retry on client errors (4xx) unless it's a rate limit (429)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    break # Break retry loop for non-retryable client errors
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Generic request error fetching news for {symbol} (Attempt {attempt + 1}/{max_retries}): {e}")
            except Exception as e: # Catch other potential errors like JSON decoding
                self.logger.error(f"Unexpected error during news fetch for {symbol} (Attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                break # Don't retry on unexpected errors

            # If we haven't returned or broken, wait and retry
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                self.logger.info(f"Retrying news fetch for {symbol} in {delay} seconds...")
                time.sleep(delay)
            else:
                self.logger.error(f"Failed to fetch news for {symbol} after {max_retries} attempts.")

        return None # Return None if all retries fail or a non-retryable error occurs

    def analyze_sentiment(self, text_batch, truncate=True):
        """
        Analyze the sentiment of a batch of text strings using the FinBERT pipeline.
        Optimized for batch processing with proper error handling.

        Args:
            text_batch (list): List of text strings to analyze.
            truncate (bool): Whether to truncate text to fit model's max token limit.

        Returns:
            list: A list of sentiment scores, one for each input text, in the range [-1, 1].
                  Returns an empty list if input is empty or an error occurs.
                  Score is calculated as positive_prob - negative_prob.
        """
        # Handle empty input
        if not text_batch or all(not text for text in text_batch):
            return [] # Return empty list for empty input

        try:
            # Filter out empty texts and keep track of original indices
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(text_batch):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i)
            
            if not valid_texts:
                return [0.0] * len(text_batch)  # Return neutral scores for all empty texts
            
            # Use the pipeline with return_all_scores=True to get probabilities for all labels
            # Note: FinBERT labels are 'positive', 'negative', 'neutral' (lowercase)
            pipeline_output = self.sentiment_pipeline(valid_texts, truncation=truncate, return_all_scores=True)

            # Initialize scores array with neutral values
            scores = [0.0] * len(text_batch)
            
            for output_idx, item_results in enumerate(pipeline_output):
                original_idx = valid_indices[output_idx]
                
                # pipeline_output is a list of lists of dicts: [[{'label': 'positive', 'score':...}, ...], ...]
                if isinstance(item_results, list):
                    pos_score = 0.0
                    neg_score = 0.0
                    neu_score = 0.0 # Keep track of neutral just in case
                    for result in item_results:
                        if isinstance(result, dict):
                            label = result.get('label', '').lower()
                            score = result.get('score', 0.0)
                            if label == 'positive':
                                pos_score = score
                            elif label == 'negative':
                                neg_score = score
                            elif label == 'neutral':
                                neu_score = score
                    # Calculate final score: positive probability - negative probability
                    # This gives a score roughly in the range [-1, 1]
                    final_score = pos_score - neg_score
                    scores[original_idx] = final_score
                else:
                    # Handle unexpected format for a specific text input
                    logger.warning(f"Unexpected result format for item: {item_results}. Assigning neutral score (0.0).")
                    scores[original_idx] = 0.0

            return scores

        except Exception as e:
            logger.error(f"Error during FinBERT sentiment analysis pipeline execution: {e}", exc_info=True)
            # Return neutral scores for all inputs on error
            return [0.0] * len(text_batch)

    def get_normalized_score(self, news_data: list, symbol: str, truncate=True):
        """
        Calculates a normalized sentiment score based on a list of news articles.

        Args:
            news_data (list | None): List of news article strings (headlines/summaries).
                                     None indicates that news fetching failed upstream.
            symbol (str): Stock symbol (used for logging).
            truncate (bool): Whether to truncate text to fit model's max token limit.

        Returns:
            float: Normalized sentiment score between 0 (most negative) and 1 (most positive),
                   or 0.5 for neutral/no news/error cases.
        """
        # Handle case where news fetching failed upstream
        if news_data is None:
            logger.warning(f"Received None for news_data for {symbol}. Returning neutral sentiment (0.5).")
            return 0.5

        # Handle case where fetching was successful but found no news
        if not news_data: # news_data is []
            return 0.5

        # Apply sentiment analysis to the batch of articles
        try:
            # Call analyze_sentiment with the full batch
            sentiment_scores = self.analyze_sentiment(news_data, truncate=truncate)
            
            if not sentiment_scores:
                return 0.5
            
            # Calculate average sentiment score
            avg_sentiment = np.mean(sentiment_scores)

            # Map from [-1, 1] to [0, 1] range
            normalized_score = (avg_sentiment + 1) / 2
            return float(normalized_score)

        except Exception as e:
            logger.error(f"Unexpected error during sentiment score normalization for {symbol}: {e}", exc_info=True)
            return 0.5 # Return neutral sentiment on unexpected error

    def _rate_limit_api_call(self):
        """Ensure minimum delay between API calls to respect rate limits."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.api_call_delay:
            sleep_time = self.api_call_delay - time_since_last_call
            time.sleep(sleep_time)
        self.last_api_call = time.time()

    def _get_symbol_sentiment_file_path(self, symbol: str) -> Path:
        """Get the file path for storing sentiment data for a symbol."""
        symbol_dir = self.sentiment_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / "daily_sentiment.parquet"

    def _load_existing_sentiment_data(self, symbol: str) -> pd.DataFrame:
        """Load existing sentiment data for a symbol, return empty DataFrame if none exists."""
        file_path = self._get_symbol_sentiment_file_path(symbol)
        if file_path.exists():
            try:
                return pd.read_parquet(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to load existing sentiment data for {symbol}: {e}")
        return pd.DataFrame(columns=['date', 'sentiment_score', 'news_count', 'model_used'])

    def _save_sentiment_data(self, symbol: str, df: pd.DataFrame):
        """Save sentiment data to Parquet file."""
        file_path = self._get_symbol_sentiment_file_path(symbol)
        try:
            df.to_parquet(file_path, index=False)
            self.logger.debug(f"Saved sentiment data for {symbol} to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save sentiment data for {symbol}: {e}")

    def process_symbol_sentiment(self, symbol: str, target_date: datetime) -> Optional[float]:
        """
        Process sentiment for a single symbol on a specific date.
        Returns the normalized sentiment score or None if processing failed.
        """
        try:
            # Rate limit API calls
            self._rate_limit_api_call()
            
            # Check if we already have sentiment for this date
            existing_data = self._load_existing_sentiment_data(symbol)
            date_str = target_date.strftime('%Y-%m-%d')
            
            if not existing_data.empty and date_str in existing_data['date'].values:
                existing_score = existing_data[existing_data['date'] == date_str]['sentiment_score'].iloc[0]
                self.logger.debug(f"Using cached sentiment for {symbol} on {date_str}: {existing_score:.3f}")
                return existing_score

            # Fetch news for the target date
            news_articles = self.fetch_news(symbol, target_date)
            
            if news_articles is None:
                self.logger.warning(f"Failed to fetch news for {symbol} on {date_str}")
                return None
            
            # Calculate sentiment score
            sentiment_score = self.get_normalized_score(news_articles, symbol)
            
            # Create new record
            new_record = pd.DataFrame([{
                'date': date_str,
                'sentiment_score': sentiment_score,
                'news_count': len(news_articles) if news_articles else 0,
                'model_used': self.model_name
            }])
            
            # Append to existing data and save
            updated_data = pd.concat([existing_data, new_record], ignore_index=True)
            updated_data = updated_data.drop_duplicates(subset=['date'], keep='last')
            updated_data = updated_data.sort_values('date')
            
            self._save_sentiment_data(symbol, updated_data)
            
            self.logger.info(f"Processed sentiment for {symbol} on {date_str}: {sentiment_score:.3f} (from {len(news_articles) if news_articles else 0} articles)")
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Error processing sentiment for {symbol} on {target_date.strftime('%Y-%m-%d')}: {e}", exc_info=True)
            return None

    def process_symbols_concurrent(self, symbols: List[str], target_date: datetime, max_workers: Optional[int] = None) -> Dict[str, float]:
        """
        Process sentiment for multiple symbols concurrently for a specific date.
        
        Args:
            symbols: List of symbol strings to process
            target_date: Date to process sentiment for
            max_workers: Maximum number of concurrent workers (defaults to self.max_workers)
            
        Returns:
            Dictionary mapping symbol to sentiment score (only successful ones)
        """
        if max_workers is None:
            max_workers = self.max_workers
            
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.process_symbol_sentiment, symbol, target_date): symbol
                for symbol in symbols
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_symbol), total=len(symbols),
                             desc=f"Processing sentiment for {target_date.strftime('%Y-%m-%d')}"):
                symbol = future_to_symbol[future]
                try:
                    sentiment_score = future.result()
                    if sentiment_score is not None:
                        results[symbol] = sentiment_score
                except Exception as e:
                    self.logger.error(f"Failed to process sentiment for {symbol}: {e}")
        
        self.logger.info(f"Successfully processed sentiment for {len(results)}/{len(symbols)} symbols on {target_date.strftime('%Y-%m-%d')}")
        return results

    def process_historical_sentiment(self, symbols: List[str], start_date: datetime, end_date: datetime,
                                   max_workers: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Process sentiment for multiple symbols across a date range.
        
        Args:
            symbols: List of symbol strings to process
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Nested dictionary: {symbol: {date_str: sentiment_score}}
        """
        if max_workers is None:
            max_workers = self.max_workers
            
        # Generate date range (business days only)
        date_range = pd.bdate_range(start=start_date, end=end_date)
        
        all_results = {}
        
        self.logger.info(f"Processing sentiment for {len(symbols)} symbols across {len(date_range)} business days")
        
        # Process each date
        for current_date in tqdm(date_range, desc="Processing dates"):
            current_date_dt = current_date.to_pydatetime()
            daily_results = self.process_symbols_concurrent(symbols, current_date_dt, max_workers)
            
            # Organize results by symbol
            for symbol, score in daily_results.items():
                if symbol not in all_results:
                    all_results[symbol] = {}
                all_results[symbol][current_date.strftime('%Y-%m-%d')] = score
        
        return all_results

    def get_all_symbols(self) -> List[str]:
        """Load all symbols from the configuration file."""
        try:
            config = self.load_symbols_config()
            all_symbols = []
            
            # Extract symbols from all categories
            for category, symbols in config.items():
                if isinstance(symbols, list):
                    all_symbols.extend(symbols)
                elif isinstance(symbols, dict):
                    # Handle nested structure if any
                    for subcategory, subsymbols in symbols.items():
                        if isinstance(subsymbols, list):
                            all_symbols.extend(subsymbols)
            
            # Remove duplicates and return
            unique_symbols = list(set(all_symbols))
            self.logger.info(f"Loaded {len(unique_symbols)} unique symbols from configuration")
            return unique_symbols
            
        except Exception as e:
            self.logger.error(f"Failed to load symbols from configuration: {e}")
            return []

    # --------------------------------------------------------------------------
    # Database storage method (kept for compatibility)
    # --------------------------------------------------------------------------
    def store_symbol_sentiment(self, symbol: str, score: float, source: str = "alpaca_news", trade_journal_id: str = None):
        """Persist a sentiment record for a given symbol into the SentimentAnalysis table."""
        session = SessionLocal()
        try:
            sentiment_record = SentimentAnalysis(
                symbol=symbol,
                score=score,
                timestamp=datetime.utcnow(),
                source=source,
                raw_data="",
                trade_journal_id=trade_journal_id
            )
            session.add(sentiment_record)
            session.commit()
            self.logger.info(f"Stored sentiment {score:.3f} for {symbol} in DB.")
        except Exception as e:
            self.logger.error(f"Failed to store sentiment for {symbol}: {e}")
            session.rollback()
        finally:
            session.close()


def main():
    """
    Main execution function for processing sentiment scores for all symbols.
    This function can be called to execute the sentiment processing pipeline.
    """
    import argparse
    from datetime import datetime, timedelta
    
    parser = argparse.ArgumentParser(description='Process sentiment scores for trading symbols')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)',
                       default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)',
                       default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--symbols', type=str, nargs='*', help='Specific symbols to process (default: all)')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum concurrent workers')
    parser.add_argument('--batch-size', type=int, help='Batch size for sentiment processing')
    parser.add_argument('--single-date', type=str, help='Process single date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize analyzer
    analyzer = NewsSentimentAnalyzer(
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    # Get symbols to process
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = analyzer.get_all_symbols()
    
    if not symbols:
        logger.error("No symbols to process")
        return
    
    logger.info(f"Processing sentiment for {len(symbols)} symbols")
    
    try:
        if args.single_date:
            # Process single date
            target_date = datetime.strptime(args.single_date, '%Y-%m-%d')
            results = analyzer.process_symbols_concurrent(symbols, target_date)
            logger.info(f"Processed {len(results)} symbols for {args.single_date}")
        else:
            # Process date range
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
            
            results = analyzer.process_historical_sentiment(symbols, start_date, end_date)
            
            total_processed = sum(len(dates) for dates in results.values())
            logger.info(f"Successfully processed {total_processed} symbol-date combinations")
            
    except Exception as e:
        logger.error(f"Error during sentiment processing: {e}", exc_info=True)


if __name__ == "__main__":
    main()