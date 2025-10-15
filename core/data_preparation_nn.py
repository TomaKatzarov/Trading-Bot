"""
Neural Network Data Preparation Module

This module provides the NNDataPreparer class for preparing data for Neural Network training
as part of the NN/RL strategy implementation. It handles data loading, feature selection,
preprocessing, and preparation for NN model training.

Created: 2025-05-23
Phase: 2 - Implementation of core/data_preparation_nn.py (Part 1)
"""

import os
import json
import importlib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    numba_spec = importlib.util.find_spec("numba")
    if numba_spec is None:
        raise ImportError("numba not installed")
    numba_module = importlib.import_module("numba")
    jit = numba_module.jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = None
    logger.warning("Numba not available. Label generation will use slower Python loops.")
    logger.warning("Numba not available. Label generation will use slower Python loops.")


if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def _compute_labels_fast(
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        prediction_horizon: int,
        profit_target: float,
        stop_loss_target: float
    ) -> np.ndarray:
        """
        Numba-optimized function to compute labels vectorized.
        
        Args:
            close_prices: Array of closing prices
            high_prices: Array of high prices
            low_prices: Array of low prices
            prediction_horizon: Number of hours to look ahead
            profit_target: Profit target ratio (e.g., 0.05 for 5%)
            stop_loss_target: Stop loss ratio (e.g., 0.02 for 2%)
            
        Returns:
            Array of labels (1 for profit hit first, 0 otherwise, NaN for insufficient data)
        """
        n = len(close_prices)
        labels = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            # Check if we have enough future data
            if i + prediction_horizon >= n:
                continue  # Leave as NaN
                
            current_close = close_prices[i]
            profit_target_price = current_close * (1.0 + profit_target)
            stop_loss_price = current_close * (1.0 - stop_loss_target)

            # Look ahead in the prediction window
            profit_hit = False
            stop_loss_hit = False
            
            for j in range(1, prediction_horizon + 1):
                if i + j >= n:
                    break
                    
                future_high = high_prices[i + j]
                future_low = low_prices[i + j]

                # Check for stop loss first (conservative approach)
                if future_low <= stop_loss_price:
                    stop_loss_hit = True
                    break

                # Check for profit target
                if future_high >= profit_target_price:
                    profit_hit = True
                    break

            # Assign label
            if profit_hit and not stop_loss_hit:
                labels[i] = 1.0
            elif stop_loss_hit or not profit_hit:
                labels[i] = 0.0
                
        return labels
else:
    # Fallback Python implementation if Numba is not available
    def _compute_labels_fast(
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        prediction_horizon: int,
        profit_target: float,
        stop_loss_target: float
    ) -> np.ndarray:
        """Python fallback for label computation."""
        n = len(close_prices)
        labels = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            if i + prediction_horizon >= n:
                continue
                
            current_close = close_prices[i]
            profit_target_price = current_close * (1.0 + profit_target)
            stop_loss_price = current_close * (1.0 - stop_loss_target)

            profit_hit = False
            stop_loss_hit = False
            
            for j in range(1, min(prediction_horizon + 1, n - i)):
                future_high = high_prices[i + j]
                future_low = low_prices[i + j]

                if future_low <= stop_loss_price:
                    stop_loss_hit = True
                    break

                if future_high >= profit_target_price:
                    profit_hit = True
                    break

            if profit_hit and not stop_loss_hit:
                labels[i] = 1.0
            else:
                labels[i] = 0.0
                
        return labels


class NNDataPreparer:
    """
    Neural Network Data Preparation class for the trading bot.
    
    This class handles:
    - Loading historical data with technical indicators and sentiment
    - Asset ID mapping for multi-symbol training
    - Feature selection and preprocessing
    - Data preparation for NN model training
    
    Based on feature_set_NN.md specifications and asset_id_embedding_strategy.md.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the NNDataPreparer with configuration.
        
        Args:
            config (dict): Configuration dictionary containing:
                - symbols_config_path: Path to symbols.json
                - feature_list: List of features to select
                - nan_handling_features: Strategy for handling NaN values
                - lookback_window: Number of timesteps for sequences (default: 24)
                - prediction_horizon: Hours to look ahead for labels (default: 8)
                - data_base_path: Base path for data files (default: 'data')
                - Other configuration parameters
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be provided as a dictionary")

        required_keys = ['symbols_config_path']
        missing_keys = [key for key in required_keys if not config.get(key)]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        symbols_config_path = config['symbols_config_path']
        if not os.path.exists(symbols_config_path):
            raise FileNotFoundError(f"symbols_config_path does not exist: {symbols_config_path}")

        # Store a defensive copy so we can safely apply defaults without mutating the caller's config
        self.config = dict(config)

        feature_list = self.config.get('feature_list')
        if feature_list is None:
            logger.warning("Configuration missing 'feature_list'; defaulting to empty list (all available columns will be considered)")
            self.config['feature_list'] = []
        elif len(feature_list) == 0:
            logger.warning("Configuration provided an empty 'feature_list'; all available columns will be considered")

        self.raw_data_cache = {}
        self.scalers = {}
        self.asset_id_map = None
        
        # Load or create asset ID mapping
        self._load_or_create_asset_id_mapping()
        
        logger.info("NNDataPreparer initialized successfully")
    
    def _load_or_create_asset_id_mapping(self):
        """
        Load symbol-to-integer-ID mapping from config/asset_id_mapping.json if it exists.
        If the file doesn't exist, create the mapping from symbols.json, save it, and populate self.asset_id_map.
        
        This implements the Simple Integer Mapping strategy from asset_id_embedding_strategy.md.
        """
        mapping_path = "config/asset_id_mapping.json"
        symbols_config_path = self.config.get('symbols_config_path', 'config/symbols.json')
        
        try:
            # Try to load existing mapping
            if os.path.exists(mapping_path):
                logger.info(f"Loading existing asset ID mapping from {mapping_path}")
                with open(mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    self.asset_id_map = mapping_data['symbol_to_id']
                    logger.info(f"Loaded asset ID mapping for {len(self.asset_id_map)} symbols")
            else:
                # Create new mapping from symbols.json
                logger.info(f"Creating new asset ID mapping from {symbols_config_path}")
                self._create_asset_id_mapping(symbols_config_path, mapping_path)
                
        except Exception as e:
            logger.error(f"Error loading/creating asset ID mapping: {e}")
            raise
    
    def _create_asset_id_mapping(self, symbols_config_path: str, mapping_path: str):
        """
        Create asset ID mapping from symbols.json and save to mapping_path.
        
        Args:
            symbols_config_path: Path to symbols.json
            mapping_path: Path to save the mapping file
        """
        try:
            # Load symbols configuration
            with open(symbols_config_path, 'r') as f:
                symbols_config = json.load(f)
            
            # Collect all unique symbols from sectors, etfs, and crypto categories
            unique_symbols = set()
            
            # Add symbols from sectors
            if 'sectors' in symbols_config:
                for sector_name, symbols in symbols_config['sectors'].items():
                    unique_symbols.update(symbols)
            
            # Add symbols from ETFs
            if 'etfs' in symbols_config:
                for etf_category, symbols in symbols_config['etfs'].items():
                    unique_symbols.update(symbols)
            
            # Add symbols from crypto
            if 'crypto' in symbols_config:
                for crypto_category, symbols in symbols_config['crypto'].items():
                    unique_symbols.update(symbols)
            
            # Note: indices category symbols overlap with sector symbols and are deduplicated by using set()
            
            # Sort for consistent ordering across runs
            sorted_symbols = sorted(list(unique_symbols))
            
            # Create bidirectional mapping
            symbol_to_id = {symbol: idx for idx, symbol in enumerate(sorted_symbols)}
            id_to_symbol = {str(idx): symbol for symbol, idx in symbol_to_id.items()}
            
            # Create mapping data structure
            mapping_data = {
                "metadata": {
                    "created_date": "2025-05-23",
                    "total_symbols": len(sorted_symbols),
                    "source_config": "symbols.json",
                    "version": "1.0"
                },
                "symbol_to_id": symbol_to_id,
                "id_to_symbol": id_to_symbol
            }
            
            # Ensure config directory exists
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            
            # Save mapping to file
            with open(mapping_path, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            
            # Set the asset_id_map
            self.asset_id_map = symbol_to_id
            
            logger.info(f"Created asset ID mapping for {len(sorted_symbols)} symbols and saved to {mapping_path}")
            
        except Exception as e:
            logger.error(f"Error creating asset ID mapping: {e}")
            raise
    
    def load_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        """
        Load the Parquet file for a symbol from data/historical/{SYMBOL}/1Hour/data.parquet.
        
        This file is assumed to contain OHLCV, VWAP, 14 technical indicators, 
        2 day-of-week features, and 1 hourly forward-filled sentiment score (total 17+ base features).
        
        Args:
            symbol (str): Trading symbol to load data for
            
        Returns:
            pd.DataFrame: DataFrame with historical data and features, indexed by datetime
        """
        # Check cache first
        if symbol in self.raw_data_cache:
            logger.debug(f"Returning cached data for {symbol}")
            return self.raw_data_cache[symbol]
        
        # Construct file path
        data_base_path = self.config.get('data_base_path', 'data')
        file_path = os.path.join(data_base_path, 'historical', symbol, '1Hour', 'data.parquet')
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found for symbol {symbol}: {file_path}")
            
            # Load the Parquet file
            df = pd.read_parquet(file_path)
            
            # Normalize column names - ensure OHLCV columns are properly capitalized
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            # Apply column mapping
            df = df.rename(columns=column_mapping)
            
            # Ensure the DataFrame's index is a DateTimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                # If timestamp is a column, set it as index
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif 'date' in df.columns:
                    df = df.set_index('date')
                else:
                    # Try to convert the index to datetime
                    df.index = pd.to_datetime(df.index)
            
            # Add column mapping for case differences and naming variations
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'VWAP': 'vwap',
                'Return_1h': '1h_return',
                'Returns': 'returns'
            }
            
            # Apply column mapping
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df[new_name] = df[old_name]
            
            # Add missing required features if they don't exist
            if 'vwap' not in df.columns and all(col in df.columns for col in ['high', 'low', 'close']):
                df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Cache the data
            self.raw_data_cache[symbol] = df
            
            logger.info(f"Loaded data for {symbol}: {len(df)} rows, {len(df.columns)} columns")
            logger.debug(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for symbol {symbol}: {e}")
            raise
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select columns from the input DataFrame based on self.config.feature_list.
        
        Args:
            df (pd.DataFrame): Input DataFrame with all available features
            
        Returns:
            pd.DataFrame: DataFrame with only selected features
        """
        feature_list = self.config.get('feature_list', [])
        
        if not feature_list:
            logger.warning("No feature_list specified in config, returning all columns")
            return df
        
        try:
            # Check which features are available
            available_features = df.columns.tolist()
            missing_features = [f for f in feature_list if f not in available_features]
            
            if missing_features:
                logger.warning(f"Missing features in data: {missing_features}")
                # Use only available features from the requested list
                feature_list = [f for f in feature_list if f in available_features]
            
            if not feature_list:
                raise ValueError("No requested features found in the data")
            
            # Select the features
            selected_df = df[feature_list].copy()
            
            logger.info(f"Selected {len(feature_list)} features: {feature_list}")
            
            return selected_df
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            raise
    
    def _generate_day_of_week_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate day of week features (sine and cosine components) from the DataFrame's datetime index.
        
        Note: Based on the implementation plan, the Parquet files from Phase 1 should already contain
        these features. This method ensures they exist or adds them if missing.
        
        Args:
            df (pd.DataFrame): Input DataFrame with datetime index
            
        Returns:
            pd.DataFrame: DataFrame with day of week features added
        """
        df_copy = df.copy()
        
        # Check if day of week features already exist
        dow_features = ['DayOfWeek_sin', 'DayOfWeek_cos']
        existing_dow_features = [f for f in dow_features if f in df_copy.columns]
        
        if len(existing_dow_features) == 2:
            logger.debug("Day of week features already exist in the data")
            return df_copy
        
        try:
            # Ensure we have a datetime index
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have a DatetimeIndex to generate day of week features")
            
            # Generate day of week (0=Monday, 6=Sunday)
            day_of_week = df_copy.index.dayofweek
            
            # Generate sine and cosine components for cyclical encoding
            df_copy['DayOfWeek_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            df_copy['DayOfWeek_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            logger.info("Generated day of week features (sine and cosine components)")
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error generating day of week features: {e}")
            raise
    
    def _preprocess_single_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Preprocess data for a single symbol by loading, selecting features, and handling NaN values.
        
        Args:
            symbol (str): Trading symbol to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame for the symbol
        """
        try:
            # Load data for the symbol
            df = self.load_data_for_symbol(symbol)
            
            # Select features based on configuration
            df_selected = self._select_features(df)
            
            # Ensure day of week features exist (they should already be in the Parquet files)
            df_with_dow = self._generate_day_of_week_features(df_selected)
            
            # Handle NaN values based on configuration
            nan_handling = self.config.get('nan_handling_features', 'ffill')
            
            if nan_handling == 'ffill':
                df_processed = df_with_dow.ffill().bfill()
            elif nan_handling == 'drop':
                df_processed = df_with_dow.dropna()
            elif nan_handling == 'bfill':
                df_processed = df_with_dow.bfill().ffill()
            elif nan_handling == 'interpolate':
                df_processed = df_with_dow.interpolate().ffill().bfill()
            else:
                logger.warning(f"Unknown NaN handling method: {nan_handling}, using forward fill")
                df_processed = df_with_dow.ffill().bfill()
            
            # Log preprocessing results
            original_rows = len(df)
            processed_rows = len(df_processed)
            nan_count_before = df_selected.isna().sum().sum()
            remaining_nan_counts = df_processed.isna().sum()

            if remaining_nan_counts.any():
                problematic = remaining_nan_counts[remaining_nan_counts > 0]
                for col, count in problematic.items():
                    non_nan_values = df_processed[col].dropna()
                    if not non_nan_values.empty:
                        fill_value = float(non_nan_values.mean())
                        logger.warning(
                            f"Column {col} still has {count} NaNs after {nan_handling}; "
                            f"filling remaining gaps with column mean {fill_value:.6f}"
                        )
                    else:
                        fill_value = 0.0
                        logger.warning(
                            f"Column {col} is entirely NaN after {nan_handling}; "
                            f"filling with {fill_value} to maintain feature continuity"
                        )
                    df_processed[col] = df_processed[col].fillna(fill_value)

            nan_count_after = df_processed.isna().sum().sum()
            
            logger.info(f"Preprocessed {symbol}: {original_rows} -> {processed_rows} rows, "
                       f"NaN count: {nan_count_before} -> {nan_count_after}")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error preprocessing data for symbol {symbol}: {e}")
            raise
    
    def _generate_labels_for_symbol(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate target labels for a symbol using optimized Numba function.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data
            symbol (str): Trading symbol for logging purposes
            
        Returns:
            pd.DataFrame: DataFrame with an added 'label' column
        """
        try:
            # Get configuration parameters
            prediction_horizon = self.config.get('prediction_horizon_hours',
                                                self.config.get('prediction_horizon', 8))
            profit_target = self.config.get('profit_target', 0.025)
            stop_loss_target = self.config.get('stop_loss_target', 0.02)
            
            # Ensure required columns exist
            required_columns = ['Close', 'High', 'Low']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required price columns for {symbol}: {missing_columns}")

            logger.info(f"Generating labels for {symbol} using {'Numba-optimized' if NUMBA_AVAILABLE else 'Python'} function: "
                        f"{len(df)} rows, horizon={prediction_horizon}h, profit={profit_target:.1%}, stop_loss={stop_loss_target:.1%}")

            # Convert to NumPy arrays for optimized computation
            close_prices = df['Close'].to_numpy(dtype=np.float64)
            high_prices = df['High'].to_numpy(dtype=np.float64)
            low_prices = df['Low'].to_numpy(dtype=np.float64)

            # Call the optimized function
            labels = _compute_labels_fast(
                close_prices, high_prices, low_prices,
                prediction_horizon, profit_target, stop_loss_target
            )

            # Create DataFrame with labels
            df_with_labels = df.copy()
            df_with_labels['label'] = labels
            
            # Drop rows where label could not be determined (NaN)
            initial_rows = len(df_with_labels)
            df_with_labels.dropna(subset=['label'], inplace=True)
            final_rows = len(df_with_labels)
            
            # Convert labels to integers
            df_with_labels['label'] = df_with_labels['label'].astype(np.int32)
            
            # Log label statistics
            if final_rows > 0:
                label_counts = df_with_labels['label'].value_counts().sort_index()
                positive_ratio = label_counts.get(1, 0) / final_rows
                logger.info(f"Labels generated for {symbol}: {initial_rows} -> {final_rows} rows "
                            f"(dropped {initial_rows - final_rows} rows with insufficient future data)")
                logger.info(f"Label distribution for {symbol}: {dict(label_counts)}, "
                            f"positive ratio: {positive_ratio:.3f}")
            else:
                logger.warning(f"No labels could be generated for {symbol}")

            return df_with_labels
            
        except Exception as e:
            logger.error(f"Error generating labels for symbol {symbol}: {e}")
            raise
    
    def _generate_sequences_for_symbol(self, df_features: pd.DataFrame, df_labels: pd.DataFrame, asset_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sliding window sequences using efficient stride_tricks.
        
        Args:
            df_features (pd.DataFrame): DataFrame with preprocessed features
            df_labels (pd.DataFrame): DataFrame with the 'label' column
            asset_id (int): Integer asset ID for this symbol
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: X_symbol, y_symbol, asset_ids_symbol
        """
        try:
            lookback_window = self.config.get('lookback_window', 24)
            
            # Align features and labels to a common index
            common_index = df_features.index.intersection(df_labels.index)
            if len(common_index) < lookback_window:
                logger.warning(f"Insufficient overlapping data for asset_id {asset_id}: "
                             f"{len(common_index)} timestamps < {lookback_window} required")
                return np.array([]), np.array([]), np.array([])

            df_features_aligned = df_features.loc[common_index].sort_index()
            df_labels_aligned = df_labels.loc[common_index].sort_index()

            # Convert to NumPy arrays
            features_array = df_features_aligned.to_numpy(dtype=np.float32)
            labels_array = df_labels_aligned['label'].to_numpy(dtype=np.int32)
            
            n_timesteps, n_features = features_array.shape
            
            # Use stride_tricks for ultra-efficient windowing (zero-copy view)
            shape = (n_timesteps - lookback_window + 1, lookback_window, n_features)
            strides = (features_array.strides[0], features_array.strides[0], features_array.strides[1])
            X_symbol = np.lib.stride_tricks.as_strided(features_array, shape=shape, strides=strides).copy()
            
            # Select corresponding labels (label at end of each window)
            y_symbol = labels_array[lookback_window - 1:]
            
            # Create asset_id array
            n_sequences = len(X_symbol)
            asset_ids_symbol = np.full(n_sequences, asset_id, dtype=np.int32)
            
            logger.info(f"Generated sequences for asset_id {asset_id} using stride_tricks: "
                       f"{n_sequences} sequences, shape X: {X_symbol.shape}, shape y: {y_symbol.shape}")
            
            return X_symbol, y_symbol, asset_ids_symbol
            
        except Exception as e:
            logger.error(f"Error generating sequences for asset_id {asset_id}: {e}")
            raise
    
    def _aggregate_data_from_symbols(self, symbols: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Aggregate data from multiple symbols by processing each symbol and collecting results.
        
        Args:
            symbols (List[str]): List of symbols to process
            
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                - all_X_parts: List of X arrays from each symbol
                - all_y_parts: List of y arrays from each symbol
                - all_asset_id_parts: List of asset_id arrays from each symbol
        """
        try:
            # Initialize empty lists
            all_X_parts = []
            all_y_parts = []
            all_asset_id_parts = []
            
            logger.info(f"Aggregating data from {len(symbols)} symbols")
            
            for symbol in symbols:
                try:
                    logger.info(f"Processing symbol: {symbol}")
                    
                    # Get asset ID for this symbol
                    if symbol not in self.asset_id_map:
                        logger.warning(f"Symbol {symbol} not found in asset_id_map, skipping")
                        continue
                    
                    asset_id = self.asset_id_map[symbol]
                    
                    # Step 1: Preprocess single symbol data to get features
                    df_features = self._preprocess_single_symbol_data(symbol)
                    
                    if df_features.empty:
                        logger.warning(f"No data available for symbol {symbol}, skipping")
                        continue
                    
                    # Step 2: Generate labels using the same processed features DataFrame
                    # Note: _generate_labels_for_symbol expects the full DataFrame with price columns
                    # So we need to load the raw data again to get price columns for label generation
                    df_raw = self.load_data_for_symbol(symbol)
                    df_labels = self._generate_labels_for_symbol(df_raw, symbol)
                    
                    if df_labels.empty:
                        logger.warning(f"No labels generated for symbol {symbol}, skipping")
                        continue
                    
                    # Step 3: Generate sequences
                    X_symbol, y_symbol, asset_ids_symbol = self._generate_sequences_for_symbol(
                        df_features, df_labels, asset_id
                    )
                    
                    # Check if we got valid sequences
                    if len(X_symbol) == 0:
                        logger.warning(f"No sequences generated for symbol {symbol}, skipping")
                        continue
                    
                    # Step 4: Append to lists
                    all_X_parts.append(X_symbol)
                    all_y_parts.append(y_symbol)
                    all_asset_id_parts.append(asset_ids_symbol)
                    
                    logger.info(f"Successfully processed {symbol}: {len(X_symbol)} sequences")
                    
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
                    # Continue with next symbol instead of failing completely
                    continue
            
            logger.info(f"Aggregation complete: {len(all_X_parts)} symbols processed successfully")
            
            # Log summary statistics
            if all_X_parts:
                total_sequences = sum(len(X) for X in all_X_parts)
                total_positive_labels = sum(np.sum(y) for y in all_y_parts)
                overall_positive_ratio = total_positive_labels / total_sequences if total_sequences > 0 else 0
                
                logger.info(f"Aggregation summary: {total_sequences} total sequences, "
                           f"{total_positive_labels} positive labels, "
                           f"positive ratio: {overall_positive_ratio:.3f}")
            
            return all_X_parts, all_y_parts, all_asset_id_parts
            
        except Exception as e:
            logger.error(f"Error in data aggregation: {e}")
            raise

    def _split_data(self, X_all: np.ndarray, y_all: np.ndarray, asset_ids_all: np.ndarray) -> dict:
        """
        Split the concatenated data into train, validation, and test sets based on temporal order.
        
        For time-series data, we respect temporal order by taking the first X% for train,
        next Y% for validation, and final Z% for test. Shuffling is generally avoided.
        
        Args:
            X_all (np.ndarray): Concatenated feature arrays, shape (n_samples_total, lookback_window, n_features)
            y_all (np.ndarray): Concatenated label arrays, shape (n_samples_total,)
            asset_ids_all (np.ndarray): Concatenated asset ID arrays, shape (n_samples_total,)
            
        Returns:
            dict: Dictionary containing train/validation/test splits with keys:
                'train': {'X': X_train, 'y': y_train, 'asset_ids': asset_ids_train}
                'val': {'X': X_val, 'y': y_val, 'asset_ids': asset_ids_val}
                'test': {'X': X_test, 'y': y_test, 'asset_ids': asset_ids_test}
        """
        try:
            # Get split ratios from config
            train_ratio = self.config.get('train_ratio', 0.7)
            val_ratio = self.config.get('val_ratio', 0.15)
            test_ratio = self.config.get('test_ratio', 0.15)
            shuffle_before_split = self.config.get('shuffle_before_split', False)
            
            # Validate ratios
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                logger.warning(f"Split ratios don't sum to 1.0: {total_ratio}, normalizing...")
                train_ratio /= total_ratio
                val_ratio /= total_ratio
                test_ratio /= total_ratio
            
            n_samples = len(X_all)
            logger.info(f"Splitting {n_samples} samples with ratios - train: {train_ratio:.3f}, "
                       f"val: {val_ratio:.3f}, test: {test_ratio:.3f}")
            
            # Calculate split indices
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            if shuffle_before_split:
                logger.warning("Shuffling before split is enabled - this may break temporal order for time-series data")
                # Create random indices
                indices = np.random.permutation(n_samples)
                X_all = X_all[indices]
                y_all = y_all[indices]
                asset_ids_all = asset_ids_all[indices]
            
            # Perform temporal splits
            X_train = X_all[:train_end]
            y_train = y_all[:train_end]
            asset_ids_train = asset_ids_all[:train_end]
            
            X_val = X_all[train_end:val_end]
            y_val = y_all[train_end:val_end]
            asset_ids_val = asset_ids_all[train_end:val_end]
            
            X_test = X_all[val_end:]
            y_test = y_all[val_end:]
            asset_ids_test = asset_ids_all[val_end:]
            
            # Log split statistics
            logger.info(f"Data split completed:")
            logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/n_samples:.1%})")
            logger.info(f"  Val:   {len(X_val)} samples ({len(X_val)/n_samples:.1%})")
            logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/n_samples:.1%})")
            
            # Log label distributions
            for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
                if len(y_split) > 0:
                    unique, counts = np.unique(y_split, return_counts=True)
                    label_dist = dict(zip(unique, counts))
                    pos_ratio = label_dist.get(1, 0) / len(y_split)
                    logger.info(f"  {split_name} label distribution: {label_dist}, positive ratio: {pos_ratio:.3f}")
            
            # Create return dictionary
            splits = {
                'train': {
                    'X': X_train,
                    'y': y_train,
                    'asset_ids': asset_ids_train
                },
                'val': {
                    'X': X_val,
                    'y': y_val,
                    'asset_ids': asset_ids_val
                },
                'test': {
                    'X': X_test,
                    'y': y_test,
                    'asset_ids': asset_ids_test
                }
            }
            
            return splits
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise

    def _apply_scaling(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply feature scaling to the train, validation, and test sets.
        
        Fits scalers on training data and applies to all sets. Handles 3D arrays by
        reshaping to 2D for fitting/transforming, then reshaping back to 3D.
        
        Args:
            X_train (np.ndarray): Training features, shape (n_train, lookback_window, n_features)
            X_val (np.ndarray): Validation features, shape (n_val, lookback_window, n_features)
            X_test (np.ndarray): Test features, shape (n_test, lookback_window, n_features)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Scaled X_train, X_val, X_test
        """
        try:
            # Get scaling configuration
            scaling_method = self.config.get('scaling_method', 'standard')
            scaling_method_map = self.config.get('scaling_method_map', {})
            
            logger.info(f"Applying {scaling_method} scaling to features")
            
            # Get original shapes
            train_shape = X_train.shape
            val_shape = X_val.shape
            test_shape = X_test.shape
            
            n_train, lookback_window, n_features = train_shape
            
            # Reshape to 2D for fitting scalers: (n_samples * lookback_window, n_features)
            X_train_2d = X_train.reshape(-1, n_features)
            X_val_2d = X_val.reshape(-1, n_features)
            X_test_2d = X_test.reshape(-1, n_features)
            
            logger.info(f"Reshaped for scaling - Train: {train_shape} -> {X_train_2d.shape}")
            
            # Initialize scalers dictionary
            self.scalers = {}
            
            # Apply scaling per feature or globally
            if scaling_method_map:
                # Per-feature scaling (if specified in config)
                logger.info("Applying per-feature scaling based on scaling_method_map")
                
                X_train_scaled_2d = X_train_2d.copy()
                X_val_scaled_2d = X_val_2d.copy()
                X_test_scaled_2d = X_test_2d.copy()
                
                for feature_idx in range(n_features):
                    feature_scaling_method = scaling_method_map.get(feature_idx, scaling_method)
                    
                    # Create scaler based on method
                    if feature_scaling_method == 'standard':
                        scaler = StandardScaler()
                    elif feature_scaling_method == 'robust':
                        scaler = RobustScaler()
                    else:
                        logger.warning(f"Unknown scaling method {feature_scaling_method}, using StandardScaler")
                        scaler = StandardScaler()
                    
                    # Fit on training data for this feature
                    scaler.fit(X_train_2d[:, feature_idx:feature_idx+1])
                    
                    # Transform all sets for this feature
                    X_train_scaled_2d[:, feature_idx:feature_idx+1] = scaler.transform(X_train_2d[:, feature_idx:feature_idx+1])
                    X_val_scaled_2d[:, feature_idx:feature_idx+1] = scaler.transform(X_val_2d[:, feature_idx:feature_idx+1])
                    X_test_scaled_2d[:, feature_idx:feature_idx+1] = scaler.transform(X_test_2d[:, feature_idx:feature_idx+1])
                    
                    # Store scaler
                    self.scalers[f'feature_{feature_idx}'] = scaler
                
            else:
                # Global scaling (default)
                logger.info(f"Applying global {scaling_method} scaling")
                
                # Create scaler based on method
                if scaling_method == 'standard':
                    scaler = StandardScaler()
                elif scaling_method == 'robust':
                    scaler = RobustScaler()
                else:
                    logger.warning(f"Unknown scaling method {scaling_method}, using StandardScaler")
                    scaler = StandardScaler()
                
                # Fit on training data
                scaler.fit(X_train_2d)
                
                # Transform all sets
                X_train_scaled_2d = scaler.transform(X_train_2d)
                X_val_scaled_2d = scaler.transform(X_val_2d)
                X_test_scaled_2d = scaler.transform(X_test_2d)
                
                # Store scaler
                self.scalers['global'] = scaler
            
            # Reshape back to 3D
            X_train_scaled = X_train_scaled_2d.reshape(train_shape)
            X_val_scaled = X_val_scaled_2d.reshape(val_shape)
            X_test_scaled = X_test_scaled_2d.reshape(test_shape)
            
            logger.info(f"Scaling completed. Scalers stored: {list(self.scalers.keys())}")
            
            # Log scaling statistics
            logger.info(f"Scaled data statistics:")
            logger.info(f"  Train - mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
            logger.info(f"  Val   - mean: {X_val_scaled.mean():.4f}, std: {X_val_scaled.std():.4f}")
            logger.info(f"  Test  - mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")
            
            return X_train_scaled, X_val_scaled, X_test_scaled
            
        except Exception as e:
            logger.error(f"Error applying scaling: {e}")
            raise

    def _calculate_sample_weights(self, y_train: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate sample weights for training data to handle class imbalance.
        
        Args:
            y_train (np.ndarray): Training labels
            
        Returns:
            Optional[np.ndarray]: Sample weights array, or None if not calculated
        """
        try:
            calculate_weights = self.config.get('calculate_sample_weights', False)
            
            if not calculate_weights:
                logger.info("Sample weight calculation disabled in config")
                return None
            
            weight_strategy = self.config.get('sample_weight_strategy', 'inverse_frequency')
            
            logger.info(f"Calculating sample weights using strategy: {weight_strategy}")
            
            # Calculate class distribution
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            class_distribution = dict(zip(unique_classes, class_counts))
            
            logger.info(f"Training class distribution: {class_distribution}")
            
            if weight_strategy == 'inverse_frequency':
                # Use sklearn's compute_sample_weight with 'balanced' strategy
                sample_weights = compute_sample_weight('balanced', y_train)
                
            elif weight_strategy == 'manual':
                # Manual weight calculation (inverse frequency)
                total_samples = len(y_train)
                n_classes = len(unique_classes)
                
                # Calculate weights: total_samples / (n_classes * class_count)
                class_weights = {}
                for class_label, count in class_distribution.items():
                    class_weights[class_label] = total_samples / (n_classes * count)
                
                # Apply weights to samples
                sample_weights = np.array([class_weights[label] for label in y_train])
                
            else:
                logger.warning(f"Unknown sample weight strategy: {weight_strategy}, using inverse_frequency")
                sample_weights = compute_sample_weight('balanced', y_train)
            
            # Log weight statistics
            weight_stats = {
                'min': sample_weights.min(),
                'max': sample_weights.max(),
                'mean': sample_weights.mean(),
                'std': sample_weights.std()
            }
            
            logger.info(f"Sample weights calculated: {len(sample_weights)} weights")
            logger.info(f"Weight statistics: {weight_stats}")
            
            # Log average weight per class
            for class_label in unique_classes:
                class_mask = y_train == class_label
                avg_weight = sample_weights[class_mask].mean()
                logger.info(f"  Class {class_label}: average weight = {avg_weight:.4f}")
            
            return sample_weights
            
        except Exception as e:
            logger.error(f"Error calculating sample weights: {e}")
            raise

    def save_scalers(self, path: str):
        """
        Save the fitted scalers to disk using joblib.
        
        Args:
            path (str): Path to save the scalers file
        """
        try:
            if not self.scalers:
                logger.warning("No scalers to save")
                return
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save scalers dictionary
            joblib.dump(self.scalers, path)
            
            logger.info(f"Saved {len(self.scalers)} scalers to {path}")
            
        except Exception as e:
            logger.error(f"Error saving scalers to {path}: {e}")
            raise

    def load_scalers(self, path: str):
        """
        Load fitted scalers from disk using joblib.
        
        Args:
            path (str): Path to load the scalers file from
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Scalers file not found: {path}")
            
            # Load scalers dictionary
            self.scalers = joblib.load(path)
            
            logger.info(f"Loaded {len(self.scalers)} scalers from {path}")
            
        except Exception as e:
            logger.error(f"Error loading scalers from {path}: {e}")
            raise

    def get_prepared_data_for_training(self) -> dict:
        """
        Main orchestration method that prepares data for NN training.
        
        This method calls all the helper methods in the correct order:
        1. Aggregate data from symbols
        2. Concatenate the parts
        3. Split into train/val/test
        4. Apply scaling
        5. Calculate sample weights (if enabled)
        6. Save scalers
        7. Return final dictionary
        
        Returns:
            dict: Dictionary containing prepared data with keys:
                'train': {'X': X_train_scaled, 'y': y_train, 'asset_ids': asset_ids_train, 'sample_weights': weights (optional)}
                'val': {'X': X_val_scaled, 'y': y_val, 'asset_ids': asset_ids_val}
                'test': {'X': X_test_scaled, 'y': y_test, 'asset_ids': asset_ids_test}
                'scalers': self.scalers
                'asset_id_map': self.asset_id_map
        """
        try:
            logger.info("Starting data preparation for NN training")
            
            # Step 1: Get symbols list from config
            symbols_list = self.config.get('symbols_list', [])
            if not symbols_list:
                # If no symbols_list provided, use all symbols from asset_id_map
                symbols_list = list(self.asset_id_map.keys())
                logger.info(f"No symbols_list in config, using all {len(symbols_list)} symbols from asset_id_map")
            
            logger.info(f"Preparing data for {len(symbols_list)} symbols")
            
            # Step 2: Aggregate data from symbols
            logger.info("Step 1/6: Aggregating data from symbols...")
            X_parts, y_parts, asset_id_parts = self._aggregate_data_from_symbols(symbols_list)
            
            if not X_parts:
                raise ValueError("No data was successfully processed from any symbols")
            
            # Step 3: Concatenate parts into single arrays
            logger.info("Step 2/6: Concatenating data parts...")
            X_all = np.concatenate(X_parts, axis=0)
            y_all = np.concatenate(y_parts, axis=0)
            asset_ids_all = np.concatenate(asset_id_parts, axis=0)
            
            logger.info(f"Concatenated data shapes - X: {X_all.shape}, y: {y_all.shape}, asset_ids: {asset_ids_all.shape}")
            
            # Step 4: Split data into train/val/test
            logger.info("Step 3/6: Splitting data...")
            splits = self._split_data(X_all, y_all, asset_ids_all)
            
            # Step 5: Apply scaling
            logger.info("Step 4/6: Applying scaling...")
            X_train_scaled, X_val_scaled, X_test_scaled = self._apply_scaling(
                splits['train']['X'], splits['val']['X'], splits['test']['X']
            )
            
            # Update splits with scaled data
            splits['train']['X'] = X_train_scaled
            splits['val']['X'] = X_val_scaled
            splits['test']['X'] = X_test_scaled
            
            # Step 6: Calculate sample weights (if enabled)
            logger.info("Step 5/6: Calculating sample weights...")
            sample_weights = self._calculate_sample_weights(splits['train']['y'])
            if sample_weights is not None:
                splits['train']['sample_weights'] = sample_weights
            
            # Step 7: Save scalers
            logger.info("Step 6/6: Saving scalers...")
            output_path_scalers = self.config.get('output_path_scalers', 'models/scalers.joblib')
            self.save_scalers(output_path_scalers)
            
            # Step 8: Prepare final return dictionary
            final_data = {
                'train': splits['train'],
                'val': splits['val'],  # Standard key name
                'test': splits['test'],
                'scalers': self.scalers,
                'asset_id_map': self.asset_id_map
            }
            
            # Log final summary
            logger.info("Data preparation completed successfully!")
            logger.info("Final data summary:")
            for split_name in ['train', 'val', 'test']:
                split_data = final_data[split_name]
                logger.info(f"  {split_name.capitalize()}: X shape {split_data['X'].shape}, "
                           f"y shape {split_data['y'].shape}, "
                           f"asset_ids shape {split_data['asset_ids'].shape}")
                if 'sample_weights' in split_data:
                    logger.info(f"    Sample weights shape: {split_data['sample_weights'].shape}")
            
            logger.info(f"Scalers: {len(final_data['scalers'])} fitted scalers")
            logger.info(f"Asset ID map: {len(final_data['asset_id_map'])} symbols")
            
            return final_data
            
        except Exception as e:
            logger.error(f"Error in get_prepared_data_for_training: {e}")
            raise

    def get_prepared_data_for_rl_training(self) -> dict:
        """
        Prepare data specifically for RL training WITHOUT label generation or filtering.
        Generates train/val/test splits following Phase 3 structure for proper evaluation.
        
        This method is designed for Reinforcement Learning where the agent needs access to
        the FULL, UNFILTERED dataset. Unlike supervised learning, RL doesn't need pre-computed
        labels or profit/stop-loss filtering - the agent will discover its own reward patterns.
        
        Key differences from get_prepared_data_for_training():
        1. NO label generation (no profit target / stop loss calculations)
        2. NO filtering based on labels (preserves 100% of data)
        3. Returns raw OHLCV + technical indicators + sentiment + temporal features
        4. Maintains chronological order for time-series integrity
        5. Generates train/val/test temporal splits (default 70/15/15)
        6. Compatible with Phase 3 data structure (data/phase3_splits/SYMBOL/)
        
        Returns:
            dict: Dictionary containing prepared data with keys:
                'symbols_data': Dict[symbol, Dict[split, pd.DataFrame]] - Data per symbol per split
                'symbols_list': List[str] - List of processed symbols
                'asset_id_map': Dict[str, int] - Symbol to ID mapping
                'feature_columns': List[str] - List of feature column names
                'split_info': Dict - Information about train/val/test splits
                'date_range': Dict[str, str] - Start and end dates
        """
        try:
            logger.info("="*80)
            logger.info("Starting data preparation for RL training (NO FILTERING MODE)")
            logger.info("="*80)
            
            # Step 1: Get symbols list from config
            symbols_list = self.config.get('symbols_list', [])
            if not symbols_list:
                # If no symbols_list provided, use all symbols from asset_id_map
                symbols_list = list(self.asset_id_map.keys())
                logger.info(f"No symbols_list in config, using all {len(symbols_list)} symbols from asset_id_map")
            
            # Get split ratios (default 80/10/10 for RL training - more data for agent)
            train_ratio = self.config.get('train_ratio', 0.80)
            val_ratio = self.config.get('val_ratio', 0.10)
            test_ratio = self.config.get('test_ratio', 0.10)
            
            # Validate split ratios
            total_ratio = train_ratio + val_ratio + test_ratio
            if not np.isclose(total_ratio, 1.0):
                logger.warning(f"Split ratios sum to {total_ratio}, normalizing to 1.0")
                train_ratio /= total_ratio
                val_ratio /= total_ratio
                test_ratio /= total_ratio
            
            logger.info(f"Preparing RL data for {len(symbols_list)} symbols")
            logger.info(f"Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%} (optimized for RL)")
            
            # Step 2: Load and preprocess data for each symbol WITHOUT label generation
            symbols_data = {}
            split_info = {
                'train': {'total_rows': 0, 'symbols': []},
                'val': {'total_rows': 0, 'symbols': []},
                'test': {'total_rows': 0, 'symbols': []}
            }
            feature_columns = None
            date_range = {'start': None, 'end': None}
            
            for symbol in symbols_list:
                try:
                    logger.info(f"Processing symbol: {symbol} for RL training")
                    
                    # Get asset ID
                    if symbol not in self.asset_id_map:
                        logger.warning(f"Symbol {symbol} not found in asset_id_map, skipping")
                        continue
                    
                    # Load and preprocess data (NO label generation)
                    df_processed = self._preprocess_single_symbol_data(symbol)
                    
                    if df_processed.empty:
                        logger.warning(f"No data available for {symbol} after preprocessing, skipping")
                        continue
                    
                    # Add asset_id column
                    asset_id = self.asset_id_map[symbol]
                    df_processed['asset_id'] = asset_id
                    
                    # Track feature columns (should be consistent across symbols)
                    if feature_columns is None:
                        feature_columns = df_processed.columns.tolist()
                    
                    # Track date range
                    symbol_start = df_processed.index.min()
                    symbol_end = df_processed.index.max()
                    if date_range['start'] is None or symbol_start < pd.Timestamp(date_range['start']):
                        date_range['start'] = str(symbol_start)
                    if date_range['end'] is None or symbol_end > pd.Timestamp(date_range['end']):
                        date_range['end'] = str(symbol_end)
                    
                    # Generate chronological train/val/test splits
                    n_rows = len(df_processed)
                    train_end_idx = int(n_rows * train_ratio)
                    val_end_idx = int(n_rows * (train_ratio + val_ratio))
                    
                    # Split data chronologically (NO SHUFFLING for time-series)
                    df_train = df_processed.iloc[:train_end_idx].copy()
                    df_val = df_processed.iloc[train_end_idx:val_end_idx].copy()
                    df_test = df_processed.iloc[val_end_idx:].copy()
                    
                    # Store splits for this symbol
                    symbols_data[symbol] = {
                        'train': df_train,
                        'val': df_val,
                        'test': df_test,
                        'full': df_processed  # Keep full data for reference
                    }
                    
                    # Update split info
                    split_info['train']['total_rows'] += len(df_train)
                    split_info['train']['symbols'].append(symbol)
                    split_info['val']['total_rows'] += len(df_val)
                    split_info['val']['symbols'].append(symbol)
                    split_info['test']['total_rows'] += len(df_test)
                    split_info['test']['symbols'].append(symbol)
                    
                    logger.info(f"✓ {symbol}: Total {len(df_processed):,} rows → "
                               f"Train {len(df_train):,} | Val {len(df_val):,} | Test {len(df_test):,}")
                    logger.info(f"  Date ranges: {symbol_start.date()} to {symbol_end.date()}")
                    logger.info(f"  Train: {df_train.index.min().date()} to {df_train.index.max().date()}")
                    logger.info(f"  Val:   {df_val.index.min().date()} to {df_val.index.max().date()}")
                    logger.info(f"  Test:  {df_test.index.min().date()} to {df_test.index.max().date()}")
                    
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol} for RL training: {e}")
                    continue
            
            if not symbols_data:
                raise ValueError("No data was successfully processed from any symbols for RL training")
            
            # Calculate total rows across all splits
            total_rows = (split_info['train']['total_rows'] + 
                         split_info['val']['total_rows'] + 
                         split_info['test']['total_rows'])
            
            # Step 3: Prepare final return dictionary
            final_data = {
                'symbols_data': symbols_data,
                'symbols_list': list(symbols_data.keys()),
                'asset_id_map': self.asset_id_map,
                'feature_columns': feature_columns,
                'split_info': split_info,
                'total_rows': total_rows,
                'date_range': date_range,
                'mode': 'RL_TRAINING',
                'no_filtering': True,
                'no_labels': True,
                'split_ratios': {
                    'train': train_ratio,
                    'val': val_ratio,
                    'test': test_ratio
                }
            }
            
            # Log final summary
            logger.info("="*80)
            logger.info("RL data preparation completed successfully!")
            logger.info("="*80)
            logger.info(f"Symbols processed: {len(symbols_data)}")
            logger.info(f"Total timesteps: {total_rows:,} rows (100% preserved, no filtering)")
            logger.info(f"  Train: {split_info['train']['total_rows']:,} rows ({train_ratio:.1%})")
            logger.info(f"  Val:   {split_info['val']['total_rows']:,} rows ({val_ratio:.1%})")
            logger.info(f"  Test:  {split_info['test']['total_rows']:,} rows ({test_ratio:.1%})")
            logger.info(f"Features per symbol: {len(feature_columns)} columns")
            logger.info(f"Date range: {date_range['start']} to {date_range['end']}")
            logger.info(f"Asset ID map: {len(self.asset_id_map)} symbols")
            
            # Log per-symbol summary
            logger.info("\nPer-symbol data summary:")
            logger.info("-" * 80)
            for symbol in sorted(symbols_data.keys()):
                splits = symbols_data[symbol]
                df_full = splits['full']
                df_train = splits['train']
                df_val = splits['val']
                df_test = splits['test']
                logger.info(f"  {symbol:8s}: Total {len(df_full):,} rows | "
                           f"Train {len(df_train):,} | Val {len(df_val):,} | Test {len(df_test):,}")
                logger.info(f"            asset_id={df_full['asset_id'].iloc[0]:3d} | "
                           f"dates: {df_full.index.min().date()} to {df_full.index.max().date()}")
            logger.info("-" * 80)
            
            # Verify no NaN values
            logger.info("\nData quality checks:")
            for symbol, splits in symbols_data.items():
                total_nans = 0
                for split_name in ['train', 'val', 'test']:
                    df = splits[split_name]
                    nan_count = df.isna().sum().sum()
                    total_nans += nan_count
                
                if total_nans > 0:
                    logger.warning(f"  {symbol}: {total_nans} NaN values detected across splits")
                else:
                    logger.info(f"  {symbol}: ✓ No NaN values (clean data)")
            
            logger.info("="*80)
            logger.info("IMPORTANT: This data is ready for RL training without any filtering")
            logger.info("The RL agent will learn directly from the full historical data")
            logger.info("="*80)
            
            return final_data
            
        except Exception as e:
            logger.error(f"Error in get_prepared_data_for_rl_training: {e}")
            raise


# Example usage and configuration
if __name__ == "__main__":
    # Example configuration
    config = {
        'symbols_config_path': 'config/symbols.json',
        'feature_list': [
            # Technical indicators (14)
            'SMA_10', 'SMA_20', 'MACD', 'MACD_signal', 'MACD_histogram',
            'RSI_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'ATR_14',
            'BB_bandwidth', 'OBV', 'Volume_SMA_20', 'Returns_1h',
            # Sentiment (1)
            'sentiment_score_hourly_ffill',
            # Day of week (2)
            'DayOfWeek_sin', 'DayOfWeek_cos'
        ],
        'nan_handling_features': 'ffill',
        'lookback_window': 24,
        'prediction_horizon': 8,
        'profit_target': 0.025,  # +2.5%
        'stop_loss_target': 0.02,  # -2%
        'data_base_path': 'data',
        # Data splitting configuration
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'shuffle_before_split': False,  # Keep temporal order for time-series
        # Scaling configuration
        'scaling_method': 'standard',  # 'standard' or 'robust'
        'scaling_method_map': {},  # Optional per-feature scaling
        # Sample weights configuration
        'calculate_sample_weights': True,
        'sample_weight_strategy': 'inverse_frequency',  # 'inverse_frequency' or 'manual'
        # Output paths
        'output_path_scalers': 'models/scalers.joblib',
        # Symbols list (optional - if not provided, uses all symbols from asset_id_map)
        'symbols_list': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']  # Example subset
    }
    
    # Initialize the data preparer
    preparer = NNDataPreparer(config)
    
    # Example: Load and preprocess data for AAPL
    try:
        aapl_data = preparer._preprocess_single_symbol_data('AAPL')
        print(f"AAPL data shape: {aapl_data.shape}")
        print(f"AAPL columns: {aapl_data.columns.tolist()}")
        print(f"Asset ID for AAPL: {preparer.asset_id_map.get('AAPL', 'Not found')}")
        
        # Example: Generate labels for AAPL
        aapl_raw = preparer.load_data_for_symbol('AAPL')
        aapl_labels = preparer._generate_labels_for_symbol(aapl_raw, 'AAPL')
        print(f"AAPL labels shape: {aapl_labels.shape}")
        print(f"AAPL label distribution: {aapl_labels['label'].value_counts().to_dict()}")
        
        # Example: Generate sequences for AAPL
        asset_id = preparer.asset_id_map['AAPL']
        X, y, asset_ids = preparer._generate_sequences_for_symbol(aapl_data, aapl_labels, asset_id)
        print(f"AAPL sequences - X shape: {X.shape}, y shape: {y.shape}, asset_ids shape: {asset_ids.shape}")
        
        # Example: Full data preparation pipeline
        print("\n" + "="*50)
        print("FULL DATA PREPARATION PIPELINE EXAMPLE")
        print("="*50)
        
        # Use the main orchestration method
        prepared_data = preparer.get_prepared_data_for_training()
        
        print("Data preparation completed!")
        print(f"Train set: X shape {prepared_data['train']['X'].shape}, y shape {prepared_data['train']['y'].shape}")
        print(f"Val set: X shape {prepared_data['val']['X'].shape}, y shape {prepared_data['val']['y'].shape}")
        print(f"Test set: X shape {prepared_data['test']['X'].shape}, y shape {prepared_data['test']['y'].shape}")
        
        if 'sample_weights' in prepared_data['train']:
            print(f"Sample weights shape: {prepared_data['train']['sample_weights'].shape}")
        
        print(f"Number of scalers: {len(prepared_data['scalers'])}")
        print(f"Asset ID map size: {len(prepared_data['asset_id_map'])}")
        
    except Exception as e:
        print(f"Error processing AAPL: {e}")