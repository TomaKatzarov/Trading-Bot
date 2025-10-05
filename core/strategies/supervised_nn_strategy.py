"""
SupervisedNNStrategy: Trading strategy class for Neural Network signals.

This module implements the core trading logic for the supervised Neural Network
approach, translating NN model predictions into trade entry/exit decisions.

Part of Task 1.3.2: Implement trade entry/exit rules aligned with NN signals.
Part of Task 1.3.3: Connect model inference to backtester data feed.
"""

from typing import Dict, Any, Optional, Tuple
import logging
import torch
import joblib
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# MLflow integration for loading artifacts
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class SupervisedNNStrategy:
    """
    Trading strategy class that generates trade actions based on Neural Network predictions.
    
    This strategy implements the core logic for:
    - Loading NN models and scalers for inference
    - Preparing input data from backtester feed
    - Performing model inference at each time step
    - Entering LONG positions when NN prediction probability exceeds signal_threshold
    - Exiting LONG positions based on time limits or probability thresholds
    - Managing position states (FLAT, LONG)
    
    The strategy is designed to work with the backtesting engine and receives
    NN model prediction probabilities to make trading decisions.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SupervisedNNStrategy with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing strategy parameters.
                Expected keys:
                - signal_threshold (float, optional): Minimum probability to trigger BUY signal. Default: 0.7
                - exit_threshold (float, optional): Probability below which to exit LONG position. Default: None
                - max_holding_period_hours (int, optional): Maximum hours to hold a position. Default: 8
                - model_path (str, optional): Path to the PyTorch model file (required for inference)
                - scaler_path (str, optional): Path to the scaler file (required for inference)
                - asset_id_map_path (str, optional): Path to the asset ID mapping JSON file
                - feature_list (list, optional): List of feature names to use (required for inference)
                - lookback_window (int, optional): Number of historical bars to use for input sequence
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model-related attributes
        self.model = None
        self.scaler = None
        self.asset_id_map = None
        self.latest_prediction_probabilities = None # To store the latest prediction probability
        self.latest_true_label = None # To store the latest true label (if available)
        
        # Log strategy initialization
        self.logger.info(f"Initializing SupervisedNNStrategy with config: {config}")
        
        # Load model dependencies if paths are provided
        if self.config.get('model_path') and self.config.get('scaler_path'):
            self.load_dependencies()
        else:
            self.logger.info("Model/scaler paths not provided - dependencies not loaded. Call load_dependencies() manually if needed.")
    
    def load_dependencies(self) -> None:
        """
        Load the PyTorch model, scaler, and optionally asset ID mapping from file paths.
        
        This method loads all required dependencies for model inference:
        - PyTorch model from model_path
        - Scaler object from scaler_path  
        - Asset ID mapping from asset_id_map_path (if provided)
        
        Raises:
            FileNotFoundError: If required model or scaler files are not found
            Exception: If there are errors loading the model or scaler
        """
        try:
            # Load PyTorch model
            model_path = self.config.get('model_path')
            if not model_path:
                raise ValueError("model_path must be specified in config")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = torch.load(model_path, map_location=device, weights_only=False)
            self.model.eval()
            self.logger.info(f"Successfully loaded model from {model_path} on device: {device}")
            
            # Load scaler
            scaler_path = self.config.get('scaler_path')
            if not scaler_path:
                raise ValueError("scaler_path must be specified in config")
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Successfully loaded scaler from {scaler_path}")
            
            # Load asset ID mapping (optional)
            asset_id_map_path = self.config.get('asset_id_map_path')
            if asset_id_map_path and os.path.exists(asset_id_map_path):
                with open(asset_id_map_path, 'r') as f:
                    asset_id_data = json.load(f)
                    self.asset_id_map = asset_id_data.get('symbol_to_id', {})
                self.logger.info(f"Successfully loaded asset ID mapping from {asset_id_map_path}")
            else:
                self.asset_id_map = None
                if asset_id_map_path:
                    self.logger.warning(f"Asset ID mapping file not found: {asset_id_map_path}")
                else:
                    self.logger.info("No asset ID mapping path provided")
                    
        except Exception as e:
            self.logger.error(f"Error loading dependencies: {str(e)}")
            raise

    def load_dependencies_from_mlflow(self, run_id: str, artifact_path: str = "best_model") -> None:
        """
        Load model, scaler, and asset ID mapping from MLflow artifacts.
        
        This method enables loading trained models directly from MLflow tracking server,
        supporting Activity 7.3: Integration with Backtesting Framework.
        
        Args:
            run_id: MLflow run ID containing the model artifacts
            artifact_path: Path to model artifact within the run (default: "best_model")
            
        Raises:
            ImportError: If MLflow is not available
            FileNotFoundError: If artifacts cannot be found
            Exception: If there are errors loading the artifacts
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not available. Install mlflow to use this feature.")
        
        try:
            self.logger.info(f"Loading dependencies from MLflow run: {run_id}")
            
            # Download model artifact
            model_uri = f"runs:/{run_id}/{artifact_path}"
            self.logger.info(f"Downloading model from: {model_uri}")
            
            # Load PyTorch model from MLflow
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = mlflow.pytorch.load_model(model_uri, map_location=device)
            self.model.eval()
            self.logger.info(f"Successfully loaded model from MLflow on device: {device}")
            
            # Try to load scaler and asset_id_map from run artifacts
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            
            # Download artifacts directory
            artifact_dir = client.download_artifacts(run_id, "")
            artifact_dir_path = Path(artifact_dir)
            
            # Load scaler if available
            scaler_paths = list(artifact_dir_path.glob("**/scalers.joblib"))
            if scaler_paths:
                scaler_path = scaler_paths[0]
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"Successfully loaded scaler from MLflow artifacts")
            else:
                self.logger.warning("No scaler found in MLflow artifacts")
            
            # Load asset ID mapping if available
            asset_map_paths = list(artifact_dir_path.glob("**/asset_id_mapping.json"))
            if asset_map_paths:
                with open(asset_map_paths[0], 'r') as f:
                    asset_id_data = json.load(f)
                    self.asset_id_map = asset_id_data.get('symbol_to_id', {})
                self.logger.info(f"Successfully loaded asset ID mapping from MLflow artifacts")
            else:
                self.asset_id_map = None
                self.logger.warning("No asset ID mapping found in MLflow artifacts")
            
            # Extract configuration from run parameters if available
            if run.data.params:
                self.logger.info("Loading configuration from MLflow run parameters...")
                # Update config with MLflow parameters
                if 'lookback_window' in run.data.params:
                    self.config['lookback_window'] = int(run.data.params['lookback_window'])
                if 'signal_threshold' in run.data.params:
                    self.config.setdefault('signal_threshold', float(run.data.params.get('signal_threshold', 0.7)))
                    
        except Exception as e:
            self.logger.error(f"Error loading dependencies from MLflow: {str(e)}")
            raise
    
    def prepare_input_sequence(
        self, 
        historical_data: pd.DataFrame, 
        symbol: str = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare input sequence from historical data for model inference.
        
        This method:
        1. Selects required features from historical data
        2. Ensures sufficient data for lookback window
        3. Applies scaling transformation
        4. Reshapes data for model input
        5. Handles asset ID if mapping is available
        
        Args:
            historical_data (pd.DataFrame): Historical market data with features
            symbol (str, optional): Symbol for asset ID lookup
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Feature tensor shaped (1, lookback_window, n_features)
                - Asset ID tensor shaped (1,) or None if no mapping
        """
        try:
            # Get configuration parameters
            feature_list = self.config.get('feature_list', [])
            lookback_window = self.config.get('lookback_window', 24)
            
            if not feature_list:
                raise ValueError("feature_list must be specified in config")
            
            # Check if we have enough data
            if len(historical_data) < lookback_window:
                self.logger.warning(
                    f"Insufficient historical data: {len(historical_data)} < {lookback_window} required"
                )
                return None, None
            
            # Select features and get the last lookback_window rows
            try:
                feature_data = historical_data[feature_list].tail(lookback_window)
            except KeyError as e:
                missing_features = set(feature_list) - set(historical_data.columns)
                raise ValueError(f"Missing features in historical data: {missing_features}")
            
            # Apply scaling
            if self.scaler is None:
                raise RuntimeError("Scaler not loaded. Call load_dependencies() first.")
            
            # Convert to numpy and apply scaling
            feature_array = feature_data.values  # Shape: (lookback_window, n_features)
            scaled_features = self.scaler.transform(feature_array)
            
            # Reshape for model input: (1, lookback_window, n_features)
            feature_tensor = torch.tensor(
                scaled_features.reshape(1, lookback_window, -1), 
                dtype=torch.float32
            )
            
            # Handle asset ID
            asset_id_tensor = None
            if self.asset_id_map and symbol:
                if symbol in self.asset_id_map:
                    asset_id = self.asset_id_map[symbol]
                    asset_id_tensor = torch.tensor([asset_id], dtype=torch.long)
                    self.logger.debug(f"Asset ID for {symbol}: {asset_id}")
                else:
                    self.logger.warning(f"Symbol {symbol} not found in asset ID mapping")
            
            self.logger.debug(
                f"Prepared input sequence: features shape {feature_tensor.shape}, "
                f"asset_id: {asset_id_tensor is not None}"
            )
            
            return feature_tensor, asset_id_tensor
            
        except Exception as e:
            self.logger.error(f"Error preparing input sequence: {str(e)}")
            return None, None
    
    def get_model_prediction(
        self, 
        feature_tensor: torch.Tensor, 
        asset_id_tensor: Optional[torch.Tensor] = None
    ) -> float:
        """
        Perform model inference and return prediction probability.
        
        This method:
        1. Ensures model is in eval mode
        2. Performs forward pass with or without asset ID
        3. Applies appropriate activation (sigmoid/softmax) if needed
        4. Extracts and returns the relevant probability
        
        Args:
            feature_tensor (torch.Tensor): Input features shaped (1, lookback_window, n_features)
            asset_id_tensor (torch.Tensor, optional): Asset ID tensor shaped (1,)
            
        Returns:
            float: Prediction probability for BUY_SIGNAL class [0.0, 1.0]
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_dependencies() first.")
            
            # Ensure model is in eval mode
            self.model.eval()
            
            # Perform inference without gradient computation
            with torch.no_grad():
                # Forward pass - adapt based on model architecture
                if asset_id_tensor is not None:
                    # Model expects both features and asset IDs
                    try:
                        output = self.model(feature_tensor, asset_id_tensor)
                    except TypeError:
                        # Fallback: model might not support asset IDs
                        self.logger.warning("Model does not support asset IDs, using features only")
                        output = self.model(feature_tensor)
                else:
                    # Model expects only features
                    output = self.model(feature_tensor)
                
                # Handle different output formats
                if output.dim() == 2 and output.size(1) > 1:
                    # Multi-class output - apply softmax and take first class probability
                    probabilities = torch.softmax(output, dim=1)
                    prediction_probability = probabilities[0, 0].item()  # First class (BUY_SIGNAL)
                elif output.dim() == 2 and output.size(1) == 1:
                    # Single output - apply sigmoid
                    prediction_probability = torch.sigmoid(output[0, 0]).item()
                else:
                    # Single scalar output - apply sigmoid
                    prediction_probability = torch.sigmoid(output).item()
                
                self.logger.debug(f"Model prediction probability: {prediction_probability:.4f}")
                
                return prediction_probability
                
        except Exception as e:
            self.logger.error(f"Error during model inference: {str(e)}")
            # Return neutral probability on error
            return 0.5
    
    def generate_trade_action(
        self, 
        prediction_probability: float, 
        current_position_status: str, 
        time_in_position_hours: int = 0
    ) -> str:
        """
        Generate trade action based on NN prediction and current position status.
        
        This method implements the core trading logic:
        1. If FLAT and prediction >= signal_threshold: BUY
        2. If LONG and time >= max_holding_period: SELL
        3. If LONG and prediction < exit_threshold (if configured): SELL
        4. Otherwise: HOLD
        
        Args:
            prediction_probability (float): NN model output probability for BUY_SIGNAL [0.0, 1.0]
            current_position_status (str): Current position status ("FLAT", "LONG")
            time_in_position_hours (int, optional): Hours the current position has been held. Default: 0
            
        Returns:
            str: Trade action ("BUY", "SELL", "HOLD")
        """
        # Retrieve configuration parameters with defaults
        signal_threshold = self.config.get('signal_threshold', 0.7)
        exit_threshold = self.config.get('exit_threshold')  # Can be None
        max_holding_period_hours = self.config.get('max_holding_period_hours', 8)
        
        # Log input parameters for debugging
        self.logger.debug(
            f"generate_trade_action called: prob={prediction_probability:.4f}, "
            f"position={current_position_status}, time_in_pos={time_in_position_hours}h"
        )
        
        # Entry logic: FLAT -> BUY
        if current_position_status == "FLAT":
            if prediction_probability >= signal_threshold:
                self.logger.info(
                    f"BUY signal: probability {prediction_probability:.4f} >= threshold {signal_threshold}"
                )
                return "BUY"
            else:
                self.logger.debug(
                    f"HOLD (FLAT): probability {prediction_probability:.4f} < threshold {signal_threshold}"
                )
                return "HOLD"
        
        # Exit logic: LONG -> SELL
        elif current_position_status == "LONG":
            # Exit due to maximum holding period
            if time_in_position_hours >= max_holding_period_hours:
                self.logger.info(
                    f"SELL signal: max holding period reached ({time_in_position_hours}h >= {max_holding_period_hours}h)"
                )
                return "SELL"
            
            # Exit due to probability dropping below exit threshold (if configured)
            if exit_threshold is not None and prediction_probability <= exit_threshold:
                self.logger.info(
                    f"SELL signal: probability {prediction_probability:.4f} <= exit_threshold {exit_threshold}"
                )
                return "SELL"
            
            # Continue holding
            self.logger.debug(
                f"HOLD (LONG): time={time_in_position_hours}h < max={max_holding_period_hours}h, "
                f"prob={prediction_probability:.4f}"
            )
            return "HOLD"
        
        # Default case: HOLD for any unrecognized position status
        else:
            self.logger.warning(f"Unknown position status: {current_position_status}. Defaulting to HOLD.")
            return "HOLD"
    
    def on_bar_data(
        self, 
        bar_data: Dict[str, Any], 
        historical_window_df: pd.DataFrame, 
        current_portfolio_status: Dict[str, Any]
    ) -> str:
        """
        Main entry point called by the backtester for each bar.
        
        This method orchestrates the complete inference pipeline:
        1. Extract symbol from bar data
        2. Prepare input sequence from historical data
        3. Perform model inference if sufficient data
        4. Generate trade action based on prediction and current position
        
        Args:
            bar_data (Dict[str, Any]): Current bar data containing symbol and OHLCV
            historical_window_df (pd.DataFrame): Historical data window with features
            current_portfolio_status (Dict[str, Any]): Current portfolio state
                Expected keys:
                - position_type (str): Current position ("FLAT", "LONG")
                - time_in_position_hours (int): Hours in current position
                
        Returns:
            str: Trade action ("BUY", "SELL", "HOLD")
        """
        try:
            # Extract symbol from bar data
            symbol = bar_data.get('symbol')
            if not symbol:
                self.logger.warning("No symbol found in bar_data")
                return "HOLD"
            
            # Prepare input sequence
            feature_tensor, asset_id_tensor = self.prepare_input_sequence(
                historical_window_df, symbol
            )
            
            # Check if we have valid input data
            if feature_tensor is None:
                self.logger.warning("Insufficient data for input sequence, returning HOLD")
                return "HOLD"
            
            # Get model prediction
            prediction_probability = self.get_model_prediction(feature_tensor, asset_id_tensor)
            
            # Store for metrics and optimization
            self.latest_prediction_probabilities = prediction_probability
            # Assuming true_label can be passed in bar_data or derived.
            # For now, we'll assume it's part of bar_data for simplicity in this backtest context.
            # In a real validation scenario, this would come from your validation dataset.
            self.latest_true_label = bar_data.get('true_label', None) # Assuming 'true_label' might be in bar_data
            
            # Extract current portfolio status
            current_position = current_portfolio_status.get('position_type', 'FLAT')
            time_in_position = current_portfolio_status.get('time_in_position_hours', 0)
            
            # Generate trade action
            action = self.generate_trade_action(
                prediction_probability, current_position, time_in_position
            )
            
            # Log the decision process
            self.logger.info(
                f"Bar processed - Symbol: {symbol}, Prediction: {prediction_probability:.4f}, "
                f"Position: {current_position}, Time: {time_in_position}h, Action: {action}"
            )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error in on_bar_data: {str(e)}")
            return "HOLD"