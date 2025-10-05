#!/usr/bin/env python3
"""
Neural Network Training Script for Trading Signal Prediction

This script implements the training infrastructure and core training loop for custom
neural network models as specified in Section 3 (Activity 3.5) and Section 4 
(Activities 4.1-4.7) of the operational plan.

Key Features:
- Configurable via CLI arguments or YAML configuration
- Support for multiple model architectures (MLP, LSTM+Attention, GRU+Attention, CNN-LSTM)
- Focal Loss and Weighted Binary Cross-Entropy loss functions
- AdamW and Adam optimizers with learning rate scheduling
- Comprehensive metrics tracking and MLflow integration
- Early stopping and model checkpointing
- GPU optimization integration

Author: Flow-Code
Date: 2025-05-28
Version: 1.0
"""

import argparse
import logging
import os
import platform
import shutil
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

# Configure logging early so optional dependency fallbacks can use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    log_loss
)
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models.nn_architectures import create_model, get_model_info
from core.data_preparation_nn import NNDataPreparer
from utils.gpu_utils import setup_gpu
# Import new experiment management modules
from core.experiment_management.config_manager import ConfigurationManager
from core.experiment_management.enhanced_logging import EnhancedMLflowLogger
from core.experiment_management.experiment_organizer import ExperimentOrganizer
from core.experiment_management.reporting import ExperimentReporter
import joblib

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

# MLflow integration
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Experiment tracking will be disabled.")

# Optuna integration for HPO
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. HPO features will be disabled.")


def _detect_torch_compile_support() -> bool:
    """Return True if the environment has the required compiler for torch.compile."""
    if not hasattr(torch, 'compile'):
        return False

    system = platform.system().lower()

    if system == 'windows':
        return shutil.which('cl') is not None
    if system == 'linux':
        # GCC or Clang is acceptable; check for either to cover most distros
        return shutil.which('gcc') is not None or shutil.which('clang') is not None
    if system == 'darwin':
        return shutil.which('clang') is not None

    return False

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Focal Loss focuses learning on hard examples by down-weighting
    easy examples. Particularly effective for imbalanced datasets.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (typically 0.25-0.75)
            gamma: Focusing parameter (typically 1.0-3.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions of shape (batch_size, 1) or (batch_size,)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Focal loss value
        """
        # Ensure inputs are properly shaped
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
        
        # Compute binary cross entropy with logits
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        
        # Compute p_t (probabilities) by applying sigmoid to inputs (logits)
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading sequence data.
    
    Handles loading and batching of sequence features, labels, and asset IDs
    produced by NNDataPreparer.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        asset_ids: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ):
        """
        Initialize trading dataset.
        
        Args:
            features: Feature sequences of shape (n_samples, lookback_window, n_features)
            labels: Binary labels of shape (n_samples,)
            asset_ids: Asset IDs of shape (n_samples,)
            sample_weights: Optional sample weights of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.asset_ids = torch.LongTensor(asset_ids)
        self.sample_weights = torch.FloatTensor(sample_weights) if sample_weights is not None else None
        
        # Validate shapes
        assert len(self.features) == len(self.labels) == len(self.asset_ids)
        if self.sample_weights is not None:
            assert len(self.sample_weights) == len(self.features)
            
        logger.info(f"Dataset initialized with {len(self)} samples")
        logger.info(f"Feature shape: {self.features.shape}")
        logger.info(f"Label distribution: {torch.bincount(self.labels)}")
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label, asset_id)
        """
        return self.features[idx], self.labels[idx], self.asset_ids[idx]

class ModelTrainer:
    """
    Main training class for neural network models.
    
    Handles the complete training pipeline including data loading,
    model training, evaluation, and checkpointing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = GradScaler('cuda', enabled=torch.cuda.is_available() and self.config.get('training_config', {}).get('use_amp', True))
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # Initialize experiment management components
        self.enhanced_logger = None
        self.experiment_organizer = None
        self.reporter = None
        
        # Setup GPU optimization
        if torch.cuda.is_available():
            gpu_info = setup_gpu()
            logger.info(f"GPU setup: {gpu_info}")
        
        # Setup enhanced experiment management if available
        if MLFLOW_AVAILABLE and config.get('use_mlflow', True):
            self._setup_enhanced_experiment_management()
            
    def _setup_enhanced_experiment_management(self):
        """Setup enhanced experiment management with new modules."""
        try:
            # Initialize experiment organizer
            from core.experiment_management.experiment_organizer import (
                ExperimentMetadata, ExperimentStage, ExperimentType
            )
            
            self.experiment_organizer = ExperimentOrganizer()
            
            # Create experiment metadata
            model_type = self.config.get('model_type', 'unknown')
            parent_run_id = self.config.get('parent_run_id')
            
            # Determine experiment type and stage
            if parent_run_id:
                exp_type = ExperimentType.HPO_TRIAL
                purpose = 'hpo_trial'
            else:
                exp_type = ExperimentType.SINGLE_TRAINING
                purpose = self.config.get('purpose', 'training')
            
            metadata = ExperimentMetadata(
                stage=ExperimentStage.DEVELOPMENT,
                type=exp_type,
                model_type=model_type,
                dataset_version='v1.0',
                purpose=purpose,
                parent_run_id=parent_run_id,
                custom_tags=self.config.get('tags', {})
            )
            
            # Generate experiment and run names
            experiment_name = self.experiment_organizer.generate_experiment_name(metadata)
            run_name = self.experiment_organizer.generate_run_name(metadata)
            
            # Initialize enhanced logger
            self.enhanced_logger = EnhancedMLflowLogger(
                experiment_name=experiment_name,
                run_name=run_name
            )
            
            # Start logging session
            self.enhanced_logger.start_run()
            
            # Note: Environment and git info are logged automatically in start_run()
            # Log configuration
            try:
                self.enhanced_logger.log_configuration(self.config)
            except Exception as e:
                logger.warning(f"Could not log configuration: {e}")
            
            # Create and apply comprehensive tags
            experiment_tags = self.experiment_organizer.create_comprehensive_tags(metadata, self.config)
            mlflow.set_tags(experiment_tags)
            
            # Initialize reporter
            self.reporter = ExperimentReporter()
            
            logger.info(f"Enhanced experiment management initialized")
            logger.info(f"Experiment: {experiment_name}")
            logger.info(f"Run: {run_name}")
            
        except Exception as e:
            logger.warning(f"Failed to setup enhanced experiment management: {e}")
            # Fallback to basic MLflow setup
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Fallback MLflow setup for compatibility with robust experiment handling."""
        experiment_name = self.config.get('experiment_name', 'nn_training')
        
        try:
            # Robust experiment setup: handle missing and deleted experiments
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                # Experiment doesn't exist, create it
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            elif experiment.lifecycle_stage == 'deleted':
                # Experiment was deleted, restore it
                mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
                experiment_id = experiment.experiment_id
                logger.info(f"Restored deleted MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                # Experiment exists and is active
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
            
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            
            # Enhanced run naming with model type and timestamp
            model_type = self.config['model_type']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{model_type}_{timestamp}"
            mlflow.start_run(run_name=run_name)
            
            # Log configuration
            mlflow.log_params(self.config)
            
            # Log git commit if available
            try:
                import subprocess
                git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
                mlflow.log_param('git_commit', git_commit)
            except Exception as e:
                logger.warning(f"Could not log git commit: {e}")
                mlflow.log_param('git_commit', 'unknown')
                
            # Log environment details
            mlflow.log_params({
                'python_version': sys.version.split()[0],
                'pytorch_version': torch.__version__,
                'numpy_version': np.__version__,
                'mlflow_version': mlflow.__version__
            })
            
            # Log GPU details if available
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
                mlflow.log_params({
                    'gpu_name': gpu_name,
                    'gpu_memory_gb': f"{gpu_memory:.2f}",
                    'gpu_count': torch.cuda.device_count()
                })
            
            # Set comprehensive tags
            tags = {
                'model_type': model_type,
                'purpose': self.config.get('purpose', 'training'),
                'status': 'running',
                'task': 'binary_classification'
            }
            if 'tags' in self.config:
                tags.update(self.config['tags'])
            mlflow.set_tags(tags)
            
            logger.info(f"MLflow tracking initialized for experiment: {experiment_name}")
            logger.info(f"Run name: {run_name}")
            logger.info(f"Tags: {tags}")
            
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            # Continue training without MLflow
            self.config['use_mlflow'] = False
    
    def load_data(self) -> None:
        """Load and prepare training data."""
        logger.info("Loading training data...")
        
        # Check if we should use pre-generated training data
        use_pregenerated = self.config.get('use_pregenerated_data', False)
        pregenerated_data_path = self.config.get('pregenerated_data_path', 'data/training_data')
          # Resolve path relative to project root if it's a relative path
        if not Path(pregenerated_data_path).is_absolute():
            # Get project root (parent of training directory)
            project_root = Path(__file__).parent.parent
            resolved_path = project_root / pregenerated_data_path
            
            # If the resolved path doesn't exist, try interpreting it as relative to project root directly
            if not resolved_path.exists():
                # Try removing the leading "../" if present
                cleaned_path = pregenerated_data_path
                if cleaned_path.startswith('../'):
                    cleaned_path = cleaned_path[3:]
                alternative_path = project_root / cleaned_path
                if alternative_path.exists():
                    resolved_path = alternative_path
        else:
            resolved_path = Path(pregenerated_data_path)
        
        if use_pregenerated and resolved_path.exists():
            logger.info(f"Loading pre-generated training data from {resolved_path}")
            prepared_data = self._load_pregenerated_data(str(resolved_path))
        else:
            if use_pregenerated:
                logger.warning(f"Pre-generated data path {resolved_path} not found, generating data on-the-fly")
                logger.info(f"Original path: {pregenerated_data_path}")
                logger.info(f"Resolved path: {resolved_path}")
                logger.info(f"Path exists: {resolved_path.exists()}")
            
            # Initialize data preparer
            data_config = self.config['data_config']
            data_preparer = NNDataPreparer(data_config)
            
            # Get prepared data
            prepared_data = data_preparer.get_prepared_data_for_training()
        
        # Extract data splits
        train_data = prepared_data['train']
        val_data = prepared_data['validation']
        test_data = prepared_data['test']
        
        # Create datasets
        train_dataset = TradingDataset(
            train_data['X'], train_data['y'], train_data['asset_ids'],
            train_data.get('sample_weights')
        )
        
        val_dataset = TradingDataset(
            val_data['X'], val_data['y'], val_data['asset_ids']
        )
        
        test_dataset = TradingDataset(
            test_data['X'], test_data['y'], test_data['asset_ids']
        )
        
        # Create data loaders
        batch_size = self.config['training_config']['batch_size']
        
        # Handle class imbalance with weighted sampling if specified
        if self.config['training_config'].get('use_weighted_sampling', False) and train_data.get('sample_weights') is not None:
            sampler = WeightedRandomSampler(
                weights=train_data['sample_weights'],
                num_samples=len(train_dataset),
                replacement=True
            )
            shuffle = False # Sampler handles shuffling
        else:
            sampler = None
            shuffle = True

        # Determine drop_last for training loader
        train_drop_last = False
        if self.config.get('model_type') == 'cnn_lstm':
            train_drop_last = True
            logger.info("Using drop_last=True for training DataLoader for CNN_LSTM to avoid BatchNorm1d issues with batch size 1.")
        
        # Optimization: Add prefetch_factor for better GPU utilization
        num_workers = self.config.get('num_workers', 4)
        prefetch_factor = 2 if num_workers > 0 else None
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            drop_last=train_drop_last,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor
        )
        
        # Store scalers and asset mapping for later use
        self.scalers = prepared_data['scalers']
        self.asset_id_map = prepared_data['asset_id_map']
        
        # Update model configuration with actual data dimensions
        train_X_shape = train_data['X'].shape
        self.config['model_config']['n_features'] = train_X_shape[2]  # Number of features
        self.config['model_config']['lookback_window'] = train_X_shape[1]  # Sequence length
        self.config['model_config']['num_assets'] = len(self.asset_id_map)  # Number of unique assets
        
        logger.info(f"Updated model config with data dimensions:")
        logger.info(f"  Features: {self.config['model_config']['n_features']}")
        logger.info(f"  Lookback window: {self.config['model_config']['lookback_window']}")
        logger.info(f"  Number of assets: {self.config['model_config']['num_assets']}")
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        
        # Enhanced data logging if available
        if self.enhanced_logger:
            # Prepare data info for each split
            train_info = {
                'X': train_data['X'],
                'y': train_data['y'],
                'n_samples': len(train_dataset),
                'class_distribution': np.bincount(train_data['y']).tolist()
            }
            val_info = {
                'X': val_data['X'],
                'y': val_data['y'],
                'n_samples': len(val_dataset),
                'class_distribution': np.bincount(val_data['y']).tolist()
            }
            test_info = {
                'X': test_data['X'],
                'y': test_data['y'],
                'n_samples': len(test_dataset),
                'class_distribution': np.bincount(test_data['y']).tolist()
            }
            
            try:
                self.enhanced_logger.log_data_info(train_info, val_info, test_info)
            except Exception as e:
                logger.warning(f"Failed to log data info: {e}")
            
            # Log scalers if available
            if self.scalers:
                try:
                    self.enhanced_logger.log_scalers(self.scalers)
                except Exception as e:
                    logger.warning(f"Failed to log scalers: {e}")
    
    def _load_pregenerated_data(self, data_path: str) -> Dict[str, Any]:
        """
        Load pre-generated training data from disk.
        
        Args:
            data_path: Path to the directory containing pre-generated data
            
        Returns:
            Dict[str, Any]: Dictionary containing loaded data splits and metadata
        """
        import json
        
        data_dir = Path(data_path)
        logger.info(f"Loading pre-generated data from {data_dir}")
        
        # Load metadata
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Data generated on: {metadata.get('generation_timestamp', 'unknown')}")
            logger.info(f"Symbols processed: {metadata.get('num_symbols', 'unknown')}")
        
        # Load data splits
        prepared_data = {}
        
        # Load train data
        train_X = np.load(data_dir / "train_X.npy")
        train_y = np.load(data_dir / "train_y.npy")
        train_asset_ids = np.load(data_dir / "train_asset_ids.npy")
        
        prepared_data['train'] = {
            'X': train_X,
            'y': train_y,
            'asset_ids': train_asset_ids
        }
        
        # Load validation data
        val_X = np.load(data_dir / "val_X.npy")
        val_y = np.load(data_dir / "val_y.npy")
        val_asset_ids = np.load(data_dir / "val_asset_ids.npy")
        
        prepared_data['validation'] = {
            'X': val_X,
            'y': val_y,
            'asset_ids': val_asset_ids
        }
        
        # Load test data
        test_X = np.load(data_dir / "test_X.npy")
        test_y = np.load(data_dir / "test_y.npy")
        test_asset_ids = np.load(data_dir / "test_asset_ids.npy")
        
        prepared_data['test'] = {
            'X': test_X,
            'y': test_y,
            'asset_ids': test_asset_ids
        }
        
        # Load scalers
        scalers_path = data_dir / "scalers.joblib"
        if scalers_path.exists():
            prepared_data['scalers'] = joblib.load(scalers_path)
        else:
            logger.warning("No scalers found in pre-generated data")
            prepared_data['scalers'] = {}
        
        # Load asset ID mapping
        asset_map_path = data_dir / "asset_id_mapping.json"
        if asset_map_path.exists():
            with open(asset_map_path, 'r') as f:
                prepared_data['asset_id_map'] = json.load(f)
        else:
            logger.warning("No asset ID mapping found in pre-generated data")
            prepared_data['asset_id_map'] = {}
        logger.info(f"Loaded pre-generated data:")
        logger.info(f"  Train: {train_X.shape}")
        logger.info(f"  Val: {val_X.shape}")
        logger.info(f"  Test: {test_X.shape}")
        
        return prepared_data
    def create_model(self) -> None:
        """Create and initialize the neural network model."""
        logger.info("Creating model...")
        
        model_config = self.config['model_config'].copy()
        model_type = self.config['model_type']
        
        # Create model
        self.model = create_model(model_type, model_config)
        self.model.to(self.device)
        
        # Apply torch.compile() for PyTorch 2.0+ optimization
        use_compile = self.config.get('training_config', {}).get('use_torch_compile', True)
        compile_supported = _detect_torch_compile_support()

        if use_compile and compile_supported and torch.cuda.is_available():
            try:
                logger.info("Applying torch.compile() for performance optimization...")
                self.model = torch.compile(self.model, mode='default')
                logger.info("torch.compile() applied successfully")
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile(): {e}. Continuing without compilation.")
        elif use_compile and not compile_supported:
            logger.info("Skipping torch.compile(): required native toolchain not detected. Set training_config.use_torch_compile=False to silence this message.")
        
        # Log model information
        model_info = get_model_info(self.model)
        logger.info(f"Model created: {model_info['model_type']}")
        logger.info(f"Parameters: {model_info['trainable_parameters']:,}")
        logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
        
        # Enhanced logging if available
        if self.enhanced_logger:
            # Log model architecture details
            try:
                self.enhanced_logger.log_model_architecture(
                    model=self.model,
                    input_shape=(1, model_config['lookback_window'], model_config['n_features']),
                    config=model_config
                )
            except Exception as e:
                logger.warning(f"Could not log model architecture: {e}")
        elif MLFLOW_AVAILABLE and self.config.get('use_mlflow', True):
            # Fallback to basic MLflow logging (only if MLflow is enabled)
            mlflow.log_params({
                'model_parameters': model_info['trainable_parameters'],
                'model_size_mb': model_info['model_size_mb']
            })
    
    def create_optimizer_and_scheduler(self) -> None:
        """Create optimizer and learning rate scheduler."""
        training_config = self.config['training_config']
        
        # Create optimizer
        optimizer_type = training_config.get('optimizer', 'adamw').lower()
        lr = training_config['learning_rate']

        # Enforce optional minimum learning rate to avoid pathological values from HPO
        min_learning_rate = training_config.get('min_learning_rate')
        if min_learning_rate is not None and lr < min_learning_rate:
            logger.warning(
                "Configured learning rate %.2e is below min_learning_rate %.2e. "
                "Overriding with the minimum value for stability.",
                lr,
                min_learning_rate
            )
            lr = float(min_learning_rate)
            training_config['learning_rate'] = lr
        weight_decay = training_config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=training_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=training_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Create scheduler
        scheduler_type = training_config.get('scheduler', 'reduce_on_plateau').lower()
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize validation F1
                factor=training_config.get('scheduler_factor', 0.5),
                patience=training_config.get('scheduler_patience', 5)
            )
        elif scheduler_type == 'cosine_annealing':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['epochs'],
                eta_min=lr * 0.01
            )
        elif scheduler_type == 'one_cycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                epochs=training_config['epochs'],
                steps_per_epoch=len(self.train_loader)
            )
        else:
            self.scheduler = None
        
        logger.info(f"Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")
    
    def create_loss_function(self) -> None:
        """Create loss function."""
        loss_config = self.config['training_config']['loss_function']
        loss_type = loss_config['type'].lower()
        
        if loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=loss_config.get('alpha', 0.25),
                gamma=loss_config.get('gamma', 2.0)
            )
        elif loss_type == 'weighted_bce':
            # Calculate class weights from training data
            train_labels = []
            for _, labels, _ in self.train_loader:
                train_labels.extend(labels.numpy())
            
            pos_weight = torch.tensor(
                (len(train_labels) - sum(train_labels)) / sum(train_labels)
            ).to(self.device)
            
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
        logger.info(f"Loss function: {loss_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Check if in HPO mode
        is_hpo_mode = hasattr(self, '_in_hpo_mode') and self._in_hpo_mode
        
        # Track timing for speed calculations
        epoch_start_time = time.time()        # Create enhanced progress bar
        if is_hpo_mode:
            # For HPO mode, disable the visual progress bar but keep the iterator
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}",
                        disable=True, leave=False)
        else:
            # Full enhanced progress bar for normal mode
            pbar = tqdm(self.train_loader, desc=f"[Training] Epoch {self.current_epoch+1}",
                        leave=False, ncols=120, file=sys.stdout, ascii=True, unit='batch',
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        for batch_idx, (features, labels, asset_ids) in enumerate(pbar):
            batch_start_time = time.time()
            
            # Move to device with non_blocking for better pipeline performance
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            asset_ids = asset_ids.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True) # Optimization: set_to_none=True
            
            # Forward pass with autocast for AMP
            use_amp_runtime = torch.cuda.is_available() and self.config.get('training_config', {}).get('use_amp', True)
            with autocast('cuda', enabled=use_amp_runtime):
                outputs = self.model(features, asset_ids)
                if outputs.ndim > 1 and outputs.shape[-1] == 1: # Ensure it's (batch, 1) before squeeze
                    outputs = outputs.squeeze(-1)
                loss = self.criterion(outputs, labels.float())
            
            # Backward pass with GradScaler
            self.scaler.scale(loss).backward()
            
            # Optimizer step with GradScaler
            if self.config['training_config'].get('gradient_clip_norm'):
                self.scaler.unscale_(self.optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training_config']['gradient_clip_norm']
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update scheduler if OneCycleLR
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Accumulate metrics
            total_loss += loss.detach().item()
            current_loss = loss.detach().item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Convert outputs (logits) to probabilities for metrics
            # This is always needed now since models output logits
            probs = torch.sigmoid(outputs.detach())
            
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Calculate real-time metrics for display
            batch_time = time.time() - batch_start_time
            
            # Calculate accuracy for current batch
            batch_probs = probs.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            batch_preds = (batch_probs >= 0.5).astype(int)
            batch_accuracy = accuracy_score(batch_labels, batch_preds) * 100
            
            # Calculate processing speed
            samples_per_sec = len(features) / batch_time if batch_time > 0 else 0
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Enhanced progress bar updates
            if is_hpo_mode:
                # Update every 50 batches in HPO mode for reduced overhead
                if batch_idx % 50 == 0:
                    pbar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Avg': f'{avg_loss:.4f}',
                        'Acc': f'{batch_accuracy:.1f}%'
                    })
            else:
                # Comprehensive real-time updates for normal mode
                elapsed_time = time.time() - epoch_start_time
                batches_remaining = len(self.train_loader) - (batch_idx + 1)
                avg_batch_time = elapsed_time / (batch_idx + 1) if batch_idx > 0 else batch_time
                eta_seconds = batches_remaining * avg_batch_time
                
                # Format ETA
                if eta_seconds > 3600:
                    eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
                elif eta_seconds > 60:
                    eta_str = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s"
                else:
                    eta_str = f"{int(eta_seconds)}s"
                
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Avg': f'{avg_loss:.4f}',
                    'Acc': f'{batch_accuracy:.1f}%',
                    'LR': f'{current_lr:.2e}',
                    'Speed': f'{samples_per_sec:.0f}s/s',
                    'ETA': eta_str
                })
          # Ensure progress bar is properly closed
        pbar.close()
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = self._compute_metrics(all_predictions, all_labels, avg_loss)
        
        return metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]: # Added epoch parameter
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        # Check if in HPO mode
        is_hpo_mode = hasattr(self, '_in_hpo_mode') and self._in_hpo_mode
          # Track timing for speed calculations
        val_start_time = time.time()
        
        # Create enhanced progress bar for validation
        if is_hpo_mode:
            # For HPO mode, disable the visual progress bar but keep the iterator
            pbar = tqdm(self.val_loader, desc="Validation", 
                        disable=True, leave=False)
        else:
            # Full enhanced progress bar for normal mode
            pbar = tqdm(self.val_loader, desc="[Validation]", leave=False, 
                        ncols=110, ascii=True, file=sys.stdout, unit='batch',
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        with torch.no_grad():
            for batch_idx, (features, labels, asset_ids) in enumerate(pbar):
                batch_start_time = time.time()
                
                # Move to device
                features = features.to(self.device, non_blocking=True) # Optimization: non_blocking
                labels = labels.to(self.device, non_blocking=True)     # Optimization: non_blocking
                asset_ids = asset_ids.to(self.device, non_blocking=True) # Optimization: non_blocking for asset_ids
                
                # Forward pass with autocast for AMP
                use_amp_runtime = torch.cuda.is_available() and self.config.get('training_config', {}).get('use_amp', True)
                with autocast('cuda', enabled=use_amp_runtime):
                    outputs = self.model(features, asset_ids)
                    if outputs.ndim > 1 and outputs.shape[-1] == 1: # Ensure it's (batch, 1) before squeeze
                        outputs = outputs.squeeze(-1)
                    loss = self.criterion(outputs, labels.float())
                total_loss += loss.item()
                current_loss = loss.item()
                total_loss += loss.item()
                current_loss = loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # Convert outputs (logits) to probabilities for metrics
                # This is always needed now since models output logits
                probs = torch.sigmoid(outputs)
                
                all_predictions.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Calculate real-time metrics for display
                batch_time = time.time() - batch_start_time
                
                # Calculate accuracy for current batch
                batch_probs = probs.cpu().numpy()
                batch_labels = labels.cpu().numpy()
                batch_preds = (batch_probs >= 0.5).astype(int)
                batch_accuracy = accuracy_score(batch_labels, batch_preds) * 100
                
                # Calculate processing speed
                samples_per_sec = len(features) / batch_time if batch_time > 0 else 0
                
                # Enhanced progress bar updates
                if is_hpo_mode:
                    # Update every 25 batches in HPO mode
                    if batch_idx % 25 == 0:
                        pbar.set_postfix({
                            'Loss': f'{current_loss:.4f}',
                            'Acc': f'{batch_accuracy:.1f}%'
                        })
                else:
                    # Comprehensive real-time updates for normal mode
                    elapsed_time = time.time() - val_start_time
                    batches_remaining = len(self.val_loader) - (batch_idx + 1)
                    avg_batch_time = elapsed_time / (batch_idx + 1) if batch_idx > 0 else batch_time
                    eta_seconds = batches_remaining * avg_batch_time
                    
                    # Format ETA
                    if eta_seconds > 60:
                        eta_str = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s"
                    else:
                        eta_str = f"{int(eta_seconds)}s"
                    
                    pbar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Avg': f'{avg_loss:.4f}',
                        'Acc': f'{batch_accuracy:.1f}%',
                        'Speed': f'{samples_per_sec:.0f}s/s',
                        'ETA': eta_str
                    })
        
        # Ensure progress bar is properly closed
        pbar.close()
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = self._compute_metrics(all_predictions, all_labels, avg_loss)

        # Log evaluation plots (e.g., confusion matrix)
        log_freq = self.config['training_config'].get('log_confusion_matrix_freq', 0)
        # Ensure epoch is an integer for modulo operation
        current_epoch_int = int(epoch) # epoch is now a direct parameter

        if self.enhanced_logger and log_freq > 0 and current_epoch_int % log_freq == 0:
            np_all_labels = np.array(all_labels)
            np_all_predictions = np.array(all_predictions) # These are probabilities
            binary_preds = (np_all_predictions >= 0.5).astype(int)
            
            self.enhanced_logger.log_evaluation_plots(
                y_true=np_all_labels,
                y_pred=binary_preds,
                y_prob=np_all_predictions,
                current_epoch=current_epoch_int,
                stage="validation"
            )
        
        return metrics
    
    def _compute_metrics(self, predictions: List[float], labels: List[int], loss: float) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics for binary classification.
        
        Enhanced to include:
        - Standard metrics: Accuracy, Precision, Recall, F1
        - Positive class (class 1) specific metrics: Precision_pos, Recall_pos, F1_pos
        - AUC metrics: ROC-AUC, PR-AUC
        - Confusion Matrix: TN, FP, FN, TP
        - Log-Loss (Binary Cross-Entropy)
        
        Args:
            predictions: List of predicted probabilities
            labels: List of true binary labels
            loss: Pre-computed loss value
            
        Returns:
            Dict containing all computed metrics
        """
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Convert probabilities to binary predictions
        binary_preds = (predictions >= 0.5).astype(int)
        
        # Compute standard metrics with edge case handling
        accuracy = accuracy_score(labels, binary_preds)
        precision = precision_score(labels, binary_preds, zero_division=0)
        recall = recall_score(labels, binary_preds, zero_division=0)
        f1 = f1_score(labels, binary_preds, zero_division=0)
        
        # Compute positive class (class 1) specific metrics
        # These are critical for imbalanced classification
        precision_pos = 0.0
        recall_pos = 0.0
        f1_pos = 0.0
        
        if len(np.unique(labels)) > 1:
            try:
                from sklearn.metrics import precision_recall_fscore_support
                precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                    labels, binary_preds, average=None, zero_division=0
                )
                
                # Extract positive class (class 1) metrics if available
                if len(precision_per_class) > 1:
                    precision_pos = float(precision_per_class[1])
                    recall_pos = float(recall_per_class[1])
                    f1_pos = float(f1_per_class[1])
            except (ValueError, IndexError) as e:
                logger.debug(f"Could not compute positive class metrics: {e}")
        
        # AUC metrics (handle edge cases)
        try:
            roc_auc = roc_auc_score(labels, predictions)
        except (ValueError, IndexError):
            roc_auc = 0.0
        
        try:
            pr_auc = average_precision_score(labels, predictions)
        except (ValueError, IndexError):
            pr_auc = 0.0
        
        # Compute Log-Loss with edge case handling
        try:
            # Clip predictions to avoid log(0) errors
            predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
            logloss = log_loss(labels, predictions_clipped)
        except (ValueError, IndexError):
            logloss = 0.0
        
        # Compute Confusion Matrix
        try:
            cm = confusion_matrix(labels, binary_preds)
            # Handle different confusion matrix shapes
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                # All predictions and labels are the same class
                if labels[0] == 0:
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
            else:
                # Unexpected shape, set to zeros
                tn, fp, fn, tp = 0, 0, 0, 0
        except (ValueError, IndexError):
            tn, fp, fn, tp = 0, 0, 0, 0
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_pos': precision_pos,
            'recall_pos': recall_pos,
            'f1_pos': f1_pos,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'log_loss': logloss,
            'confusion_matrix_tn': int(tn),
            'confusion_matrix_fp': int(fp),
            'confusion_matrix_fn': int(fn),
            'confusion_matrix_tp': int(tp)
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint with enhanced naming convention."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'scalers': self.scalers,
            'asset_id_map': self.asset_id_map
        }
        
        # Create enhanced filename with comprehensive information
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.config.get('model_type', 'unknown')
        f1_score_val = metrics.get('f1', 0.0) # Renamed to avoid conflict
        epoch_val = self.current_epoch # Renamed to avoid conflict
        
        # Enhanced filename format: epoch_modeltype_f1score_timestamp.pt
        enhanced_filename = f"epoch{epoch_val:03d}_{model_type}_f1{f1_score_val:.4f}_{timestamp}.pt"
        
        # Save regular checkpoint with enhanced naming
        checkpoint_path = output_dir / enhanced_filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {enhanced_filename}")
        
        # Save best model with both enhanced and legacy naming
        if is_best:
            # Enhanced best model filename
            best_enhanced_filename = f"BEST_epoch{epoch_val:03d}_{model_type}_f1{f1_score_val:.4f}_{timestamp}.pt"
            best_enhanced_path = output_dir / best_enhanced_filename
            torch.save(checkpoint, best_enhanced_path)
            
            # Legacy best model filename for compatibility
            best_legacy_path = output_dir / 'best_model.pt'
            torch.save(checkpoint, best_legacy_path)
            
            logger.info(f"New best model saved with F1: {metrics['f1']:.4f}")
            logger.info(f"Best model files: {best_enhanced_filename} & best_model.pt")
            
            # Log model artifact to MLflow with signature
            if MLFLOW_AVAILABLE:
                try:
                    import mlflow.models as mlflow_models
                    import numpy as np
                    
                    # Create sample inputs (batch_size=1, lookback=24, features=n_features)
                    lookback = self.config['model_config'].get('lookback_window', 24)
                    n_features = self.config['model_config'].get('n_features', 23)
                    
                    # Create dictionary input example with both required inputs
                    sample_features = torch.randn(1, lookback, n_features).to(self.device)
                    sample_asset_ids = torch.zeros(1, dtype=torch.long).to(self.device)
                    
                    # Get model output for signature
                    with torch.no_grad():
                        output_example = self.model(sample_features, sample_asset_ids)
                    
                    # Create signature with named inputs for clarity
                    # Use a simplified signature that describes the main input
                    signature = mlflow_models.infer_signature(
                        sample_features.cpu().numpy(),
                        output_example.cpu().numpy()
                    )
                    
                    # Log model with signature only (skip input_example to avoid validation issues)
                    mlflow.pytorch.log_model(
                        pytorch_model=self.model,
                        artifact_path="best_model",
                        signature=signature
                    )
                    logger.info("Model logged to MLflow with signature")
                except Exception as e:
                    logger.warning(f"Failed to log model with signature: {e}, logging without signature")
                    mlflow.pytorch.log_model(self.model, "best_model")

    def save_final_model(self) -> Path:
        """
        Save the final, trained model and associated artifacts.
        This method is intended to be called after a definitive training run.
        
        Returns:
            Path to the saved final model artifact.
        """
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Consolidate all necessary components for deployment/inference
        final_artifact = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scalers': self.scalers,
            'asset_id_map': self.asset_id_map,
            'training_metadata': {
                'best_epoch': self.best_epoch,
                'best_metric_value': self.best_metric,
                'training_timestamp': datetime.now().isoformat()
            }
        }
        
        # Define a clear, final model filename
        model_type = self.config.get('model_type', 'unknown')
        final_model_filename = f"final_model_{model_type}.pth"
        final_model_path = output_dir / final_model_filename
        
        # Save the final artifact
        torch.save(final_artifact, final_model_path)
        logger.info(f"Final model artifact saved to: {final_model_path}")

        # Log the final model to MLflow if active
        if self.enhanced_logger and mlflow.active_run():
            try:
                self.enhanced_logger.log_final_model(
                    model=self.model,
                    artifact_path="final_production_model",
                    full_artifact_path=str(final_model_path)
                )
                logger.info("Final model logged to MLflow artifacts.")
            except Exception as e:
                logger.warning(f"Failed to log final model to MLflow: {e}")
    
    def evaluate_model_on_test_set(self, model_path: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate the trained model on the hold-out test set.
        
        This function implements Activity 7.2: Test Set Evaluation.
        It loads the best model checkpoint, performs inference on the test set,
        calculates comprehensive metrics, and logs results to MLflow.
        
        Args:
            model_path: Path to model checkpoint. If None, uses the best_model.pt from output_dir
            
        Returns:
            Dict containing all test set evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("EVALUATING MODEL ON TEST SET")
        logger.info("=" * 80)
        
        # Determine model path
        if model_path is None:
            model_path = Path(self.config['output_dir']) / 'best_model.pt'
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
        logger.info(f"Best validation metrics from training:")
        for metric, value in checkpoint['metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Perform inference on test set
        logger.info("\nPerforming inference on test set...")
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for features, labels, asset_ids in tqdm(self.test_loader, desc="Test Set Inference"):
                # Move to device
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                asset_ids = asset_ids.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(features, asset_ids)
                if outputs.ndim > 1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)
                
                # Calculate loss
                loss = self.criterion(outputs, labels.float())
                total_loss += loss.item()
                
                # Convert logits to probabilities
                probs = torch.sigmoid(outputs)
                
                all_predictions.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute comprehensive metrics
        avg_loss = total_loss / len(self.test_loader)
        test_metrics = self._compute_metrics(all_predictions, all_labels, avg_loss)
        
        # Log test metrics
        logger.info("\n" + "=" * 80)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Test Samples: {len(all_labels)}")
        logger.info(f"\nClassification Metrics:")
        logger.info(f"  Loss:      {test_metrics['loss']:.4f}")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {test_metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC:    {test_metrics['pr_auc']:.4f}")
        logger.info(f"  Log-Loss:  {test_metrics['log_loss']:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Negatives:  {test_metrics['confusion_matrix_tn']}")
        logger.info(f"  False Positives: {test_metrics['confusion_matrix_fp']}")
        logger.info(f"  False Negatives: {test_metrics['confusion_matrix_fn']}")
        logger.info(f"  True Positives:  {test_metrics['confusion_matrix_tp']}")
        logger.info("=" * 80)
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE and mlflow.active_run():
            logger.info("\nLogging test metrics to MLflow...")
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            # Log confusion matrix plot if enhanced logger available
            if self.enhanced_logger:
                np_all_labels = np.array(all_labels)
                np_all_predictions = np.array(all_predictions)
                binary_preds = (np_all_predictions >= 0.5).astype(int)
                
                self.enhanced_logger.log_evaluation_plots(
                    y_true=np_all_labels,
                    y_pred=binary_preds,
                    y_prob=np_all_predictions,
                    current_epoch=checkpoint['epoch'],
                    stage="test"
                )
            
            logger.info("Test metrics logged to MLflow successfully")
        
        return test_metrics

        return final_model_path
    
    def train(self) -> None:
        """Main training loop."""
        # Training configuration
        epochs = self.config['training_config']['epochs']
        patience = self.config['training_config'].get('early_stopping_patience', 15)
        min_epochs = self.config['training_config'].get('min_epochs', 0)
        if min_epochs > epochs:
            logger.warning("min_epochs (%d) exceeds total epochs (%d); clamping to epochs", min_epochs, epochs)
            min_epochs = epochs
        min_epochs = max(0, min_epochs)
        monitor_metric = self.config['training_config'].get('monitor_metric', 'f1')
          # Initialize colorama for cross-platform colored output
        colorama.init(autoreset=True)
        
        # Determine if we're in HPO mode to simplify output
        is_hpo_mode = hasattr(self, '_in_hpo_mode') and self._in_hpo_mode
        
        # Ensure proper terminal handling for progress bars
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            # Terminal supports ANSI escape sequences
            os.environ['TERM'] = 'xterm-256color'
        
        # Print training header (simplified for HPO mode)
        if not is_hpo_mode:
            print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{Style.BRIGHT}*** STARTING NEURAL NETWORK TRAINING ***{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
            model_type = self.config.get('model_type', 'UNKNOWN')
            print(f"{Fore.CYAN}Model: {Style.BRIGHT}{model_type.upper()}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Epochs: {Style.BRIGHT}{epochs}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Batch Size: {Style.BRIGHT}{self.config['training_config']['batch_size']}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Learning Rate: {Style.BRIGHT}{self.config['training_config']['learning_rate']}{Style.RESET_ALL}")
            if min_epochs > 0:
                print(f"{Fore.CYAN}Minimum Epochs Before Early Stop: {Style.BRIGHT}{min_epochs}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Device: {Style.BRIGHT}{self.device}{Style.RESET_ALL}")
            # Main training loop with overall progress (simplified for HPO mode)
        training_start_time = time.time()
        
        if is_hpo_mode:
            # For HPO mode, disable the visual progress bar but keep the iterator
            epoch_pbar = tqdm(range(epochs), desc="🤖 HPO Training",
                              disable=True, leave=False)
        else:
            # Enhanced comprehensive progress bar for normal mode
            epoch_pbar = tqdm(range(epochs), desc="[Training] Neural Network",
                              ncols=130, file=sys.stdout, ascii=True, unit='epoch',
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        for epoch_idx in epoch_pbar: # Renamed epoch to epoch_idx to avoid conflict with parameter
            epoch_start_time = time.time()
            self.current_epoch = epoch_idx
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_metrics.append(train_metrics)
            
            # Validate epoch
            val_metrics = self.validate_epoch(epoch=self.current_epoch) # Pass current_epoch
            self.val_metrics.append(val_metrics)
            
            # Update scheduler (except OneCycleLR which updates per batch)
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[monitor_metric])
                else:
                    self.scheduler.step()
            
            # Calculate epoch time and total training time
            epoch_time = time.time() - epoch_start_time
            total_training_time = time.time() - training_start_time
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check for improvement
            current_metric = val_metrics[monitor_metric]
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch_idx
                self.patience_counter = 0
                if not is_hpo_mode:
                    epoch_pbar.write(f"{Fore.GREEN}{Style.BRIGHT}*** NEW BEST MODEL! F1: {self.best_metric:.4f}{Style.RESET_ALL}")
            else:
                self.patience_counter += 1
            
            # Calculate estimated time remaining
            avg_epoch_time = total_training_time / (epoch_idx + 1)
            epochs_remaining = epochs - (epoch_idx + 1)
            eta_seconds = epochs_remaining * avg_epoch_time
            
            # Format ETA
            if eta_seconds > 3600:
                eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
            elif eta_seconds > 60:
                eta_str = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s"
            else:
                eta_str = f"{int(eta_seconds)}s"
            
            # Enhanced epoch progress bar updates
            if is_hpo_mode:
                # Simplified but informative updates for HPO mode
                epoch_pbar.set_postfix({
                    'Val F1': f'{val_metrics["f1"]:.4f}',
                    'Best': f'{self.best_metric:.4f}',
                    'Loss': f'{val_metrics["loss"]:.4f}',
                    'ETA': eta_str
                })
                
                # Minimal logging for HPO mode - only when there's improvement or every 5 epochs
                if is_best or epoch_idx % 5 == 0:
                    print(f"Epoch {epoch_idx+1}/{epochs}: Val F1={val_metrics['f1']:.4f}, Loss={val_metrics['loss']:.4f}", file=sys.stderr)
            else:
                # Comprehensive updates for normal mode
                epoch_pbar.set_postfix({
                    'Val F1': f'{val_metrics["f1"]:.4f}',
                    'Train F1': f'{train_metrics["f1"]:.4f}',
                    'Best F1': f'{self.best_metric:.4f}',
                    'Val Loss': f'{val_metrics["loss"]:.4f}',
                    'LR': f'{current_lr:.2e}',
                    'Time': f'{epoch_time:.1f}s',
                    'ETA': eta_str,
                    'Patience': f'{self.patience_counter}/{patience}'
                })
                
                # Detailed epoch results output
                epoch_pbar.write(f"\n{Fore.BLUE}{'-'*80}{Style.RESET_ALL}")
                epoch_pbar.write(f"{Fore.BLUE}{Style.BRIGHT}[EPOCH] {epoch_idx+1}/{epochs} RESULTS{Style.RESET_ALL}")
                epoch_pbar.write(f"{Fore.BLUE}{'-'*80}{Style.RESET_ALL}")
                epoch_pbar.write(f"{Fore.GREEN}[Train]   - Loss: {Style.BRIGHT}{train_metrics['loss']:.4f}{Style.RESET_ALL}, "
                               f"F1: {Style.BRIGHT}{train_metrics['f1']:.4f}{Style.RESET_ALL}, "
                               f"Acc: {Style.BRIGHT}{train_metrics['accuracy']:.4f}{Style.RESET_ALL}")
                epoch_pbar.write(f"{Fore.YELLOW}[Validation] - Loss: {Style.BRIGHT}{val_metrics['loss']:.4f}{Style.RESET_ALL}, "
                               f"F1: {Style.BRIGHT}{val_metrics['f1']:.4f}{Style.RESET_ALL}, "
                               f"Acc: {Style.BRIGHT}{val_metrics['accuracy']:.4f}{Style.RESET_ALL}")
                epoch_pbar.write(f"{Fore.CYAN}[LR]: {Style.BRIGHT}{current_lr:.6f}{Style.RESET_ALL}")
                epoch_pbar.write(f"{Fore.MAGENTA}[Time]: {Style.BRIGHT}{epoch_time:.1f}s{Style.RESET_ALL}, "
                               f"Remaining: {Style.BRIGHT}{eta_str}{Style.RESET_ALL}")
                if is_best:
                    epoch_pbar.write(f"{Fore.GREEN}{Style.BRIGHT}*** Best performance so far!{Style.RESET_ALL}")
                else:
                    epoch_pbar.write(f"{Fore.BLUE}[Patience]: {Style.BRIGHT}{self.patience_counter}/{patience}{Style.RESET_ALL}")
            
            # Save checkpoint
            # MLflow per-epoch metric logging with step parameter
            if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
                try:
                    # Log all training metrics with step
                    for metric_name, metric_value in train_metrics.items():
                        mlflow.log_metric(f"train_{metric_name}", metric_value, step=epoch_idx)
                    
                    # Log all validation metrics with step
                    for metric_name, metric_value in val_metrics.items():
                        mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch_idx)
                    
                    # Log learning rate with step
                    mlflow.log_metric("learning_rate", current_lr, step=epoch_idx)
                    
                except Exception as e:
                    logger.warning(f"MLflow per-epoch logging failed at epoch {epoch_idx}: {e}")
            
            self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                if (epoch_idx + 1) < min_epochs:
                    if not is_hpo_mode:
                        epoch_pbar.write(
                            f"{Fore.YELLOW}{Style.BRIGHT}[WAIT] Early stopping patience reached but minimum epochs ({min_epochs}) not met yet{Style.RESET_ALL}"
                        )
                else:
                    if not is_hpo_mode:
                        epoch_pbar.write(
                            f"{Fore.RED}{Style.BRIGHT}[STOP] Early stopping triggered after {patience} epochs without improvement{Style.RESET_ALL}"
                        )
                    break
          # Ensure epoch progress bar is properly closed
        epoch_pbar.close()
        
        # Training completion summary (simplified for HPO mode)
        if not is_hpo_mode:
            print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{Style.BRIGHT}*** TRAINING COMPLETED! ***{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[Best] F1 Score: {Style.BRIGHT}{self.best_metric:.4f}{Style.RESET_ALL} (Epoch {self.best_epoch+1})")
            print(f"{Fore.CYAN}[Total] Epochs: {Style.BRIGHT}{self.current_epoch + 1}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}\n")
        
        logger.info(f"Training completed. Best F1: {self.best_metric:.4f} at epoch {self.best_epoch}")
        
        # Enhanced logging and reporting if available
        if self.enhanced_logger:
            # Log final training summary with final metrics
            training_summary = {
                'best_val_f1': self.best_metric,
                'best_epoch': self.best_epoch,
                'total_epochs': self.current_epoch + 1,
                'training_time_seconds': sum(epoch_time for epoch_time in [1.0] * (self.current_epoch + 1)),  # Approximate
                'final_lr': self.optimizer.param_groups[0]['lr']
            }
            
            # Add final train and val losses if available
            if self.train_metrics:
                training_summary['final_train_loss'] = self.train_metrics[-1]['loss']
                training_summary['final_train_f1'] = self.train_metrics[-1]['f1']
            if self.val_metrics:
                training_summary['final_val_loss'] = self.val_metrics[-1]['loss']
                training_summary['final_val_f1'] = self.val_metrics[-1]['f1']
            
            for key, value in training_summary.items():
                mlflow.log_metric(key, value)
            
            # Generate learning curves
            try:
                self.enhanced_logger.log_training_plots(
                    train_metrics=self.train_metrics,
                    val_metrics=self.val_metrics
                )
            except Exception as e:
                logger.warning(f"Could not log training plots: {e}")
            
            # Save scalers with enhanced logging
            scalers_path = Path(self.config['output_dir']) / 'scalers.joblib'
            joblib.dump(self.scalers, scalers_path)
            try:
                self.enhanced_logger.log_scalers(self.scalers)
            except Exception as e:
                logger.warning(f"Could not log scalers: {e}")
            
            # Generate training report if reporter is available
            if self.reporter:
                try:
                    report_path = Path(self.config['output_dir']) / 'training_report.html'
                    self.reporter.generate_training_report(
                        run_id=mlflow.active_run().info.run_id,
                        output_path=str(report_path)
                    )
                    mlflow.log_artifact(str(report_path))
                    logger.info(f"Training report generated: {report_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate training report: {e}")
            
            # End enhanced logging session
            self.enhanced_logger.end_run()
            
        elif MLFLOW_AVAILABLE and self.config.get('use_mlflow', True):
            # Fallback to basic MLflow logging
            try:
                # Check if there's an active run, if not start one
                if mlflow.active_run() is None:
                    self._setup_mlflow()
                
                # Log summary metrics including final train/val losses
                summary_metrics = {
                    'best_val_f1': self.best_metric,
                    'best_epoch': self.best_epoch,
                    'total_epochs': self.current_epoch + 1
                }
                
                # Add final train and val losses if available
                if self.train_metrics:
                    summary_metrics['final_train_loss'] = self.train_metrics[-1]['loss']
                    summary_metrics['final_train_f1'] = self.train_metrics[-1]['f1']
                if self.val_metrics:
                    summary_metrics['final_val_loss'] = self.val_metrics[-1]['loss']
                    summary_metrics['final_val_f1'] = self.val_metrics[-1]['f1']
                
                mlflow.log_metrics(summary_metrics)
                
                # Save scalers as artifacts
                scalers_path = Path(self.config['output_dir']) / 'scalers.joblib'
                joblib.dump(self.scalers, scalers_path)
                mlflow.log_artifact(str(scalers_path))
                
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        # Final success message
        print(f"{Fore.GREEN}{Style.BRIGHT}[SUCCESS] Training pipeline executed successfully!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[MLflow] Model saved and logged to MLflow{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[MLflow] Check MLflow UI for detailed metrics and model artifacts{Style.RESET_ALL}\n")

    def train_with_pruning(self, trial: Optional['optuna.Trial'] = None) -> Dict[str, float]:
        """
        Training loop with Optuna pruning support for HPO.
        
        Args:
            trial: Optuna trial object for pruning
            
        Returns:
            Best validation metrics achieved during training
        """
        if not OPTUNA_AVAILABLE and trial is not None:
            logger.warning("Optuna not available, falling back to regular training")
            self.train()
            return {}
        
        # Training configuration
        epochs = self.config['training_config']['epochs']
        patience = self.config['training_config'].get('early_stopping_patience', 15)
        min_epochs = self.config['training_config'].get('min_epochs', 0)
        if min_epochs > epochs:
            logger.warning("min_epochs (%d) exceeds total epochs (%d); clamping to epochs", min_epochs, epochs)
            min_epochs = epochs
        min_epochs = max(0, min_epochs)
        monitor_metric = self.config['training_config'].get('monitor_metric', 'f1')
        
        # Track best metrics for return
        best_metrics = {}
        
        # Check if we're in HPO mode (MLflow managed externally)
        in_hpo_mode = hasattr(self, '_in_hpo_mode') and self._in_hpo_mode
        
        # Main training loop
        for epoch_idx in range(epochs): # Renamed epoch to epoch_idx
            self.current_epoch = epoch_idx
            
            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch(epoch=self.current_epoch) # Pass current_epoch
            
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Update scheduler
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[monitor_metric])
                else:
                    self.scheduler.step()
            
            # Check for improvement
            current_metric = val_metrics[monitor_metric]
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch_idx
                self.patience_counter = 0
                best_metrics = val_metrics.copy()
                
                # Save best model (skip file saving in HPO mode for performance)
                if not in_hpo_mode:
                    self.save_checkpoint(val_metrics, is_best=True)
            else:
                self.patience_counter += 1
            
            # MLflow logging (only if MLflow run is active)
            if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
                try:
                    # Log all train and validation metrics
                    train_log_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
                    val_log_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
                    
                    mlflow.log_metrics(train_log_metrics, step=epoch_idx)
                    mlflow.log_metrics(val_log_metrics, step=epoch_idx)
                    mlflow.log_metric('learning_rate', self.optimizer.param_groups[0]['lr'], step=epoch_idx)
                except Exception as e:
                    logger.warning(f"MLflow logging failed at epoch {epoch_idx}: {e}")
            
            # Optuna pruning
            if trial is not None:
                trial.report(current_metric, epoch_idx)
                if trial.should_prune():
                    logger.info(f"Trial pruned at epoch {epoch_idx}")
                    raise optuna.TrialPruned()
            
            # Early stopping
            if self.patience_counter >= patience:
                if (epoch_idx + 1) < min_epochs:
                    logger.info(
                        "Early stopping patience reached at epoch %d but min_epochs=%d not satisfied; continuing training.",
                        epoch_idx,
                        min_epochs
                    )
                else:
                    logger.info(f"Early stopping at epoch {epoch_idx}. Best {monitor_metric}: {self.best_metric:.4f}")
                    break
        
        # Ensure we have best metrics to return
        if not best_metrics and self.val_metrics:
            best_metrics = self.val_metrics[-1].copy()
        
        # Add prefixed metrics for HPO
        prefixed_metrics = {}
        for key, value in best_metrics.items():
            prefixed_metrics[f'validation_{key}'] = value
            if key == 'f1':
                prefixed_metrics['validation_f1_score_positive_class'] = value
            elif key == 'precision':
                prefixed_metrics['validation_precision_positive_class'] = value
            elif key == 'recall':
                prefixed_metrics['validation_recall_positive_class'] = value
        logger.info(f"Training completed. Best {monitor_metric}: {self.best_metric:.4f} at epoch {self.best_epoch}")
        
        # Log a final summary of the best metrics for this trial
        if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
            final_summary_metrics = {f"final_{k}": v for k, v in prefixed_metrics.items()}
            final_summary_metrics['final_best_epoch'] = self.best_epoch
            mlflow.log_metrics(final_summary_metrics)
            
        return prefixed_metrics

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'model_type': 'lstm',
        'experiment_name': 'nn_training_default',
        'output_dir': 'training/runs/default',
        'use_mlflow': True,
        'num_workers': 4,
        
        'data_config': {
            'symbols_config_path': '../config/symbols.json',
            'feature_list': [
                'open', 'high', 'low', 'close', 'volume', 'vwap',
                'SMA_20', 'EMA_12', 'RSI_14', 'MACD_line', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_middle', 'BB_lower', 'BB_bandwidth',
                'DayOfWeek_sin', 'DayOfWeek_cos', 'sentiment_score_hourly_ffill'
            ],
            'lookback_window': 24,
            'profit_target': 0.05,
            'stop_loss_target': 0.02,
            'prediction_horizon_hours': 8,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'scaling_method': 'standard',
            'nan_handling_features': 'ffill',
            'sample_weights_strategy': 'balanced'
        },
        
        'model_config': {
            'n_features': 17,
            'num_assets': 154,
            'asset_embedding_dim': 8,
            'lstm_hidden_dim': 64,
            'lstm_num_layers': 2,
            'attention_dim极': 64,
            'dropout_rate': 0.3,
            'use_layer_norm': True
        },
        
        'training_config': {
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'optimizer': 'adamw',
            'weight_decay': 1e-5,
            'scheduler': 'reduce_on_plateau',
            'scheduler_factor': 0.5,
            'scheduler_patience': 5,
            'gradient_clip_norm': 1.0,
            'early_stopping_patience': 15,
            'monitor_metric': 'f1',
            'use_weighted_sampling': False,
            'use_amp': True, # Added use_amp flag
            'log_confusion_matrix_freq': 5, # Log confusion matrix every N epochs, 0 to disable
            'loss_function': {
                'type': 'focal',
                'alpha': 0.25,
                'gamma': 2.0
            }
        }
    }

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Neural Network for Trading Signal Prediction')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--model-type', type=str, choices=['mlp', 'lstm', 'gru', 'cnn_lstm'],
                       help='Model architecture type')
    parser.add_argument('--output-dir', type=str, help='Output directory for models and logs')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--use-pregenerated', action='store_true',
                       help='Use pre-generated training data from data/training_data')
    parser.add_argument('--pregenerated-path', type=str, default='data/training_data',
                       help='Path to pre-generated training data directory')
    
    args = parser.parse_args()
    
    # Enhanced configuration management
    try:
        config_manager = ConfigurationManager()
        
        if args.config:
            # Load from file and apply CLI overrides
            config = config_manager.load_base_config(args.config)
            cli_overrides = {}
            
            if args.model_type:
                cli_overrides['model_type'] = args.model_type
            if args.output_dir:
                cli_overrides['output_dir'] = args.output_dir
            if args.epochs:
                cli_overrides['training_config.epochs'] = args.epochs
            if args.batch_size:
                cli_overrides['training_config.batch_size'] = args.batch_size
            if args.learning_rate:
                cli_overrides['training_config.learning_rate'] = args.learning_rate
            if args.use_pregenerated:
                cli_overrides['use_pregenerated_data'] = True
                cli_overrides['pregenerated_data_path'] = args.pregenerated_path
                
            config = config_manager.override_config(config, cli_overrides)
        else:
            # Create default config and apply CLI overrides
            config = config_manager.create_default_config('neural_network')
            
            if args.model_type:
                config['model_type'] = args.model_type
            if args.output_dir:
                config['output_dir'] = args.output_dir
            if args.epochs:
                config['training_config']['epochs'] = args.epochs
            if args.batch_size:
                config['training_config']['batch_size'] = args.batch_size
            if args.learning_rate:
                config['training_config']['learning_rate'] = args.learning_rate
            if args.use_pregenerated:
                config['use_pregenerated_data'] = True
                config['pregenerated_data_path'] = args.pregenerated_path
        
        # Validate configuration
        config = config_manager.validate_config(config)
        
    except Exception as e:
        logger.warning(f"Enhanced config management failed: {e}")
        # Fallback to original configuration loading
        if args.config:
            config = load_config(args.config)
        else:
            config = create_default_config()
        
        # Override config with command line arguments (fallback method)
        if args.model_type:
            config['model_type'] = args.model_type
        if args.output_dir:
            config['output_dir'] = args.output_dir
        if args.epochs:
            config['training_config']['epochs'] = args.epochs
        if args.batch_size:
            config['training_config']['batch_size'] = args.batch_size
        if args.learning_rate:
            config['training_config']['learning_rate'] = args.learning_rate
        if args.use_pregenerated:
            config['use_pregenerated_data'] = True
            config['pregenerated_data_path'] = args.pregenerated_path
    
    # Check GPU availability
    if args.gpu and not torch.cuda.is_available():
        logger.error("GPU requested but CUDA not available")
        sys.exit(1)
    
    # Create trainer and run training
    try:
        trainer = ModelTrainer(config)
        trainer.load_data()
        trainer.create_model()
        trainer.create_optimizer_and_scheduler()
        trainer.create_loss_function()
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if MLFLOW_AVAILABLE:
            mlflow.end_run(status='FAILED')
        raise

if __name__ == '__main__':
    main()