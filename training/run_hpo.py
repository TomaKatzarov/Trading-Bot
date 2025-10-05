#!/usr/bin/env python3
"""
Hyperparameter Optimization (HPO) Script for Neural Network Models

This script implements the HPO strategy as outlined in Section 5 (Activities 5.1-5.6)
of the operational plan. It uses Optuna for Bayesian optimization and integrates with
the existing training infrastructure.

Key Features:
- Optuna-based hyperparameter optimization with TPE sampler
- Configurable search spaces for all model architectures
- Integration with existing train_nn_model.py infrastructure
- MLflow logging for all trials and results
- Pruning support for early stopping of unpromising trials
- Comprehensive results analysis and best trial identification

Author: Flow-Code
Date: 2025-05-28
Version: 1.0
"""

import argparse
import logging
import os
import sys
import time
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*experimental.*")
warnings.filterwarnings("ignore", message=".*MLflowCallback.*")
warnings.filterwarnings("ignore", message=".*torch version.*")
warnings.filterwarnings("ignore", message=".*packages were not found.*")

import optuna
from optuna.integration import MLflowCallback
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, RandomSampler
import torch
import numpy as np
import mlflow
import mlflow.pytorch

# MLflow availability flag
MLFLOW_AVAILABLE = True

# Progress bar imports
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    try:
        from tqdm import tqdm
        RICH_AVAILABLE = False
        TQDM_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False
        TQDM_AVAILABLE = False

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train_nn_model import ModelTrainer, create_default_config, load_config
from utils.gpu_utils import setup_gpu
# Import new experiment management modules
from core.experiment_management.config_manager import ConfigurationManager
from core.experiment_management.enhanced_logging import EnhancedMLflowLogger
from core.experiment_management.experiment_organizer import ExperimentOrganizer
from core.experiment_management.reporting import ExperimentReporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize console for rich output
if RICH_AVAILABLE:
    console = Console()

def _generate_candidate_data_paths(preferred_path: Optional[str] = None) -> List[Path]:
    """Generate candidate directories for pre-generated data detection."""
    candidates: List[Path] = []
    seen: set[str] = set()

    def add_path(path: Path):
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen:
            candidates.append(resolved)
            seen.add(key)

    search_bases = [Path.cwd(), Path(__file__).parent, project_root]

    if preferred_path:
        preferred = Path(preferred_path)
        if preferred.is_absolute():
            add_path(preferred)
        else:
            for base in search_bases:
                add_path(base / preferred)

    default_paths = [
        Path("../data/training_data"),
        Path("data/training_data"),
        Path("../data/training_data_v2_full"),
        Path("data/training_data_v2_full")
    ]

    for default_path in default_paths:
        if default_path.is_absolute():
            add_path(default_path)
        else:
            for base in search_bases:
                add_path(base / default_path)

    return candidates


def check_pregenerated_data(preferred_path: Optional[str] = None) -> Tuple[bool, Optional[Path]]:
    """Check if pre-generated training data exists and return the resolved path if available."""
    required_files = {
        "train_X.npy", "train_y.npy", "train_asset_ids.npy",
        "val_X.npy", "val_y.npy", "val_asset_ids.npy",
        "test_X.npy", "test_y.npy", "test_asset_ids.npy",
        "scalers.joblib", "metadata.json", "asset_id_mapping.json"
    }

    candidates = _generate_candidate_data_paths(preferred_path)

    for candidate in candidates:
        if not candidate.exists():
            continue

        missing_required = [f for f in required_files if not (candidate / f).exists()]
        if not missing_required:
            return True, candidate

    # No fully qualified dataset found, but return first candidate for reference if it exists
    for candidate in candidates:
        if candidate.exists():
            return False, candidate

    return False, candidates[0] if candidates else None


def display_data_status(preferred_path: Optional[str] = None) -> Tuple[bool, Optional[Path]]:
    """Display training data status with rich formatting."""
    data_exists, resolved_path = check_pregenerated_data(preferred_path)
    location_text = str(resolved_path) if resolved_path else "N/A"

    if RICH_AVAILABLE:
        if data_exists:
            console.print(Panel(
                "[green]✓ Pre-generated training data found![/green]\n"
                f"[cyan]Location:[/cyan] {location_text}\n"
                "[cyan]Status:[/cyan] Will use pre-generated dataset",
                title="[bold green]Training Data Status[/bold green]",
                border_style="green"
            ))
        else:
            console.print(Panel(
                "[yellow]⚠ No pre-generated training data found[/yellow]\n"
                f"[cyan]Last checked location:[/cyan] {location_text}\n"
                "[cyan]Will generate data during training[/cyan]\n"
                "[dim]Consider running data preparation first for faster HPO[/dim]",
                title="[bold yellow]Training Data Status[/bold yellow]",
                border_style="yellow"
            ))
    else:
        if data_exists:
            logger.info(f"✓ Pre-generated training data found at {location_text} - will use existing dataset")
        else:
            logger.info(f"⚠ No pre-generated training data found (checked {location_text}) - will generate data during training")

    return data_exists, resolved_path

# Check MLflow availability
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Experiment tracking will be limited.")


class HPOSearchSpace:
    """Defines search spaces for hyperparameter optimization."""
    
    @staticmethod
    def get_common_search_space() -> Dict[str, Any]:
        """Get common hyperparameters for all model types."""
        return {
            'learning_rate': ('log_uniform', 1e-5, 1e-2),
            'batch_size': ('categorical', [128, 256, 512]),  # Optimized for RTX 5070 Ti
            'weight_decay': ('log_uniform', 1e-6, 1e-3),
            'dropout_rate': ('uniform', 0.1, 0.5),
            'focal_alpha': ('uniform', 0.25, 0.9),
            'focal_gamma': ('uniform', 0.5, 3.0),
            'scheduler_factor': ('uniform', 0.3, 0.8),
            'scheduler_patience': ('int', 2, 5),  # Faster convergence for HPO
            'gradient_clip_norm': ('uniform', 0.5, 5.0),
            'early_stopping_patience': ('int', 3, 8)  # More aggressive stopping for HPO
        }
    
    @staticmethod
    def get_mlp_search_space() -> Dict[str, Any]:
        """Get MLP-specific hyperparameters."""
        common = HPOSearchSpace.get_common_search_space()
        mlp_specific = {
            'hidden_layers': ('int', 2, 4),
            'hidden_dim_1': ('int', 64, 512),
            'hidden_dim_2': ('int', 32, 256),
            'hidden_dim_3': ('int', 16, 128),
            'asset_embedding_dim': ('int', 4, 32)
        }
        return {**common, **mlp_specific}
    
    @staticmethod
    def get_lstm_search_space() -> Dict[str, Any]:
        """Get LSTM-specific hyperparameters."""
        common = HPOSearchSpace.get_common_search_space()
        lstm_specific = {
            'lstm_hidden_dim': ('int', 32, 256),
            'lstm_num_layers': ('int', 1, 3),
            'attention_dim': ('int', 32, 128),
            'asset_embedding_dim': ('int', 4, 32),
            'use_layer_norm': ('categorical', [True, False])
        }
        return {**common, **lstm_specific}
    
    @staticmethod
    def get_gru_search_space() -> Dict[str, Any]:
        """Get GRU-specific hyperparameters."""
        common = HPOSearchSpace.get_common_search_space()
        gru_specific = {
            'gru_hidden_dim': ('int', 32, 256),
            'gru_num_layers': ('int', 1, 3),
            'attention_dim': ('int', 32, 128),
            'asset_embedding_dim': ('int', 4, 32),
            'use_layer_norm': ('categorical', [True, False])
        }
        return {**common, **gru_specific}
    
    @staticmethod
    def get_cnn_lstm_search_space() -> Dict[str, Any]:
        """Get CNN-LSTM-specific hyperparameters."""
        common = HPOSearchSpace.get_common_search_space()
        cnn_lstm_specific = {
            'cnn_filters_1': ('int', 16, 128),
            'cnn_filters_2': ('int', 16, 128),
            'cnn_kernel_size_1': ('int', 3, 7),
            'cnn_kernel_size_2': ('int', 3, 5),
            'cnn_stride': ('int', 1, 2),
            'use_max_pooling': ('categorical', [True, False]),
            'lstm_hidden_dim': ('int', 32, 256),
            'lstm_num_layers': ('int', 1, 2),
            'attention_dim': ('int', 32, 128),
            'asset_embedding_dim': ('int', 4, 32),
            'use_layer_norm': ('categorical', [True, False])
        }
        return {**common, **cnn_lstm_specific}


class HPOObjective:
    """Objective function for Optuna optimization with enhanced experiment management."""
    
    def __init__(self, base_config: Dict[str, Any], model_type: str,
                 target_metric: str = 'validation_f1_score_positive_class', use_pregenerated: bool = None,
                 pregenerated_data_path: Optional[str] = None,
                 experiment_organizer: Optional[ExperimentOrganizer] = None,
                 enhanced_logger: Optional[EnhancedMLflowLogger] = None,
                 config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize HPO objective with enhanced experiment management.
        
        Args:
            base_config: Base configuration for training
            model_type: Type of model ('mlp', 'lstm', 'gru', 'cnn_lstm')
            target_metric: Metric to optimize
            use_pregenerated: Whether to use pre-generated data (auto-detected if None)
            experiment_organizer: Optional experiment organizer for parent/child relationships
            enhanced_logger: Optional enhanced MLflow logger
            config_manager: Optional configuration manager
        """
        self.base_config = base_config.copy()
        self.model_type = model_type
        self.target_metric = target_metric
        self.experiment_organizer = experiment_organizer
        self.enhanced_logger = enhanced_logger
        self.config_manager = config_manager
        self.pregenerated_data_path = pregenerated_data_path

        # Auto-detect pre-generated data if not specified
        data_available, resolved_path = check_pregenerated_data(pregenerated_data_path)
        self.resolved_pregenerated_path = resolved_path

        if use_pregenerated is None:
            self.use_pregenerated = data_available
        else:
            self.use_pregenerated = use_pregenerated and data_available
            if use_pregenerated and not data_available:
                logger.warning("Pre-generated data requested but required dataset was not found. Falling back to data generation for trials.")

        logger.info(f"Initialized HPO objective for {model_type} model")
        logger.info(f"Target metric: {target_metric}")
        logger.info(f"Use pre-generated data: {self.use_pregenerated}")
        logger.info(f"Pre-generated data path: {self.resolved_pregenerated_path if self.resolved_pregenerated_path else 'not available'}")
        logger.info(f"Enhanced experiment management: {self.experiment_organizer is not None}")
        
        # Get search space for model type
        if model_type == 'mlp':
            self.search_space = HPOSearchSpace.get_mlp_search_space()
        elif model_type == 'lstm':
            self.search_space = HPOSearchSpace.get_lstm_search_space()
        elif model_type == 'gru':
            self.search_space = HPOSearchSpace.get_gru_search_space()
        elif model_type == 'cnn_lstm':
            self.search_space = HPOSearchSpace.get_cnn_lstm_search_space()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for a single trial with enhanced experiment management.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Target metric value to optimize
        """
        # Ensure no active MLflow run at start
        try:
            if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
                mlflow.end_run()
                logger.warning(f"Ended active MLflow run before starting trial {trial.number}")
        except Exception as e:
            logger.warning(f"Error checking/ending active MLflow run: {e}")
        
        # Sample hyperparameters
        trial_config = self._sample_hyperparameters(trial)
        
        # Create trial-specific configuration
        config = self._create_trial_config(trial_config, trial.number)
        
        # Disable MLflow auto-setup in trainer for HPO trials
        config['use_mlflow'] = False
        
        trainer = None
        try:
            # Create trainer without MLflow auto-setup
            trainer = ModelTrainer(config)
            
            # Set HPO mode flag to disable nested progress bars
            trainer._in_hpo_mode = True
            
            # Setup MLflow specifically for this trial
            if MLFLOW_AVAILABLE:
                experiment_name = f"hpo_{self.model_type}_trials"
                mlflow.set_experiment(experiment_name)
                
                # Start trial run with unique name
                run_name = f"{self.model_type}_trial_{trial.number}_{datetime.now().strftime('%H%M%S')}"
                mlflow.start_run(run_name=run_name)
                
                # Log trial parameters
                mlflow.log_params(trial_config)
                mlflow.log_param('trial_number', trial.number)
                mlflow.log_param('model_type', self.model_type)
                mlflow.log_param('target_metric', self.target_metric)
                
                # Set tags
                mlflow.set_tags({
                    'trial_number': str(trial.number),
                    'model_type': self.model_type,
                    'hpo_trial': 'true',
                    'study_type': 'optuna'
                })
            
            # Load data and setup model
            trainer.load_data()
            trainer.create_model()
            trainer.create_optimizer_and_scheduler()
            trainer.create_loss_function()
            
            # Train model with pruning callback
            best_metrics = trainer.train_with_pruning(trial)
            
            # Get target metric value
            if isinstance(best_metrics, dict) and self.target_metric in best_metrics:
                target_value = best_metrics[self.target_metric]
            else:
                logger.warning(f"Target metric {self.target_metric} not found in results")
                target_value = 0.0
            
            # Log final trial result
            if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
                mlflow.log_metric('final_target_value', target_value)
                mlflow.set_tag('trial_status', 'completed')
            
            logger.info(f"Trial {trial.number} completed: {self.target_metric} = {target_value:.4f}")
            return float(target_value)
            
        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} pruned")
            if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
                mlflow.set_tag('trial_status', 'pruned')
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
                mlflow.set_tag('trial_status', 'failed')
                mlflow.log_param('error_message', str(e))
            return float('-inf') if self.target_metric.endswith('_loss') else 0.0
        finally:
            # Always ensure MLflow run is ended
            try:
                if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
                    mlflow.end_run()
            except Exception as e:
                logger.warning(f"Error ending MLflow run for trial {trial.number}: {e}")
            
            # Clean up trainer resources
            if trainer is not None:
                try:
                    # Clean up model and tensors
                    if hasattr(trainer, 'model') and trainer.model is not None:
                        del trainer.model
                    if hasattr(trainer, 'train_loader'):
                        del trainer.train_loader
                    if hasattr(trainer, 'val_loader'):
                        del trainer.val_loader
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Error cleaning up trainer for trial {trial.number}: {e}")
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for the trial."""
        sampled = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config[0]
            
            if param_type == 'uniform':
                sampled[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
            elif param_type == 'log_uniform':
                sampled[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
            elif param_type == 'int':
                sampled[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_type == 'categorical':
                sampled[param_name] = trial.suggest_categorical(param_name, param_config[1])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        return sampled
    
    def _create_trial_config(self, trial_params: Dict[str, Any], trial_number: int) -> Dict[str, Any]:
        """Create configuration for the trial."""
        config = self.base_config.copy()
        
        # Update model type
        config['model_type'] = self.model_type
        
        # Update training configuration
        training_config = config['training_config']
        training_config['learning_rate'] = trial_params['learning_rate']
        training_config['batch_size'] = trial_params['batch_size']
        training_config['weight_decay'] = trial_params['weight_decay']
        training_config['scheduler_factor'] = trial_params['scheduler_factor']
        training_config['scheduler_patience'] = trial_params['scheduler_patience']
        training_config['gradient_clip_norm'] = trial_params['gradient_clip_norm']
        training_config['early_stopping_patience'] = trial_params['early_stopping_patience']
        
        # Update loss function parameters
        training_config['loss_function']['alpha'] = trial_params['focal_alpha']
        training_config['loss_function']['gamma'] = trial_params['focal_gamma']
        
        # Update model configuration
        model_config = config['model_config']
        model_config['dropout_rate'] = trial_params['dropout_rate']
        model_config['asset_embedding_dim'] = trial_params['asset_embedding_dim']
        
        # Model-specific parameters
        if self.model_type == 'mlp':
            model_config['hidden_layers'] = trial_params['hidden_layers']
            model_config['hidden_dim_1'] = trial_params['hidden_dim_1']
            model_config['hidden_dim_2'] = trial_params['hidden_dim_2']
            if trial_params['hidden_layers'] >= 3:
                model_config['hidden_dim_3'] = trial_params['hidden_dim_3']
        
        elif self.model_type in ['lstm', 'cnn_lstm']:
            model_config['lstm_hidden_dim'] = trial_params['lstm_hidden_dim']
            model_config['lstm_num_layers'] = trial_params['lstm_num_layers']
            model_config['attention_dim'] = trial_params['attention_dim']
            model_config['use_layer_norm'] = trial_params['use_layer_norm']
            
            if self.model_type == 'cnn_lstm':
                model_config['cnn_filters'] = (trial_params['cnn_filters_1'], trial_params['cnn_filters_2'])
                model_config['cnn_kernel_sizes'] = (trial_params['cnn_kernel_size_1'], trial_params['cnn_kernel_size_2'])
                model_config['cnn_stride'] = trial_params['cnn_stride']
                model_config['use_max_pooling'] = trial_params['use_max_pooling']
        
        elif self.model_type == 'gru':
            model_config['gru_hidden_dim'] = trial_params['gru_hidden_dim']
            model_config['gru_num_layers'] = trial_params['gru_num_layers']
            model_config['attention_dim'] = trial_params['attention_dim']
            model_config['use_layer_norm'] = trial_params['use_layer_norm']
        
        # Update output directory for trial
        config['output_dir'] = f"{config['output_dir']}/trial_{trial_number}"
        
        # Add trial metadata
        config['hpo_trial_number'] = trial_number
        config['hpo_model_type'] = self.model_type
        
        # Add use_pregenerated parameter to top-level config (as expected by ModelTrainer)
        config['use_pregenerated_data'] = self.use_pregenerated
        if self.resolved_pregenerated_path:
            config['pregenerated_data_path'] = str(self.resolved_pregenerated_path)
        elif self.pregenerated_data_path:
            config['pregenerated_data_path'] = self.pregenerated_data_path
        else:
            config['pregenerated_data_path'] = '../data/training_data'
        
        return config


class HPOManager:
    """Manages hyperparameter optimization studies with enhanced experiment management."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HPO manager with enhanced experiment management.
        
        Args:
            config: HPO configuration
        """
        self.config = config
        self.study_name = config.get('study_name', f"hpo_study_{int(time.time())}")
        self.storage_url = config.get('storage_url', f"sqlite:///hpo_studies/{self.study_name}.db")
        
        # Create storage directory
        storage_dir = Path("hpo_studies")
        storage_dir.mkdir(exist_ok=True)
        
        # Initialize enhanced experiment management components
        self.experiment_organizer = None
        self.enhanced_logger = None
        self.config_manager = None
        self.experiment_reporter = None
        
        # Setup enhanced experiment management
        self._setup_enhanced_experiment_management()
        
        # Setup MLflow if available (fallback)
        if MLFLOW_AVAILABLE and not self.enhanced_logger:
            self._setup_mlflow()
    
    def _setup_enhanced_experiment_management(self):
        """Setup enhanced experiment management system."""
        try:
            # Initialize experiment organizer for HPO study
            self.experiment_organizer = ExperimentOrganizer(
                experiment_name=f"HPO_{self.study_name}",
                tags={'experiment_type': 'hpo', 'study_name': self.study_name}
            )
            
            # Initialize enhanced logger
            self.enhanced_logger = EnhancedMLflowLogger()
            
            # Initialize config manager
            self.config_manager = ConfigurationManager()
            
            # Initialize experiment reporter
            self.experiment_reporter = ExperimentReporter()
            
            logger.info("Enhanced experiment management initialized for HPO")
            
        except Exception as e:
            logger.warning(f"Failed to setup enhanced experiment management: {e}")
            logger.warning("Falling back to basic MLflow setup")
    
    def _setup_mlflow(self):
        """Setup MLflow for HPO tracking (fallback)."""
        experiment_name = f"HPO_{self.study_name}"
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set: {experiment_name}")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow experiment: {e}")
    
    def create_study(self, model_type: str, direction: str = 'maximize') -> optuna.Study:
        """
        Create Optuna study.
        
        Args:
            model_type: Type of model for the study
            direction: Optimization direction ('maximize' or 'minimize')
            
        Returns:
            Optuna study object
        """
        study_name_full = f"{self.study_name}_{model_type}"
        
        # Configure sampler
        sampler_type = self.config.get('sampler', 'tpe')
        if sampler_type == 'tpe':
            sampler = TPESampler(seed=42)
        elif sampler_type == 'random':
            sampler = RandomSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        
        # Configure pruner
        pruner_type = self.config.get('pruner', 'median')
        if pruner_type == 'median':
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner_type == 'hyperband':
            pruner = HyperbandPruner(min_resource=10, max_resource=100)
        else:
            pruner = None
        
        # Create study
        study = optuna.create_study(
            study_name=study_name_full,
            storage=self.storage_url,
            load_if_exists=True,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        logger.info(f"Created study: {study_name_full}")
        logger.info(f"Sampler: {sampler_type}, Pruner: {pruner_type}")
        
        return study
    def run_optimization(self, model_type: str, base_config: Dict[str, Any],
                        n_trials: int = 100, target_metric: str = 'validation_f1_score_positive_class',
                        use_pregenerated: bool = None, pregenerated_data_path: Optional[str] = None) -> optuna.Study:
        """
        Run hyperparameter optimization with enhanced experiment management.
        
        Args:
            model_type: Type of model to optimize
            base_config: Base configuration for training
            n_trials: Number of trials to run
            target_metric: Metric to optimize
            use_pregenerated: Whether to use pre-generated data
            pregenerated_data_path: Preferred path to pre-generated data
            
        Returns:
            Completed study object
        """
        logger.info(f"Starting HPO for {model_type} with {n_trials} trials")
        logger.info(f"Target metric: {target_metric}")
        
        # Start parent experiment run for this model's HPO
        parent_run_id = None
        if self.experiment_organizer:
            try:
                parent_run_id = self.experiment_organizer.start_parent_run(
                    run_name=f"HPO_{model_type}",
                    tags={'model_type': model_type, 'hpo_parent': 'true'}
                )
                
                # Log HPO configuration
                if self.enhanced_logger:
                    self.enhanced_logger.log_hyperparameters({
                        'n_trials': n_trials,
                        'target_metric': target_metric,
                        'model_type': model_type,
                        'sampler': self.config.get('sampler', 'tpe'),
                        'pruner': self.config.get('pruner', 'median')
                    })
                    self.enhanced_logger.log_metadata({
                        'experiment_type': 'hpo_study',
                        'study_name': self.study_name,
                        'model_type': model_type
                    })
            except Exception as e:
                logger.warning(f"Failed to start parent experiment run: {e}")
        
        # Create study
        study = self.create_study(model_type)
        
        # Create objective function with enhanced experiment management
        objective = HPOObjective(
            base_config,
            model_type,
            target_metric,
            use_pregenerated,
            pregenerated_data_path=pregenerated_data_path or self.config.get('pregenerated_data_path'),
            experiment_organizer=self.experiment_organizer,
            enhanced_logger=self.enhanced_logger,
            config_manager=self.config_manager
        )
        
        # Setup MLflow callback if available and enhanced logging is not being used
        callbacks = []
        if MLFLOW_AVAILABLE and not self.enhanced_logger:
            try:
                mlflow_callback = MLflowCallback(
                    tracking_uri=mlflow.get_tracking_uri(),
                    metric_name=target_metric
                )
                callbacks.append(mlflow_callback)
            except Exception as e:
                logger.warning(f"Failed to setup MLflow callback: {e}")
        
        # Run optimization with progress tracking
        try:
            if RICH_AVAILABLE:
                # Rich progress bar
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task(f"[cyan]Optimizing {model_type.upper()}", total=n_trials)
                    def progress_callback(study, trial):
                        progress.update(task, advance=1)
                        if trial.state == optuna.trial.TrialState.COMPLETE:
                            progress.console.print(f"[green]✓[/green] Trial {trial.number}: {trial.value:.4f}")
                        elif trial.state == optuna.trial.TrialState.PRUNED:
                            progress.console.print(f"[yellow]⚠[/yellow] Trial {trial.number}: Pruned")
                        elif trial.state == optuna.trial.TrialState.FAIL:
                            progress.console.print(f"[red]✗[/red] Trial {trial.number}: Failed")
                    
                    callbacks.append(progress_callback)
                    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
            
            elif TQDM_AVAILABLE:
                # TQDM progress bar
                with tqdm(total=n_trials, desc=f"HPO {model_type.upper()}", unit="trial") as pbar:
                    def tqdm_callback(study, trial):
                        pbar.update(1)
                        pbar.set_postfix({
                            'best': f"{study.best_value:.4f}" if study.best_value else "N/A",
                            'state': trial.state.name
                        })
                    
                    callbacks.append(tqdm_callback)
                    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
            
            else:
                # Basic progress logging
                def basic_callback(study, trial):
                    if trial.number % 10 == 0:
                        logger.info(f"Trial {trial.number}/{n_trials} completed")
                
                callbacks.append(basic_callback)
                study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
            
            logger.info(f"HPO completed for {model_type}")
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best value: {study.best_value:.4f}")
            logger.info(f"Best params: {study.best_params}")
            
            # Log final HPO results if enhanced logger is available
            if self.enhanced_logger:
                try:
                    final_results = {
                        'best_trial_number': study.best_trial.number,
                        'best_value': study.best_value,
                        'n_trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                        'n_trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                        'n_trials_failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
                    }
                    self.enhanced_logger.log_metrics(final_results)
                    self.enhanced_logger.log_hyperparameters(study.best_params)
                    
                    # Generate and log HPO study visualization if reporter is available
                    if self.experiment_reporter:
                        try:
                            study_report = self.experiment_reporter.generate_hpo_study_report(
                                study, model_type, target_metric
                            )
                            self.enhanced_logger.log_artifact(
                                study_report, 
                                f"hpo_study_report_{model_type}.html"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to generate HPO study report: {e}")
                            
                except Exception as e:
                    logger.warning(f"Failed to log final HPO results: {e}")
            
        except KeyboardInterrupt:
            logger.info("HPO interrupted by user")
            # Log interruption if enhanced logger is available
            if self.enhanced_logger:
                self.enhanced_logger.log_metadata({'hpo_status': 'interrupted'})
        except Exception as e:
            logger.error(f"HPO failed: {e}")
            # Log failure if enhanced logger is available
            if self.enhanced_logger:
                self.enhanced_logger.log_metadata({'hpo_status': 'failed', 'error': str(e)})
            raise
        finally:
            # End parent experiment run if it was started
            if self.experiment_organizer and parent_run_id:
                try:
                    self.experiment_organizer.end_run()
                except Exception as e:
                    logger.warning(f"Failed to end parent experiment run: {e}")
        
        return study
    def analyze_results(self, study: optuna.Study, model_type: str) -> Dict[str, Any]:
        """
        Analyze HPO results with enhanced reporting.
        
        Args:
            study: Completed study object
            model_type: Type of model
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing HPO results for {model_type}")
        
        # Basic statistics
        n_trials = len(study.trials)
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        # Best trial information
        best_trial = study.best_trial
        best_value = study.best_value
        best_params = study.best_params
        
        # Parameter importance (if available)
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except Exception:
            param_importance = {}
        
        # Enhanced analysis with experiment reporter
        convergence_data = None
        visualization_paths = []
        
        if self.experiment_reporter:
            try:
                # Generate convergence analysis
                convergence_data = self.experiment_reporter.analyze_hpo_convergence(study)
                
                # Generate visualizations
                viz_paths = self.experiment_reporter.create_hpo_visualizations(
                    study, model_type, output_dir=f"hpo_results/{model_type}_visualizations"
                )
                visualization_paths.extend(viz_paths)
                
                logger.info(f"Generated {len(visualization_paths)} HPO visualizations")
                
            except Exception as e:
                logger.warning(f"Failed to generate enhanced analysis: {e}")
          # Compile results
        results = {
            'model_type': model_type,
            'study_name': study.study_name,
            'n_trials': n_trials,
            'n_complete': n_complete,
            'n_pruned': n_pruned,
            'n_failed': n_failed,
            'best_trial_number': best_trial.number,
            'best_value': best_value,
            'best_params': best_params,
            'param_importance': param_importance,
            'study_direction': study.direction.name,
            'convergence_data': convergence_data,
            'visualization_paths': visualization_paths
        }
        
        # Log results
        logger.info(f"HPO Analysis Results for {model_type}:")
        logger.info(f"  Total trials: {n_trials}")
        logger.info(f"  Complete: {n_complete}, Pruned: {n_pruned}, Failed: {n_failed}")
        logger.info(f"  Best trial: {best_trial.number} (value: {best_value:.4f})")
        logger.info(f"  Top 3 important parameters:")
        for i, (param, importance) in enumerate(sorted(param_importance.items(), 
                                                      key=lambda x: x[1], reverse=True)[:3]):
            logger.info(f"    {i+1}. {param}: {importance:.4f}")
        
        # Enhanced logging of results
        if self.enhanced_logger:
            try:
                self.enhanced_logger.log_metrics({
                    f'hpo_analysis_{model_type}_n_trials': n_trials,
                    f'hpo_analysis_{model_type}_completion_rate': n_complete / n_trials if n_trials > 0 else 0,
                    f'hpo_analysis_{model_type}_best_value': best_value
                })
                
                # Log convergence data if available
                if convergence_data:
                    self.enhanced_logger.log_metrics({
                        f'hpo_convergence_{model_type}_{k}': v 
                        for k, v in convergence_data.items() 
                        if isinstance(v, (int, float))
                    })
                
            except Exception as e:
                logger.warning(f"Failed to log enhanced HPO analysis: {e}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save HPO results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_file = output_path / f"hpo_results_{results['model_type']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"HPO results saved to {results_file}")

    def train_and_save_best_model(self, study: optuna.Study, model_type: str, base_config: Dict[str, Any],
                                  use_pregenerated: bool, pregenerated_data_path: Optional[str]):
        """
        Train the best model from HPO study and save it.
        
        Args:
            study: Completed Optuna study object.
            model_type: The model type (e.g., 'lstm').
            base_config: The base configuration used for the HPO study.
            use_pregenerated: Whether to use pre-generated data.
            pregenerated_data_path: Resolved path to the pre-generated dataset, if available.
        """
        logger.info(f"--- Training and Saving Best {model_type.upper()} Model ---")
        
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")

        # Create a clean configuration for the final training run
        final_config = base_config.copy()
        
        # Use HPOObjective's logic to populate the config from best_params
        # This avoids duplicating the complex config creation logic
        objective = HPOObjective(
            base_config,
            model_type,
            use_pregenerated=use_pregenerated,
            pregenerated_data_path=pregenerated_data_path or self.config.get('pregenerated_data_path')
        )
        final_config = objective._create_trial_config(best_params, trial_number=-1) # -1 indicates final run

        # Set a clear output directory for the final model
        final_model_dir = Path(f"models/hpo_derived/{study.study_name}")
        final_model_dir.mkdir(parents=True, exist_ok=True)
        final_config['output_dir'] = str(final_model_dir)
        
        # Ensure MLflow is enabled for this final, important run
        final_config['use_mlflow'] = True
        final_config['experiment_name'] = f"HPO_Final_Models"
        final_config['tags'] = {
            'purpose': 'hpo_best_model',
            'study_name': study.study_name,
            'model_type': model_type,
            'source': 'hpo_retraining'
        }

        # Apply conservative training safeguards for the definitive retraining pass
        final_training_cfg = final_config.setdefault('training_config', {})

        # Guarantee a reasonable minimum epoch count before patience-based stopping
        recommended_min_epochs = max(15, int(final_training_cfg.get('min_epochs', 0)))
        if final_training_cfg.get('min_epochs', 0) < recommended_min_epochs:
            logger.info(
                "Raising min_epochs for final %s retraining to %d to avoid premature stopping.",
                model_type,
                recommended_min_epochs
            )
            final_training_cfg['min_epochs'] = recommended_min_epochs

        # Expand patience window if the HPO trial used an aggressive early-stop setting
        recommended_patience = max(20, int(final_training_cfg.get('early_stopping_patience', 0)))
        if final_training_cfg.get('early_stopping_patience', 0) < recommended_patience:
            logger.info(
                "Increasing early_stopping_patience for final %s retraining to %d.",
                model_type,
                recommended_patience
            )
            final_training_cfg['early_stopping_patience'] = recommended_patience

        # Prevent ultra-low learning rates discovered during HPO from crippling the full retrain
        min_final_lr = max(1e-4, float(final_training_cfg.get('min_learning_rate', 0) or 0))
        if final_training_cfg.get('learning_rate') is not None and final_training_cfg['learning_rate'] < min_final_lr:
            logger.info(
                "Boosting learning rate for final %s retraining from %.2e to %.2e.",
                model_type,
                final_training_cfg['learning_rate'],
                min_final_lr
            )
            final_training_cfg['learning_rate'] = min_final_lr
        final_training_cfg['min_learning_rate'] = max(min_final_lr, 1e-4)

        logger.info(f"Starting final training for best {model_type} model...")
        
        try:
            trainer = ModelTrainer(final_config)
            trainer.load_data()
            trainer.create_model()
            trainer.create_optimizer_and_scheduler()
            trainer.create_loss_function()
            
            # Run the full training process
            trainer.train()

            # Explicitly save the final model artifact
            final_model_path = trainer.save_final_model()
            logger.info(f"✅ Best {model_type} model trained and saved to {final_model_path}")

            # Log the final model path to the parent HPO experiment if possible
            if self.enhanced_logger and mlflow.active_run():
                self.enhanced_logger.log_metadata({
                    f'best_{model_type}_model_path': str(final_model_path)
                })

        except Exception as e:
            logger.error(f"❌ Failed to train and save the best {model_type} model: {e}", exc_info=True)


def create_hpo_config() -> Dict[str, Any]:
    """Create default HPO configuration."""
    return {
        'study_name': f"nn_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'sampler': 'tpe',  # 'tpe' or 'random'
        'pruner': 'median',  # 'median', 'hyperband', or None
        'n_trials': 100,
        'target_metric': 'validation_f1_score_positive_class',
        'direction': 'maximize',
        'storage_url': None,  # Will be auto-generated
        'output_dir': 'hpo_results',
        'models_to_optimize': ['mlp', 'lstm', 'gru', 'cnn_lstm']
    }


def load_hpo_config(config_path: str) -> Dict[str, Any]:
    """Load HPO configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def normalize_hpo_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw configuration structures into the format expected by the HPO manager."""
    if raw_config is None:
        return create_hpo_config()

    normalized = create_hpo_config()
    detected_new_schema = False

    # Direct overrides for known keys
    direct_keys = ['study_name', 'sampler', 'pruner', 'n_trials', 'target_metric',
                   'direction', 'storage_url', 'output_dir', 'models_to_optimize']
    for key in direct_keys:
        if key in raw_config and raw_config[key] is not None:
            normalized[key] = raw_config[key]

    # Handle new schema blocks if present
    experiment_cfg = raw_config.get('experiment_config', {})
    if experiment_cfg:
        detected_new_schema = True
        normalized['study_name'] = experiment_cfg.get('name', normalized['study_name'])
        if 'n_trials' in experiment_cfg and 'n_trials' not in raw_config:
            normalized['n_trials'] = experiment_cfg['n_trials']
        if experiment_cfg.get('output_dir') and 'output_dir' not in raw_config:
            normalized['output_dir'] = experiment_cfg['output_dir']

    optimization_cfg = raw_config.get('optimization', {})
    if optimization_cfg:
        detected_new_schema = True
        if 'metric' in optimization_cfg and 'target_metric' not in raw_config:
            normalized['target_metric'] = optimization_cfg['metric']
        if 'direction' in optimization_cfg and 'direction' not in raw_config:
            normalized['direction'] = optimization_cfg['direction']

    model_cfg = raw_config.get('model_config', {})
    if not normalized.get('models_to_optimize') and model_cfg.get('type'):
        detected_new_schema = True
        normalized['models_to_optimize'] = [model_cfg['type']]

    # Ensure we always have at least one model to optimize
    if not normalized.get('models_to_optimize'):
        normalized['models_to_optimize'] = create_hpo_config()['models_to_optimize']

    if detected_new_schema:
        logger.info("Normalized advanced HPO configuration schema for compatibility")

    return normalized


def main():
    """Main HPO function."""
    parser = argparse.ArgumentParser(description='Run Hyperparameter Optimization for Neural Network Models')
    parser.add_argument('--config', type=str, help='Path to HPO configuration YAML file')
    parser.add_argument('--base-config', type=str, help='Path to base training configuration YAML file')
    parser.add_argument('--model-type', type=str, choices=['mlp', 'lstm', 'gru', 'cnn_lstm'],
                       help='Specific model type to optimize (if not specified, optimizes all)')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of trials per model')
    parser.add_argument('--study-name', type=str, help='Name for the HPO study')
    parser.add_argument('--target-metric', type=str, default='validation_f1_score_positive_class',
                       help='Metric to optimize')
    parser.add_argument('--output-dir', type=str, default='hpo_results', help='Output directory for results')
    parser.add_argument('--use-pregenerated', action='store_true',
                       help='Force use of pre-generated data (auto-detected by default)')
    parser.add_argument('--no-pregenerated', action='store_true',
                       help='Force data generation (ignore pre-generated data)')
    
    args = parser.parse_args()
    
    # Display welcome message and data status
    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold blue]🚀 Neural Network Hyperparameter Optimization[/bold blue]\n"
            "[cyan]Optimizing model performance through systematic hyperparameter search[/cyan]",
            title="[bold green]HPO System[/bold green]",
            border_style="blue"
        ))
    else:
        logger.info("🚀 Starting Neural Network Hyperparameter Optimization")
    
    # Track requested use of pre-generated data (actual status resolved after config loading)
    use_pregenerated = None
    if args.use_pregenerated:
        use_pregenerated = True
    elif args.no_pregenerated:
        use_pregenerated = False
      # Load configurations with enhanced configuration management
    config_manager = None
    try:
        config_manager = ConfigurationManager()
        logger.info("Enhanced configuration management initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize enhanced configuration management: {e}")
        logger.warning("Falling back to basic configuration loading")
    
    if args.config:
        if config_manager:
            try:
                raw_hpo_config = config_manager.load_config(args.config)
            except Exception as e:
                logger.warning(f"Enhanced config loading failed: {e}, using fallback")
                raw_hpo_config = load_hpo_config(args.config)
        else:
            raw_hpo_config = load_hpo_config(args.config)
    else:
        raw_hpo_config = create_hpo_config()

    hpo_config = normalize_hpo_config(raw_hpo_config)
    
    if args.base_config:
        if config_manager:
            try:
                base_config = config_manager.load_config(args.base_config)
            except Exception as e:
                logger.warning(f"Enhanced config loading failed: {e}, using fallback")
                base_config = load_config(args.base_config)
        else:
            base_config = load_config(args.base_config)
    else:
        base_config = create_default_config()
    
    # Validate configurations if config manager is available
    if config_manager:
        try:
            base_config = config_manager.validate_config(base_config)
            # Add HPO-specific validation
            if 'hpo' not in base_config:
                base_config['hpo'] = {'enabled': True}
            base_config['hpo']['enabled'] = True
            base_config['hpo']['n_trials'] = hpo_config.get('n_trials', 100)
            logger.info("Configuration validation completed")
        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}")

    # Resolve pre-generated data path preferences
    preferred_data_path = None
    hpo_data_cfg = raw_hpo_config.get('data_config') if isinstance(raw_hpo_config, dict) else None
    if hpo_data_cfg and hpo_data_cfg.get('pregenerated_path'):
        preferred_data_path = hpo_data_cfg['pregenerated_path']

    base_data_cfg = base_config.get('data_config', {})
    if base_config.get('pregenerated_data_path'):
        preferred_data_path = base_config['pregenerated_data_path']
    elif base_data_cfg.get('pregenerated_data_path'):
        preferred_data_path = base_data_cfg['pregenerated_data_path']

    data_available, resolved_data_path = display_data_status(preferred_data_path)
    if resolved_data_path:
        resolved_path_str = str(resolved_data_path)
        base_config['pregenerated_data_path'] = resolved_path_str
        base_config.setdefault('data_config', {})['pregenerated_data_path'] = resolved_path_str
        hpo_config['pregenerated_data_path'] = resolved_path_str
    elif preferred_data_path:
        base_config['pregenerated_data_path'] = preferred_data_path
        base_config.setdefault('data_config', {})['pregenerated_data_path'] = preferred_data_path
        hpo_config['pregenerated_data_path'] = preferred_data_path

    if use_pregenerated is None:
        use_pregenerated = data_available
    elif use_pregenerated and not data_available:
        logger.warning("Pre-generated data requested via CLI but not found. Falling back to on-the-fly generation.")
        use_pregenerated = False

    base_config['use_pregenerated_data'] = use_pregenerated

    # Override with command line arguments
    if args.model_type:
        hpo_config['models_to_optimize'] = [args.model_type]
    if args.n_trials:
        hpo_config['n_trials'] = args.n_trials
    if args.study_name:
        hpo_config['study_name'] = args.study_name
    if args.target_metric:
        hpo_config['target_metric'] = args.target_metric
    if args.output_dir:
        hpo_config['output_dir'] = args.output_dir
    
    # Setup GPU if available
    if torch.cuda.is_available():
        gpu_info = setup_gpu()
        logger.info(f"GPU setup: {gpu_info}")
    
    # Create HPO manager
    hpo_manager = HPOManager(hpo_config)
    
    # Run optimization for each model type
    models_to_optimize = hpo_config['models_to_optimize']
    all_results = {}
    
    for model_type in models_to_optimize:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting HPO for {model_type.upper()} model")
        logger.info(f"{'='*60}")
        
        try:
            # Run optimization
            study = hpo_manager.run_optimization(
                model_type=model_type,
                base_config=base_config,
                n_trials=hpo_config['n_trials'],
                target_metric=hpo_config['target_metric'],
                use_pregenerated=use_pregenerated,
                pregenerated_data_path=hpo_config.get('pregenerated_data_path')
            )
            
            # Analyze results
            results = hpo_manager.analyze_results(study, model_type)
            all_results[model_type] = results
            
            # Save individual results
            hpo_manager.save_results(results, hpo_config['output_dir'])

            # Train and save the best model from this study
            if study.best_trial:
                hpo_manager.train_and_save_best_model(
                    study,
                    model_type,
                    base_config,
                    use_pregenerated,
                    hpo_config.get('pregenerated_data_path')
                )
            else:
                logger.warning(f"No best trial found for {model_type}, skipping final model training.")
            
        except Exception as e:
            logger.error(f"HPO failed for {model_type}: {e}")
            continue
    
    # Save combined results
    if all_results:
        combined_results = {
            'hpo_config': hpo_config,
            'raw_hpo_config': raw_hpo_config,
            'base_config': base_config,
            'results_by_model': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        output_path = Path(hpo_config['output_dir'])
        combined_file = output_path / "hpo_combined_results.json"
        with open(combined_file, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        logger.info(f"\n{'='*60}")
        logger.info("HPO SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Combined results saved to {combined_file}")
        
        # Print summary of best results
        for model_type, results in all_results.items():
            logger.info(f"{model_type.upper()}: Best value = {results['best_value']:.4f} "
                       f"(Trial {results['best_trial_number']})")
        
        # Find overall best model
        best_model = max(all_results.items(), key=lambda x: x[1]['best_value'])
        logger.info(f"\nOverall best model: {best_model[0].upper()} "
                   f"(value: {best_model[1]['best_value']:.4f})")
        
        logger.info(f"\nHPO completed successfully!")
    else:
        logger.error("No successful HPO runs completed")
        sys.exit(1)


if __name__ == '__main__':
    main()