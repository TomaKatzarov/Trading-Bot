"""
Centralized Configuration Management for Advanced Experiment Tracking

This module provides robust configuration handling with YAML support,
CLI argument integration, and validation for neural network training experiments.

Part of Activity 6.1: Centralized Configuration Management
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Centralized configuration manager for neural network training experiments.
    
    Provides robust YAML/CLI configuration handling with validation,
    defaults management, and experiment-specific overrides.
    """
    
    def __init__(self, base_config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            base_config_path: Path to base configuration file
        """
        self.base_config_path = base_config_path
        self.config = {}
        self.config_history = []
        
    def load_base_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load base configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        if config_path is None:
            config_path = self.base_config_path
            
        if config_path is None:
            raise ValueError("No configuration path provided")
            
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Record configuration loading
            self.config_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'load_base',
                'source': str(config_path),
                'config_keys': list(config.keys()) if config else []
            })
            
            logger.info(f"Loaded base configuration from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration for neural network training.
        
        Returns:
            Default configuration dictionary
        """
        default_config = {
            # Model configuration
            'model_type': 'mlp',
            'model_config': {
                'mlp': {
                    'hidden_sizes': [256, 128, 64],
                    'dropout_rate': 0.3,
                    'activation': 'relu',
                    'batch_norm': True
                },
                'lstm': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'bidirectional': False
                },
                'gru': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'bidirectional': False
                }
            },
            
            # Training configuration
            'training': {
                'batch_size': 512,
                'learning_rate': 0.001,
                'num_epochs': 100,
                'patience': 15,
                'min_delta': 1e-6,
                'optimizer': 'adamw',
                'scheduler': 'reduce_on_plateau',
                'weight_decay': 1e-4,
                'gradient_clipping': 1.0
            },
            
            # Data configuration
            'data_config': {
                'lookback_window': 24,
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
                'features': 'all',
                'target_config': {
                    'profit_target': 0.05,
                    'stop_loss': 0.02,
                    'time_horizon': 8
                },
                'scaling': {
                    'method': 'standard',
                    'robust_for_volume': True
                },
                'augmentation': {
                    'enabled': False,
                    'oversampling_ratio': 2.0,
                    'noise_level': 0.001,
                    'time_shift_range': 2
                }
            },
            
            # Loss function configuration
            'loss_config': {
                'type': 'focal',
                'focal_alpha': 0.25,
                'focal_gamma': 2.0,
                'class_weights': None,
                'label_smoothing': 0.0
            },
            
            # Experiment tracking
            'experiment_name': 'nn_training_default',
            'use_mlflow': True,
            'save_artifacts': True,
            'plot_metrics': True,
            
            # HPO configuration
            'hpo': {
                'enabled': False,
                'n_trials': 100,
                'timeout': None,
                'study_name': None,
                'direction': 'maximize',
                'metric': 'val_f1'
            },
            
            # Environment and infrastructure
            'device': 'auto',
            'mixed_precision': False,
            'compile_model': False,
            'num_workers': 4,
            'pin_memory': True,
            
            # Logging and debugging
            'log_level': 'INFO',
            'debug_mode': False,
            'profile_training': False,
            
            # Tags for organization
            'tags': {
                'stage': 'development',
                'version': '1.0',
                'framework': 'pytorch'
            }
        }
        
        # Record default config creation
        self.config_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'create_default',
            'source': 'internal',
            'config_keys': list(default_config.keys())
        })
        
        return default_config
    
    def apply_cli_overrides(self, config: Dict[str, Any], 
                          cli_args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
        """
        Apply command-line argument overrides to configuration.
        
        Args:
            config: Base configuration dictionary
            cli_args: Parsed CLI arguments
            
        Returns:
            Configuration with CLI overrides applied
        """
        if cli_args is None:
            return config
            
        # Track overrides
        overrides = {}
        
        # Model type override
        if hasattr(cli_args, 'model_type') and cli_args.model_type:
            config['model_type'] = cli_args.model_type
            overrides['model_type'] = cli_args.model_type
            
        # Training parameter overrides
        if hasattr(cli_args, 'batch_size') and cli_args.batch_size:
            config['training']['batch_size'] = cli_args.batch_size
            overrides['training.batch_size'] = cli_args.batch_size
            
        if hasattr(cli_args, 'learning_rate') and cli_args.learning_rate:
            config['training']['learning_rate'] = cli_args.learning_rate
            overrides['training.learning_rate'] = cli_args.learning_rate
            
        if hasattr(cli_args, 'num_epochs') and cli_args.num_epochs:
            config['training']['num_epochs'] = cli_args.num_epochs
            overrides['training.num_epochs'] = cli_args.num_epochs
            
        # Experiment name override
        if hasattr(cli_args, 'experiment_name') and cli_args.experiment_name:
            config['experiment_name'] = cli_args.experiment_name
            overrides['experiment_name'] = cli_args.experiment_name
            
        # HPO overrides
        if hasattr(cli_args, 'enable_hpo') and cli_args.enable_hpo:
            config['hpo']['enabled'] = True
            overrides['hpo.enabled'] = True
            
        if hasattr(cli_args, 'n_trials') and cli_args.n_trials:
            config['hpo']['n_trials'] = cli_args.n_trials
            overrides['hpo.n_trials'] = cli_args.n_trials
            
        # Debug mode override
        if hasattr(cli_args, 'debug') and cli_args.debug:
            config['debug_mode'] = True
            config['log_level'] = 'DEBUG'
            overrides['debug_mode'] = True
            
        # Record CLI overrides
        if overrides:
            self.config_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'cli_override',
                'source': 'command_line',
                'overrides': overrides
            })
            
            logger.info(f"Applied CLI overrides: {overrides}")
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and set reasonable defaults.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        validation_errors = []
        
        # Validate model type
        valid_model_types = ['mlp', 'lstm', 'gru', 'cnn_lstm']
        if config.get('model_type') not in valid_model_types:
            validation_errors.append(f"Invalid model_type. Must be one of {valid_model_types}")
        
        # Validate training parameters
        training = config.get('training', {})
        if training.get('batch_size', 0) <= 0:
            validation_errors.append("batch_size must be positive")
        if training.get('learning_rate', 0) <= 0:
            validation_errors.append("learning_rate must be positive")
        if training.get('num_epochs', 0) <= 0:
            validation_errors.append("num_epochs must be positive")
        
        # Validate data configuration
        data_config = config.get('data_config', {})
        if data_config.get('lookback_window', 0) <= 0:
            validation_errors.append("lookback_window must be positive")
        
        symbols = data_config.get('symbols', [])
        if not symbols or len(symbols) == 0:
            validation_errors.append("At least one symbol must be specified")
        
        # Validate target configuration
        target_config = data_config.get('target_config', {})
        profit_target = target_config.get('profit_target', 0)
        stop_loss = target_config.get('stop_loss', 0)
        if profit_target <= 0:
            validation_errors.append("profit_target must be positive")
        if stop_loss <= 0:
            validation_errors.append("stop_loss must be positive")
        
        # Validate experiment name
        experiment_name = config.get('experiment_name', '')
        if not experiment_name or not isinstance(experiment_name, str):
            config['experiment_name'] = 'nn_training_default'
            logger.warning("Invalid experiment_name, using default")
        
        # Validate HPO configuration
        hpo_config = config.get('hpo', {})
        if hpo_config.get('enabled', False):
            if hpo_config.get('n_trials', 0) <= 0:
                validation_errors.append("HPO n_trials must be positive when HPO is enabled")
        
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")
        
        # Record validation
        self.config_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'validate',
            'source': 'internal',
            'status': 'passed'
        })
        
        return config
    
    def save_config(self, config: Dict[str, Any], save_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        config_with_meta = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'config_history': self.config_history,
                'version': '1.0'
            },
            'config': config
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_with_meta, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def get_experiment_config(self, base_config_path: Optional[str] = None,
                            cli_args: Optional[argparse.Namespace] = None,
                            experiment_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get complete experiment configuration with all overrides applied.
        
        Args:
            base_config_path: Path to base configuration file
            cli_args: Parsed CLI arguments
            experiment_overrides: Additional experiment-specific overrides
            
        Returns:
            Complete experiment configuration
        """
        # Start with default configuration
        config = self.create_default_config()
        
        # Load base configuration if provided
        if base_config_path:
            try:
                base_config = self.load_base_config(base_config_path)
                config = self._deep_merge(config, base_config)
            except Exception as e:
                logger.warning(f"Failed to load base config: {e}")
        
        # Apply CLI overrides
        config = self.apply_cli_overrides(config, cli_args)
        
        # Apply experiment-specific overrides
        if experiment_overrides:
            config = self._deep_merge(config, experiment_overrides)
            self.config_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'experiment_override',
                'source': 'experiment_specific',
                'overrides': experiment_overrides
            })
        
        # Validate final configuration
        config = self.validate_config(config)
        
        # Store final configuration
        self.config = config
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_cli_parser(self) -> argparse.ArgumentParser:
        """
        Get command-line argument parser with all configuration options.
        
        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(description='Neural Network Training Configuration')
        
        # Configuration file
        parser.add_argument('--config', type=str, help='Path to configuration YAML file')
        
        # Model configuration
        parser.add_argument('--model-type', choices=['mlp', 'lstm', 'gru', 'cnn_lstm'],
                          help='Model architecture type')
        
        # Training parameters
        parser.add_argument('--batch-size', type=int, help='Training batch size')
        parser.add_argument('--learning-rate', type=float, help='Learning rate')
        parser.add_argument('--num-epochs', type=int, help='Number of training epochs')
        
        # Experiment configuration
        parser.add_argument('--experiment-name', type=str, help='Experiment name for tracking')
        
        # HPO configuration
        parser.add_argument('--enable-hpo', action='store_true', help='Enable hyperparameter optimization')
        parser.add_argument('--n-trials', type=int, help='Number of HPO trials')
        
        # Debugging
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        
        # Output
        parser.add_argument('--save-config', type=str, help='Save final configuration to file')
        
        return parser


def load_config(config_path: Optional[str] = None, 
               cli_args: Optional[argparse.Namespace] = None,
               experiment_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to load and process configuration.
    
    Args:
        config_path: Path to configuration file
        cli_args: Parsed CLI arguments
        experiment_overrides: Additional overrides
        
    Returns:
        Processed configuration dictionary
    """
    config_manager = ConfigurationManager()
    return config_manager.get_experiment_config(
        base_config_path=config_path,
        cli_args=cli_args,
        experiment_overrides=experiment_overrides
    )


def create_default_config(save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create and optionally save default configuration.
    
    Args:
        save_path: Optional path to save configuration
        
    Returns:
        Default configuration dictionary
    """
    config_manager = ConfigurationManager()
    config = config_manager.create_default_config()
    
    if save_path:
        config_manager.save_config(config, save_path)
    
    return config


if __name__ == '__main__':
    # CLI interface for configuration management
    parser = ConfigurationManager().get_cli_parser()
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigurationManager()
    config = config_manager.get_experiment_config(
        base_config_path=args.config,
        cli_args=args
    )
    
    print("Final Configuration:")
    print(yaml.dump(config, default_flow_style=False, indent=2))
    
    # Save configuration if requested
    if args.save_config:
        config_manager.save_config(config, args.save_config)
