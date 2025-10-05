"""
Enhanced MLflow Logging System for Advanced Experiment Tracking

This module provides comprehensive MLflow logging capabilities including
model architecture summaries, scalers, data information, and enhanced
environment metadata tracking.

Part of Activity 6.2: Enhanced MLflow Logging
"""

import os
import json
import pickle
import tempfile
import subprocess
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
import torch
import torch.nn as nn

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

logger = logging.getLogger(__name__)


class EnhancedMLflowLogger:
    """
    Enhanced MLflow logger with comprehensive experiment tracking capabilities.
    
    Provides advanced logging for model architecture, scalers, data information,
    git commits, environment details, and automated visualizations.
    """
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None):
        """
        Initialize enhanced MLflow logger.
        
        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional run name (auto-generated if None)
            tags: Optional tags for the run
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.run_id = None
        self.is_active = False
        
        # Tracking state
        self.logged_artifacts = []
        self.logged_metrics = {}
        self.logged_params = {}
        
    def start_run(self) -> str:
        """
        Start MLflow run with enhanced initialization and robust experiment handling.
        
        Returns:
            MLflow run ID
        """
        try:
            # Robust experiment setup: handle missing and deleted experiments
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            if experiment is None:
                # Experiment doesn't exist, create it
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
            elif experiment.lifecycle_stage == 'deleted':
                # Experiment was deleted, restore it
                mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
                experiment_id = experiment.experiment_id
                logger.info(f"Restored deleted MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                # Experiment exists and is active
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
            
            # Set the experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Generate run name if not provided
            if self.run_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.run_name = f"run_{timestamp}"
            
            # Start run
            run = mlflow.start_run(run_name=self.run_name)
            self.run_id = run.info.run_id
            self.is_active = True
            
            # Log initial tags
            initial_tags = {
                'framework': 'pytorch',
                'task': 'binary_classification',
                'status': 'running',
                'created_at': datetime.now().isoformat()
            }
            initial_tags.update(self.tags)
            mlflow.set_tags(initial_tags)
            
            # Log comprehensive environment information
            self._log_environment_info()
            
            # Log git information
            self._log_git_info()
            
            logger.info(f"Started MLflow run: {self.run_name} (ID: {self.run_id})")
            return self.run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise
    
    def _log_environment_info(self) -> None:
        """Log comprehensive environment and system information."""
        try:
            # Python environment
            import sys
            env_info = {
                'python_version': sys.version.split()[0],
                'python_executable': sys.executable,
                'platform_system': platform.system(),
                'platform_release': platform.release(),
                'platform_machine': platform.machine(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
            
            # Package versions
            package_versions = {}
            try:
                import torch
                package_versions['pytorch'] = torch.__version__
            except ImportError:
                pass
            
            try:
                import numpy
                package_versions['numpy'] = numpy.__version__
            except ImportError:
                pass
            
            try:
                import pandas
                package_versions['pandas'] = pandas.__version__
            except ImportError:
                pass
            
            try:
                import sklearn
                package_versions['scikit_learn'] = sklearn.__version__
            except ImportError:
                pass
            
            try:
                import mlflow
                package_versions['mlflow'] = mlflow.__version__
            except ImportError:
                pass
            
            # GPU information
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info.update({
                    'gpu_available': True,
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                    'cuda_version': torch.version.cuda
                })
            else:
                gpu_info['gpu_available'] = False
            
            # Log all environment information
            mlflow.log_params(env_info)
            mlflow.log_params({f"pkg_{k}": v for k, v in package_versions.items()})
            mlflow.log_params(gpu_info)
            
            # Create and log environment summary as artifact
            env_summary = {
                'environment': env_info,
                'packages': package_versions,
                'gpu': gpu_info,
                'timestamp': datetime.now().isoformat()
            }
            
            self._log_json_artifact(env_summary, 'environment_info.json')
            
        except Exception as e:
            logger.warning(f"Failed to log environment info: {e}")
    
    def _log_git_info(self) -> None:
        """Log git repository information."""
        try:
            # Get git commit hash
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            
            # Get git branch
            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            
            # Get git status (check for uncommitted changes)
            git_status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            
            has_uncommitted = len(git_status) > 0
            
            # Get commit message
            git_message = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%B'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            
            # Get commit date
            git_date = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%cd', '--date=iso'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()
            
            git_info = {
                'git_commit': git_commit,
                'git_branch': git_branch,
                'git_has_uncommitted_changes': has_uncommitted,
                'git_commit_message': git_message[:100],  # Truncate long messages
                'git_commit_date': git_date
            }
            
            # Log params with error handling for duplicates
            try:
                mlflow.log_params(git_info)
            except Exception as param_error:
                logger.debug(f"Could not log git params (may already be logged): {param_error}")
            
            # Log detailed git info as artifact
            detailed_git_info = {
                'commit': git_commit,
                'branch': git_branch,
                'has_uncommitted_changes': has_uncommitted,
                'commit_message': git_message,
                'commit_date': git_date,
                'status': git_status,
                'timestamp': datetime.now().isoformat()
            }
            
            self._log_json_artifact(detailed_git_info, 'git_info.json')
            
        except subprocess.CalledProcessError:
            logger.warning("Not in a git repository or git not available")
        except Exception as e:
            logger.warning(f"Failed to log git info: {e}")
    
    def log_model_architecture(self, model: nn.Module, input_shape: Tuple[int, ...],
                             config: Optional[Dict[str, Any]] = None) -> None:
        """
        Log comprehensive model architecture information.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (batch_size, ...)
            config: Optional model configuration dictionary
        """
        try:
            # Create model summary
            model_summary = self._create_model_summary(model, input_shape)
            
            # Log model configuration
            if config:
                mlflow.log_params({f"model_{k}": v for k, v in config.items()})
            
            # Log model parameters count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            mlflow.log_params({
                'model_total_params': total_params,
                'model_trainable_params': trainable_params,
                'model_non_trainable_params': total_params - trainable_params
            })
            
            # Save model summary as artifact
            self._log_text_artifact(model_summary, 'model_summary.txt')
            
            # Save model architecture visualization if possible
            try:
                self._save_model_architecture_diagram(model, input_shape)
            except Exception as e:
                logger.warning(f"Could not create model architecture diagram: {e}")
            
            logger.info("Model architecture logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log model architecture: {e}")
    
    def _create_model_summary(self, model: nn.Module, input_shape: Tuple[int, ...]) -> str:
        """Create detailed model summary string."""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("MODEL ARCHITECTURE SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Model: {model.__class__.__name__}")
        summary_lines.append(f"Input shape: {input_shape}")
        summary_lines.append("")
        
        # Layer-by-layer summary
        summary_lines.append("Layer-by-Layer Summary:")
        summary_lines.append("-" * 80)
        summary_lines.append(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':<15}")
        summary_lines.append("=" * 80)
        
        total_params = 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                num_params = sum(p.numel() for p in module.parameters())
                total_params += num_params
                
                # Try to get output shape (simplified)
                output_shape = "Unknown"
                try:
                    if hasattr(module, 'out_features'):
                        output_shape = f"(-1, {module.out_features})"
                    elif hasattr(module, 'num_features'):
                        output_shape = f"(-1, {module.num_features})"
                except:
                    pass
                
                summary_lines.append(f"{name:<30} {output_shape:<20} {num_params:<15,}")
        
        summary_lines.append("=" * 80)
        summary_lines.append(f"Total params: {total_params:,}")
        summary_lines.append(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        summary_lines.append(f"Non-trainable params: {total_params - sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)
    
    def _save_model_architecture_diagram(self, model: nn.Module, input_shape: Tuple[int, ...]) -> None:
        """Save model architecture diagram (simplified visualization)."""
        # Create a simple text-based architecture diagram
        diagram_lines = []
        diagram_lines.append("Model Architecture Flow:")
        diagram_lines.append("=" * 50)
        
        prev_shape = input_shape
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_name = module.__class__.__name__
                
                # Estimate output shape
                if hasattr(module, 'out_features'):
                    current_shape = f"(..., {module.out_features})"
                elif hasattr(module, 'num_features'):
                    current_shape = f"(..., {module.num_features})"
                else:
                    current_shape = "(..., ?)"
                
                diagram_lines.append(f"{prev_shape}")
                diagram_lines.append("    ↓")
                diagram_lines.append(f"[{module_name}]")
                diagram_lines.append("    ↓")
                
                prev_shape = current_shape
        
        diagram_lines.append(f"{prev_shape}")
        
        self._log_text_artifact("\n".join(diagram_lines), 'model_architecture_diagram.txt')
    
    def log_data_info(self, train_data: Dict[str, Any], val_data: Dict[str, Any],
                     test_data: Optional[Dict[str, Any]] = None,
                     feature_names: Optional[List[str]] = None) -> None:
        """
        Log comprehensive data information including sample counts, feature info, and class distribution.
        
        Args:
            train_data: Training data dictionary with 'X', 'y', 'asset_ids'
            val_data: Validation data dictionary
            test_data: Optional test data dictionary
            feature_names: Optional list of feature names
        """
        try:
            data_info = {}
            
            # Sample counts
            data_info['train_samples'] = len(train_data['y'])
            data_info['val_samples'] = len(val_data['y'])
            if test_data:
                data_info['test_samples'] = len(test_data['y'])
            
            # Feature information
            if 'X' in train_data:
                X_shape = train_data['X'].shape
                data_info['input_shape'] = list(X_shape)
                data_info['sequence_length'] = X_shape[1] if len(X_shape) > 2 else 1
                data_info['num_features'] = X_shape[-1]
            
            # Class distribution
            train_class_dist = np.bincount(train_data['y'])
            val_class_dist = np.bincount(val_data['y'])
            
            data_info['train_class_0'] = int(train_class_dist[0])
            data_info['train_class_1'] = int(train_class_dist[1]) if len(train_class_dist) > 1 else 0
            data_info['train_class_imbalance_ratio'] = float(train_class_dist[1] / train_class_dist[0]) if train_class_dist[0] > 0 and len(train_class_dist) > 1 else 0.0
            
            data_info['val_class_0'] = int(val_class_dist[0])
            data_info['val_class_1'] = int(val_class_dist[1]) if len(val_class_dist) > 1 else 0
            data_info['val_class_imbalance_ratio'] = float(val_class_dist[1] / val_class_dist[0]) if val_class_dist[0] > 0 and len(val_class_dist) > 1 else 0.0
            
            if test_data:
                test_class_dist = np.bincount(test_data['y'])
                data_info['test_class_0'] = int(test_class_dist[0])
                data_info['test_class_1'] = int(test_class_dist[1]) if len(test_class_dist) > 1 else 0
                data_info['test_class_imbalance_ratio'] = float(test_class_dist[1] / test_class_dist[0]) if test_class_dist[0] > 0 and len(test_class_dist) > 1 else 0.0
            
            # Asset ID information
            if 'asset_ids' in train_data:
                unique_assets = np.unique(train_data['asset_ids'])
                data_info['num_unique_assets'] = len(unique_assets)
                data_info['asset_ids'] = unique_assets.tolist()
            
            # Log as parameters
            mlflow.log_params(data_info)
            
            # Create detailed data summary
            data_summary = {
                'data_info': data_info,
                'feature_names': feature_names,
                'class_distribution': {
                    'train': train_class_dist.tolist(),
                    'validation': val_class_dist.tolist(),
                    'test': test_class_dist.tolist() if test_data else None
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self._log_json_artifact(data_summary, 'data_info.json')
            
            # Create and log class distribution plot
            self._plot_class_distribution(train_class_dist, val_class_dist, 
                                        test_class_dist if test_data else None)
            
            logger.info("Data information logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log data info: {e}")
    
    def _plot_class_distribution(self, train_dist: np.ndarray, val_dist: np.ndarray,
                               test_dist: Optional[np.ndarray] = None) -> None:
        """Create and log class distribution plot."""
        try:
            fig, axes = plt.subplots(1, 3 if test_dist is not None else 2, figsize=(15, 5))
            if test_dist is None:
                axes = [axes[0], axes[1]]
            
            # Training distribution
            axes[0].bar(['Class 0', 'Class 1'], train_dist)
            axes[0].set_title('Training Class Distribution')
            axes[0].set_ylabel('Count')
            
            # Validation distribution
            axes[1].bar(['Class 0', 'Class 1'], val_dist)
            axes[1].set_title('Validation Class Distribution')
            axes[1].set_ylabel('Count')
            
            # Test distribution (if available)
            if test_dist is not None:
                axes[2].bar(['Class 0', 'Class 1'], test_dist)
                axes[2].set_title('Test Class Distribution')
                axes[2].set_ylabel('Count')
            
            plt.tight_layout()
            self._log_figure(fig, 'class_distribution.png')
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Failed to create class distribution plot: {e}")
    
    def log_scalers(self, scalers: Dict[str, Any]) -> None:
        """
        Log data scalers as artifacts.
        
        Args:
            scalers: Dictionary of scalers (e.g., {'feature_scaler': scaler})
        """
        try:
            for scaler_name, scaler in scalers.items():
                # Save scaler as pickle
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    pickle.dump(scaler, f)
                    temp_path = f.name
                
                mlflow.log_artifact(temp_path, f'scalers/{scaler_name}.pkl')
                os.unlink(temp_path)
                
                # Log scaler metadata
                scaler_info = {
                    'type': scaler.__class__.__name__,
                    'feature_names': getattr(scaler, 'feature_names_in_', None),
                    'n_features': getattr(scaler, 'n_features_in_', None),
                    'mean': getattr(scaler, 'mean_', None),
                    'scale': getattr(scaler, 'scale_', None)
                }
                
                # Convert numpy arrays to lists for JSON serialization
                for key, value in scaler_info.items():
                    if hasattr(value, 'tolist'):
                        scaler_info[key] = value.tolist()
                
                self._log_json_artifact(scaler_info, f'scalers/{scaler_name}_info.json')
            
            logger.info(f"Logged {len(scalers)} scalers")
            
        except Exception as e:
            logger.error(f"Failed to log scalers: {e}")
    
    def log_training_plots(self, train_metrics: List[Dict[str, float]], 
                          val_metrics: List[Dict[str, float]]) -> None:
        """
        Create and log training visualization plots.
        
        Args:
            train_metrics: List of training metrics per epoch
            val_metrics: List of validation metrics per epoch
        """
        try:
            # Create learning curves
            self._plot_learning_curves(train_metrics, val_metrics)
            
            # Create metrics comparison plot
            self._plot_metrics_comparison(train_metrics, val_metrics)
            
            logger.info("Training plots logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log training plots: {e}")
    
    def _plot_learning_curves(self, train_metrics: List[Dict[str, float]],
                            val_metrics: List[Dict[str, float]]) -> None:
        """Create learning curves plot."""
        if not train_metrics or not val_metrics:
            return
        
        epochs = range(1, len(train_metrics) + 1)
        
        # Get available metrics
        metric_keys = set(train_metrics[0].keys()) & set(val_metrics[0].keys())
        if 'epoch' in metric_keys:
            metric_keys.remove('epoch')
        
        n_metrics = len(metric_keys)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metric_keys):
            if i >= len(axes):
                break
                
            train_values = [m.get(metric, 0) for m in train_metrics]
            val_values = [m.get(metric, 0) for m in val_metrics]
            
            axes[i].plot(epochs, train_values, label=f'Train {metric}', marker='o')
            axes[i].plot(epochs, val_values, label=f'Val {metric}', marker='s')
            axes[i].set_title(f'{metric.title()} Learning Curve')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.title())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metric_keys), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        self._log_figure(fig, 'learning_curves.png')
        plt.close(fig)
    
    def _plot_metrics_comparison(self, train_metrics: List[Dict[str, float]],
                               val_metrics: List[Dict[str, float]]) -> None:
        """Create metrics comparison plot."""
        if not train_metrics or not val_metrics:
            return
        
        # Get final metrics
        final_train = train_metrics[-1]
        final_val = val_metrics[-1]
        
        metric_keys = set(final_train.keys()) & set(final_val.keys())
        if 'epoch' in metric_keys:
            metric_keys.remove('epoch')
        
        if not metric_keys:
            return
        
        metrics = list(metric_keys)
        train_values = [final_train.get(m, 0) for m in metrics]
        val_values = [final_val.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
        ax.bar(x + width/2, val_values, width, label='Validation', alpha=0.8)
        
        ax.set_title('Final Metrics Comparison')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._log_figure(fig, 'metrics_comparison.png')
        plt.close(fig)
    
    def log_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None,
                           class_names: Optional[List[str]] = None,
                           current_epoch: int = 0,
                           stage: str = "eval") -> None:
        """
        Create and log evaluation plots (confusion matrix, ROC, PR curves).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for ROC/PR curves)
            class_names: Optional class names
            current_epoch: The current epoch number
            stage: The stage of evaluation (e.g., 'validation', 'test')
        """
        try:
            if class_names is None:
                class_names = ['Class 0', 'Class 1']
            
            # Confusion matrix
            self._plot_confusion_matrix(y_true, y_pred, class_names, current_epoch, stage)
            
            # ROC and PR curves (if probabilities available)
            if y_prob is not None:
                self._plot_roc_curve(y_true, y_prob, current_epoch, stage)
                self._plot_pr_curve(y_true, y_prob, current_epoch, stage)
            
            logger.info(f"Evaluation plots for epoch {current_epoch} ({stage}) logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log evaluation plots: {e}")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str], current_epoch: int, stage: str) -> None:
        """Create confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'Confusion Matrix - {stage.title()} Epoch {current_epoch}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        filename = f'{stage}_confusion_matrix_epoch_{current_epoch:03d}.png'
        self._log_figure(fig, filename)
        plt.close(fig)
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, current_epoch: int, stage: str) -> None:
        """Create ROC curve plot."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {stage.title()} Epoch {current_epoch}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{stage}_roc_curve_epoch_{current_epoch:03d}.png'
        self._log_figure(fig, filename)
        plt.close(fig)
    
    def _plot_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray, current_epoch: int, stage: str) -> None:
        """Create Precision-Recall curve plot."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {stage.title()} Epoch {current_epoch}')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{stage}_pr_curve_epoch_{current_epoch:03d}.png'
        self._log_figure(fig, filename)
        plt.close(fig)
    
    def _log_figure(self, figure, filename: str) -> None:
        """Log matplotlib figure as artifact."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            figure.savefig(f.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(f.name, filename)
            os.unlink(f.name)
    
    def _log_text_artifact(self, content: str, filename: str) -> None:
        """Log text content as artifact."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            mlflow.log_artifact(f.name, filename)
            os.unlink(f.name)
    
    def _log_json_artifact(self, data: Dict[str, Any], filename: str) -> None:
        """Log JSON data as artifact."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            mlflow.log_artifact(f.name, filename)
            os.unlink(f.name)
    
    def end_run(self, status: str = 'FINISHED') -> None:
        """
        End MLflow run.
        
        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')
        """
        if self.is_active:
            # Update final tags
            mlflow.set_tag('status', status.lower())
            mlflow.set_tag('ended_at', datetime.now().isoformat())
            
            mlflow.end_run()
            self.is_active = False
            logger.info(f"Ended MLflow run: {self.run_name} with status: {status}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if self.is_active:
            mlflow.log_params(params)
            self.logged_params.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        if self.is_active:
            mlflow.log_metrics(metrics, step=step)
            for key, value in metrics.items():
                if key not in self.logged_metrics:
                    self.logged_metrics[key] = []
                self.logged_metrics[key].append(value)
    
    def log_artifact(self, artifact_path: str, artifact_folder: Optional[str] = None) -> None:
        """Log artifact to MLflow."""
        if self.is_active:
            mlflow.log_artifact(artifact_path, artifact_folder)
            self.logged_artifacts.append(artifact_path)
    
    def log_model(self, model: nn.Module, model_name: str = "model") -> None:
        """Log PyTorch model to MLflow."""
        if self.is_active:
            mlflow.pytorch.log_model(model, model_name)
            logger.info(f"Model logged as '{model_name}'")


def create_enhanced_logger(experiment_name: str, run_name: Optional[str] = None,
                         tags: Optional[Dict[str, str]] = None) -> EnhancedMLflowLogger:
    """
    Create and start enhanced MLflow logger.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Optional run name
        tags: Optional tags for the run
        
    Returns:
        Started EnhancedMLflowLogger instance
    """
    logger_instance = EnhancedMLflowLogger(experiment_name, run_name, tags)
    logger_instance.start_run()
    return logger_instance
