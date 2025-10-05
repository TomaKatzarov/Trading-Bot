"""
Experiment Organization and Tagging System

This module provides structured experiment naming conventions, comprehensive
tagging systems, and parent/child run relationships for HPO studies.

Part of Activity 6.3: Experiment Organization and Tagging
"""

import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

logger = logging.getLogger(__name__)


class ExperimentStage(Enum):
    """Experiment stage enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    VALIDATION = "validation"
    PRODUCTION = "production"
    RESEARCH = "research"


class ExperimentType(Enum):
    """Experiment type enumeration."""
    SINGLE_TRAINING = "single_training"
    HPO_STUDY = "hpo_study"
    HPO_TRIAL = "hpo_trial"
    ABLATION_STUDY = "ablation_study"
    BASELINE = "baseline"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"


@dataclass
class ExperimentMetadata:
    """Container for experiment metadata."""
    stage: ExperimentStage
    type: ExperimentType
    model_type: str
    dataset_version: str
    purpose: str
    researcher: Optional[str] = None
    parent_run_id: Optional[str] = None
    study_name: Optional[str] = None
    trial_number: Optional[int] = None
    custom_tags: Optional[Dict[str, str]] = None


class ExperimentOrganizer:
    """
    Manages experiment organization with structured naming and comprehensive tagging.
    
    Provides consistent naming conventions, hierarchical organization,
    and parent/child relationships for complex experiment workflows.
    """
    
    def __init__(self):
        """Initialize experiment organizer."""
        self.naming_patterns = {
            ExperimentType.SINGLE_TRAINING: "{stage}_{model_type}_{purpose}_{timestamp}",
            ExperimentType.HPO_STUDY: "HPO_{stage}_{model_type}_{purpose}_{timestamp}",
            ExperimentType.HPO_TRIAL: "HPO_trial_{parent_study}_{trial:03d}",
            ExperimentType.ABLATION_STUDY: "ABLATION_{stage}_{model_type}_{component}_{timestamp}",
            ExperimentType.BASELINE: "BASELINE_{stage}_{model_type}_{timestamp}",
            ExperimentType.COMPARISON: "COMPARE_{stage}_{models}_{timestamp}",
            ExperimentType.ANALYSIS: "ANALYSIS_{stage}_{focus}_{timestamp}"
        }
    
    def generate_experiment_name(self, metadata: ExperimentMetadata) -> str:
        """
        Generate structured experiment name based on metadata.
        
        Args:
            metadata: Experiment metadata
            
        Returns:
            Generated experiment name
        """
        pattern = self.naming_patterns.get(metadata.type)
        if not pattern:
            raise ValueError(f"No naming pattern for experiment type: {metadata.type}")
        
        # Prepare substitution variables
        substitutions = {
            'stage': metadata.stage.value,
            'model_type': metadata.model_type,
            'purpose': self._sanitize_name(metadata.purpose),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'dataset_version': metadata.dataset_version
        }
        
        # Add type-specific substitutions
        if metadata.type == ExperimentType.HPO_TRIAL:
            substitutions['parent_study'] = self._extract_study_name(metadata.parent_run_id)
            substitutions['trial'] = metadata.trial_number or 0
        elif metadata.type == ExperimentType.ABLATION_STUDY:
            substitutions['component'] = self._sanitize_name(metadata.purpose)
        elif metadata.type == ExperimentType.COMPARISON:
            substitutions['models'] = self._sanitize_name(metadata.model_type)
        elif metadata.type == ExperimentType.ANALYSIS:
            substitutions['focus'] = self._sanitize_name(metadata.purpose)
        
        try:
            experiment_name = pattern.format(**substitutions)
        except KeyError as e:
            logger.warning(f"Missing substitution variable: {e}")
            # Fallback to basic naming
            experiment_name = f"{metadata.stage.value}_{metadata.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self._sanitize_name(experiment_name)
    
    def generate_run_name(self, metadata: ExperimentMetadata, 
                         additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate structured run name within an experiment.
        
        Args:
            metadata: Experiment metadata
            additional_info: Additional information for run naming
            
        Returns:
            Generated run name
        """
        base_name = f"{metadata.model_type}_{metadata.purpose}"
        
        # Add type-specific information
        if metadata.type == ExperimentType.HPO_TRIAL:
            base_name = f"trial_{metadata.trial_number:03d}_{metadata.model_type}"
        elif metadata.type == ExperimentType.SINGLE_TRAINING:
            timestamp = datetime.now().strftime("%H%M%S")
            base_name = f"{metadata.model_type}_{timestamp}"
        
        # Add additional info if provided
        if additional_info:
            info_parts = []
            for key, value in additional_info.items():
                if isinstance(value, (int, float)):
                    info_parts.append(f"{key}_{value}")
                elif isinstance(value, str) and len(value) <= 10:
                    info_parts.append(f"{key}_{self._sanitize_name(value)}")
            
            if info_parts:
                base_name += "_" + "_".join(info_parts)
        
        return self._sanitize_name(base_name)
    
    def create_comprehensive_tags(self, metadata: ExperimentMetadata,
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Create comprehensive tags for experiment organization.
        
        Args:
            metadata: Experiment metadata
            config: Optional configuration dictionary
            
        Returns:
            Dictionary of tags
        """
        tags = {
            # Core experiment metadata
            'stage': metadata.stage.value,
            'type': metadata.type.value,
            'model_type': metadata.model_type,
            'dataset_version': metadata.dataset_version,
            'purpose': metadata.purpose,
            'created_date': datetime.now().strftime("%Y-%m-%d"),
            'created_time': datetime.now().strftime("%H:%M:%S"),
            'framework': 'pytorch',
            'task': 'binary_classification',
            
            # Organization tags
            'project': 'trading_bot_ai',
            'phase': 'neural_network_training',
            'status': 'running'
        }
        
        # Add researcher information
        if metadata.researcher:
            tags['researcher'] = metadata.researcher
        
        # Add parent/child relationship tags
        if metadata.parent_run_id:
            tags['parent_run_id'] = metadata.parent_run_id
            tags['is_child_run'] = 'true'
        else:
            tags['is_child_run'] = 'false'
        
        # Add HPO-specific tags
        if metadata.type in [ExperimentType.HPO_STUDY, ExperimentType.HPO_TRIAL]:
            tags['hpo_enabled'] = 'true'
            if metadata.study_name:
                tags['study_name'] = metadata.study_name
            if metadata.trial_number is not None:
                tags['trial_number'] = str(metadata.trial_number)
        else:
            tags['hpo_enabled'] = 'false'
        
        # Add configuration-derived tags
        if config:
            self._add_config_tags(tags, config)
        
        # Add custom tags
        if metadata.custom_tags:
            tags.update(metadata.custom_tags)
        
        return tags
    
    def _add_config_tags(self, tags: Dict[str, str], config: Dict[str, Any]) -> None:
        """Add tags derived from configuration."""
        # Training configuration tags
        training_config = config.get('training', {})
        if 'optimizer' in training_config:
            tags['optimizer'] = str(training_config['optimizer'])
        if 'learning_rate' in training_config:
            tags['learning_rate'] = f"{training_config['learning_rate']:.0e}"
        if 'batch_size' in training_config:
            tags['batch_size'] = str(training_config['batch_size'])
        
        # Data configuration tags
        data_config = config.get('data_config', {})
        if 'lookback_window' in data_config:
            tags['lookback_window'] = str(data_config['lookback_window'])
        if 'symbols' in data_config:
            symbols = data_config['symbols']
            if isinstance(symbols, list):
                tags['num_symbols'] = str(len(symbols))
                if len(symbols) <= 5:  # Only tag if manageable number
                    tags['symbols'] = "_".join(symbols)
        
        # Loss configuration tags
        loss_config = config.get('loss_config', {})
        if 'type' in loss_config:
            tags['loss_type'] = str(loss_config['type'])
        
        # Model configuration tags
        model_config = config.get('model_config', {})
        model_type = config.get('model_type', '')
        if model_type in model_config:
            model_specific = model_config[model_type]
            if 'hidden_size' in model_specific:
                tags['hidden_size'] = str(model_specific['hidden_size'])
            elif 'hidden_sizes' in model_specific:
                sizes = model_specific['hidden_sizes']
                if isinstance(sizes, list):
                    tags['architecture'] = "_".join(map(str, sizes))
    
    def create_hpo_parent_run(self, metadata: ExperimentMetadata,
                            study_config: Dict[str, Any]) -> str:
        """
        Create parent run for HPO study.
        
        Args:
            metadata: Experiment metadata for HPO study
            study_config: HPO study configuration
            
        Returns:
            Parent run ID
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available for HPO parent run creation")
        
        # Generate experiment and run names
        experiment_name = self.generate_experiment_name(metadata)
        run_name = f"HPO_PARENT_{metadata.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Create parent run
        parent_run = mlflow.start_run(run_name=run_name)
        parent_run_id = parent_run.info.run_id
        
        # Create comprehensive tags for parent run
        parent_tags = self.create_comprehensive_tags(metadata)
        parent_tags.update({
            'is_hpo_parent': 'true',
            'hpo_study_name': metadata.study_name or 'default_study',
            'hpo_status': 'running'
        })
        
        mlflow.set_tags(parent_tags)
        
        # Log HPO study configuration
        mlflow.log_params({
            f"hpo_{k}": v for k, v in study_config.items()
            if isinstance(v, (str, int, float, bool))
        })
        
        # Log study metadata
        study_metadata = {
            'study_name': metadata.study_name,
            'study_type': 'optuna_hpo',
            'created_at': datetime.now().isoformat(),
            'parent_run_id': parent_run_id,
            'expected_trials': study_config.get('n_trials', 'unknown')
        }
        
        for key, value in study_metadata.items():
            mlflow.log_param(f"study_{key}", value)
        
        logger.info(f"Created HPO parent run: {run_name} (ID: {parent_run_id})")
        
        return parent_run_id
    
    def create_hpo_child_run(self, parent_run_id: str, trial_number: int,
                           trial_params: Dict[str, Any],
                           base_metadata: ExperimentMetadata) -> Tuple[str, str]:
        """
        Create child run for HPO trial.
        
        Args:
            parent_run_id: Parent run ID
            trial_number: Trial number
            trial_params: Trial parameters
            base_metadata: Base experiment metadata
            
        Returns:
            Tuple of (child_run_id, child_run_name)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available for HPO child run creation")
        
        # Create child metadata
        child_metadata = ExperimentMetadata(
            stage=base_metadata.stage,
            type=ExperimentType.HPO_TRIAL,
            model_type=base_metadata.model_type,
            dataset_version=base_metadata.dataset_version,
            purpose=base_metadata.purpose,
            researcher=base_metadata.researcher,
            parent_run_id=parent_run_id,
            study_name=base_metadata.study_name,
            trial_number=trial_number,
            custom_tags=base_metadata.custom_tags
        )
        
        # Generate child run name
        child_run_name = self.generate_run_name(child_metadata, trial_params)
        
        # Get parent run to extract experiment info
        parent_run = mlflow.get_run(parent_run_id)
        experiment_id = parent_run.info.experiment_id
        
        # Start child run in the same experiment
        child_run = mlflow.start_run(
            run_name=child_run_name,
            experiment_id=experiment_id,
            nested=True
        )
        child_run_id = child_run.info.run_id
        
        # Create comprehensive tags for child run
        child_tags = self.create_comprehensive_tags(child_metadata)
        child_tags.update({
            'is_hpo_child': 'true',
            'parent_run_id': parent_run_id,
            'trial_number': str(trial_number),
            'hpo_trial_status': 'running'
        })
        
        mlflow.set_tags(child_tags)
        
        # Log trial parameters
        mlflow.log_params(trial_params)
        
        # Log trial metadata
        trial_metadata = {
            'trial_number': trial_number,
            'parent_run_id': parent_run_id,
            'trial_start_time': datetime.now().isoformat()
        }
        
        for key, value in trial_metadata.items():
            mlflow.log_param(f"trial_{key}", value)
        
        logger.info(f"Created HPO child run: {child_run_name} (ID: {child_run_id})")
        
        return child_run_id, child_run_name
    
    def finalize_hpo_parent_run(self, parent_run_id: str, 
                              best_trial_id: Optional[str] = None,
                              study_summary: Optional[Dict[str, Any]] = None) -> None:
        """
        Finalize HPO parent run with study results.
        
        Args:
            parent_run_id: Parent run ID
            best_trial_id: ID of the best trial run
            study_summary: Summary of the HPO study
        """
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            # Get parent run
            parent_run = mlflow.get_run(parent_run_id)
            
            # Update parent run with study results
            with mlflow.start_run(run_id=parent_run_id):
                # Update status
                mlflow.set_tag('hpo_status', 'completed')
                mlflow.set_tag('completed_at', datetime.now().isoformat())
                
                # Log best trial information
                if best_trial_id:
                    mlflow.log_param('best_trial_run_id', best_trial_id)
                    mlflow.set_tag('best_trial_run_id', best_trial_id)
                
                # Log study summary
                if study_summary:
                    for key, value in study_summary.items():
                        if isinstance(value, (str, int, float, bool)):
                            mlflow.log_param(f"study_final_{key}", value)
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (str, int, float, bool)):
                                    mlflow.log_param(f"study_final_{key}_{sub_key}", sub_value)
                
                # Log completion timestamp
                mlflow.log_param('study_completed_at', datetime.now().isoformat())
            
            logger.info(f"Finalized HPO parent run: {parent_run_id}")
            
        except Exception as e:
            logger.error(f"Failed to finalize HPO parent run: {e}")
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use in experiment/run names."""
        if not isinstance(name, str):
            name = str(name)
        
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Limit length
        if len(sanitized) > 50:
            sanitized = sanitized[:50].rstrip('_')
        
        return sanitized
    
    def _extract_study_name(self, parent_run_id: Optional[str]) -> str:
        """Extract study name from parent run ID."""
        if not parent_run_id:
            return "unknown_study"
        
        try:
            if MLFLOW_AVAILABLE:
                parent_run = mlflow.get_run(parent_run_id)
                study_name = parent_run.data.tags.get('hpo_study_name', parent_run_id[:8])
                return self._sanitize_name(study_name)
        except Exception:
            pass
        
        return parent_run_id[:8] if parent_run_id else "unknown"
    
    def get_experiment_hierarchy(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment hierarchy showing parent/child relationships.
        
        Args:
            experiment_id: MLflow experiment ID
            
        Returns:
            Dictionary describing experiment hierarchy
        """
        if not MLFLOW_AVAILABLE:
            return {}
        
        try:
            # Get all runs in experiment
            experiment = mlflow.get_experiment(experiment_id)
            runs = mlflow.search_runs(experiment_ids=[experiment_id])
            
            hierarchy = {
                'experiment_name': experiment.name,
                'experiment_id': experiment_id,
                'total_runs': len(runs),
                'parent_runs': [],
                'orphaned_runs': []
            }
            
            # Organize runs by parent/child relationships
            parent_runs = {}
            child_runs = {}
            orphaned_runs = []
            
            for _, run in runs.iterrows():
                run_id = run['run_id']
                tags = run.get('tags', {})
                
                if tags.get('is_hpo_parent') == 'true':
                    parent_runs[run_id] = {
                        'run_id': run_id,
                        'run_name': run.get('run_name', ''),
                        'status': run.get('status', ''),
                        'children': []
                    }
                elif tags.get('is_hpo_child') == 'true':
                    parent_id = tags.get('parent_run_id')
                    if parent_id:
                        if parent_id not in child_runs:
                            child_runs[parent_id] = []
                        child_runs[parent_id].append({
                            'run_id': run_id,
                            'run_name': run.get('run_name', ''),
                            'trial_number': tags.get('trial_number', ''),
                            'status': run.get('status', '')
                        })
                    else:
                        orphaned_runs.append({
                            'run_id': run_id,
                            'run_name': run.get('run_name', ''),
                            'status': run.get('status', '')
                        })
                else:
                    orphaned_runs.append({
                        'run_id': run_id,
                        'run_name': run.get('run_name', ''),
                        'status': run.get('status', '')
                    })
            
            # Attach children to parents
            for parent_id, children in child_runs.items():
                if parent_id in parent_runs:
                    parent_runs[parent_id]['children'] = children
            
            hierarchy['parent_runs'] = list(parent_runs.values())
            hierarchy['orphaned_runs'] = orphaned_runs
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Failed to get experiment hierarchy: {e}")
            return {}


def create_experiment_metadata(stage: str, exp_type: str, model_type: str,
                             dataset_version: str, purpose: str,
                             researcher: Optional[str] = None,
                             parent_run_id: Optional[str] = None,
                             study_name: Optional[str] = None,
                             trial_number: Optional[int] = None,
                             custom_tags: Optional[Dict[str, str]] = None) -> ExperimentMetadata:
    """
    Create experiment metadata object.
    
    Args:
        stage: Experiment stage (development, testing, validation, production, research)
        exp_type: Experiment type (single_training, hpo_study, hpo_trial, etc.)
        model_type: Model type (mlp, lstm, gru, etc.)
        dataset_version: Dataset version identifier
        purpose: Purpose description
        researcher: Optional researcher name
        parent_run_id: Optional parent run ID for child runs
        study_name: Optional study name for HPO
        trial_number: Optional trial number for HPO trials
        custom_tags: Optional custom tags
        
    Returns:
        ExperimentMetadata object
    """
    return ExperimentMetadata(
        stage=ExperimentStage(stage),
        type=ExperimentType(exp_type),
        model_type=model_type,
        dataset_version=dataset_version,
        purpose=purpose,
        researcher=researcher,
        parent_run_id=parent_run_id,
        study_name=study_name,
        trial_number=trial_number,
        custom_tags=custom_tags
    )
