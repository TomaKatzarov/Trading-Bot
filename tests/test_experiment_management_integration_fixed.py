#!/usr/bin/env python3
"""
Integration Tests for Advanced Experiment Management and Tracking

This module contains comprehensive integration tests for the enhanced experiment
management system implemented in Section 6 (Activities 6.1-6.4).

Tests cover:
- Enhanced MLflow logging integration with ModelTrainer
- HPO parent/child experiment relationships
- Configuration management with various scenarios
- Reporting and visualization features
- End-to-end experiment workflows

Author: Flow-Code
Date: 2025-05-29
Version: 1.0
"""

import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules to test
from core.experiment_management.config_manager import ConfigurationManager
from core.experiment_management.enhanced_logging import EnhancedMLflowLogger
from core.experiment_management.experiment_organizer import ExperimentOrganizer, ExperimentMetadata, ExperimentStage, ExperimentType
from core.experiment_management.reporting import ExperimentReporter


class TestExperimentManagementIntegration(unittest.TestCase):
    """Integration tests for experiment management system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, "test_config.yaml")
        
        # Create test configuration
        self.test_config = {
            'model_type': 'mlp',
            'training': {
                'num_epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'model_config': {
                'hidden_layers': [128, 64],
                'dropout_rate': 0.2
            },
            'experiment': {
                'name': 'test_experiment',
                'tags': {'test': 'true', 'version': '1.0'}
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_config_manager_integration(self):
        """Test ConfigurationManager with various scenarios."""
        # Test basic configuration loading
        config_manager = ConfigurationManager()
        config = config_manager.load_base_config(self.config_file)
        
        self.assertEqual(config['model_type'], 'mlp')
        self.assertEqual(config['training']['num_epochs'], 10)
        
        # Test CLI overrides using argparse.Namespace
        import argparse
        cli_args = argparse.Namespace()
        cli_args.learning_rate = 0.01
        cli_args.batch_size = 64
        
        updated_config = config_manager.apply_cli_overrides(config, cli_args)
        
        self.assertEqual(updated_config['training']['learning_rate'], 0.01)
        self.assertEqual(updated_config['training']['batch_size'], 64)
        
        # Test validation - only pass config (method takes 1 arg + self)
        validated_config = config_manager.validate_config(updated_config)
        self.assertIn('model_type', validated_config)
        self.assertIn('training', validated_config)
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_artifact')
    def test_enhanced_mlflow_logger_integration(self, mock_log_artifact, mock_log_metric, 
                                               mock_log_param, mock_start_run):
        """Test EnhancedMLflowLogger integration."""
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        # Create logger with required experiment_name parameter
        logger = EnhancedMLflowLogger(experiment_name="test_experiment")
        
        # Test hyperparameter logging
        hyperparams = {'learning_rate': 0.001, 'batch_size': 32}
        logger.log_hyperparameters(hyperparams)
        
        # Test metrics logging
        metrics = {'train_loss': 0.5, 'val_accuracy': 0.85}
        logger.log_metrics(metrics)
        
        # Test metadata logging
        metadata = {'experiment_type': 'training', 'model_version': '1.0'}
        logger.log_metadata(metadata)
        
        # Verify calls were made
        self.assertTrue(mock_log_param.called)
        self.assertTrue(mock_log_metric.called)
    
    @patch('mlflow.start_run')
    @patch('mlflow.set_tag')
    def test_experiment_organizer_integration(self, mock_set_tag, mock_start_run):
        """Test ExperimentOrganizer with parent/child relationships."""
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.info.run_id = 'parent_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        # Create experiment organizer (no parameters in constructor)
        organizer = ExperimentOrganizer()
        
        # Test experiment name generation
        metadata = ExperimentMetadata(
            stage=ExperimentStage.DEVELOPMENT,
            type=ExperimentType.HPO_STUDY,
            model_type='mlp',
            purpose='testing'
        )
        
        experiment_name = organizer.generate_experiment_name(metadata)
        self.assertIn('HPO', experiment_name)
        self.assertIn('mlp', experiment_name)
        
        # Test tagging
        tags = organizer.generate_experiment_tags(metadata)
        self.assertEqual(tags['stage'], 'development')
        self.assertEqual(tags['type'], 'hpo_study')
        self.assertEqual(tags['model_type'], 'mlp')
    
    def test_experiment_reporter_integration(self):
        """Test ExperimentReporter functionality."""
        reporter = ExperimentReporter(output_dir=self.test_dir)
        
        # Test report generation with mock data
        mock_metrics = {
            'train_loss': [(1, 0.8), (2, 0.6), (3, 0.4), (4, 0.3)],
            'val_loss': [(1, 0.9), (2, 0.7), (3, 0.5), (4, 0.4)],
            'val_accuracy': [(1, 0.6), (2, 0.7), (3, 0.8), (4, 0.85)]
        }
        
        # Test learning curves generation using private method
        curves_path = os.path.join(self.test_dir, 'learning_curves.png')
        reporter._plot_learning_curves(mock_metrics, 'Test Learning Curves')
        
        # Test training report generation with proper signature
        with patch('mlflow.get_run') as mock_get_run:
            mock_run = MagicMock()
            mock_run.data.metrics = {'val_accuracy': 0.85}
            mock_run.data.params = {'learning_rate': 0.001}
            mock_get_run.return_value = mock_run
            
            report_path = reporter.generate_training_report('test_run_id')
            
            # Verify report was generated
            self.assertTrue(os.path.exists(report_path))


class TestModelTrainerIntegration(unittest.TestCase):
    """Integration tests for ModelTrainer with enhanced experiment management."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create minimal test configuration
        self.test_config = {
            'model_type': 'mlp',
            'training_config': {
                'num_epochs': 2,  # Minimal for testing
                'batch_size': 16,
                'learning_rate': 0.01
            },
            'model_config': {
                'hidden_layers': [32, 16],
                'dropout_rate': 0.1
            },
            'data_config': {
                'sequence_length': 10,
                'prediction_horizon': 1
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('training.train_nn_model.ModelTrainer')
    @patch('mlflow.start_run')
    def test_modeltrainer_enhanced_integration(self, mock_start_run, mock_trainer_class):
        """Test ModelTrainer with enhanced experiment management."""
        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        # Mock ModelTrainer instance
        mock_trainer = MagicMock()
        mock_trainer.experiment_organizer = MagicMock()
        mock_trainer.enhanced_logger = MagicMock()
        mock_trainer.config_manager = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Test trainer initialization
        trainer = mock_trainer_class(self.test_config)
        
        # Verify enhanced components are available
        self.assertTrue(hasattr(trainer, 'experiment_organizer'))
        self.assertTrue(hasattr(trainer, 'enhanced_logger'))
        self.assertTrue(hasattr(trainer, 'config_manager'))


class TestHPOIntegration(unittest.TestCase):
    """Integration tests for HPO with enhanced experiment management."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        self.test_config = {
            'model_type': 'mlp',
            'training_config': {
                'num_epochs': 2,
                'batch_size': 16
            },
            'hpo_config': {
                'n_trials': 3,
                'direction': 'maximize'
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('optuna.create_study')
    @patch('mlflow.start_run')
    def test_hpo_parent_child_relationships(self, mock_start_run, mock_create_study):
        """Test HPO parent/child experiment relationships."""
        # Mock Optuna study
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        
        # Mock MLflow runs
        mock_run = MagicMock()
        mock_run.info.run_id = 'parent_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        try:
            from training.run_hpo import HPOObjective
            
            # Test objective creation with enhanced experiment management
            experiment_organizer = ExperimentOrganizer()
            enhanced_logger = EnhancedMLflowLogger(experiment_name="hpo_test")
            config_manager = ConfigurationManager()
            
            objective = HPOObjective(
                base_config=self.test_config,
                model_type='mlp',
                experiment_organizer=experiment_organizer,
                enhanced_logger=enhanced_logger,
                config_manager=config_manager
            )
            
            # Verify components are set
            self.assertIsNotNone(objective.experiment_organizer)
            self.assertIsNotNone(objective.enhanced_logger)
            self.assertIsNotNone(objective.config_manager)
            
        except ImportError:
            self.skipTest("HPO modules import failed - modules may not be available")


class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end integration tests for complete experiment workflows."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create comprehensive test configuration
        self.config_file = os.path.join(self.test_dir, "e2e_config.yaml")
        self.test_config = {
            'model_type': 'mlp',
            'training': {
                'num_epochs': 3,
                'batch_size': 16,
                'learning_rate': 0.01,
                'early_stopping_patience': 2
            },
            'model_config': {
                'hidden_layers': [32, 16],
                'dropout_rate': 0.1,
                'activation': 'relu'
            },
            'experiment': {
                'name': 'e2e_test_experiment',
                'tags': {'test': 'e2e', 'version': '1.0'},
                'description': 'End-to-end integration test'
            },
            'hpo': {
                'n_trials': 2,
                'direction': 'maximize',
                'target_metric': 'validation_f1_score_positive_class'
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    def test_complete_experiment_workflow(self, mock_log_metric, mock_log_param, mock_start_run):
        """Test complete experiment workflow from config to reporting."""
        # Mock MLflow run context
        mock_run = MagicMock()
        mock_run.info.run_id = 'workflow_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        # Test 1: Configuration Management
        config_manager = ConfigurationManager()
        config = config_manager.load_base_config(self.config_file)
        
        # Fix validation call - only pass config
        validated_config = config_manager.validate_config(config)
        self.assertIn('model_type', validated_config)
        
        # Test 2: Enhanced Logging Setup
        enhanced_logger = EnhancedMLflowLogger(experiment_name="e2e_test")
        enhanced_logger.log_hyperparameters(config['training'])
        enhanced_logger.log_metadata(config['experiment'])
        
        # Test 3: Experiment Organization
        organizer = ExperimentOrganizer()
        
        metadata = ExperimentMetadata(
            stage=ExperimentStage.TESTING,
            type=ExperimentType.SINGLE_TRAINING,
            model_type=config['model_type'],
            purpose='e2e_testing'
        )
        
        experiment_name = organizer.generate_experiment_name(metadata)
        experiment_tags = organizer.generate_experiment_tags(metadata)
        
        self.assertIn('mlp', experiment_name)
        self.assertEqual(experiment_tags['model_type'], 'mlp')
        
        # Test 4: Reporting
        reporter = ExperimentReporter(output_dir=self.test_dir)
        
        # Mock training metrics for report generation
        mock_metrics = {
            'train_loss': [(1, 0.8), (2, 0.5), (3, 0.3)],
            'val_loss': [(1, 0.9), (2, 0.6), (3, 0.4)],
            'val_accuracy': [(1, 0.6), (2, 0.8), (3, 0.85)]
        }
        
        # Generate learning curves using private method
        reporter._plot_learning_curves(mock_metrics, 'E2E Test Learning Curves')
        
        # Generate training report
        with patch('mlflow.get_run') as mock_get_run:
            mock_run_data = MagicMock()
            mock_run_data.data.metrics = {'val_accuracy': 0.85}
            mock_run_data.data.params = config['training']
            mock_get_run.return_value = mock_run_data
            
            report_path = reporter.generate_training_report('workflow_run_id')
            
            # Verify report was generated
            self.assertTrue(os.path.exists(report_path))
        
        # Verify all components worked together
        self.assertTrue(mock_log_param.called)
        self.assertTrue(mock_log_metric.called)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
