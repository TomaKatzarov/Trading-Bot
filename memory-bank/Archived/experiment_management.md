# Advanced Experiment Management and Tracking Guide

## Overview
The Advanced Experiment Management and Tracking system provides robust capabilities for structured experiment management, enhanced traceability, automated reporting, configuration validation, scalable hyperparameter optimization (HPO), and rich visualizations. This system streamlines the machine learning development lifecycle by ensuring consistency, reproducibility, and comprehensive insights into experimental results.

## Core Modules

### `core/experiment_management/config_manager.py`
*   **Main Class**: `ConfigurationManager`
*   **Functionalities**: Handles loading, validation, and management of experiment configurations from YAML files. Supports command-line interface (CLI) overrides, ensures configuration history, and provides schema validation to prevent errors.

### `core/experiment_management/enhanced_logging.py`
*   **Main Class**: `EnhancedMLflowLogger`
*   **Functionalities**: Extends MLflow's logging capabilities to capture comprehensive details of each experiment. This includes logging model architecture summaries, data characteristics (e.g., scaler parameters, feature distributions), environment details, and automatically generating and logging various plots (e.g., training history, feature importance, prediction vs. actual).

### `core/experiment_management/experiment_organizer.py`
*   **Main Class**: `ExperimentOrganizer`
*   **Functionalities**: Manages the organization of MLflow runs and experiments. It ensures structured naming conventions, applies comprehensive tags for easy filtering and search, and establishes parent-child relationships for HPO studies, making it easy to navigate and compare related runs.

### `core/experiment_management/reporting.py`
*   **Main Class**: `ExperimentReporter`
*   **Functionalities**: Facilitates the automated generation of various reports. It can produce detailed training reports, HPO study summaries, and model comparison reports. Reports are saved in a designated `reports/` directory and can include interactive HTML elements for dynamic analysis.

## Integration with Training Scripts
These modules are seamlessly integrated into the main training and HPO scripts:
*   `training/train_nn_model.py`: Utilizes `ConfigurationManager` for loading training parameters, `ExperimentOrganizer` for setting up the MLflow run, and `EnhancedMLflowLogger` for comprehensive logging of training metrics, model artifacts, and plots.
*   `training/run_hpo.py`: Leverages `ConfigurationManager` for HPO specific configurations, `ExperimentOrganizer` for managing parent-child MLflow runs for each HPO trial, and `EnhancedMLflowLogger` for logging individual trial results.

## MLflow Usage
With these enhancements, users can expect to find the following in MLflow:
*   **Comprehensive Parameters and Metrics**: All configuration parameters and key performance metrics are logged for each run.
*   **Detailed Tags for Organization**: Runs are tagged with relevant information such as experiment type, model version, dataset used, and custom identifiers, enabling powerful filtering and grouping.
*   **Logged Artifacts**: Important files like model architecture summaries, scaler objects, data information, and automatically generated plots (e.g., loss curves, prediction distributions) are logged as artifacts.
*   **Parent/Child Run Structures for HPO Studies**: HPO studies are organized with a parent run representing the study and child runs for each hyperparameter combination, simplifying analysis of HPO results.
*   **Code Version and Environment Details**: The Git commit hash of the code used for the run and details about the execution environment are automatically logged, ensuring full reproducibility.

## Configuration Management (`ConfigurationManager`)
Configurations are defined using YAML files, providing a human-readable and structured way to manage experiment settings.
Example:
```yaml
# training/config_templates/hpo_example.yaml
experiment:
  name: "HPO_Study_NN_Model"
  tags:
    project: "TradingBot"
    model_type: "NeuralNetwork"
    task: "HyperparameterOptimization"
model:
  input_dim: 10
  hidden_layers: [64, 32]
  output_dim: 1
  activation: "relu"
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: "Adam"
hpo:
  param_space:
    learning_rate: [0.0001, 0.001, 0.01]
    batch_size: [16, 32, 64]
  num_trials: 10
```
CLI override capabilities allow for quick adjustments without modifying the YAML file directly (e.g., `python train_nn_model.py --model.learning_rate 0.005`). Refer to example configurations like `training/config_templates/hpo_example.yaml` for detailed structure.

## Reporting (`ExperimentReporter`)
The `ExperimentReporter` class simplifies the generation of various reports.
```python
from core.experiment_management.reporting import ExperimentReporter

# Initialize the reporter
reporter = ExperimentReporter(mlflow_tracking_uri="mlruns")

# Generate a training report for a specific run ID
reporter.generate_training_report(
    run_id="your_training_run_id",
    output_path="reports/training_summary.html"
)

# Generate an HPO study report for a parent run ID
reporter.generate_hpo_report(
    parent_run_id="your_hpo_parent_run_id",
    output_path="reports/hpo_study_summary.html"
)

# Generate a model comparison report for multiple run IDs
reporter.generate_model_comparison_report(
    run_ids=["run_id_1", "run_id_2", "run_id_3"],
    output_path="reports/model_comparison.html"
)
```
Reports are saved in the `reports/` directory and can include interactive HTML for dynamic exploration of results.

## Usage Examples

### Basic Training with Enhanced Experiment Management
```python
import torch
import torch.nn as nn
import torch.optim as optim
from core.experiment_management.config_manager import ConfigurationManager
from core.experiment_management.experiment_organizer import ExperimentOrganizer
from core.experiment_management.enhanced_logging import EnhancedMLflowLogger
from training.train_nn_model import train_model # Assuming train_model is in train_nn_model.py

def main():
    config_path = "training/config_templates/basic_training_example.yaml" # Example config path
    config_manager = ConfigurationManager(config_path)
    config = config_manager.get_config()

    organizer = ExperimentOrganizer(config)
    logger = EnhancedMLflowLogger()

    with organizer.start_run(run_name="Basic_NN_Training") as run:
        logger.log_config(config)
        logger.log_environment_details()

        # Dummy data for demonstration
        X_train = torch.randn(100, config.model.input_dim)
        y_train = torch.randn(100, config.model.output_dim)
        X_val = torch.randn(20, config.model.input_dim)
        y_val = torch.randn(20, config.model.output_dim)

        # Define a simple model
        model = nn.Sequential(
            nn.Linear(config.model.input_dim, config.model.hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(config.model.hidden_layers[0], config.model.output_dim)
        )
        optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
        criterion = nn.MSELoss()

        # Train the model (simplified call)
        history = train_model(model, X_train, y_train, X_val, y_val, optimizer, criterion, config.training.epochs, logger)

        logger.log_model_summary(model, "model_architecture.txt")
        logger.log_training_history(history)
        logger.log_metrics({"final_loss": history['val_loss'][-1]})

if __name__ == "__main__":
    main()
```

### HPO with Enhanced Experiment Management
```python
import torch
import torch.nn as nn
import torch.optim as optim
from core.experiment_management.config_manager import ConfigurationManager
from core.experiment_management.experiment_organizer import ExperimentOrganizer
from core.experiment_management.enhanced_logging import EnhancedMLflowLogger
from training.run_hpo import run_hpo_study # Assuming run_hpo_study is in run_hpo.py

def main():
    config_path = "training/config_templates/hpo_example.yaml" # Example HPO config path
    config_manager = ConfigurationManager(config_path)
    config = config_manager.get_config()

    organizer = ExperimentOrganizer(config)
    logger = EnhancedMLflowLogger()

    # Dummy data for demonstration
    X_train = torch.randn(100, config.model.input_dim)
    y_train = torch.randn(100, config.model.output_dim)
    X_val = torch.randn(20, config.model.input_dim)
    y_val = torch.randn(20, config.model.output_dim)

    # Run HPO study
    run_hpo_study(config, X_train, y_train, X_val, y_val, organizer, logger)

if __name__ == "__main__":
    main()
```

### Generating Reports
```python
from core.experiment_management.reporting import ExperimentReporter

def main():
    reporter = ExperimentReporter(mlflow_tracking_uri="mlruns")

    # Example: Generate a training report
    # Replace 'your_training_run_id' with an actual MLflow run ID
    try:
        reporter.generate_training_report(
            run_id="your_training_run_id",
            output_path="reports/training_report_example.html"
        )
        print("Training report generated successfully.")
    except Exception as e:
        print(f"Could not generate training report: {e}")

    # Example: Generate an HPO report
    # Replace 'your_hpo_parent_run_id' with an actual MLflow parent run ID for an HPO study
    try:
        reporter.generate_hpo_report(
            parent_run_id="your_hpo_parent_run_id",
            output_path="reports/hpo_report_example.html"
        )
        print("HPO report generated successfully.")
    except Exception as e:
        print(f"Could not generate HPO report: {e}")

    # Example: Generate a model comparison report
    # Replace with actual MLflow run IDs you want to compare
    try:
        reporter.generate_model_comparison_report(
            run_ids=["run_id_1", "run_id_2", "run_id_3"],
            output_path="reports/model_comparison_example.html"
        )
        print("Model comparison report generated successfully.")
    except Exception as e:
        print(f"Could not generate model comparison report: {e}")

if __name__ == "__main__":
    main()