# Output Management Strategy

This document outlines the strategy for managing outputs from training runs, Hyperparameter Optimization (HPO) studies, and other experiments within the project. Its purpose is to provide a clear and predictable structure for all generated artifacts, including models, logs, and results.

## 1. Directory Structure

The following directories are used to store all experiment outputs. All paths are relative to the project root.

-   **`models/`**: This directory is the central repository for all final, trained, and usable models.
    -   **`models/hpo_derived/{study_name}/`**: Contains models that are the result of an HPO study. After an HPO run completes, the best model is automatically retrained and saved here.
        -   `final_model_{model_type}.pth`: The definitive, best model artifact for a given model type from the HPO study. This file contains the model's state dictionary, the configuration used to train it, and the necessary data scalers.
    -   **`models/manual_runs/{run_name}/`**: Contains models generated from manual (non-HPO) training runs.

-   **`hpo_results/`**: This directory stores the detailed results and artifacts from HPO studies.
    -   `hpo_results_{model_type}.json`: A JSON file containing the comprehensive results for a specific model's HPO study, including best parameters, trial values, and parameter importance.
    -   `hpo_combined_results.json`: A summary JSON file that aggregates the results from all model types optimized in a single HPO execution.
    -   `{model_type}_visualizations/`: Contains visualization plots generated during the HPO study, such as parameter importance and optimization history.

-   **`hpo_studies/`**: This directory contains the SQLite database files (`.db`) that store the complete history of every Optuna study. This allows for resuming studies and detailed post-hoc analysis.
    -   `{study_name}.db`: The SQLite database for a specific HPO study.

-   **`mlruns/`**: This is the default output directory for MLflow. It contains all the logged experiments, including metrics, parameters, artifacts, and model files for every run. The structure is managed by MLflow, but it serves as the backend for the MLflow UI.

-   **`training/runs/`**: This directory is the default output location for standard, non-HPO training runs initiated via `train_nn_model.py`. It contains model checkpoints and logs for individual training sessions.

## 2. HPO Workflow and Model Retrieval

The primary goal of the HPO process is to identify the best set of hyperparameters. The workflow ensures that this discovery translates directly into a usable model artifact:

1.  **Optimization**: The `run_hpo.py` script executes the Optuna study, testing numerous hyperparameter combinations to find the set that maximizes the target metric (e.g., `validation_f1_score`).
2.  **Automated Retraining**: Once the study for a given model type is complete, the system automatically triggers a final, full training run using the best-found hyperparameters.
3.  **Final Model Saving**: The model resulting from this final training run is saved as the definitive artifact to:
    ```
    models/hpo_derived/{study_name}/final_model_{model_type}.pth
    ```
4.  **MLflow Logging**: This final model is also logged as an artifact within the MLflow experiment, making it easily accessible and downloadable directly from the MLflow UI.

**To find the best model after an HPO run, navigate to the `models/hpo_derived/{study_name}/` directory.**

## 3. Standard Training Runs

When running a standard training session using `train_nn_model.py`, the outputs are handled as follows:

-   **Checkpoints**: Intermediate model checkpoints are saved to the directory specified in the configuration (defaults to `training/runs/{run_name}`).
-   **Best Model**: The best-performing model from the training session is saved as `best_model.pt` within the same output directory.
-   **Final Model**: Upon completion, a final production-ready model is saved as `final_model_{model_type}.pth` in the output directory.

This structured approach ensures that all experiment outputs are organized, predictable, and easily accessible, streamlining the process of model evaluation and deployment.