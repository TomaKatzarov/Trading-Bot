# Operational Plan: 1.4 Train Custom NN Models & Tune Hyperparameters

**Document Version:** 1.0
**Date:** 2025-05-28
**Author:** Roo (AI Architect)
**Based on Project Task ID:** 1.4 (from [`memory-bank/progress.md`](memory-bank/progress.md:60))

## 0. Introduction

This document outlines the detailed operational plan for Task 1.4: "Train Custom NN Models & Tune Hyperparameters" as part of Phase 1 (New Strategy): "Foundation & Custom NN (Supervised Baseline)" of the project. The primary goal of this task is to develop, train, tune, and evaluate custom Neural Network (NN) models to serve as a robust supervised baseline for the subsequent Reinforcement Learning (RL) phase.

This plan leverages insights from various project documents, including:
*   [`memory-bank/progress.md`](memory-bank/progress.md:1) for task breakdown.
*   [`memory-bank/strategic_plan_nn_rl.md`](memory-bank/strategic_plan_nn_rl.md:1) for high-level strategy.
*   [`memory-bank/feature_set_NN.md`](memory-bank/feature_set_NN.md:1) for data and feature specifications.
*   [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:1) for lessons learned and improvement strategies.
*   [`memory-bank/Literature Review on Neural Network Architectures.md`](memory-bank/Literature%20Review%20on%20Neural%20Network%20Architectures.md:1) for architectural guidance.
*   Other context documents like [`memory-bank/asset_id_embedding_strategy.md`](memory-bank/asset_id_embedding_strategy.md:1), [`memory-bank/systemPatterns.md`](memory-bank/systemPatterns.md:1), and [`memory-bank/techContext.md`](memory-bank/techContext.md:1).

## 1. Prerequisites and Data Finalization

This section details the necessary checks and finalization steps for the data that will be used to train the NN models. It assumes the completion of Task 1.2, which developed the [`core/data_preparation_nn.py`](core/data_preparation_nn.py:1) module.

**Sub-Tasks:**
*   **1.4.2 Data Preparation & Preprocessing Confirmation** (from [`memory-bank/progress.md`](memory-bank/progress.md:62))
*   **1.4.4.A Aggregate Data for Combined Training** (from [`memory-bank/progress.md`](memory-bank/progress.md:64))

**Activities:**

1.  **Verify `core/data_preparation_nn.py` Outputs:**
    *   Confirm that the [`core/data_preparation_nn.py`](core/data_preparation_nn.py:1) script successfully generates data in the expected format (NumPy arrays for features `X`, labels `y`, and `asset_ids`).
    *   Validate output shapes, data types (e.g., `float32` for features, `int32` for labels/IDs), and sequence lengths based on the lookback window (24-48 hours, starting with 24 as per [`memory-bank/feature_set_NN.md`](memory-bank/feature_set_NN.md:9)).
    *   Ensure generated labels correctly reflect the +5% profit / -2% stop-loss target within an 8-hour horizon, as specified in [`memory-bank/feature_set_NN.md`](memory-bank/feature_set_NN.md:11) and implemented in [`core/data_preparation_nn.py`](core/data_preparation_nn.py:288).
    *   Confirm that per-symbol `.npz` files (or equivalent aggregated format) are correctly generated and loadable.

2.  **Finalize Data Splits:**
    *   Review and confirm the train/validation/test splitting strategy implemented in [`core/data_preparation_nn.py`](core/data_preparation_nn.py:307) (default 70/15/15 temporal split).
    *   Ensure strict chronological order is maintained to prevent data leakage, especially for time-series data.
    *   Verify stratification by symbol if combined training is used, to ensure representation of all assets in all sets.
    *   Document the exact datasets (e.g., date ranges, symbols) used for each split for reproducibility.

3.  **Implement and Validate Data Augmentation Strategies:**
    *   Based on [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:30):
        *   **Oversampling:** Implement minority class (BUY_SIGNAL) oversampling using techniques like `WeightedRandomSampler` in PyTorch DataLoaders or explicit duplication.
        *   **Noise Injection (Jittering):** For positive samples, add small Gaussian noise to technical indicator values. Define noise level (e.g., 0.1-0.5% of feature standard deviation).
        *   **Time Shifting:** Slightly shift input windows (e.g., by 1-2 hours) for positive samples, ensuring the label remains valid.
    *   Carefully validate that augmentation techniques do not introduce unrealistic artifacts or label leakage.
    *   Integrate augmentation into the training data pipeline.

4.  **Confirm Normalization/Standardization Procedures:**
    *   Verify that `StandardScaler` (for general features) and `RobustScaler` (for skewed features like volume, if applicable) are used as per [`memory-bank/feature_set_NN.md`](memory-bank/feature_set_NN.md:74) and [`memory-bank/strategic_plan_nn_rl.md`](memory-bank/strategic_plan_nn_rl.md:54).
    *   Ensure scalers are fitted **only** on the training data and then applied to validation and test sets.
    *   Confirm that scalers are saved by [`core/data_preparation_nn.py`](core/data_preparation_nn.py:311) (e.g., `scalers.joblib`) for consistent application during inference/backtesting.

5.  **Verify Input/Output Formatting for NN Models:**
    *   **Input:**
        *   Features (`X`): `(n_samples, lookback_window, n_features)`
        *   Asset IDs (`asset_ids`): `(n_samples,)` for embedding lookup, as per [`memory-bank/asset_id_embedding_strategy.md`](memory-bank/asset_id_embedding_strategy.md:100).
    *   **Output (Labels `y`):**
        *   Binary classification: `(n_samples,)` with values {0, 1} for NO_BUY_SIGNAL / BUY_SIGNAL.
    *   Ensure data loaders provide batches in these formats to the models.

6.  **Data Aggregation for Combined Training:**
    *   Confirm that data from multiple symbols is correctly aggregated by [`core/data_preparation_nn.py`](core/data_preparation_nn.py:290), including the `asset_id` feature.
    *   Ensure the `asset_id_map` from `config/asset_id_mapping.json` is correctly used and passed along with the data.

## 2. Custom NN Model Architecture Specifications

This section defines the NN architectures to be implemented and evaluated, drawing from project documentation.

**Sub-Tasks:**
*   **1.4.1 Finalize NN Architecture & Define Supervised Learning Task** (from [`memory-bank/progress.md`](memory-bank/progress.md:61))
*   **1.4.4.H Implement Attention Mechanism & Evaluate GRU** (from [`memory-bank/progress.md`](memory-bank/progress.md:71)) (Partially, architecture definition)

**Activities:**

1.  **Define Supervised Learning Task:**
    *   **Task Type:** Binary Classification.
    *   **Target Variable:** `BUY_SIGNAL` (1) vs. `NO_BUY_SIGNAL` (0).
    *   **Labeling Criteria:** +5% profit before -2% stop-loss within an 8-hour future window, as detailed in [`memory-bank/feature_set_NN.md`](memory-bank/feature_set_NN.md:73).

2.  **MLP (Multi-Layer Perceptron) - Baseline:**
    *   **Rationale:** Simple, fast to train, serves as a crucial performance benchmark (as suggested in [`memory-bank/Literature Review on Neural Network Architectures.md`](memory-bank/Literature%20Review%20on%20Neural%20Network%20Architectures.md:44) and [`memory-bank/strategic_plan_nn_rl.md`](memory-bank/strategic_plan_nn_rl.md:25)).
    *   **Architecture:**
        *   Input layer: Flattened sequence `(lookback_window * n_features)`. Asset ID embedding can be concatenated after flattening or used to create separate MLPs per asset group if simpler.
        *   Hidden layers: 2-3 fully connected layers (e.g., 128, 64, 32 units) with ReLU activation.
        *   Output layer: Single neuron with Sigmoid activation for binary classification.
        *   Regularization: Dropout layers after hidden layers.
    *   **File:** Implement in `core/models/nn_architectures.py` (or similar).

3.  **LSTM (Long Short-Term Memory) / GRU (Gated Recurrent Unit) - Primary Sequential Models:**
    *   **Rationale:** Proven for financial time series, capable of capturing temporal dependencies (as per [`memory-bank/Literature Review on Neural Network Architectures.md`](memory-bank/Literature%20Review%20on%20Neural%20Network%20Architectures.md:74) and selected in [`memory-bank/strategic_plan_nn_rl.md`](memory-bank/strategic_plan_nn_rl.md:26)). GRU as a lighter alternative.
    *   **Architecture (LSTM/GRU Core):**
        *   Input: Sequence `(batch_size, lookback_window, n_features)`.
        *   Asset ID Embedding: Use `nn.Embedding` layer for `asset_ids`, concatenate embeddings to input features at each time step or as an initial hidden state component (as per [`memory-bank/asset_id_embedding_strategy.md`](memory-bank/asset_id_embedding_strategy.md:117)). Embedding dimension: 8-16 (tunable).
        *   Recurrent Layers: 1-2 layers of LSTM/GRU (e.g., 64-128 units per layer).
        *   Normalization: Layer Normalization within recurrent blocks (as suggested in [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:17)).
        *   Output from RNN: Output of the last time step, or an aggregation (e.g., mean/max pooling) of all time steps.
    *   **Attention Mechanism (Enhancement for LSTM/GRU):**
        *   Implement self-attention or additive attention layer after the LSTM/GRU layers to weigh the importance of different time steps in the sequence (as recommended in [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:15)).
        *   The attention output will then be fed to the final classification layer.
    *   **Final Classification Layer:** Fully connected layer with Sigmoid activation.
    *   **Regularization:** Dropout after recurrent layers and attention layer.
    *   **File:** Implement in `core/models/nn_architectures.py`.

4.  **CNN-LSTM Hybrid (Exploratory, with improvements):**
    *   **Rationale:** Combines CNN's local feature extraction with LSTM's temporal modeling. Previous attempts underperformed (Diagnostic Report), so focus on remediations.
    *   **Architecture:**
        *   Input: Sequence `(batch_size, lookback_window, n_features)`.
        *   Asset ID Embedding: As above.
        *   1D CNN Layers: 1-2 layers of 1D convolutions (e.g., 32-64 filters, kernel size 3-5) with ReLU activation. Apply to input sequence.
        *   Normalization: Batch Normalization after CNN layers (as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:17)).
        *   Optional: Max Pooling layer after CNNs (use cautiously to preserve temporal resolution).
        *   LSTM Layers: Feed output of CNN layers into 1-2 LSTM layers.
        *   Attention: Optionally add attention after LSTM.
        *   Final Classification Layer: Fully connected layer with Sigmoid activation.
    *   **Regularization:** Dropout.
    *   **File:** Implement in `core/models/nn_architectures.py`.

## 3. Development and Training Environment Setup

This section covers the setup of the environment required for model development and training.

**Sub-Tasks:**
*   **1.4.3 NN Model Implementation & Training Infrastructure Setup** (from [`memory-bank/progress.md`](memory-bank/progress.md:63)) (Partially, environment setup)

**Activities:**

1.  **Hardware Allocation:**
    *   Primary GPU: NVIDIA RTX 5070 Ti 16GB (as per [`memory-bank/techContext.md`](memory-bank/techContext.md:93)).
    *   Ensure CUDA drivers and toolkit are up-to-date and compatible with PyTorch version.

2.  **Software Frameworks and Libraries:**
    *   **PyTorch:** Version 2.2 or later (as specified in [`memory-bank/strategic_plan_nn_rl.md`](memory-bank/strategic_plan_nn_rl.md:34)).
    *   **Optuna:** For hyperparameter optimization (already used/planned as per [`memory-bank/progress.md`](memory-bank/progress.md:66)).
    *   **MLflow:** For experiment tracking (already used/planned as per [`memory-bank/progress.md`](memory-bank/progress.md:49)).
    *   **NumPy, Pandas, Scikit-learn:** For data manipulation and evaluation.
    *   **Joblib:** For saving/loading scalers and potentially models.
    *   Update `requirements.txt` with all necessary packages and versions.

3.  **Containerization (Optional but Recommended):**
    *   Consider using Docker to create a reproducible training environment.
    *   Develop a `Dockerfile` specifying OS, Python version, CUDA version, and all dependencies.
    *   This ensures consistency across development and potential future deployment.

4.  **Version Control Practices:**
    *   **Code:** Use Git for all scripts (training, models, evaluation). Maintain a clear branching strategy (e.g., `develop`, `feature/task-1.4`).
    *   **Data:** Use DVC for versioning datasets and scalers (as mentioned in [`memory-bank/strategic_plan_nn_rl.md`](memory-bank/strategic_plan_nn_rl.md:58)).
    *   **Models:** Trained model artifacts (weights, architecture) should be versioned, potentially using MLflow's artifact tracking or Git LFS if small enough.
    *   **Experiments:** All experiment configurations and results tracked via MLflow.

5.  **Training Scripts Structure:**
    *   Develop modular training scripts (e.g., `training/train_nn_model.py`).
    *   Script should be configurable via CLI arguments or a configuration file (e.g., YAML) for model type, hyperparameters, data paths, etc.
    *   Integrate with [`utils/gpu_utils.py`](utils/gpu_utils.py:1) for optimized GPU settings (Flash Attention, TF32) and dynamic batch sizing if applicable (as per [`memory-bank/techContext.md`](memory-bank/techContext.md:75)).

## 4. Detailed Training Regimen and Loop Implementation

This section details the specifics of the model training process.

**Sub-Tasks:**
*   **1.4.3 NN Model Implementation & Training Infrastructure Setup** (from [`memory-bank/progress.md`](memory-bank/progress.md:63)) (Partially, training loop)
*   **1.4.4.I Advanced Regularization & Imbalance Mitigation** (from [`memory-bank/progress.md`](memory-bank/progress.md:76)) (Partially, training aspects)

**Activities:**

1.  **Loss Function Selection:**
    *   **Primary:** Focal Loss to handle class imbalance (as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:31)).
        *   Tunable parameters: `alpha` (weighting factor, e.g., 0.5-0.8) and `gamma` (focusing parameter, e.g., 1.0-2.0).
    *   **Alternative/Comparison:** Weighted Binary Cross-Entropy. Calculate weights based on inverse class frequency from the training set.
    *   Implement flexibility to switch between loss functions.

2.  **Optimizer Selection:**
    *   **Primary:** AdamW (Adam with decoupled weight decay).
    *   **Alternative:** Adam.
    *   Initial learning rate: e.g., 1e-4 to 5e-4 (tunable).

3.  **Key Performance Metrics (During Training & Evaluation):**
    *   Loss (overall, and per class if applicable).
    *   Accuracy.
    *   **Precision, Recall, F1-score (especially for the positive 'BUY_SIGNAL' class).**
    *   Area Under Precision-Recall Curve (PR-AUC).
    *   Area Under ROC Curve (ROC-AUC).
    *   Confusion Matrix.
    *   Log these metrics for both training and validation sets at each epoch.

4.  **Training Parameters:**
    *   **Epochs:** Sufficiently large number (e.g., 50-200), controlled by early stopping. Diagnostic Report suggests allowing more epochs with patience ([`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:22)).
    *   **Batch Size:** Determined by GPU memory (RTX 5070 Ti 16GB). [`utils/gpu_utils.py`](utils/gpu_utils.py:1) can help estimate optimal batch sizes. Start with e.g., 32, 64, 128.
    *   **Learning Rate Schedule:**
        *   Implement `ReduceLROnPlateau` based on validation loss or F1-score (as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:21)).
        *   Consider `CosineAnnealingLR` or `OneCycleLR` as alternatives.

5.  **Regularization Techniques Implementation:**
    *   **Dropout:** Apply strategically in MLP, CNN, and RNN layers (tunable rate, e.g., 0.2-0.5) (as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:24)).
    *   **Weight Decay (L2 Regularization):** Include in the optimizer (AdamW handles this naturally). Tune decay factor (e.g., 1e-5 to 1e-3) (as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:23)).
    *   **Batch Normalization / Layer Normalization:** As specified in model architectures.
    *   **Gradient Clipping:** Implement to prevent exploding gradients, especially for RNNs (e.g., clip norm to 1.0 or 5.0) (as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:27)).

6.  **Early Stopping and Model Checkpointing:**
    *   **Early Stopping:** Monitor a key validation metric (e.g., validation F1-score for positive class, or validation loss). Stop training if the metric doesn't improve for a defined `patience` number of epochs (e.g., 10-15 epochs, as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:22)).
    *   **Model Checkpointing:** Save the model weights (and optimizer state) that achieve the best performance on the monitored validation metric. Also, save periodic checkpoints.

7.  **Sample Weights for Class Imbalance:**
    *   If not fully addressed by Focal Loss or oversampling, use `sample_weights` in the training loop, calculated by [`core/data_preparation_nn.py`](core/data_preparation_nn.py:309).

8.  **Curriculum Learning (Exploratory):**
    *   Based on [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:36):
        *   Phase 1: Train on an easier task (e.g., smaller profit target like +2%, more frequent).
        *   Phase 2: Fine-tune on the actual +5% target using weights from Phase 1.
    *   This is an advanced technique; implement if initial models struggle significantly.

## 5. Robust Hyperparameter Optimization (HPO) Strategy

This section outlines the strategy for systematically tuning hyperparameters to find optimal model configurations.

The HPO framework has been successfully implemented using **Optuna**.
Key scripts include the main HPO orchestrator [`training/run_hpo.py`](training/run_hpo.py) and a user-friendly quick start script [`training/run_hpo_quick_start.py`](training/run_hpo_quick_start.py).
An example HPO configuration is available at [`training/config_templates/hpo_example.yaml`](training/config_templates/hpo_example.yaml).
Comprehensive guidance on using the HPO framework can be found in the [HPO Usage Guide](../../docs/hpo_usage_guide.md).

**Sub-Tasks:**
*   **1.4.4.C Hyperparameter Tuning on Combined Data (Optuna for LSTM)** (from [`memory-bank/progress.md`](memory-bank/progress.md:66))
*   **1.4.4.F Optuna Hyperparameter Study for CNN-LSTM** (from [`memory-bank/progress.md`](memory-bank/progress.md:69))
*   **1.4.4.G Optuna Hyperparameter Study for MLP** (from [`memory-bank/progress.md`](memory-bank/progress.md:70))
*   **1.4.4.J Hyperparameter Re-tuning for Enhanced Model** (from [`memory-bank/progress.md`](memory-bank/progress.md:81))

**Activities:**

1.  **Identify Critical Hyperparameters for Tuning:** [COMPLETED]
    *   **Common:** Learning rate, batch size, optimizer parameters (e.g., Adam betas).
    *   **Architecture-specific:**
        *   MLP: Number of layers, units per layer, dropout rates.
        *   LSTM/GRU: Number of layers, hidden units, dropout rates, attention dimension (if applicable).
        *   CNN-LSTM: CNN filter counts, kernel sizes, strides, pooling strategy, LSTM parameters.
    *   **Regularization:** Weight decay factor, dropout rates.
    *   **Loss Function:** Focal Loss `alpha` and `gamma`.
    *   **Data Augmentation:** Noise levels, oversampling ratios (if made tunable).

2.  **Chosen HPO Methodology:** [COMPLETED]
    *   **Primary:** Bayesian Optimization using Optuna (as it's already familiar to the project). Optuna's TPE sampler is suitable.
    *   **Consider for extensive searches:** Population-Based Training (PBT) if computational resources allow and simpler methods plateau (as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:38)).

3.  **HPO Tool Setup:** [COMPLETED]
    *   Integrate Optuna with the training scripts.
    *   Set up an Optuna study, specifying the direction of optimization (e.g., maximize validation F1-score for positive class or PR-AUC).
    *   Use Optuna's pruning callbacks for early stopping of unpromising trials.

4.  **Define Search Spaces/Ranges for Each Hyperparameter:** [COMPLETED]
    *   Example Ranges:
        *   Learning Rate: Log-uniform (1e-5 to 1e-2).
        *   Batch Size: Categorical (32, 64, 128, 256).
        *   LSTM/GRU Units: Integer (32 to 256, step 32).
        *   Dropout Rate: Uniform (0.1 to 0.5).
        *   Focal Loss Alpha: Uniform (0.25 to 0.9).
        *   Focal Loss Gamma: Uniform (0.5 to 3.0).
    *   Define these carefully based on literature and initial manual experiments.

5.  **Evaluation Protocol for HPO Runs:** [COMPLETED]
    *   **Metric:** Primary metric for Optuna to optimize (e.g., validation set F1-score for 'BUY_SIGNAL' class, or PR-AUC).
    *   **Trials:** Run a significant number of trials (e.g., 50-200 per model architecture, as suggested in [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:38)).
    *   **Cross-Validation (Optional but Recommended for Robustness):**
        *   If computationally feasible, use k-fold time-series cross-validation within each HPO trial on the combined training/validation data. Average the validation metric across folds.
        *   Alternatively, use a single, fixed validation set derived from the temporal split.
    *   Log all trial parameters and results using MLflow.

## 6. Comprehensive Experiment Tracking and Management [COMPLETED]

Ensuring reproducibility and traceability of all experiments is crucial.

The Advanced Experiment Management system has been successfully implemented, providing robust capabilities for configuration, logging, organization, and reporting.
Core components are located in the `core/experiment_management/` directory (`config_manager.py`, `enhanced_logging.py`, `experiment_organizer.py`, `reporting.py`).
These components are integrated into `training/train_nn_model.py` and `training/run_hpo.py`.
Comprehensive guidance on this system can be found in the [Advanced Experiment Management and Tracking Guide](../../docs/experiment_management.md).

**Activities:**

1.  **Experiment Tracking Tool:** [COMPLETED]
    *   **MLflow:** Standardize on MLflow for tracking all training experiments (already in use for backtesting).
    *   Set up an MLflow tracking server if not already in place (can be local).

2.  **Logged Items for Each Experiment Run:** [COMPLETED]
    *   **Parameters:**
        *   All hyperparameters (model-specific, optimizer, loss function, regularization).
        *   Data version/source (DVC hash).
        *   Data preparation config (lookback window, feature list, augmentation settings).
        *   Random seeds used.
    *   **Metrics:**
        *   All metrics listed in Section 4.3, for training and validation sets, logged per epoch.
        *   Final metrics on the validation set for the best epoch.
    *   **Model Artifacts:**
        *   Saved model weights (best checkpoint).
        *   Model architecture definition (e.g., `model.summary()` or script).
        *   Saved scalers (if not globally managed by DVC for this run).
        *   Optuna study object (if applicable).
    *   **Visualizations:**
        *   Learning curves (loss, metrics vs. epoch).
        *   Precision-Recall curves, ROC curves for validation set.
        *   Confusion matrices.
    *   **Environment Metadata:**
        *   Git commit hash of the code.
        *   Python environment (`requirements.txt` or Conda env file).
        *   Hardware used (GPU type).
        *   Execution logs (stdout/stderr).

## 7. Rigorous Model Evaluation and Selection Protocol

This section defines how trained models will be evaluated and the best one(s) selected.

**Sub-Tasks:**
*   **1.4.4.D Select Best HPs & Final Model (Focal Loss LSTM)** (from [`memory-bank/progress.md`](memory-bank/progress.md:67))
*   **1.4.4.E Re-train Best Model & Fix Saving Logic** (from [`memory-bank/progress.md`](memory-bank/progress.md:68))
*   **1.4.5 Model Evaluation & Baseline Performance Documentation** (from [`memory-bank/progress.md`](memory-bank/progress.md:85))

**Activities:**

1.  **Validation Set Evaluation:**
    *   After HPO, select the top N hyperparameter configurations for each model architecture.
    *   Re-train models with these configurations multiple times (e.g., 3-5 runs with different random seeds) to assess stability and average performance on the validation set.
    *   Metrics: Focus on F1-score (positive class), Precision (positive class), PR-AUC, and profit-based metrics from a simulated backtest on the validation set (as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:42)).

2.  **Classification Threshold Optimization:**
    *   For the selected model(s), determine the optimal classification threshold (default 0.5 for sigmoid) on the validation set.
    *   Iterate through thresholds (e.g., 0.1 to 0.9) and select the one that maximizes a chosen metric (e.g., F1-score, or a custom metric balancing precision and recall, or validation set profit).
    *   This is crucial as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:28) and [`memory-bank/progress.md`](memory-bank/progress.md:57) (already done for backtester).

3.  **Error Analysis:**
    *   For the best model(s), perform in-depth error analysis on validation set predictions:
        *   Analyze confusion matrices.
        *   Qualitatively review false positives and false negatives: What patterns is the model missing or misinterpreting? (as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:46)).
        *   Identify if errors are concentrated in specific symbols, time periods, or market conditions.

4.  **Feature Importance Analysis (1.4.5.4):**
    *   For the final selected model, use techniques like SHAP or permutation importance to understand which input features are most influential (as per [`memory-bank/Diagnostic Report and Remediation Plan for LSTMCNN-LSTM Model Underperformance.md`](memory-bank/Diagnostic%20Report%20and%20Remediation%20Plan%20for%20LSTMCNN-LSTM%20Model%20Underperformance.md:46)).
    *   This aids interpretability and can guide future feature engineering.

5.  **Model Selection Criteria:**
    *   Primary: Positive-class F1-score ≥ 0.25 on the validation set (as per [`memory-bank/progress.md`](memory-bank/progress.md:86)).
    *   Secondary:
        *   Positive-class Precision (e.g., > 0.25-0.3).
        *   PR-AUC.
        *   Demonstrable positive return in a simulated backtest on the validation set using the [`core/strategies/supervised_nn_strategy.py`](core/strategies/supervised_nn_strategy.py:1) and the backtesting engine (Task 1.3).
        *   Model stability (consistent performance across multiple runs).
        *   Model complexity and inference speed.
    *   Select the model architecture and hyperparameter configuration that best meets these criteria.

6.  **Final Test Set Evaluation (1.4.5.1):**
    *   Once a final model is selected and its threshold optimized, evaluate it **once** on the hold-out test set.
    *   Report all relevant metrics. This provides an unbiased estimate of generalization performance.

7.  **Backtesting Integration and Evaluation (1.4.5.2, 1.4.5.3):**
    *   Integrate the final selected model and its scaler with the [`core/strategies/supervised_nn_strategy.py`](core/strategies/supervised_nn_strategy.py:1).
    *   Run comprehensive backtests using the engine developed in Task 1.3 on the test set period for multiple symbols.
    *   Evaluate using metrics from [`memory-bank/progress.md`](memory-bank/progress.md:54) (Sharpe, Sortino, Calmar, Max Drawdown, Profit Factor, etc.).

## 8. Resource Allocation and Timeline Estimates

**Sub-Tasks:** This section covers the overall timeline for 1.4.

**Resources:**
*   **Human Effort:**
    *   1 Lead AI/ML Engineer.
    *   Estimate: 6-10 person-weeks for the entire Task 1.4, considering its iterative nature and HPO.
*   **Computational Resources:**
    *   Dedicated use of RTX 5070 Ti 16GB GPU for training and HPO.
    *   CPU for data preprocessing and some HPO orchestration.
    *   Storage for datasets, models, and experiment logs.

**Projected Timeline (assuming prerequisites are met):**

*   **Week 1-2: Initial Setup & Baseline Models (MLP, Basic LSTM/GRU)**
    *   1.4.1 Finalize NN Architecture & Define Supervised Learning Task
    *   1.4.2 Data Preparation & Preprocessing Confirmation
    *   1.4.3 NN Model Implementation & Training Infrastructure Setup (for MLP, basic LSTM/GRU)
    *   1.4.4.A Aggregate Data for Combined Training
    *   1.4.4.B Initial Runs on Combined Data (MLP, basic LSTM/GRU)
    *   1.4.4.G Optuna Hyperparameter Study for MLP (initial run)
*   **Week 3-5: Advanced Sequential Models & Initial HPO**
    *   1.4.4.H Implement Attention Mechanism & Evaluate GRU (Architecture implementation)
    *   1.4.4.I Advanced Regularization & Imbalance Mitigation (Implementation in training loop)
    *   Training and initial HPO (1.4.4.C) for LSTM/GRU with Attention.
*   **Week 6-8: CNN-LSTM (Exploratory) & Comprehensive HPO**
    *   Implementation and training of CNN-LSTM (if prioritized).
    *   1.4.4.F Optuna Hyperparameter Study for CNN-LSTM.
    *   1.4.4.J Hyperparameter Re-tuning for Enhanced Models (LSTM/GRU+Attention, potentially CNN-LSTM).
*   **Week 9-10: Model Evaluation, Selection, and Documentation**
    *   1.4.4.D Select Best HPs & Final Model
    *   1.4.4.E Re-train Best Model & Fix Saving Logic
    *   1.4.5 Model Evaluation & Baseline Performance Documentation (Validation, Error Analysis, Feature Importance, Test Set Eval, Backtesting)
    *   1.4.6 Final Documentation & Memory Bank Update

**Total Estimated Timeline for Task 1.4: 10 weeks.** (This aligns with the overall Phase 1 timeline of Apr-Sep 2025, where 1.4 is a major component).

## 9. Documentation Requirements and Key Deliverables

**Sub-Tasks:**
*   **1.4.5.5 Compile Baseline Performance Report** (from [`memory-bank/progress.md`](memory-bank/progress.md:91))
*   **1.4.6 Final Documentation & Memory Bank Update** (from [`memory-bank/progress.md`](memory-bank/progress.md:92))

**Documentation:**

1.  **Model Architecture Document:** Detailed descriptions of each implemented architecture, including layer configurations, activation functions, and rationale. Store in `docs/model_architectures.md`.
2.  **Training Procedure Document:** Step-by-step guide on how to run training scripts, configure experiments, and interpret results. Store in `docs/training_procedure_nn.md`.
3.  **HPO Results Summary:** Document the HPO process for each model, including search spaces, best hyperparameters found, and performance of different configurations. Store in `docs/hpo_summary_nn.md`.
4.  **Experiment Logs:** All MLflow experiment IDs and key findings should be linked or summarized for easy access.
5.  **Code Documentation:** Comprehensive docstrings and comments in all new/modified Python scripts (`core/models/nn_architectures.py`, `training/train_nn_model.py`, etc.).

**Key Deliverables:**

1.  **Trained Model Artifacts:**
    *   Saved weights (`.pt` or `.pth` files) for the best performing model(s) for each architecture.
    *   Corresponding model architecture definitions/scripts.
    *   Associated scalers (`scalers.joblib`).
    *   Asset ID mapping file (`config/asset_id_mapping.json`) if updated.
2.  **Training and Evaluation Scripts:**
    *   Finalized Python scripts for training, HPO, and evaluation.
3.  **HPO Logs and Reports:**
    *   Optuna study results/databases.
    *   Visualizations of HPO process (e.g., parallel coordinate plots, optimization history).
4.  **Baseline Performance Report (1.4.5.5):**
    *   A comprehensive report summarizing:
        *   Methodology used for model development and training.
        *   Experiments conducted.
        *   Performance metrics (classification and backtesting) for all evaluated models on validation and test sets.
        *   Feature importance analysis results.
        *   Error analysis insights.
        *   The final selected baseline model, its configuration, and its performance.
        *   Lessons learned and recommendations for Phase 2 (RL).
    *   Store as `memory-bank/reports/NN_Phase1_Baseline_Performance_Report.md`.
5.  **Updated Memory Bank Documents (1.4.6):**
    *   Relevant sections of [`memory-bank/progress.md`](memory-bank/progress.md:1), [`memory-bank/activeContext.md`](memory-bank/activeContext.md:1), [`memory-bank/strategic_plan_nn_rl.md`](memory-bank/strategic_plan_nn_rl.md:1), etc., updated to reflect the outcomes of Task 1.4.