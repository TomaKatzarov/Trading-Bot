# Hyperparameter Optimization (HPO) Usage Guide

This guide explains how to use the hyperparameter optimization system for neural network models in the trading bot project.

## Overview

The HPO system uses Optuna for Bayesian optimization to automatically find the best hyperparameters for neural network models. It supports all model architectures (MLP, LSTM, GRU, CNN-LSTM) and integrates with MLflow for experiment tracking.

## Quick Start

### 1. Basic HPO Run

Run HPO for a specific model with default settings:

```bash
# Optimize LSTM model with 25 trials
python training/run_hpo_quick_start.py --model lstm --trials 25

# Optimize all models with 20 trials each
python training/run_hpo_quick_start.py --trials 20
```

### 2. Using Pre-generated Training Data

For faster HPO (recommended for extensive studies):

```bash
# First, generate training data
python scripts/generate_combined_training_data.py

# Then run HPO with pre-generated data
python training/run_hpo_quick_start.py --model gru --use-pregenerated --trials 30
```

### 3. Custom Study Configuration

For advanced users, create a custom configuration file:

```bash
# Copy and modify the example configuration
cp training/config_templates/hpo_example.yaml my_hpo_config.yaml

# Run with custom configuration
python training/run_hpo.py --config my_hpo_config.yaml
```

## HPO System Components

### 1. Main HPO Script (`training/run_hpo.py`)

The comprehensive HPO orchestration script with full configuration options:

- **Features:**
  - Configurable search spaces for all model types
  - Multiple sampler options (TPE, Random)
  - Pruning support (Median, Hyperband)
  - MLflow integration
  - Distributed optimization support
  - Comprehensive results analysis

- **Usage:**
  ```bash
  python training/run_hpo.py [options]
  ```

### 2. Quick Start Script (`training/run_hpo_quick_start.py`)

Simplified interface for common HPO tasks:

- **Features:**
  - Sensible defaults for quick experimentation
  - Interactive confirmation
  - Simplified command-line interface
  - Automatic configuration generation

- **Usage:**
  ```bash
  python training/run_hpo_quick_start.py [options]
  ```

### 3. Configuration Templates

Pre-configured templates for different use cases:

- `training/config_templates/hpo_example.yaml`: Comprehensive configuration example
- `training/config_templates/lstm_baseline.yaml`: LSTM model baseline
- `training/config_templates/mlp_baseline.yaml`: MLP model baseline

## Hyperparameter Search Spaces

### Common Parameters (All Models)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `learning_rate` | log_uniform | 1e-5 to 1e-2 | Learning rate for optimizer |
| `batch_size` | categorical | [32, 64, 128, 256] | Training batch size |
| `weight_decay` | log_uniform | 1e-6 to 1e-3 | L2 regularization strength |
| `dropout_rate` | uniform | 0.1 to 0.5 | Dropout probability |
| `focal_alpha` | uniform | 0.25 to 0.9 | Focal loss alpha parameter |
| `focal_gamma` | uniform | 0.5 to 3.0 | Focal loss gamma parameter |
| `early_stopping_patience` | int | 10 to 25 | Early stopping patience |

### Model-Specific Parameters

#### MLP
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `hidden_layers` | int | 2 to 4 | Number of hidden layers |
| `hidden_dim_1` | int | 64 to 512 | First hidden layer size |
| `hidden_dim_2` | int | 32 to 256 | Second hidden layer size |
| `asset_embedding_dim` | int | 4 to 32 | Asset embedding dimension |

#### LSTM/GRU
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `lstm_hidden_dim` | int | 32 to 256 | LSTM hidden dimension |
| `lstm_num_layers` | int | 1 to 3 | Number of LSTM layers |
| `attention_dim` | int | 32 to 128 | Attention mechanism dimension |
| `use_layer_norm` | categorical | [True, False] | Use layer normalization |

#### CNN-LSTM
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `cnn_filters_1` | int | 16 to 128 | First CNN layer filters |
| `cnn_kernel_size_1` | int | 3 to 7 | First CNN kernel size |
| `use_max_pooling` | categorical | [True, False] | Use max pooling |

## Configuration Options

### HPO Configuration

```yaml
# Basic settings
study_name: "my_hpo_study"
n_trials: 100
target_metric: "validation_f1_score_positive_class"
models_to_optimize: ["lstm", "gru"]

# Optimization settings
sampler: "tpe"  # or "random"
pruner: "median"  # or "hyperband", null
direction: "maximize"

# Output settings
output_dir: "hpo_results"
```

### Base Training Configuration

The HPO system uses the standard training configuration as a base and overrides hyperparameters during optimization. Key settings:

```yaml
training_config:
  epochs: 100
  early_stopping_patience: 15
  monitor_metric: "f1"
  loss_function:
    type: "focal"
    alpha: 0.25  # Will be optimized
    gamma: 2.0   # Will be optimized
```

## Results Analysis

### Output Files

HPO generates several output files:

1. **Individual Results:** `hpo_results_{model_type}.json`
   - Best hyperparameters for each model
   - Trial statistics and parameter importance
   - Performance metrics

2. **Combined Results:** `hpo_combined_results.json`
   - Summary of all model optimizations
   - Comparison across model types
   - Overall best configuration

3. **Optuna Database:** `hpo_studies/{study_name}.db`
   - Complete trial history
   - Can be loaded for further analysis

### MLflow Integration

All HPO trials are logged to MLflow:

- **Experiments:** Each HPO study creates an MLflow experiment
- **Runs:** Each trial is logged as a separate run
- **Metrics:** All validation metrics are tracked
- **Parameters:** All hyperparameters are logged
- **Artifacts:** Best models are saved as artifacts

Access MLflow UI:
```bash
mlflow ui
```

### Parameter Importance Analysis

The system automatically calculates parameter importance using Optuna's built-in analysis:

```python
import optuna
study = optuna.load_study(study_name="my_study", storage="sqlite:///hpo_studies/my_study.db")
importance = optuna.importance.get_param_importances(study)
```

## Best Practices

### 1. Start Small
- Begin with 20-30 trials for initial exploration
- Use quick start script for experimentation
- Gradually increase trials for final optimization

### 2. Use Pre-generated Data
- Generate training data once: `python scripts/generate_combined_training_data.py`
- Use `--use-pregenerated` flag for faster HPO
- Especially important for extensive studies (100+ trials)

### 3. Monitor Progress
- Check MLflow UI regularly during optimization
- Use pruning to stop unpromising trials early
- Monitor GPU memory usage for large models

### 4. Iterative Refinement
- Start with broad search spaces
- Narrow ranges based on initial results
- Use custom search space overrides for fine-tuning

### 5. Resource Management
- Use appropriate batch sizes for your GPU
- Consider distributed optimization for large studies
- Monitor disk space for MLflow artifacts

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in search space
   - Use smaller model architectures
   - Enable gradient checkpointing

2. **Slow HPO Progress**
   - Use pre-generated training data
   - Enable pruning
   - Reduce number of epochs for initial trials

3. **Poor Optimization Results**
   - Check search space ranges
   - Increase number of trials
   - Verify target metric is appropriate

4. **MLflow Connection Issues**
   - Check MLflow server status
   - Verify tracking URI configuration
   - Use local file storage as fallback

### Debug Mode

Enable verbose logging for debugging:

```bash
python training/run_hpo_quick_start.py --verbose --model lstm --trials 5
```

## Advanced Usage

### Custom Search Spaces

Override default search spaces in configuration:

```yaml
custom_search_spaces:
  common:
    learning_rate:
      type: "log_uniform"
      low: 1e-4
      high: 1e-3
  lstm:
    lstm_hidden_dim:
      type: "int"
      low: 64
      high: 128
      step: 32
```

### Distributed Optimization

Use PostgreSQL backend for distributed HPO:

```yaml
storage_url: "postgresql://user:pass@localhost/optuna"
```

### Multi-Objective Optimization

Extend the objective function to optimize multiple metrics:

```python
# In custom objective function
return [f1_score, precision, -training_time]  # Minimize training time
```

## Integration with Training Pipeline

### Using HPO Results

After HPO completion, use the best hyperparameters for final training:

```bash
# Extract best parameters from HPO results
python scripts/extract_best_hpo_params.py --study-name my_study --model lstm

# Train final model with best parameters
python training/train_nn_model.py --config best_lstm_config.yaml
```

### Automated Pipeline

Create automated pipeline for HPO → Training → Evaluation:

```bash
# 1. Run HPO
python training/run_hpo_quick_start.py --model lstm --trials 50

# 2. Extract best config
python scripts/extract_best_hpo_params.py --study-name latest --model lstm

# 3. Train final model
python training/train_nn_model.py --config best_lstm_config.yaml

# 4. Evaluate with backtesting
python scripts/run_backtest_with_model.py --model-path best_model.pt
```

## Performance Optimization

### GPU Utilization
- Use appropriate batch sizes for your GPU memory
- Enable mixed precision training for faster trials
- Monitor GPU utilization during HPO

### Storage Optimization
- Use SQLite for single-machine studies
- Use PostgreSQL for distributed studies
- Clean up old trial artifacts regularly

### Time Optimization
- Use pruning to stop poor trials early
- Pre-generate training data
- Use smaller validation sets for HPO

## Conclusion

The HPO system provides a comprehensive solution for optimizing neural network hyperparameters. Start with the quick start script for experimentation, then move to the full HPO system for production studies. Always monitor results through MLflow and use the analysis tools to understand parameter importance and optimization progress.