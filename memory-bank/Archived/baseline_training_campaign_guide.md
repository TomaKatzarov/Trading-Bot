# Phase 2: Baseline Model Training Campaign Guide

**Document Version:** 1.0  
**Date:** 2025-09-30  
**Status:** Ready for Execution

## Overview

This guide documents the Phase 2 Baseline Training Campaign for establishing performance benchmarks across all 4 neural network architectures before proceeding to hyperparameter optimization (Phase 3).

## Campaign Objectives

1. **Train 4 Neural Network Architectures** with standardized baseline configurations
2. **Establish Performance Benchmarks** for each architecture type
3. **Identify Best Candidates** for intensive hyperparameter optimization
4. **Document Baseline Performance** for comparison with optimized models

## Model Architectures

### 1. MLP (Multi-Layer Perceptron)
- **Purpose:** Baseline feedforward architecture
- **Config:** `training/config_templates/mlp_baseline.yaml`
- **Key Parameters:**
  - Hidden dims: [256, 128, 64]
  - Dropout: 0.3
  - Batch norm: True

### 2. LSTM with Attention
- **Purpose:** Primary recurrent model
- **Config:** `training/config_templates/lstm_baseline.yaml`
- **Key Parameters:**
  - Hidden size: 128
  - Layers: 2
  - Attention dim: 64

### 3. GRU with Attention
- **Purpose:** Alternative recurrent model
- **Config:** `training/config_templates/gru_baseline.yaml`
- **Key Parameters:**
  - Hidden size: 128
  - Layers: 2
  - Attention dim: 64

### 4. CNN-LSTM Hybrid
- **Purpose:** Exploratory architecture
- **Config:** `training/config_templates/cnn_lstm_baseline.yaml`
- **Key Parameters:**
  - CNN channels: [32, 64]
  - LSTM hidden: 128
  - Attention dim: 64

## Common Baseline Configuration

All models share these standardized settings:

### Data Configuration
- **Lookback window:** 24 hours
- **Batch size:** 128
- **Train/Val/Test split:** 70/15/15
- **Profit target:** +5%
- **Stop loss:** -2%
- **Prediction horizon:** 8 hours

### Training Configuration
- **Epochs:** 100 (with early stopping)
- **Learning rate:** 0.001
- **Optimizer:** AdamW
- **Weight decay:** 0.01
- **Loss function:** Focal Loss (α=0.25, γ=2.0)
- **Gradient clipping:** 1.0
- **Early stopping patience:** 15 epochs
- **LR scheduler:** ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 7 epochs
  - Mode: max (monitoring F1-score)

## Prerequisites

### Required Data
The campaign requires pre-generated combined training data. The orchestration script will automatically:
1. Check if `data/training_data/` exists
2. Generate it if missing using `scripts/generate_combined_training_data.py`
3. Verify data quality before proceeding

### Required Files
```
data/training_data/
├── train_X.npy
├── train_y.npy
├── train_asset_ids.npy
├── val_X.npy
├── val_y.npy
├── val_asset_ids.npy
├── test_X.npy
├── test_y.npy
├── test_asset_ids.npy
├── scalers.joblib
├── asset_id_mapping.json
└── metadata.json
```

### System Requirements
- **GPU:** NVIDIA RTX 5070 Ti 16GB (or equivalent)
- **CUDA:** Compatible version with PyTorch 2.2+
- **RAM:** 16GB+ recommended
- **Storage:** ~10GB for data and models

## Execution

### Quick Start
```bash
# Run the complete campaign
python scripts/run_baseline_training_campaign.py
```

### Manual Execution (Individual Models)
If you prefer to train models individually:

```bash
# 1. Generate training data (if needed)
python scripts/generate_combined_training_data.py

# 2. Train individual models
python training/train_nn_model.py --config training/config_templates/mlp_baseline.yaml --use-pregenerated-data
python training/train_nn_model.py --config training/config_templates/lstm_baseline.yaml --use-pregenerated-data
python training/train_nn_model.py --config training/config_templates/gru_baseline.yaml --use-pregenerated-data
python training/train_nn_model.py --config training/config_templates/cnn_lstm_baseline.yaml --use-pregenerated-data
```

## Expected Duration

### Per-Model Estimates
- **MLP:** 3-4 hours (simpler architecture)
- **LSTM:** 5-6 hours (recurrent with attention)
- **GRU:** 5-6 hours (recurrent with attention)
- **CNN-LSTM:** 6-7 hours (hybrid architecture)

### Total Campaign
- **Sequential execution:** 19-23 hours
- **With early stopping:** May finish earlier (12-18 hours typical)

## Monitoring Progress

### Real-Time Monitoring
The orchestration script displays:
- Model being trained
- Epoch progress with tqdm bars
- Training/validation metrics per epoch
- Best performance achieved
- Early stopping triggers

### MLflow UI
Access detailed experiment tracking:
```bash
# Start MLflow UI (if not running)
mlflow ui

# Access at: http://localhost:5000
```

**MLflow Organization:**
- Experiment: "Baseline_Training_Production"
- Tags for each run:
  - `model_type`: mlp | lstm | gru | cnn_lstm
  - `stage`: PRODUCTION
  - `purpose`: baseline_benchmark
  - `phase`: baseline_training

## Success Criteria

### Minimum Requirements
- **All 4 models train without errors**
- **Validation F1-score > 0.20** for positive class
- **Best models saved** for each architecture
- **All runs logged to MLflow**
- **Test set evaluation completed**

### Quality Indicators
- Convergence within 100 epochs
- Stable validation metrics (not oscillating)
- Reasonable precision/recall balance
- No severe overfitting (train vs val gap < 0.15)

## Output Artifacts

### Per-Model Outputs
```
training/runs/<model_type>_baseline/
├── best_model.pt                    # Best model weights
├── scalers.joblib                   # Feature scalers
├── training_history.csv             # Epoch-by-epoch metrics
├── confusion_matrix.png             # Final confusion matrix
└── learning_curves.png              # Training curves
```

### MLflow Artifacts
- Model weights
- Training configuration
- Metrics history
- Visualizations (confusion matrices, ROC curves)
- System metadata (git hash, environment)

## Post-Campaign Analysis

### 1. Compare Model Performance
```python
# View in MLflow UI or programmatically
import mlflow

experiment = mlflow.get_experiment_by_name("Baseline_Training_Production")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Compare metrics
comparison = runs[['tags.model_type', 'metrics.val_f1_score_positive_class', 
                   'metrics.val_pr_auc', 'metrics.test_f1_score_positive_class']]
print(comparison.sort_values('metrics.val_f1_score_positive_class', ascending=False))
```

### 2. Identify Top Performers
Look for models with:
- **Highest validation F1-score**
- **Best precision-recall balance**
- **Stable training (low variance)**
- **Good test set generalization**

### 3. Prepare Phase 3 Recommendations
Based on baseline results, prioritize models for HPO:
- **Primary HPO target:** Best performing architecture
- **Secondary HPO targets:** Close competitors
- **De-prioritize:** Models significantly underperforming

## Troubleshooting

### Data Generation Fails
```bash
# Check data availability
python scripts/check_available_data.py

# Verify specific symbols
ls data/historical/*/1Hour/data.parquet

# Manual data generation with verbose output
python scripts/generate_combined_training_data.py 2>&1 | tee data_gen.log
```

### Training Fails
- **GPU OOM:** Reduce batch_size in config
- **NaN loss:** Check data scaling, reduce learning rate
- **No improvement:** Verify data quality, check class balance

### Slow Training
- **Check GPU utilization:** `nvidia-smi`
- **Verify data loading:** Monitor CPU usage
- **Consider reducing:** num_workers if I/O bound

## Next Steps After Completion

### Phase 3: Hyperparameter Optimization
1. **Review baseline results** in MLflow
2. **Select top 1-2 architectures** for intensive HPO
3. **Run HPO campaigns:**
   ```bash
   python training/run_hpo.py --model-type lstm --n-trials 100
   ```
4. **Compare optimized vs baseline** performance

### Documentation Requirements
- Update `memory-bank/progress.md`
- Document baseline performance in `memory-bank/decisionLog.md`
- Create performance comparison report
- Update `memory-bank/activeContext.md`

## Campaign Checklist

- [ ] Prerequisites verified (GPU, data, configs)
- [ ] Training data generated and validated
- [ ] MLflow server running
- [ ] Sufficient disk space available
- [ ] Campaign execution started
- [ ] All 4 models trained successfully
- [ ] Results reviewed in MLflow
- [ ] Best performers identified
- [ ] Memory Bank updated
- [ ] Phase 3 priorities determined

## Support and References

### Key Documentation
- Training Infrastructure: `docs/experiment_management.md`
- HPO Framework: `docs/hpo_usage_guide.md`
- Data Preparation: `docs/nn_data_preparer_usage.md`
- Operational Plan: `memory-bank/project_plans/plan_1.4_train_tune_nn_models.md`

### Configuration Files
- Baseline configs: `training/config_templates/*_baseline.yaml`
- HPO examples: `training/config_templates/hpo_*.yaml`

---

**Document Status:** Ready for Production  
**Last Updated:** 2025-09-30  
**Maintained By:** AI Development Team