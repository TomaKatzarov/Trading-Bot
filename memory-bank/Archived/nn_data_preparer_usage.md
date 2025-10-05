# Neural Network Data Preparer Usage Guide

This document provides comprehensive usage instructions for the `NNDataPreparer` class in `core/data_preparation_nn.py`.

## Overview

The `NNDataPreparer` class is responsible for preparing financial time-series data for Neural Network training as part of the AI-powered trading system. It handles data loading, feature selection, preprocessing, label generation, sequence creation, and data splitting for multi-symbol training.

## Key Features

- **Multi-Symbol Support**: Processes data from multiple trading symbols simultaneously
- **Asset ID Embedding**: Simple integer mapping strategy for multi-symbol training compatibility
- **Feature Engineering**: Supports 18 base features including technical indicators, sentiment, and temporal features
- **Label Generation**: Forward-looking profit/stop-loss target logic with configurable parameters
- **Sequence Generation**: Sliding window approach for time-series data
- **Data Scaling**: StandardScaler and RobustScaler support with proper train-only fitting
- **Class Imbalance Handling**: Sample weight calculation and oversampling strategies
- **Temporal Data Splitting**: Respects chronological order for realistic evaluation

## Installation and Setup

Ensure you have the required dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

The module expects the following directory structure:

```
project_root/
├── config/
│   ├── symbols.json          # Symbol configuration
│   └── asset_id_mapping.json # Auto-generated asset ID mapping
├── data/
│   └── historical/
│       ├── AAPL/
│       │   └── 1Hour/
│       │       └── data.parquet
│       ├── MSFT/
│       │   └── 1Hour/
│       │       └── data.parquet
│       └── ...
└── models/
    └── scalers.joblib        # Saved feature scalers
```

## Basic Usage

### 1. Configuration Setup

```python
from core.data_preparation_nn import NNDataPreparer

# Define configuration
config = {
    # Data source configuration
    'symbols_config_path': 'config/symbols.json',
    'data_base_path': 'data',
    
    # Feature selection (18 base features from feature_set_NN.md)
    'feature_list': [
        # Technical indicators (14)
        'SMA_10', 'SMA_20', 'MACD', 'MACD_signal', 'MACD_histogram',
        'RSI_14', 'Stoch_K', 'Stoch_D', 'ADX_14', 'ATR_14',
        'BB_bandwidth', 'OBV', 'Volume_SMA_20', 'Returns_1h',
        # Sentiment (1)
        'sentiment_score_hourly_ffill',
        # Day of week (2)
        'DayOfWeek_sin', 'DayOfWeek_cos'
    ],
    
    # Data preprocessing
    'nan_handling_features': 'ffill',  # 'ffill', 'drop', 'bfill', 'interpolate'
    
    # Sequence generation
    'lookback_window': 24,        # Number of timesteps for sequences (24 hours)
    'prediction_horizon': 8,      # Hours to look ahead for labels
    
    # Label generation
    'profit_target': 0.05,        # +5% profit target
    'stop_loss_target': 0.02,     # -2% stop loss
    
    # Data splitting
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'shuffle_before_split': False,  # Keep temporal order
    
    # Feature scaling
    'scaling_method': 'standard',   # 'standard' or 'robust'
    
    # Class imbalance handling
    'calculate_sample_weights': True,
    'sample_weight_strategy': 'inverse_frequency',
    
    # Output paths
    'output_path_scalers': 'models/scalers.joblib',
    
    # Symbol selection (optional - if not provided, uses all symbols)
    'symbols_list': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
}
```

### 2. Initialize and Prepare Data

```python
# Initialize the data preparer
preparer = NNDataPreparer(config)

# Prepare data for training (main orchestration method)
prepared_data = preparer.get_prepared_data_for_training()
```

### 3. Access Prepared Data

```python
# Access training data
train_data = prepared_data['train']
X_train = train_data['X']          # Shape: (n_samples, lookback_window, n_features)
y_train = train_data['y']          # Shape: (n_samples,)
asset_ids_train = train_data['asset_ids']  # Shape: (n_samples,)
sample_weights = train_data['sample_weights']  # Shape: (n_samples,) - if enabled

# Access validation data
val_data = prepared_data['val']
X_val = val_data['X']
y_val = val_data['y']
asset_ids_val = val_data['asset_ids']

# Access test data
test_data = prepared_data['test']
X_test = test_data['X']
y_test = test_data['y']
asset_ids_test = test_data['asset_ids']

# Access scalers and asset mapping
scalers = prepared_data['scalers']
asset_id_map = prepared_data['asset_id_map']

print(f"Training data shape: {X_train.shape}")
print(f"Feature names: {config['feature_list']}")
print(f"Asset ID mapping: {asset_id_map}")
```

## Advanced Usage

### Custom Feature Selection

```python
# Use a subset of features
config['feature_list'] = [
    'SMA_10', 'SMA_20', 'RSI_14', 'MACD',
    'sentiment_score_hourly_ffill',
    'DayOfWeek_sin', 'DayOfWeek_cos'
]
```

### Per-Feature Scaling

```python
# Apply different scaling methods to different features
config['scaling_method_map'] = {
    0: 'standard',    # SMA_10
    1: 'standard',    # SMA_20
    2: 'robust',      # RSI_14 (use robust for bounded indicators)
    3: 'standard',    # MACD
    4: 'robust',      # sentiment_score_hourly_ffill
    5: 'standard',    # DayOfWeek_sin
    6: 'standard'     # DayOfWeek_cos
}
```

### Custom Label Generation Parameters

```python
# Adjust profit/loss targets and prediction horizon
config.update({
    'profit_target': 0.03,        # 3% profit target (easier)
    'stop_loss_target': 0.015,    # 1.5% stop loss
    'prediction_horizon': 12      # 12-hour prediction horizon
})
```

### Manual Sample Weight Strategy

```python
config.update({
    'calculate_sample_weights': True,
    'sample_weight_strategy': 'manual'  # Custom weight calculation
})
```

## Individual Method Usage

### Load Data for Single Symbol

```python
preparer = NNDataPreparer(config)

# Load raw data for a specific symbol
aapl_data = preparer.load_data_for_symbol('AAPL')
print(f"AAPL data shape: {aapl_data.shape}")
print(f"Date range: {aapl_data.index.min()} to {aapl_data.index.max()}")
```

### Generate Labels

```python
# Generate labels for a symbol
aapl_labels = preparer._generate_labels_for_symbol(aapl_data, 'AAPL')
label_distribution = aapl_labels['label'].value_counts()
print(f"Label distribution: {label_distribution}")
```

### Create Sequences

```python
# Preprocess features
aapl_features = preparer._preprocess_single_symbol_data('AAPL')

# Generate sequences
asset_id = preparer.asset_id_map['AAPL']
X, y, asset_ids = preparer._generate_sequences_for_symbol(
    aapl_features, aapl_labels, asset_id
)
print(f"Generated {len(X)} sequences for AAPL")
```

## Data Output Format

The `get_prepared_data_for_training()` method returns a dictionary with the following structure:

```python
{
    'train': {
        'X': np.ndarray,           # Shape: (n_train, lookback_window, n_features)
        'y': np.ndarray,           # Shape: (n_train,) - binary labels (0/1)
        'asset_ids': np.ndarray,   # Shape: (n_train,) - integer asset IDs
        'sample_weights': np.ndarray  # Shape: (n_train,) - optional
    },
    'val': {
        'X': np.ndarray,           # Shape: (n_val, lookback_window, n_features)
        'y': np.ndarray,           # Shape: (n_val,)
        'asset_ids': np.ndarray    # Shape: (n_val,)
    },
    'test': {
        'X': np.ndarray,           # Shape: (n_test, lookback_window, n_features)
        'y': np.ndarray,           # Shape: (n_test,)
        'asset_ids': np.ndarray    # Shape: (n_test,)
    },
    'scalers': dict,               # Fitted sklearn scalers
    'asset_id_map': dict          # Symbol to integer ID mapping
}
```

## Data Types and Shapes

- **X arrays**: `np.float32`, shape `(n_samples, lookback_window, n_features)`
- **y arrays**: `np.int32`, shape `(n_samples,)` with values 0 (no signal) or 1 (buy signal)
- **asset_ids arrays**: `np.int32`, shape `(n_samples,)` with integer asset IDs
- **sample_weights**: `np.float64`, shape `(n_samples,)` with positive weights

## Error Handling

The class includes comprehensive error handling:

```python
try:
    prepared_data = preparer.get_prepared_data_for_training()
except FileNotFoundError as e:
    print(f"Data file not found: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

- **Caching**: Raw data is cached in memory to avoid repeated file reads
- **Memory Usage**: Large datasets may require significant RAM for sequence generation
- **Processing Time**: Multi-symbol processing can be time-intensive for large symbol lists

## Integration with Neural Network Training

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(prepared_data['train']['X'])
y_train_tensor = torch.LongTensor(prepared_data['train']['y'])

# Create dataset and dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Use sample weights in loss function if available
if 'sample_weights' in prepared_data['train']:
    sample_weights = torch.FloatTensor(prepared_data['train']['sample_weights'])
    # Apply weights in your loss calculation
```

## Troubleshooting

### Common Issues

1. **Missing Data Files**: Ensure all symbols in `symbols_list` have corresponding data files
2. **Insufficient Data**: Some symbols may not have enough data for sequence generation
3. **Memory Issues**: Reduce `symbols_list` or `lookback_window` for large datasets
4. **Label Imbalance**: Adjust `profit_target` and `stop_loss_target` if no positive labels are generated

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check data quality:

```python
# Verify data shapes and distributions
for split_name in ['train', 'val', 'test']:
    split_data = prepared_data[split_name]
    print(f"{split_name}: X shape {split_data['X'].shape}, y shape {split_data['y'].shape}")
    print(f"  Positive ratio: {np.mean(split_data['y']):.3f}")
    print(f"  Unique asset IDs: {np.unique(split_data['asset_ids'])}")
```

## Best Practices

1. **Feature Selection**: Start with the full 18-feature set, then experiment with subsets
2. **Label Parameters**: Begin with 5%/2% profit/stop targets, adjust based on data characteristics
3. **Temporal Splitting**: Always use `shuffle_before_split: False` for time-series data
4. **Scaling**: Use `standard` scaling for most features, `robust` for bounded indicators
5. **Sample Weights**: Enable for imbalanced datasets, but monitor for over-weighting
6. **Validation**: Always validate data quality and label distributions before training

This comprehensive guide should help you effectively use the `NNDataPreparer` class for your neural network training data preparation needs.