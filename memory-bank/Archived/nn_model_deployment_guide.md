# Neural Network Model Deployment Guide

**Document Version:** 1.0  
**Date:** 2025-09-30  
**Activity:** 7.5 - Model Deployment Considerations  
**Author:** Flow-Code

## Table of Contents
1. [Overview](#overview)
2. [Loading Trained Models](#loading-trained-models)
3. [Data Preprocessing for Inference](#data-preprocessing-for-inference)
4. [Model Serving Options](#model-serving-options)
5. [Production Monitoring](#production-monitoring)
6. [Version Control Best Practices](#version-control-best-practices)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide outlines the essential steps and considerations for deploying trained Neural Network models for live trading or inference in the TradingBotAI system. Proper deployment ensures:

- **Consistency:** Inference preprocessing matches training preprocessing exactly
- **Reliability:** Models perform as expected in production
- **Monitoring:** Performance degradation is detected early
- **Maintainability:** Clear version control and rollback procedures

---

## Loading Trained Models

### Method 1: Loading from Local Checkpoint

```python
import torch
import joblib
import json

# Load model checkpoint
checkpoint_path = "models/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Extract components
model = checkpoint['model']  # Or reconstruct from state_dict
model.eval()

# Load scaler
scaler = checkpoint['scalers']  # Or from separate file
scaler_path = "models/scalers.joblib"
scaler = joblib.load(scaler_path)

# Load asset ID mapping
with open("config/asset_id_mapping.json", 'r') as f:
    asset_id_data = json.load(f)
    asset_id_map = asset_id_data['symbol_to_id']
```

### Method 2: Loading from MLflow Artifacts

```python
import mlflow
import mlflow.pytorch

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:///path/to/mlruns")

# Load model from MLflow run
run_id = "abc123def456"
model_uri = f"runs:/{run_id}/best_model"
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# Download additional artifacts
client = mlflow.tracking.MlflowClient()
artifact_dir = client.download_artifacts(run_id, "")

# Load scaler and mapping from artifacts
scaler = joblib.load(f"{artifact_dir}/scalers.joblib")
with open(f"{artifact_dir}/asset_id_mapping.json", 'r') as f:
    asset_id_map = json.load(f)['symbol_to_id']
```

### Method 3: Using SupervisedNNStrategy

```python
from core.strategies.supervised_nn_strategy import SupervisedNNStrategy

# Option A: Load from local files
config = {
    'model_path': 'models/best_model.pt',
    'scaler_path': 'models/scalers.joblib',
    'asset_id_map_path': 'config/asset_id_mapping.json',
    'signal_threshold': 0.7,
    'max_holding_period_hours': 8,
    'feature_list': ['SMA_10', 'SMA_20', 'MACD_line', ...]  # Full feature list
}

strategy = SupervisedNNStrategy(config)
# Dependencies loaded automatically

# Option B: Load from MLflow
config = {'signal_threshold': 0.7, 'max_holding_period_hours': 8}
strategy = SupervisedNNStrategy(config)
strategy.load_dependencies_from_mlflow(run_id="abc123def456")
```

---

## Data Preprocessing for Inference

### Critical Requirements

**⚠️ CRITICAL:** Preprocessing for inference MUST exactly match training preprocessing to avoid distribution shift and degraded performance.

### Preprocessing Pipeline

```python
import pandas as pd
import numpy as np

def preprocess_for_inference(raw_data: pd.DataFrame, 
                            feature_list: List[str],
                            scaler,
                            lookback_window: int = 24) -> np.ndarray:
    """
    Preprocess raw data for model inference.
    
    Args:
        raw_data: DataFrame with historical market data
        feature_list: List of feature column names (must match training)
        scaler: Fitted scaler from training
        lookback_window: Number of time steps to use
        
    Returns:
        Preprocessed array ready for model input
    """
    # 1. Select features (must match training feature order)
    feature_data = raw_data[feature_list].tail(lookback_window)
    
    # 2. Handle missing values (same strategy as training)
    # Example: forward fill then backward fill
    feature_data = feature_data.ffill().bfill()
    
    # 3. Apply scaling transformation
    scaled_features = scaler.transform(feature_data.values)
    
    # 4. Reshape for model: (1, lookback_window, n_features)
    input_tensor = scaled_features.reshape(1, lookback_window, -1)
    
    return input_tensor
```

### Common Preprocessing Pitfalls

❌ **Don't:**
- Change feature order between training and inference
- Use different scaling parameters
- Skip handling of missing values
- Include future data (look-ahead bias)
- Use different datetime handling or timezone conversion

✅ **Do:**
- Use exact same feature list and order
- Apply the same scaler fitted during training
- Handle NaN values consistently
- Ensure temporal alignment
- Verify data types match training

---

## Model Serving Options

### Option 1: Direct Integration (Current Approach)

Integrate model directly into trading decision loop:

```python
from core.strategies.supervised_nn_strategy import SupervisedNNStrategy

# Initialize once
strategy = SupervisedNNStrategy(config)

# In trading loop
while trading:
    # Get latest market data
    historical_data = fetch_latest_data(symbol, hours=24)
    
    # Prepare input and get prediction
    feature_tensor, asset_id_tensor = strategy.prepare_input_sequence(
        historical_data, symbol
    )
    
    prediction_prob = strategy.get_model_prediction(
        feature_tensor, asset_id_tensor
    )
    
    # Generate trading action
    action = strategy.generate_trade_action(
        prediction_probability=prediction_prob,
        current_position_status=current_position,
        time_in_position_hours=time_in_position
    )
```

### Option 2: RESTful API Service

Create a Flask/FastAPI service for model serving:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd

app = FastAPI()

# Load model globally (once at startup)
model = None
scaler = None
strategy = None

@app.on_event("startup")
async def load_model():
    global model, scaler, strategy
    config = {...}  # Load from file
    strategy = SupervisedNNStrategy(config)

class PredictionRequest(BaseModel):
    symbol: str
    historical_data: List[Dict]  # List of OHLCV dicts
    
class PredictionResponse(BaseModel):
    prediction_probability: float
    recommended_action: str
    timestamp: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.historical_data)
        
        # Get prediction
        feature_tensor, asset_id_tensor = strategy.prepare_input_sequence(
            df, request.symbol
        )
        
        if feature_tensor is None:
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        prediction = strategy.get_model_prediction(feature_tensor, asset_id_tensor)
        
        # Generate action
        action = strategy.generate_trade_action(
            prediction_probability=prediction,
            current_position_status="FLAT"
        )
        
        return PredictionResponse(
            prediction_probability=float(prediction),
            recommended_action=action,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Option 3: Batch Inference Service

For processing multiple symbols or time periods:

```python
class BatchInferenceEngine:
    """Efficient batch inference for multiple symbols."""
    
    def __init__(self, model_path: str, scaler_path: str, config: Dict):
        self.model = self._load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.config = config
    
    def predict_batch(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Run inference on multiple symbols.
        
        Args:
            data_dict: Dictionary of {symbol: historical_dataframe}
            
        Returns:
            Dictionary of {symbol: prediction_probability}
        """
        predictions = {}
        
        # Prepare batch
        feature_batch = []
        asset_id_batch = []
        symbols_list = []
        
        for symbol, df in data_dict.items():
            features = self.preprocess_data(df)
            if features is not None:
                feature_batch.append(features)
                asset_id = self.config['asset_id_map'].get(symbol, 0)
                asset_id_batch.append(asset_id)
                symbols_list.append(symbol)
        
        if not feature_batch:
            return predictions
        
        # Batch inference
        with torch.no_grad():
            feature_tensor = torch.tensor(np.vstack(feature_batch), dtype=torch.float32)
            asset_id_tensor = torch.tensor(asset_id_batch, dtype=torch.long)
            
            outputs = self.model(feature_tensor, asset_id_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
        
        # Map back to symbols
        for symbol, prob in zip(symbols_list, probabilities):
            predictions[symbol] = float(prob)
        
        return predictions
```

### Option 4: ONNX Export (Advanced)

Export PyTorch model to ONNX for optimized inference:

```python
import torch.onnx

# Create dummy input for tracing
dummy_features = torch.randn(1, 24, 23)  # (batch, lookback, features)
dummy_asset_ids = torch.tensor([0], dtype=torch.long)

# Export to ONNX
model.eval()
torch.onnx.export(
    model,
    (dummy_features, dummy_asset_ids),
    "models/model.onnx",
    input_names=['features', 'asset_ids'],
    output_names=['logits'],
    dynamic_axes={
        'features': {0: 'batch_size'},
        'asset_ids': {0: 'batch_size'}
    },
    opset_version=14
)

# Load and run ONNX model
import onnxruntime as ort

session = ort.InferenceSession("models/model.onnx")
outputs = session.run(
    None,
    {
        'features': features.numpy(),
        'asset_ids': asset_ids.numpy()
    }
)
```

---

## Production Monitoring

### Performance Monitoring

Monitor model performance in production to detect degradation:

```python
class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self, alert_thresholds: Dict[str, float]):
        self.alert_thresholds = alert_thresholds
        self.prediction_history = []
        self.performance_history = []
    
    def log_prediction(self, prediction: float, symbol: str, 
                      actual_outcome: Optional[bool] = None):
        """Log prediction for monitoring."""
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'prediction': prediction,
            'actual_outcome': actual_outcome
        })
    
    def check_distribution_shift(self, window_size: int = 1000):
        """Check for distribution shift in predictions."""
        if len(self.prediction_history) < window_size:
            return None
        
        recent = self.prediction_history[-window_size:]
        historical = self.prediction_history[-2*window_size:-window_size]
        
        # Calculate statistics
        recent_mean = np.mean([p['prediction'] for p in recent])
        historical_mean = np.mean([p['prediction'] for p in historical])
        
        # Alert if significant shift
        shift = abs(recent_mean - historical_mean)
        if shift > self.alert_thresholds.get('distribution_shift', 0.1):
            return {
                'alert': 'Distribution Shift Detected',
                'recent_mean': recent_mean,
                'historical_mean': historical_mean,
                'shift': shift
            }
        
        return None
    
    def calculate_rolling_accuracy(self, window_size: int = 100):
        """Calculate rolling accuracy on predictions with known outcomes."""
        predictions_with_outcomes = [
            p for p in self.prediction_history 
            if p['actual_outcome'] is not None
        ]
        
        if len(predictions_with_outcomes) < window_size:
            return None
        
        recent = predictions_with_outcomes[-window_size:]
        
        # Calculate accuracy
        correct = sum(
            1 for p in recent 
            if (p['prediction'] >= 0.5) == p['actual_outcome']
        )
        
        accuracy = correct / len(recent)
        
        # Alert if below threshold
        if accuracy < self.alert_thresholds.get('min_accuracy', 0.55):
            return {
                'alert': 'Low Accuracy Detected',
                'rolling_accuracy': accuracy,
                'window_size': window_size
            }
        
        return {'rolling_accuracy': accuracy}
```

### Key Metrics to Monitor

| Metric | Threshold | Action |
|--------|-----------|--------|
| Rolling Accuracy | < 55% over 100 predictions | Investigate model drift |
| Prediction Mean Shift | > 0.1 from historical mean | Check for distribution changes |
| Inference Latency | > 100ms (p95) | Optimize or scale resources |
| Memory Usage | > 80% of available | Implement batch processing |
| Error Rate | > 5% of requests | Debug preprocessing pipeline |

### Logging Strategy

```python
import logging
from datetime import datetime

class InferenceLogger:
    """Structured logging for model inference."""
    
    def __init__(self, log_path: str):
        self.logger = logging.getLogger('inference')
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_inference(self, symbol: str, prediction: float, 
                     features: Dict, latency_ms: float):
        """Log inference details."""
        self.logger.info(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'prediction': float(prediction),
            'features_summary': {
                k: float(v) for k, v in list(features.items())[:5]
            },
            'latency_ms': latency_ms
        }))
```

---

## Data Preprocessing for Inference

### Preprocessing Consistency Checklist

✅ **Before Deployment:**
- [ ] Verify exact feature list matches training
- [ ] Confirm feature order is identical
- [ ] Validate scaler parameters (mean, std) are from training set only
- [ ] Test preprocessing with sample data
- [ ] Document any special handling (NaN strategy, outliers, etc.)

### Feature Engineering Consistency

```python
class ProductionPreprocessor:
    """Ensures consistent preprocessing between training and inference."""
    
    def __init__(self, config_path: str):
        """Load preprocessing configuration from training."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load scaler
        self.scaler = joblib.load(self.config['scaler_path'])
        
        # Store feature metadata
        self.feature_list = self.config['feature_list']
        self.lookback_window = self.config['lookback_window']
        self.nan_strategy = self.config.get('nan_handling', 'ffill')
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input data meets requirements."""
        # Check columns
        missing_features = set(self.feature_list) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Check data sufficiency
        if len(df) < self.lookback_window:
            raise ValueError(
                f"Insufficient data: {len(df)} < {self.lookback_window} required"
            )
        
        # Check for excessive NaNs
        nan_percentage = df[self.feature_list].isna().sum().sum() / (len(df) * len(self.feature_list))
        if nan_percentage > 0.1:  # >10% NaN is suspicious
            raise ValueError(f"Too many NaN values: {nan_percentage:.2%}")
        
        return True
    
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Apply preprocessing pipeline."""
        self.validate_input(df)
        
        # Select features
        feature_data = df[self.feature_list].tail(self.lookback_window)
        
        # Handle NaN (must match training strategy)
        if self.nan_strategy == 'ffill':
            feature_data = feature_data.ffill().bfill()
        elif self.nan_strategy == 'drop':
            feature_data = feature_data.dropna()
        # Add other strategies as needed
        
        # Apply scaling
        scaled = self.scaler.transform(feature_data.values)
        
        # Reshape
        return scaled.reshape(1, self.lookback_window, -1)
```

### Real-Time Data Alignment

```python
def align_realtime_data(historical_df: pd.DataFrame,
                       realtime_bar: Dict,
                       feature_calculator) -> pd.DataFrame:
    """
    Align real-time data with historical for seamless inference.
    
    Args:
        historical_df: Historical data DataFrame
        realtime_bar: Latest bar data from market feed
        feature_calculator: Object to calculate technical indicators
        
    Returns:
        Updated DataFrame ready for inference
    """
    # Convert realtime bar to DataFrame row
    new_row = pd.DataFrame([realtime_bar])
    
    # Append to historical
    combined = pd.concat([historical_df, new_row], ignore_index=False)
    
    # Recalculate indicators if needed
    # (Some indicators need recent history to compute latest value)
    combined = feature_calculator.calculate_all_indicators(combined)
    
    # Return most recent data
    return combined.tail(25)  # +1 for potential indicator calculation
```

---

## Model Serving Options

### Development/Testing Environment

```bash
# Run backtest on historical data
python scripts/run_nn_backtest.py \
    --model-source local \
    --model-path models/best_model.pt \
    --scaler-path models/scalers.joblib \
    --symbols AAPL MSFT \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

### Production Deployment Checklist

- [ ] Model artifacts versioned and stored securely
- [ ] Preprocessing configuration documented
- [ ] Error handling and fallbacks implemented
- [ ] Monitoring and alerting configured
- [ ] Rollback procedure tested
- [ ] Load testing completed
- [ ] Security review conducted (if API-based)
- [ ] Performance benchmarks established

---

## Version Control Best Practices

### Model Versioning Strategy

```
models/
├── v1.0/
│   ├── model.pt
│   ├── scalers.joblib
│   ├── asset_id_mapping.json
│   ├── config.json
│   └── metadata.json
├── v1.1/
│   └── ...
└── production/  # Symlink to current production version
    └── -> v1.0/
```

### Metadata Tracking

```json
{
  "model_version": "1.0",
  "training_date": "2025-09-30",
  "mlflow_run_id": "abc123def456",
  "git_commit": "a1b2c3d4",
  "training_data_version": "20250930",
  "architecture": "lstm_attention",
  "performance_metrics": {
    "test_f1": 0.2847,
    "test_precision": 0.3021,
    "validation_sharpe": 1.42
  },
  "hyperparameters": {
    "lstm_hidden_dim": 128,
    "attention_dim": 64,
    "learning_rate": 0.0003
  },
  "deployed_date": "2025-10-01",
  "deployed_by": "trading_team"
}
```

### Git Workflow

```bash
# Tag model releases
git tag -a model-v1.0 -m "LSTM+Attention model, F1=0.28, Sharpe=1.42"
git push origin model-v1.0

# Track code version with model
echo "git_commit: $(git rev-parse HEAD)" >> models/v1.0/metadata.json
```

### MLflow Model Registry

```python
# Register model in MLflow
import mlflow

client = mlflow.tracking.MlflowClient()

# Register model
model_uri = f"runs:/{run_id}/best_model"
mv = mlflow.register_model(model_uri, "TradingSignalPredictor")

# Transition to production
client.transition_model_version_stage(
    name="TradingSignalPredictor",
    version=mv.version,
    stage="Production"
)

# Load production model
production_model = mlflow.pyfunc.load_model(
    model_uri="models:/TradingSignalPredictor/Production"
)
```

---

## Performance Optimization

### Inference Latency Optimization

```python
# Option 1: Torch JIT Compilation
model_scripted = torch.jit.script(model)
model_scripted.save("models/model_scripted.pt")

# Load and use
model = torch.jit.load("models/model_scripted.pt")
model.eval()

# Option 2: Half Precision (FP16) on GPU
model = model.half()  # Convert to FP16
features = features.half()

# Option 3: Batch Processing
def batch_predict(models: List, data_batch: List) -> List:
    """Process multiple predictions in one forward pass."""
    # Combine into batch
    feature_batch = torch.stack([d['features'] for d in data_batch])
    asset_id_batch = torch.tensor([d['asset_id'] for d in data_batch])
    
    # Single forward pass
    with torch.no_grad():
        outputs = model(feature_batch, asset_id_batch)
        probabilities = torch.sigmoid(outputs)
    
    return probabilities.cpu().numpy()
```

### Memory Management

```python
# Cleanup unused tensors
import gc

def run_inference_with_cleanup(data):
    """Run inference with proper memory management."""
    try:
        result = model(data)
        result_cpu = result.cpu().numpy()
        
        # Delete GPU tensors
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result_cpu
        
    finally:
        gc.collect()
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Prediction probabilities differ from training

**Cause:** Preprocessing mismatch  
**Solution:**
```python
# Verify scaler parameters
print("Scaler mean:", scaler.mean_)
print("Scaler scale:", scaler.scale_)

# Check feature values before scaling
print("Raw features:", feature_data.describe())

# Check scaled features
print("Scaled features:", scaled_features.describe())
```

#### Issue: Model outputs constant predictions

**Cause:** Model not in eval mode or batch norm issues  
**Solution:**
```python
# Ensure eval mode
model.eval()

# Disable dropout/batch norm
with torch.no_grad():
    predictions = model(features, asset_ids)
```

#### Issue: Slow inference latency

**Cause:** Inefficient data loading or model size  
**Solution:**
```python
# Profile inference
import time

start = time.time()
# ... preprocessing ...
prep_time = time.time() - start

start = time.time()
# ... model inference ...
inference_time = time.time() - start

print(f"Preprocessing: {prep_time*1000:.2f}ms")
print(f"Inference: {inference_time*1000:.2f}ms")
```

#### Issue: Out of memory errors

**Cause:** Large batch size or model size  
**Solution:**
```python
# Reduce batch size
batch_size = 16  # Instead of 64

# Use gradient checkpointing (if training)
model.gradient_checkpointing_enable()

# Clear cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## Security Considerations

### Model Security

1. **Model File Integrity:**
   - Use checksums (MD5/SHA256) to verify model files
   - Store models in secure locations with access controls

2. **Input Validation:**
   - Validate all input data before inference
   - Sanitize any user-provided symbols or parameters
   - Rate limit API endpoints to prevent abuse

3. **Secure Credentials:**
   - Never hardcode API keys in model code
   - Use environment variables or secure vaults
   - Rotate credentials regularly

### Example: Input Validation

```python
def validate_trading_input(symbol: str, data: pd.DataFrame) -> bool:
    """Validate input data for security and correctness."""
    # Whitelist allowed symbols
    ALLOWED_SYMBOLS = set([...])  # Load from config
    if symbol not in ALLOWED_SYMBOLS:
        raise ValueError(f"Symbol {symbol} not authorized")
    
    # Check data size limits
    if len(data) > 10000:  # Reasonable limit
        raise ValueError("Input data too large")
    
    # Validate data types
    if not all(dtype in ['float64', 'int64'] for dtype in data.dtypes):
        raise ValueError("Invalid data types in input")
    
    # Check for injection attempts in column names
    for col in data.columns:
        if not col.replace('_', '').isalnum():
            raise ValueError(f"Invalid column name: {col}")
    
    return True
```

---

## Best Practices Summary

### DO ✅

1. **Use exact same preprocessing** as training
2. **Version all artifacts** (model, scaler, config, code)
3. **Monitor performance** continuously in production
4. **Implement fallbacks** for model failures
5. **Test thoroughly** before deployment
6. **Document everything** (config, quirks, edge cases)
7. **Use MLflow** for experiment tracking and model registry
8. **Implement proper logging** for debugging
9. **Validate inputs** before inference
10. **Have rollback procedures** ready

### DON'T ❌

1. **Don't skip preprocessing validation**
2. **Don't use different feature scaling**
3. **Don't ignore model monitoring**
4. **Don't deploy without testing**
5. **Don't hardcode configurations**
6. **Don't forget error handling**
7. **Don't mix training and inference code**
8. **Don't ignore version control**
9. **Don't skip security reviews**
10. **Don't assume model will perform indefinitely**

---

## Quick Start Guide

### 1. Prepare for Deployment

```bash
# Verify model artifacts
python -c "import torch; checkpoint = torch.load('models/best_model.pt'); print('Model loaded successfully')"

# Check scaler
python -c "import joblib; scaler = joblib.load('models/scalers.joblib'); print(f'Scaler: {type(scaler)}')"

# Validate configuration
python -c "import json; config = json.load(open('config/deployment_config.json')); print('Config valid')"
```

### 2. Run Test Inference

```python
# test_inference.py
from core.strategies.supervised_nn_strategy import SupervisedNNStrategy
import pandas as pd

# Load strategy
strategy = SupervisedNNStrategy({
    'model_path': 'models/best_model.pt',
    'scaler_path': 'models/scalers.joblib',
    'feature_list': [...]  # Your features
})

# Load test data
test_data = pd.read_parquet('data/test_sample.parquet')

# Run inference
feature_tensor, asset_id_tensor = strategy.prepare_input_sequence(test_data, 'AAPL')
prediction = strategy.get_model_prediction(feature_tensor, asset_id_tensor)

print(f"Prediction: {prediction:.4f}")
```

### 3. Deploy to Production

```bash
# Run backtest to validate
python scripts/run_nn_backtest.py \
    --model-source local \
    --model-path models/best_model.pt \
    --scaler-path models/scalers.joblib \
    --symbols AAPL \
    --start-date 2024-11-01 \
    --end-date 2024-11-30

# If backtest successful, deploy
# (Integration with your trading system)
```

---

## Continuous Improvement

### Retraining Strategy

1. **Scheduled Retraining:** Retrain model weekly/monthly with latest data
2. **Event-Triggered:** Retrain when performance drops below threshold
3. **A/B Testing:** Deploy new model to subset of symbols first
4. **Shadow Mode:** Run new model in parallel, compare with production

### Model Lifecycle

```
[Training] → [Validation] → [Staging] → [Production] → [Monitoring] → [Retraining]
     ↑                                                         |
     └─────────────────────────────────────────────────────────┘
```

---

## References

- [Training Script Documentation](training_procedure_nn.md)
- [Experiment Management Guide](experiment_management.md)
- [HPO Usage Guide](hpo_usage_guide.md)
- [Backtesting Documentation](backtesting_logging.md)

---

**Last Updated:** 2025-09-30  
**Document Owner:** Flow-Code  
**Review Cycle:** Monthly