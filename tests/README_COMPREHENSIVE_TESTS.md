# Comprehensive Training Pipeline Test Suite

This directory contains a comprehensive test suite designed to validate every aspect of the SAC training pipeline with surgical precision before training begins.

## 🎯 Test Coverage

### 1. **Data Validation & Configuration** (`test_data_validation_comprehensive.py`)
- ✅ Data format compliance (Parquet, Phase 3 structure)
- ✅ OHLCV data integrity (no nulls, infinites, valid ranges)
- ✅ OHLC constraint validation (high ≥ close, low ≤ close, etc.)
- ✅ Technical indicator validity (RSI [0,100], proper calculations)
- ✅ Train/val/test split consistency
- ✅ Timestamp chronology and frequency
- ✅ Configuration loading from YAML
- ✅ Feature engineering correctness

### 2. **Reward Calculation Accuracy** (`test_reward_calculation_accuracy.py`)
- ✅ PnL reward scaling with profit percentage
- ✅ Positive rewards for profits, negative for losses
- ✅ Loss penalty multiplier effects
- ✅ Reward clipping to configured bounds
- ✅ Position sizing multipliers (small/medium/large)
- ✅ Exit type multipliers (partial/full)
- ✅ Transaction cost proportional scaling
- ✅ Diversity bonus for varied actions
- ✅ Diversity penalty for action collapse
- ✅ Component weight aggregation
- ✅ ROI-based scaling
- ✅ Forced exit penalties
- ✅ Numerical stability (zero equity, NaN handling)

### 3. **Action Space Behavior** (`test_action_space_behavior.py`)
- ✅ Continuous action mapping to discrete actions
- ✅ Hold threshold enforcement
- ✅ Buy/sell signal interpretation
- ✅ Action magnitude affecting position size
- ✅ Action masking (anti-exploit defense)
  - Blocks excessive buy concentration
  - Enforces minimum hold periods
  - Progressive position building
- ✅ Multi-position support (concurrent positions)
- ✅ Max positions limit enforcement
- ✅ Position sizing constraints
  - Respects max_position_pct
  - Respects available capital
  - Enforces min_trade_value
- ✅ Trade execution validation
  - Portfolio state updates
  - Commission/slippage costs
  - Position closing logic
- ✅ Action smoothing window

### 4. **End-to-End Training Pipeline** (`test_comprehensive_training_pipeline.py`)
- ✅ Configuration loading from YAML
- ✅ RewardConfig construction from dict
- ✅ PortfolioConfig construction from dict
- ✅ TradingConfig assembly with nested configs
- ✅ Data file existence and readability
- ✅ Required technical indicators presence
- ✅ No missing values in data
- ✅ Sequential timestamps
- ✅ Environment initialization (discrete & continuous)
- ✅ Environment reset produces valid observations
- ✅ Configuration propagation to environment
- ✅ Full episode execution without errors
- ✅ Episode termination at configured length
- ✅ Portfolio state updates
- ✅ Evaluation mode consistency
- ✅ Multi-environment vectorization (SubprocVecEnv)
- ✅ Edge case handling (missing columns, insufficient data)
- ✅ SAC integration (instantiation, training smoke test)

### 5. **Reward Infrastructure (Phase A2)** (`test_reward_infrastructure_e2e.py`)
- ✅ RewardConfig matches phase_a2_sac_sharpe.yaml
- ✅ Sharpe gate initialization state
- ✅ Component weight application in aggregation
- ✅ Negative ROI penalty respects multipliers
- ✅ Positive PnL reward clipping
- ✅ Diversity penalty on action collapse
- ✅ Small positive ROI generates reward
- ✅ ROI negative scale reduces loss magnitude
- ✅ Reward clip bounds total reward

### 6. **Reward Shaper Stage 2** (`test_reward_shaper_stage2.py`)
- ✅ Sharpe gate requires voluntary closes
- ✅ Forced exit penalty scales with loss
- ✅ Time decay penalty after threshold
- ✅ Sharpe gate mechanics

## 🚀 Quick Start

### Run All Tests
```bash
# Activate environment
./activate_rl_env.bat  # Windows
source activate_rl_env.sh  # Linux/Mac

# Run comprehensive test suite
python tests/run_comprehensive_tests.py
```

### Run Specific Test Suite
```bash
# Run only data validation tests
python tests/run_comprehensive_tests.py --suite "Data Validation"

# Run only reward calculation tests
python tests/run_comprehensive_tests.py --suite "Reward"

# Run with verbose output
python tests/run_comprehensive_tests.py --verbose
```

### Quick Validation (Skip Lengthy Tests)
```bash
python tests/run_comprehensive_tests.py --quick
```

### Generate Test Report
```bash
python tests/run_comprehensive_tests.py --report-file test_report.txt
```

### Run Individual Test Files
```bash
# Data validation
pytest tests/test_data_validation_comprehensive.py -v

# Reward accuracy
pytest tests/test_reward_calculation_accuracy.py -v

# Action space
pytest tests/test_action_space_behavior.py -v

# Full pipeline
pytest tests/test_comprehensive_training_pipeline.py -v
```

## 📊 Test Report Example

```
================================================================================
COMPREHENSIVE TRAINING PIPELINE TEST REPORT
Generated: 2025-10-15 14:30:22
================================================================================

SUMMARY
--------------------------------------------------------------------------------
Total Test Suites:   6
Passed:              6 ✓
Failed:              0 ✗
Skipped:             0 ⊘
Total Duration:      45.32s

DETAILED RESULTS
--------------------------------------------------------------------------------

1. Data Validation & Configuration
   Status:      ✓ PASSED
   Duration:    8.21s

2. Reward Calculation Accuracy
   Status:      ✓ PASSED
   Duration:    12.45s

[... more results ...]

================================================================================
✓ ALL TESTS PASSED - TRAINING PIPELINE READY
================================================================================
```

## 🔧 Troubleshooting

### Missing Dependencies
```bash
pip install pytest numpy pandas torch stable-baselines3
```

### Test Data Generation
Tests automatically generate synthetic data in temporary directories. No manual data preparation needed.

### Configuration Files
Some tests require `training/config_templates/phase_a2_sac_sharpe.yaml`. If missing, those specific tests will be skipped (not fail).

## 📝 Adding New Tests

### Test File Structure
```python
"""Test description."""
from __future__ import annotations

import pytest
from core.rl.environments import ...

@pytest.fixture
def my_fixture():
    """Fixture description."""
    # Setup
    yield data
    # Teardown

def test_my_feature():
    """Test that feature works correctly."""
    # Arrange
    data = setup_test_data()
    
    # Act
    result = function_under_test(data)
    
    # Assert
    assert result == expected_value
```

### Register in Master Runner
Edit `tests/run_comprehensive_tests.py` and add to `TEST_SUITES`:
```python
{
    "name": "My New Test Suite",
    "file": "tests/test_my_new_suite.py",
    "description": "What this suite validates",
}
```

## 🎯 Pre-Training Checklist

Before starting a training run, ensure:

1. ✅ All tests pass: `python tests/run_comprehensive_tests.py`
2. ✅ Data splits exist: `data/phase3_splits/{symbol}/train.parquet`
3. ✅ Configuration valid: Check YAML syntax and parameter values
4. ✅ GPU available: `torch.cuda.is_available()` returns `True`
5. ✅ Sufficient disk space: >10GB for checkpoints and logs
6. ✅ MLflow configured: `mlflow ui` accessible

## 📚 Related Documentation

- **Training Guide**: `docs/rl_training_guide.md`
- **Configuration Reference**: `training/config_templates/README.md`
- **Reward Shaping**: `docs/reward_shaping_guide.md`
- **Environment Spec**: `core/rl/environments/README.md`

## 🤝 Contributing

When adding new features to the training pipeline:

1. Write tests first (TDD approach)
2. Ensure tests cover edge cases
3. Update this README with test descriptions
4. Run full test suite before committing
5. Add test coverage to CI/CD pipeline

## 📞 Support

If tests fail unexpectedly:

1. Check test output for specific failure details
2. Verify environment setup: `python --version`, `pip list`
3. Review configuration files for syntax errors
4. Check data integrity: `data/*/metadata.json`
5. Consult logs: `logs/test_run_*.log`

---

**Last Updated**: October 15, 2025  
**Test Suite Version**: 1.0.0  
**Compatibility**: Python 3.10+, PyTorch 2.0+, SB3 2.0+
