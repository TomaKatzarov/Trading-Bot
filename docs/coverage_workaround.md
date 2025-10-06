# Coverage Workflow for PyTorch-Based RL Modules

Running `pytest --cov` with the current PyTorch build raises
``RuntimeError: function '_has_torch_function' already has a docstring`` because
coverage instrumentation re-applies native docstrings. Until the upstream issue
is resolved, use the standard `coverage` CLI to gather metrics:

```bash
source trading_rl_env/Scripts/activate
coverage run -m pytest tests/test_feature_encoder.py
coverage report -m core/rl/policies/feature_encoder.py
```

This workflow executes the same test suite and produces accurate coverage
figures without triggering the docstring regression. The latest run reports
**100%** line coverage for `core/rl/policies/feature_encoder.py`.
