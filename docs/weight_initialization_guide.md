# Weight Initialization & Transfer Guide

## Overview
Proper weight initialization is essential for stable PPO training across 143 symbol agents. This guide summarizes the standardized strategies introduced in Task 2.4 and documents how to apply them in practice, verify outcomes, and reason about optional weight transfer from supervised learning (SL) checkpoints.

## Initialization Strategies

All helpers are implemented in `core/rl/policies/initialization.py` and exposed via `__all__` for convenient importing.

### 1. Xavier (Glorot) Initialization
**Used for:** Transformer encoder blocks (attention + feedforward)

**Formula:**
$$
W \sim \mathcal{U}\left[-\sqrt{\frac{6}{\text{fan}_\text{in} + \text{fan}_\text{out}}}, \sqrt{\frac{6}{\text{fan}_\text{in} + \text{fan}_\text{out}}}\right]
$$

**Why:** Preserves variance through GELU/tanh-style activations common in transformer layers and stabilizes residual paths.

**API:** `init_encoder(module, strategy="xavier_uniform", gain=1.0)`

### 2. Orthogonal Initialization
**Used for:** Actor and critic multilayer perceptrons

**Formula:**
$$
W = Q \cdot \text{gain}, \qquad Q = \text{orthonormal basis from QR decomposition of a random Gaussian matrix}
$$

**Gains:** Hidden layers use $\sqrt{2}$, actor outputs use $0.01$ (per PPO best practices), critic outputs use $1.0$.

**Why:** Encourages diverse exploration and mitigates vanishing/exploding gradients in deeper value/policy heads.

**API:**

```python
init_actor(actor, strategy="orthogonal", gain=0.01, hidden_gain=math.sqrt(2.0))
init_critic(critic, strategy="orthogonal", gain=1.0, hidden_gain=math.sqrt(2.0))
```

### 3. He (Kaiming) Initialization
**Used for:** ReLU-heavy modules (optional alternative to orthogonal)

**Formula:**
$$
W \sim \mathcal{U}\left[-\sqrt{\frac{6}{\text{fan}_\text{in}}}, \sqrt{\frac{6}{\text{fan}_\text{in}}}\right] \quad \text{or} \quad \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan}_\text{in}}}\right)
$$

**Why:** Maintains activation scale for ReLU/Leaky ReLU features.

**API:** `he_uniform_init(module, gain=math.sqrt(2.0))`, `he_normal_init(module, gain=math.sqrt(2.0))`

### Verification Helper
Use `verify_initialization(module, strategy, tolerance=0.1, gain=1.0)` to compare empirical mean/std with theoretical expectations, detect NaNs/Infs, and confirm orthogonality.

## Practical Checklist
1. **Import helpers** in policy modules:
   ```python
   from core.rl.policies.initialization import init_encoder, init_actor, init_critic
   ```
2. **Apply during module construction**:
   ```python
   self.encoder.apply(lambda m: init_encoder(m, "xavier_uniform"))
   self.actor.apply(lambda m: init_actor(m, "orthogonal", gain=0.01))
   self.critic.apply(lambda m: init_critic(m, "orthogonal", gain=1.0))
   ```
   (See `core/rl/policies/feature_encoder.py` and `core/rl/policies/symbol_agent.py` for concrete implementations.)
3. **Verify stats** as part of smoke tests or debugging sessions:
   ```python
   report = verify_initialization(self.encoder, "xavier_uniform", tolerance=0.15)
   assert report["passed"], report["checks"]
   ```
4. **Parameter sharing audit** (optional but recommended before large-scale training):
   ```python
   manager = SharedEncoderManager(shared_encoder)
   manager.register_agent(symbol_agent)
   manager.print_report()
   ```
   Measured savings for 143 agents: 97.29% parameter reduction (12.8M params with sharing vs. 472.8M without; ≈1.71 GB memory saved).

## Supervised → RL Transfer (Experimental)
The SL models underperformed severely in backtesting, so RL training defaults to randomized initialization. However, the utilities in `core/rl/policies/sl_to_rl_transfer.py` allow optional migration experiments:

1. **Load an SL checkpoint** (must contain `model.pt`):
   ```python
   from core.rl.policies.sl_to_rl_transfer import load_sl_checkpoint
   checkpoint = load_sl_checkpoint(Path("models/sl_checkpoints/aapl_lstm"))
   ```
2. **Map compatible tensors** to the RL encoder:
   ```python
   transferred = transfer_sl_features_to_encoder(encoder, checkpoint)
   ```
   A warning is emitted if mismatched shapes are skipped.
3. **Bundle experiment artifacts**:
   ```python
   experiment = create_sl_transfer_experiment(encoder, Path("models/sl_checkpoints/aapl_lstm"), freeze=False)
   ```

⚠️ **Warning:** Retire the experiment if it lags behind random initialization by >10% cumulative reward after 20k PPO steps.

## References
- Glorot & Bengio (2010) — Understanding the difficulty of training deep feedforward neural networks
- He et al. (2015) — Delving deep into rectifiers
- Saxe et al. (2014) — Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
- Schulman et al. (2017) — Proximal Policy Optimization Algorithms (actor gain heuristics)

## Testing & Validation
- `pytest tests/test_initialization.py -v` covers initialization statistics, weight sharing math, and SL transfer warnings.
- Run `pytest tests/test_symbol_agent.py tests/test_feature_encoder.py -v` after refactors that touch initialization to ensure downstream policies remain stable.
- Capture parameter savings data in future documentation updates (e.g., add screenshot of `SharedEncoderManager.print_report()` output once available).

## Next Steps
- Integrate verification hooks into training pipelines (log `verify_initialization` outputs to MLflow).
- Automate savings calculations in Phase 3 training scripts to guard against accidental encoder duplication.
- Evaluate SL transfer only after PPO exhibits stable learning curves; consider curriculum warm-starts as an alternative.
