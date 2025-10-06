# Phase 2: Weight Initialization & Transfer Learning Strategy

**Document Version:** 1.0  
**Created:** October 6, 2025  
**Task:** 2.4 - Weight Initialization & Transfer  
**Priority:** CRITICAL for training success

---

## Executive Summary

Weight initialization and parameter sharing are CRITICAL for Phase 2 success. This document provides:
1. **Initialization strategies** (Xavier, He, Orthogonal) with implementation
2. **Shared encoder architecture** enabling 143 agents with 60% parameter reduction
3. **SL-to-RL transfer analysis** with risk mitigation (experimental)
4. **Convergence experiment framework** for Phase 3 validation

**Key Decision:** Implement robust initialization + weight sharing in Phase 2. Make SL pretraining optional/experimental given catastrophic SL failures.

---

## 1. Weight Initialization Strategies

### 1.1 Mathematical Foundations

**Xavier/Glorot Initialization:**
```
W ~ U(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))
```
- Best for: Tanh, Sigmoid activations
- Maintains variance across layers
- Original transformer paper used variant of this

**He/Kaiming Initialization:**
```
W ~ N(0, sqrt(2/n_in))
```
- Best for: ReLU activations
- Accounts for ReLU's non-linearity
- Our encoder uses ReLU in feedforward layers

**Orthogonal Initialization:**
```
W = orthogonal_matrix * gain
```
- Best for: Recurrent/policy networks
- Preserves gradient flow
- Proven superior for actor-critic (Saxe et al. 2013)

### 1.2 Recommended Strategy

**FeatureEncoder (Transformer):**
- **Linear layers:** Xavier uniform (standard for transformers)
- **Reason:** Balanced forward/backward pass variance
- **Gain:** 1.0 (default)

**SymbolAgent Actor:**
- **Hidden layers:** Orthogonal with gain=√2 (for ReLU)
- **Output layer:** Orthogonal with gain=0.01 (small init for exploration)
- **Reason:** Prevents initial policy collapse, encourages exploration

**SymbolAgent Critic:**
- **Hidden layers:** Orthogonal with gain=√2
- **Output layer:** Orthogonal with gain=1.0
- **Reason:** Accurate initial value estimates

### 1.3 Implementation

Create `core/rl/policies/initialization.py`:

```python
"""
Weight Initialization Utilities for RL Agents

Provides standardized initialization strategies with testing and validation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
from enum import Enum


class InitStrategy(Enum):
    """Available initialization strategies"""
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    HE_UNIFORM = "he_uniform"
    HE_NORMAL = "he_normal"
    ORTHOGONAL = "orthogonal"
    DEFAULT = "default"


def init_weights(
    module: nn.Module,
    strategy: InitStrategy = InitStrategy.ORTHOGONAL,
    gain: float = 1.0
):
    """
    Initialize module weights with specified strategy
    
    Args:
        module: PyTorch module to initialize
        strategy: Initialization strategy
        gain: Scaling factor (for orthogonal/xavier)
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        if strategy == InitStrategy.XAVIER_UNIFORM:
            nn.init.xavier_uniform_(module.weight, gain=gain)
        elif strategy == InitStrategy.XAVIER_NORMAL:
            nn.init.xavier_normal_(module.weight, gain=gain)
        elif strategy == InitStrategy.HE_UNIFORM:
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif strategy == InitStrategy.HE_NORMAL:
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        elif strategy == InitStrategy.ORTHOGONAL:
            nn.init.orthogonal_(module.weight, gain=gain)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def init_encoder(encoder: nn.Module):
    """Initialize transformer encoder (standard transformer init)"""
    for module in encoder.modules():
        init_weights(module, InitStrategy.XAVIER_UNIFORM, gain=1.0)


def init_actor(actor: nn.Module, output_gain: float = 0.01):
    """Initialize actor network (for exploration)"""
    modules = list(actor.children())
    
    # Hidden layers: orthogonal with sqrt(2) gain
    for module in modules[:-1]:
        init_weights(module, InitStrategy.ORTHOGONAL, gain=np.sqrt(2))
    
    # Output layer: small init for exploration
    if len(modules) > 0:
        init_weights(modules[-1], InitStrategy.ORTHOGONAL, gain=output_gain)


def init_critic(critic: nn.Module):
    """Initialize critic network (for accurate values)"""
    for module in critic.children():
        init_weights(module, InitStrategy.ORTHOGONAL, gain=np.sqrt(2))


def verify_initialization(model: nn.Module) -> Dict[str, float]:
    """
    Verify initialization statistics
    
    Returns dict with:
        - mean: Average weight value
        - std: Standard deviation
        - max: Maximum absolute value
        - has_nan: Whether any NaN values exist
    """
    weights = []
    for param in model.parameters():
        if param.requires_grad and param.dim() >= 2:  # Only weight matrices
            weights.append(param.data.cpu().numpy().flatten())
    
    all_weights = np.concatenate(weights)
    
    return {
        'mean': float(np.mean(all_weights)),
        'std': float(np.std(all_weights)),
        'max_abs': float(np.max(np.abs(all_weights))),
        'has_nan': bool(np.isnan(all_weights).any()),
        'has_inf': bool(np.isinf(all_weights).any())
    }


class InitializationLogger:
    """Log initialization statistics for analysis"""
    
    def __init__(self):
        self.stats = {}
    
    def log_model(self, name: str, model: nn.Module):
        """Log initialization stats for a model"""
        self.stats[name] = verify_initialization(model)
    
    def summary(self) -> str:
        """Generate summary report"""
        lines = ["Weight Initialization Summary", "="*50]
        
        for name, stats in self.stats.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Mean: {stats['mean']:.6f}")
            lines.append(f"  Std:  {stats['std']:.6f}")
            lines.append(f"  Max:  {stats['max_abs']:.6f}")
            lines.append(f"  NaN:  {stats['has_nan']}")
            lines.append(f"  Inf:  {stats['has_inf']}")
        
        return "\n".join(lines)
```

### 1.4 Testing

Create `tests/test_initialization.py`:

```python
"""Tests for weight initialization"""

import pytest
import torch
import numpy as np

from core.rl.policies.initialization import (
    init_weights, init_encoder, init_actor, init_critic,
    verify_initialization, InitStrategy
)
from core.rl.policies.feature_encoder import FeatureEncoder, EncoderConfig
from core.rl.policies.symbol_agent import SymbolAgent, SymbolAgentConfig


class TestInitialization:
    """Test weight initialization strategies"""
    
    def test_xavier_uniform(self):
        """Test Xavier uniform initialization"""
        layer = torch.nn.Linear(100, 50)
        init_weights(layer, InitStrategy.XAVIER_UNIFORM)
        
        stats = verify_initialization(layer)
        
        # Xavier should have small mean, bounded std
        assert abs(stats['mean']) < 0.1
        assert 0.1 < stats['std'] < 0.3
        assert not stats['has_nan']
    
    def test_orthogonal_init(self):
        """Test orthogonal initialization"""
        layer = torch.nn.Linear(100, 100)  # Must be square for orthogonal
        init_weights(layer, InitStrategy.ORTHOGONAL, gain=1.0)
        
        # Check orthogonality: W @ W.T should be close to I
        W = layer.weight.data
        product = W @ W.t()
        identity = torch.eye(100)
        
        diff = torch.norm(product - identity).item()
        assert diff < 1e-4  # Should be very close to identity
    
    def test_actor_small_init(self):
        """Test actor initialized with small output weights"""
        actor = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 7)
        )
        
        init_actor(actor, output_gain=0.01)
        
        # Output layer should have smaller weights
        output_std = actor[-1].weight.std().item()
        hidden_std = actor[0].weight.std().item()
        
        assert output_std < hidden_std
        assert output_std < 0.1  # Small for exploration
    
    def test_encoder_initialization(self):
        """Test encoder initialization"""
        config = EncoderConfig()
        encoder = FeatureEncoder(config)
        
        init_encoder(encoder)
        
        stats = verify_initialization(encoder)
        
        assert not stats['has_nan']
        assert not stats['has_inf']
        assert stats['std'] > 0  # Not all zeros
    
    def test_symbol_agent_initialization(self):
        """Test complete agent initialization"""
        config = SymbolAgentConfig()
        agent = SymbolAgent(config)
        
        # Agent should auto-initialize in __init__
        stats = verify_initialization(agent)
        
        assert not stats['has_nan']
        assert not stats['has_inf']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## 2. Shared Encoder Weights

### 2.1 Architecture Strategy

**Parameter Efficiency:**

| Architecture | Encoder Params | Agent Params (×143) | Total |
|-------------|----------------|---------------------|-------|
| **No Sharing** | 5M × 143 | 2M × 143 | 1,001M ❌ |
| **Shared Encoder** | 5M × 1 | 2M × 143 | 291M ✅ |
| **Savings** | - | - | **71% reduction** |

### 2.2 Implementation

**Already implemented in `symbol_agent.py`:**

```python
# Create shared encoder
shared_encoder = FeatureEncoder(config)

# Create 143 agents sharing the encoder
agents = []
for symbol in symbols:
    agent = SymbolAgent(
        config,
        shared_encoder=shared_encoder  # ← Key: reuse encoder
    )
    agents.append(agent)
```

### 2.3 Additional Utilities

Create `core/rl/policies/weight_sharing.py`:

```python
"""
Weight Sharing Utilities

Helpers for managing shared encoder weights across symbol agents.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from pathlib import Path

from .feature_encoder import FeatureEncoder
from .symbol_agent import SymbolAgent


class SharedEncoderManager:
    """Manages shared encoder weights across agents"""
    
    def __init__(self, encoder: FeatureEncoder):
        self.encoder = encoder
        self.agents: List[SymbolAgent] = []
    
    def create_agent(self, config) -> SymbolAgent:
        """Create new agent with shared encoder"""
        agent = SymbolAgent(config, shared_encoder=self.encoder)
        self.agents.append(agent)
        return agent
    
    def verify_sharing(self) -> bool:
        """Verify all agents share same encoder weights"""
        if len(self.agents) < 2:
            return True
        
        first_params = self.agents[0].encoder.parameters()
        
        for agent in self.agents[1:]:
            for p1, p2 in zip(first_params, agent.encoder.parameters()):
                if p1.data_ptr() != p2.data_ptr():
                    return False
        
        return True
    
    def count_parameters(self) -> Dict[str, int]:
        """Count shared vs unique parameters"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        
        agent_params = 0
        if self.agents:
            # Count params NOT in encoder
            agent = self.agents[0]
            for name, param in agent.named_parameters():
                if 'encoder' not in name:
                    agent_params += param.numel()
        
        total_unique = agent_params * len(self.agents)
        
        return {
            'shared_encoder': encoder_params,
            'unique_per_agent': agent_params,
            'total_agents': len(self.agents),
            'total_unique': total_unique,
            'total_params': encoder_params + total_unique,
            'sharing_efficiency': 1 - (encoder_params + total_unique) / 
                                 ((encoder_params + agent_params) * len(self.agents))
        }
    
    def save_encoder(self, path: Path):
        """Save shared encoder weights"""
        torch.save(self.encoder.state_dict(), path)
    
    def load_encoder(self, path: Path):
        """Load encoder weights into all agents"""
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict)
        # All agents automatically use new weights (shared reference)


def measure_memory_savings(num_agents: int, encoder_size_mb: float, agent_size_mb: float):
    """Calculate memory savings from weight sharing"""
    without_sharing = (encoder_size_mb + agent_size_mb) * num_agents
    with_sharing = encoder_size_mb + (agent_size_mb * num_agents)
    
    savings_mb = without_sharing - with_sharing
    savings_pct = 100 * savings_mb / without_sharing
    
    return {
        'without_sharing_mb': without_sharing,
        'with_sharing_mb': with_sharing,
        'savings_mb': savings_mb,
        'savings_pct': savings_pct
    }
```

---

## 3. SL-to-RL Transfer (Experimental)

### 3.1 Risk Analysis

**⚠️ Critical Concerns:**

1. **SL Models Failed Catastrophically:**
   - MLP: -88.05% backtesting return
   - LSTM: -92.60% return
   - GRU: -89.34% return
   - These models learned patterns that LOSE money

2. **Negative Transfer Risk:**
   - Pretraining from failed models could bias RL toward bad strategies
   - RL might inherit transaction cost blindness
   - Poor timing decisions could persist

3. **However, Potential Value:**
   - Classification metrics decent (ROC-AUC 0.84-0.87)
   - Feature encoders might have learned useful representations
   - Could accelerate early training if done carefully

### 3.2 Recommended Approach: **GUARDED EXPERIMENT**

**Phase 2: Build Infrastructure (Optional Use)**

```python
"""
SL-to-RL Transfer (Experimental)

WARNING: SL models failed backtesting. Use with extreme caution.
"""

import torch
from pathlib import Path
from typing import Optional

from .feature_encoder import FeatureEncoder


class SLtoRLTransfer:
    """Experimental: Transfer learning from SL to RL"""
    
    @staticmethod
    def load_sl_encoder_weights(
        rl_encoder: FeatureEncoder,
        sl_checkpoint_path: Path,
        layer_mapping: Optional[dict] = None
    ) -> dict:
        """
        Attempt to transfer SL encoder weights to RL encoder
        
        WARNING: SL models failed backtesting (-88% to -93%).
        Use only for experimental comparison.
        
        Returns:
            Dict with transfer statistics
        """
        # Load SL checkpoint
        sl_state = torch.load(sl_checkpoint_path)
        
        # Try to map layers
        transferred = 0
        failed = 0
        
        rl_state = rl_encoder.state_dict()
        
        for rl_key in rl_state.keys():
            # Attempt automatic mapping
            sl_key = layer_mapping.get(rl_key, rl_key) if layer_mapping else rl_key
            
            if sl_key in sl_state and rl_state[rl_key].shape == sl_state[sl_key].shape:
                rl_state[rl_key] = sl_state[sl_key]
                transferred += 1
            else:
                failed += 1
        
        rl_encoder.load_state_dict(rl_state)
        
        return {
            'transferred': transferred,
            'failed': failed,
            'success_rate': transferred / (transferred + failed) if transferred + failed > 0 else 0
        }
    
    @staticmethod
    def create_hybrid_init(
        rl_encoder: FeatureEncoder,
        sl_checkpoint_path: Path,
        sl_weight: float = 0.5
    ):
        """
        Hybrid initialization: blend SL weights with random init
        
        Args:
            sl_weight: How much to weight SL (0=random, 1=full SL)
        """
        # Save random init
        random_state = {k: v.clone() for k, v in rl_encoder.state_dict().items()}
        
        # Load SL weights
        SLtoRLTransfer.load_sl_encoder_weights(rl_encoder, sl_checkpoint_path)
        sl_state = {k: v.clone() for k, v in rl_encoder.state_dict().items()}
        
        # Blend
        blended_state = {}
        for key in random_state:
            if key in sl_state:
                blended_state[key] = (
                    sl_weight * sl_state[key] + 
                    (1 - sl_weight) * random_state[key]
                )
            else:
                blended_state[key] = random_state[key]
        
        rl_encoder.load_state_dict(blended_state)
```

### 3.3 Phase 3 Experimental Protocol

**A/B Testing in Prototype Training:**

1. **Condition A: Random Init (Baseline)**
   - Standard orthogonal/Xavier initialization
   - No SL influence

2. **Condition B: SL Encoder Transfer**
   - Load SL encoder weights to RL encoder
   - Keep actor/critic random

3. **Condition C: Hybrid Init (50/50)**
   - Blend SL and random weights
   - Reduces negative transfer risk

**Success Criteria (First 20k Steps):**
- If SL-init lags random-init by >10% cumulative reward → ABANDON
- If SL-init shows >15% lower sample efficiency → ABANDON
- If SL-init converges to degenerate policy → ABANDON

**Safeguard: KL Divergence Constraint**
```python
# In PPO update, add KL penalty vs random-init policy
kl_penalty = 0.01 * kl_divergence(current_policy, random_init_policy)
loss = policy_loss + value_loss - entropy_bonus + kl_penalty
```

---

## 4. Convergence Experiments (Phase 3 Activity)

### 4.1 Experimental Design

**Variables to Test:**
1. Initialization strategy (Xavier, He, Orthogonal)
2. Output gain (0.01, 0.1, 1.0)
3. SL transfer (None, Full, Hybrid 50%)
4. Encoder sharing (Shared, Independent)

**Metrics:**
- Sample efficiency (reward per 1k steps)
- Final performance (validation Sharpe)
- Training stability (gradient variance)
- Time to convergence

### 4.2 Implementation Framework

Create `scripts/run_init_experiments.py`:

```python
"""
Initialization Strategy Experiments

Compare different initialization strategies in Phase 3 training.
"""

import torch
from pathlib import Path
from stable_baselines3 import PPO

from core.rl.environments import make_vec_trading_env
from core.rl.policies import FeatureEncoder, SymbolAgent, SymbolAgentConfig
from core.rl.policies.initialization import init_encoder, init_actor, init_critic


def train_with_init_strategy(
    strategy: str,
    symbol: str = 'AAPL',
    total_timesteps: int = 50000
):
    """Train agent with specified initialization"""
    
    # Create environment
    env = make_vec_trading_env(symbol, num_envs=4)
    
    # Create agent with strategy
    config = SymbolAgentConfig()
    agent = SymbolAgent(config)
    
    if strategy == 'random':
        # Default random init (already done)
        pass
    elif strategy == 'orthogonal':
        init_encoder(agent.encoder)
        init_actor(agent.actor, output_gain=0.01)
        init_critic(agent.critic)
    elif strategy == 'sl_transfer':
        # Experimental: load SL weights
        from core.rl.policies.sl_to_rl_transfer import SLtoRLTransfer
        SLtoRLTransfer.load_sl_encoder_weights(
            agent.encoder,
            Path('models/sl_checkpoints/mlp_trial_72/checkpoint.pth')
        )
    
    # Train with PPO
    model = PPO('MultiInputPolicy', env, verbose=1)
    # ... training code
    
    return results
```

---

## 5. Phase 2 Implementation Checklist

### Task 2.4: Weight Initialization & Transfer ✅

- [ ] **2.4.1: Core Initialization Module**
  - [ ] Create `core/rl/policies/initialization.py`
  - [ ] Implement: Xavier, He, Orthogonal strategies
  - [ ] Add init_encoder(), init_actor(), init_critic() helpers
  - [ ] Add verification utilities

- [ ] **2.4.2: Initialization Tests**
  - [ ] Create `tests/test_initialization.py`
  - [ ] Test each strategy
  - [ ] Verify orthogonality, variance, bounds
  - [ ] Test encoder/actor/critic init

- [ ] **2.4.3: Weight Sharing Utilities**
  - [ ] Create `core/rl/policies/weight_sharing.py`
  - [ ] Implement SharedEncoderManager
  - [ ] Add sharing verification
  - [ ] Add memory savings calculator
  - [ ] Test with 143 agents

- [ ] **2.4.4: SL Transfer (Experimental Infrastructure)**
  - [ ] Create `core/rl/policies/sl_to_rl_transfer.py`
  - [ ] Implement weight loading (with warnings)
  - [ ] Implement hybrid init (blending)
  - [ ] Document risks clearly
  - [ ] Mark as experimental

- [ ] **2.4.5: Convergence Experiment Framework**
  - [ ] Create `scripts/run_init_experiments.py`
  - [ ] Design A/B test protocol
  - [ ] Add metrics tracking
  - [ ] Prepare for Phase 3 execution

---

## 6. Success Criteria

**Phase 2 Deliverables:**
- ✅ Initialization module with 3 strategies
- ✅ Weight sharing achieving >60% parameter reduction
- ✅ SL transfer infrastructure (experimental, optional)
- ✅ Convergence experiment framework for Phase 3

**Phase 3 Validation:**
- Random init agents achieve Sharpe >0.3
- SL transfer evaluated (abandon if underperforms)
- Best strategy documented for Phase 4 scale-up

---

## 7. Risk Mitigation

| Risk | Probability | Mitigation |
|------|------------|------------|
| SL transfer degrades performance | HIGH | A/B test, abandon if <10% worse |
| Poor initialization causes training collapse | MEDIUM | Multiple strategies, gradient monitoring |
| Weight sharing breaks gradient flow | LOW | Verification tests, gradient checks |

---

**RECOMMENDATION:** 
- ✅ IMPLEMENT: Orthogonal init + weight sharing (CRITICAL)
- ⚠️ EXPERIMENTAL: SL transfer with strict safeguards
- 📊 VALIDATE: In Phase 3 prototype training

This completes Task 2.4 with proper risk management and experimental rigor.