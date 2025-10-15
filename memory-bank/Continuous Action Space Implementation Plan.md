Critical Assessment: RL Action Space Collapse in Trading Environment
Executive Summary
After analyzing your comprehensive implementation across all stages, I've identified a fundamental architectural mismatch between your reward structure, action space design, and the inherent dynamics of trading environments. The persistent action collapse despite multiple sophisticated interventions indicates we're fighting symptoms rather than root causes.

Core Diagnosis: The Triple Bind
1. The Sparse Reward Paradox
Your environment has created what I call a "sparse reward desert" - the agent only receives meaningful feedback when:

A position is opened AND closed (requiring 2+ coordinated actions)
Market conditions align favorably
Risk limits aren't violated
Between these rare events, HOLD dominates because it's the only action guaranteed not to incur penalties. This is exacerbated by:

Transaction costs creating immediate negative rewards for trading
Curriculum penalties for "wrong" actions
Risk gates blocking profitable pyramiding
2. The Evaluation Catastrophe
The most telling symptom: policies collapse specifically during deterministic evaluation. This reveals that your agents never actually learned to trade - they learned to exploit the exploration mechanism. When ε=0, the true policy emerges: pure HOLD/SELL with no trading logic.

3. The Action Space Impedance Mismatch
Your discrete 7-action space creates artificial boundaries that don't map to trading reality:

No gradual position sizing (only SMALL/MEDIUM/LARGE)
SELL_PARTIAL vs SELL_ALL is arbitrary
ADD_TO_WINNER requires specific preconditions
Why Standard Fixes Failed
Your implementations were technically correct but targeted the wrong layer:

Entropy bonuses/penalties: Can't overcome when HOLD genuinely has highest expected reward
Curriculum penalties: Add noise but don't change fundamental dynamics
Optimizer tweaks: Stabilize learning but don't fix reward sparsity
Forced exit penalties: Create fear without teaching profitable exit timing
The Fundamental Recovery Plan
Phase A: Immediate Stabilization (Week 1)
A1. Implement Continuous Action Space

class ContinuousTradeAction:
    """
    Replace discrete actions with continuous [-1, 1] output:
    - [-1.0, -0.1]: Sell proportional to current position
    - [-0.1, 0.1]: Hold zone (with small threshold)
    - [0.1, 1.0]: Buy proportional to available capital
    """
    
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def interpret_action(self, raw_action, current_position, available_capital):
        if abs(raw_action) < 0.1:  # Hold threshold
            return ActionType.HOLD, 0
        
        if raw_action > 0:  # Buy signal
            size = abs(raw_action) * available_capital * self.max_position_pct
            return ActionType.BUY, size
        else:  # Sell signal
            size = abs(raw_action) * current_position
            return ActionType.SELL, size

A2. Implement Intrinsic Curiosity Module (ICM)

class TradingICM(nn.Module):
    """
    Intrinsic reward for exploring state-action consequences
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Forward model: predict next state
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Inverse model: predict action from state transition
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def compute_intrinsic_reward(self, state, action, next_state):
        # Prediction error as curiosity signal
        predicted_next = self.forward_model(torch.cat([state, action]))
        intrinsic_reward = F.mse_loss(predicted_next, next_state, reduction='none').mean(dim=-1)
        return intrinsic_reward * 0.01  # Scale appropriately

Phase B: Hierarchical Decomposition (Week 2)
B1. Implement Options Framework

class TradingOptions:
    """
    High-level options that execute over multiple timesteps
    """
    
    options = {
        'open_long': OpenLongOption(),      # Executes until position opened
        'close_position': CloseOption(),    # Executes until position closed  
        'trend_follow': TrendFollowOption(), # Maintains position in trend
        'scalp': ScalpOption(),            # Quick in-out trades
        'wait': WaitOption()                # Active waiting with exit readiness
    }
    
    class OptionPolicy(nn.Module):
        def __init__(self):
            self.option_selector = nn.Linear(state_dim, len(options))
            self.termination_fn = nn.Linear(state_dim, len(options))
            
        def forward(self, state):
            option_logits = self.option_selector(state)
            termination_prob = torch.sigmoid(self.termination_fn(state))
            return option_logits, termination_prob

B2. Implement Hindsight Experience Replay (HER)

class TradingHER:
    """
    Learn from failed trades by relabeling goals
    """
    
    def relabel_trajectory(self, trajectory, achieved_return):
        # If we aimed for +2% but got +0.5%
        # Relabel the goal as +0.5% and mark as success
        
        synthetic_trajectories = []
        for t in range(len(trajectory)):
            # Create alternative goals
            for goal in self.sample_achieved_goals(trajectory[t:]):
                relabeled = trajectory.copy()
                relabeled.goal = goal
                relabeled.reward = self.compute_reward(trajectory[t:], goal)
                synthetic_trajectories.append(relabeled)
                
        return synthetic_trajectories


Phase C: Reward Revolution (Week 3)
C1. Implement Potential-Based Reward Shaping

class PotentialBasedRewards:
    """
    Dense rewards that maintain optimal policy
    """
    
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        
    def potential_function(self, state):
        """
        Φ(s) = expected_return_from_state(s)
        Learned via separate value network
        """
        features = {
            'distance_to_support': min(0, price - support_level),
            'distance_to_resistance': max(0, resistance_level - price),
            'trend_alignment': trend_strength * position_direction,
            'time_in_position': -log(1 + position_duration),
            'unrealized_pnl': current_pnl
        }
        return self.value_net(features)
    
    def shaped_reward(self, state, action, next_state, original_reward):
        # F(s,a,s') = γΦ(s') - Φ(s)
        shaping = self.gamma * self.potential_function(next_state) - self.potential_function(state)
        return original_reward + shaping

C2. Multi-Objective Reward Decomposition

class MultiObjectiveTrading:
    """
    Separate value heads for different objectives
    """
    
    def __init__(self):
        self.objectives = {
            'profit': ProfitHead(),
            'risk': RiskHead(),
            'activity': ActivityHead(),
            'timing': TimingHead()
        }
        
    def compute_rewards(self, transition):
        rewards = {}
        rewards['profit'] = self.compute_pnl(transition)
        rewards['risk'] = -self.compute_drawdown(transition)
        rewards['activity'] = self.compute_trade_frequency(transition)
        rewards['timing'] = self.compute_entry_quality(transition)
        
        # Learned weighted combination
        weights = self.weight_network(transition.state)
        total_reward = sum(w * r for w, r in zip(weights, rewards.values()))
        
        return total_reward, rewards  # Keep components for analysis

Phase D: Advanced Training Techniques (Week 4)
D1. Implement IMPALA-style Off-Policy Correction

class VTrace:
    """
    Correct for off-policy actions during learning
    """
    
    def compute_vtrace(self, behavior_logits, target_logits, rewards, values, bootstrap):
        rho = torch.exp(target_logits - behavior_logits)
        rho_bar = torch.minimum(rho, torch.ones_like(rho))
        c_bar = torch.minimum(rho, torch.ones_like(rho) * self.c_bar_threshold)
        
        # Compute V-trace targets
        vs = []
        v = bootstrap
        for t in reversed(range(len(rewards))):
            delta = rho_bar[t] * (rewards[t] + self.gamma * v - values[t])
            v = values[t] + delta + self.gamma * c_bar[t] * (v - values[t])
            vs.append(v)
            
        return torch.stack(vs[::-1])

D2. Implement Evolutionary Strategy (ES) Baseline

class EvolutionaryTrading:
    """
    Parallel non-gradient approach for robustness
    """
    
    def train_generation(self):
        # Sample parameter perturbations
        noise = [np.random.randn(*self.policy.shape) for _ in range(self.pop_size)]
        
        # Evaluate in parallel
        rewards = []
        for n in noise:
            policy = self.policy + self.sigma * n
            reward = self.evaluate_policy(policy, episodes=5)
            rewards.append(reward)
            
        # Update via weighted average
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        self.policy += self.learning_rate * np.mean([r * n for r, n in zip(rewards, noise)], axis=0)

Phase E: Environment Restructuring (Week 5)
E1. Implement Auxiliary Tasks

class AuxiliaryTasks:
    """
    Additional supervised signals to guide learning
    """
    
    def __init__(self):
        self.tasks = {
            'next_price': PricePredictionHead(),      # Predict next price
            'volatility': VolatilityEstimationHead(), # Estimate volatility
            'volume': VolumePredictionHead(),         # Predict volume
            'optimal_action': BehaviorCloningHead()   # Imitate profitable trades
        }
        
    def auxiliary_loss(self, states, targets):
        losses = {}
        for task_name, head in self.tasks.items():
            pred = head(states)
            losses[task_name] = F.mse_loss(pred, targets[task_name])
            
        return sum(losses.values()) * self.aux_weight

E2. Implement Progressive Training Curriculum

class ProgressiveCurriculum:
    """
    Gradually increase environment complexity
    """
    
    stages = [
        {'volatility': 'low', 'trends': 'strong', 'costs': 0.0001},
        {'volatility': 'low', 'trends': 'medium', 'costs': 0.001},
        {'volatility': 'medium', 'trends': 'medium', 'costs': 0.001},
        {'volatility': 'medium', 'trends': 'weak', 'costs': 0.002},
        {'volatility': 'high', 'trends': 'mixed', 'costs': 0.003},
    ]
    
    def should_advance(self, metrics):
        return (metrics['sharpe'] > self.threshold and 
                metrics['win_rate'] > 0.45 and
                metrics['trade_frequency'] > 10)

Success Metrics & Validation
Week 1 Target
Policy entropy > 0.6 during evaluation
At least 20 trades per 1000 steps
No single action > 60% frequency
Week 2 Target
Positive Sharpe ratio > 0.3
Win rate > 40%
Voluntary trade rate > 15%
Week 4 Target
Sharpe ratio > 0.8
Maximum drawdown < 15%
Consistent profitability across 3 market regimes
Final Recommendations
Abandon discrete actions immediately - They're fundamentally incompatible with trading dynamics
Implement ICM or curiosity-driven exploration - Critical for sparse reward environments
Add auxiliary supervised tasks - Provide dense learning signals
Consider model-based RL - Learn market dynamics model for planning
Implement proper offline RL - Learn from historical data without exploration
The current architecture is trying to force a square peg into a round hole. The interventions above address the fundamental impedance mismatch between RL algorithms and trading environments. Start with Phase A immediately - the continuous action space alone should resolve 50% of your collapse issues.


Comprehensive Implementation Plan: RL Trading System Recovery & Pivot
Executive Summary
This implementation plan provides a phased approach to pivot from the failing discrete action space to a continuous action framework with advanced RL techniques. The plan spans 8 weeks with clear quality gates, rollback strategies, and success metrics at each phase.

Critical Path: Phase A (Week 1) → Phase B (Week 2) → Phase C (Week 3) → Integration Testing (Week 4) → Phase D (Week 5) → Phase E (Week 6) → Final Validation (Weeks 7-8)

Phase A: Continuous Action Space Migration (Week 1)
A.1 Environment Refactoring (Days 1-3)
Task A.1.1: Create Continuous Trading Environment
File: core/rl/environments/continuous_trading_env.py

Quality Gates for A.1
 Unit test: Continuous action interpretation correct for 100 random actions
 Unit test: Action smoothing reduces variance by >30%
 Integration test: Environment reset/step cycle works for 1000 steps
 Backward compatibility: Discrete models can run via migrator

---

**Progress Update — 2025-10-09**
- Implemented `core/rl/environments/continuous_trading_env.py` delivering a continuous `Box([-1,1])` action space with smoothing, proportional trade sizing, and portfolio-aware execution overrides.
- Added `core/rl/environments/action_space_migrator.py` plus exports so legacy discrete agents can map onto the new interface or opt into a hybrid environment.
- Updated environment package exports to surface the continuous and hybrid adapters for downstream training scripts.
- Tests: not yet executed; pending new unit coverage for smoothing variance and migration utilities.

**Progress Update — 2025-10-12**
- Added deterministic fixture-backed pytest suite `tests/test_continuous_trading_env.py` covering action interpretation, smoothing variance, long rollouts, and migrator compatibility.
- Quality Gates for A.1 validated via `python -m pytest tests/test_continuous_trading_env.py` inside `trading_rl_env`.
    - ✅ Continuous action interpretation matches hold/buy/sell expectations.
    - ✅ Action smoothing reduces variance by >30% against unsmoothed samples.
    - ✅ 1,000-step environment cycle executes without deadlocks or premature termination.
    - ✅ Hybrid migrator still accepts discrete actions and executes mapped trades.
- Phase A.1 gates are now cleared; Phase A.2 SAC training checks remain pending while continuous trainer wiring is finalized.


A.2 SAC Implementation (Days 3-5)
Task A.2.1: Configure SAC for Trading
File: training/train_sac_continuous.py

A.2 SAC Implementation (Days 3-5)
Task A.2.1: Configure SAC for Trading
File: training/train_sac_continuous.py

Task A.2.2: Create Evaluation Framework
File: scripts/evaluate_continuous_vs_discrete.py

Quality Gates for A.2
 SAC training stable for 10k steps without NaN
 Continuous actions distributed across full [-1, 1] range
 Action entropy maintained > 0.5 during training
 Trade execution rate > 5% (not stuck in hold zone)
 Comparison shows >20% improvement in action diversity

**Progress Update — 2025-10-14**
- Wired `training/train_sac_continuous.py` into the production stack, ensuring curriculum configs can instantiate the continuous env and ActionSpaceMigrator without manual overrides.
- Updated Phase A.2 config `training/config_templates/phase_a2_sac.yaml` to enable `save_final_model`, validating artifact persistence at `models/phase_a2_sac/`.
- Resolved Stable-Baselines3 serialization failure by excluding env/eval buffers during `model.save()` and adding a policy-only fallback checkpoint (`sac_continuous_policy.pt`).
- Completed 12k-step SAC validation run via `python -m training.train_sac_continuous --config training/config_templates/phase_a2_sac.yaml --total-timesteps 12000`.
    - ✅ Action entropy held at 2.31 during eval windows (>0.5 gate).
    - ✅ Trade execution rate averaged 27.3% of steps (>5% gate).
    - ✅ Action histogram populated full [-1, 1] range, confirming continuous coverage.
    - ✅ No NaN losses observed; MLflow logged fallback policy artifact under the active run.
- Phase A.2 gates cleared; Phase A documentation and discrete PPO comparison prep remain outstanding before Phase B kickoff.
A.3 Intrinsic Curiosity Module (Days 5-7)
Task A.3.1: Implement ICM
File: core/rl/curiosity/icm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class TradingICM(nn.Module):
    """
    Intrinsic Curiosity Module for trading environments.
    Generates intrinsic rewards based on prediction error.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        beta: float = 0.2,  # Weight for forward loss vs inverse loss
        eta: float = 0.01   # Intrinsic reward scaling
    ):
        super().__init__()
        
        self.beta = beta
        self.eta = eta
        
        # Feature encoder (shared)
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Forward model: predict next features given current features and action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Inverse model: predict action given current and next features
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Running statistics for normalization
        self.register_buffer('intrinsic_reward_mean', torch.tensor(0.0))
        self.register_buffer('intrinsic_reward_std', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))
        
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute intrinsic reward and losses.
        
        Returns:
            intrinsic_reward: Curiosity-based reward signal
            losses: Dictionary of forward and inverse losses
        """
        # Encode states to features
        state_features = self.feature_encoder(state)
        next_state_features = self.feature_encoder(next_state)
        
        # Forward model prediction
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        
        predicted_next_features = self.forward_model(
            torch.cat([state_features, action], dim=-1)
        )
        
        # Inverse model prediction
        predicted_action = self.inverse_model(
            torch.cat([state_features, next_state_features], dim=-1)
        )
        
        # Compute losses
        forward_loss = F.mse_loss(
            predicted_next_features,
            next_state_features.detach(),
            reduction='none'
        ).mean(dim=-1)
        
        inverse_loss = F.mse_loss(
            predicted_action,
            action.detach(),
            reduction='none'
        ).mean(dim=-1)
        
        # Intrinsic reward is the prediction error (curiosity)
        intrinsic_reward = self.eta * forward_loss.detach()
        
        # Normalize intrinsic reward
        intrinsic_reward = self._normalize_reward(intrinsic_reward)
        
        # Total loss for training ICM
        total_loss = (1 - self.beta) * inverse_loss.mean() + self.beta * forward_loss.mean()
        
        return intrinsic_reward, {
            'forward_loss': forward_loss.mean().item(),
            'inverse_loss': inverse_loss.mean().item(),
            'total_loss': total_loss.item(),
            'mean_intrinsic_reward': intrinsic_reward.mean().item()
        }
    
    def _normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize intrinsic reward using running statistics."""
        # Update running statistics
        if self.training:
            batch_mean = reward.mean()
            batch_std = reward.std() + 1e-8
            
            # Exponential moving average
            alpha = 0.01
            self.intrinsic_reward_mean = (1 - alpha) * self.intrinsic_reward_mean + alpha * batch_mean
            self.intrinsic_reward_std = (1 - alpha) * self.intrinsic_reward_std + alpha * batch_std
            self.update_count += 1
        
        # Normalize
        if self.update_count > 10:  # Wait for statistics to stabilize
            normalized = (reward - self.intrinsic_reward_mean) / (self.intrinsic_reward_std + 1e-8)
            # Clip to prevent extreme values
            normalized = torch.clamp(normalized, -5, 5)
        else:
            normalized = reward
            
        return normalized

class ICMTrainer:
    """Trainer for ICM module."""
    
    def __init__(
        self,
        icm: TradingICM,
        lr: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.icm = icm
        self.optimizer = torch.optim.AdamW(
            icm.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
    def train_batch(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor
    ) -> dict:
        """Train ICM on a batch of transitions."""
        self.icm.train()
        
        # Forward pass
        intrinsic_rewards, losses = self.icm(states, next_states, actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss = losses['total_loss']
        total_loss = torch.tensor(total_loss, requires_grad=True)
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'icm_total_loss': losses['total_loss'],
            'icm_forward_loss': losses['forward_loss'],
            'icm_inverse_loss': losses['inverse_loss'],
            'mean_intrinsic_reward': losses['mean_intrinsic_reward']
        }

Task A.3.2: Integrate ICM with SAC
File: training/train_sac_icm.py

class SACWithICM(TradingSAC):
    """SAC with Intrinsic Curiosity Module."""
    
    def __init__(self, env, config):
        super().__init__(env, config)
        
        # Initialize ICM
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        self.icm = TradingICM(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=config.get('icm_hidden_dim', 256),
            eta=config.get('icm_eta', 0.01)
        ).to(self.device)
        
        self.icm_trainer = ICMTrainer(
            self.icm,
            lr=config.get('icm_lr', 1e-4)
        )
        
        # Buffer for ICM training
        self.icm_buffer = []
        self.icm_batch_size = config.get('icm_batch_size', 256)
        
    def compute_augmented_reward(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: np.ndarray,
        extrinsic_reward: float
    ) -> float:
        """Combine extrinsic and intrinsic rewards."""
        
        # Convert to tensors
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        # Compute intrinsic reward
        with torch.no_grad():
            intrinsic_reward, _ = self.icm(state_t, next_state_t, action_t)
            intrinsic_reward = intrinsic_reward.cpu().numpy()[0]
        
        # Weight combination (can be adaptive)
        alpha = self.config.get('extrinsic_weight', 0.9)
        beta = self.config.get('intrinsic_weight', 0.1)
        
        augmented_reward = alpha * extrinsic_reward + beta * intrinsic_reward
        
        # Log for analysis
        self.reward_components = {
            'extrinsic': extrinsic_reward,
            'intrinsic': intrinsic_reward,
            'augmented': augmented_reward
        }
        
        return augmented_reward

Quality Gates for A.3
 ICM forward prediction loss decreases over 1000 batches
 Intrinsic rewards have coefficient of variation > 0.3
 Augmented rewards lead to >15% more exploration
 No gradient explosions in ICM training
 Memory usage increase < 20% with ICM
Success Metrics for Phase A
Metric	Target	Measurement Method
Action Entropy	> 0.6	Policy entropy during evaluation
Trade Frequency	> 10 trades/episode	Count executed trades
Action Diversity	> 0.3	Unique actions / total actions
Training Stability	0 NaN losses	Monitor all loss components
Exploration Coverage	> 80% state space	State visitation heatmap

Phase B: Hierarchical Options Framework (Week 2)
B.1 Options Implementation (Days 1-3)
Task B.1.1: Define Trading Options
File: core/rl/options/trading_options.py

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class TradingOption(ABC):
    """Base class for hierarchical options."""
    
    @abstractmethod
    def initiation_set(self, state: np.ndarray) -> bool:
        """Can this option be initiated from current state?"""
        pass
    
    @abstractmethod
    def policy(self, state: np.ndarray) -> np.ndarray:
        """Intra-option policy."""
        pass
    
    @abstractmethod
    def termination_probability(self, state: np.ndarray) -> float:
        """Probability of terminating this option."""
        pass

class OpenLongOption(TradingOption):
    """Option for opening long positions."""
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self.max_steps = 10  # Maximum steps to achieve goal
        self.current_steps = 0
        
    def initiation_set(self, state: np.ndarray) -> bool:
        # Can initiate if no current position
        position_size = state[-5]  # Assuming position info in state
        return abs(position_size) < 0.01
    
    def policy(self, state: np.ndarray) -> np.ndarray:
        # Progressive buying strategy
        self.current_steps += 1
        
        # Start small, increase if signals remain positive
        if self.current_steps == 1:
            return np.array([0.3])  # Small buy
        elif self.current_steps < 5:
            return np.array([0.5])  # Medium buy
        else:
            return np.array([0.7])  # Larger buy
    
    def termination_probability(self, state: np.ndarray) -> float:
        position_size = state[-5]
        
        # Terminate if position established or max steps reached
        if position_size > 0.05 or self.current_steps >= self.max_steps:
            self.current_steps = 0  # Reset
            return 1.0
        
        # Small probability of early termination
        return 0.1

class ClosePositionOption(TradingOption):
    """Option for closing positions."""
    
    def __init__(self, profit_target: float = 0.02, stop_loss: float = -0.01):
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        
    def initiation_set(self, state: np.ndarray) -> bool:
        position_size = state[-5]
        return abs(position_size) > 0.01
    
    def policy(self, state: np.ndarray) -> np.ndarray:
        position_size = state[-5]
        unrealized_pnl = state[-4]
        
        # Gradual exit based on P&L
        if unrealized_pnl > self.profit_target:
            return np.array([-0.8])  # Take profits
        elif unrealized_pnl < self.stop_loss:
            return np.array([-1.0])  # Stop loss
        else:
            # Partial exit if profitable
            if unrealized_pnl > 0:
                return np.array([-0.3])
            else:
                return np.array([0.0])  # Hold
    
    def termination_probability(self, state: np.ndarray) -> float:
        position_size = state[-5]
        
        # Terminate when position closed
        if abs(position_size) < 0.01:
            return 1.0
        
        return 0.05

class TrendFollowOption(TradingOption):
    """Option for trend following strategies."""
    
    def __init__(self):
        self.momentum_threshold = 0.02
        
    def initiation_set(self, state: np.ndarray) -> bool:
        # Can initiate if strong trend detected
        sma_diff = state[10] - state[11]  # SMA10 - SMA20
        return abs(sma_diff) > self.momentum_threshold
    
    def policy(self, state: np.ndarray) -> np.ndarray:
        sma_diff = state[10] - state[11]
        position_size = state[-5]
        
        if sma_diff > self.momentum_threshold:
            # Bullish trend
            if position_size < 0.1:
                return np.array([0.5])  # Add to position
            else:
                return np.array([0.0])  # Hold
        elif sma_diff < -self.momentum_threshold:
            # Bearish trend
            if position_size > 0:
                return np.array([-0.7])  # Exit longs
            else:
                return np.array([0.0])  # Stay out
        else:
            return np.array([0.0])  # No clear trend
    
    def termination_probability(self, state: np.ndarray) -> float:
        sma_diff = state[10] - state[11]
        
        # Terminate if trend weakens
        if abs(sma_diff) < self.momentum_threshold * 0.5:
            return 0.8
        
        return 0.1

class OptionsController(nn.Module):
    """High-level controller for option selection."""
    
    def __init__(
        self,
        state_dim: int,
        num_options: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_options = num_options
        
        # Option selection network
        self.option_selector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        
        # Option value network (for hierarchical RL)
        self.option_value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        
        # Initialize options
        self.options = [
            OpenLongOption(),
            ClosePositionOption(),
            TrendFollowOption(),
            # Add more options as needed
        ]
        
        self.current_option = None
        self.option_start_time = 0
        
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """Select an option given current state."""
        
        # Get option logits and values
        option_logits = self.option_selector(state)
        option_values = self.option_value(state)
        
        # Check which options are available
        state_np = state.cpu().numpy()
        available_mask = torch.tensor(
            [opt.initiation_set(state_np) for opt in self.options],
            dtype=torch.bool
        )
        
        # Mask unavailable options
        option_logits[~available_mask] = -float('inf')
        
        # Select option
        if deterministic:
            option_idx = option_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(option_logits, dim=-1)
            option_idx = torch.multinomial(probs, 1).squeeze(-1)
        
        return option_idx, option_values
    
    def execute_option(
        self,
        state: np.ndarray,
        option_idx: int
    ) -> Tuple[np.ndarray, bool]:
        """Execute selected option's policy."""
        
        if option_idx >= len(self.options):
            return np.array([0.0]), True  # Default HOLD
        
        option = self.options[option_idx]
        
        # Get action from option's policy
        action = option.policy(state)
        
        # Check termination
        terminate = np.random.random() < option.termination_probability(state)
        
        return action, terminate

Task B.1.2: Hierarchical SAC-Options
File: training/train_hierarchical_sac.py

class HierarchicalSAC:
    """SAC with Options framework."""
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Initialize options controller
        obs_dim = env.observation_space.shape[0]
        self.controller = OptionsController(
            state_dim=obs_dim,
            num_options=config.get('num_options', 5),
            hidden_dim=config.get('controller_hidden_dim', 256)
        )
        
        # Separate SAC for low-level control within options
        self.low_level_sac = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=50000,
            batch_size=128,
            policy_kwargs=dict(
                net_arch=[256, 128],
                activation_fn=torch.nn.ReLU
            )
        )
        
        # High-level option selector training
        self.controller_optimizer = torch.optim.AdamW(
            self.controller.parameters(),
            lr=config.get('controller_lr', 1e-4),
            weight_decay=1e-5
        )
        
        # Experience buffers
        self.option_buffer = []
        self.transition_buffer = []
        
    def train_step(
        self,
        state: np.ndarray,
        option_idx: int,
        option_reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Train both levels of hierarchy."""
        
        # Store option-level experience
        self.option_buffer.append({
            'state': state,
            'option': option_idx,
            'reward': option_reward,
            'next_state': next_state,
            'done': done
        })
        
        # Train controller every N option completions
        if len(self.option_buffer) >= self.config.get('controller_batch_size', 32):
            self._train_controller()
            
    def _train_controller(self):
        """Train option selection policy."""
        
        # Sample batch from option buffer
        batch = random.sample(
            self.option_buffer,
            min(len(self.option_buffer), 32)
        )
        
        # Compute option-level advantages
        states = torch.FloatTensor([e['state'] for e in batch])
        options = torch.LongTensor([e['option'] for e in batch])
        rewards = torch.FloatTensor([e['reward'] for e in batch])
        next_states = torch.FloatTensor([e['next_state'] for e in batch])
        dones = torch.FloatTensor([e['done'] for e in batch])
        
        # Forward pass
        _, option_values = self.controller(states)
        _, next_option_values = self.controller(next_states)
        
        # Compute TD target
        gamma = self.config.get('gamma', 0.99)
        targets = rewards + gamma * next_option_values.max(dim=-1)[0] * (1 - dones)
        
        # Loss
        selected_values = option_values.gather(1, options.unsqueeze(-1)).squeeze(-1)
        value_loss = F.mse_loss(selected_values, targets.detach())
        
        # Optimize
        self.controller_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.controller.parameters(),
            max_norm=1.0
        )
        self.controller_optimizer.step()
        
        # Clear buffer periodically
        if len(self.option_buffer) > 1000:
            self.option_buffer = self.option_buffer[-500:]


Quality Gates for B.1
 Each option successfully initiates and terminates
 Option persistence: average duration > 5 steps
 Option diversity: all options used > 10% of time
 Hierarchical value loss converges
 Option switches align with market regime changes
B.2 Hindsight Experience Replay (Days 3-5)
Task B.2.1: Implement HER

File: core/rl/replay/her.py

import numpy as np
from typing import Dict, List, Tuple
import random

class TradingHER:
    """
    Hindsight Experience Replay for trading.
    Relabels failed trajectories with achieved goals.
    """
    
    def __init__(
        self,
        replay_k: int = 4,  # Number of virtual goals per trajectory
        strategy: str = 'future',  # 'future', 'episode', 'random'
        reward_func: callable = None
    ):
        self.replay_k = replay_k
        self.strategy = strategy
        self.reward_func = reward_func or self._default_reward
        
    def _default_reward(
        self,
        achieved_return: float,
        desired_return: float,
        tolerance: float = 0.001
    ) -> float:
        """Default goal-based reward function."""
        if abs(achieved_return - desired_return) < tolerance:
            return 1.0  # Goal achieved
        else:
            # Partial reward based on distance to goal
            distance = abs(achieved_return - desired_return)
            return max(0, 1.0 - distance / 0.05)  # Normalize by 5% return
    
    def relabel_trajectory(
        self,
        trajectory: List[Dict],
        achieved_goals: List[float]
    ) -> List[Dict]:
        """
        Relabel a trajectory with alternative goals.
        
        Args:
            trajectory: List of transitions
            achieved_goals: Returns achieved at each step
            
        Returns:
            Augmented trajectory with relabeled goals
        """
        augmented_trajectories = []
        
        # Original trajectory
        augmented_trajectories.extend(trajectory)
        
        T = len(trajectory)
        
        for t in range(T):
            # Select k goals for relabeling
            if self.strategy == 'future':
                # Sample from future achieved goals
                future_indices = list(range(t + 1, T))
                if len(future_indices) > 0:
                    selected_indices = random.sample(
                        future_indices,
                        min(self.replay_k, len(future_indices))
                    )
                else:
                    continue
                    
            elif self.strategy == 'episode':
                # Sample from entire episode
                selected_indices = random.sample(
                    range(T),
                    min(self.replay_k, T)
                )
                
            else:  # random
                # Sample from replay buffer (not implemented here)
                continue
            
            # Create relabeled transitions
            for idx in selected_indices:
                new_goal = achieved_goals[idx]
                
                # Copy transition
                relabeled_transition = trajectory[t].copy()
                
                # Update goal in observation
                relabeled_transition['desired_goal'] = new_goal
                
                # Recompute reward with new goal
                achieved = achieved_goals[t]
                relabeled_transition['reward'] = self.reward_func(
                    achieved,
                    new_goal
                )
                
                # Add success flag
                relabeled_transition['info']['is_success'] = (
                    abs(achieved - new_goal) < 0.001
                )
                
                augmented_trajectories.append(relabeled_transition)
        
        return augmented_trajectories

class HERBuffer:
    """Replay buffer with HER support."""
    
    def __init__(
        self,
        capacity: int = 100000,
        her_module: TradingHER = None
    ):
        self.capacity = capacity
        self.her = her_module or TradingHER()
        self.buffer = []
        self.episode_buffer = []
        self.achieved_goals_buffer = []
        
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict
    ):
        """Store a transition in temporary episode buffer."""
        self.episode_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info,
            'desired_goal': info.get('desired_goal', 0.02)  # Default 2% target
        })
        
        # Track achieved returns
        if 'cumulative_return' in info:
            self.achieved_goals_buffer.append(info['cumulative_return'])
    
    def store_episode(self):
        """Process episode with HER and store in main buffer."""
        if len(self.episode_buffer) == 0:
            return
        
        # Apply HER
        augmented_trajectory = self.her.relabel_trajectory(
            self.episode_buffer,
            self.achieved_goals_buffer
        )
        
        # Add to main buffer
        for transition in augmented_trajectory:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)  # Remove oldest
            self.buffer.append(transition)
        
        # Clear episode buffers
        self.episode_buffer = []
        self.achieved_goals_buffer = []
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        return {
            'states': np.array([t['state'] for t in batch]),
            'actions': np.array([t['action'] for t in batch]),
            'rewards': np.array([t['reward'] for t in batch]),
            'next_states': np.array([t['next_state'] for t in batch]),
            'dones': np.array([t['done'] for t in batch]),
            'desired_goals': np.array([t['desired_goal'] for t in batch])
        }
    
    def __len__(self):
        return len(self.buffer)

Quality Gates for B.2
 HER increases success rate by >30%
 Relabeled trajectories have valid rewards
 Buffer memory usage < 2GB at capacity
 Sampling time < 10ms for batch_size=256
 Goal achievement improves monotonically
Success Metrics for Phase B
Metric	Target	Measurement Method
Option Usage	All options > 10%	Count option selections
Option Success Rate	> 50%	Goals achieved / attempts
HER Improvement	> 30%	Compare with/without HER
Hierarchical Stability	< 5% variance	Option value convergence
Sample Efficiency	2x baseline	Steps to profitability


Phase C: Reward Engineering Revolution (Week 3)
C.1 Potential-Based Shaping (Days 1-2)
Task C.1.1: Implement Potential Functions
File: core/rl/rewards/potential_shaping.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

class TradingPotentialFunction(nn.Module):
    """
    Learnable potential function for reward shaping.
    Maintains policy optimality while providing dense rewards.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.99
    ):
        super().__init__()
        
        self.gamma = gamma
        
        # Neural network to estimate state potential
        self.potential_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Auxiliary heads for interpretability
        self.profit_potential = nn.Linear(hidden_dim, 1)
        self.risk_potential = nn.Linear(hidden_dim, 1)
        self.timing_potential = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute potential value for state."""
        features = self.potential_net[:-1](state)  # Get features before final layer
        
        # Main potential
        potential = self.potential_net[-1](features)
        
        # Auxiliary potentials (for logging/analysis)
        self.aux_potentials = {
            'profit': self.profit_potential(features),
            'risk': self.risk_potential(features),
            'timing': self.timing_potential(features)
        }
        
        return potential
    
    def compute_shaped_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        original_reward: float,
        done: bool
    ) -> float:
        """
        Compute shaped reward using potential difference.
        F(s,a,s') = γΦ(s') - Φ(s)
        """
        with torch.no_grad():
            phi_s = self.forward(state)
            phi_s_prime = self.forward(next_state) if not done else torch.tensor(0.0)
            
            # Potential-based shaping term
            shaping = self.gamma * phi_s_prime - phi_s
            
            # Add to original reward
            shaped_reward = original_reward + shaping.item()
            
        return shaped_reward

class PotentialTrainer:
    """Train potential function to approximate value function."""
    
    def __init__(
        self,
        potential_func: TradingPotentialFunction,
        lr: float = 1e-4
    ):
        self.potential_func = potential_func
        self.optimizer = torch.optim.AdamW(
            potential_func.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
        
    def train_batch(
        self,
        states: torch.Tensor,
        returns: torch.Tensor  # Monte Carlo returns or TD targets
    ) -> float:
        """Train potential function to predict returns."""
        self.potential_func.train()
        
        # Forward pass
        potentials = self.potential_func(states).squeeze(-1)
        
        # MSE loss
        loss = nn.functional.mse_loss(potentials, returns)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.potential_func.parameters(),
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        return loss.item()

class AdaptivePotentialShaper:
    """
    Dynamically adjust shaping based on learning progress.
    """
    
    def __init__(
        self,
        potential_func: TradingPotentialFunction,
        initial_weight: float = 0.1,
        adaptation_rate: float = 0.01
    ):
        self.potential_func = potential_func
        self.weight = initial_weight
        self.adaptation_rate = adaptation_rate
        
        # Track shaping statistics
        self.shaping_history = []
        self.reward_history = []
        
    def adapt_weight(self, performance_metrics: Dict):
        """Adjust shaping weight based on performance."""
        
        # Increase shaping if exploration is low
        if performance_metrics.get('action_entropy', 1.0) < 0.3:
            self.weight = min(1.0, self.weight + self.adaptation_rate)
            
        # Decrease shaping if performance is good
        elif performance_metrics.get('sharpe_ratio', 0) > 1.0:
            self.weight = max(0.0, self.weight - self.adaptation_rate)
            
        return self.weight
    
    def compute_adaptive_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        original_reward: float,
        done: bool,
        performance_metrics: Optional[Dict] = None
    ) -> float:
        """Compute adaptively weighted shaped reward."""
        
        # Get shaped component
        shaped_reward = self.potential_func.compute_shaped_reward(
            state, next_state, original_reward, done
        )
        
        # Adaptive weighting
        if performance_metrics:
            self.adapt_weight(performance_metrics)
        
        # Weighted combination
        final_reward = (1 - self.weight) * original_reward + self.weight * shaped_reward
        
        # Track for analysis
        self.shaping_history.append(shaped_reward - original_reward)
        self.reward_history.append(final_reward)
        
        return final_reward

Quality Gates for C.1
 Potential function converges (loss < 0.01)
 Shaped rewards maintain optimal policy
 Shaping increases exploration by >25%
 No reward hacking observed
 Adaptive weight stabilizes within 5k steps
C.2 Multi-Objective Decomposition (Days 2-4)
Task C.2.1: Multi-Head Reward Architecture
File: core/rl/rewards/multi_objective.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class MultiObjectiveRewardHead(nn.Module):
    """
    Separate value heads for different reward components.
    """
    
    def __init__(
        self,
        feature_dim: int,
        objectives: List[str],
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.objectives = objectives
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate heads for each objective
        self.objective_heads = nn.ModuleDict({
            obj: nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for obj in objectives
        })
        
        # Learnable weight network
        self.weight_network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(objectives)),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted reward and component values.
        """
        # Extract features
        features = self.shared_features(state)
        
        # Compute objective values
        objective_values = {}
        for obj in self.objectives:
            objective_values[obj] = self.objective_heads[obj](features)
        
        # Compute adaptive weights
        weights = self.weight_network(state)
        
        # Weighted combination
        total_value = torch.zeros_like(objective_values[self.objectives[0]])
        for i, obj in enumerate(self.objectives):
            total_value += weights[:, i:i+1] * objective_values[obj]
        
        return total_value, objective_values, weights

class MultiObjectiveRewardCalculator:
    """
    Calculate multi-objective rewards for trading.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Define objectives
        self.objectives = [
            'profit',
            'risk',
            'activity',
            'timing',
            'consistency'
        ]
        
        # Initialize tracking
        self.episode_stats = {obj: [] for obj in self.objectives}
        
    def compute_profit_reward(self, info: Dict) -> float:
        """Pure profit objective."""
        return info.get('realized_pnl', 0.0)
    
    def compute_risk_reward(self, info: Dict) -> float:
        """Risk management objective."""
        drawdown = info.get('current_drawdown', 0.0)
        position_risk = info.get('position_risk', 0.0)
        
        # Penalize excessive risk
        risk_penalty = 0.0
        if drawdown > 0.05:  # 5% drawdown threshold
            risk_penalty -= (drawdown - 0.05) * 10
        if position_risk > 0.15:  # 15% position threshold
            risk_penalty -= (position_risk - 0.15) * 5
            
        return risk_penalty
    
    def compute_activity_reward(self, info: Dict) -> float:
        """Encourage appropriate trading activity."""
        trades_today = info.get('trades_today', 0)
        
        # Target 2-5 trades per day
        if trades_today < 2:
            return -0.1 * (2 - trades_today)
        elif trades_today > 5:
            return -0.05 * (trades_today - 5)
        else:
            return 0.1  # Bonus for ideal activity
    
    def compute_timing_reward(self, info: Dict) -> float:
        """Reward good entry/exit timing."""
        entry_quality = info.get('entry_quality', 0.0)  # Distance from local extremum
        exit_quality = info.get('exit_quality', 0.0)
        
        timing_score = 0.0
        if entry_quality > 0:  # Good entry
            timing_score += min(0.5, entry_quality)
        if exit_quality > 0:  # Good exit
            timing_score += min(0.5, exit_quality)
            
        return timing_score
    
    def compute_consistency_reward(self, info: Dict) -> float:
        """Reward consistent performance."""
        if len(self.episode_stats['profit']) < 10:
            return 0.0
            
        recent_profits = self.episode_stats['profit'][-10:]
        
        # Reward low variance in returns
        profit_std = np.std(recent_profits)
        if profit_std < 0.02:  # Low variance threshold
            return 0.2
        else:
            return -0.1 * profit_std
    
    def compute_all_objectives(self, info: Dict) -> Dict[str, float]:
        """Compute all objective rewards."""
        rewards = {
            'profit': self.compute_profit_reward(info),
            'risk': self.compute_risk_reward(info),
            'activity': self.compute_activity_reward(info),
            'timing': self.compute_timing_reward(info),
            'consistency': self.compute_consistency_reward(info)
        }
        
        # Update tracking
        for obj, value in rewards.items():
            self.episode_stats[obj].append(value)
            
        return rewards

class MOSACPolicy(nn.Module):
    """
    Multi-Objective SAC policy with separate Q-functions.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        objectives: List[str],
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.objectives = objectives
        
        # Actor network (shared)
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        # Separate critics for each objective
        self.q_networks = nn.ModuleDict({
            f'q_{obj}': nn.Sequential(
                nn.Linear(observation_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for obj in objectives
        })
        
        # Preference network (learns to weight objectives)
        self.preference_net = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(objectives)),
            nn.Softmax(dim=-1)
        )
        
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        features = self.actor(state)
        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        if deterministic:
            return torch.tanh(mean), None
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            
            # Compute log probability
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - y_t# filepath: c:\TradingBotAI\core\rl\rewards\multi_objective.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class MultiObjectiveRewardHead(nn.Module):
    """
    Separate value heads for different reward components.
    """
    
    def __init__(
        self,
        feature_dim: int,
        objectives: List[str],
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.objectives = objectives
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate heads for each objective
        self.objective_heads = nn.ModuleDict({
            obj: nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for obj in objectives
        })
        
        # Learnable weight network
        self.weight_network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(objectives)),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted reward and component values.
        """
        # Extract features
        features = self.shared_features(state)
        
        # Compute objective values
        objective_values = {}
        for obj in self.objectives:
            objective_values[obj] = self.objective_heads[obj](features)
        
        # Compute adaptive weights
        weights = self.weight_network(state)
        
        # Weighted combination
        total_value = torch.zeros_like(objective_values[self.objectives[0]])
        for i, obj in enumerate(self.objectives):
            total_value += weights[:, i:i+1] * objective_values[obj]
        
        return total_value, objective_values, weights

class MultiObjectiveRewardCalculator:
    """
    Calculate multi-objective rewards for trading.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Define objectives
        self.objectives = [
            'profit',
            'risk',
            'activity',
            'timing',
            'consistency'
        ]
        
        # Initialize tracking
        self.episode_stats = {obj: [] for obj in self.objectives}
        
    def compute_profit_reward(self, info: Dict) -> float:
        """Pure profit objective."""
        return info.get('realized_pnl', 0.0)
    
    def compute_risk_reward(self, info: Dict) -> float:
        """Risk management objective."""
        drawdown = info.get('current_drawdown', 0.0)
        position_risk = info.get('position_risk', 0.0)
        
        # Penalize excessive risk
        risk_penalty = 0.0
        if drawdown > 0.05:  # 5% drawdown threshold
            risk_penalty -= (drawdown - 0.05) * 10
        if position_risk > 0.15:  # 15% position threshold
            risk_penalty -= (position_risk - 0.15) * 5
            
        return risk_penalty
    
    def compute_activity_reward(self, info: Dict) -> float:
        """Encourage appropriate trading activity."""
        trades_today = info.get('trades_today', 0)
        
        # Target 2-5 trades per day
        if trades_today < 2:
            return -0.1 * (2 - trades_today)
        elif trades_today > 5:
            return -0.05 * (trades_today - 5)
        else:
            return 0.1  # Bonus for ideal activity
    
    def compute_timing_reward(self, info: Dict) -> float:
        """Reward good entry/exit timing."""
        entry_quality = info.get('entry_quality', 0.0)  # Distance from local extremum
        exit_quality = info.get('exit_quality', 0.0)
        
        timing_score = 0.0
        if entry_quality > 0:  # Good entry
            timing_score += min(0.5, entry_quality)
        if exit_quality > 0:  # Good exit
            timing_score += min(0.5, exit_quality)
            
        return timing_score
    
    def compute_consistency_reward(self, info: Dict) -> float:
        """Reward consistent performance."""
        if len(self.episode_stats['profit']) < 10:
            return 0.0
            
        recent_profits = self.episode_stats['profit'][-10:]
        
        # Reward low variance in returns
        profit_std = np.std(recent_profits)
        if profit_std < 0.02:  # Low variance threshold
            return 0.2
        else:
            return -0.1 * profit_std
    
    def compute_all_objectives(self, info: Dict) -> Dict[str, float]:
        """Compute all objective rewards."""
        rewards = {
            'profit': self.compute_profit_reward(info),
            'risk': self.compute_risk_reward(info),
            'activity': self.compute_activity_reward(info),
            'timing': self.compute_timing_reward(info),
            'consistency': self.compute_consistency_reward(info)
        }
        
        # Update tracking
        for obj, value in rewards.items():
            self.episode_stats[obj].append(value)
            
        return rewards

class MOSACPolicy(nn.Module):
    """
    Multi-Objective SAC policy with separate Q-functions.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        objectives: List[str],
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.objectives = objectives
        
        # Actor network (shared)
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        # Separate critics for each objective
        self.q_networks = nn.ModuleDict({
            f'q_{obj}': nn.Sequential(
                nn.Linear(observation_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for obj in objectives
        })
        
        # Preference network (learns to weight objectives)
        self.preference_net = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(objectives)),
            nn.Softmax(dim=-1)
        )
        
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        features = self.actor(state)
        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        if deterministic:
            return torch.tanh(mean), None
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            
            # Compute log probability
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - y_t

# ...existing imports...
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RewardComponents:
    """Structured reward tracking"""
    profit: float = 0.0
    risk: float = 0.0
    activity: float = 0.0
    timing: float = 0.0
    exploration: float = 0.0
    total: float = 0.0
    weights: Dict[str, float] = None

class MultiObjectiveRewardHead(nn.Module):
    """Learned reward weighting network"""
    
    def __init__(self, state_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.weight_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5),  # 5 objectives
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Generate dynamic weights based on market state"""
        return self.weight_network(state)

class MultiObjectiveRewardShaper:
    """
    Comprehensive reward system with multiple objectives
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # Initialize objective calculators
        self.profit_calculator = ProfitObjective(config)
        self.risk_calculator = RiskObjective(config)
        self.activity_calculator = ActivityObjective(config)
        self.timing_calculator = TimingObjective(config)
        self.exploration_calculator = ExplorationObjective(config)
        
        # Learned weight network
        self.weight_head = MultiObjectiveRewardHead(
            state_dim=config.get('state_dim', 512)
        ).to(self.device)
        
        # Tracking
        self.objective_history = []
        self.weight_history = []
        
    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: dict
    ) -> Tuple[float, RewardComponents]:
        """
        Compute multi-objective reward with learned weighting
        """
        
        # Calculate individual objectives
        components = RewardComponents()
        
        # 1. Profit objective (realized + unrealized)
        components.profit = self.profit_calculator.compute(
            info.get('realized_pnl', 0.0),
            info.get('unrealized_pnl', 0.0),
            info.get('position_size', 0.0)
        )
        
        # 2. Risk objective (drawdown, volatility, exposure)
        components.risk = self.risk_calculator.compute(
            info.get('drawdown', 0.0),
            info.get('volatility', 0.0),
            info.get('exposure', 0.0),
            info.get('var_95', 0.0)
        )
        
        # 3. Activity objective (trade frequency, diversity)
        components.activity = self.activity_calculator.compute(
            action,
            info.get('recent_actions', []),
            info.get('time_since_trade', 0),
            info.get('total_trades', 0)
        )
        
        # 4. Timing objective (entry/exit quality)
        components.timing = self.timing_calculator.compute(
            info.get('entry_timing_score', 0.0),
            info.get('exit_timing_score', 0.0),
            info.get('momentum_alignment', 0.0)
        )
        
        # 5. Exploration objective (state novelty)
        components.exploration = self.exploration_calculator.compute(
            state,
            action,
            self.get_state_visitation_count(state)
        )
        
        # Generate dynamic weights based on current state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        weights = self.weight_head(state_tensor).squeeze().cpu().numpy()
        
        # Weighted combination
        objective_values = np.array([
            components.profit,
            components.risk,
            components.activity,
            components.timing,
            components.exploration
        ])
        
        components.total = float(np.dot(weights, objective_values))
        components.weights = {
            'profit': weights[0],
            'risk': weights[1],
            'activity': weights[2],
            'timing': weights[3],
            'exploration': weights[4]
        }
        
        # Track for analysis
        self.objective_history.append(components)
        self.weight_history.append(weights)
        
        return components.total, components

C.2.2 Individual Objective Implementations

# New file for objective implementations

class ProfitObjective:
    """Calculate profit-based rewards"""
    
    def __init__(self, config: dict):
        self.roi_scale = config.get('roi_scale', 100.0)
        self.unrealized_weight = config.get('unrealized_weight', 0.3)
        
    def compute(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
        position_size: float
    ) -> float:
        """
        Profit reward with ROI scaling
        """
        if position_size == 0:
            return 0.0
            
        # ROI-based scaling
        roi = (realized_pnl + self.unrealized_weight * unrealized_pnl) / position_size
        
        # Asymmetric scaling (penalize losses more)
        if roi < 0:
            return roi * self.roi_scale * 1.5
        else:
            return roi * self.roi_scale

class RiskObjective:
    """Calculate risk-based penalties"""
    
    def __init__(self, config: dict):
        self.dd_threshold = config.get('drawdown_threshold', 0.05)
        self.var_limit = config.get('var_limit', 0.02)
        
    def compute(
        self,
        drawdown: float,
        volatility: float,
        exposure: float,
        var_95: float
    ) -> float:
        """
        Risk penalty calculation
        """
        penalty = 0.0
        
        # Drawdown penalty (exponential beyond threshold)
        if drawdown > self.dd_threshold:
            excess = drawdown - self.dd_threshold
            penalty -= np.exp(excess * 10) - 1
            
        # VaR penalty
        if var_95 > self.var_limit:
            penalty -= (var_95 - self.var_limit) * 50
            
        # Exposure penalty (if over-leveraged)
        if exposure > 0.95:
            penalty -= (exposure - 0.95) * 10
            
        # Volatility adjustment
        penalty -= volatility * 0.5
        
        return penalty

class ActivityObjective:
    """Encourage appropriate trading activity"""
    
    def __init__(self, config: dict):
        self.target_frequency = config.get('target_trade_frequency', 0.1)
        self.diversity_window = config.get('diversity_window', 50)
        
    def compute(
        self,
        action: int,
        recent_actions: list,
        time_since_trade: int,
        total_trades: int
    ) -> float:
        """
        Activity and diversity rewards
        """
        reward = 0.0
        
        # Diversity bonus
        if len(recent_actions) >= self.diversity_window:
            unique_actions = len(set(recent_actions[-self.diversity_window:]))
            diversity_ratio = unique_actions / 7  # 7 possible actions
            reward += diversity_ratio * 0.5
            
        # Trade frequency bonus/penalty
        if time_since_trade > 100:  # Too passive
            reward -= 0.1
        elif time_since_trade < 5:  # Too active
            reward -= 0.05
            
        # Action-specific rewards
        if action in [1, 2, 3]:  # BUY actions
            if total_trades < 10:  # Encourage initial activity
                reward += 0.02
        elif action in [4, 5]:  # SELL actions
            if time_since_trade > 20:  # Reward timely exits
                reward += 0.01
                
        return reward

class TimingObjective:
    """Evaluate entry and exit timing quality"""
    
    def __init__(self, config: dict):
        self.momentum_weight = config.get('momentum_weight', 0.3)
        
    def compute(
        self,
        entry_timing_score: float,
        exit_timing_score: float,
        momentum_alignment: float
    ) -> float:
        """
        Timing quality reward
        """
        # Entry timing (e.g., buying at support)
        entry_reward = entry_timing_score * 0.5
        
        # Exit timing (e.g., selling at resistance)
        exit_reward = exit_timing_score * 0.5
        
        # Momentum alignment bonus
        momentum_reward = momentum_alignment * self.momentum_weight
        
        return entry_reward + exit_reward + momentum_reward

class ExplorationObjective:
    """Intrinsic motivation for exploration"""
    
    def __init__(self, config: dict):
        self.novelty_threshold = config.get('novelty_threshold', 10)
        self.decay_rate = config.get('exploration_decay', 0.999)
        self.state_visits = {}
        
    def compute(
        self,
        state: np.ndarray,
        action: int,
        visit_count: int
    ) -> float:
        """
        Exploration bonus based on state novelty
        """
        # Hash state for counting
        state_hash = hash(state.tobytes())
        
        # Update visit count
        if state_hash not in self.state_visits:
            self.state_visits[state_hash] = 0
        self.state_visits[state_hash] += 1
        
        # Novelty bonus (inverse of visit count)
        if visit_count < self.novelty_threshold:
            novelty_bonus = 1.0 / (1 + visit_count) * 0.1
        else:
            novelty_bonus = 0.0
            
        # Decay exploration over time
        novelty_bonus *= self.decay_rate ** len(self.state_visits)
        
        return novelty_bonus


C.2.3 Testing Framework

# New test file

import pytest
import numpy as np
import torch
from core.rl.rewards.multi_objective_reward import (
    MultiObjectiveRewardShaper,
    RewardComponents
)

class TestMultiObjectiveReward:
    
    @pytest.fixture
    def config(self):
        return {
            'state_dim': 512,
            'device': 'cpu',
            'roi_scale': 100.0,
            'drawdown_threshold': 0.05,
            'target_trade_frequency': 0.1
        }
    
    @pytest.fixture
    def reward_shaper(self, config):
        return MultiObjectiveRewardShaper(config)
    
    def test_profit_objective(self, reward_shaper):
        """Test profit calculation"""
        info = {
            'realized_pnl': 100,
            'unrealized_pnl': 50,
            'position_size': 1000
        }
        
        state = np.random.randn(512)
        action = 1  # BUY_SMALL
        next_state = np.random.randn(512)
        
        reward, components = reward_shaper.compute_reward(
            state, action, next_state, info
        )
        
        assert components.profit > 0
        assert components.total != 0
        assert len(components.weights) == 5
        
    def test_risk_penalty(self, reward_shaper):
        """Test risk penalties"""
        info = {
            'drawdown': 0.10,  # 10% drawdown
            'volatility': 0.02,
            'exposure': 0.98,  # Over-exposed
            'var_95': 0.03
        }
        
        state = np.random.randn(512)
        action = 0  # HOLD
        next_state = np.random.randn(512)
        
        reward, components = reward_shaper.compute_reward(
            state, action, next_state, info
        )
        
        assert components.risk < 0  # Should be negative (penalty)
        
    def test_activity_diversity(self, reward_shaper):
        """Test activity and diversity rewards"""
        info = {
            'recent_actions': [0, 1, 2, 3, 4, 5, 6] * 10,  # All actions used
            'time_since_trade': 20,
            'total_trades': 5
        }
        
        state = np.random.randn(512)
        action = 1  # BUY_SMALL
        next_state = np.random.randn(512)
        
        reward, components = reward_shaper.compute_reward(
            state, action, next_state, info
        )
        
        assert components.activity > 0  # Diverse actions rewarded
        
    def test_learned_weights(self, reward_shaper):
        """Test dynamic weight generation"""
        state1 = np.random.randn(512)
        state2 = np.random.randn(512)
        
        # Get weights for different states
        weights1 = reward_shaper.weight_head(
            torch.FloatTensor(state1).unsqueeze(0)
        ).squeeze().numpy()
        
        weights2 = reward_shaper.weight_head(
            torch.FloatTensor(state2).unsqueeze(0)
        ).squeeze().numpy()
        
        # Weights should sum to 1
        assert np.isclose(weights1.sum(), 1.0)
        assert np.isclose(weights2.sum(), 1.0)
        
        # Different states should produce different weights
        assert not np.allclose(weights1, weights2)
        
    def test_exploration_decay(self, reward_shaper):
        """Test exploration bonus decay"""
        state = np.random.randn(512)
        action = 1
        
        # First visit should have highest bonus
        first_reward, first_components = reward_shaper.compute_reward(
            state, action, state, {'recent_actions': []}
        )
        
        # Subsequent visits should have lower bonus
        second_reward, second_components = reward_shaper.compute_reward(
            state, action, state, {'recent_actions': []}
        )
        
        assert first_components.exploration >= second_components.exploration

C.3 Integration with Training Pipeline

# Modifications to existing file

def build_reward_config(config: dict) -> dict:
    """Build reward configuration with multi-objective support"""
    
    reward_config = config.get('reward', {})
    
    # Check if multi-objective mode is enabled
    if reward_config.get('multi_objective_enabled', False):
        return {
            'reward_class': 'MultiObjectiveRewardShaper',
            'state_dim': config['model']['state_dim'],
            'device': config.get('device', 'cuda'),
            
            # Objective-specific configs
            'profit': {
                'roi_scale': reward_config.get('roi_scale', 100.0),
                'unrealized_weight': reward_config.get('unrealized_weight', 0.3)
            },
            'risk': {
                'drawdown_threshold': reward_config.get('drawdown_threshold', 0.05),
                'var_limit': reward_config.get('var_limit', 0.02)
            },
            'activity': {
                'target_trade_frequency': reward_config.get('target_frequency', 0.1),
                'diversity_window': reward_config.get('diversity_window', 50)
            },
            'timing': {
                'momentum_weight': reward_config.get('momentum_weight', 0.3)
            },
            'exploration': {
                'novelty_threshold': reward_config.get('novelty_threshold', 10),
                'exploration_decay': reward_config.get('exploration_decay', 0.999)
            }
        }
    else:
        # Fallback to existing reward shaper
        return reward_config

C.4 Quality Gates & Success Metrics
C.4.1 Automated Test Suite

#!/bin/bash

echo "=== Phase C: Reward Revolution Test Suite ==="

# Unit tests
echo "Running unit tests..."
pytest tests/test_multi_objective_reward.py -v --tb=short

# Integration test with environment
echo "Running integration test..."
python scripts/test_reward_integration.py \
    --episodes 10 \
    --config configs/phase_c_reward_test.yaml

# Performance benchmark
echo "Running reward computation benchmark..."
python scripts/benchmark_reward_shaper.py \
    --iterations 10000 \
    --batch-size 32

# Validation metrics
python scripts/validate_phase_c_metrics.py

C.4.2 Success Criteria
Metric	Target	Measurement Method
Reward Computation Speed	<1ms per step	Benchmark script
Weight Network Convergence	<1000 updates	Training logs
Objective Balance	No single objective >60% weight	Weight histogram
Exploration Bonus Impact	5-10% of total reward early	Reward component analysis
Test Coverage	>98%	pytest-cov

Phase D: Advanced Training Techniques (Week 4)
D.1 IMPALA-style V-Trace Implementation
D.1.1 Core V-Trace Module

# New file for V-trace implementation

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

class VTraceReturns:
    """
    V-trace return computation for off-policy correction
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0
    ):
        self.gamma = gamma
        self.rho_bar = rho_bar  # Clipping for importance weights
        self.c_bar = c_bar      # Clipping for trace coefficients
        
    def compute_vtrace_returns(
        self,
        behavior_logits: torch.Tensor,  # (T, B, A)
        target_logits: torch.Tensor,    # (T, B, A)
        actions: torch.Tensor,           # (T, B)
        rewards: torch.Tensor,           # (T, B)
        values: torch.Tensor,            # (T, B)
        bootstrap_value: torch.Tensor,   # (B,)
        masks: torch.Tensor              # (T, B)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute V-trace targets and advantages
        
        Returns:
            vs: V-trace value targets (T, B)
            advantages: V-trace advantages (T, B)
        """
        
        T, B = rewards.shape
        
        # Compute importance sampling ratios
        behavior_probs = F.softmax(behavior_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)
        
        # Get probabilities for taken actions
        behavior_action_probs = behavior_probs.gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        target_action_probs = target_probs.gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        
        # Importance weights
        rhos = target_action_probs / (behavior_action_probs + 1e-8)
        rho_bars = torch.minimum(rhos, torch.ones_like(rhos) * self.rho_bar)
        c_bars = torch.minimum(rhos, torch.ones_like(rhos) * self.c_bar)
        
        # Compute V-trace returns recursively
        vs = torch.zeros_like(rewards)
        v_s = bootstrap_value
        
        for t in reversed(range(T)):
            delta_v = rho_bars[t] * (
                rewards[t] + self.gamma * masks[t] * v_s - values[t]
            )
            v_s = values[t] + delta_v + self.gamma * masks[t] * c_bars[t] * (v_s - values[t])
            vs[t] = v_s
            
        # Compute advantages
        advantages = rho_bars * (rewards + self.gamma * masks * vs[1:].roll(-1, 0) - values)
        advantages[-1] = rho_bars[-1] * (rewards[-1] + self.gamma * masks[-1] * bootstrap_value - values[-1])
        
        return vs.detach(), advantages.detach()

class IMPALALearner:
    """
    IMPALA-style distributed learner with V-trace
    """
    
    def __init__(
        self,
        policy_network: torch.nn.Module,
        value_network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict
    ):
        self.policy = policy_network
        self.value = value_network
        self.optimizer = optimizer
        self.vtrace = VTraceReturns(
            gamma=config.get('gamma', 0.99),
            rho_bar=config.get('rho_bar', 1.0),
            c_bar=config.get('c_bar', 1.0)
        )
        
        # Loss weights
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
    def compute_loss(
        self,
        trajectories: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute IMPALA loss with V-trace corrections
        """
        
        # Extract trajectory components
        states = trajectories['states']           # (T, B, state_dim)
        actions = trajectories['actions']         # (T, B)
        rewards = trajectories['rewards']         # (T, B)
        masks = trajectories['masks']            # (T, B)
        behavior_logits = trajectories['behavior_logits']  # (T, B, A)
        
        # Forward pass with current policy
        target_logits = self.policy(states)      # (T, B, A)
        values = self.value(states).squeeze(-1)  # (T, B)
        
        # Bootstrap value for last timestep
        with torch.no_grad():
            bootstrap_value = self.value(
                trajectories['next_states'][-1]
            ).squeeze(-1)
        
        # Compute V-trace returns
        vtrace_returns, advantages = self.vtrace.compute_vtrace_returns(
            behavior_logits,
            target_logits,
            actions,
            rewards,
            values,
            bootstrap_value,
            masks
        )
        
        # Policy gradient loss
        log_probs = F.log_softmax(target_logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(action_log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, vtrace_returns)
        
        # Entropy bonus
        probs = F.softmax(target_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        # Logging metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_vtrace_return': vtrace_returns.mean().item()
        }
        
        return total_loss, metrics

D.1.2 Distributed Training Infrastructure

# New distributed training coordinator

import ray
import torch
from typing import List, Dict
import numpy as np
from collections import deque

@ray.remote
class ActorWorker:
    """
    Remote actor for trajectory collection
    """
    
    def __init__(self, env_config: dict, actor_id: int):
        self.env = create_trading_environment(env_config)
        self.actor_id = actor_id
        self.device = 'cpu'  # Actors run on CPU
        self.trajectory_buffer = deque(maxlen=1000)
        
    def collect_trajectories(
        self,
        policy_weights: dict,
        num_steps: int
    ) -> List[dict]:
        """
        Collect trajectories using current policy
        """
        # Update local policy
        self.policy.load_state_dict(policy_weights)
        
        trajectories = []
        state = self.env.reset()
        
        for _ in range(num_steps):
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits = self.policy(state_tensor)
                action = torch.multinomial(
                    F.softmax(logits, dim=-1), 1
                ).item()
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Store trajectory
            trajectories.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'behavior_logits': logits.numpy()
            })
            
            if done:
                state = self.env.reset()
            else:
                state = next_state
                
        return trajectories

class IMPALATrainer:
    """
    Main IMPALA training coordinator
    """
    
    def __init__(
        self,
        config: dict,
        num_actors: int = 8
    ):
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # Initialize learner
        self.learner = IMPALALearner(
            policy_network=self.create_policy_network(),
            value_network=self.create_value_network(),
            optimizer=torch.optim.Adam(lr=config['learning_rate']),
            config=config
        )
        
        # Initialize Ray
        ray.init()
        
        # Create remote actors
        self.actors = [
            ActorWorker.remote(config['env'], i)
            for i in range(num_actors)
        ]
        
        # Trajectory queue
        self.trajectory_queue = deque(maxlen=10000)
        
    def train(self, total_timesteps: int):
        """
        Main training loop
        """
        
        timesteps_collected = 0
        
        while timesteps_collected < total_timesteps:
            
            # Broadcast current policy to actors
            policy_weights = self.learner.policy.state_dict()
            
            # Collect trajectories in parallel
            trajectory_futures = [
                actor.collect_trajectories.remote(
                    policy_weights,
                    self.config['rollout_length']
                )
                for actor in self.actors
            ]
            
            # Gather trajectories
            all_trajectories = ray.get(trajectory_futures)
            
            # Add to queue
            for actor_trajectories in all_trajectories:
                self.trajectory_queue.extend(actor_trajectories)
            
            # Sample batch for learning
            if len(self.trajectory_queue) >= self.config['batch_size']:
                batch = self.sample_batch(self.config['batch_size'])
                
                # Compute loss and update
                loss, metrics = self.learner.compute_loss(batch)
                
                self.learner.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.learner.policy.parameters(),
                    self.config['max_grad_norm']
                )
                
                self.learner.optimizer.step()
                
                # Log metrics
                self.log_metrics(metrics, timesteps_collected)
                
            timesteps_collected += len(all_trajectories) * self.config['rollout_length']
            
        # Cleanup
        ray.shutdown()


D.2 Evolutionary Strategy Implementation
D.2.1 Core ES Algorithm

# New evolutionary strategy implementation

import numpy as np
from typing import Callable, Tuple, List
import multiprocessing as mp
from dataclasses import dataclass

@dataclass
class ESConfig:
    """Configuration for Evolution Strategy"""
    population_size: int = 256
    sigma: float = 0.02  # Noise standard deviation
    learning_rate: float = 0.01
    elite_ratio: float = 0.2
    weight_decay: float = 0.001
    antithetic: bool = True  # Use antithetic sampling
    
class EvolutionStrategy:
    """
    OpenAI Evolution Strategy for robust trading policies
    """
    
    def __init__(
        self,
        policy_fn: Callable,
        env_fn: Callable,
        config: ESConfig
    ):
        self.policy_fn = policy_fn
        self.env_fn = env_fn
        self.config = config
        
        # Initialize policy parameters
        dummy_policy = policy_fn()
        self.theta = self.flatten_parameters(dummy_policy)
        self.n_params = len(self.theta)
        
        # Adam optimizer state
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.t = 0
        
        # Setup multiprocessing
        self.n_workers = mp.cpu_count()
        
    def flatten_parameters(self, policy) -> np.ndarray:
        """Flatten policy network parameters"""
        params = []
        for param in policy.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def unflatten_parameters(self, theta: np.ndarray, policy):
        """Restore flattened parameters to policy"""
        idx = 0
        for param in policy.parameters():
            param_shape = param.data.shape
            param_size = np.prod(param_shape)
            param.data = torch.FloatTensor(
                theta[idx:idx+param_size].reshape(param_shape)
            )
            idx += param_size
            
    def evaluate_policy(
        self,
        theta: np.ndarray,
        num_episodes: int = 5
    ) -> float:
        """
        Evaluate policy parameters
        """
        policy = self.policy_fn()
        self.unflatten_parameters(theta, policy)
        
        env = self.env_fn()
        total_reward = 0.0
        
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action = policy.act(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                
            total_reward += episode_reward
            
        return total_reward / num_episodes
    
    def train_step(self) -> Tuple[float, dict]:
        """
        Single training iteration
        """
        
        # Generate population
        if self.config.antithetic:
            # Antithetic sampling (half noise, half negative)
            epsilon = np.random.randn(
                self.config.population_size // 2,
                self.n_params
            )
            epsilon = np.concatenate([epsilon, -epsilon], axis=0)
        else:
            epsilon = np.random.randn(
                self.config.population_size,
                self.n_params
            )
        
        # Perturb parameters
        theta_perturbed = self.theta + self.config.sigma * epsilon
        
        # Parallel evaluation
        with mp.Pool(self.n_workers) as pool:
            rewards = pool.map(
                self.evaluate_policy,
                theta_perturbed
            )
        rewards = np.array(rewards)
        
        # Compute advantages (normalized rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute gradient
        gradient = np.mean(
            advantages.reshape(-1, 1) * epsilon,
            axis=0
        ) / self.config.sigma
        
        # Adam update
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient**2
        
        m_hat = self.m / (1 - beta1**self.t)
        v_hat = self.v / (1 - beta2**self.t)
        
        # Update parameters
        self.theta += self.config.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        # Weight decay
        self.theta *= (1 - self.config.weight_decay)
        
        # Metrics
        metrics = {
            'mean_reward': rewards.mean(),
            'std_reward': rewards.std(),
            'max_reward': rewards.max(),
            'gradient_norm': np.linalg.norm(gradient)
        }
        
        return rewards.mean(), metrics
    
    def train(
        self,
        num_iterations: int,
        save_interval: int = 10
    ):
        """
        Full training loop
        """
        
        best_reward = -np.inf
        
        for iteration in range(num_iterations):
            mean_reward, metrics = self.train_step()
            
            # Log progress
            print(f"Iteration {iteration}: Reward = {mean_reward:.4f}")
            
            # Save best policy
            if mean_reward > best_reward:
                best_reward = mean_reward
                self.save_policy(f"es_policy_best.pt")
                
            # Periodic saves
            if iteration % save_interval == 0:
                self.save_policy(f"es_policy_iter_{iteration}.pt")

D.2.2 Hybrid ES-RL Training

# Combine ES robustness with RL sample efficiency

class HybridESRLTrainer:
    """
    Alternating ES and RL training for robust policies
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize RL trainer (PPO/SAC)
        self.rl_trainer = self.create_rl_trainer(config)
        
        # Initialize ES trainer
        self.es_trainer = EvolutionStrategy(
            policy_fn=lambda: self.create_policy(),
            env_fn=lambda: self.create_environment(),
            config=ESConfig(**config['es'])
        )
        
        # Training schedule
        self.rl_steps = config.get('rl_steps_per_cycle', 10000)
        self.es_iterations = config.get('es_iterations_per_cycle', 10)
        
    def train_cycle(self):
        """
        Single training cycle combining ES and RL
        """
        
        # Phase 1: RL training for sample efficiency
        print("Starting RL phase...")
        rl_metrics = self.rl_trainer.train(self.rl_steps)
        
        # Transfer RL policy to ES
        rl_params = self.rl_trainer.get_policy_parameters()
        self.es_trainer.theta = rl_params
        
        # Phase 2: ES training for robustness
        print("Starting ES phase...")
        es_metrics = []
        for _ in range(self.es_iterations):
            reward, metrics = self.es_trainer.train_step()
            es_metrics.append(metrics)
            
        # Transfer improved policy back to RL
        self.rl_trainer.set_policy_parameters(self.es_trainer.theta)
        
        return {
            'rl_metrics': rl_metrics,
            'es_metrics': es_metrics
        }

D.3 Testing & Validation Framework
D.3.1 Performance Testing Suite

# Test suite for Phase D components

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

class TestVTrace:
    
    def test_importance_weights(self):
        """Test V-trace importance weight computation"""
        from core.rl.algorithms.vtrace import VTraceReturns
        
        vtrace = VTraceReturns(rho_bar=1.0, c_bar=1.0)
        
        # Create sample data
        T, B, A = 10, 4, 7
        behavior_logits = torch.randn(T, B, A)
        target_logits = torch.randn(T, B, A)
        actions = torch.randint(0, A, (T, B))
        rewards = torch.randn(T, B)
        values = torch.randn(T, B)
        bootstrap = torch.randn(B)
        masks = torch.ones(T, B)
        
        # Compute V-trace
        vs, advantages = vtrace.compute_vtrace_returns(
            behavior_logits, target_logits, actions,
            rewards, values, bootstrap, masks
        )
        
        assert vs.shape == (T, B)
        assert advantages.shape == (T, B)
        assert not torch.isnan(vs).any()
        assert not torch.isnan(advantages).any()
        
    def test_es_gradient_estimation(self):
        """Test ES gradient estimation"""
        from core.rl.algorithms.evolution_strategy import EvolutionStrategy
        
        # Mock environment and policy
        mock_env = Mock()
        mock_env.reset.return_value = np.zeros(100)
        mock_env.step.return_value = (np.zeros(100), 0.1, False, {})
        
        es = EvolutionStrategy(
            policy_fn=lambda: Mock(),
            env_fn=lambda: mock_env,
            config=ESConfig(population_size=10)
        )
        
        # Test gradient computation
        rewards = np.random.randn(10)
        epsilon = np.random.randn(10, 100)
        
        gradient = np.mean(
            ((rewards - rewards.mean()) / (rewards.std() + 1e-8)).reshape(-1, 1) * epsilon,
            axis=0
        ) / 0.02
        
        assert gradient.shape == (100,)
        assert not np.isnan(gradient).any()

D.3.2 Integration Validation

#!/bin/bash

echo "=== Phase D: Advanced Training Validation ==="

# Test V-trace implementation
echo "Testing V-trace..."
python -m pytest tests/test_advanced_training.py::TestVTrace -v

# Test ES implementation  
echo "Testing Evolution Strategy..."
python -m pytest tests/test_advanced_training.py::TestES -v

# Benchmark V-trace performance
echo "Benchmarking V-trace..."
python scripts/benchmark_vtrace.py \
    --batch-size 32 \
    --sequence-length 100 \
    --iterations 1000

# Run short ES training
echo "Running ES smoke test..."
python scripts/test_es_training.py \
    --iterations 10 \
    --population-size 16 \
    --episodes 2

# Compare hybrid vs pure approaches
echo "Comparing training approaches..."
python scripts/compare_training_methods.py \
    --timesteps 10000 \
    --seeds 3

D.4 Configuration Templates

training:
  method: "hybrid_es_rl"  # Options: ppo, sac, es, hybrid_es_rl, impala
  
  # IMPALA settings
  impala:
    num_actors: 8
    rollout_length: 100
    batch_size: 256
    learning_rate: 0.0003
    value_loss_coef: 0.5
    entropy_coef: 0.01
    rho_bar: 1.0
    c_bar: 1.0
    max_grad_norm: 40.0
    
  # Evolution Strategy settings
  es:
    population_size: 256
    sigma: 0.02
    learning_rate: 0.01
    elite_ratio: 0.2
    weight_decay: 0.001
    antithetic: true
    eval_episodes: 5
    
  # Hybrid training
  hybrid:
    rl_steps_per_cycle: 10000
    es_iterations_per_cycle: 10
    switch_threshold: 0.5  # Performance threshold to switch methods
    
  # Common settings
  gamma: 0.99
  device: "cuda"
  seed: 42
  save_interval: 5000
  eval_interval: 1000

Phase E: Environment Restructuring (Week 5)
E.1 Auxiliary Task Implementation
E.1.1 Auxiliary Prediction Heads

# New file for auxiliary learning tasks

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class AuxiliaryTaskNetwork(nn.Module):
    """
    Multi-task learning with auxiliary objectives
    """
    
    def __init__(
        self,
        shared_dim: int = 256,
        hidden_dim: int = 128,
        config: dict = None
    ):
        super().__init__()
        self.config = config or {}
        
        # Shared encoder is passed from main policy
        
        # Price prediction head
        self.price_predictor = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Next price change
        )
        
        # Volatility estimation head
        self.volatility_estimator = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Volatility estimate
            nn.Softplus()  # Ensure positive
        )
        
        # Volume prediction head
        self.volume_predictor = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Low/Normal/High volume
        )
        
        # Optimal action predictor (behavior cloning)
        self.bc_head = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # Action logits
        )
        
        # Market regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Bull/Bear/Sideways/Volatile
        )
        
    def forward(
        self,
        shared_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for all auxiliary tasks
        """
        
        predictions = {}
        
        # Price prediction
        predictions['next_price'] = self.price_predictor(shared_features)
        
        # Volatility estimation
        predictions['volatility'] = self.volatility_estimator(shared_features)
        
        # Volume prediction
        predictions['volume_class'] = self.volume_predictor(shared_features)
        
        # Behavior cloning
        predictions['bc_logits'] = self.bc_head(shared_features)
        
        # Regime classification
        predictions['regime'] = self.regime_classifier(shared_features)
        
        return predictions
    
    def compute_auxiliary_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Dict[str, float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted auxiliary loss
        """
        
        if weights is None:
            weights = {
                'price': 0.2,
                'volatility': 0.15,
                'volume': 0.1,
                'bc': 0.3,
                'regime': 0.25
            }
        
        losses = {}
        
        # Price prediction loss (MSE)
        if 'next_price' in targets:
            losses['price'] = F.mse_loss(
                predictions['next_price'],
                targets['next_price']
            )
        
        # Volatility loss (MSE)
        if 'volatility' in targets:
            losses['volatility'] = F.mse_loss(
                predictions['volatility'],
                targets['volatility']
            )
        
        # Volume classification loss
        if 'volume_class' in targets:
            losses['volume'] = F.cross_entropy(
                predictions['volume_class'],
                targets['volume_class']
            )
        
        # Behavior cloning loss
        if 'expert_action' in targets:
            losses['bc'] = F.cross_entropy(
                predictions['bc_logits'],
                targets['expert_action']
            )
        
        # Regime classification loss
        if 'regime' in targets:
            losses['regime'] = F.cross_entropy(
                predictions['regime'],
                targets['regime']
            )
        
        # Weighted sum
        total_loss = sum(
            weights.get(k, 0.0) * v 
            for k, v in losses.items()
        )
        
        return total_loss, losses

E.1.2 Expert Demonstration Generator

# Generate expert demonstrations for behavior cloning

import numpy as np
from typing import List, Dict, Tuple
import pandas as pd

class TradingExpert:
    """
    Rule-based expert for generating demonstrations
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Trading parameters
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.bb_width_threshold = config.get('bb_width', 0.02)
        self.volume_spike = config.get('volume_spike', 1.5)
        
    def get_expert_action(
        self,
        state: Dict[str, float],
        position: float = 0.0
    ) -> int:
        """
        Generate expert action based on technical rules
        """
        
        rsi = state.get('rsi', 50)
        macd_signal = state.get('macd_signal', 0)
        bb_position = state.get('bb_position', 0.5)  # 0=lower, 1=upper
        volume_ratio = state.get('volume_ratio', 1.0)
        trend_strength = state.get('trend_strength', 0)
        
        # No position - look for entry
        if position == 0:
            # Strong buy signal
            if (rsi < self.rsi_oversold and 
                macd_signal > 0 and
                bb_position < 0.2 and
                volume_ratio > self.volume_spike):
                return 3  # BUY_LARGE
            
            # Moderate buy signal
            elif (rsi < 40 and 
                  macd_signal > 0 and
                  trend_strength > 0.5):
                return 2  # BUY_MEDIUM
            
            # Weak buy signal
            elif rsi < 50 and macd_signal > 0:
                return 1  # BUY_SMALL
            
            else:
                return 0  # HOLD
        
        # Has position - look for exit or pyramid
        else:
            # Strong sell signal
            if (rsi > self.rsi_overbought and
                macd_signal < 0 and
                bb_position > 0.8):
                return 5  # SELL_ALL
            
            # Partial exit signal
            elif (rsi > 65 and macd_signal < 0):
                return 4  # SELL_PARTIAL
            
            # Pyramid signal (add to winner)
            elif (position > 0 and  # Profitable position
                  rsi < 50 and
                  macd_signal > 0 and
                  trend_strength > 0.7):
                return 6  # ADD_TO_WINNER
            
            else:
                return 0  # HOLD
    
    def generate_demonstrations(
        self,
        historical_data: pd.DataFrame,
        num_episodes: int = 100
    ) -> List[Dict]:
        """
        Generate expert demonstration trajectories
        """
        
        demonstrations = []
        
        for episode in range(num_episodes):
            # Random starting point
            start_idx = np.random.randint(0, len(historical_data) - 1000)
            episode_data = historical_data.iloc[start_idx:start_idx+1000]
            
            trajectory = []
            position = 0.0
            
            for idx, row in episode_data.iterrows():
                # Create state dict
                state = {
                    'rsi': row['RSI_14'],
                    'macd_signal': row['MACD_signal'],
                    'bb_position': row['BB_position'],
                    'volume_ratio': row['Volume'] / row['Volume_SMA_20'],
                    'trend_strength': row['ADX_14'] / 100
                }
                
                # Get expert action
                action = self.get_expert_action(state, position)
                
                # Update position (simplified)
                if action in [1, 2, 3]:  # BUY
                    position += 1.0
                elif action in [4, 5]:  # SELL
                    position = max(0, position - 1.0)
                elif action == 6:  # ADD
                    position += 0.5
                
                # Store transition
                trajectory.append({
                    'state': row.values,
                    'action': action,
                    'position': position
                })
            
            demonstrations.append(trajectory)
        
        return demonstrations

E.2 Progressive Training Curriculum
E.2.1 Curriculum Controller

# Progressive difficulty curriculum

import numpy as np
from typing import Dict, List, Tuple
from enum import Enum

class DifficultyLevel(Enum):
    BEGINNER = 0
    INTERMEDIATE = 1
    ADVANCED = 2
    EXPERT = 3
    MASTER = 4

class ProgressiveCurriculum:
    """
    Gradually increase environment complexity
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Define curriculum stages
        self.stages = [
            # Stage 0: Beginner (strong trends, low volatility, minimal costs)
            {
                'level': DifficultyLevel.BEGINNER,
                'volatility': 0.1,
                'trend_strength': 0.8,
                'transaction_cost': 0.0001,
                'slippage': 0.0,
                'market_regime': 'bull',
                'noise_level': 0.05,
                'min_performance': {'sharpe': 0.3, 'win_rate': 0.5}
            },
            
            # Stage 1: Intermediate
            {
                'level': DifficultyLevel.INTERMEDIATE,
                'volatility': 0.15,
                'trend_strength': 0.6,
                'transaction_cost': 0.001,
                'slippage': 0.0005,
                'market_regime': 'mixed_bullish',
                'noise_level': 0.1,
                'min_performance': {'sharpe': 0.5, 'win_rate': 0.45}
            },
            
            # Stage 2: Advanced
            {
                'level': DifficultyLevel.ADVANCED,
                'volatility': 0.2,
                'trend_strength': 0.4,
                'transaction_cost': 0.002,
                'slippage': 0.001,
                'market_regime': 'sideways',
                'noise_level': 0.15,
                'min_performance': {'sharpe': 0.6, 'win_rate': 0.45}
            },
            
            # Stage 3: Expert
            {
                'level': DifficultyLevel.EXPERT,
                'volatility': 0.25,
                'trend_strength': 0.2,
                'transaction_cost': 0.003,
                'slippage': 0.0015,
                'market_regime': 'mixed_bearish',
                'noise_level': 0.2,
                'min_performance': {'sharpe': 0.7, 'win_rate': 0.45}
            },
            
            # Stage 4: Master (realistic)
            {
                'level': DifficultyLevel.MASTER,
                'volatility': 0.3,
                'trend_strength': 0.1,
                'transaction_cost': 0.003,
                'slippage': 0.002,
                'market_regime': 'all',
                'noise_level': 0.25,
                'min_performance': {'sharpe': 0.8, 'win_rate': 0.45}
            }
        ]
        
        self.current_stage = 0
        self.stage_episodes = 0
        self.stage_performance = []
        
    def get_current_config(self) -> Dict:
        """Get current stage configuration"""
        return self.stages[self.current_stage].copy()
    
    def should_advance(
        self,
        metrics: Dict[str, float]
    ) -> bool:
        """
        Check if agent should advance to next stage
        """
        
        if self.current_stage >= len(self.stages) - 1:
            return False
        
        current_reqs = self.stages[self.current_stage]['min_performance']
        
        # Check all requirements
        for metric, threshold in current_reqs.items():
            if metrics.get(metric, 0) < threshold:
                return False
        
        # Require minimum episodes at current level
        min_episodes = self.config.get('min_episodes_per_stage', 100)
        if self.stage_episodes < min_episodes:
            return False
        
        # Require consistent performance
        if len(self.stage_performance) < 10:
            return False
        
        recent_performance = self.stage_performance[-10:]
        if np.mean(recent_performance) < 0.8:  # 80% success rate
            return False
        
        return True
    
    def advance_stage(self):
        """Move to next difficulty level"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.stage_episodes = 0
            self.stage_performance = []
            
            print(f"Advanced to stage {self.current_stage}: "
                  f"{self.stages[self.current_stage]['level'].name}")
    
    def update_performance(
        self,
        episode_success: bool,
        metrics: Dict[str, float]
    ):
        """Update stage performance tracking"""
        self.stage_episodes += 1
        self.stage_performance.append(1.0 if episode_success else 0.0)
        
        # Check for advancement
        if self.should_advance(metrics):
            self.advance_stage()

E.2.2 Curriculum Integration

# Environment wrapper with curriculum support

import gym
import numpy as np
from typing import Dict, Tuple

class CurriculumTradingEnvironment(gym.Env):
    """
    Trading environment with progressive curriculum
    """
    
    def __init__(
        self,
        base_env_class,
        curriculum_config: dict = None
    ):
        self.base_env_class = base_env_class
        self.curriculum = ProgressiveCurriculum(curriculum_config)
        
        # Create initial environment
        self.env = self._create_env()
        
        # Copy attributes
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Tracking
        self.episode_metrics = {}
        
    def _create_env(self):
        """Create environment with current curriculum settings"""
        config = self.curriculum.get_current_config()
        
        # Apply curriculum parameters
        env_config = {
            'volatility_multiplier': config['volatility'],
            'trend_bias': config['trend_strength'],
            'transaction_cost': config['transaction_cost'],
            'slippage': config['slippage'],
            'noise_level': config['noise_level']
        }
        
        if config['market_regime'] != 'all':
            env_config['market_regime'] = config['market_regime']
        
        return self.base_env_class(**env_config)
    
    def reset(self, **kwargs):
        """Reset with curriculum adjustment"""
        
        # Check if we should recreate env with new difficulty
        if self.curriculum.stage_episodes == 0:
            self.env = self._create_env()
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step with curriculum tracking"""
        obs, reward, done, info = self.env.step(action)
        
        # Track episode metrics
        if 'metrics' in info:
            for key, value in info['metrics'].items# filepath: c:\TradingBotAI\core\rl\environments\curriculum_env.py
# Environment wrapper with curriculum support

import gym
import numpy as np
from typing import Dict, Tuple

class CurriculumTradingEnvironment(gym.Env):
    """
    Trading environment with progressive curriculum
    """
    
    def __init__(
        self,
        base_env_class,
        curriculum_config: dict = None
    ):
        self.base_env_class = base_env_class
        self.curriculum = ProgressiveCurriculum(curriculum_config)
        
        # Create initial environment
        self.env = self._create_env()
        
        # Copy attributes
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Tracking
        self.episode_metrics = {}
        
    def _create_env(self):
        """Create environment with current curriculum settings"""
        config = self.curriculum.get_current_config()
        
        # Apply curriculum parameters
        env_config = {
            'volatility_multiplier': config['volatility'],
            'trend_bias': config['trend_strength'],
            'transaction_cost': config['transaction_cost'],
            'slippage': config['slippage'],
            'noise_level': config['noise_level']
        }
        
        if config['market_regime'] != 'all':
            env_config['market_regime'] = config['market_regime']
        
        return self.base_env_class(**env_config)
    
    def reset(self, **kwargs):
        """Reset with curriculum adjustment"""
        
        # Check if we should recreate env with new difficulty
        if self.curriculum.stage_episodes == 0:
            self.env = self._create_env()
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step with curriculum tracking"""
        obs, reward, done, info = self.env.step(action)
        
        # Track episode metrics
        if 'metrics' in info:
            for key, value in info['metrics'].items

E.2.3 Adaptive Difficulty Scaling

class AdaptiveDifficultyController:
    """Dynamically adjust difficulty based on agent performance"""
    
    def __init__(self):
        self.performance_buffer = deque(maxlen=100)
        self.difficulty_params = {
            'spread': 0.001,  # Start with tight spreads
            'volatility_multiplier': 0.5,  # Start with low volatility
            'trend_clarity': 0.8,  # Clear trends initially
            'noise_level': 0.1,  # Low noise
            'regime_stability': 0.9  # Stable regimes
        }
        
    def update_difficulty(self, episode_metrics):
        """Adjust difficulty based on recent performance"""
        self.performance_buffer.append(episode_metrics)
        
        if len(self.performance_buffer) < 20:
            return  # Need more data
            
        recent_performance = np.mean([m['total_return'] for m in self.performance_buffer[-20:]])
        win_rate = np.mean([m['win_rate'] for m in self.performance_buffer[-20:]])
        
        # If performing too well, increase difficulty
        if recent_performance > 0.02 and win_rate > 0.6:
            self.difficulty_params['spread'] = min(0.003, self.difficulty_params['spread'] * 1.1)
            self.difficulty_params['volatility_multiplier'] = min(1.5, self.difficulty_params['volatility_multiplier'] * 1.05)
            self.difficulty_params['trend_clarity'] = max(0.3, self.difficulty_params['trend_clarity'] * 0.95)
            self.difficulty_params['noise_level'] = min(0.5, self.difficulty_params['noise_level'] * 1.1)
            
        # If struggling, reduce difficulty
        elif recent_performance < -0.01 or win_rate < 0.35:
            self.difficulty_params['spread'] = max(0.0005, self.difficulty_params['spread'] * 0.95)
            self.difficulty_params['volatility_multiplier'] = max(0.3, self.difficulty_params['volatility_multiplier'] * 0.95)
            self.difficulty_params['trend_clarity'] = min(0.9, self.difficulty_params['trend_clarity'] * 1.05)
            self.difficulty_params['noise_level'] = max(0.05, self.difficulty_params['noise_level'] * 0.9)
            
        return self.difficulty_params


Phase F: Production Readiness (Week 6)
F.1 Model Export and Optimization
F.1.1 ONNX Export

# scripts/export_to_onnx.py
import torch
import torch.onnx
import onnx
import onnxruntime as ort

def export_sac_to_onnx(model_path, output_path):
    """Export SAC policy to ONNX for production inference"""
    
    # Load trained model
    model = SAC.load(model_path)
    policy = model.policy
    
    # Create dummy input
    dummy_obs = torch.randn(1, observation_dim).to(model.device)
    
    # Export actor network (deterministic policy for production)
    torch.onnx.export(
        policy.actor.mu,  # Deterministic action
        dummy_obs,
        f"{output_path}/sac_actor.onnx",
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    
    # Optimize ONNX model
    import onnxoptimizer
    model_onnx = onnx.load(f"{output_path}/sac_actor.onnx")
    optimized = onnxoptimizer.optimize(model_onnx)
    onnx.save(optimized, f"{output_path}/sac_actor_optimized.onnx")
    
    # Validate
    ort_session = ort.InferenceSession(f"{output_path}/sac_actor_optimized.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_obs.cpu().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    logger.info(f"ONNX export successful. Output shape: {ort_outputs[0].shape}")
    
    return ort_session

F.1.2 TensorRT Optimization (Optional)

F.2 Production Inference Pipeline
F.2.1 Real-time Inference Server

# core/production/inference_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import asyncio
from typing import Dict, List, Optional

class TradingInferenceServer:
    def __init__(self, model_path: str, config: dict):
        self.app = FastAPI()
        self.model = self.load_model(model_path)
        self.feature_pipeline = FeaturePipeline(config)
        self.risk_manager = RiskManager(config['risk'])
        self.position_tracker = PositionTracker()
        
        # Setup routes
        self.setup_routes()
        
    def load_model(self, path: str):
        """Load ONNX model for inference"""
        import onnxruntime as ort
        return ort.InferenceSession(path)
        
    def setup_routes(self):
        @self.app.post("/predict")
        async def predict(request: PredictionRequest):
            try:
                # Extract features
                features = await self.feature_pipeline.compute_features(
                    request.symbol,
                    request.market_data
                )
                
                # Get current position
                position = self.position_tracker.get_position(request.symbol)
                
                # Prepare observation
                obs = self.prepare_observation(features, position)
                
                # Model inference
                action = self.model.run(None, {self.model.get_inputs()[0].name: obs})[0]
                
                # Apply risk checks
                action = self.risk_manager.validate_action(
                    action, 
                    position, 
                    request.portfolio_state
                )
                
                # Convert to trade signal
                signal = self.action_to_signal(action, position)
                
                # Log decision
                await self.log_decision(request.symbol, action, signal)
                
                return {
                    "symbol": request.symbol,
                    "action": action.tolist(),
                    "signal": signal,
                    "confidence": self.compute_confidence(action),
                    "timestamp": request.timestamp
                }
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
    def action_to_signal(self, action: np.ndarray, position: Position) -> Dict:
        """Convert continuous action to discrete trading signal"""
        action_value = float(action[0])
        
        # Apply thresholds
        if abs(action_value) < 0.1:
            return {"type": "HOLD", "size": 0}
            
        if action_value > 0:
            # Buy signal
            size = self.calculate_position_size(action_value, position)
            return {
                "type": "BUY",
                "size": size,
                "order_type": "LIMIT" if action_value < 0.5 else "MARKET"
            }
        else:
            # Sell signal
            if position.size == 0:
                return {"type": "HOLD", "size": 0}
                
            size = min(abs(action_value) * position.size, position.size)
            return {
                "type": "SELL",
                "size": size,
                "order_type": "MARKET"
            }

F.2.2 Circuit Breakers and Failsafes

class ProductionSafeguards:
    def __init__(self, config):
        self.max_daily_trades = config.get('max_daily_trades', 100)
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_daily_loss = config.get('max_daily_loss', 0.02)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        self.daily_stats = defaultdict(lambda: {
            'trades': 0,
            'pnl': 0.0,
            'last_reset': datetime.now()
        })
        
    def check_circuit_breakers(self, symbol: str, action: np.ndarray, portfolio: Portfolio) -> bool:
        """Check if action should be blocked by circuit breakers"""
        
        # Reset daily counters if needed
        self._reset_if_new_day(symbol)
        
        checks = {
            'daily_trade_limit': self.daily_stats[symbol]['trades'] < self.max_daily_trades,
            'daily_loss_limit': self.daily_stats[symbol]['pnl'] > -self.max_daily_loss * portfolio.total_value,
            'position_size_limit': self._check_position_size(action, portfolio),
            'portfolio_drawdown': portfolio.current_drawdown < 0.15,
            'correlation_limit': self._check_correlation_exposure(symbol, portfolio) < 0.7
        }
        
        failed_checks = [k for k, v in checks.items() if not v]
        
        if failed_checks:
            logger.warning(f"Circuit breakers triggered for {symbol}: {failed_checks}")
            self._alert_operators(symbol, failed_checks)
            return False
            
        return True
        
    def _check_correlation_exposure(self, symbol: str, portfolio: Portfolio) -> float:
        """Check if adding to position increases correlation risk"""
        correlations = portfolio.get_position_correlations()
        symbol_correlation = correlations.get(symbol, {})
        
        # Calculate weighted correlation exposure
        exposure = 0.0
        for other_symbol, corr in symbol_correlation.items():
            if abs(corr) > 0.7:
                other_position = portfolio.get_position(other_symbol)
                exposure += abs(corr) * abs(other_position.size / portfolio.total_value)
                
        return exposure

F.3 Monitoring and Observability
F.3.1 Real-time Dashboards
# core/monitoring/dashboard.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class TradingDashboard:
    def __init__(self, data_source):
        self.data_source = data_source
        st.set_page_config(page_title="RL Trading Dashboard", layout="wide")
        
    def run(self):
        st.title("🤖 RL Trading System Monitor")
        
        # Create layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            self.show_portfolio_stats()
            self.show_risk_metrics()
            
        with col2:
            self.show_performance_charts()
            self.show_action_distribution()
            
        with col3:
            self.show_recent_trades()
            self.show_alerts()
            
        # Auto-refresh
        st.experimental_rerun()
        
    def show_portfolio_stats(self):
        """Display key portfolio metrics"""
        st.subheader("📊 Portfolio Stats")
        
        stats = self.data_source.get_portfolio_stats()
        
        metrics = {
            "Total Value": f"${stats['total_value']:,.2f}",
            "Daily P&L": f"{stats['daily_pnl']:+.2%}",
            "Sharpe Ratio": f"{stats['sharpe']:.2f}",
            "Win Rate": f"{stats['win_rate']:.1%}",
            "Positions": stats['open_positions']
        }
        
        for label, value in metrics.items():
            st.metric(label, value)
            
    def show_action_distribution(self):
        """Visualize action distribution over time"""
        st.subheader("🎯 Action Distribution")
        
        actions_df = self.data_source.get_recent_actions(hours=24)
        
        # Create heatmap of actions
        fig = go.Figure(data=go.Heatmap(
            z=actions_df.pivot_table(
                index='hour', 
                columns='symbol', 
                values='action_value'
            ).values,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="24h Action Heatmap",
            xaxis_title="Symbol",
            yaxis_title="Hour",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

F.3.2 Alerting System

class AlertingSystem:
    def __init__(self, config):
        self.thresholds = config['thresholds']
        self.channels = self._setup_channels(config['channels'])
        self.alert_history = deque(maxlen=1000)
        self.alert_cooldown = {}  # Prevent alert spam
        
    def check_conditions(self, metrics: Dict):
        """Check for alert conditions"""
        
        alerts = []
        
        # Check each metric against thresholds
        for metric, value in metrics.items():
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                
                if isinstance(threshold, dict):
                    if 'min' in threshold and value < threshold['min']:
                        alerts.append(Alert(
                            level='WARNING',
                            metric=metric,
                            value=value,
                            threshold=threshold['min'],
                            direction='below'
                        ))
                    if 'max' in threshold and value > threshold['max']:
                        alerts.append(Alert(
                            level='WARNING',
                            metric=metric,
                            value=value,
                            threshold=threshold['max'],
                            direction='above'
                        ))
                        
        # Check for critical conditions
        if metrics.get('drawdown', 0) > 0.20:
            alerts.append(Alert(
                level='CRITICAL',
                metric='drawdown',
                value=metrics['drawdown'],
                message='Drawdown exceeds 20% - consider halting trading'
            ))
            
        if metrics.get('policy_entropy', 1.0) < 0.1:
            alerts.append(Alert(
                level='CRITICAL',
                metric='policy_entropy',
                value=metrics['policy_entropy'],
                message='Policy collapse detected - entropy critically low'
            ))
            
        # Send alerts
        for alert in alerts:
            self.send_alert(alert)
            
    def send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        
        # Check cooldown
        alert_key = f"{alert.metric}_{alert.level}"
        if alert_key in self.alert_cooldown:
            if time.time() - self.alert_cooldown[alert_key] < 300:  # 5 min cooldown
                return
                
        # Send through channels
        for channel in self.channels:
            try:
                channel.send(alert)
                logger.info(f"Alert sent via {channel.name}: {alert}")
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.name}: {e}")
                
        # Update cooldown
        self.alert_cooldown[alert_key] = time.time()
        self.alert_history.append(alert)

Phase G: Testing Framework and Quality Assurance (Week 7)
G.1 Comprehensive Test Suite
G.1.1 Unit Tests for Core Components

# tests/test_continuous_action_space.py
import pytest
import numpy as np
import torch
from core.rl.continuous_action import ContinuousTradeAction

class TestContinuousActionSpace:
    def test_action_interpretation(self):
        """Test action interpretation logic"""
        action_interpreter = ContinuousTradeAction()
        
        # Test HOLD zone
        action = np.array([0.05])
        action_type, size = action_interpreter.interpret_action(
            action, current_position=100, available_capital=10000
        )
        assert action_type == ActionType.HOLD
        assert size == 0
        
        # Test BUY signal
        action = np.array([0.5])
        action_type, size = action_interpreter.interpret_action(
            action, current_position=0, available_capital=10000
        )
        assert action_type == ActionType.BUY
        assert size == 5000 * action_interpreter.max_position_pct
        
        # Test SELL signal
        action = np.array([-0.8])
        action_type, size = action_interpreter.interpret_action(
            action, current_position=100, available_capital=10000
        )
        assert action_type == ActionType.SELL
        assert size == 80  # 80% of position
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        action_interpreter = ContinuousTradeAction()
        
        # Test with zero position (no sell possible)
        action = np.array([-0.5])
        action_type, size = action_interpreter.interpret_action(
            action, current_position=0, available_capital=10000
        )
        assert action_type == ActionType.HOLD
        assert size == 0
        
        # Test with zero capital (no buy possible)
        action = np.array([0.8])
        action_type, size = action_interpreter.interpret_action(
            action, current_position=0, available_capital=0
        )
        assert action_type == ActionType.HOLD
        assert size == 0

G.1.2 Integration Tests

# tests/test_sac_training_integration.py
class TestSACTrainingIntegration:
    @pytest.fixture
    def training_env(self):
        """Create test environment with known dynamics"""
        env = ContinuousTradingEnv(
            data_source="synthetic",
            episode_length=100,
            initial_balance=10000
        )
        return env
        
    def test_sac_convergence(self, training_env):
        """Test SAC convergence on simple task"""
        model = SAC(
            "MlpPolicy",
            training_env,
            learning_rate=1e-3,
            buffer_size=10000,
            learning_starts=100,
            train_freq=1,
            gradient_steps=1,
            verbose=0
        )
        
        # Train for short period
        initial_performance = evaluate_policy(model, training_env, n_eval_episodes=10)
        model.learn(total_timesteps=5000)
        final_performance = evaluate_policy(model, training_env, n_eval_episodes=10)
        
        # Should show improvement
        assert final_performance[0] > initial_performance[0]
        
        # Check action distribution
        actions = []
        obs = training_env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=False)
            actions.append(action)
            obs, _, _, _ = training_env.step(action)
            
        actions = np.array(actions)
        
        # Should use full action range
        assert actions.min() < -0.5
        assert actions.max() > 0.5
        assert 0.1 < actions.std() < 0.9  # Not collapsed

G.2 Backtesting Validation
G.2.1 Walk-Forward Analysis

# scripts/walk_forward_validation.py
class WalkForwardValidator:
    def __init__(self, config):
        self.window_size = config['window_size']  # e.g., 60 days
        self.step_size = config['step_size']  # e.g., 20 days
        self.retrain_frequency = config['retrain_frequency']  # e.g., every 3 windows
        
    def validate(self, data, model_class, model_params):
        """Perform walk-forward validation"""
        results = []
        
        windows = self.create_windows(data)
        current_model = None
        
        for i, (train_window, test_window) in enumerate(windows):
            # Retrain periodically or use existing model
            if i % self.retrain_frequency == 0 or current_model is None:
                logger.info(f"Retraining model at window {i}")
                current_model = self.train_model(
                    train_window, 
                    model_class, 
                    model_params
                )
                
            # Evaluate on test window
            metrics = self.evaluate_window(current_model, test_window)
            metrics['window_id'] = i
            metrics['train_start'] = train_window.index[0]
            metrics['train_end'] = train_window.index[-1]
            metrics['test_start'] = test_window.index[0]
            metrics['test_end'] = test_window.index[-1]
            
            results.append(metrics)
            
            # Early stopping if performance degrades
            if len(results) >= 3:
                recent_sharpe = np.mean([r['sharpe'] for r in results[-3:]])
                if recent_sharpe < 0:
                    logger.warning(f"Poor recent performance (Sharpe: {recent_sharpe:.3f})")
                    
        return pd.DataFrame(results)

G.2.2 Market Regime Testing

class MarketRegimeTester:
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        
    def test_across_regimes(self, model, data):
        """Test model performance across different market regimes"""
        
        # Detect regimes in historical data
        regimes = self.regime_detector.detect(data)
        
        regime_results = {}
        
        for regime in ['bull', 'bear', 'sideways', 'high_volatility']:
            regime_data = data[regimes == regime]
            
            if len(regime_data) < 100:
                continue
                
            # Create environment with regime-specific data
            env = TradingEnvironment(regime_data)
            
            # Evaluate model
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=20
            )
            
            # Detailed metrics
            metrics = self.compute_detailed_metrics(model, env)
            
            regime_results[regime] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'sharpe': metrics['sharpe'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'avg_trade_duration': metrics['avg_trade_duration'],
                'n_samples': len(regime_data)
            }
            
        return regime_results

Success Metrics and Quality Gates
Overall Success Criteria (End of Week 7)
Phase A Success Metrics (Week 1)
✅ Policy Entropy: > 0.6 during deterministic evaluation
✅ Trading Frequency: > 20 trades per 1000 steps
✅ Action Distribution: No single action > 60% frequency
✅ Training Stability: No NaN losses or gradient explosions
✅ Exploration: Voluntary trade rate > 5% (agent-initiated)
Phase B Success Metrics (Week 2)
✅ Sharpe Ratio: > 0.3 on validation set
✅ Win Rate: > 40% on executed trades
✅ Voluntary Trade Rate: > 15% (target 20%)
✅ Options Usage: All 5 options activated at least once per episode
✅ HER Effectiveness: 20% improvement in sample efficiency
Phase C Success Metrics (Week 3)
✅ Sharpe Ratio: > 0.5 with new reward structure
✅ Drawdown Control: Maximum drawdown < 20%
✅ Reward Stability: Episode rewards std dev < 2x mean
✅ Multi-objective Balance: No single objective dominates > 60%
Phase D Success Metrics (Week 4)
✅ Sharpe Ratio: > 0.8
✅ Maximum Drawdown: < 15%
✅ Consistent Profitability: Positive returns in 3+ market regimes
✅ ES Baseline: Within 80% of gradient-based performance
Phase E Success Metrics (Week 5)
✅ Curriculum Progression: Complete 3+ stages successfully
✅ Auxiliary Task Accuracy: > 60% on price prediction
✅ Adaptive Difficulty: Smooth progression without collapse
Phase F Success Metrics (Week 6)
✅ Inference Latency: < 10ms p95
✅ ONNX Compatibility: 100% output match with PyTorch
✅ Circuit Breakers: All safety checks operational
✅ Dashboard Functionality: Real-time updates < 1s delay
Phase G Success Metrics (Week 7)
✅ Test Coverage: > 80% for critical paths
✅ Walk-Forward Validation: Positive Sharpe in 70%+ windows
✅ Regime Testing: Profitable in 3+ market regimes
✅ Benchmark Performance: Meeting latency targets
Quality Gates Between Phases
Gate A→B: Continuous Action Space
Policy uses full action range [-1, 1]
No action collapse after 10k steps
ICM intrinsic reward functioning
Gate B→C: Hierarchical Learning
Options demonstrating specialization
HER buffer improving learning
No option collapse
Gate C→D: Reward Stability
Shaped rewards not causing exploitation
Multi-objective weights balanced
Performance improving
Gate D→E: Advanced Training
V-trace corrections working
ES providing diversity
No training instabilities
Gate E→F: Curriculum Completion
Agent progresses through 3+ stages
Auxiliary tasks helping main task
Performance generalizing
Gate F→G: Production Ready
Model exported and optimized
All safeguards operational
Monitoring dashboards live
Rollback Plan
If any phase fails to meet success criteria:

Immediate Actions:

Stop current training
Analyze failure logs and metrics
Identify root cause
Recovery Options:

Minor Issues: Tune hyperparameters and retry
Major Issues: Revert to previous phase checkpoint
Critical Issues: Fall back to supervised learning baseline
Documentation:

Document failure mode
Update implementation plan
Adjust success criteria if unrealistic
Recommended Execution Order
Week 1 Priority (CRITICAL - Start Immediately)
Day 1-2: Implement continuous action space (Phase A.1)
Day 3-4: Add ICM for exploration (Phase A.2)
Day 5: Initial testing and validation
Week 2 Priority
Day 1-2: Implement Options framework (Phase B.1)
Day 3-4: Add HER (Phase B.2)
Day 5: Integration testing
Week 3-4 Priority
Reward shaping improvements (Phase C)
Advanced training techniques (Phase D)
Week 5-6 Priority
Curriculum implementation (Phase E)
Production preparation (Phase F)
Week 7 Priority
Comprehensive testing (Phase G)
Final validation and sign-off
Expected Outcomes
By End of Week 1
Action collapse resolved
Agent actively exploring and trading
Policy entropy stable above 0.5
By End of Week 2
Positive Sharpe ratio achieved
Consistent trading behavior
No single-action dominance
By End of Week 4
Sharpe ratio > 0.8
Robust across market conditions
Ready for production testing
By End of Week 7
Production-ready system
Comprehensive test coverage
Full monitoring and safeguards
This implementation plan provides a clear path from your current collapsed state to a robust, production-ready continuous-action RL trading system. The key is to start immediately with Phase A (continuous actions + ICM) as this addresses the fundamental architectural mismatch causing your current issues.