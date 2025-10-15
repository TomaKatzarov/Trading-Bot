# SAC Advanced Modules Integration Architecture
## Complete Data Flow: Options → HER → V-Trace → Training Loop

**Document Purpose:** Clarify exact integration points for Hierarchical Options, HER, and Advanced Techniques into the SAC training pipeline.

---

## 🔄 COMPLETE INTEGRATION ARCHITECTURE

### High-Level Module Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAC Training Pipeline                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  PHASE D: V-Trace Off-Policy Correction (Optional)     │   │
│  │  • Modifies advantage computation                       │   │
│  │  • Enables distributed training                         │   │
│  └────────────────────────────────────────────────────────┘   │
│                           ▲                                      │
│                           │                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  PHASE C: Multi-Objective Reward Shaper                │   │
│  │  • Computes weighted reward components                  │   │
│  │  • Learns dynamic objective weights                     │   │
│  └────────────────────────────────────────────────────────┘   │
│                           ▲                                      │
│                           │                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  PHASE B: HER Replay Buffer                            │   │
│  │  • Stores transitions with goals                        │   │
│  │  • Relabels failed trajectories                         │   │
│  │  • Provides augmented training batches                  │   │
│  └────────────────────────────────────────────────────────┘   │
│                           ▲                                      │
│                           │                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  PHASE B: Hierarchical Options Controller              │   │
│  │  • Selects high-level option (5 choices)               │   │
│  │  • Executes option's continuous policy                  │   │
│  │  • Manages option termination                           │   │
│  └────────────────────────────────────────────────────────┘   │
│                           ▲                                      │
│                           │                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  PHASE A: Base SAC Algorithm                           │   │
│  │  • Continuous actor-critic                              │   │
│  │  • Automatic temperature tuning                         │   │
│  │  • Twin Q-networks                                      │   │
│  └────────────────────────────────────────────────────────┘   │
│                           ▲                                      │
│                           │                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  PHASE A: Continuous Trading Environment               │   │
│  │  • Box(-1, 1) action space                             │   │
│  │  • Smoothed actions                                     │   │
│  │  • Proportional position sizing                         │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 MODULE 1: HIERARCHICAL OPTIONS INTEGRATION

### A. Options Controller Wrapping SAC

**File:** `training/train_sac_with_options.py` (Phase B extension of `train_sac_continuous.py`)

```python
"""
SAC with Hierarchical Options Integration
Extends base SAC with option-level decision making.
"""

import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from core.rl.options import OptionsController, TradingOptions

class HierarchicalSACTrainer:
    """
    SAC trainer with hierarchical options framework.
    
    INTEGRATION POINT 1: Options wrap SAC's action selection
    INTEGRATION POINT 2: Option-level Q-values guide selection
    INTEGRATION POINT 3: Termination handled at option level
    """
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # PHASE A: Base SAC (already trained)
        self.base_sac = SAC.load(config['base_sac_checkpoint'])
        
        # PHASE B: Add Options Controller
        self.options_controller = OptionsController(
            state_dim=config['state_dim'],
            num_options=5,
            hidden_dim=256
        ).to(config['device'])
        
        # Separate optimizer for options (meta-learning)
        self.options_optimizer = torch.optim.Adam(
            self.options_controller.parameters(),
            lr=config['options_lr']
        )
        
        # Track current option and steps
        self.current_option_idx = None
        self.option_step_count = 0
        self.option_episode_returns = []
        
    def select_action(self, state: np.ndarray) -> float:
        """
        HIERARCHICAL ACTION SELECTION
        
        Flow:
        1. Check if need new option (first step OR option terminated)
        2. If yes: Use OptionsController to select option
        3. Execute option's intra-option policy to get continuous action
        4. Return continuous action [-1, 1] for environment
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Step 1: Check option termination
        if self.current_option_idx is None or self._should_terminate_option(state):
            # Step 2: Select new option
            self.current_option_idx, option_values = self.options_controller.select_option(
                state_tensor,
                deterministic=False  # Explore over options
            )
            self.option_step_count = 0
            print(f"[OPTIONS] Selected option {self.current_option_idx} "
                  f"(Q-value: {option_values[0, self.current_option_idx]:.3f})")
        
        # Step 3: Execute option's policy
        option = self.options_controller.options[self.current_option_idx]
        continuous_action = option.policy(state, self.option_step_count)
        
        self.option_step_count += 1
        
        # Return continuous action for environment
        return continuous_action
    
    def _should_terminate_option(self, state: np.ndarray) -> bool:
        """Check if current option should terminate"""
        if self.current_option_idx is None:
            return True
        
        option = self.options_controller.options[self.current_option_idx]
        terminate_prob = option.termination_probability(state, self.option_step_count)
        
        # Stochastic termination
        return np.random.random() < terminate_prob
    
    def train_step(self, batch: dict):
        """
        HIERARCHICAL TRAINING
        
        Two-level optimization:
        1. Train base SAC on continuous actions (low-level)
        2. Train options controller on option Q-values (high-level)
        """
        
        # LEVEL 1: Train base SAC (continuous actions)
        # This uses standard SAC loss on actual executed actions
        sac_loss = self._train_base_sac(batch)
        
        # LEVEL 2: Train options controller (meta-policy)
        # This uses option-level returns to update selection policy
        options_loss = self._train_options_controller(batch)
        
        return {
            'sac_loss': sac_loss,
            'options_loss': options_loss,
            'current_option': self.current_option_idx,
            'option_steps': self.option_step_count
        }
    
    def _train_base_sac(self, batch):
        """Standard SAC training on continuous actions"""
        # Delegate to SB3's SAC implementation
        self.base_sac.train(gradient_steps=1)
        return self.base_sac.logger.name_to_value.get('train/loss', 0)
    
    def _train_options_controller(self, batch):
        """
        Train options controller using option-level returns.
        
        Goal: Maximize cumulative return of selected options
        Uses policy gradient on option selection logits
        """
        states = torch.FloatTensor(batch['states'])
        option_indices = torch.LongTensor(batch['option_indices'])
        option_returns = torch.FloatTensor(batch['option_returns'])
        
        # Forward pass: Get option logits and values
        option_logits = self.options_controller.option_selector(states)
        option_values = self.options_controller.option_value(states)
        
        # Policy gradient loss (REINFORCE-style)
        log_probs = torch.log_softmax(option_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, option_indices.unsqueeze(-1)).squeeze()
        
        # Advantage = actual return - predicted value (baseline)
        advantages = option_returns - option_values.gather(1, option_indices.unsqueeze(-1)).squeeze()
        
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        
        # Value function loss
        value_loss = torch.nn.functional.mse_loss(
            option_values.gather(1, option_indices.unsqueeze(-1)).squeeze(),
            option_returns
        )
        
        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backprop
        self.options_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.options_controller.parameters(), 1.0)
        self.options_optimizer.step()
        
        return total_loss.item()


# USAGE EXAMPLE
def train_hierarchical_sac():
    """Main training loop with options"""
    
    config = {
        'base_sac_checkpoint': 'models/phase_a2_sac/final_model.zip',
        'state_dim': 512,
        'options_lr': 1e-4,
        'device': 'cuda'
    }
    
    env = make_continuous_env(symbol='SPY')
    trainer = HierarchicalSACTrainer(env, config)
    
    for episode in range(1000):
        state = env.reset()
        episode_return = 0
        
        while True:
            # Hierarchical action selection
            action = trainer.select_action(state)
            
            next_state, reward, done, info = env.step(action)
            
            # Store transition with option metadata
            transition = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'option_idx': trainer.current_option_idx,
                'option_step': trainer.option_step_count
            }
            
            # Train both levels
            if episode > 10:  # Warmup
                trainer.train_step(transition)
            
            episode_return += reward
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode}: Return = {episode_return:.2f}")
```

**Key Integration Points:**
1. **Action Selection:** Options controller selects high-level option → option executes low-level continuous action
2. **Training:** Two-level optimization (base SAC for actions + options controller for meta-policy)
3. **Termination:** Options manage their own lifecycle, SAC sees continuous action stream

---

## 📦 MODULE 2: HER REPLAY BUFFER INTEGRATION

### B. HER Buffer Replacing Standard Replay

**File:** `training/train_sac_with_options.py` (continued integration)

```python
"""
Hindsight Experience Replay Integration
Replaces standard SAC replay buffer with goal-conditioned HER buffer.
"""

from core.rl.replay import HERReplayBuffer

class HierarchicalSACWithHER(HierarchicalSACTrainer):
    """
    Extends HierarchicalSACTrainer with HER replay buffer.
    
    INTEGRATION POINT 1: Replace SB3's standard replay buffer
    INTEGRATION POINT 2: Episode-level trajectory storage
    INTEGRATION POINT 3: Goal relabeling during sampling
    """
    
    def __init__(self, env, config):
        super().__init__(env, config)
        
        # PHASE B: Replace standard buffer with HER
        self.replay_buffer = HERReplayBuffer(
            capacity=config.get('buffer_size', 100000),
            her_ratio=config.get('her_ratio', 0.8)
        )
        
        # Track episode trajectory for HER
        self.episode_trajectory = []
        self.episode_goals = []
        
    def collect_rollout(self):
        """
        MODIFIED ROLLOUT COLLECTION
        
        Changes from base SAC:
        1. Store full episode trajectory (not individual transitions)
        2. Track cumulative returns for goal relabeling
        3. Process episode with HER after termination
        """
        
        state = self.env.reset()
        episode_return = 0
        done = False
        
        # Clear episode buffers
        self.episode_trajectory = []
        self.episode_goals = []
        
        while not done:
            # Hierarchical action selection (from Module 1)
            action = self.select_action(state)
            
            next_state, reward, done, info = self.env.step(action)
            
            episode_return += reward
            
            # Store transition WITH goal metadata
            transition = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'option_idx': self.current_option_idx,
                'desired_goal': info.get('desired_goal', 0.02),  # 2% target return
                'achieved_goal': episode_return  # Cumulative return so far
            }
            
            # Add to episode buffer (not main buffer yet)
            self.episode_trajectory.append(transition)
            self.episode_goals.append(episode_return)
            
            state = next_state
        
        # CRITICAL: Process entire episode with HER
        self._process_episode_with_her()
        
        return episode_return
    
    def _process_episode_with_her(self):
        """
        Apply HER to episode trajectory.
        
        Process:
        1. Add original trajectory to buffer
        2. For each transition, create k=4 virtual goals
        3. Relabel transitions with achieved goals
        4. Add augmented transitions to buffer
        
        Result: 1 episode → ~5× transitions in buffer (1 real + 4 virtual per step)
        """
        
        # Store original episode
        for transition in self.episode_trajectory:
            self.replay_buffer.store_transition(**transition)
        
        # Apply HER relabeling
        self.replay_buffer.store_episode()
        
        print(f"[HER] Episode processed: {len(self.episode_trajectory)} transitions "
              f"→ {len(self.replay_buffer)} total in buffer")
    
    def train_step(self):
        """
        MODIFIED TRAINING STEP
        
        Changes from base SAC:
        1. Sample from HER buffer (includes relabeled goals)
        2. Goal-conditioned reward computation
        3. Both option-level and action-level training
        """
        
        if len(self.replay_buffer) < self.config['batch_size']:
            return None  # Not enough samples yet
        
        # Sample batch from HER buffer
        batch = self.replay_buffer.sample(self.config['batch_size'])
        
        # Train base SAC with goal-conditioned batch
        sac_metrics = self._train_base_sac_goal_conditioned(batch)
        
        # Train options controller
        options_metrics = self._train_options_controller(batch)
        
        return {**sac_metrics, **options_metrics}
    
    def _train_base_sac_goal_conditioned(self, batch):
        """
        Train SAC with goal-augmented states.
        
        State format: [original_state, desired_goal] (concatenated)
        Reward: Distance to goal (computed by HER)
        """
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']  # Already computed by HER
        next_states = batch['next_states']
        dones = batch['dones']
        desired_goals = batch['desired_goals']
        
        # Augment states with goals
        states_aug = np.concatenate([states, desired_goals[:, None]], axis=-1)
        next_states_aug = np.concatenate([next_states, desired_goals[:, None]], axis=-1)
        
        # Standard SAC update with augmented states
        # (Delegate to SB3 SAC implementation)
        self.base_sac.replay_buffer.add(
            states_aug, next_states_aug, actions, rewards, dones, [{}]*len(states)
        )
        
        self.base_sac.train(gradient_steps=1)
        
        return {
            'sac/actor_loss': self.base_sac.logger.name_to_value.get('train/actor_loss', 0),
            'sac/critic_loss': self.base_sac.logger.name_to_value.get('train/critic_loss', 0),
            'her/buffer_size': len(self.replay_buffer)
        }


# USAGE EXAMPLE
def train_hierarchical_sac_with_her():
    """Training loop with both Options and HER"""
    
    config = {
        'base_sac_checkpoint': 'models/phase_a2_sac/final_model.zip',
        'state_dim': 512,
        'buffer_size': 200000,  # Larger for HER
        'her_ratio': 0.8,
        'batch_size': 256,
        'device': 'cuda'
    }
    
    env = make_continuous_env(symbol='SPY')
    trainer = HierarchicalSACWithHER(env, config)
    
    for episode in range(1000):
        # Collect full episode trajectory
        episode_return = trainer.collect_rollout()
        
        # Train on augmented buffer
        if episode > 10:  # Warmup
            for _ in range(10):  # Multiple gradient steps per episode
                metrics = trainer.train_step()
                if metrics:
                    print(f"  SAC loss: {metrics['sac/actor_loss']:.4f}, "
                          f"Buffer: {metrics['her/buffer_size']}")
        
        print(f"Episode {episode}: Return = {episode_return:.2f}")
```

**Key Integration Points:**
1. **Buffer Replacement:** HER buffer stores episode trajectories, not individual transitions
2. **Relabeling:** Failed episodes become successful with virtual goals
3. **Goal Augmentation:** States concatenated with desired goals for conditioning

---

## 📦 MODULE 3: V-TRACE OFF-POLICY CORRECTION (OPTIONAL)

### C. V-Trace for Distributed Training

**File:** `training/train_sac_distributed.py` (Phase D advanced extension)

```python
"""
V-Trace Off-Policy Correction Integration
Enables distributed actor-learner architecture (IMPALA-style).
"""

import ray
import torch
from core.rl.algorithms import VTraceReturns

@ray.remote
class RemoteActor:
    """
    Remote actor for distributed rollout collection.
    
    INTEGRATION POINT: Runs on separate CPU workers
    Collects trajectories using behavior policy
    """
    
    def __init__(self, env_config, actor_id):
        self.env = make_continuous_env(**env_config)
        self.actor_id = actor_id
        self.policy = None  # Updated by learner
        
    def collect_trajectory(self, policy_weights, num_steps=100):
        """Collect trajectory with current policy"""
        
        # Update local policy
        self.policy.load_state_dict(policy_weights)
        
        trajectory = []
        state = self.env.reset()
        
        for _ in range(num_steps):
            # Get action from behavior policy
            with torch.no_grad():
                action, log_prob = self.policy.get_action_and_log_prob(state)
            
            next_state, reward, done, info = self.env.step(action)
            
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'behavior_log_prob': log_prob,  # For importance sampling
                'info': info
            })
            
            if done:
                state = self.env.reset()
            else:
                state = next_state
        
        return trajectory


class DistributedSACLearner:
    """
    Central learner with V-Trace off-policy correction.
    
    INTEGRATION POINT 1: Receives trajectories from distributed actors
    INTEGRATION POINT 2: Applies V-Trace corrections to advantages
    INTEGRATION POINT 3: Updates policy and broadcasts to actors
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        # Initialize V-Trace
        self.vtrace = VTraceReturns(
            gamma=config.get('gamma', 0.99),
            rho_bar=config.get('rho_bar', 1.0),
            c_bar=config.get('c_bar', 1.0)
        )
        
        # Base SAC policy (target policy)
        self.policy = self._build_policy()
        self.value_network = self._build_value_network()
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=3e-4)
        
        # Initialize distributed actors
        ray.init()
        self.actors = [
            RemoteActor.remote(config['env'], i)
            for i in range(config['num_actors'])
        ]
        
    def train_step(self):
        """
        DISTRIBUTED TRAINING STEP WITH V-TRACE
        
        Flow:
        1. Broadcast current policy to actors
        2. Collect trajectories in parallel (behavior policy)
        3. Compute V-Trace corrections (target vs behavior)
        4. Update policy with corrected advantages
        5. Repeat
        """
        
        # Step 1: Get current policy weights
        policy_weights = self.policy.state_dict()
        
        # Step 2: Collect trajectories in parallel
        trajectory_futures = [
            actor.collect_trajectory.remote(policy_weights, num_steps=100)
            for actor in self.actors
        ]
        trajectories = ray.get(trajectory_futures)
        
        # Step 3: Process trajectories with V-Trace
        total_loss = 0
        for trajectory in trajectories:
            loss = self._train_on_trajectory_with_vtrace(trajectory)
            total_loss += loss
        
        avg_loss = total_loss / len(trajectories)
        
        return {
            'vtrace/policy_loss': avg_loss,
            'vtrace/num_trajectories': len(trajectories),
            'vtrace/total_steps': sum(len(t) for t in trajectories)
        }
    
    def _train_on_trajectory_with_vtrace(self, trajectory):
        """
        Apply V-Trace correction and update policy.
        
        V-Trace corrects for off-policy distribution mismatch:
        - Behavior policy: What actors used to collect data
        - Target policy: What learner is currently optimizing
        """
        
        # Extract trajectory components
        states = torch.FloatTensor([t['state'] for t in trajectory])
        actions = torch.FloatTensor([t['action'] for t in trajectory])
        rewards = torch.FloatTensor([t['reward'] for t in trajectory])
        next_states = torch.FloatTensor([t['next_state'] for t in trajectory])
        dones = torch.FloatTensor([t['done'] for t in trajectory])
        behavior_log_probs = torch.FloatTensor([t['behavior_log_prob'] for t in trajectory])
        
        # Get target policy log probs (current policy)
        with torch.no_grad():
            target_log_probs = self.policy.get_log_prob(states, actions)
        
        # Get value estimates
        values = self.value_network(states).squeeze()
        with torch.no_grad():
            bootstrap_value = self.value_network(next_states[-1]).squeeze()
        
        # Convert to logits for V-Trace (V-Trace expects logits, not log probs)
        # For continuous actions, we approximate with Gaussian logits
        behavior_logits = behavior_log_probs.unsqueeze(-1)
        target_logits = target_log_probs.unsqueeze(-1)
        
        # Compute V-Trace returns and advantages
        vs, advantages = self.vtrace.compute_vtrace_returns(
            behavior_logits=behavior_logits.unsqueeze(0),  # (T, B, A) format
            target_logits=target_logits.unsqueeze(0),
            actions=actions.long().unsqueeze(0),  # Discretize for V-Trace
            rewards=rewards.unsqueeze(0),
            values=values.unsqueeze(0),
            bootstrap_value=bootstrap_value.unsqueeze(0),
            masks=(1 - dones).unsqueeze(0)
        )
        
        # Policy gradient loss with V-Trace advantages
        policy_loss = -(target_log_probs * advantages.squeeze().detach()).mean()
        
        # Value function loss (match V-Trace targets)
        value_loss = torch.nn.functional.mse_loss(values, vs.squeeze())
        
        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return total_loss.item()


# USAGE EXAMPLE
def train_distributed_sac():
    """Distributed training with V-Trace"""
    
    config = {
        'env': {'symbol': 'SPY', 'data_root': 'data/historical'},
        'num_actors': 8,  # 8 parallel workers
        'gamma': 0.99,
        'rho_bar': 1.0,
        'c_bar': 1.0,
        'device': 'cuda'
    }
    
    learner = DistributedSACLearner(config)
    
    for iteration in range(1000):
        metrics = learner.train_step()
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: "
                  f"Loss = {metrics['vtrace/policy_loss']:.4f}, "
                  f"Steps = {metrics['vtrace/total_steps']}")
```

**Key Integration Points:**
1. **Distributed Architecture:** Separate actors (CPU) and learner (GPU)
2. **Off-Policy Correction:** V-Trace handles behavior vs target policy mismatch
3. **Importance Sampling:** Automatically weights trajectories based on policy divergence

---

## 🔗 COMPLETE INTEGRATION FLOW DIAGRAM

### End-to-End Data Flow

```
EPISODE START
     │
     ├─> [1] OptionsController.select_option(state)
     │         │
     │         └─> Returns: option_idx (0-4)
     │
     ├─> [2] TradingOption[option_idx].policy(state, step)
     │         │
     │         └─> Returns: continuous_action [-1, 1]
     │
     ├─> [3] Environment.step(continuous_action)
     │         │
     │         └─> Returns: next_state, reward, done, info
     │
     ├─> [4] Multi-Objective Reward Shaper (Phase C)
     │         │
     │         ├─> Computes 5 objective components
     │         ├─> Learns dynamic weights based on state
     │         └─> Returns: shaped_reward
     │
     ├─> [5] ICM Module (Phase A.3)
     │         │
     │         ├─> Computes prediction error
     │         └─> Returns: intrinsic_reward
     │
     ├─> [6] Total reward = extrinsic + intrinsic
     │
     ├─> [7] HER Buffer.store_transition()
     │         │
     │         ├─> Stores: (state, action, reward, next_state, done, goal)
     │         └─> Tracks cumulative episode return
     │
     ├─> [8] Check option termination
     │         │
     │         ├─> If terminate: Go back to step [1]
     │         └─> Else: Continue with same option
     │
     ├─> [9] Check episode done
     │         │
     │         ├─> If done: HER.store_episode()
     │         │             │
     │         │             └─> Relabels trajectory with virtual goals
     │         │                 Creates 5× augmented transitions
     │         │
     │         └─> Else: Loop to step [2]
     │
EPISODE END
     │
     └─> [10] Training Phase (after enough episodes)
           │
           ├─> Sample batch from HER buffer
           │
           ├─> If Distributed (Phase D):
           │     │
           │     ├─> V-Trace.compute_vtrace_returns()
           │     │     │
           │     │     └─> Corrects for off-policy collection
           │     │
           │     └─> Update with corrected advantages
           │
           ├─> Train Base SAC (action-level)
           │     │
           │     ├─> Actor loss: Policy gradient
           │     ├─> Critic loss: TD error
           │     └─> Temperature: Automatic tuning
           │
           └─> Train Options Controller (meta-level)
                 │
                 ├─> Policy gradient on option selection
                 └─> Value function for options
```

---

## 🎯 CRITICAL INTEGRATION CHECKLIST

### Phase B Integration (Options + HER)

- [ ] **Replace action selection:** `env.step(sac.predict())` → `env.step(options.execute())`
- [ ] **Replace replay buffer:** `SAC.replay_buffer` → `HERReplayBuffer`
- [ ] **Add episode tracking:** Store full trajectories, not individual transitions
- [ ] **Modify reward:** Add option-level returns for meta-learning
- [ ] **Update training loop:** Two-level optimization (SAC + options)

### Phase C Integration (Multi-Objective)

- [ ] **Wrap environment reward:** `env.step()` returns base reward → multi-objective processes it
- [ ] **Add state to reward:** Reward shaper needs current state for dynamic weighting
- [ ] **Log components:** Track all 5 objective contributions separately
- [ ] **Train weight network:** Additional optimizer for learned weights

### Phase D Integration (V-Trace - Optional)

- [ ] **Setup distributed:** Initialize Ray, create remote actors
- [ ] **Store behavior policy:** Save log probs during collection
- [ ] **Compute corrections:** Apply V-Trace before policy update
- [ ] **Broadcast weights:** Send updated policy to all actors

---

## 📝 EXAMPLE: COMPLETE TRAINING SCRIPT

**File:** `training/train_sac_all_modules.py`

```python
"""
Complete integration: Base SAC + Options + HER + Multi-Objective + V-Trace
"""

def train_full_stack(config):
    """
    Training with all advanced modules integrated.
    
    Phases executed:
    - Phase A: Base SAC (already trained, load checkpoint)
    - Phase B: Add Options + HER
    - Phase C: Add Multi-Objective Rewards
    - Phase D: Enable V-Trace (optional, if distributed)
    """
    
    # Phase A: Load base SAC
    base_sac = SAC.load(config['phase_a_checkpoint'])
    
    # Phase B: Wrap with Options
    options_controller = OptionsController(state_dim=512, num_options=5)
    
    # Phase B: Replace buffer with HER
    replay_buffer = HERReplayBuffer(capacity=200000, her_ratio=0.8)
    
    # Phase C: Add multi-objective reward shaper
    reward_shaper = MultiObjectiveRewardShaper(config)
    
    # Phase D: Optional V-Trace for distributed
    if config.get('distributed', False):
        vtrace = VTraceReturns()
        actors = initialize_remote_actors(config['num_actors'])
    
    # Training loop
    for episode in range(config['num_episodes']):
        
        # Collect episode trajectory
        trajectory = []
        state = env.reset()
        episode_return = 0
        current_option = None
        
        while True:
            # [1] Option selection
            if current_option is None or should_terminate_option():
                current_option, _ = options_controller.select_option(state)
            
            # [2] Execute option policy
            action = options_controller.options[current_option].policy(state)
            
            # [3] Environment step
            next_state, base_reward, done, info = env.step(action)
            
            # [4] Multi-objective reward shaping
            shaped_reward, components = reward_shaper.compute_reward(
                state, action, next_state, info
            )
            
            # [5] ICM intrinsic reward (if enabled)
            if config.get('icm_enabled', False):
                intrinsic_reward = icm.compute_reward(state, action, next_state)
                total_reward = 0.9 * shaped_reward + 0.1 * intrinsic_reward
            else:
                total_reward = shaped_reward
            
            # [6-7] Store transition
            trajectory.append({
                'state': state,
                'action': action,
                'reward': total_reward,
                'next_state': next_state,
                'done': done,
                'option': current_option,
                'goal': info.get('goal', 0.02)
            })
            
            episode_return += total_reward
            state = next_state
            
            if done:
                break
        
        # [9] Process episode with HER
        for transition in trajectory:
            replay_buffer.store_transition(**transition)
        replay_buffer.store_episode()  # Apply relabeling
        
        # [10] Training (if enough data)
        if len(replay_buffer) >= config['batch_size']:
            
            # Sample batch
            batch = replay_buffer.sample(config['batch_size'])
            
            # Train with V-Trace if distributed
            if config.get('distributed', False):
                # Collect from remote actors
                remote_trajectories = collect_from_actors(actors)
                # Apply V-Trace corrections
                corrected_advantages = vtrace.compute_vtrace_returns(...)
                # Update with corrections
                train_with_vtrace(base_sac, corrected_advantages)
            else:
                # Standard SAC update
                base_sac.train(batch)
            
            # Train options controller
            train_options_controller(options_controller, batch)
            
            # Train multi-objective weight network
            train_weight_network(reward_shaper, batch)
        
        # Logging
        if episode % 10 == 0:
            print(f"Episode {episode}: Return = {episode_return:.2f}")
            print(f"  Reward components: {components}")
            print(f"  Buffer size: {len(replay_buffer)}")
            print(f"  Current option: {current_option}")

if __name__ == "__main__":
    config = load_config('training/config_templates/phase_all_modules.yaml')
    train_full_stack(config)
```

---

## 🔍 DEBUGGING INTEGRATION ISSUES

### Common Integration Problems

**Problem 1: Options never terminate**
- **Symptom:** Same option runs for entire episode
- **Fix:** Check `termination_probability()` returns reasonable values (0.1-0.8)
- **Debug:** Add logging in `_should_terminate_option()`

**Problem 2: HER buffer grows too fast**
- **Symptom:** Out of memory after 100 episodes
- **Fix:** Reduce `her_k` from 4 to 2, or reduce `buffer_size`
- **Debug:** Monitor `len(replay_buffer)` vs `capacity`

**Problem 3: V-Trace advantages explode**
- **Symptom:** NaN in policy loss
- **Fix:** Clip importance weights (`rho_bar=1.0` → `rho_bar=0.5`)
- **Debug:** Log `rhos` in V-Trace computation

**Problem 4: Multi-objective weights collapse to single objective**
- **Symptom:** One weight >0.95, others near 0
- **Fix:** Increase weight network regularization, reduce learning rate
- **Debug:** Log weight entropy (should be >0.5)

---

## ✅ INTEGRATION VALIDATION

### Validation Tests

```python
# Test 1: Options integration
def test_options_integration():
    env = make_continuous_env('SPY')
    options = OptionsController(state_dim=512, num_options=5)
    
    state = env.reset()
    action = options.execute_option(state, option_idx=0)
    
    assert -1.0 <= action <= 1.0, "Action out of bounds"
    assert options.current_option_idx is not None, "Option not selected"
    print("✅ Options integration validated")

# Test 2: HER buffer integration
def test_her_integration():
    buffer = HERReplayBuffer(capacity=1000, her_ratio=0.8)
    
    # Store episode
    for _ in range(50):
        buffer.store_transition(
            state=np.random.randn(512),
            action=np.random.randn(1),
            reward=np.random.randn(),
            next_state=np.random.randn(512),
            done=False,
            info={'cumulative_return': np.random.randn()}
        )
    buffer.store_episode()
    
    # Sample batch
    batch = buffer.sample(32)
    
    assert len(buffer) > 50, "HER didn't augment buffer"
    assert batch is not None, "Sampling failed"
    print(f"✅ HER integration validated (buffer size: {len(buffer)})")

# Test 3: V-Trace integration
def test_vtrace_integration():
    vtrace = VTraceReturns()
    
    T, B, A = 10, 4, 7
    behavior_logits = torch.randn(T, B, A)
    target_logits = torch.randn(T, B, A)
    actions = torch.randint(0, A, (T, B))
    rewards = torch.randn(T, B)
    values = torch.randn(T, B)
    bootstrap = torch.randn(B)
    masks = torch.ones(T, B)
    
    vs, advantages = vtrace.compute_vtrace_returns(
        behavior_logits, target_logits, actions,
        rewards, values, bootstrap, masks
    )
    
    assert not torch.isnan(vs).any(), "NaN in V-Trace returns"
    assert not torch.isnan(advantages).any(), "NaN in advantages"
    print("✅ V-Trace integration validated")

# Run all tests
if __name__ == "__main__":
    test_options_integration()
    test_her_integration()
    test_vtrace_integration()
    print("\n✅ ALL INTEGRATION TESTS PASSED")
```

---

**This document provides the complete integration architecture showing exactly how Options, HER, and V-Trace modules plug into the SAC training pipeline. Each module has clear integration points, data flow diagrams, and validation tests.**