"""Advanced training utilities for SAC continuous trading.

This module provides curriculum learning, critic stabilization, and action
collapse detection to improve training stability and performance.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CurriculumSchedule:
    """Adaptive difficulty scaling for training progression.
    
    Gradually increases position sizes and multipliers as the agent
    demonstrates proficiency, preventing early-stage collapse from
    over-aggressive incentives.
    """

    def __init__(self):
        """Initialize curriculum schedule with conservative starting point."""
        self.checkpoints = {
            -1: {  # Sentinel for initial state
                'position_size_large_multiplier': 0.6,  # Heavy penalty initially
                'max_position_pct': 0.05,  # Small positions only
                'win_bonus_multiplier': 1.5,  # Low variance rewards
                'description': 'Phase 0: Conservative training start',
            },
            0: {
                'position_size_large_multiplier': 0.6,  # Heavy penalty initially
                'max_position_pct': 0.05,  # Small positions only
                'win_bonus_multiplier': 1.5,  # Low variance rewards
                'description': 'Phase 0: Conservative training start',
            },
            100_000: {
                'position_size_large_multiplier': 0.7,
                'max_position_pct': 0.08,
                'win_bonus_multiplier': 2.0,
                'description': 'Phase 1: Modest position sizing',
            },
            200_000: {
                'position_size_large_multiplier': 0.8,
                'max_position_pct': 0.10,
                'win_bonus_multiplier': 2.5,
                'description': 'Phase 2: Standard configuration',
            },
            300_000: {
                'position_size_large_multiplier': 0.9,  # Only if Sharpe > 0
                'max_position_pct': 0.12,
                'win_bonus_multiplier': 3.0,
                'description': 'Phase 3: Advanced (requires positive Sharpe)',
            }
        }
        
        self.current_phase = 0
        self.last_checkpoint = -1  # Start before any checkpoint
    
    def get_config_updates(
        self, 
        current_step: int, 
        current_sharpe: float = -999.0
    ) -> Optional[Dict]:
        """Get configuration updates based on training progress.
        
        Args:
            current_step: Current training timestep
            current_sharpe: Current Sharpe ratio (for gating advanced phases)
            
        Returns:
            Dictionary of config updates if checkpoint reached, None otherwise
        """
        # Find applicable checkpoint
        applicable_checkpoint = 0
        for checkpoint_step in sorted(self.checkpoints.keys()):
            if current_step >= checkpoint_step:
                applicable_checkpoint = checkpoint_step
        
        # Check if we've progressed to a new phase
        if applicable_checkpoint > self.last_checkpoint:
            config = self.checkpoints[applicable_checkpoint].copy()
            
            # Gate Phase 3 on positive Sharpe
            if applicable_checkpoint == 300_000:
                if current_sharpe < 0.0:
                    logger.warning(
                        "Phase 3 curriculum (300K) requires positive Sharpe. "
                        f"Current Sharpe: {current_sharpe:.3f}. Staying in Phase 2."
                    )
                    return None
            
            self.last_checkpoint = applicable_checkpoint
            self.current_phase = list(self.checkpoints.keys()).index(applicable_checkpoint)
            
            logger.info(
                f"Curriculum advancing to {config['description']} at step {current_step}"
            )
            
            # Remove description from config updates
            config.pop('description', None)
            return config
        
        return None
    
    def get_current_phase_info(self) -> Dict:
        """Get information about current training phase."""
        checkpoint = list(self.checkpoints.keys())[self.current_phase]
        return {
            'phase': self.current_phase,
            'checkpoint': checkpoint,
            'config': self.checkpoints[checkpoint]
        }


class CriticStabilizer:
    """Monitor and stabilize critic loss during training.
    
    Detects critic explosion and applies emergency interventions to prevent
    value function collapse.
    """

    def __init__(
        self, 
        baseline_loss: float = 500.0, 
        explosion_threshold: float = 3.0,
        window_size: int = 5
    ):
        """Initialize critic stabilizer.
        
        Args:
            baseline_loss: Expected baseline critic loss
            explosion_threshold: Multiplier threshold for explosion detection
            window_size: Number of recent losses to track
        """
        self.baseline = baseline_loss
        self.threshold = explosion_threshold
        self.window_size = window_size
        
        self.loss_history = []
        self.intervention_count = 0
        self.last_intervention_step = -1
    
    def check_and_recover(
        self, 
        current_loss: float, 
        current_step: int,
        min_steps_between_interventions: int = 10_000
    ) -> Optional[Dict]:
        """Check for critic explosion and return recovery actions.
        
        Args:
            current_loss: Current critic loss value
            current_step: Current training timestep
            min_steps_between_interventions: Minimum steps before re-intervention
            
        Returns:
            Dictionary of recovery actions if explosion detected, None otherwise
        """
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        
        # Need sufficient history
        if len(self.loss_history) < self.window_size:
            return None
        
        # Check if recent average exceeds threshold
        recent_avg = np.mean(self.loss_history)
        
        if recent_avg > self.baseline * self.threshold:
            # Prevent intervention spam
            if current_step - self.last_intervention_step < min_steps_between_interventions:
                logger.warning(
                    f"Critic explosion detected (loss={recent_avg:.1f}) but "
                    f"too soon since last intervention (step {self.last_intervention_step})"
                )
                return None
            
            self.intervention_count += 1
            self.last_intervention_step = current_step
            
            logger.error(
                f"🚨 CRITIC EXPLOSION DETECTED at step {current_step}! "
                f"Loss: {recent_avg:.1f} (threshold: {self.baseline * self.threshold:.1f}). "
                f"Applying emergency intervention #{self.intervention_count}"
            )
            
            return {
                'reduce_lr': 0.5,  # Cut learning rate in half
                'increase_tau': 2.0,  # Double tau (slower target updates)
                'rollback_steps': 10_000,  # Suggest reverting to earlier checkpoint
                'reduce_batch_size': 0.75,  # Optionally reduce batch size
                'intervention_type': 'critic_explosion',
                'loss_value': recent_avg,
            }
        
        return None
    
    def get_stats(self) -> Dict:
        """Get stabilizer statistics."""
        return {
            'intervention_count': self.intervention_count,
            'last_intervention_step': self.last_intervention_step,
            'recent_losses': self.loss_history,
            'recent_avg_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
        }


class ActionCollapseDetector:
    """Monitor action distribution and detect collapse.
    
    Identifies when the agent's action distribution becomes too concentrated
    in a single bin, indicating exploration failure or local optima.
    """

    def __init__(
        self, 
        collapse_threshold: float = 0.5,
        warning_threshold: float = 0.4,
        window_size: int = 1000
    ):
        """Initialize action collapse detector.
        
        Args:
            collapse_threshold: Max concentration triggering intervention
            warning_threshold: Concentration level for warnings
            window_size: Number of recent actions to track
        """
        self.collapse_threshold = collapse_threshold
        self.warning_threshold = warning_threshold
        self.window_size = window_size
        
        self.action_window = []
        self.intervention_count = 0
        self.warning_count = 0
    
    def update(self, action: int) -> None:
        """Update action history with latest action.
        
        Args:
            action: Discrete action taken (0-19 for continuous binned)
        """
        self.action_window.append(action)
        if len(self.action_window) > self.window_size:
            self.action_window.pop(0)
    
    def detect_collapse(self, current_step: int) -> Optional[Dict]:
        """Detect action collapse and return intervention.
        
        Args:
            current_step: Current training timestep
            
        Returns:
            Dictionary of intervention actions if collapse detected, None otherwise
        """
        if len(self.action_window) < 100:  # Need minimum history
            return None
        
        # Calculate action distribution
        action_array = np.array(self.action_window)
        unique, counts = np.unique(action_array, return_counts=True)
        
        if len(counts) == 0:
            return None
        
        max_concentration = float(np.max(counts)) / float(len(self.action_window))
        dominant_action = int(unique[np.argmax(counts)])
        
        # Check for warnings
        if max_concentration > self.warning_threshold:
            self.warning_count += 1
            if self.warning_count % 10 == 1:  # Log every 10th warning
                logger.warning(
                    f"⚠️  Action concentration warning at step {current_step}: "
                    f"{max_concentration:.1%} in action {dominant_action} "
                    f"(threshold: {self.warning_threshold:.1%})"
                )
        
        # Check for collapse
        if max_concentration > self.collapse_threshold:
            self.intervention_count += 1
            
            logger.error(
                f"🚨 ACTION COLLAPSE DETECTED at step {current_step}! "
                f"Concentration: {max_concentration:.1%} in action {dominant_action} "
                f"(threshold: {self.collapse_threshold:.1%}). "
                f"Intervention #{self.intervention_count}"
            )
            
            return {
                'increase_entropy_coef': 1.5,  # Boost exploration
                'add_action_noise': 0.1,  # Inject uniform noise
                'reduce_batch_size': 0.5,  # Increase gradient variance
                'reset_optimizer': True,  # Reset optimizer momentum
                'intervention_type': 'action_collapse',
                'concentration': max_concentration,
                'dominant_action': dominant_action,
            }
        
        return None
    
    def get_distribution(self) -> Dict:
        """Get current action distribution statistics."""
        if not self.action_window:
            return {}
        
        action_array = np.array(self.action_window)
        unique, counts = np.unique(action_array, return_counts=True)
        
        distribution = {int(a): int(c) for a, c in zip(unique, counts)}
        max_concentration = float(np.max(counts)) / float(len(self.action_window))
        dominant_action = int(unique[np.argmax(counts)])
        
        # Calculate entropy
        probs = counts / len(self.action_window)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return {
            'distribution': distribution,
            'max_concentration': max_concentration,
            'dominant_action': dominant_action,
            'entropy': entropy,
            'window_size': len(self.action_window),
            'unique_actions': len(unique),
            'intervention_count': self.intervention_count,
            'warning_count': self.warning_count,
        }


__all__ = [
    "CurriculumSchedule",
    "CriticStabilizer", 
    "ActionCollapseDetector",
]
