"""Test script to validate Phase 2 recovery implementations."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from core.rl.environments.reward_shaper import RewardConfig, RewardShaper
from training.advanced_training_utils import (
    CurriculumSchedule,
    CriticStabilizer,
    ActionCollapseDetector,
)


def test_config_loading():
    """Test that the updated config loads correctly."""
    print("=" * 80)
    print("TEST 1: Config Loading")
    print("=" * 80)
    
    config_path = Path("training/config_templates/phase_a2_sac_sharpe.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    reward_cfg = config_dict['environment']['reward_config']
    
    # Check critical fixes - Updated to match current phase_a2 simplified config
    assert reward_cfg['position_size_small_multiplier'] == 1.0, "Position sizing should be neutral in phase_a2!"
    assert reward_cfg['position_size_large_multiplier'] == 1.0, "Position sizing should be neutral in phase_a2!"
    assert reward_cfg['win_bonus_multiplier'] == 2.0, "Win bonus in phase_a2!"
    assert reward_cfg['loss_penalty_multiplier'] == 1.0, "Loss penalty in phase_a2!"
    assert reward_cfg['sharpe_gate_enabled'] == False, "Sharpe gate disabled in phase_a2!"
    
    # Check new augmentations
    assert 'progressive_roi_enabled' in reward_cfg, "Progressive ROI config missing!"
    assert 'entropy_bonus_weight' in reward_cfg, "Entropy bonus config missing!"
    assert 'momentum_weight' in reward_cfg, "Momentum config missing!"
    
    # Check SAC parameters - Updated to match current phase_a2 config
    sac_cfg = config_dict['sac']
    assert sac_cfg['base_learning_rate'] == 0.00025, "LR should be 0.00025 in phase_a2!"
    assert sac_cfg['batch_size'] == 256, "Batch size should be 256 in phase_a2!"
    assert sac_cfg['tau'] == 0.005, "Tau should be 0.005 in phase_a2!"
    assert sac_cfg['ent_coef'] == 0.05, "Ent coef should be fixed 0.05 in phase_a2!"
    assert sac_cfg['ent_coef_lower_bound'] == 0.05, "Ent lower bound should be 0.05!"
    assert sac_cfg['target_entropy'] == -1.0, "Target entropy should be -1.0 in phase_a2!"
    
    print("✅ All config values correct!")
    print(f"   Position sizing: {reward_cfg['position_size_small_multiplier']:.1f} / "
          f"{reward_cfg['position_size_medium_multiplier']:.1f} / "
          f"{reward_cfg['position_size_large_multiplier']:.1f}")
    print(f"   Win/Loss multipliers: {reward_cfg['win_bonus_multiplier']:.1f} / "
          f"{reward_cfg['loss_penalty_multiplier']:.1f}")
    print(f"   Sharpe gate: {reward_cfg['sharpe_gate_enabled']}")
    print(f"   Progressive ROI: {reward_cfg['progressive_roi_enabled']}")
    print(f"   Entropy bonus weight: {reward_cfg['entropy_bonus_weight']}")
    print(f"   Momentum weight: {reward_cfg['momentum_weight']}")
    print()


def test_reward_config_creation():
    """Test RewardConfig with new parameters."""
    print("=" * 80)
    print("TEST 2: RewardConfig Creation")
    print("=" * 80)
    
    config = RewardConfig(
        progressive_roi_enabled=True,
        progressive_roi_thresholds=(0.05, 0.02, 0.0, -0.01, -0.02),
        progressive_roi_multipliers=(3.0, 2.0, 1.5, 0.5, 1.0, 2.0),
        entropy_bonus_weight=0.1,
        entropy_bonus_target=2.0,
        entropy_bonus_scale=0.5,
        momentum_weight=0.05,
        momentum_window=10,
        momentum_max_reward=0.3,
    )
    
    print("✅ RewardConfig created successfully!")
    print(f"   Progressive ROI enabled: {config.progressive_roi_enabled}")
    print(f"   Entropy bonus weight: {config.entropy_bonus_weight}")
    print(f"   Momentum weight: {config.momentum_weight}")
    print()


def test_progressive_roi():
    """Test progressive ROI reward calculation."""
    print("=" * 80)
    print("TEST 3: Progressive ROI Scaling")
    print("=" * 80)
    
    config = RewardConfig(
        progressive_roi_enabled=True,
        progressive_roi_thresholds=(0.05, 0.02, 0.0, -0.01, -0.02),
        progressive_roi_multipliers=(3.0, 2.0, 1.5, 0.5, 1.0, 2.0),
    )
    shaper = RewardShaper(config)
    
    test_cases = [
        (0.10, 3.0, "> 5% profit"),
        (0.03, 2.0, "2-5% profit"),
        (0.01, 1.5, "0-2% profit"),
        (-0.005, 0.5, "0 to -1% loss"),
        (-0.015, 1.0, "-1% to -2% loss"),
        (-0.03, 2.0, "< -2% loss"),
    ]
    
    for roi_pct, expected_mult, desc in test_cases:
        result = shaper._progressive_roi_reward(roi_pct)
        expected = roi_pct * expected_mult
        assert abs(result - expected) < 1e-6, f"Progressive ROI failed for {desc}"
        print(f"   {desc:20s}: ROI={roi_pct:6.2%} → reward={result:7.4f} (mult={expected_mult:.1f}x)")
    
    print("✅ Progressive ROI scaling works correctly!")
    print()


def test_entropy_bonus():
    """Test entropy bonus calculation."""
    print("=" * 80)
    print("TEST 4: Entropy Bonus")
    print("=" * 80)
    
    config = RewardConfig(
        entropy_bonus_weight=0.1,
        entropy_bonus_target=2.0,
        entropy_bonus_scale=0.5,
    )
    shaper = RewardShaper(config)
    
    # Test high diversity (should get full bonus)
    diversity_info_high = {
        'action_diversity_window': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Uniform
    }
    bonus_high = shaper._compute_entropy_bonus(diversity_info_high)
    print(f"   High diversity (uniform): {bonus_high:.4f} (should be ~{config.entropy_bonus_scale:.2f})")
    
    # Test low diversity (should get reduced bonus)
    diversity_info_low = {
        'action_diversity_window': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # Collapsed
    }
    bonus_low = shaper._compute_entropy_bonus(diversity_info_low)
    print(f"   Low diversity (collapsed): {bonus_low:.4f} (should be ~0.0)")
    
    # Test medium diversity
    diversity_info_med = {
        'action_diversity_window': [5, 5, 5, 6, 6, 7, 7, 8, 9, 10]  # Moderate
    }
    bonus_med = shaper._compute_entropy_bonus(diversity_info_med)
    print(f"   Medium diversity: {bonus_med:.4f}")
    
    assert bonus_high > bonus_med > bonus_low, "Entropy bonus not scaling correctly!"
    print("✅ Entropy bonus rewards diversity!")
    print()


def test_curriculum_schedule():
    """Test curriculum learning schedule."""
    print("=" * 80)
    print("TEST 5: Curriculum Schedule")
    print("=" * 80)
    
    curriculum = CurriculumSchedule()
    
    # Test phase 0
    updates = curriculum.get_config_updates(0, current_sharpe=-0.5)
    assert updates is not None, "Phase 0 should trigger!"
    print(f"   Step 0: {updates}")
    
    # Test phase 1
    updates = curriculum.get_config_updates(100_000, current_sharpe=-0.2)
    assert updates is not None, "Phase 1 should trigger!"
    print(f"   Step 100K: {updates}")
    
    # Test phase 2
    updates = curriculum.get_config_updates(200_000, current_sharpe=-0.1)
    assert updates is not None, "Phase 2 should trigger!"
    print(f"   Step 200K: {updates}")
    
    # Test phase 3 WITHOUT positive Sharpe (should block)
    updates = curriculum.get_config_updates(300_000, current_sharpe=-0.05)
    assert updates is None, "Phase 3 should be blocked by negative Sharpe!"
    print(f"   Step 300K (Sharpe=-0.05): BLOCKED ✅")
    
    # Test phase 3 WITH positive Sharpe (should pass)
    curriculum_2 = CurriculumSchedule()
    curriculum_2.get_config_updates(0)
    curriculum_2.get_config_updates(100_000)
    curriculum_2.get_config_updates(200_000)
    updates = curriculum_2.get_config_updates(300_000, current_sharpe=0.15)
    assert updates is not None, "Phase 3 should trigger with positive Sharpe!"
    print(f"   Step 300K (Sharpe=+0.15): {updates}")
    
    print("✅ Curriculum schedule works correctly!")
    print()


def test_critic_stabilizer():
    """Test critic loss monitoring."""
    print("=" * 80)
    print("TEST 6: Critic Stabilizer")
    print("=" * 80)
    
    stabilizer = CriticStabilizer(baseline_loss=500.0, explosion_threshold=3.0)
    
    # Test normal losses (no intervention)
    for i, loss in enumerate([450, 480, 520, 490, 510]):
        result = stabilizer.check_and_recover(loss, current_step=i*1000)
        assert result is None, f"Should not intervene on normal loss {loss}"
    print("   Normal losses (400-550): No intervention ✅")
    
    # Test explosion (should intervene)
    explosion_stabilizer = CriticStabilizer(baseline_loss=500.0, explosion_threshold=3.0, window_size=5)
    for i, loss in enumerate([1200, 1400, 1600, 1800, 2000]):
        if i < 4:
            explosion_stabilizer.check_and_recover(loss, current_step=(6+i)*1000)
    result = explosion_stabilizer.check_and_recover(2000, current_step=10_000)
    assert result is not None, "Should intervene on explosion!"
    assert result['reduce_lr'] == 0.5, "Should reduce LR"
    assert result['increase_tau'] == 2.0, "Should increase tau"
    print(f"   Critic explosion detected: {result}")
    
    print("✅ Critic stabilizer detects explosions!")
    print()


def test_action_collapse_detector():
    """Test action distribution monitoring."""
    print("=" * 80)
    print("TEST 7: Action Collapse Detector")
    print("=" * 80)
    
    detector = ActionCollapseDetector(collapse_threshold=0.5, warning_threshold=0.4)
    
    # Test diverse actions (no collapse)
    for _ in range(100):
        action = np.random.randint(0, 10)
        detector.update(action)
    
    result = detector.detect_collapse(current_step=1000)
    assert result is None, "Should not detect collapse with diverse actions"
    stats = detector.get_distribution()
    print(f"   Diverse distribution: entropy={stats['entropy']:.2f}, "
          f"max_conc={stats['max_concentration']:.2%} ✅")
    
    # Test collapsed actions (should detect)
    detector2 = ActionCollapseDetector(collapse_threshold=0.5)
    for _ in range(200):
        action = 5 if np.random.rand() < 0.7 else np.random.randint(0, 10)
        detector2.update(action)
    
    result = detector2.detect_collapse(current_step=5000)
    stats2 = detector2.get_distribution()
    print(f"   Collapsed distribution: entropy={stats2['entropy']:.2f}, "
          f"max_conc={stats2['max_concentration']:.2%}, "
          f"dominant={stats2['dominant_action']}")
    
    if stats2['max_concentration'] > 0.5:
        assert result is not None, "Should detect collapse!"
        print(f"   Intervention triggered: {result['intervention_type']} ✅")
    else:
        print("   Not concentrated enough for intervention (acceptable)")
    
    print("✅ Action collapse detector works!")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("PHASE 2 RECOVERY IMPLEMENTATION VALIDATION")
    print("=" * 80)
    print()
    
    try:
        test_config_loading()
        test_reward_config_creation()
        test_progressive_roi()
        test_entropy_bonus()
        test_curriculum_schedule()
        test_critic_stabilizer()
        test_action_collapse_detector()
        
        print("=" * 80)
        print("🎉 ALL TESTS PASSED! Ready for training.")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Run training with: python training/train_sac_continuous.py \\")
        print("      --config training/config_templates/phase_a2_sac_sharpe.yaml \\")
        print("      --symbols SPY --total-timesteps 400000")
        print()
        print("2. Monitor metrics in TensorBoard:")
        print("   - eval/sharpe_ratio_mean (target: +0.30)")
        print("   - train/critic_loss (watch for explosions > 1500)")
        print("   - continuous/entropy (maintain > 1.5)")
        print("   - continuous/action_bin_* (watch for >50% concentration)")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
