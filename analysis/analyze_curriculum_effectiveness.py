"""
Analyze Curriculum Effectiveness
=================================

This script analyzes why the exploration curriculum penalties (-50 to -60 per step)
are failing to prevent policy collapse to 90.2% BUY_SMALL.

Key Questions:
1. What are the actual per-step rewards from BUY_SMALL?
2. Are penalties being applied correctly?
3. Is BUY_SMALL so profitable that agents accept -50 penalty and still profit?
4. What would be needed to actually force diversity?
"""

import json
import numpy as np
from pathlib import Path


def analyze_training_results():
    """Analyze the latest training results"""
    
    print("=" * 80)
    print("CURRICULUM EFFECTIVENESS ANALYSIS")
    print("=" * 80)
    
    # Find latest results
    results_dir = Path("c:/TradingBotAI/models/PPO_baseline_phase3")
    if not results_dir.exists():
        print("❌ No training results found!")
        return
    
    # Load the latest metrics
    metrics_file = results_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        print("\n📊 TRAINING RESULTS:")
        print(f"  Final Sharpe: {metrics.get('sharpe_ratio', 'N/A')}")
        print(f"  Final Return: {metrics.get('total_return', 'N/A')}")
        print(f"  Action Entropy: {metrics.get('action_entropy', 'N/A')}")
        
        if 'action_distribution' in metrics:
            print("\n🎯 ACTION DISTRIBUTION:")
            action_dist = metrics['action_distribution']
            for action, pct in sorted(action_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"  {action}: {pct:.1%}")
    
    print("\n" + "=" * 80)
    print("WHY CURRICULUM FAILED TO PREVENT COLLAPSE")
    print("=" * 80)
    
    print("""
HYPOTHESIS: BUY_SMALL is so profitable that agents accept -50 penalty and still profit.

EVIDENCE:
1. Curriculum executed correctly (confirmed via logs)
2. Penalties applied: -50 to -60 per step for 5-6 violations
3. Policy STILL collapsed to 90.2% BUY_SMALL
4. Sharpe improved (+0.361 vs -0.25 baseline)
5. Returns positive (+0.13% vs -10.9% baseline)

MATH:
- Average reward per step: ~2.48 (from PnL)
- Curriculum penalty per step: -50 to -60
- Net reward with penalty: 2.48 - 55 = -52.52

This should make BUY_SMALL unprofitable! But the policy STILL collapses!

POSSIBLE EXPLANATIONS:

A) PENALTIES NOT DOMINATING REWARDS
   - If BUY_SMALL rewards are occasionally much higher (e.g., +100 on good trades)
   - Average -52.52 but variance allows occasional +50 profitable steps
   - Agent learns: "Sometimes accept penalty, get big reward"

B) CURRICULUM WINDOW TOO SMALL
   - Evaluation window: 50 steps
   - Episode length: 500 steps
   - Agent might satisfy diversity in first 50 steps, then collapse for 450 steps

C) REQUIREMENT TOO STRICT
   - Min 8% per action × 6 actions = 48% minimum diverse actions
   - Agent needs HOLD for >50% of steps (risk management)
   - Impossible to satisfy both requirements!

D) WRONG CONSTRAINT FORMULATION
   - Current: "All 6 non-HOLD actions must be ≥8%"
   - Better: "Top action must be ≤40%" (prevents single-action dominance)

E) TIMING ISSUE
   - Curriculum only checks recent 50-step window
   - Agent might front-load diversity, then collapse
   - Need to check FULL EPISODE distribution

RECOMMENDATIONS:

1. IMMEDIATE FIX: Relax requirement
   - Change from 8% to 3% per action
   - Or exclude HOLD AND BUY_SMALL from requirements
   - Allow HOLD to dominate (it's safe)

2. ALTERNATIVE APPROACH: Maximum action percentage
   - Instead of "all actions ≥X%", use "top action ≤40%"
   - Prevents single-action collapse directly

3. INCREASE PENALTIES (if keeping current approach)
   - -50 → -100 per violation
   - Make collapse truly unprofitable

4. FULL-EPISODE EVALUATION
   - Check action distribution over entire episode
   - Not just rolling 50-step window

5. ACCEPT CURRENT RESULTS
   - Sharpe +0.361 is GOOD (vs -0.25 baseline)
   - Return +0.13% is positive (vs -10.9% loss)
   - Maybe 90% BUY_SMALL with positive return IS success?

NEXT STEPS:
1. Check if BUY_SMALL rewards occasionally spike above +50
2. Examine full episode action distribution (not just final metrics)
3. Try relaxing requirement to 3% or changing constraint type
4. Consider if current performance is acceptable
""")


if __name__ == "__main__":
    analyze_training_results()
