#!/usr/bin/env python3
"""Deep analysis of TensorBoard training metrics."""
import sys
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def analyze_tensorboard_run(logdir: Path):
    """Extract and analyze all metrics from TensorBoard logs."""
    
    print("=" * 80)
    print("  TENSORBOARD DEEP ANALYSIS")
    print("=" * 80)
    print(f"\nLog Directory: {logdir}")
    
    # Find all event files
    event_files = list(logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        print("❌ No TensorBoard event files found!")
        return
    
    print(f"\nFound {len(event_files)} event file(s)")
    
    for event_file in event_files:
        print(f"\n📊 Analyzing: {event_file.name}")
        print("-" * 80)
        
        # Load event file
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        
        # Get all available tags
        scalar_tags = ea.Tags()["scalars"]
        print(f"\n✅ Found {len(scalar_tags)} metric tags\n")
        
        # Organize metrics by category
        categories = {
            "reward": [],
            "eval": [],
            "train": [],
            "icm": [],
            "continuous": [],
            "trading": [],
            "reward_breakdown": [],
            "reward_components": [],
        }
        
        for tag in scalar_tags:
            for cat in categories.keys():
                if tag.startswith(cat):
                    categories[cat].append(tag)
                    break
        
        # Analyze each category
        for category, tags in categories.items():
            if not tags:
                continue
            
            print(f"\n{'=' * 60}")
            print(f"  {category.upper()} METRICS ({len(tags)} tags)")
            print(f"{'=' * 60}\n")
            
            for tag in sorted(tags):
                events = ea.Scalars(tag)
                if not events:
                    continue
                
                values = [e.value for e in events]
                steps = [e.step for e in events]
                
                if len(values) == 0:
                    continue
                
                # Calculate statistics
                first_val = values[0]
                last_val = values[-1]
                min_val = min(values)
                max_val = max(values)
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Calculate trend
                if len(values) > 1:
                    trend = last_val - first_val
                    trend_pct = (trend / abs(first_val) * 100) if first_val != 0 else 0
                else:
                    trend = 0
                    trend_pct = 0
                
                # Format output
                tag_short = tag.replace(f"{category}/", "")
                
                print(f"  {tag_short:35s}")
                print(f"    Steps: {len(values):5d} | Range: [{min_val:+.6f}, {max_val:+.6f}]")
                print(f"    Mean: {mean_val:+.6f} ± {std_val:.6f}")
                print(f"    First: {first_val:+.6f} → Last: {last_val:+.6f} (Δ {trend:+.6f}, {trend_pct:+.1f}%)")
                
                # Highlight critical patterns
                if "sharpe" in tag.lower() or "return" in tag.lower():
                    if last_val < 0:
                        print(f"    ⚠️  NEGATIVE PERFORMANCE!")
                    elif last_val > 0.3:
                        print(f"    ✅ GOOD PERFORMANCE!")
                
                if "pnl" in tag.lower():
                    if last_val < 0:
                        print(f"    ⚠️  LOSING MONEY!")
                    elif last_val > 0:
                        print(f"    ✅ MAKING MONEY!")
                
                if "entropy" in tag.lower():
                    if last_val > 2.0:
                        print(f"    ⚠️  TOO RANDOM (should be 1.0-2.0)")
                    elif last_val < 1.0:
                        print(f"    ⚠️  TOO DETERMINISTIC (may overfit)")
                    else:
                        print(f"    ✅ HEALTHY ENTROPY")
                
                if "transaction_cost" in tag.lower():
                    if abs(last_val) > 0.05:
                        print(f"    ⚠️  TRANSACTION COSTS TOO HIGH!")
                
                print()
        
        # CRITICAL INSIGHTS
        print("\n" + "=" * 80)
        print("  🔍 CRITICAL INSIGHTS")
        print("=" * 80 + "\n")
        
        # Get key metrics
        key_metrics = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            if events:
                values = [e.value for e in events]
                key_metrics[tag] = {
                    "first": values[0],
                    "last": values[-1],
                    "mean": np.mean(values),
                    "max": max(values),
                    "min": min(values),
                }
        
        # Analyze reward composition
        pnl = key_metrics.get("reward_components/pnl", {}).get("last", 0)
        tx_cost = key_metrics.get("reward_components/transaction_cost", {}).get("last", 0)
        sharpe = key_metrics.get("reward_components/sharpe", {}).get("last", 0)
        
        print(f"1. REWARD BALANCE:")
        print(f"   PnL Component:          {pnl:+.6f}")
        print(f"   Transaction Cost:       {tx_cost:+.6f}")
        print(f"   Sharpe Component:       {sharpe:+.6f}")
        print(f"   Net Impact:             {pnl + tx_cost + sharpe:+.6f}")
        
        if abs(tx_cost) > abs(pnl):
            ratio = abs(tx_cost) / max(abs(pnl), 1e-6)
            print(f"   ⚠️  CRITICAL: Transaction costs {ratio:.1f}x larger than PnL!")
        
        if sharpe < -0.1:
            print(f"   ⚠️  CRITICAL: Sharpe penalty {sharpe:.4f} crushing rewards!")
        
        # Analyze action distribution
        action_mean = key_metrics.get("continuous/action_mean", {}).get("last", 0)
        action_std = key_metrics.get("continuous/action_std", {}).get("last", 0)
        entropy = key_metrics.get("continuous/entropy", {}).get("last", 0)
        
        print(f"\n2. ACTION BEHAVIOR:")
        print(f"   Action Mean:            {action_mean:+.4f} (0=neutral, +1=all buy, -1=all sell)")
        print(f"   Action Std:             {action_std:.4f}")
        print(f"   Entropy:                {entropy:.4f}")
        
        if action_mean > 0.4:
            print(f"   ⚠️  BIAS TOWARD BUYING (mean > 0.4)")
        elif action_mean < -0.4:
            print(f"   ⚠️  BIAS TOWARD SELLING (mean < -0.4)")
        
        if entropy > 2.5:
            print(f"   ⚠️  TOO MUCH EXPLORATION (entropy > 2.5)")
        
        # Analyze learning progress
        actor_loss_first = key_metrics.get("train/actor_loss", {}).get("first", 0)
        actor_loss_last = key_metrics.get("train/actor_loss", {}).get("last", 0)
        critic_loss_last = key_metrics.get("train/critic_loss", {}).get("last", 0)
        
        print(f"\n3. LEARNING DYNAMICS:")
        print(f"   Actor Loss (first):     {actor_loss_first:+.4f}")
        print(f"   Actor Loss (final):     {actor_loss_last:+.4f}")
        print(f"   Critic Loss (final):    {critic_loss_last:.6f}")
        
        if abs(actor_loss_last) > 5:
            print(f"   ⚠️  Actor not converging (loss > 5)")
        
        if critic_loss_last > 0.1:
            print(f"   ⚠️  Critic error high (should be < 0.01)")
        
        # Analyze evaluation performance
        eval_sharpe = key_metrics.get("eval/sharpe_ratio_mean", {}).get("last")
        eval_return = key_metrics.get("eval/total_return_pct_mean", {}).get("last")
        
        if eval_sharpe is not None and eval_return is not None:
            print(f"\n4. FINAL EVALUATION:")
            print(f"   Sharpe Ratio:           {eval_sharpe:+.4f}")
            print(f"   Total Return:           {eval_return:+.4f}%")
            
            if eval_sharpe < 0:
                print(f"   ❌ NEGATIVE SHARPE - Agent losing money consistently")
            elif eval_sharpe < 0.3:
                print(f"   ⚠️  LOW SHARPE - Below minimum viable (0.3)")
            elif eval_sharpe >= 0.5:
                print(f"   ✅ GOOD SHARPE - Profitable strategy!")
        
        print("\n" + "=" * 80)
        print("  💡 KEY TAKEAWAYS")
        print("=" * 80 + "\n")
        
        takeaways = []
        
        if abs(tx_cost) > abs(pnl):
            takeaways.append("1. Transaction costs dominating PnL - reduce cost penalties 5-10x")
        
        if sharpe < -0.1:
            takeaways.append("2. Sharpe penalty active despite config - disable roi_multiplier_enabled")
        
        if action_mean > 0.4:
            takeaways.append("3. Agent biased toward buying/holding - check reward asymmetry")
        
        if entropy > 2.5:
            takeaways.append("4. Too much random exploration - reduce ent_coef to 0.05")
        
        if pnl < 0:
            takeaways.append("5. PnL component negative - increase pnl_scale 5-10x")
        
        if eval_sharpe and eval_sharpe < 0:
            takeaways.append("6. Negative Sharpe in eval - agent learned to lose money")
        
        if not takeaways:
            takeaways.append("✅ Training looks healthy - minor tuning may help")
        
        for takeaway in takeaways:
            print(f"   {takeaway}")
        
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    logdir = Path("models/phase_a2_sac_sharpe/SPY/tensorboard/SAC_1")
    
    if not logdir.exists():
        print(f"❌ ERROR: Log directory not found: {logdir}")
        sys.exit(1)
    
    analyze_tensorboard_run(logdir)
