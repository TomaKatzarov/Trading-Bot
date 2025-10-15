"""Validation script to verify action exploration fixes.

This script tests the critical fixes applied to prevent policy collapse:
1. Actor initialization     # Check diversity
    unique_actions = len(action_counts)
    dominant_action_pct = max(action_counts.values()) / num_samples * 100
    
    console.print(f"\n[bold]Unique actions:[/bold] {unique_actions} / 7")
    console.print(f"[bold]Most common action:[/bold] {dominant_action_pct:.1f}%")
    
    # Note: SELL actions (4,5,6) are masked when no position exists (position=zeros)
    # This is EXPECTED and correct behavior. We only check BUY actions (0,1,2,3) for diversity.
    buy_and_hold_actions = [0, 1, 2, 3]  # HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE
    available_actions_used = sum(1 for a in buy_and_hold_actions if action_counts.get(a, 0) > 0)
    
    console.print(f"[bold]Available actions used:[/bold] {available_actions_used} / 4 (BUY+HOLD)")
    console.print("[dim]Note: SELL actions are masked (no position = can't sell) ✓[/dim]")
    
    # Verdict - adjusted for action masking
    if available_actions_used >= 3 and dominant_action_pct < 40 and entropy > 1.2:
        console.print("\n[bold green]✓ PASS:[/bold green] Diverse initial policy (gain=0.1 working)")
        return True
    elif available_actions_used <= 1 or dominant_action_pct > 90:
        console.print("\n[bold red]✗ FAIL:[/bold red] Policy already collapsed (gain still 0.01?)")
        return False
    else:
        console.print("\n[bold green]✓ PASS:[/bold green] Acceptable diversity with action masking")
        return Trueom 0.01 to 0.1
2. Entropy coefficient increased from 0.05 to 0.15
3. Reward normalization disabled

Run this before starting full Phase 3 training to verify the fixes work.

Usage:
    python analysis/validate_exploration_fix.py --symbol SPY --steps 5000
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.rl.policies.symbol_agent import SymbolAgent, SymbolAgentConfig
from core.rl.policies.feature_encoder import EncoderConfig


def test_actor_initialization():
    """Test that actor is initialized with correct gain."""
    console = Console()
    
    console.print("\n[bold cyan]Test 1: Actor Initialization Gain[/bold cyan]")
    
    encoder_config = EncoderConfig(
        technical_seq_len=24,
        technical_feature_dim=23,
        d_model=128,
        nhead=4,
        num_layers=2,
        output_dim=128,
        dropout=0.1,
    )
    
    agent_config = SymbolAgentConfig(
        encoder_config=encoder_config,
        action_dim=7,
        hidden_dim=128,
        dropout=0.1,
    )
    
    agent = SymbolAgent(agent_config)
    
    # Extract final actor layer weights
    actor_layers = [m for m in agent.actor.modules() if isinstance(m, torch.nn.Linear)]
    final_layer = actor_layers[-1]
    weights = final_layer.weight.data
    
    # Calculate statistics
    mean = weights.mean().item()
    std = weights.std().item()
    abs_max = weights.abs().max().item()
    
    # Expected: gain=0.1 → std ≈ 0.1 / sqrt(input_dim)
    input_dim = weights.shape[1]
    expected_std = 0.1 / np.sqrt(input_dim)
    
    table = Table(title="Actor Final Layer Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Expected (gain=0.1)", style="green")
    table.add_column("Old (gain=0.01)", style="red")
    
    table.add_row("Mean", f"{mean:.6f}", "~0.0", "~0.0")
    table.add_row("Std Dev", f"{std:.6f}", f"~{expected_std:.4f}", f"~{expected_std/10:.4f}")
    table.add_row("Max |weight|", f"{abs_max:.6f}", f"~{expected_std*3:.4f}", f"~{expected_std/10*3:.4f}")
    
    console.print(table)
    
    # Verify
    old_expected_std = expected_std / 10
    if abs(std - expected_std) < abs(std - old_expected_std):
        console.print("[bold green]✓ PASS:[/bold green] Actor initialized with gain=0.1 (NEW)")
        return True
    else:
        console.print("[bold red]✗ FAIL:[/bold red] Actor still using gain=0.01 (OLD)")
        return False


def test_initial_action_distribution(num_samples=1000):
    """Test that initial policy produces diverse action distribution."""
    console = Console()
    
    console.print("\n[bold cyan]Test 2: Initial Action Distribution[/bold cyan]")
    
    encoder_config = EncoderConfig(
        technical_seq_len=24,
        technical_feature_dim=23,
        d_model=128,
        nhead=4,
        num_layers=2,
        output_dim=128,
        dropout=0.1,
    )
    
    agent_config = SymbolAgentConfig(
        encoder_config=encoder_config,
        action_dim=7,
        hidden_dim=128,
        dropout=0.1,
    )
    
    agent = SymbolAgent(agent_config)
    agent.eval()
    
    # Create dummy observations
    batch_size = num_samples
    observations = {
        "technical": torch.randn(batch_size, 24, 23),
        "sl_probs": torch.ones(batch_size, 3) / 3,
        "position": torch.zeros(batch_size, 5),
        "portfolio": torch.tensor([[100000.0, 100000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * batch_size),
        "regime": torch.ones(batch_size, 10) * 0.5,
    }
    
    # Sample actions
    with torch.no_grad():
        actions, _, _ = agent(observations, deterministic=False)
    
    # Count action distribution
    action_counts = Counter(actions.cpu().numpy().tolist())
    
    table = Table(title=f"Action Distribution ({num_samples} samples)")
    table.add_column("Action", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="yellow")
    table.add_column("Expected (uniform)", style="green")
    
    action_names = ["HOLD", "BUY_SMALL", "BUY_MEDIUM", "BUY_LARGE", "SELL_PARTIAL", "SELL_ALL", "ADD_POSITION"]
    
    for action_idx, name in enumerate(action_names):
        count = action_counts.get(action_idx, 0)
        pct = count / num_samples * 100
        expected = 100 / 7
        table.add_row(name, str(count), f"{pct:.1f}%", f"{expected:.1f}%")
    
    console.print(table)
    
    # Calculate entropy
    probs = np.array([action_counts.get(i, 0) / num_samples for i in range(7)])
    probs = probs[probs > 0]  # Filter zeros for log
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(7)  # For uniform distribution
    
    console.print(f"\n[bold]Entropy:[/bold] {entropy:.3f} / {max_entropy:.3f} ({entropy/max_entropy*100:.1f}% of maximum)")
    
    # Check diversity
    unique_actions = len(action_counts)
    dominant_action_pct = max(action_counts.values()) / num_samples * 100
    
    console.print(f"[bold]Unique actions:[/bold] {unique_actions} / 7")
    console.print(f"[bold]Most common action:[/bold] {dominant_action_pct:.1f}%")
    
    # Verdict
    if unique_actions >= 5 and dominant_action_pct < 25 and entropy > 1.5:
        console.print("\n[bold green]✓ PASS:[/bold green] Diverse initial policy (gain=0.1 working)")
        return True
    elif unique_actions <= 2 or dominant_action_pct > 80:
        console.print("\n[bold red]✗ FAIL:[/bold red] Policy already collapsed (gain still 0.01?)")
        return False
    else:
        console.print("\n[bold yellow]⚠ PARTIAL:[/bold yellow] Borderline diversity (monitor closely)")
        return True


def test_entropy_coefficient_schedule():
    """Test that entropy coefficient schedule is configured correctly."""
    console = Console()
    
    console.print("\n[bold cyan]Test 3: Entropy Coefficient Schedule[/bold cyan]")
    
    # Simulate the schedule from config
    initial = 0.15
    decay = 0.9995
    minimum = 0.01
    total_timesteps = 100_000
    n_steps = 2048
    n_envs = 16
    
    updates = total_timesteps // (n_steps * n_envs)
    
    table = Table(title="Entropy Coefficient Over Training")
    table.add_column("Timestep", style="cyan")
    table.add_column("Update", style="blue")
    table.add_column("Progress", style="yellow")
    table.add_column("Entropy Coef (NEW)", style="green")
    table.add_column("Entropy Coef (OLD)", style="red")
    
    checkpoints = [0, 10_000, 30_000, 50_000, 70_000, 100_000]
    
    for timestep in checkpoints:
        update = timestep // (n_steps * n_envs)
        progress = timestep / total_timesteps * 100
        
        # New schedule
        new_ent = max(minimum, initial * (decay ** update))
        
        # Old schedule
        old_initial = 0.05
        old_decay = 0.997
        old_minimum = 0.005
        old_ent = max(old_minimum, old_initial * (old_decay ** update))
        
        table.add_row(
            f"{timestep:,}",
            str(update),
            f"{progress:.0f}%",
            f"{new_ent:.4f}",
            f"{old_ent:.4f}"
        )
    
    console.print(table)
    
    # Verdict
    final_new = max(minimum, initial * (decay ** updates))
    final_old = max(0.005, 0.05 * (0.997 ** updates))
    
    if final_new > 0.01 and final_new > final_old * 2:
        console.print(f"\n[bold green]✓ PASS:[/bold green] Entropy stays high (final: {final_new:.4f} vs old: {final_old:.4f})")
        return True
    else:
        console.print(f"\n[bold red]✗ FAIL:[/bold red] Entropy still decaying too fast")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate exploration fixes")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    console = Console()
    console.print("\n[bold magenta]═══ Action Exploration Fix Validation ═══[/bold magenta]\n")
    
    results = []
    
    # Test 1: Actor initialization
    try:
        results.append(("Actor Initialization", test_actor_initialization()))
    except Exception as e:
        console.print(f"[bold red]✗ CRASH:[/bold red] {e}")
        results.append(("Actor Initialization", False))
    
    # Test 2: Initial action distribution
    try:
        results.append(("Initial Action Distribution", test_initial_action_distribution(1000)))
    except Exception as e:
        console.print(f"[bold red]✗ CRASH:[/bold red] {e}")
        results.append(("Initial Action Distribution", False))
    
    # Test 3: Entropy schedule
    try:
        results.append(("Entropy Schedule", test_entropy_coefficient_schedule()))
    except Exception as e:
        console.print(f"[bold red]✗ CRASH:[/bold red] {e}")
        results.append(("Entropy Schedule", False))
    
    # Summary
    console.print("\n[bold magenta]═══ Summary ═══[/bold magenta]\n")
    
    summary_table = Table(title="Validation Results")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Result", style="bold")
    
    for test_name, passed in results:
        status = "[green]✓ PASS" if passed else "[red]✗ FAIL"
        summary_table.add_row(test_name, status)
    
    console.print(summary_table)
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    if total_passed == total_tests:
        console.print(f"\n[bold green]All {total_tests} tests passed! ✓[/bold green]")
        console.print("[bold green]You can proceed with Phase 3 training.[/bold green]")
        return 0
    else:
        console.print(f"\n[bold red]{total_tests - total_passed} test(s) failed! ✗[/bold red]")
        console.print("[bold red]DO NOT proceed with training until all tests pass.[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
