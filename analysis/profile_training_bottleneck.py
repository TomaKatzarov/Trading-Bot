"""Profile training to identify the actual bottleneck causing 3-minute pauses.

This script will help us understand WHERE the time is being spent.
"""
import time
import cProfile
import pstats
from pathlib import Path
import sys

# Simplified test to isolate the bottleneck
def profile_initialization():
    """Profile just the initialization phase"""
    print("="*80)
    print("PROFILING INITIALIZATION BOTTLENECK")
    print("="*80)
    
    start = time.time()
    
    # Hypothesis 1: torch.compile is slow on first run
    print("\n1. Testing torch.compile overhead...")
    import torch
    
    model = torch.nn.Sequential(
        torch.nn.Linear(578, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 6)
    ).cuda()
    
    compile_start = time.time()
    compiled = torch.compile(model)
    compile_time = time.time() - compile_start
    print(f"   torch.compile: {compile_time:.2f}s")
    
    # First forward pass (triggers compilation)
    x = torch.randn(4, 578).cuda()
    first_run_start = time.time()
    _ = compiled(x)
    torch.cuda.synchronize()
    first_run_time = time.time() - first_run_start
    print(f"   First forward pass (actual compilation): {first_run_time:.2f}s")
    
    # Second pass (should be fast)
    second_run_start = time.time()
    _ = compiled(x)
    torch.cuda.synchronize()
    second_run_time = time.time() - second_run_start
    print(f"   Second forward pass: {second_run_time:.2f}s")
    
    print(f"\nTotal test time: {time.time() - start:.2f}s")
    
    if first_run_time > 60:
        print("\n⚠️  BOTTLENECK FOUND: torch.compile first run takes >60s!")
        print("   This explains the 3-minute pause!")
        return "torch.compile"
    
    return None

def profile_replay_buffer():
    """Profile replay buffer operations"""
    print("\n" + "="*80)
    print("PROFILING REPLAY BUFFER")
    print("="*80)
    
    import numpy as np
    import torch
    from collections import namedtuple
    
    Transition = namedtuple('Transition', ['state', 'option_idx', 'option_return', 'option_steps', 'episode_return'])
    
    # Test pre-allocated buffer performance
    capacity = 10000
    state_dim = 578
    
    init_start = time.time()
    states = np.zeros((capacity, state_dim), dtype=np.float32)
    option_indices = np.zeros(capacity, dtype=np.int64)
    option_returns = np.zeros(capacity, dtype=np.float32)
    option_steps = np.zeros(capacity, dtype=np.float32)
    init_time = time.time() - init_start
    
    print(f"   Buffer initialization: {init_time:.4f}s")
    
    # Test sampling
    sample_start = time.time()
    for _ in range(100):
        indices = np.random.randint(0, 1000, size=128)
        batch_states = torch.from_numpy(states[indices]).cuda(non_blocking=True)
    sample_time = time.time() - sample_start
    
    print(f"   100 batched samples: {sample_time:.4f}s")
    
    return None

if __name__ == "__main__":
    print("Starting bottleneck profiling...")
    print(f"This will help identify the 3-minute pause\n")
    
    bottleneck = profile_initialization()
    profile_replay_buffer()
    
    if bottleneck == "torch.compile":
        print("\n" + "="*80)
        print("ROOT CAUSE IDENTIFIED!")
        print("="*80)
        print("\ntorch.compile has massive first-run overhead (60-180 seconds)")
        print("This happens TWICE:")
        print("  1. At start when compiling options controller")
        print("  2. At end when compiling for final evaluation/save")
        print("\nSOLUTION OPTIONS:")
        print("  A. Disable torch.compile (fastest for short runs)")
        print("  B. Use torch.compile(mode='reduce-overhead') - faster compilation")
        print("  C. Pre-compile with dummy forward pass")
        print("  D. Only compile for production runs >10k steps")
