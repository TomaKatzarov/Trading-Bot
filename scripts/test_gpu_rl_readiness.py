"""GPU readiness benchmark for RL workloads."""

from __future__ import annotations

import time

import torch


def test_gpu_compute() -> bool:
    """Run a suite of GPU micro-benchmarks."""

    if not torch.cuda.is_available():
        print("No GPU available")
        return False

    device = torch.device("cuda")

    # Test 1: Matrix multiplication speed
    size = 5000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU Matrix Mult (10x {size}x{size}): {gpu_time:.3f}s")

    # Test 2: Neural network forward pass
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    ).to(device)

    batch = torch.randn(256, 1024, device=device)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model(batch)
    torch.cuda.synchronize()
    nn_time = time.time() - start

    print(f"NN Forward Pass (100x batches of 256): {nn_time:.3f}s")

    # Test 3: Memory allocation
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    print(f"Peak GPU Memory Used: {max_memory_allocated:.2f} GB")
    print(f"Total GPU Memory: {total_memory:.2f} GB")

    # Reset max memory tracker for cleanliness
    torch.cuda.reset_peak_memory_stats()

    performance_ok = gpu_time < 5.0 and nn_time < 1.0
    print("GPU Ready for RL Training" if performance_ok else "GPU performance below target")
    return performance_ok


if __name__ == "__main__":
    success = test_gpu_compute()
    exit(0 if success else 1)