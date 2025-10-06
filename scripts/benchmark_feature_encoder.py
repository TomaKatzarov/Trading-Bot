"""Feature Encoder Performance Benchmarking.

Validates encoder performance for multi-agent deployment by testing inference
speed, memory usage, and scaling characteristics across CPU and GPU devices.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.rl.policies.feature_encoder import EncoderConfig, FeatureEncoder


def create_sample_batch(batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Create a batch of synthetic observations matching environment schema."""

    obs = {
        "technical": torch.randn(batch_size, 24, 23),
        "sl_probs": torch.rand(batch_size, 3),
        "position": torch.randn(batch_size, 5),
        "portfolio": torch.randn(batch_size, 8),
        "regime": torch.randn(batch_size, 10),
    }
    return {key: value.to(device=device) for key, value in obs.items()}


def benchmark_inference_speed(
    encoder: FeatureEncoder,
    device: str = "cuda",
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Measure inference latency statistics across multiple batch sizes."""

    encoder.eval()
    batch_sizes = [1, 4, 8, 16, 32, 64]
    results: Dict[str, Dict[str, float]] = {}

    for batch_size in batch_sizes:
        observations = create_sample_batch(batch_size, device=device)

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = encoder(observations)
                if device == "cuda":
                    torch.cuda.synchronize()

        timings = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = encoder(observations)
                if device == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start_time) * 1000.0)

        timings_arr = np.asarray(timings, dtype=np.float64)
        results[f"batch_{batch_size}"] = {
            "mean_ms": float(np.mean(timings_arr)),
            "std_ms": float(np.std(timings_arr)),
            "min_ms": float(np.min(timings_arr)),
            "max_ms": float(np.max(timings_arr)),
            "p50_ms": float(np.percentile(timings_arr, 50)),
            "p95_ms": float(np.percentile(timings_arr, 95)),
            "p99_ms": float(np.percentile(timings_arr, 99)),
            "per_sample_ms": float(np.mean(timings_arr) / batch_size),
            "throughput_samples_per_sec": float(batch_size / (np.mean(timings_arr) / 1000.0)),
        }

    return results


def benchmark_memory_usage(encoder: FeatureEncoder, device: str = "cuda") -> Dict[str, float]:
    """Measure parameter and activation memory consumption."""

    param_bytes = sum(p.numel() * p.element_size() for p in encoder.parameters())
    param_mb = param_bytes / (1024 ** 2)

    peak_memory_mb = 0.0
    activation_mb = 0.0

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        observations = create_sample_batch(32, device=device)
        with torch.no_grad():
            _ = encoder(observations)
            torch.cuda.synchronize()

        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        activation_mb = max(peak_memory_mb - param_mb, 0.0)

    return {
        "parameter_mb": float(param_mb),
        "peak_memory_mb": float(peak_memory_mb),
        "activation_mb": float(activation_mb),
        "parameter_count": int(sum(p.numel() for p in encoder.parameters())),
    }


def analyze_batch_scaling(speed_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Evaluate throughput scaling efficiency across batch sizes."""

    batch_sizes = [1, 4, 8, 16, 32, 64]
    throughputs = []
    for size in batch_sizes:
        metrics = speed_results.get(f"batch_{size}")
        throughputs.append(metrics["throughput_samples_per_sec"] if metrics else 0.0)

    scaling_factors = []
    if throughputs[0] > 0:
        base_throughput = throughputs[0]
        for idx in range(1, len(batch_sizes)):
            ideal_factor = batch_sizes[idx] / batch_sizes[0]
            actual_factor = throughputs[idx] / base_throughput if throughputs[idx] else 0.0
            if ideal_factor > 0:
                scaling_factors.append((actual_factor / ideal_factor) * 100.0)

    return {
        "batch_1_throughput": float(throughputs[0]) if throughputs else 0.0,
        "batch_32_throughput": float(throughputs[4]) if len(throughputs) > 4 else 0.0,
        "avg_scaling_efficiency_pct": float(np.mean(scaling_factors)) if scaling_factors else 0.0,
        "throughput_improvement_1_to_32": float(
            (throughputs[4] / throughputs[0]) if len(throughputs) > 4 and throughputs[0] > 0 else 0.0
        ),
    }


def compare_cpu_gpu(encoder_config: EncoderConfig) -> Dict[str, Dict[str, float]]:
    """Run benchmark on CPU and GPU to measure relative speedups."""

    results: Dict[str, Dict[str, float]] = {}

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        encoder = FeatureEncoder(encoder_config).to(device)
        speed_metrics = benchmark_inference_speed(
            encoder, device=device, num_iterations=50 if device == "cuda" else 20
        )
        results[device] = {
            "batch_32_p95_ms": speed_metrics["batch_32"]["p95_ms"],
            "batch_32_throughput": speed_metrics["batch_32"]["throughput_samples_per_sec"],
        }

    if "cpu" in results and "cuda" in results:
        cpu = results["cpu"]
        gpu = results["cuda"]
        results["gpu_speedup"] = {
            "latency_improvement": cpu["batch_32_p95_ms"] / gpu["batch_32_p95_ms"],
            "throughput_improvement": gpu["batch_32_throughput"] / cpu["batch_32_throughput"],
        }

    return results


def print_header(title: str) -> None:
    """Utility to print benchmark section headers."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> Dict[str, Dict[str, float]]:
    """Execute encoder performance benchmarking workflow."""

    parser = argparse.ArgumentParser(description="Benchmark Feature Encoder")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="analysis/reports")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")

    torch.backends.cudnn.benchmark = True

    print_header("FEATURE ENCODER PERFORMANCE BENCHMARK")
    print(f"Device: {args.device}")
    print(f"PyTorch Version: {torch.__version__}")
    if args.device == "cuda":
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = EncoderConfig()
    encoder = FeatureEncoder(config).to(args.device)
    encoder.eval()

    print("\nEncoder Configuration:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Hidden dim: {config.d_model}")
    print(f"  Attention heads: {config.nhead}")
    print(f"  FFN dim: {config.dim_feedforward}")

    print_header("INFERENCE SPEED BENCHMARK")
    speed_results = benchmark_inference_speed(
        encoder,
        device=args.device,
        num_iterations=args.num_iterations,
    )

    print(
        f"\n{'Batch Size':<12} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Per Sample (ms)':<17} {'Throughput (samp/s)':<22}"
    )
    print("-" * 70)
    for key in [f"batch_{size}" for size in [1, 4, 8, 16, 32, 64]]:
        metrics = speed_results[key]
        batch = key.split("_")[1]
        print(
            f"{batch:<12} {metrics['mean_ms']:<12.3f} {metrics['p95_ms']:<12.3f} "
            f"{metrics['per_sample_ms']:<17.3f} {metrics['throughput_samples_per_sec']:<22.1f}"
        )

    batch_32_p95 = speed_results["batch_32"]["p95_ms"]
    print(
        f"\nTarget (<10ms P95 for batch_size=32): "
        f"{'✅ PASS' if batch_32_p95 < 10 else '❌ FAIL'} ({batch_32_p95:.2f} ms)"
    )

    print_header("MEMORY USAGE BENCHMARK")
    memory_results = benchmark_memory_usage(encoder, device=args.device)
    print(f"\nParameter Memory: {memory_results['parameter_mb']:.2f} MB")
    print(f"Parameter Count: {memory_results['parameter_count']:,}")
    if args.device == "cuda":
        print(f"Peak Memory (batch_size=32): {memory_results['peak_memory_mb']:.2f} MB")
        print(f"Activation Memory: {memory_results['activation_mb']:.2f} MB")
        activation_target = memory_results['activation_mb'] < 100
        print(f"\nTarget (<100MB activation): {'✅ PASS' if activation_target else '❌ FAIL'}")

    print_header("BATCH SCALING ANALYSIS")
    scaling_results = analyze_batch_scaling(speed_results)
    print(f"\nBatch 1 Throughput: {scaling_results['batch_1_throughput']:.1f} samples/sec")
    print(f"Batch 32 Throughput: {scaling_results['batch_32_throughput']:.1f} samples/sec")
    print(
        f"Throughput Improvement (1→32): {scaling_results['throughput_improvement_1_to_32']:.2f}x"
    )
    print(
        f"Average Scaling Efficiency: {scaling_results['avg_scaling_efficiency_pct']:.1f}%"
    )

    cpu_gpu_results = compare_cpu_gpu(config)
    if len(cpu_gpu_results) > 1:
        print_header("CPU VS GPU COMPARISON")
        cpu_metrics = cpu_gpu_results.get("cpu")
        gpu_metrics = cpu_gpu_results.get("cuda")
        if cpu_metrics:
            print(
                f"CPU P95 (batch 32): {cpu_metrics['batch_32_p95_ms']:.2f} ms | "
                f"Throughput: {cpu_metrics['batch_32_throughput']:.1f} samples/sec"
            )
        if gpu_metrics:
            print(
                f"GPU P95 (batch 32): {gpu_metrics['batch_32_p95_ms']:.2f} ms | "
                f"Throughput: {gpu_metrics['batch_32_throughput']:.1f} samples/sec"
            )
        speedup = cpu_gpu_results.get("gpu_speedup")
        if speedup:
            print(
                f"GPU Latency Improvement: {speedup['latency_improvement']:.2f}x | "
                f"Throughput Improvement: {speedup['throughput_improvement']:.2f}x"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "device": args.device,
        "pytorch_version": torch.__version__,
        "encoder_config": {
            "num_layers": config.num_layers,
            "d_model": config.d_model,
            "nhead": config.nhead,
            "dim_feedforward": config.dim_feedforward,
        },
        "inference_speed": speed_results,
        "memory": memory_results,
        "scaling": scaling_results,
        "cpu_gpu": cpu_gpu_results,
        "targets_met": {
            "p95_latency_10ms": batch_32_p95 < 10,
            "activation_memory_100mb": (
                memory_results["activation_mb"] < 100 if args.device == "cuda" else None
            ),
            "parameter_count_5m": memory_results["parameter_count"] < 5_000_000,
            "throughput_batch32_3000": scaling_results["batch_32_throughput"] > 3000,
        },
    }

    output_path = output_dir / "feature_encoder_benchmark.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print_header("RESULTS SAVED")
    print(f"Results saved to: {output_path}")

    print_header("BENCHMARK SUMMARY")
    all_targets = [value for value in results["targets_met"].values() if value is not None]
    summary = "✅ All Targets Met!" if all(all_targets) else "⚠️ Some targets not met"
    print(f"\n{summary}")

    return results


if __name__ == "__main__":
    main()
