"""Benchmark supervised-learning checkpoint inference performance.

Executes multiple timed inference passes for each checkpoint in
``models/sl_checkpoints`` using the pregenerated dataset. The script reports
latency per sample and throughput, enabling regression tracking prior to RL
integration.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from sl_checkpoint_utils import (  # noqa: E402
    align_config_with_state_dict,
    compute_logit_statistics,
    ensure_directory,
    infer_dataset_info,
    instantiate_model_from_checkpoint,
    list_checkpoint_artifacts,
    load_asset_mapping,
    load_checkpoint_bundle,
    load_dataset_arrays,
    run_batched_inference,
    sanitize_model_config,
    summarize_model,
)

LOGGER = logging.getLogger("sl_checkpoint_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/sl_checkpoints"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/training_data_v2_final"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/sl_inference_benchmarks.json"))
    parser.add_argument("--log-file", type=Path, default=Path("logs/sl_inference_benchmarks.log"))
    parser.add_argument("--sample-limit", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def configure_logging(log_path: Path) -> None:
    ensure_directory(log_path.parent)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)


def select_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_asset_ids(asset_ids: np.ndarray, num_assets: int) -> None:
    if asset_ids.size == 0:
        return
    max_id = int(asset_ids.max())
    if max_id >= num_assets:
        raise ValueError(
            f"Asset ID {max_id} exceeds embedding size ({num_assets}). Update metadata or dataset."
        )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_file)

    LOGGER.info("Starting SL inference benchmark")
    LOGGER.info("Checkpoint directory: %s", args.checkpoint_dir)
    LOGGER.info("Dataset directory: %s", args.dataset_dir)

    artifacts = list_checkpoint_artifacts(args.checkpoint_dir)
    if not artifacts:
        LOGGER.error("No checkpoints discovered under %s", args.checkpoint_dir)
        return 1

    mapping = load_asset_mapping(args.dataset_dir / "asset_id_mapping.json")
    features, asset_ids = load_dataset_arrays(
        args.dataset_dir,
        split=args.split,
        sample_limit=args.sample_limit,
    )
    dataset_info = infer_dataset_info(
        features,
        asset_ids,
        mapping,
        metadata_path=args.dataset_dir / "metadata.json",
    )

    device = select_device(args.device)
    LOGGER.info("Using device: %s", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ensure_directory(args.output_json.parent)

    summary: Dict[str, Any] = {
        "dataset": {
            "split": args.split,
            "num_samples": int(features.shape[0]),
            "lookback_window": dataset_info.lookback_window,
            "n_features": dataset_info.n_features,
        },
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "device": device.type,
        "results": [],
    }

    overall_success = True

    for artifact in artifacts:
        checkpoint_name = artifact.checkpoint_path.parent.name
        LOGGER.info("\n=== Benchmarking %s ===", checkpoint_name)
        record: Dict[str, Any] = {
            "checkpoint": checkpoint_name,
            "status": "ok",
        }
        try:
            bundle = load_checkpoint_bundle(artifact)
            model_type = bundle["model_type"]
            raw_config = bundle["raw_config"]
            state_dict = bundle["state_dict"]

            sanitized_config = sanitize_model_config(model_type, raw_config, dataset_info)
            sanitized_config = align_config_with_state_dict(model_type, sanitized_config, state_dict)
            validate_asset_ids(asset_ids, int(sanitized_config.get("num_assets", dataset_info.inferred_num_assets)))

            model = instantiate_model_from_checkpoint(model_type, sanitized_config, state_dict)

            inference_output = run_batched_inference(
                model,
                features,
                asset_ids,
                device,
                batch_size=args.batch_size,
                repeats=args.repeats,
                warmup=args.warmup,
            )

            record.update(
                {
                    "model_type": model_type,
                    "epoch": bundle["epoch"],
                    "metrics": {k: float(v) for k, v in bundle["metrics"].items()},
                    "parameter_summary": summarize_model(model),
                    "avg_latency_ms_per_sample": inference_output["avg_latency_ms_per_sample"],
                    "total_latency_s": inference_output["total_latency_s"],
                    "samples_per_second": inference_output["samples_per_second"],
                    "logit_stats": compute_logit_statistics(inference_output["logits"]),
                }
            )
            LOGGER.info(
                "%s: %.3f ms/sample (%.1f samples/sec)",
                checkpoint_name,
                record["avg_latency_ms_per_sample"],
                record["samples_per_second"],
            )
        except Exception as exc:  # pragma: no cover - defensive
            overall_success = False
            record["status"] = "error"
            record["error_message"] = str(exc)
            LOGGER.exception("Benchmark failed for %s", checkpoint_name)

        summary["results"].append(record)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Benchmark summary written to %s", args.output_json)

    return 0 if overall_success else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
