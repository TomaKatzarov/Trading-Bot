#!/usr/bin/env python3
"""Comprehensive test-set evaluation for Phase 3 HPO models.

This script loads the enhanced Phase 3 dataset, runs inference with the top HPO
trial checkpoints, and produces a full evaluation report including
classification metrics, probabilistic scores, confusion matrices, and
threshold analysis to determine the optimal production operating point.

Usage (defaults evaluate the three Phase 3 winners):
    python scripts/evaluate_hpo_models_on_test.py

You can evaluate custom checkpoints via repeated --model-checkpoint flags:
    python scripts/evaluate_hpo_models_on_test.py \
        --model-checkpoint lstm_prod=path/to/checkpoint.pt \
        --model-checkpoint gru_prod=path/to/checkpoint.pt

Outputs:
- Rich console summary comparing validation vs test metrics.
- JSON report with per-model metrics and threshold curves under
  reports/phase4/test_set_evaluation/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional at runtime
    def tqdm(iterable: Iterable, **_: object) -> Iterable:  # type: ignore
        return iterable

# Project-local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.nn_architectures import create_model

DEFAULT_DATASET = PROJECT_ROOT / "data" / "training_data_v2_final"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase4" / "test_set_evaluation"
DEFAULT_MODEL_PATHS = {
    "mlp_trial72": PROJECT_ROOT / "models" / "hpo_derived" / "HPO_GRU_Production_mlp" / "best_model.pt",
    "lstm_trial62": PROJECT_ROOT / "models" / "hpo_derived" / "HPO_GRU_Production_lstm" / "best_model.pt",
    "gru_trial93": PROJECT_ROOT / "models" / "hpo_derived" / "HPO_GRU_Production_gru" / "best_model.pt",
}

MetricsDict = Dict[str, float]


def _resolve_path(path_str: str, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _load_asset_mapping(dataset_dir: Path) -> Optional[Mapping[str, int]]:
    mapping_path = dataset_dir / "asset_id_mapping.json"
    if not mapping_path.exists():
        return None
    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("symbol_to_id", {})


def load_dataset(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    X = np.load(dataset_dir / "test_X.npy")
    y = np.load(dataset_dir / "test_y.npy")
    asset_ids = np.load(dataset_dir / "test_asset_ids.npy")

    metadata_path = dataset_dir / "metadata.json"
    metadata: Dict[str, float] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    return X, y, asset_ids, metadata


@dataclass
class DatasetInfo:
    n_features: int
    lookback_window: int
    num_assets_dataset: int
    num_assets_mapping: Optional[int]


def infer_dataset_info(X: np.ndarray, asset_ids: np.ndarray, asset_mapping: Optional[Mapping[str, int]]) -> DatasetInfo:
    n_features = int(X.shape[2])
    lookback = int(X.shape[1])
    num_assets_dataset = int(asset_ids.max()) + 1 if asset_ids.size else 1
    num_assets_mapping = len(asset_mapping) if asset_mapping else None
    return DatasetInfo(n_features, lookback, num_assets_dataset, num_assets_mapping)


@dataclass
class CheckpointBundle:
    path: Path
    model_type: str
    raw_model_config: Dict[str, float]
    validation_metrics: MetricsDict
    training_config: Dict[str, float]
    trial_id: Optional[str]
    epoch: int


def load_checkpoint(path: Path) -> CheckpointBundle:
    if path.is_dir():
        path = path / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_type = str(config.get("model_type", "unknown")).lower()
    model_config = dict(config.get("model_config", {}))
    val_metrics = {k: float(v) for k, v in checkpoint.get("metrics", {}).items()}
    trial_id = None
    tags = config.get("tags") or {}
    if isinstance(tags, dict):
        trial_id = tags.get("hpo_trial")
    epoch = int(checkpoint.get("epoch", -1))

    return CheckpointBundle(
        path=path,
        model_type=model_type,
        raw_model_config=model_config,
        validation_metrics=val_metrics,
        training_config=dict(config.get("training_config", {})),
        trial_id=trial_id,
        epoch=epoch,
    )


def sanitize_model_config(model_type: str, raw_config: Mapping[str, float], dataset_info: DatasetInfo) -> Dict[str, float]:
    cfg: Dict[str, float] = {}
    cfg["n_features"] = dataset_info.n_features
    cfg["lookback_window"] = dataset_info.lookback_window

    inferred_num_assets = dataset_info.num_assets_dataset
    if dataset_info.num_assets_mapping is not None:
        inferred_num_assets = max(inferred_num_assets, dataset_info.num_assets_mapping)
    inferred_num_assets = max(inferred_num_assets, int(raw_config.get("num_assets", inferred_num_assets) or inferred_num_assets))
    cfg["num_assets"] = inferred_num_assets

    cfg["asset_embedding_dim"] = int(raw_config.get("asset_embedding_dim", 8) or 8)
    dropout_val = float(raw_config.get("dropout_rate", 0.3) or 0.3)
    cfg["dropout_rate"] = float(min(max(dropout_val, 0.0), 0.8))

    attention_key = "attention_dim" if "attention_dim" in raw_config else "attention_dim极"

    if model_type == "mlp":
        hidden_layers = int(raw_config.get("hidden_layers", 3) or 3)
        hidden_dims: List[int] = []
        for idx in range(1, hidden_layers + 1):
            key = f"hidden_dim_{idx}"
            if key in raw_config:
                hidden_dims.append(int(raw_config[key]))
        if not hidden_dims and "hidden_dims" in raw_config:
            raw_dims = raw_config["hidden_dims"]
            if isinstance(raw_dims, (list, tuple)):
                hidden_dims = [int(v) for v in raw_dims if v]
        if not hidden_dims:
            hidden_dims = [128, 64, 32]
        cfg["hidden_dims"] = tuple(hidden_dims)

    elif model_type == "lstm":
        cfg["lstm_hidden_dim"] = int(raw_config.get("lstm_hidden_dim", raw_config.get("hidden_dim_1", 64)) or 64)
        cfg["lstm_num_layers"] = int(raw_config.get("lstm_num_layers", 2) or 2)
        cfg["attention_dim"] = int(raw_config.get(attention_key, 64) or 64)
        cfg["use_layer_norm"] = bool(raw_config.get("use_layer_norm", True))

    elif model_type == "gru":
        cfg["gru_hidden_dim"] = int(raw_config.get("gru_hidden_dim", raw_config.get("hidden_dim_1", 64)) or 64)
        cfg["gru_num_layers"] = int(raw_config.get("gru_num_layers", 2) or 2)
        cfg["attention_dim"] = int(raw_config.get(attention_key, 64) or 64)
        cfg["use_layer_norm"] = bool(raw_config.get("use_layer_norm", True))

    elif model_type == "cnn_lstm":
        # Not used in default run but kept for completeness.
        cfg["lstm_hidden_dim"] = int(raw_config.get("lstm_hidden_dim", 128) or 128)
        cfg["lstm_num_layers"] = int(raw_config.get("lstm_num_layers", 2) or 2)
        cfg["attention_dim"] = int(raw_config.get(attention_key, 64) or 64)
        cfg["use_layer_norm"] = bool(raw_config.get("use_layer_norm", True))
        cfg["cnn_filters"] = tuple(raw_config.get("cnn_filters", (32, 64)))
        cfg["cnn_kernel_sizes"] = tuple(raw_config.get("cnn_kernel_sizes", (3, 3)))
        cfg["cnn_stride"] = int(raw_config.get("cnn_stride", 1) or 1)
        cfg["use_max_pooling"] = bool(raw_config.get("use_max_pooling", True))
    else:
        raise ValueError(f"Unsupported model type '{model_type}' in checkpoint; cannot evaluate.")

    return cfg


def clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value
    return cleaned


def align_config_with_state_dict(
    model_type: str,
    config: Mapping[str, float],
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    cfg = dict(config)

    asset_weight = state_dict.get("asset_embedding.weight")
    if asset_weight is not None:
        cfg["num_assets"] = int(asset_weight.shape[0])
        cfg["asset_embedding_dim"] = int(asset_weight.shape[1])

    if model_type == "mlp":
        linear_weights = []
        for key, value in state_dict.items():
            if key.startswith("mlp.") and key.endswith(".weight"):
                try:
                    layer_index = int(key.split(".")[1])
                except (IndexError, ValueError):
                    continue
                linear_weights.append((layer_index, int(value.shape[0])))
        if linear_weights:
            linear_weights.sort(key=lambda item: item[0])
            hidden_dims = [dim for _, dim in linear_weights[:-1]]  # Exclude final output layer
            if hidden_dims:
                cfg["hidden_dims"] = tuple(hidden_dims)

    elif model_type == "lstm":
        hidden_dim = None
        for key, value in state_dict.items():
            if key.startswith("lstm.weight_hh_l0"):
                hidden_dim = int(value.shape[1])
                break
        if hidden_dim is not None:
            cfg["lstm_hidden_dim"] = hidden_dim
        num_layers = sum(1 for key in state_dict if key.startswith("lstm.weight_hh_l"))
        if num_layers:
            cfg["lstm_num_layers"] = num_layers
        if "attention.query.weight" in state_dict:
            cfg["attention_dim"] = int(state_dict["attention.query.weight"].shape[0])
        cfg["use_layer_norm"] = any(key.startswith("layer_norm") for key in state_dict)

    elif model_type == "gru":
        hidden_dim = None
        for key, value in state_dict.items():
            if key.startswith("gru.weight_hh_l0"):
                hidden_dim = int(value.shape[1])
                break
        if hidden_dim is not None:
            cfg["gru_hidden_dim"] = hidden_dim
        num_layers = sum(1 for key in state_dict if key.startswith("gru.weight_hh_l"))
        if num_layers:
            cfg["gru_num_layers"] = num_layers
        if "attention.query.weight" in state_dict:
            cfg["attention_dim"] = int(state_dict["attention.query.weight"].shape[0])
        cfg["use_layer_norm"] = any(key.startswith("layer_norm") for key in state_dict)

    return cfg


def compute_classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> MetricsDict:
    y_pred = (probs >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = math.nan

    pr_auc = average_precision_score(y_true, probs)
    eps = 1e-7
    logloss = log_loss(y_true, np.clip(probs, eps, 1 - eps))

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        # Handle single-class predictions
        tn = fp = fn = tp = 0
        if np.all(y_pred == 1):
            tp = int(np.sum(y_true == 1))
            fn = int(np.sum(y_true == 0))
        elif np.all(y_pred == 0):
            tn = int(np.sum(y_true == 0))
            fp = int(np.sum(y_true == 1))

    metrics: MetricsDict = {
        "accuracy": float(accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1": float(f1_macro),
        "precision_pos": float(precision_pos),
        "recall_pos": float(recall_pos),
        "f1_pos": float(f1_pos),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "log_loss": float(logloss),
        "loss": float(logloss),
        "confusion_matrix_tn": float(tn),
        "confusion_matrix_fp": float(fp),
        "confusion_matrix_fn": float(fn),
        "confusion_matrix_tp": float(tp),
    }
    return metrics


def compute_threshold_grid(y_true: np.ndarray, probs: np.ndarray, step: float = 0.05) -> List[Dict[str, float]]:
    thresholds = np.arange(step, 1.0, step)
    thresholds = np.append(thresholds, [0.5])
    thresholds = np.unique(np.clip(thresholds, 1e-4, 1 - 1e-4))
    grid: List[Dict[str, float]] = []
    for thr in thresholds:
        y_pred = (probs >= thr).astype(int)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        grid.append({
            "threshold": float(thr),
            "precision_pos": float(prec),
            "recall_pos": float(rec),
            "f1_pos": float(f1),
        })
    return sorted(grid, key=lambda row: row["threshold"])


def summarise_best_threshold(threshold_grid: Sequence[Dict[str, float]]) -> Dict[str, float]:
    best_entry = max(threshold_grid, key=lambda row: (row["f1_pos"], row["recall_pos"]))
    return dict(best_entry)


def compute_pr_curve(y_true: np.ndarray, probs: np.ndarray, max_points: int = 500) -> List[Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    curve: List[Dict[str, float]] = []
    for thr, prec, rec in zip(np.append(thresholds, 1.0), precision, recall):
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        curve.append({
            "threshold": float(thr),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })
    if len(curve) > max_points:
        indices = np.linspace(0, len(curve) - 1, num=max_points, dtype=int)
        curve = [curve[i] for i in indices]
    return curve


def evaluate_model(
    checkpoint: CheckpointBundle,
    dataset_info: DatasetInfo,
    test_loader: DataLoader,
    device: torch.device,
    threshold_step: float,
    pr_curve_points: int,
) -> Tuple[MetricsDict, List[Dict[str, float]], Dict[str, float], List[Dict[str, float]], float]:
    sanitized_config = sanitize_model_config(checkpoint.model_type, checkpoint.raw_model_config, dataset_info)

    raw_state = torch.load(checkpoint.path, map_location="cpu", weights_only=False)["model_state_dict"]
    state_dict = clean_state_dict(raw_state)
    final_config = align_config_with_state_dict(checkpoint.model_type, sanitized_config, state_dict)

    model = create_model(checkpoint.model_type, final_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_probs: List[float] = []
    all_labels: List[int] = []

    start_time = time.time()
    with torch.no_grad():
        for features, labels, asset_ids in tqdm(test_loader, desc=f"Evaluating {checkpoint.model_type.upper()}", leave=False):
            features = features.to(device)
            labels = labels.to(device)
            asset_ids = asset_ids.to(device)

            logits = model(features, asset_ids)
            if logits.ndim > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    inference_time = time.time() - start_time

    probs_array = np.asarray(all_probs, dtype=np.float64)
    labels_array = np.asarray(all_labels, dtype=np.int64)

    metrics_default = compute_classification_metrics(labels_array, probs_array, threshold=0.5)
    threshold_grid = compute_threshold_grid(labels_array, probs_array, step=threshold_step)
    best_threshold = summarise_best_threshold(threshold_grid)
    pr_curve = compute_pr_curve(labels_array, probs_array, max_points=pr_curve_points)

    return metrics_default, threshold_grid, best_threshold, pr_curve, inference_time


def build_test_loader(X: np.ndarray, y: np.ndarray, asset_ids: np.ndarray, batch_size: int, num_workers: int) -> DataLoader:
    features = torch.from_numpy(X.astype(np.float32, copy=False))
    labels = torch.from_numpy(y.astype(np.int64, copy=False))
    assets = torch.from_numpy(asset_ids.astype(np.int64, copy=False))

    dataset = TensorDataset(features, labels, assets)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def parse_model_overrides(overrides: Optional[Sequence[str]]) -> Dict[str, Path]:
    if not overrides:
        return {name: path for name, path in DEFAULT_MODEL_PATHS.items()}

    mapping: Dict[str, Path] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --model-checkpoint entry '{item}'. Expected format name=/path/to/checkpoint.pt")
        name, path_str = item.split("=", 1)
        mapping[name.strip()] = _resolve_path(path_str.strip())
    return mapping


def format_delta(current: float, reference: Optional[float]) -> str:
    if reference is None or math.isnan(reference):
        return "n/a"
    delta = current - reference
    sign = "+" if delta >= 0 else ""
    return f"{current:.3f} ({sign}{delta:.3f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HPO models on the Phase 3 test split")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to enhanced dataset directory")
    parser.add_argument("--model-checkpoint", dest="model_checkpoints", action="append",
                        help="Named checkpoint in the form alias=/path/to/checkpoint.pt")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON results")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="Step size for threshold sweep")
    parser.add_argument("--pr-curve-points", type=int, default=500, help="Maximum points stored for the precision-recall curve")

    args = parser.parse_args()

    dataset_dir = _resolve_path(args.dataset)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test, test_asset_ids, metadata = load_dataset(dataset_dir)
    asset_mapping = _load_asset_mapping(dataset_dir)
    dataset_info = infer_dataset_info(X_test, test_asset_ids, asset_mapping)

    if dataset_info.n_features != 23:
        print(f"[warning] Detected {dataset_info.n_features} features instead of expected 23.")
    if metadata.get("test_samples") and metadata["test_samples"] != int(len(y_test)):
        print(f"[warning] Metadata test_samples={metadata['test_samples']} but loaded {len(y_test)} rows.")

    model_paths = parse_model_overrides(args.model_checkpoints)

    test_loader = build_test_loader(X_test, y_test, test_asset_ids, args.batch_size, args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    evaluation_results = {}
    console_rows: List[Tuple[str, str, str, str, str, str]] = []

    for name, checkpoint_path in model_paths.items():
        checkpoint = load_checkpoint(checkpoint_path)
        metrics_default, threshold_grid, best_threshold, pr_curve, inference_time = evaluate_model(
            checkpoint, dataset_info, test_loader, device, args.threshold_step, args.pr_curve_points
        )

        optimal_threshold = best_threshold
        val_metrics = checkpoint.validation_metrics or {}

        evaluation_results[name] = {
            "alias": name,
            "model_type": checkpoint.model_type,
            "checkpoint": str(checkpoint.path),
            "trial_id": checkpoint.trial_id,
            "epoch": checkpoint.epoch,
            "validation_metrics": val_metrics,
            "test_metrics_threshold_0_50": metrics_default,
            "threshold_grid": threshold_grid,
            "optimal_threshold": optimal_threshold,
            "precision_recall_curve": pr_curve,
            "inference_time_sec": inference_time,
            "samples_per_second": float(len(y_test) / max(inference_time, 1e-6)),
            "dataset": {
                "test_samples": int(len(y_test)),
                "positive_ratio": float(np.mean(y_test)),
            },
        }

        val_f1 = val_metrics.get("f1_pos") or val_metrics.get("f1")
        val_recall = val_metrics.get("recall_pos") or val_metrics.get("recall")

        display_name = f"{name} [{checkpoint.model_type.upper()}]"
        if checkpoint.trial_id:
            display_name += f" ({checkpoint.trial_id})"
        console_rows.append((
            display_name,
            format_delta(metrics_default["f1_pos"], val_f1),
            format_delta(metrics_default["recall_pos"], val_recall),
            f"{metrics_default['precision_pos']:.3f}",
            f"{optimal_threshold['threshold']:.2f}",
            f"{optimal_threshold['f1_pos']:.3f}"
        ))

    best_model_name = max(evaluation_results.items(), key=lambda item: item[1]["optimal_threshold"]["f1_pos"])[0]

    now_utc = datetime.now(UTC)
    timestamp = now_utc.strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"test_evaluation_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({
            "generated_at": now_utc.isoformat(),
            "device": str(device),
            "dataset_dir": str(dataset_dir),
            "results": evaluation_results,
            "best_model": best_model_name,
        }, f, indent=2)

    header = (
        f"\nTest Set Evaluation Summary (dataset: {dataset_dir})\n"
        + "=" * 88
        + "\nModel                       Test F1+ (Δ vs Val)   Test Recall+ (Δ)   Test Prec+   Opt Thr   Opt F1+\n"
        + "-" * 88
    )
    print(header)
    for row in console_rows:
        print(f"{row[0]:<29} {row[1]:>18} {row[2]:>20} {row[3]:>11} {row[4]:>8} {row[5]:>9}")
    print("-" * 88)
    print(f"Best production candidate (max F1_pos): {best_model_name}")
    print(f"Detailed report written to: {report_path}\n")


if __name__ == "__main__":
    main()
