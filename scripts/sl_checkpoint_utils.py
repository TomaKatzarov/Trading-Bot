"""Utility functions for SL checkpoint loading and benchmarking.

Provides helpers to load supervised-learning checkpoints, sanitize model
configurations, instantiate compatible PyTorch models, and run inference
utilities shared between validation and benchmarking scripts.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from joblib import load as joblib_load


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.models.nn_architectures import create_model, get_model_info

# Allow numpy scalar objects when loading legacy checkpoints using public API.
torch.serialization.add_safe_globals([np.generic])


@dataclass
class DatasetInfo:
    """Metadata describing the pregenerated supervised-learning dataset."""

    n_features: int
    lookback_window: int
    num_assets_dataset: int
    num_assets_mapping: Optional[int]

    @property
    def inferred_num_assets(self) -> int:
        """Infer the maximum number of assets from dataset and mapping."""
        values = [self.num_assets_dataset]
        if self.num_assets_mapping is not None:
            values.append(self.num_assets_mapping)
        return max(values)


@dataclass
class CheckpointArtifacts:
    """Bundle of artifacts associated with a single SL checkpoint."""

    checkpoint_path: Path
    scalers_path: Optional[Path]
    metadata_path: Optional[Path]


@dataclass
class LoadedCheckpoint:
    """Container for a loaded checkpoint and instantiated model."""

    model_type: str
    model: torch.nn.Module
    state_dict: Mapping[str, torch.Tensor]
    raw_config: Mapping[str, object]
    sanitized_config: Mapping[str, object]
    metrics: Mapping[str, float]
    epoch: int
    metadata: Optional[Mapping[str, object]]
    scalers: Optional[object]


def list_checkpoint_artifacts(root_dir: Path) -> List[CheckpointArtifacts]:
    """Discover checkpoint directories under ``root_dir``.

    Parameters
    ----------
    root_dir:
        Base directory containing per-model checkpoint folders.

    Returns
    -------
    List[CheckpointArtifacts]
        One entry per discovered checkpoint directory.
    """
    artifacts: List[CheckpointArtifacts] = []
    if not root_dir.exists():
        return artifacts

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        checkpoint_path = child / "model.pt"
        scalers_path = child / "scalers.joblib"
        metadata_path = child / "metadata.json"
        artifacts.append(
            CheckpointArtifacts(
                checkpoint_path=checkpoint_path,
                scalers_path=scalers_path if scalers_path.exists() else None,
                metadata_path=metadata_path if metadata_path.exists() else None,
            )
        )
    return artifacts


def load_asset_mapping(mapping_path: Path) -> Optional[Mapping[str, int]]:
    """Load the asset ID mapping file if available."""
    if not mapping_path.exists():
        return None
    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = data.get("symbol_to_id")
    if isinstance(mapping, dict):
        return mapping
    return None


def load_dataset_arrays(
    dataset_dir: Path,
    split: str = "test",
    sample_limit: Optional[int] = 2048,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature and asset-id arrays for the requested split.

    Parameters
    ----------
    dataset_dir:
        Directory containing ``*_X.npy`` and ``*_asset_ids.npy`` files.
    split:
        Dataset split to load (``train``, ``val``, or ``test``).
    sample_limit:
        Optional maximum number of samples to return. If ``None`` all rows are
        returned. When fewer samples than ``sample_limit`` exist, the full array
        is used. Selection is random but reproducible via ``rng_seed``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature tensor of shape ``(N, lookback, n_features)`` and asset IDs of
        shape ``(N,)``.
    """
    features_path = dataset_dir / f"{split}_X.npy"
    asset_ids_path = dataset_dir / f"{split}_asset_ids.npy"
    if not features_path.exists() or not asset_ids_path.exists():
        raise FileNotFoundError(
            f"Missing dataset split '{split}' under {dataset_dir}."
        )

    features = np.load(features_path)
    asset_ids = np.load(asset_ids_path)
    if features.shape[0] != asset_ids.shape[0]:
        raise ValueError(
            "Mismatched number of samples between features and asset IDs: "
            f"{features.shape[0]} vs {asset_ids.shape[0]}"
        )

    if sample_limit is not None and features.shape[0] > sample_limit:
        rng = np.random.default_rng(rng_seed)
        indices = rng.choice(features.shape[0], size=sample_limit, replace=False)
        indices.sort()
        features = features[indices]
        asset_ids = asset_ids[indices]

    return features.astype(np.float32), asset_ids.astype(np.int64)


def infer_dataset_info(
    features: np.ndarray,
    asset_ids: np.ndarray,
    asset_mapping: Optional[Mapping[str, int]],
    metadata_path: Optional[Path] = None,
) -> DatasetInfo:
    """Infer dataset properties required for model construction."""
    if features.ndim != 3:
        raise ValueError(
            f"Expected features with 3 dimensions, got shape {features.shape}."
        )

    num_assets_dataset = int(asset_ids.max()) + 1 if asset_ids.size else 1
    num_assets_mapping = len(asset_mapping) if asset_mapping is not None else None
    lookback_window = int(features.shape[1])
    n_features = int(features.shape[2])

    if metadata_path and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        n_features = int(metadata.get("feature_count", n_features))
        lookback_window = int(metadata.get("lookback_window", lookback_window))
        num_assets_metadata = metadata.get("num_symbols")
        if isinstance(num_assets_metadata, int):
            num_assets_dataset = max(num_assets_dataset, num_assets_metadata)

    return DatasetInfo(
        n_features=n_features,
        lookback_window=lookback_window,
        num_assets_dataset=num_assets_dataset,
        num_assets_mapping=num_assets_mapping,
    )


def _clamp_dropout(value: float) -> float:
    return float(min(max(value, 0.0), 0.8))


def sanitize_model_config(
    model_type: str,
    raw_config: Mapping[str, object],
    dataset_info: DatasetInfo,
) -> Dict[str, object]:
    """Normalize raw model configuration extracted from checkpoint files."""
    cfg: Dict[str, object] = {
        "n_features": dataset_info.n_features,
        "lookback_window": dataset_info.lookback_window,
        "num_assets": max(
            dataset_info.inferred_num_assets,
            int(raw_config.get("num_assets", dataset_info.inferred_num_assets) or dataset_info.inferred_num_assets),
        ),
        "asset_embedding_dim": int(raw_config.get("asset_embedding_dim", 8) or 8),
        "dropout_rate": _clamp_dropout(float(raw_config.get("dropout_rate", 0.3) or 0.3)),
    }

    attention_key = "attention_dim" if "attention_dim" in raw_config else "attention_dim极"

    if model_type == "mlp":
        hidden_layers = int(raw_config.get("hidden_layers", 3) or 3)
        hidden_dims: List[int] = []
        for idx in range(1, hidden_layers + 1):
            key = f"hidden_dim_{idx}"
            if key in raw_config:
                try:
                    hidden_dims.append(int(raw_config[key]))
                except (TypeError, ValueError):
                    continue
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
        cfg["lstm_hidden_dim"] = int(raw_config.get("lstm_hidden_dim", 128) or 128)
        cfg["lstm_num_layers"] = int(raw_config.get("lstm_num_layers", 2) or 2)
        cfg["attention_dim"] = int(raw_config.get(attention_key, 64) or 64)
        cfg["use_layer_norm"] = bool(raw_config.get("use_layer_norm", True))
        cfg["cnn_filters"] = tuple(raw_config.get("cnn_filters", (32, 64)))
        cfg["cnn_kernel_sizes"] = tuple(raw_config.get("cnn_kernel_sizes", (3, 3)))
        cfg["cnn_stride"] = int(raw_config.get("cnn_stride", 1) or 1)
        cfg["use_max_pooling"] = bool(raw_config.get("use_max_pooling", True))
    else:
        raise ValueError(f"Unsupported model type '{model_type}' in checkpoint configuration.")

    return cfg


def clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove wrapping prefixes added by DDP or compiled graphs."""
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
    config: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    """Adjust sanitized configuration using tensor shapes from ``state_dict``."""
    cleaned_state = clean_state_dict(state_dict)
    cfg = dict(config)

    asset_weight = cleaned_state.get("asset_embedding.weight")
    if asset_weight is not None:
        cfg["num_assets"] = int(asset_weight.shape[0])
        cfg["asset_embedding_dim"] = int(asset_weight.shape[1])

    if model_type == "mlp":
        linear_weights: List[Tuple[int, int]] = []
        for key, tensor in cleaned_state.items():
            if key.startswith("mlp.") and key.endswith(".weight"):
                parts = key.split(".")
                if len(parts) < 3:
                    continue
                try:
                    layer_index = int(parts[1])
                except ValueError:
                    continue
                linear_weights.append((layer_index, int(tensor.shape[0])))
        if linear_weights:
            linear_weights.sort(key=lambda item: item[0])
            hidden_dims = [dim for _, dim in linear_weights[:-1]]
            if hidden_dims:
                cfg["hidden_dims"] = tuple(hidden_dims)

    elif model_type == "lstm":
        hidden_dim = None
        for key, tensor in cleaned_state.items():
            if key.startswith("lstm.weight_hh_l0"):
                hidden_dim = int(tensor.shape[1])
                break
        if hidden_dim is not None:
            cfg["lstm_hidden_dim"] = hidden_dim
        num_layers = sum(1 for key in cleaned_state if key.startswith("lstm.weight_hh_l"))
        if num_layers:
            cfg["lstm_num_layers"] = num_layers
        attention_weight = cleaned_state.get("attention.query.weight")
        if attention_weight is not None:
            cfg["attention_dim"] = int(attention_weight.shape[0])
        cfg["use_layer_norm"] = any("layer_norm" in key for key in cleaned_state)

    elif model_type == "gru":
        hidden_dim = None
        for key, tensor in cleaned_state.items():
            if key.startswith("gru.weight_hh_l0"):
                hidden_dim = int(tensor.shape[1])
                break
        if hidden_dim is not None:
            cfg["gru_hidden_dim"] = hidden_dim
        num_layers = sum(1 for key in cleaned_state if key.startswith("gru.weight_hh_l"))
        if num_layers:
            cfg["gru_num_layers"] = num_layers
        attention_weight = cleaned_state.get("attention.query.weight")
        if attention_weight is not None:
            cfg["attention_dim"] = int(attention_weight.shape[0])
        cfg["use_layer_norm"] = any("layer_norm" in key for key in cleaned_state)

    return cfg


def load_checkpoint_bundle(artifact: CheckpointArtifacts) -> Dict[str, object]:
    """Load checkpoint file and associated metadata/scalers if present."""
    if not artifact.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {artifact.checkpoint_path}")

    checkpoint = torch.load(artifact.checkpoint_path, map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("config", {})
    model_type = str(raw_config.get("model_type", "unknown")).lower()
    metrics = checkpoint.get("metrics", {})
    epoch = int(checkpoint.get("epoch", -1))

    metadata: Optional[Mapping[str, object]] = None
    if artifact.metadata_path and artifact.metadata_path.exists():
        with artifact.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    scalers = None
    if artifact.scalers_path and artifact.scalers_path.exists():
        scalers = joblib_load(artifact.scalers_path)

    return {
        "model_type": model_type,
        "raw_config": raw_config.get("model_config", {}),
        "state_dict": checkpoint.get("model_state_dict", {}),
        "metrics": metrics,
        "epoch": epoch,
        "metadata": metadata,
        "scalers": scalers,
    }


def instantiate_model_from_checkpoint(
    model_type: str,
    sanitized_config: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
) -> torch.nn.Module:
    """Create and hydrate a model instance from checkpoint data."""
    model = create_model(model_type, dict(sanitized_config))
    model.load_state_dict(clean_state_dict(state_dict))
    model.eval()
    return model


def summarize_model(model: torch.nn.Module) -> Dict[str, float]:
    """Return a lightweight summary with parameter counts."""
    info = get_model_info(model)
    return {
        "trainable_parameters": float(info["trainable_parameters"]),
        "total_parameters": float(info["total_parameters"]),
        "model_size_mb": float(info["model_size_mb"]),
    }


def ensure_directory(path: Path) -> None:
    """Create a directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def compute_logit_statistics(logits: torch.Tensor) -> Dict[str, float]:
    """Compute descriptive statistics for model logits."""
    if logits.numel() == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": float(torch.min(logits).item()),
        "max": float(torch.max(logits).item()),
        "mean": float(torch.mean(logits).item()),
        "std": float(torch.std(logits).item()),
    }


def run_batched_inference(
    model: torch.nn.Module,
    features: np.ndarray,
    asset_ids: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
    repeats: int = 1,
    warmup: int = 0,
) -> Dict[str, object]:
    """Run batched inference and measure latency statistics.

    Parameters
    ----------
    model:
        Instantiated PyTorch model configured for evaluation.
    features:
        Feature tensor shaped ``(N, lookback, n_features)``.
    asset_ids:
        Asset identifiers corresponding to ``features`` (shape ``(N,)``).
    device:
        Torch device to execute inference on.
    batch_size:
        Batch size for inference.
    repeats:
        Number of timed passes to execute (results averaged).
    warmup:
        Number of untimed warm-up passes prior to measurement.

    Returns
    -------
    Dict[str, object]
        Dictionary containing latency metrics and logits from the final pass.
    """

    if features.shape[0] != asset_ids.shape[0]:
        raise ValueError("Features and asset_ids must contain the same number of rows.")

    model = model.to(device)
    model.eval()

    feature_tensor = torch.from_numpy(features)
    asset_tensor = torch.from_numpy(asset_ids)

    total_samples = feature_tensor.shape[0]
    if total_samples == 0:
        return {
            "logits": torch.empty(0),
            "avg_latency_ms_per_sample": float("nan"),
            "total_latency_s": 0.0,
            "samples_per_second": float("nan"),
        }

    def _iterate_batches():
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            yield feature_tensor[start:end], asset_tensor[start:end]

    with torch.no_grad():
        for _ in range(warmup):
            for batch_features, batch_assets in _iterate_batches():
                batch_features = batch_features.to(device)
                batch_assets = batch_assets.to(device)
                model(batch_features, batch_assets)

    measured_time = 0.0
    logits_list: List[torch.Tensor] = []

    with torch.no_grad():
        for repeat_idx in range(repeats):
            for batch_features, batch_assets in _iterate_batches():
                batch_features = batch_features.to(device)
                batch_assets = batch_assets.to(device)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                logits = model(batch_features, batch_assets)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                measured_time += time.perf_counter() - start_time

                if repeat_idx == repeats - 1:
                    logits_list.append(logits.detach().cpu())

    total_samples_processed = total_samples * repeats
    avg_latency_ms = (measured_time / total_samples_processed) * 1000
    samples_per_second = total_samples_processed / measured_time if measured_time > 0 else float("nan")

    return {
        "logits": torch.cat(logits_list, dim=0) if logits_list else torch.empty(0),
        "avg_latency_ms_per_sample": float(avg_latency_ms),
        "total_latency_s": float(measured_time),
        "samples_per_second": float(samples_per_second),
    }