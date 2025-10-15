"""Utility functions for SL checkpoint loading and benchmarking.

Provides helpers to load supervised-learning checkpoints, sanitize model
configurations, instantiate compatible PyTorch models, and run inference
utilities shared between validation and benchmarking scripts.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import warnings

import numpy as np
import torch
from joblib import load as joblib_load

try:  # pragma: no cover - optional dependency guard
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover - sklearn optional in runtime envs
    InconsistentVersionWarning = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.models.nn_architectures import create_model, get_model_info

# Allow numpy scalar objects when loading legacy checkpoints using public API.
torch.serialization.add_safe_globals([np.generic])

LOGGER = logging.getLogger(__name__)


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
    asset_mapping: Optional[Mapping[str, int]]
    checkpoint_dir: Path
    default_asset_id: Optional[int] = None
    device: torch.device = torch.device("cpu")
    scaler_mean: Optional[torch.Tensor] = None
    scaler_scale: Optional[torch.Tensor] = None


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

    attention_key = next((key for key in raw_config.keys() if "attention_dim" in key), "attention_dim")

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

    if InconsistentVersionWarning is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            checkpoint = torch.load(artifact.checkpoint_path, map_location="cpu", weights_only=False)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        try:
            if InconsistentVersionWarning is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    scalers = joblib_load(artifact.scalers_path)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scalers = joblib_load(artifact.scalers_path)
        except Exception:
            scalers = None
    if scalers is None:
        scalers = checkpoint.get("scalers")

    return {
        "model_type": model_type,
        "raw_config": raw_config.get("model_config", {}),
        "state_dict": checkpoint.get("model_state_dict", {}),
        "metrics": metrics,
        "epoch": epoch,
        "metadata": metadata,
        "scalers": scalers,
        "asset_mapping": checkpoint.get("asset_id_map"),
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


def _resolve_checkpoint_artifacts(path: Path) -> Tuple[Path, CheckpointArtifacts]:
    """Return the checkpoint directory and associated artifact bundle."""

    if path.is_dir():
        base_dir = path
        checkpoint_file = base_dir / "model.pt"
    else:
        base_dir = path.parent
        checkpoint_file = path

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_file}")

    scalers_path = base_dir / "scalers.joblib"
    metadata_path = base_dir / "metadata.json"

    artifact = CheckpointArtifacts(
        checkpoint_path=checkpoint_file,
        scalers_path=scalers_path if scalers_path.exists() else None,
        metadata_path=metadata_path if metadata_path.exists() else None,
    )
    return base_dir, artifact


def _infer_dataset_from_config(
    raw_config: Mapping[str, object],
    metadata: Optional[Mapping[str, object]],
    asset_mapping: Optional[Mapping[str, int]],
) -> DatasetInfo:
    """Best-effort dataset metadata reconstruction from checkpoint artifacts."""

    def _coerce_int(value: Any, default: int) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    meta_dict: Optional[Dict[str, Any]] = dict(metadata) if isinstance(metadata, Mapping) else None

    n_features = _coerce_int(raw_config.get("n_features"), meta_dict.get("feature_count") if meta_dict else 1)
    lookback = _coerce_int(raw_config.get("lookback_window"), meta_dict.get("lookback_window") if meta_dict else 1)
    num_assets_raw = _coerce_int(raw_config.get("num_assets"), len(asset_mapping) if asset_mapping else 1)
    num_assets_map = len(asset_mapping) if asset_mapping else None

    if meta_dict and "num_symbols" in meta_dict:
        num_assets_raw = max(num_assets_raw, _coerce_int(meta_dict.get("num_symbols"), num_assets_raw))

    return DatasetInfo(
        n_features=max(1, n_features),
        lookback_window=max(1, lookback),
        num_assets_dataset=max(1, num_assets_raw),
        num_assets_mapping=num_assets_map,
    )


def _resolve_device(preference: Optional[str | torch.device]) -> torch.device:
    if isinstance(preference, torch.device):
        if preference.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA device %s requested but unavailable; falling back to CPU", preference)
            return torch.device("cpu")
        return preference

    if isinstance(preference, str):
        normalized = preference.strip().lower()
        if normalized in {"auto", "default", ""}:
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if normalized == "cpu":
            return torch.device("cpu")
        if normalized.startswith("cuda"):
            if torch.cuda.is_available():
                try:
                    return torch.device(normalized)
                except Exception:
                    LOGGER.warning("Invalid CUDA device string '%s'; defaulting to cuda", preference)
                    return torch.device("cuda")
            LOGGER.warning("CUDA requested (%s) but unavailable; using CPU", preference)
            return torch.device("cpu")
        LOGGER.warning("Unrecognized device preference '%s'; defaulting to CPU", preference)
        return torch.device("cpu")

    return torch.device("cpu")


def _prepare_scaler_tensors(
    scalers: Any,
    device: torch.device,
    flatten_dim: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    scaler = _select_scaler(scalers)
    if scaler is None:
        return None, None

    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        return None, None

    mean_arr = np.asarray(mean, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32)

    if mean_arr.size != flatten_dim or scale_arr.size != flatten_dim:
        LOGGER.debug(
            "Scaler dimensionality mismatch (expected=%s, mean=%s, scale=%s)",
            flatten_dim,
            mean_arr.size,
            scale_arr.size,
        )
        return None, None

    scale_arr = np.where(scale_arr == 0.0, 1.0, scale_arr)
    mean_tensor = torch.from_numpy(mean_arr).to(device=device)
    scale_tensor = torch.from_numpy(scale_arr).to(device=device)
    return mean_tensor, scale_tensor


def load_sl_checkpoint(checkpoint_path: Path | str, device: Optional[str | torch.device] = None) -> LoadedCheckpoint:
    """Hydrate a supervised-learning checkpoint for inference within RL."""

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"SL checkpoint path does not exist: {path}")

    base_dir, artifact = _resolve_checkpoint_artifacts(path)
    bundle = load_checkpoint_bundle(artifact)

    model_type = bundle.get("model_type", "unknown")
    if not model_type or model_type == "unknown":
        raise ValueError(f"Unable to determine model type for checkpoint at {base_dir}")

    raw_config = dict(bundle.get("raw_config", {}))
    metadata = bundle.get("metadata")
    if isinstance(metadata, Mapping):
        metadata = dict(metadata)
        metadata.setdefault("checkpoint_dir", str(base_dir))

    asset_mapping_raw = bundle.get("asset_mapping")
    asset_mapping = dict(asset_mapping_raw) if isinstance(asset_mapping_raw, Mapping) else None
    dataset_info = _infer_dataset_from_config(raw_config, metadata, asset_mapping)

    sanitized_config = sanitize_model_config(model_type, raw_config, dataset_info)
    state_dict = bundle.get("state_dict", {})
    sanitized_config = align_config_with_state_dict(model_type, sanitized_config, state_dict)

    target_device = _resolve_device(device)
    model = instantiate_model_from_checkpoint(model_type, sanitized_config, state_dict)
    model = model.to(target_device)

    lookback = int(sanitized_config.get("lookback_window", dataset_info.lookback_window))
    n_features = int(sanitized_config.get("n_features", dataset_info.n_features))
    flatten_dim = max(1, lookback * n_features)
    scaler_mean, scaler_scale = _prepare_scaler_tensors(bundle.get("scalers"), target_device, flatten_dim)

    return LoadedCheckpoint(
        model_type=model_type,
        model=model,
        state_dict=clean_state_dict(state_dict),
        raw_config=raw_config,
        sanitized_config=sanitized_config,
        metrics=bundle.get("metrics", {}),
        epoch=int(bundle.get("epoch", -1)),
        metadata=metadata,
        scalers=bundle.get("scalers"),
        asset_mapping=asset_mapping,
        checkpoint_dir=base_dir,
        device=target_device,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )


def _select_scaler(scalers: Any) -> Optional[Any]:
    if scalers is None:
        return None
    if isinstance(scalers, Mapping):
        for key in ("global", "standard", "default"):
            candidate = scalers.get(key)
            if candidate is not None and hasattr(candidate, "transform"):
                return candidate
        for candidate in scalers.values():
            if hasattr(candidate, "transform"):
                return candidate
        return None
    if hasattr(scalers, "transform"):
        return scalers
    return None


def run_inference(
    bundle: LoadedCheckpoint,
    input_tensor: Any,
    asset_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a forward pass for RL feature augmentation using SL logits."""

    if bundle is None:
        raise ValueError("bundle must be a LoadedCheckpoint instance")

    if isinstance(input_tensor, torch.Tensor):
        features_tensor = input_tensor.to(device=bundle.device, dtype=torch.float32, non_blocking=True)
    else:
        features_tensor = torch.as_tensor(input_tensor, dtype=torch.float32, device=bundle.device)

    if features_tensor.ndim == 2:
        features_tensor = features_tensor.unsqueeze(0)
    elif features_tensor.ndim != 3:
        raise ValueError(
            f"Expected input tensor with shape (batch, lookback, features); received shape {tuple(features_tensor.shape)}"
        )

    features_tensor = features_tensor.contiguous()
    batch_size, lookback, feature_dim = features_tensor.shape

    if bundle.scaler_mean is not None and bundle.scaler_scale is not None:
        flat = features_tensor.view(batch_size, -1)
        if bundle.scaler_mean.numel() == flat.shape[1]:
            mean = bundle.scaler_mean
            scale = bundle.scaler_scale
            if mean.device != flat.device:
                mean = mean.to(flat.device)
                scale = scale.to(flat.device)
            flat = (flat - mean) / scale
            features_tensor = flat.view(batch_size, lookback, feature_dim)
        else:
            LOGGER.debug(
                "Scaler tensor mismatch for checkpoint %s (expected=%s, got=%s)",
                bundle.checkpoint_dir,
                flat.shape[1],
                bundle.scaler_mean.numel(),
            )
    elif bundle.scalers is not None:
        scaler = _select_scaler(bundle.scalers)
        if scaler is not None:
            features_cpu = features_tensor.detach().cpu()
            flat_cpu = features_cpu.view(batch_size, -1).numpy()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    transformed = scaler.transform(flat_cpu)
                features_tensor = (
                    torch.as_tensor(transformed, dtype=torch.float32, device=bundle.device)
                    .view(batch_size, lookback, feature_dim)
                )
            except Exception:
                pass

    resolved_asset = asset_id
    if resolved_asset is None:
        resolved_asset = bundle.default_asset_id
    if resolved_asset is None and bundle.asset_mapping:
        try:
            resolved_asset = next(iter(bundle.asset_mapping.values()))
        except StopIteration:
            resolved_asset = None
    if resolved_asset is None:
        resolved_asset = 0

    asset_tensor = torch.full(
        (batch_size,),
        int(resolved_asset),
        dtype=torch.long,
        device=bundle.device,
    )

    bundle.model.eval()
    with torch.no_grad():
        logits = bundle.model(features_tensor, asset_tensor).squeeze(-1)
        probabilities = torch.sigmoid(logits)

    logits_cpu = logits.detach().cpu()
    probs_cpu = probabilities.detach().cpu()

    if probs_cpu.numel() == 1:
        return {
            "probability": float(probs_cpu.item()),
            "logit": float(logits_cpu.item()),
        }

    return {
        "probability": probs_cpu.tolist(),
        "logits": logits_cpu.tolist(),
    }


__all__ = [
    "DatasetInfo",
    "CheckpointArtifacts",
    "LoadedCheckpoint",
    "list_checkpoint_artifacts",
    "load_asset_mapping",
    "load_dataset_arrays",
    "infer_dataset_info",
    "sanitize_model_config",
    "clean_state_dict",
    "align_config_with_state_dict",
    "load_checkpoint_bundle",
    "instantiate_model_from_checkpoint",
    "summarize_model",
    "ensure_directory",
    "compute_logit_statistics",
    "run_batched_inference",
    "load_sl_checkpoint",
    "run_inference",
]