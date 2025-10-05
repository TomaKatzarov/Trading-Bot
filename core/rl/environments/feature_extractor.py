"""
Feature extraction utilities for the RL trading environment.

This module provides a configurable, high-performance feature extraction pipeline
that transforms raw market data into RL-ready observation tensors. It supports
selective feature inclusion, rolling normalization schemes, and an internal LRU
cache to accelerate repeated window requests during training or evaluation.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["FeatureConfig", "FeatureExtractor"]


@dataclass
class FeatureConfig:
    """Configuration container for the feature extractor.

    Attributes
    ----------
    include_ohlcv:
        Whether to include base OHLCV price and volume information.
    include_technical:
        Include pre-computed technical indicators (moving averages, momentum, etc.).
    include_sentiment:
        Include sentiment signals such as NLP-derived scores.
    include_temporal:
        Include cyclical encodings for temporal context (e.g., day-of-week).
    normalize_method:
        Strategy used to normalize features. Valid options: ``"zscore"``,
        ``"minmax"``, ``"robust"``, or ``"none"``.
    normalize_window:
        Rolling window (in timesteps) used for normalization statistics.
    feature_subset:
        Optional whitelist of feature names to include. Useful for ablations or
        curriculum setups.
    cache_size:
        Maximum number of feature windows to retain in the LRU cache.
    clip_range:
        Numeric range applied after normalization to guard against outliers.
    allow_partial_windows:
        If ``True``, the extractor will pad insufficient history with NaNs and
        allow extraction. When ``False`` a ``ValueError`` is raised instead.
    pad_value:
        Value used to replace NaNs when ``allow_partial_windows`` is enabled.
    dtype:
        Output dtype returned by ``extract_window`` / ``extract_batch``.
    strict_feature_check:
        When ``True`` the extractor raises if optional features are missing.
    """

    include_ohlcv: bool = True
    include_technical: bool = True
    include_sentiment: bool = True
    include_temporal: bool = True

    normalize_method: str = "zscore"
    normalize_window: int = 252

    feature_subset: Optional[Sequence[str]] = None
    cache_size: int = 1024

    clip_range: Tuple[float, float] = (-10.0, 10.0)
    allow_partial_windows: bool = False
    pad_value: float = 0.0
    dtype: np.dtype = np.float32
    strict_feature_check: bool = True

    def as_dict(self) -> Dict[str, object]:
        """Return the configuration as a plain dictionary for logging."""

        return asdict(self)


@dataclass
class _CacheStats:
    """Lightweight tracker for cache performance metrics."""

    hits: int = 0
    misses: int = 0

    def register_hit(self):
        self.hits += 1

    def register_miss(self):
        self.misses += 1

    def summary(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100.0) if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate_pct": hit_rate,
        }


class _LRUWindowCache:
    """Simple LRU cache tailored for numpy window arrays."""

    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError("cache_size must be positive")
        self.maxsize = int(maxsize)
        self._store: "OrderedDict[Tuple[int, bool], np.ndarray]" = OrderedDict()

    def get(self, key: Tuple[int, bool]) -> Optional[np.ndarray]:
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value  # move to end (most recently used)
        return value

    def put(self, key: Tuple[int, bool], value: np.ndarray) -> None:
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self.maxsize:
            self._store.popitem(last=False)
        # store an immutable view to protect cached value
        readonly = np.array(value, copy=True)
        readonly.setflags(write=False)
        self._store[key] = readonly

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)


class FeatureExtractor:
    """Core feature extraction pipeline used by the trading environment.

    The extractor expects a tidy ``pandas.DataFrame`` whose index is sorted by
    timestamp (ascending) and whose columns contain the desired feature inputs.
    Feature groups can be toggled via :class:`FeatureConfig`, and normalization
    statistics are pre-computed to accelerate repeated calls.
    """

    _OHLCV_COLUMNS: Tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
    )

    _TECH_COLUMNS: Tuple[str, ...] = (
        "SMA_10",
        "SMA_20",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "RSI_14",
        "Stochastic_K",
        "Stochastic_D",
        "ADX_14",
        "ATR_14",
        "BB_bandwidth",
        "OBV",
        "Volume_SMA_20",
        "Return_1h",
    )

    _SENTIMENT_COLUMNS: Tuple[str, ...] = (
        "sentiment_score_hourly_ffill",
    )

    _TEMPORAL_COLUMNS: Tuple[str, ...] = (
        "DayOfWeek_sin",
        "DayOfWeek_cos",
    )

    _ALLOWED_NORMALIZERS: Tuple[str, ...] = ("zscore", "minmax", "robust", "none")

    def __init__(
        self,
        data: pd.DataFrame,
        config: FeatureConfig,
        lookback_window: int = 24,
        sl_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.lookback_window = int(lookback_window)

        if self.lookback_window <= 0:
            raise ValueError("lookback_window must be positive")

        if config.normalize_method not in self._ALLOWED_NORMALIZERS:
            raise ValueError(
                f"normalize_method must be one of {self._ALLOWED_NORMALIZERS}, "
                f"got '{config.normalize_method}'"
            )

        self.data = self._prepare_dataframe(data)
        self.feature_cols = self._define_feature_columns()

        self._feature_matrix = self.data[self.feature_cols].to_numpy(copy=False)
        self._cache = _LRUWindowCache(config.cache_size)
        self._cache_stats = _CacheStats()
        self.sl_models: Dict[str, Any] = sl_models or {}

        self._normalization_params: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        if config.normalize_method != "none":
            self._normalization_params = self._compute_normalization_params()

        logger.info(
            "FeatureExtractor initialized | features=%s | normalize=%s | window=%s",
            len(self.feature_cols),
            config.normalize_method,
            self.lookback_window,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_window(self, end_idx: int, normalize: bool = True) -> np.ndarray:
        """Return a single feature window ending *just before* ``end_idx``.

        Parameters
        ----------
        end_idx:
            Positional index (exclusive) of the final timestep. Equivalent to the
            index that would be passed to ``pandas.DataFrame.iloc`` for slicing.
        normalize:
            Whether to apply normalization according to the configured method.

        Returns
        -------
        np.ndarray
            Array with shape ``(lookback_window, num_features)`` and dtype from
            :class:`FeatureConfig`.
        """

        key = (int(end_idx), bool(normalize))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_stats.register_hit()
            return cached.copy()

        self._cache_stats.register_miss()
        window = self._extract_window(end_idx)

        if normalize and self._normalization_params is not None:
            window = self._normalize_window(window, end_idx)

        window = window.astype(self.config.dtype, copy=False)
        window = np.clip(window, *self.config.clip_range)
        self._cache.put(key, window)

        return window.copy()

    def extract_batch(
        self, end_indices: Sequence[int], normalize: bool = True
    ) -> np.ndarray:
        """Vectorized variant of :meth:`extract_window`.

        Parameters
        ----------
        end_indices:
            Iterable of positional indices. Each index is processed independently.
        normalize:
            Whether to apply normalization to each window.

        Returns
        -------
        np.ndarray
            Array with shape ``(batch, lookback_window, num_features)``.
        """

        batch = [self.extract_window(idx, normalize=normalize) for idx in end_indices]
        if not batch:
            return np.empty((0, self.lookback_window, len(self.feature_cols)), dtype=self.config.dtype)
        return np.stack(batch, axis=0)

    def get_sl_predictions(
        self,
        end_idx: int,
        window: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return SL model probabilities for the specified window."""

        if window is None:
            window = self.extract_window(end_idx, normalize=True)

        if not self.sl_models:
            return np.array([0.5, 0.5, 0.5], dtype=np.float32)

        input_tensor = window[np.newaxis, :, :]
        probs: List[float] = []

        try:  # Local import to avoid heavy dependency when unused
            from scripts.sl_checkpoint_utils import run_inference as _run_inference  # type: ignore
        except Exception:  # pragma: no cover - graceful degradation when unavailable
            _run_inference = None

        if _run_inference is None:
            return np.array([0.5, 0.5, 0.5], dtype=np.float32)

        for model_name in ("mlp", "lstm", "gru"):
            bundle = self.sl_models.get(model_name)
            if bundle is None:
                probs.append(0.5)
                continue

            try:
                inference_out = _run_inference(bundle, input_tensor)
                if isinstance(inference_out, dict) and "probability" in inference_out:
                    probs.append(float(inference_out["probability"]))
                elif isinstance(inference_out, (tuple, list)) and len(inference_out) > 0:
                    probs.append(float(inference_out[0]))
                else:
                    probs.append(float(inference_out))
            except Exception as exc:  # pragma: no cover - inference failures
                logger.warning("SL inference failed for %s: %s", model_name, exc)
                probs.append(0.5)

        if not probs:
            return np.array([0.5, 0.5, 0.5], dtype=np.float32)

        return np.array(probs, dtype=np.float32)

    def prefetch(self, end_indices: Iterable[int], normalize: bool = True) -> None:
        """Warm the cache by extracting a series of indices without returning them."""

        for idx in end_indices:
            self.extract_window(idx, normalize=normalize)

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature names."""

        return list(self.feature_cols)

    def get_feature_count(self) -> int:
        """Return the number of features currently configured."""

        return len(self.feature_cols)

    def validate_index(self, idx: int) -> bool:
        """Check whether a given index has sufficient history for extraction."""

        if idx > len(self.data):
            return False
        if idx < 0:
            return False
        return idx >= self.lookback_window or self.config.allow_partial_windows

    def get_valid_range(self) -> Tuple[int, int]:
        """Return the inclusive range of valid end indices."""

        min_valid = self.lookback_window if not self.config.allow_partial_windows else 1
        return min_valid, len(self.data)

    def get_cache_stats(self) -> Dict[str, float]:
        """Return cache statistics for monitoring and debugging."""

        return self._cache_stats.summary()

    def clear_cache(self) -> None:
        """Empty the internal cache and reset statistics."""

        self._cache.clear()
        self._cache_stats = _CacheStats()

    def update_data(self, data: pd.DataFrame) -> None:
        """Replace the underlying dataframe and recompute dependent state.

        This is useful when switching between symbols or refreshing with new
        historical data during evaluation. Cache and normalization statistics are
        rebuilt to ensure consistency.
        """

        self.data = self._prepare_dataframe(data)
        self.feature_cols = self._define_feature_columns()
        self._feature_matrix = self.data[self.feature_cols].to_numpy(copy=False)
        self.clear_cache()

        if self.config.normalize_method != "none":
            self._normalization_params = self._compute_normalization_params()
        else:
            self._normalization_params = None

    def reconfigure(self, config: FeatureConfig) -> None:
        """Update configuration and recompute derived artifacts."""

        self.config = config
        if config.normalize_method not in self._ALLOWED_NORMALIZERS:
            raise ValueError(
                f"normalize_method must be one of {self._ALLOWED_NORMALIZERS}, "
                f"got '{config.normalize_method}'"
            )

        self._cache = _LRUWindowCache(config.cache_size)
        self._cache_stats = _CacheStats()
        self.feature_cols = self._define_feature_columns()
        self._feature_matrix = self.data[self.feature_cols].to_numpy(copy=False)

        if config.normalize_method != "none":
            self._normalization_params = self._compute_normalization_params()
        else:
            self._normalization_params = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("dataframe is empty; cannot extract features")

        df = data.copy()
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)

        return df

    def _define_feature_columns(self) -> List[str]:
        columns: List[str] = []

        if self.config.include_ohlcv:
            columns.extend(self._ensure_columns(self._OHLCV_COLUMNS))

        if self.config.include_technical:
            columns.extend(self._ensure_columns(self._TECH_COLUMNS))

        if self.config.include_sentiment:
            columns.extend(self._ensure_columns(self._SENTIMENT_COLUMNS))

        if self.config.include_temporal:
            columns.extend(self._ensure_columns(self._TEMPORAL_COLUMNS))

        if self.config.feature_subset is not None:
            subset = [col for col in self.config.feature_subset if col in columns]
            if self.config.strict_feature_check and len(subset) != len(self.config.feature_subset):
                missing = set(self.config.feature_subset) - set(columns)
                if missing:
                    raise ValueError(f"Requested feature_subset columns missing: {missing}")
            columns = subset

        if not columns:
            raise ValueError("No features selected; please check FeatureConfig")

        return columns

    def _ensure_columns(self, required: Sequence[str]) -> List[str]:
        available = []
        missing = []
        for col in required:
            if col in self.data.columns:
                available.append(col)
            else:
                missing.append(col)

        if missing and self.config.strict_feature_check:
            raise ValueError(f"Missing required features in dataframe: {missing}")

        return available

    def _compute_normalization_params(self) -> Dict[str, Dict[str, np.ndarray]]:
        logger.debug("Computing normalization parameters | method=%s", self.config.normalize_method)
        method = self.config.normalize_method
        window = max(int(self.config.normalize_window), 2)

        params: Dict[str, Dict[str, np.ndarray]] = {}

        for col in self.feature_cols:
            series = self.data[col].astype(float)

            if method == "zscore":
                rolling_mean = series.rolling(window=window, min_periods=20).mean()
                rolling_std = series.rolling(window=window, min_periods=20).std(ddof=0)
                params[col] = {
                    "mean": rolling_mean.to_numpy(dtype=float, copy=False),
                    "std": rolling_std.to_numpy(dtype=float, copy=False),
                }
            elif method == "minmax":
                rolling_min = series.rolling(window=window, min_periods=20).min()
                rolling_max = series.rolling(window=window, min_periods=20).max()
                params[col] = {
                    "min": rolling_min.to_numpy(dtype=float, copy=False),
                    "max": rolling_max.to_numpy(dtype=float, copy=False),
                }
            elif method == "robust":
                rolling_median = series.rolling(window=window, min_periods=20).median()
                rolling_q1 = series.rolling(window=window, min_periods=20).quantile(0.25)
                rolling_q3 = series.rolling(window=window, min_periods=20).quantile(0.75)
                params[col] = {
                    "median": rolling_median.to_numpy(dtype=float, copy=False),
                    "iqr": (rolling_q3 - rolling_q1).to_numpy(dtype=float, copy=False),
                }
            else:  # pragma: no cover - guarded earlier
                raise RuntimeError(f"Unsupported normalization method: {method}")

        return params

    def _extract_window(self, end_idx: int) -> np.ndarray:
        idx = int(end_idx)
        start_idx = idx - self.lookback_window

        if start_idx < 0:
            if not self.config.allow_partial_windows:
                raise ValueError(
                    f"Index {end_idx} has insufficient history for lookback {self.lookback_window}"
                )
            start_idx = 0

        raw_window = self._feature_matrix[start_idx:idx]
        if raw_window.shape[0] < self.lookback_window:
            if not self.config.allow_partial_windows:
                raise ValueError(
                    f"Index {end_idx} yielded window of size {raw_window.shape[0]}, "
                    f"expected {self.lookback_window}"
                )
            pad_rows = self.lookback_window - raw_window.shape[0]
            padding = np.full((pad_rows, raw_window.shape[1]), self.config.pad_value, dtype=raw_window.dtype)
            raw_window = np.vstack([padding, raw_window])

        return np.array(raw_window, copy=True)

    def _normalize_window(self, window: np.ndarray, end_idx: int) -> np.ndarray:
        method = self.config.normalize_method
        if method == "zscore":
            return self._zscore_normalize(window, end_idx)
        if method == "minmax":
            return self._minmax_normalize(window, end_idx)
        if method == "robust":
            return self._robust_normalize(window, end_idx)
        return window

    def _zscore_normalize(self, window: np.ndarray, end_idx: int) -> np.ndarray:
        params = self._normalization_params
        if params is None:
            return window

        normalized = window.astype(float, copy=True)
        anchor = int(end_idx) - 1

        for i, col in enumerate(self.feature_cols):
            stats = params.get(col)
            if stats is None:
                continue
            mean_arr = stats["mean"]
            std_arr = stats["std"]
            mean = mean_arr[anchor] if 0 <= anchor < len(mean_arr) else np.nan
            std = std_arr[anchor] if 0 <= anchor < len(std_arr) else np.nan

            if not np.isfinite(std) or std < 1e-8:
                normalized[:, i] = 0.0
                continue
            if not np.isfinite(mean):
                mean = 0.0

            normalized[:, i] = (normalized[:, i] - mean) / std

        return normalized

    def _minmax_normalize(self, window: np.ndarray, end_idx: int) -> np.ndarray:
        params = self._normalization_params
        if params is None:
            return window

        normalized = window.astype(float, copy=True)
        anchor = int(end_idx) - 1

        for i, col in enumerate(self.feature_cols):
            stats = params.get(col)
            if stats is None:
                continue
            min_arr = stats["min"]
            max_arr = stats["max"]
            min_val = min_arr[anchor] if 0 <= anchor < len(min_arr) else np.nan
            max_val = max_arr[anchor] if 0 <= anchor < len(max_arr) else np.nan

            if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val - min_val < 1e-8:
                normalized[:, i] = 0.5
                continue

            normalized[:, i] = (normalized[:, i] - min_val) / (max_val - min_val)

        return normalized

    def _robust_normalize(self, window: np.ndarray, end_idx: int) -> np.ndarray:
        params = self._normalization_params
        if params is None:
            return window

        normalized = window.astype(float, copy=True)
        anchor = int(end_idx) - 1

        for i, col in enumerate(self.feature_cols):
            stats = params.get(col)
            if stats is None:
                continue
            median_arr = stats["median"]
            iqr_arr = stats["iqr"]
            median = median_arr[anchor] if 0 <= anchor < len(median_arr) else np.nan
            iqr = iqr_arr[anchor] if 0 <= anchor < len(iqr_arr) else np.nan

            if not np.isfinite(iqr) or iqr < 1e-8:
                normalized[:, i] = 0.0
                continue
            if not np.isfinite(median):
                median = 0.0

            normalized[:, i] = (normalized[:, i] - median) / iqr

        return normalized

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def compute_feature_statistics(self) -> pd.DataFrame:
        """Return descriptive statistics for the selected feature columns."""

        stats = self.data[self.feature_cols].describe().transpose()
        stats["missing_ratio"] = (
            self.data[self.feature_cols].isna().sum() / len(self.data)
        )
        return stats

    def iter_feature_windows(
        self,
        start_idx: int,
        end_idx: int,
        step: int,
        normalize: bool = True,
    ):
        """Yield successive windows between ``start_idx`` and ``end_idx``."""

        for idx in range(start_idx, end_idx, step):
            if not self.validate_index(idx):
                continue
            yield idx, self.extract_window(idx, normalize=normalize)

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        return (
            "FeatureExtractor(features=%s, lookback=%s, normalize=%s, cache_size=%s)"
            % (
                len(self.feature_cols),
                self.lookback_window,
                self.config.normalize_method,
                self.config.cache_size,
            )
        )
