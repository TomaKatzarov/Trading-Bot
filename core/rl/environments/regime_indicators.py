"""Market regime indicator computation utilities for the RL trading environment.

This module exposes :class:`RegimeIndicators`, a helper that pre-computes a
collection of normalized market state descriptors (volatility, trend, breadth,
and momentum) to augment the trading environment observations. All indicators are
scaled to the [0, 1] interval and cached internally for efficient repeated
lookups during RL rollouts.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

logger = logging.getLogger(__name__)

__all__ = ["RegimeIndicators"]

_NEUTRAL = 0.5


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _percentile_of_last(window: np.ndarray) -> float:
    """Return percentile rank of the last element within a rolling window.

    The function expects a 1-D floating array representing a chronological
    sequence. NaN values are ignored. If the final value is NaN or the window is
    empty after filtering, ``np.nan`` is returned.
    """

    if window.size == 0:
        return np.nan

    # Get the last element (most recent value)
    last_element = window[window.size - 1]
    
    # If the last element is NaN, return NaN
    if np.isnan(last_element):
        return np.nan

    # Count valid (non-NaN) values and compare with last_element
    valid_count = 0
    count_le = 0
    for i in range(window.size):
        if not np.isnan(window[i]):
            valid_count += 1
            if window[i] <= last_element:
                count_le += 1
    
    if valid_count == 0:
        return np.nan
    
    return float(count_le) / float(valid_count)


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _trend_strength_stat(window: np.ndarray) -> float:
    """Compute a normalized slope statistic for trend strength analysis."""

    if window.size == 0:
        return np.nan

    # Calculate mean and std for valid values
    valid_sum = 0.0
    valid_count = 0
    last = np.nan
    
    for i in range(window.size):
        if not np.isnan(window[i]):
            valid_sum += window[i]
            valid_count += 1
            last = window[i]
    
    if valid_count == 0 or np.isnan(last):
        return np.nan
    
    mean = valid_sum / valid_count
    
    # Calculate standard deviation
    var_sum = 0.0
    for i in range(window.size):
        if not np.isnan(window[i]):
            diff = window[i] - mean
            var_sum += diff * diff
    
    std = np.sqrt(var_sum / valid_count)
    
    if std < 1e-8:
        return min(max(last, -3.0), 3.0)
    
    value = last / (std + 1e-8)
    return min(max(value, -3.0), 3.0)


class RegimeIndicators:
    """Compute market regime indicators for a single symbol.

    Parameters
    ----------
    data:
        Pandas DataFrame containing OHLCV and supporting technical columns.
    volatility_scale:
        Scale factor used to normalize annualized realized volatility into the
        [0, 1] range. Defaults to 0.5 (roughly 50% annualized volatility).
    neutral_value:
        Fallback value used when insufficient history exists. Defaults to 0.5.
    atr_period:
        Period used when estimating ATR if the source dataframe does not supply
        ``ATR_{atr_period}``.
    volume_avg_window:
        Window for average volume comparisons. Defaults to 20.
    """

    _REGIME_NAMES: Sequence[str] = (
        "realized_volatility",
        "volatility_regime",
        "trend_strength",
        "trend_regime",
        "price_vs_sma50",
        "price_vs_sma200",
        "volume_regime",
        "atr_regime",
        "momentum_regime",
        "volatility_trend",
    )

    def __init__(
        self,
        data: pd.DataFrame,
        volatility_scale: float = 0.5,
        neutral_value: float = _NEUTRAL,
        atr_period: int = 14,
        volume_avg_window: int = 20,
    ) -> None:
        if volatility_scale <= 0:
            raise ValueError("volatility_scale must be positive")
        if volume_avg_window <= 0:
            raise ValueError("volume_avg_window must be positive")
        if atr_period <= 0:
            raise ValueError("atr_period must be positive")

        self.data = self._prepare_dataframe(data)
        self.volatility_scale = float(volatility_scale)
        self.neutral_value = float(neutral_value)
        self.atr_period = int(atr_period)
        self.volume_avg_window = int(volume_avg_window)

        self._regime_frame: Optional[pd.DataFrame] = None
        self._precompute_indicators()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_regime_vector(self, idx: int) -> np.ndarray:
        """Return the regime indicator vector at the provided positional index."""

        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for data length {len(self.data)}")

        frame = self._regime_frame
        assert frame is not None  # for type checkers

        vector = frame.iloc[idx].to_numpy(dtype=np.float32, copy=True)
        vector = np.clip(vector, 0.0, 1.0, out=vector)
        return vector

    def get_regime_dataframe(self) -> pd.DataFrame:
        """Return a copy of the pre-computed regime indicator dataframe."""

        frame = self._regime_frame
        assert frame is not None
        return frame.copy()

    def get_regime_names(self) -> List[str]:
        """Return the ordered list of regime indicator names."""

        return list(self._REGIME_NAMES)

    def iter_vectors(self, indices: Iterable[int]) -> Iterable[np.ndarray]:
        """Yield regime vectors for a collection of positional indices."""

        for idx in indices:
            yield self.get_regime_vector(idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("dataframe is empty; cannot compute regime indicators")

        df = data.copy()
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)

        required_columns = {"close"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"RegimeIndicators requires columns {missing}")

        return df

    def _precompute_indicators(self) -> None:
        logger.info("Precomputing regime indicators…")

        close = self.data["close"].astype(float)
        volume = self.data.get("volume")

        # Use numba engine for rolling operations if available
        engine = "numba" if HAS_NUMBA else None
        engine_kwargs = {"nopython": True, "nogil": True, "parallel": False} if HAS_NUMBA else {}

        returns = close.pct_change()
        realized_vol = returns.rolling(window=20, min_periods=20).std(ddof=0) * math.sqrt(252.0)
        realized_vol_norm = (realized_vol / self.volatility_scale).clip(0.0, 1.0)

        vol_regime = self._rolling_percentile(realized_vol, window=252, min_periods=60, engine=engine, engine_kwargs=engine_kwargs)

        sma_20 = self.data.get("SMA_20")
        if sma_20 is None:
            sma_20 = close.rolling(window=20, min_periods=20).mean()

        sma_50 = self.data.get("SMA_50")
        if sma_50 is None:
            sma_50 = self.data.get("SMA_20", sma_20)
        sma_200 = self.data.get("SMA_200")
        if sma_200 is None:
            sma_200 = self.data.get("SMA_20", sma_20)

        sma_slope = (sma_20 - sma_20.shift(20)) / (sma_20.shift(20) + 1e-8)
        trend_strength_raw = sma_slope.rolling(window=60, min_periods=30).apply(
            _trend_strength_stat, raw=True, engine=engine, engine_kwargs=engine_kwargs
        )
        trend_strength = (trend_strength_raw / 3.0).clip(-1.0, 1.0)
        trend_strength_unit = (trend_strength * 0.5) + 0.5

        trend_regime = np.full_like(sma_slope, self.neutral_value, dtype=float)
        trend_regime[sma_slope > 0.005] = 1.0
        trend_regime[sma_slope < -0.005] = 0.0

        price_vs_sma50 = ((close - sma_50) / (sma_50 + 1e-8)).clip(-0.5, 0.5)
        price_vs_sma50_unit = price_vs_sma50 / 0.5 * 0.5 + 0.5

        price_vs_sma200 = ((close - sma_200) / (sma_200 + 1e-8)).clip(-0.5, 0.5)
        price_vs_sma200_unit = price_vs_sma200 / 0.5 * 0.5 + 0.5

        if volume is not None:
            vol_ma = volume.rolling(window=self.volume_avg_window, min_periods=5).mean()
            volume_regime = (volume / (vol_ma + 1e-8)).clip(0.0, 3.0) / 3.0
        else:
            volume_regime = pd.Series(self.neutral_value, index=self.data.index)

        atr_series = self.data.get(f"ATR_{self.atr_period}")
        if atr_series is None:
            atr_series = self._compute_atr(self.atr_period)

        atr_regime = self._rolling_percentile(atr_series, window=252, min_periods=60, engine=engine, engine_kwargs=engine_kwargs)

        roc = close.pct_change(20)
        momentum_regime = self._rolling_percentile(roc, window=60, min_periods=30, engine=engine, engine_kwargs=engine_kwargs)

        vol_change = realized_vol.diff(20)
        vol_trend = np.full_like(vol_change, self.neutral_value, dtype=float)
        vol_trend[vol_change > 0] = 1.0
        vol_trend[vol_change < 0] = 0.0
        small_change = vol_change.abs() < 0.01
        vol_trend[small_change.fillna(False)] = self.neutral_value

        regime_frame = pd.DataFrame(
            {
                "realized_volatility": realized_vol_norm,
                "volatility_regime": vol_regime,
                "trend_strength": trend_strength_unit,
                "trend_regime": trend_regime,
                "price_vs_sma50": price_vs_sma50_unit,
                "price_vs_sma200": price_vs_sma200_unit,
                "volume_regime": volume_regime,
                "atr_regime": atr_regime,
                "momentum_regime": momentum_regime,
                "volatility_trend": vol_trend,
            },
            index=self.data.index,
        )

        regime_frame = regime_frame.apply(lambda col: col.fillna(self.neutral_value))
        regime_frame = regime_frame.clip(0.0, 1.0)

        self.realized_vol = realized_vol  # type: ignore[attr-defined]
        self.vol_regime = vol_regime
        self.trend_strength = trend_strength_unit
        self.trend_regime = trend_regime
        self.price_vs_sma50 = price_vs_sma50_unit
        self.price_vs_sma200 = price_vs_sma200_unit
        self.volume_regime = volume_regime
        self.atr_regime = atr_regime
        self.momentum_regime = momentum_regime
        self.vol_trend = vol_trend

        self._regime_frame = regime_frame

        logger.info("Regime indicators precomputed")

    def _rolling_percentile(
        self,
        series: pd.Series,
        window: int,
        min_periods: int,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict] = None,
    ) -> pd.Series:
        if engine_kwargs is None:
            engine_kwargs = {}
        return series.rolling(window=window, min_periods=min_periods).apply(
            _percentile_of_last, raw=True, engine=engine, engine_kwargs=engine_kwargs
        ).fillna(self.neutral_value)

    def _compute_atr(self, period: int) -> pd.Series:
        high = self.data.get("high")
        low = self.data.get("low")
        close = self.data["close"]

        if high is None or low is None:
            logger.warning("Missing high/low columns; ATR fallback defaults to neutral value")
            return pd.Series(self.neutral_value, index=self.data.index)

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.rolling(window=period, min_periods=period).mean()
        return atr.fillna(atr.mean()).fillna(self.neutral_value)