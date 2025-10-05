"""Test suite for FeatureExtractor and RegimeIndicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.rl.environments.feature_extractor import FeatureConfig, FeatureExtractor
from core.rl.environments.regime_indicators import RegimeIndicators


@pytest.fixture(scope="module")
def sample_data() -> pd.DataFrame:
    """Create reproducible synthetic market data for testing."""

    n_rows = 500
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="h")

    price = 100.0 + np.cumsum(rng.normal(loc=0.0, scale=0.5, size=n_rows))
    noise = rng.normal(loc=0.0, scale=0.1, size=n_rows)
    high_spread = np.abs(rng.normal(loc=0.0, scale=0.5, size=n_rows))
    low_spread = np.abs(rng.normal(loc=0.0, scale=0.5, size=n_rows))

    volume = rng.integers(low=1_000, high=10_000, size=n_rows)
    sentiment = rng.uniform(low=0.4, high=0.6, size=n_rows)

    data = pd.DataFrame(
        {
            "open": price + noise,
            "high": price + high_spread,
            "low": price - low_spread,
            "close": price,
            "volume": volume,
            "vwap": price + rng.normal(loc=0.0, scale=0.05, size=n_rows),
            "SMA_10": price,
            "SMA_20": price,
            "MACD": rng.normal(loc=0.0, scale=0.5, size=n_rows),
            "MACD_signal": rng.normal(loc=0.0, scale=0.3, size=n_rows),
            "MACD_hist": rng.normal(loc=0.0, scale=0.2, size=n_rows),
            "RSI_14": rng.uniform(low=30.0, high=70.0, size=n_rows),
            "Stochastic_K": rng.uniform(low=20.0, high=80.0, size=n_rows),
            "Stochastic_D": rng.uniform(low=20.0, high=80.0, size=n_rows),
            "ADX_14": rng.uniform(low=10.0, high=40.0, size=n_rows),
            "ATR_14": rng.uniform(low=0.5, high=2.0, size=n_rows),
            "BB_bandwidth": rng.uniform(low=0.01, high=0.05, size=n_rows),
            "OBV": np.cumsum(rng.integers(low=-1_000, high=1_000, size=n_rows)),
            "Volume_SMA_20": rng.integers(low=1_000, high=10_000, size=n_rows),
            "Return_1h": rng.normal(loc=0.0, scale=0.01, size=n_rows),
            "sentiment_score_hourly_ffill": sentiment,
            "DayOfWeek_sin": np.sin(np.arange(n_rows) * 2 * np.pi / 7),
            "DayOfWeek_cos": np.cos(np.arange(n_rows) * 2 * np.pi / 7),
        },
        index=dates,
    )

    return data


class TestFeatureExtractor:
    """Tests covering FeatureExtractor behaviour."""

    def test_initialization(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="zscore")
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        assert extractor.get_feature_count() == 23
        assert len(extractor.get_feature_names()) == 23

    def test_window_extraction(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="none")
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        window = extractor.extract_window(100)

        assert window.shape == (24, 23)
        assert window.dtype == np.float32

    def test_extract_batch(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="none")
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        batch = extractor.extract_batch([110, 115, 120])
        empty = extractor.extract_batch([])

        assert batch.shape == (3, 24, 23)
        assert empty.shape == (0, 24, 23)

    def test_normalization_zscore(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="zscore", normalize_window=50)
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        window = extractor.extract_window(100, normalize=True)

        assert window.shape == (24, 23)
        assert np.all(np.abs(window) <= 10.0)

    def test_normalization_minmax(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="minmax", normalize_window=50)
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        window = extractor.extract_window(120, normalize=True)

        assert window.shape == (24, 23)
        assert np.all((0.0 <= window) & (window <= 1.0))

    def test_normalization_robust(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="robust", normalize_window=60)
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        window = extractor.extract_window(150, normalize=True)

        assert window.shape == (24, 23)
        assert np.all(np.abs(window) <= 10.0)

    def test_caching(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="none")
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        extractor.extract_window(200)
        stats_first = extractor.get_cache_stats()

        extractor.extract_window(200)
        stats_second = extractor.get_cache_stats()

        assert stats_second["hits"] > stats_first["hits"]

    def test_feature_subset(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(
            normalize_method="none",
            feature_subset=("close", "volume", "RSI_14"),
        )
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        assert extractor.get_feature_count() == 3
        assert set(extractor.get_feature_names()) == {"close", "volume", "RSI_14"}

    def test_valid_range(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig()
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        min_idx, max_idx = extractor.get_valid_range()

        assert min_idx == 24
        assert max_idx == len(sample_data)
        assert extractor.validate_index(24)
        assert not extractor.validate_index(23)

    def test_sl_predictions_fallback(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="zscore")
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        preds = extractor.get_sl_predictions(180)

        assert preds.shape == (3,)
        assert np.allclose(preds, 0.5)


class TestRegimeIndicators:
    """Tests covering RegimeIndicators behaviour."""

    def test_initialization(self, sample_data: pd.DataFrame) -> None:
        regime = RegimeIndicators(sample_data)

        names = regime.get_regime_names()
        assert len(names) == 10

    def test_regime_vector_shape(self, sample_data: pd.DataFrame) -> None:
        regime = RegimeIndicators(sample_data)
        vector = regime.get_regime_vector(100)

        assert vector.shape == (10,)
        assert vector.dtype == np.float32

    def test_regime_values_range(self, sample_data: pd.DataFrame) -> None:
        regime = RegimeIndicators(sample_data)
        vector = regime.get_regime_vector(150)

        assert np.all(vector >= 0.0)
        assert np.all(vector <= 1.0)

    def test_early_index_handling(self, sample_data: pd.DataFrame) -> None:
        regime = RegimeIndicators(sample_data)
        vector = regime.get_regime_vector(10)

        assert vector.shape == (10,)
        assert not np.any(np.isnan(vector))

    def test_iter_vectors(self, sample_data: pd.DataFrame) -> None:
        regime = RegimeIndicators(sample_data)
        indices = [50, 75, 100]
        vectors = list(regime.iter_vectors(indices))

        assert len(vectors) == len(indices)
        for vec in vectors:
            assert vec.shape == (10,)
            assert np.all((0.0 <= vec) & (vec <= 1.0))


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    pytest.main([__file__, "-v"])  # type: ignore[arg-type]
