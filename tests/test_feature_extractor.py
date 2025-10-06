"""Test suite for FeatureExtractor and RegimeIndicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
import types

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

    def test_partial_window_padding(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(
            normalize_method="none",
            allow_partial_windows=True,
            pad_value=-1.0,
        )
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        window = extractor.extract_window(5, normalize=False)
        pad_rows = extractor.lookback_window - 5

        assert window.shape == (extractor.lookback_window, extractor.get_feature_count())
        assert np.allclose(window[:pad_rows], -1.0)
        assert np.any(window[pad_rows:] != -1.0)

    def test_missing_columns_strict_raises(self, sample_data: pd.DataFrame) -> None:
        data = sample_data.drop(columns=["sentiment_score_hourly_ffill"])
        config = FeatureConfig(normalize_method="none")

        with pytest.raises(ValueError):
            FeatureExtractor(data, config, lookback_window=24)

    def test_missing_columns_non_strict(self, sample_data: pd.DataFrame) -> None:
        data = sample_data.drop(columns=["sentiment_score_hourly_ffill"])
        config = FeatureConfig(normalize_method="none", strict_feature_check=False)

        extractor = FeatureExtractor(data, config, lookback_window=24)

        assert "sentiment_score_hourly_ffill" not in extractor.get_feature_names()

    def test_minmax_normalization_fallback(self, sample_data: pd.DataFrame) -> None:
        patched = sample_data.copy()
        patched["close"] = 1.0
        config = FeatureConfig(normalize_method="minmax", normalize_window=30)
        extractor = FeatureExtractor(patched, config, lookback_window=24)

        window = extractor.extract_window(100, normalize=True)
        idx = extractor.get_feature_names().index("close")

        assert np.allclose(window[:, idx], 0.5)

    def test_zscore_normalization_handles_zero_std(self, sample_data: pd.DataFrame) -> None:
        patched = sample_data.copy()
        patched["close"] = 7.0
        config = FeatureConfig(normalize_method="zscore", normalize_window=30)
        extractor = FeatureExtractor(patched, config, lookback_window=24)

        window = extractor.extract_window(90, normalize=True)
        idx = extractor.get_feature_names().index("close")

        assert np.allclose(window[:, idx], 0.0)

    def test_robust_normalization_handles_zero_iqr(self, sample_data: pd.DataFrame) -> None:
        patched = sample_data.copy()
        patched["close"] = -3.0
        config = FeatureConfig(normalize_method="robust", normalize_window=30)
        extractor = FeatureExtractor(patched, config, lookback_window=24)

        window = extractor.extract_window(110, normalize=True)
        idx = extractor.get_feature_names().index("close")

        assert np.allclose(window[:, idx], 0.0)

    def test_reconfigure_and_update_data(self, sample_data: pd.DataFrame) -> None:
        base_config = FeatureConfig(normalize_method="none")
        extractor = FeatureExtractor(sample_data, base_config, lookback_window=24)

        trimmed = sample_data.iloc[::2].copy()
        extractor.update_data(trimmed)
        assert extractor.get_valid_range()[1] == len(trimmed)

        new_config = FeatureConfig(normalize_method="minmax", normalize_window=25)
        extractor.reconfigure(new_config)

        window = extractor.extract_window(extractor.get_valid_range()[0] + 1, normalize=True)

        assert np.isfinite(window).all()
        assert np.all((0.0 <= window) & (window <= 1.0))

    def test_cache_size_validation(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="none", cache_size=0)

        with pytest.raises(ValueError):
            FeatureExtractor(sample_data, config, lookback_window=24)

    def test_feature_config_as_dict(self) -> None:
        config = FeatureConfig(normalize_method="minmax", normalize_window=50, feature_subset=("close",))

        config_dict = config.as_dict()

        assert config_dict["normalize_method"] == "minmax"
        assert config_dict["normalize_window"] == 50
        assert tuple(config_dict["feature_subset"]) == ("close",)

    def test_validate_index_bounds(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig()
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        assert extractor.validate_index(24)
        assert not extractor.validate_index(len(sample_data) + 1)
        assert not extractor.validate_index(-1)

    def test_prefetch_and_clear_cache(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="none")
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        extractor.prefetch(range(50, 55))
        stats_after_prefetch = extractor.get_cache_stats()
        assert stats_after_prefetch["misses"] >= 1

        extractor.extract_window(54)
        stats_after_hit = extractor.get_cache_stats()
        assert stats_after_hit["hits"] >= 1

        extractor.clear_cache()
        cleared_stats = extractor.get_cache_stats()
        assert cleared_stats["total"] == 0

    def test_get_sl_predictions_with_stubbed_inference(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="zscore")

        module_name = "scripts.sl_checkpoint_utils"
        fake_module = types.ModuleType(module_name)

        def run_inference(bundle, input_tensor):
            if bundle == "mlp":
                return {"probability": 0.91}
            if bundle == "lstm":
                return [0.27]
            if bundle == "gru":
                return 0.13
            raise RuntimeError("unexpected bundle")

        fake_module.run_inference = run_inference  # type: ignore[attr-defined]
        sys.modules[module_name] = fake_module

        try:
            extractor = FeatureExtractor(
                sample_data,
                config,
                lookback_window=24,
                sl_models={"mlp": "mlp", "lstm": "lstm", "gru": "gru"},
            )

            window = extractor.extract_window(200)
            probs = extractor.get_sl_predictions(200, window=window)

            assert np.allclose(probs, [0.91, 0.27, 0.13])
        finally:
            sys.modules.pop(module_name, None)

    def test_prepare_dataframe_validations(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="none")

        with pytest.raises(TypeError):
            FeatureExtractor(123, config, lookback_window=24)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            FeatureExtractor(pd.DataFrame(), config, lookback_window=24)

        shuffled = sample_data.sample(frac=1.0, random_state=0)
        extractor = FeatureExtractor(shuffled, config, lookback_window=24)
        assert extractor.data.index.is_monotonic_increasing

    def test_feature_subset_invalid_request(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(
            normalize_method="none",
            feature_subset=("nonexistent",),
        )

        with pytest.raises(ValueError):
            FeatureExtractor(sample_data, config, lookback_window=24)

    def test_compute_feature_statistics(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="none")
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        stats = extractor.compute_feature_statistics()

        assert "missing_ratio" in stats.columns
        assert not stats.isna().any().any()

    def test_iter_feature_windows(self, sample_data: pd.DataFrame) -> None:
        config = FeatureConfig(normalize_method="none")
        extractor = FeatureExtractor(sample_data, config, lookback_window=24)

        windows = list(extractor.iter_feature_windows(20, 80, 10, normalize=False))

        indices = [idx for idx, _ in windows]
        assert all(extractor.validate_index(idx) for idx in indices)
        assert indices[0] >= extractor.get_valid_range()[0]

    def test_reconfigure_invalid_method(self, sample_data: pd.DataFrame) -> None:
        extractor = FeatureExtractor(sample_data, FeatureConfig(), lookback_window=24)
        bad_config = FeatureConfig(normalize_method="bogus")

        with pytest.raises(ValueError):
            extractor.reconfigure(bad_config)


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
