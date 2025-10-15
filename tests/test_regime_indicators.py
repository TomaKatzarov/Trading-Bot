import numpy as np
import pandas as pd
import pytest

import core.rl.environments.regime_indicators as regime_module
from core.rl.environments.regime_indicators import (
    RegimeIndicators,
    _percentile_of_last,
    _trend_strength_stat,
)


@pytest.fixture(scope="module")
def sample_dataframe() -> pd.DataFrame:
    periods = 400
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    base = np.linspace(100.0, 150.0, periods)
    trend = np.linspace(0.0, 10.0, periods)
    osc = np.sin(np.linspace(0.0, 12.0, periods))

    data = {
        "open": base + 0.1,
        "high": base + 0.5,
        "low": base - 0.5,
        "close": base,
        "volume": 1_000_000 + np.arange(periods) * 10.0,
        "vwap": base + 0.25,
        "SMA_10": base + osc,
        "SMA_20": base - osc,
        "ATR_14": 2.5 + 0.1 * osc,
    }

    return pd.DataFrame(data, index=index)


def test_regime_vector_shape_and_range(sample_dataframe: pd.DataFrame) -> None:
    indicators = RegimeIndicators(sample_dataframe)
    idx = 180
    vector = indicators.get_regime_vector(idx)

    assert vector.shape == (10,)
    assert vector.dtype == np.float32
    assert np.all(vector >= 0.0) and np.all(vector <= 1.0)


def test_regime_names(sample_dataframe: pd.DataFrame) -> None:
    indicators = RegimeIndicators(sample_dataframe)
    names = indicators.get_regime_names()

    assert len(names) == 10
    assert names[0] == "realized_volatility"


def test_regime_dataframe_alignment(sample_dataframe: pd.DataFrame) -> None:
    indicators = RegimeIndicators(sample_dataframe)
    frame = indicators.get_regime_dataframe()

    assert list(frame.columns) == indicators.get_regime_names()
    assert len(frame) == len(sample_dataframe)
    assert np.all(frame.to_numpy() >= 0.0)
    assert np.all(frame.to_numpy() <= 1.0)


def test_out_of_range_index(sample_dataframe: pd.DataFrame) -> None:
    indicators = RegimeIndicators(sample_dataframe)
    with pytest.raises(IndexError):
        indicators.get_regime_vector(len(sample_dataframe))


def test_percentile_of_last_edge_cases() -> None:
    assert np.isnan(_percentile_of_last(np.array([])))
    all_nan = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(_percentile_of_last(all_nan))

    window = np.array([0.1, 0.2, 0.2, 0.3])
    percentile = _percentile_of_last(window)
    assert pytest.approx(percentile, rel=1e-6) == 1.0


def test_percentile_of_last_trailing_nan() -> None:
    """Test that _percentile_of_last returns NaN when the last value is NaN.
    
    Note: We test this directly rather than monkeypatching np.isnan, as
    monkeypatching numpy functions breaks Numba's JIT compilation.
    """
    # Test with trailing NaN
    window = np.array([0.1, np.nan])
    assert np.isnan(_percentile_of_last(window))
    
    # Test with only NaN values
    window_all_nan = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(_percentile_of_last(window_all_nan))
    
    # Test with NaN in middle but valid last value
    window_valid_last = np.array([0.1, np.nan, 0.5])
    result = _percentile_of_last(window_valid_last)
    assert not np.isnan(result)
    # Last value 0.5 is greater than 0.1, so percentile should be 1.0
    assert result == pytest.approx(1.0)


def test_trend_strength_stat_edge_cases() -> None:
    assert np.isnan(_trend_strength_stat(np.array([])))
    assert np.isnan(_trend_strength_stat(np.array([np.nan, np.nan])))

    flat_series = np.array([1.0, 1.0, 1.0])
    assert _trend_strength_stat(flat_series) == pytest.approx(1.0)


def test_dataframe_validation_errors() -> None:
    with pytest.raises(TypeError):
        RegimeIndicators(data=[1, 2, 3])  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        RegimeIndicators(pd.DataFrame())

    with pytest.raises(ValueError):
        RegimeIndicators(pd.DataFrame({"open": [1.0]}))


def test_invalid_constructor_parameters(sample_dataframe: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        RegimeIndicators(sample_dataframe, volatility_scale=0.0)

    with pytest.raises(ValueError):
        RegimeIndicators(sample_dataframe, volume_avg_window=0)

    with pytest.raises(ValueError):
        RegimeIndicators(sample_dataframe, atr_period=0)


def test_volume_regime_defaults_to_neutral() -> None:
    periods = 120
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = pd.Series(100.0 + np.linspace(0.0, 1.0, periods), index=index)
    df = pd.DataFrame({"close": close})

    indicators = RegimeIndicators(df)
    volume_regime = indicators.volume_regime

    assert np.allclose(volume_regime.to_numpy(), indicators.neutral_value)


def test_atr_fallback_when_high_low_missing() -> None:
    periods = 120
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = pd.Series(150.0 + np.sin(np.linspace(0.0, 4.0, periods)), index=index)
    df = pd.DataFrame({"close": close})

    indicators = RegimeIndicators(df)
    atr_series = indicators._compute_atr(indicators.atr_period)

    assert np.allclose(atr_series.to_numpy(), indicators.neutral_value)


def test_volatility_trend_neutral_for_small_changes() -> None:
    periods = 120
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = pd.Series(200.0 + 1e-4 * np.sin(np.linspace(0.0, 2.0, periods)), index=index)
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 0.1,
            "low": close - 0.1,
        }
    )

    indicators = RegimeIndicators(df)
    vol_trend = np.asarray(indicators.vol_trend)
    valid = vol_trend[~np.isnan(vol_trend)]

    assert np.allclose(valid, indicators.neutral_value)


def test_iter_vectors_returns_expected_sequences(sample_dataframe: pd.DataFrame) -> None:
    indicators = RegimeIndicators(sample_dataframe)
    indices = [10, 20, 30]
    vectors = list(indicators.iter_vectors(indices))

    assert len(vectors) == len(indices)
    for vector in vectors:
        assert vector.shape == (10,)
        assert np.all((0.0 <= vector) & (vector <= 1.0))


def test_dataframe_with_unsorted_index_sorted_internally(sample_dataframe: pd.DataFrame) -> None:
    reversed_df = sample_dataframe.iloc[::-1]
    indicators = RegimeIndicators(reversed_df)

    assert indicators.data.index.is_monotonic_increasing
