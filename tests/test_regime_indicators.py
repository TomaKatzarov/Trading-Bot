import numpy as np
import pandas as pd
import pytest

from core.rl.environments import RegimeIndicators


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
