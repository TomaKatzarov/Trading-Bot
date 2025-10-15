#!/usr/bin/env python3
"""Phase 3 data preparation pipeline for RL prototype training.

Creates clean train/validation/test splits for the 10-symbol Phase 3 portfolio,
fits per-symbol feature scalers on training data, and attaches supervised
learning baseline probabilities for downstream comparison.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sl_checkpoint_utils import (  # noqa: E402
    DatasetInfo,
    align_config_with_state_dict,
    instantiate_model_from_checkpoint,
    list_checkpoint_artifacts,
    load_checkpoint_bundle,
    sanitize_model_config,
)


LOGGER = logging.getLogger("phase3_data_preparation")

PHASE3_SYMBOLS = (
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "XOM",
)

LOOKBACK_WINDOW = 24
TARGET_START = pd.Timestamp("2023-10-01 00:00:00", tz="UTC")
TARGET_END = pd.Timestamp("2025-10-31 23:00:00", tz="UTC")
TIMEFRAME = "1Hour"

# Feature definitions (matches supervised-learning dataset)
FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "SMA_10",
    "SMA_20",
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
    "RSI_14",
    "Stoch_K",
    "Stoch_D",
    "ADX_14",
    "ATR_14",
    "BB_bandwidth",
    "OBV",
    "Volume_SMA_20",
    "1h_return",
    "sentiment_score_hourly_ffill",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
]

SCALE_FEATURES = FEATURE_COLUMNS

SL_MODEL_DIRECTORIES = {
    "mlp": "mlp_trial72_epoch3",
    "lstm": "lstm_trial62_epoch1",
    "gru": "gru_trial93_epoch4",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


@dataclass
class SymbolMetadata:
    symbol: str
    total_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_period: Tuple[str, str]
    val_period: Tuple[str, str]
    test_period: Tuple[str, str]
    trading_hours: List[int]
    scaler_path: str


def configure_logging(verbose: bool = False) -> None:
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/historical"))
    parser.add_argument("--output-root", type=Path, default=Path("data/phase3_splits"))
    parser.add_argument(
        "--asset-map",
        type=Path,
        default=Path("data/training_data_v2_final/asset_id_mapping.json"),
    )
    parser.add_argument(
        "--global-scaler",
        type=Path,
        default=Path("data/training_data_v2_final/scalers.joblib"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("models/sl_checkpoints"),
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Optional subset of symbols to process (defaults to Phase 3 portfolio)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_symbol_frame(symbol: str, data_root: Path) -> pd.DataFrame:
    file_path = data_root / symbol / TIMEFRAME / "data.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found for {symbol}: {file_path}")

    df = pd.read_parquet(file_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column for {symbol}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[(df.index >= TARGET_START) & (df.index <= TARGET_END)]
    return df


def infer_trading_hours(index: pd.DatetimeIndex) -> List[int]:
    weekday_index = index[index.weekday < 5]
    if weekday_index.empty:
        return []
    return sorted(int(hour) for hour in np.unique(weekday_index.hour))


def build_expected_index(trading_hours: Iterable[int]) -> pd.DatetimeIndex:
    if not trading_hours:
        return pd.DatetimeIndex([], tz="UTC")
    business_days = pd.bdate_range(TARGET_START.normalize(), TARGET_END.normalize(), tz="UTC")
    instants: List[pd.Timestamp] = []
    for day in business_days:
        for hour in trading_hours:
            instants.append(day + pd.Timedelta(hours=int(hour)))
    expected = pd.DatetimeIndex(instants, tz="UTC")
    expected = expected[(expected >= TARGET_START) & (expected <= TARGET_END)]
    return expected


def forward_fill_columns(df: pd.DataFrame, missing_index: pd.DatetimeIndex) -> None:
    if missing_index.empty:
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    # explicit handling for volume/returns on synthesized rows
    for col in ("volume", "Volume"):
        if col in df.columns:
            df.loc[missing_index, col] = 0.0
    if "Volume_SMA_20" in df.columns:
        df.loc[missing_index, "Volume_SMA_20"] = 0.0
    if "Return_1h" in df.columns:
        df["Return_1h"] = df["Return_1h"].fillna(0.0)


def ensure_lowercase_ohlcv(df: pd.DataFrame) -> None:
    mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "VWAP": "vwap",
    }
    for source, target in mapping.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    ensure_lowercase_ohlcv(df)
    if "vwap" not in df.columns and {"high", "low", "close"}.issubset(df.columns):
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    if "close" in df.columns:
        df["1h_return"] = df["close"].pct_change().fillna(0.0)
        df["Return_1h"] = df["1h_return"]
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        df["HL_diff"] = df["high"] - df["low"]
        df["OHLC_avg"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    if "close" in df.columns:
        df["SMA_10"] = df["close"].rolling(window=10, min_periods=1).mean()
        df["SMA_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    if "close" in df.columns:
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        df["MACD_line"] = macd_line
        df["MACD_signal"] = macd_signal
        df["MACD_hist"] = macd_line - macd_signal

    if "DayOfWeek_sin" not in df.columns or "DayOfWeek_cos" not in df.columns:
        day = df.index.dayofweek
        df["DayOfWeek_sin"] = np.sin(2 * np.pi * day / 7.0)
        df["DayOfWeek_cos"] = np.cos(2 * np.pi * day / 7.0)

    if "sentiment_score_hourly_ffill" in df.columns:
        sentiment_series = df["sentiment_score_hourly_ffill"].copy()
    else:
        sentiment_series = pd.Series(0.5, index=df.index, dtype=float)
    df["sentiment_score_hourly_ffill"] = sentiment_series.ffill().fillna(0.5)
    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURE_COLUMNS)
    available = set(df.columns)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    df = df.dropna(subset=FEATURE_COLUMNS)
    return df


def compute_split_boundaries(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    total_span = TARGET_END - TARGET_START
    train_end = TARGET_START + pd.to_timedelta(total_span.total_seconds() * TRAIN_RATIO, unit="s")
    val_end = train_end + pd.to_timedelta(total_span.total_seconds() * VAL_RATIO, unit="s")
    return train_end, val_end


def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_cutoff, val_cutoff = compute_split_boundaries(df)
    train_df = df[df.index <= train_cutoff].copy()
    val_df = df[(df.index > train_cutoff) & (df.index <= val_cutoff)].copy()
    test_df = df[df.index > val_cutoff].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more splits are empty; review date coverage")
    return train_df, val_df, test_df


def apply_scaler(
    scaler: StandardScaler,
    df: pd.DataFrame,
    features: List[str],
) -> None:
    scaled_values = scaler.transform(df[features])
    df.loc[:, features] = scaled_values


def sliding_windows(values: np.ndarray, lookback: int) -> np.ndarray:
    num_samples = values.shape[0] - lookback + 1
    if num_samples <= 0:
        return np.empty((0, lookback, values.shape[1]), dtype=np.float32)
    try:
        windows = np.lib.stride_tricks.sliding_window_view(values, (lookback, values.shape[1]))
        return windows.reshape(num_samples, lookback, values.shape[1]).astype(np.float32)
    except AttributeError:  # fallback for older numpy
        buffer = np.empty((num_samples, lookback, values.shape[1]), dtype=np.float32)
        for idx in range(num_samples):
            buffer[idx] = values[idx : idx + lookback]
        return buffer


class Phase3Preparer:
    def __init__(
        self,
        data_root: Path,
        output_root: Path,
        asset_map_path: Path,
        global_scaler_path: Path,
        checkpoint_root: Path,
    ) -> None:
        self.data_root = data_root
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

        with asset_map_path.open("r", encoding="utf-8") as f:
            self.asset_mapping: Mapping[str, int] = json.load(f)
        if not self.asset_mapping:
            raise ValueError("Asset ID mapping is empty")

        scalers_dict = joblib.load(global_scaler_path)
        if isinstance(scalers_dict, dict):
            self.global_scaler: StandardScaler = scalers_dict.get("global")
        else:
            self.global_scaler = scalers_dict
        if self.global_scaler is None:
            raise ValueError("Global scaler not found in joblib file")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sl_models = self._load_sl_models(checkpoint_root)
        LOGGER.info("Using device %s for SL inference", self.device)

    def _load_sl_models(self, checkpoint_root: Path) -> Dict[str, torch.nn.Module]:
        artifacts = {
            artifact.checkpoint_path.parent.name: artifact
            for artifact in list_checkpoint_artifacts(checkpoint_root)
        }
        dataset_info = DatasetInfo(
            n_features=len(FEATURE_COLUMNS),
            lookback_window=LOOKBACK_WINDOW,
            num_assets_dataset=len(self.asset_mapping),
            num_assets_mapping=len(self.asset_mapping),
        )
        models: Dict[str, torch.nn.Module] = {}
        for alias, directory in SL_MODEL_DIRECTORIES.items():
            artifact = artifacts.get(directory)
            if artifact is None:
                raise FileNotFoundError(f"Checkpoint directory '{directory}' not found under {checkpoint_root}")
            bundle = load_checkpoint_bundle(artifact)
            model_type = bundle["model_type"]
            sanitized_config = sanitize_model_config(model_type, bundle["raw_config"], dataset_info)
            sanitized_config = align_config_with_state_dict(model_type, sanitized_config, bundle["state_dict"])
            model = instantiate_model_from_checkpoint(model_type, sanitized_config, bundle["state_dict"])
            model.eval()
            model.to(self.device)
            models[alias] = model
            LOGGER.info("Loaded %s model '%s'", alias.upper(), directory)
        return models

    def _run_sl_inference(self, features: np.ndarray, asset_id: int) -> Dict[str, np.ndarray]:
        if features.shape[0] < LOOKBACK_WINDOW:
            return {alias: np.empty(0, dtype=np.float32) for alias in self.sl_models}

        scaled = self.global_scaler.transform(features)
        windows = sliding_windows(scaled.astype(np.float32), LOOKBACK_WINDOW)
        asset_ids = np.full((windows.shape[0],), asset_id, dtype=np.int64)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(windows),
            torch.from_numpy(asset_ids),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)

        probabilities: Dict[str, np.ndarray] = {}
        for alias, model in self.sl_models.items():
            outputs: List[np.ndarray] = []
            with torch.no_grad():
                for batch_features, batch_asset_ids in loader:
                    batch_features = batch_features.to(self.device)
                    batch_asset_ids = batch_asset_ids.to(self.device)
                    logits = model(batch_features, batch_asset_ids).squeeze(-1)
                    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                    outputs.append(probs)
            if outputs:
                probabilities[alias] = np.concatenate(outputs, axis=0)
            else:
                probabilities[alias] = np.empty(0, dtype=np.float32)
        return probabilities

    def _attach_sl_probs(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        asset_id = self.asset_mapping.get(symbol)
        if asset_id is None:
            raise KeyError(f"Symbol {symbol} missing from asset-id mapping")

        feature_matrix = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        probabilities = self._run_sl_inference(feature_matrix, asset_id)
        for alias, values in probabilities.items():
            series = pd.Series(data=np.nan, index=df.index, dtype=np.float32)
            if values.size:
                series.iloc[LOOKBACK_WINDOW - 1 :] = values
            df[f"sl_prob_{alias}"] = series.ffill().bfill().clip(0.0, 1.0)
        return df.iloc[LOOKBACK_WINDOW - 1 :].copy()

    def prepare_symbol(self, symbol: str) -> SymbolMetadata:
        LOGGER.info("Processing %s", symbol)
        df_raw = load_symbol_frame(symbol, self.data_root)
        trading_hours = infer_trading_hours(df_raw.index)
        expected_index = build_expected_index(trading_hours)
        df = df_raw.reindex(expected_index)
        missing_index = expected_index.difference(df_raw.index)
        forward_fill_columns(df, missing_index)
        df = enrich_features(df)
        df = drop_incomplete_rows(df)
        df["symbol"] = symbol

        df = self._attach_sl_probs(df, symbol)

        train_df, val_df, test_df = split_dataframe(df)

        scaler = StandardScaler()
        scaler.fit(train_df[SCALE_FEATURES])

        apply_scaler(scaler, train_df, SCALE_FEATURES)
        apply_scaler(scaler, val_df, SCALE_FEATURES)
        apply_scaler(scaler, test_df, SCALE_FEATURES)

        symbol_dir = self.output_root / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(symbol_dir / "train.parquet")
        val_df.to_parquet(symbol_dir / "val.parquet")
        test_df.to_parquet(symbol_dir / "test.parquet")
        scaler_path = symbol_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)

        metadata = {
            "symbol": symbol,
            "total_rows": int(df.shape[0]),
            "train_rows": int(train_df.shape[0]),
            "val_rows": int(val_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
            "train_period": {
                "start": train_df.index.min().isoformat(),
                "end": train_df.index.max().isoformat(),
            },
            "val_period": {
                "start": val_df.index.min().isoformat(),
                "end": val_df.index.max().isoformat(),
            },
            "test_period": {
                "start": test_df.index.min().isoformat(),
                "end": test_df.index.max().isoformat(),
            },
            "trading_hours": trading_hours,
            "scaled_features": SCALE_FEATURES,
            "sl_prob_columns": [f"sl_prob_{alias}" for alias in SL_MODEL_DIRECTORIES],
        }
        (symbol_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info(
            "%s splits saved (train=%d, val=%d, test=%d)",
            symbol,
            train_df.shape[0],
            val_df.shape[0],
            test_df.shape[0],
        )
        return SymbolMetadata(
            symbol=symbol,
            total_rows=int(df.shape[0]),
            train_rows=int(train_df.shape[0]),
            val_rows=int(val_df.shape[0]),
            test_rows=int(test_df.shape[0]),
            train_period=(metadata["train_period"]["start"], metadata["train_period"]["end"]),
            val_period=(metadata["val_period"]["start"], metadata["val_period"]["end"]),
            test_period=(metadata["test_period"]["start"], metadata["test_period"]["end"]),
            trading_hours=trading_hours,
            scaler_path=str(scaler_path),
        )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    preparer = Phase3Preparer(
        data_root=args.data_root,
        output_root=args.output_root,
        asset_map_path=args.asset_map,
        global_scaler_path=args.global_scaler,
        checkpoint_root=args.checkpoint_root,
    )

    target_symbols: Iterable[str]
    if args.symbols:
        target_symbols = [sym.upper() for sym in args.symbols]
    else:
        target_symbols = PHASE3_SYMBOLS

    metadata_records: List[SymbolMetadata] = []
    for symbol in target_symbols:
        metadata_records.append(preparer.prepare_symbol(symbol))

    phase3_metadata = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "symbols": [metadata.__dict__ for metadata in metadata_records],
    }
    with (args.output_root / "phase3_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(phase3_metadata, f, indent=2)

    LOGGER.info("Phase 3 data preparation complete. Artifacts saved to %s", args.output_root)


if __name__ == "__main__":  # pragma: no cover
    main()
