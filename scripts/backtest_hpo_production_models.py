#!/usr/bin/env python3
"""Comprehensive backtesting campaign for Phase 3 HPO production models.

This script evaluates the three Phase 3 production checkpoints (MLP Trial 72,
LSTM Trial 62, GRU Trial 93) alongside ensemble variants to validate trading
profitability over the full two-year evaluation window. It reproduces the
risk-management and label logic used during training, applies transaction
costs, supports configurable position sizing, and emits rich analytics
artifacts (JSON/CSV/plots) for downstream reporting.

Usage examples
--------------
Run the full campaign (all models + ensembles):
    python scripts/backtest_hpo_production_models.py

Limit to specific models:
    python scripts/backtest_hpo_production_models.py --models mlp ensemble_weighted_val

Quick smoke test (last 30 days, 6 symbols):
    python scripts/backtest_hpo_production_models.py --quick-test

Plot only from an existing results file:
    python scripts/backtest_hpo_production_models.py --plot-only --results-file backtesting/results/backtest_campaign_20251004_190000.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

try:
    import seaborn as sns  # type: ignore
except ImportError:  # pragma: no cover - optional at runtime
    sns = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional at runtime
    plt = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    def tqdm(iterable: Iterable, **_: object) -> Iterable:  # type: ignore
        return iterable

# Project-local imports (lazy to keep CLI discoverable)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.nn_architectures import create_model

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

BACKTEST_CONFIG: Dict[str, object] = {
    "initial_capital": 100_000.0,
    "position_sizing": {
        "strategy": "equal_weight",  # or "kelly_criterion"
        "max_positions": 20,
        "max_position_size": 0.10,  # Fraction of capital per position
    },
    "trading_costs": {
        "commission_rate": 0.001,   # 0.1% per side
        "slippage_bps": 5,          # 5 basis points = 0.05%
    },
    "thresholds": {
        "default": 0.50,
        "mlp_optimal": 0.55,  # From test evaluation
        "conservative": 0.60,
        "aggressive": 0.40,
    },
    "risk_management": {
        "stop_loss": 0.02,    # 2%
        "take_profit": 0.025, # 2.5%
        "max_hold_hours": 8,
    },
}

FEATURE_LIST: List[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
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
    "Return_1h",
    "sentiment_score_hourly_ffill",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
]

MODEL_DEFAULTS: Dict[str, Dict[str, object]] = {
    "mlp": {
        "path": PROJECT_ROOT / "models" / "hpo_derived" / "HPO_GRU_Production_mlp" / "best_model.pt",
        "threshold": BACKTEST_CONFIG["thresholds"]["mlp_optimal"],
        "val_weight": 0.355,
        "test_weight": 0.306,
        "color": "#1f77b4",
    },
    "lstm": {
        "path": PROJECT_ROOT / "models" / "hpo_derived" / "HPO_GRU_Production_lstm" / "best_model.pt",
        "threshold": BACKTEST_CONFIG["thresholds"]["default"],
        "val_weight": 0.329,
        "test_weight": 0.289,
        "color": "#ff7f0e",
    },
    "gru": {
        "path": PROJECT_ROOT / "models" / "hpo_derived" / "HPO_GRU_Production_gru" / "best_model.pt",
        "threshold": BACKTEST_CONFIG["thresholds"]["default"],
        "val_weight": 0.334,
        "test_weight": 0.269,
        "color": "#2ca02c",
    },
}

LOOKBACK_WINDOW = 24
RISK_FREE_RATE = 0.02  # 2% annual risk-free assumption

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckpointBundle:
    """Container for checkpoint metadata and artifacts."""

    name: str
    path: Path
    model_type: str
    model_config: Dict[str, object]
    validation_metrics: Dict[str, float]
    scaler: joblib
    asset_map: Mapping[str, int]
    epoch: int


@dataclass
class ModelRuntime:
    """Runtime objects required for inference."""

    bundle: CheckpointBundle
    model: torch.nn.Module
    device: torch.device


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_time: pd.Timestamp
    entry_price: float
    entry_price_effective: float
    entry_cost: float
    commission_in: float
    strategy: str
    trigger_prob: float
    threshold: float

    def mark_to_market(self, price: float) -> float:
        return self.quantity * price


@dataclass
class TradeRecord:
    timestamp_open: pd.Timestamp
    timestamp_close: pd.Timestamp
    symbol: str
    strategy: str
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    duration_hours: float
    reason: str
    trigger_prob: float
    threshold: float


@dataclass
class StrategySpec:
    name: str
    label: str
    strategy_type: str  # "single", "majority", "weighted"
    models: Tuple[str, ...]
    threshold: float
    thresholds_map: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    majority_required: int = 2
    color: Optional[str] = None


@dataclass
class StrategyResult:
    equity_curve: pd.DataFrame
    trades: List[TradeRecord]
    metrics: Dict[str, object]
    console_summary: Dict[str, object]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("backtest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Utility functions for model loading
# ---------------------------------------------------------------------------


def _resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _load_asset_mapping(mapping_path: Optional[Path]) -> Mapping[str, int]:
    """Load symbol-to-asset-id mapping."""

    if mapping_path is None:
        mapping_path = PROJECT_ROOT / "config" / "asset_id_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Asset ID mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("symbol_to_id", data)


def _clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value
    return cleaned


def _infer_model_config(model_type: str, raw_config: Mapping[str, object], state_dict: Mapping[str, torch.Tensor], n_features: int) -> Dict[str, object]:
    """Infer the minimal model configuration required for reconstruction."""

    cfg: Dict[str, object] = {
        "n_features": n_features,
        "lookback_window": LOOKBACK_WINDOW,
    }

    asset_weights = state_dict.get("asset_embedding.weight")
    if asset_weights is not None:
        cfg["num_assets"] = int(asset_weights.shape[0])
        cfg["asset_embedding_dim"] = int(asset_weights.shape[1])
    else:
        cfg["num_assets"] = int(raw_config.get("num_assets", 154) or 154)
        cfg["asset_embedding_dim"] = int(raw_config.get("asset_embedding_dim", 8) or 8)

    if model_type == "mlp":
        hidden_dims: List[int] = []
        for key, weight in state_dict.items():
            if key.startswith("mlp.") and key.endswith(".weight"):
                parts = key.split(".")
                if len(parts) >= 3 and parts[1].isdigit():
                    idx = int(parts[1])
                    if idx not in (len(hidden_dims) + 1,):
                        hidden_dims.append(int(weight.shape[0]))
                    else:
                        hidden_dims.append(int(weight.shape[0]))
        if not hidden_dims:
            hidden_dims = [128, 64, 32]
        cfg["hidden_dims"] = tuple(hidden_dims[:-1]) if len(hidden_dims) > 1 else tuple(hidden_dims)
        cfg["dropout_rate"] = float(raw_config.get("dropout_rate", 0.3) or 0.3)

    elif model_type == "lstm":
        cfg["lstm_hidden_dim"] = int(raw_config.get("lstm_hidden_dim", 64) or 64)
        cfg["lstm_num_layers"] = int(raw_config.get("lstm_num_layers", 2) or 2)
        cfg["attention_dim"] = int(raw_config.get("attention_dim", raw_config.get("attention_dim极", 64)) or 64)
        cfg["use_layer_norm"] = bool(raw_config.get("use_layer_norm", True))
        cfg["dropout_rate"] = float(raw_config.get("dropout_rate", 0.3) or 0.3)

    elif model_type == "gru":
        cfg["gru_hidden_dim"] = int(raw_config.get("gru_hidden_dim", 64) or 64)
        cfg["gru_num_layers"] = int(raw_config.get("gru_num_layers", 2) or 2)
        cfg["attention_dim"] = int(raw_config.get("attention_dim", raw_config.get("attention_dim极", 64)) or 64)
        cfg["use_layer_norm"] = bool(raw_config.get("use_layer_norm", True))
        cfg["dropout_rate"] = float(raw_config.get("dropout_rate", 0.3) or 0.3)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return cfg


def load_checkpoint_bundle(name: str, checkpoint_path: Path, asset_map: Mapping[str, int]) -> CheckpointBundle:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise ValueError(f"model_state_dict missing in checkpoint: {checkpoint_path}")
    model_state = _clean_state_dict(model_state)

    model_type = str(config.get("model_type", config.get("hpo_model_type", "unknown"))).lower()
    raw_model_cfg = dict(config.get("model_config", {}))
    metrics = {k: float(v) for k, v in checkpoint.get("metrics", {}).items()}

    scaler_dict = checkpoint.get("scalers")
    if not scaler_dict:
        scalers_path = checkpoint_path.parent / "scalers.joblib"
        if not scalers_path.exists():
            raise FileNotFoundError(f"Scaler file missing for checkpoint {checkpoint_path}")
        scaler_dict = joblib.load(scalers_path)
    scaler = scaler_dict["global"]

    n_features = len(scaler.mean_)
    model_cfg = _infer_model_config(model_type, raw_model_cfg, model_state, n_features)

    bundle = CheckpointBundle(
        name=name,
        path=checkpoint_path,
        model_type=model_type,
        model_config=model_cfg,
        validation_metrics=metrics,
        scaler=scaler,
        asset_map=asset_map,
        epoch=int(checkpoint.get("epoch", -1)),
    )
    return bundle


def instantiate_model(bundle: CheckpointBundle, device: torch.device) -> ModelRuntime:
    state = torch.load(bundle.path, map_location="cpu", weights_only=False)
    state_dict = _clean_state_dict(state["model_state_dict"])
    model = create_model(bundle.model_type, bundle.model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return ModelRuntime(bundle=bundle, model=model, device=device)


# ---------------------------------------------------------------------------
# Data preparation and prediction generation
# ---------------------------------------------------------------------------


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize column names that differ only by case
    if "vwap" not in df.columns and "VWAP" in df.columns:
        df["vwap"] = df["VWAP"]
    if "Open" not in df.columns and "open" in df.columns:
        df["Open"] = df["open"]
    if "High" not in df.columns and "high" in df.columns:
        df["High"] = df["high"]
    if "Low" not in df.columns and "low" in df.columns:
        df["Low"] = df["low"]
    if "Close" not in df.columns and "close" in df.columns:
        df["Close"] = df["close"]
    if "Volume" not in df.columns and "volume" in df.columns:
        df["Volume"] = df["volume"]
    return df


def load_symbol_dataframe(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parquet_path = PROJECT_ROOT / "data" / "historical" / symbol / "1Hour" / "data.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing historical data for {symbol}: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    df = df.loc[(df.index >= start) & (df.index <= end)].copy()
    df = _ensure_feature_columns(df)
    df.dropna(subset=["Close"], inplace=True)
    return df


def sliding_windows(features: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    if features.shape[0] < window:
        raise ValueError("Insufficient rows for lookback window")
    n_rows, n_features = features.shape
    shape = (n_rows - window + 1, window, n_features)
    strides = (features.strides[0], features.strides[0], features.strides[1])
    windows = np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides)
    return windows.copy()


def batched_inference(runtime: ModelRuntime, feature_windows: NDArray[np.float32], asset_id: int, batch_size: int = 512) -> NDArray[np.float32]:
    device = runtime.device
    model = runtime.model
    total = feature_windows.shape[0]
    probs: List[float] = []
    asset_tensor = torch.full((batch_size,), asset_id, dtype=torch.long, device=device)
    with torch.no_grad():
        for start_idx in range(0, total, batch_size):
            end_idx = min(total, start_idx + batch_size)
            batch = torch.from_numpy(feature_windows[start_idx:end_idx]).to(device)
            if asset_tensor.shape[0] != end_idx - start_idx:
                asset_tensor = torch.full((end_idx - start_idx,), asset_id, dtype=torch.long, device=device)
            logits = model(batch, asset_tensor)
            if logits.ndim == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            probs_batch = torch.sigmoid(logits).detach().cpu().numpy()
            probs.extend(probs_batch.tolist())
    return np.asarray(probs, dtype=np.float32)


def process_symbol(
    symbol: str,
    runtimes: Mapping[str, ModelRuntime],
    thresholds: Mapping[str, float],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    df_raw = load_symbol_dataframe(symbol, start, end)
    if len(df_raw) < LOOKBACK_WINDOW + 1:
        raise ValueError(f"Symbol {symbol} has insufficient data ({len(df_raw)} rows)")

    feature_df = df_raw[FEATURE_LIST].copy()
    if feature_df.isna().any().any():
        feature_df.fillna(method="ffill", inplace=True)
        feature_df.fillna(method="bfill", inplace=True)
    features_np = feature_df.to_numpy(dtype=np.float32)

    rows: Dict[str, NDArray[np.float32]] = {}
    timestamps = feature_df.index[LOOKBACK_WINDOW - 1 :]
    prices = df_raw.loc[timestamps, ["Open", "High", "Low", "Close"]].astype(np.float32)

    for name, runtime in runtimes.items():
        scaler = runtime.bundle.scaler
        scaled = scaler.transform(features_np)
        windows = sliding_windows(scaled.astype(np.float32), LOOKBACK_WINDOW)
        asset_map = runtime.bundle.asset_map
        if symbol not in asset_map:
            raise KeyError(f"Symbol {symbol} missing in asset map for model {name}")
        asset_id = int(asset_map[symbol])
        probs = batched_inference(runtime, windows, asset_id)
        rows[f"prob_{name}"] = probs

    frame = prices.copy()
    frame["symbol"] = symbol
    for col_name, values in rows.items():
        frame[col_name] = values
    frame["threshold_mlp"] = thresholds.get("mlp", BACKTEST_CONFIG["thresholds"]["default"])
    frame["threshold_lstm"] = thresholds.get("lstm", BACKTEST_CONFIG["thresholds"]["default"])
    frame["threshold_gru"] = thresholds.get("gru", BACKTEST_CONFIG["thresholds"]["default"])
    return frame


# ---------------------------------------------------------------------------
# Strategy evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_signal(row: pd.Series, spec: StrategySpec) -> Tuple[bool, float]:
    """Return (should_enter, probability_used)."""
    if spec.strategy_type == "single":
        model = spec.models[0]
        prob = float(row[f"prob_{model}"])
        threshold = spec.thresholds_map.get(model, spec.threshold)
        return prob >= threshold, prob

    if spec.strategy_type == "majority":
        votes = 0
        model_probs: List[float] = []
        for model in spec.models:
            prob = float(row[f"prob_{model}"])
            model_probs.append(prob)
            threshold = spec.thresholds_map.get(model, spec.threshold)
            if prob >= threshold:
                votes += 1
        prob_used = float(np.mean(model_probs)) if model_probs else 0.0
        return votes >= spec.majority_required, prob_used

    if spec.strategy_type == "weighted":
        weighted = 0.0
        weight_sum = 0.0
        for model in spec.models:
            w = spec.weights.get(model, 0.0)
            prob = float(row[f"prob_{model}"])
            weighted += prob * w
            weight_sum += w
        prob_used = weighted / weight_sum if weight_sum else 0.0
        return prob_used >= spec.threshold, prob_used

    raise ValueError(f"Unsupported strategy type: {spec.strategy_type}")


def position_fraction(prob: float, sizing_cfg: Mapping[str, object], risk_cfg: Mapping[str, float]) -> float:
    strategy = str(sizing_cfg.get("strategy", "equal_weight"))
    max_fraction = float(sizing_cfg.get("max_position_size", 0.10))
    if strategy == "equal_weight":
        return max_fraction

    if strategy == "kelly_criterion":
        stop_loss = float(risk_cfg.get("stop_loss", 0.02))
        take_profit = float(risk_cfg.get("take_profit", 0.025))
        b = take_profit / stop_loss if stop_loss > 0 else 1.0
        kelly = ((b * prob) - (1 - prob)) / b
        return max(0.0, min(max_fraction, kelly))

    return max_fraction


def update_equity_snapshot(
    timestamp: pd.Timestamp,
    cash: float,
    positions: Mapping[str, Position],
    rows: pd.DataFrame,
) -> Dict[str, float]:
    positions_value = 0.0
    for symbol, pos in positions.items():
        price = float(rows.loc[symbol, "Close"] if symbol in rows.index else pos.entry_price)
        positions_value += pos.quantity * price
    equity = cash + positions_value
    return {
        "timestamp": timestamp,
        "cash": cash,
        "positions_value": positions_value,
        "total_equity": equity,
    }


def run_strategy(
    spec: StrategySpec,
    market_data: pd.DataFrame,
    config: Mapping[str, object],
) -> StrategyResult:
    initial_capital = float(config["initial_capital"])
    sizing_cfg: Mapping[str, object] = config["position_sizing"]  # type: ignore
    risk_cfg: Mapping[str, float] = config["risk_management"]  # type: ignore
    costs_cfg: Mapping[str, float] = config["trading_costs"]  # type: ignore

    commission_rate = float(costs_cfg.get("commission_rate", 0.0))
    slippage = float(costs_cfg.get("slippage_bps", 0.0)) / 10_000.0
    stop_loss = float(risk_cfg.get("stop_loss", 0.02))
    take_profit = float(risk_cfg.get("take_profit", 0.025))
    max_hold_hours = float(risk_cfg.get("max_hold_hours", 8))
    max_positions = int(sizing_cfg.get("max_positions", 20))

    timestamps = market_data.index.get_level_values("timestamp").unique()
    cash = initial_capital
    positions: Dict[str, Position] = {}
    trades: List[TradeRecord] = []
    equity_rows: List[Dict[str, float]] = []

    for timestamp in tqdm(timestamps, desc=f"Simulating {spec.label}"):
        rows = market_data.xs(timestamp, level="timestamp")

        # Exit logic
        for symbol in list(positions.keys()):
            if symbol not in rows.index:
                continue
            pos = positions[symbol]
            row = rows.loc[symbol]
            holding_hours = (timestamp - pos.entry_time).total_seconds() / 3600.0
            should_exit = False
            exit_reason = "hold"
            exit_price_base = float(row["Close"])

            if timestamp > pos.entry_time:
                low_price = float(row["Low"])
                high_price = float(row["High"])
                stop_price = pos.entry_price * (1 - stop_loss)
                take_price = pos.entry_price * (1 + take_profit)

                if low_price <= stop_price:
                    should_exit = True
                    exit_reason = "stop_loss"
                    exit_price_base = stop_price
                elif high_price >= take_price:
                    should_exit = True
                    exit_reason = "take_profit"
                    exit_price_base = take_price
                elif holding_hours >= max_hold_hours:
                    should_exit = True
                    exit_reason = "timeout"
                    exit_price_base = float(row["Close"])

            if should_exit:
                exit_price_effective = exit_price_base * (1 - slippage)
                exit_value = pos.quantity * exit_price_effective
                exit_commission = exit_value * commission_rate
                cash += exit_value - exit_commission
                pnl = exit_value - exit_commission - pos.entry_cost - pos.commission_in
                return_pct = pnl / pos.entry_cost if pos.entry_cost else 0.0
                trade = TradeRecord(
                    timestamp_open=pos.entry_time,
                    timestamp_close=timestamp,
                    symbol=symbol,
                    strategy=spec.name,
                    entry_price=pos.entry_price_effective,
                    exit_price=exit_price_effective,
                    pnl=pnl,
                    return_pct=return_pct,
                    duration_hours=holding_hours,
                    reason=exit_reason,
                    trigger_prob=pos.trigger_prob,
                    threshold=pos.threshold,
                )
                trades.append(trade)
                del positions[symbol]

        # Compute equity snapshot for sizing decisions
        positions_value_running = 0.0
        for sym, pos in positions.items():
            price_ref = float(rows.loc[sym, "Close"] if sym in rows.index else pos.entry_price)
            positions_value_running += pos.quantity * price_ref
        current_equity = cash + positions_value_running

        # Entry logic
        for symbol, row in rows.iterrows():
            if symbol in positions:
                continue
            if len(positions) >= max_positions:
                break
            allowed_cash = cash
            if allowed_cash <= 0:
                break

            should_enter, prob = evaluate_signal(row, spec)
            if not should_enter or prob <= 0.0:
                continue

            fraction = position_fraction(prob, sizing_cfg, risk_cfg)
            if fraction <= 0.0:
                continue

            position_value = min(fraction * current_equity, allowed_cash)
            if position_value <= 0:
                continue

            gross_price = float(row["Close"]) * (1 + slippage)
            effective_denom = gross_price * (1 + commission_rate)
            if effective_denom <= 0:
                continue
            quantity = position_value / effective_denom
            if quantity <= 0:
                continue
            entry_cost = quantity * gross_price
            commission_in = entry_cost * commission_rate
            total_debit = entry_cost + commission_in
            if total_debit > cash:
                continue

            cash -= total_debit
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_time=timestamp,
                entry_price=float(row["Close"]),
                entry_price_effective=gross_price,
                entry_cost=entry_cost,
                commission_in=commission_in,
                strategy=spec.name,
                trigger_prob=prob,
                threshold=spec.threshold,
            )
            positions[symbol] = position
            positions_value_running += position.mark_to_market(float(row["Close"]))
            current_equity = cash + positions_value_running

        equity_rows.append(update_equity_snapshot(timestamp, cash, positions, rows))

    # Liquidate remaining positions at final timestamp
    if positions:
        final_timestamp = timestamps[-1]
        rows = market_data.xs(final_timestamp, level="timestamp")
        for symbol, pos in list(positions.items()):
            price = float(rows.loc[symbol, "Close"] if symbol in rows.index else pos.entry_price)
            exit_price_effective = price * (1 - slippage)
            exit_value = pos.quantity * exit_price_effective
            exit_commission = exit_value * commission_rate
            cash += exit_value - exit_commission
            pnl = exit_value - exit_commission - pos.entry_cost - pos.commission_in
            holding_hours = (final_timestamp - pos.entry_time).total_seconds() / 3600.0
            trade = TradeRecord(
                timestamp_open=pos.entry_time,
                timestamp_close=final_timestamp,
                symbol=symbol,
                strategy=spec.name,
                entry_price=pos.entry_price_effective,
                exit_price=exit_price_effective,
                pnl=pnl,
                return_pct=pnl / pos.entry_cost if pos.entry_cost else 0.0,
                duration_hours=holding_hours,
                reason="end_of_period",
                trigger_prob=pos.trigger_prob,
                threshold=pos.threshold,
            )
            trades.append(trade)
            del positions[symbol]
        equity_rows.append(update_equity_snapshot(final_timestamp, cash, positions, rows))

    equity_df = pd.DataFrame(equity_rows)
    equity_df = equity_df.drop_duplicates(subset="timestamp", keep="last")
    equity_df.set_index("timestamp", inplace=True)
    metrics = compute_all_metrics(equity_df, trades, initial_capital)

    console_summary = {
        "Total Return": metrics["returns"]["total_return_percent"],
        "Annualized Return": metrics["returns"]["annualized_return_percent"],
        "Sharpe": metrics["risk"]["sharpe_ratio"],
        "Sortino": metrics["risk"]["sortino_ratio"],
        "Max Drawdown": metrics["risk"]["max_drawdown_percent"],
        "Calmar": metrics["risk"]["calmar_ratio"],
        "Total Trades": metrics["trading"]["total_trades"],
        "Win Rate": metrics["trading"]["win_rate_percent"],
        "Profit Factor": metrics["trading"]["profit_factor"],
        "Avg Profit": metrics["trading"].get("avg_profit_per_trade", 0.0),
        "Avg Loss": metrics["trading"].get("avg_loss_per_trade", 0.0),
        "Avg Duration": metrics["trading"].get("avg_duration_hours", 0.0),
        "Best Trade": metrics["trading"].get("best_trade_percent", 0.0),
        "Worst Trade": metrics["trading"].get("worst_trade_percent", 0.0),
    }

    return StrategyResult(equity_curve=equity_df, trades=trades, metrics=metrics, console_summary=console_summary)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def max_drawdown(series: pd.Series) -> Tuple[float, pd.Series]:
    cumulative = series.cummax()
    drawdown = (series - cumulative) / cumulative
    return float(drawdown.min()), drawdown


def compute_returns_metrics(equity: pd.DataFrame, initial_capital: float) -> Dict[str, object]:
    total_return = equity["total_equity"].iloc[-1] / initial_capital - 1
    start = equity.index[0]
    end = equity.index[-1]
    total_days = max((end - start).total_seconds() / 86400.0, 1.0)
    years = total_days / 365.0
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return

    hourly_returns = equity["total_equity"].pct_change(fill_method=None).dropna()
    daily_equity = equity["total_equity"].resample("1D").last().ffill().dropna()
    daily_returns = daily_equity.pct_change(fill_method=None).dropna()

    monthly_returns = (
        daily_equity.resample("1ME").last().pct_change(fill_method=None).dropna()
    )

    cumulative_series = (equity["total_equity"] / initial_capital) - 1
    cumulative_records = [
        {"timestamp": ts.isoformat(), "equity": float(val), "return_pct": float(cumulative_series.loc[ts])}
        for ts, val in equity["total_equity"].items()
    ]

    return {
        "total_return_percent": float(total_return * 100),
        "annualized_return_percent": float(annualized_return * 100),
        "hourly_returns": hourly_returns.tolist(),
        "monthly_returns": {ts.strftime("%Y-%m"): float(val * 100) for ts, val in monthly_returns.items()},
        "cumulative_returns": cumulative_records,
    }


def compute_risk_metrics(equity: pd.DataFrame, returns_metrics: Mapping[str, object]) -> Dict[str, float]:
    hourly_returns = np.asarray(returns_metrics.get("hourly_returns", []), dtype=np.float64)
    daily_equity = equity["total_equity"].resample("1D").last().ffill().dropna()
    daily_returns = daily_equity.pct_change(fill_method=None).dropna()
    annual_volatility = float(daily_returns.std() * math.sqrt(252))

    max_dd, drawdown_series = max_drawdown(equity["total_equity"])
    annual_return = float(returns_metrics["annualized_return_percent"]) / 100.0
    sharpe = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility > 0 else 0.0

    downside = daily_returns[daily_returns < 0]
    downside_dev = float(downside.std() * math.sqrt(252)) if not downside.empty else 0.0
    sortino = (annual_return - RISK_FREE_RATE) / downside_dev if downside_dev > 0 else 0.0

    calmar = annual_return / abs(max_dd) if max_dd != 0 else math.inf

    var_95 = float(np.percentile(daily_returns, 5)) if len(daily_returns) > 0 else 0.0
    var_99 = float(np.percentile(daily_returns, 1)) if len(daily_returns) > 0 else 0.0

    return {
        "annualized_volatility_percent": annual_volatility * 100,
        "max_drawdown_percent": float(max_dd * 100),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "var_95_percent": float(var_95 * 100),
        "var_99_percent": float(var_99 * 100),
        "drawdown_series": drawdown_series.tolist(),
    }


def compute_trading_metrics(trades: Sequence[TradeRecord], initial_capital: float) -> Dict[str, float]:
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate_percent": 0.0,
            "profit_factor": 0.0,
            "avg_profit_per_trade": 0.0,
            "avg_loss_per_trade": 0.0,
            "avg_duration_hours": 0.0,
            "best_trade_percent": 0.0,
            "worst_trade_percent": 0.0,
            "total_pnl": 0.0,
        }

    profits = [t.pnl for t in trades]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    win_rate = len(wins) / total_trades * 100

    avg_profit = np.mean([t.pnl for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
    avg_duration = np.mean([t.duration_hours for t in trades]) if trades else 0.0

    best_trade = max((t.return_pct for t in trades), default=0.0)
    worst_trade = min((t.return_pct for t in trades), default=0.0)

    consecutive_wins = consecutive_losses = max_wins = max_losses = 0
    for trade in trades:
        if trade.pnl > 0:
            consecutive_wins += 1
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            consecutive_wins = 0
        max_wins = max(max_wins, consecutive_wins)
        max_losses = max(max_losses, consecutive_losses)

    return {
        "total_trades": total_trades,
        "win_rate_percent": float(win_rate),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf" if gross_profit > 0 else 0.0),
        "avg_profit_per_trade": float(avg_profit),
        "avg_loss_per_trade": float(avg_loss),
        "avg_duration_hours": float(avg_duration),
        "best_trade_percent": float(best_trade * 100),
        "worst_trade_percent": float(worst_trade * 100),
        "total_pnl": float(sum(profits)),
        "consecutive_wins": max_wins,
        "consecutive_losses": max_losses,
    }


def compute_symbol_breakdown(trades: Sequence[TradeRecord]) -> Dict[str, object]:
    pnl_by_symbol: MutableMapping[str, float] = defaultdict(float)
    wins_by_symbol: MutableMapping[str, int] = defaultdict(int)
    counts_by_symbol: MutableMapping[str, int] = defaultdict(int)
    for trade in trades:
        pnl_by_symbol[trade.symbol] += trade.pnl
        counts_by_symbol[trade.symbol] += 1
        if trade.pnl > 0:
            wins_by_symbol[trade.symbol] += 1
    returns_by_symbol = sorted(((sym, pnl) for sym, pnl in pnl_by_symbol.items()), key=lambda x: x[1], reverse=True)
    win_rates = {
        sym: (wins_by_symbol[sym] / counts_by_symbol[sym]) * 100 if counts_by_symbol[sym] else 0.0
        for sym in counts_by_symbol
    }
    top_symbols = returns_by_symbol[:10]
    bottom_symbols = returns_by_symbol[-10:][::-1]
    return {
        "pnl_by_symbol": {sym: pnl for sym, pnl in returns_by_symbol},
        "win_rate_by_symbol": win_rates,
        "top_symbols": top_symbols,
        "bottom_symbols": bottom_symbols,
    }


def compute_all_metrics(equity: pd.DataFrame, trades: Sequence[TradeRecord], initial_capital: float) -> Dict[str, object]:
    returns = compute_returns_metrics(equity, initial_capital)
    risk = compute_risk_metrics(equity, returns)
    trading = compute_trading_metrics(trades, initial_capital)
    symbol_breakdown = compute_symbol_breakdown(trades)
    return {
        "returns": returns,
        "risk": risk,
        "trading": trading,
        "per_symbol": symbol_breakdown,
    }


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def compute_buy_and_hold_baseline(start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, object]:
    df = load_symbol_dataframe("SPY", start, end)
    if df.empty:
        raise ValueError("SPY data unavailable for baseline computation")
    df = df.sort_index()
    start_price = float(df["Close"].iloc[0])
    end_price = float(df["Close"].iloc[-1])
    total_return = (end_price / start_price) - 1
    daily_returns = df["Close"].resample("1D").last().pct_change(fill_method=None).dropna()
    annual_return = (1 + total_return) ** (365 / max((end - start).days, 1)) - 1
    annual_volatility = float(daily_returns.std() * math.sqrt(252))
    max_dd, _ = max_drawdown(df["Close"])
    sharpe = (annual_return - RISK_FREE_RATE) / annual_volatility if annual_volatility > 0 else 0.0
    return {
        "total_return_percent": total_return * 100,
        "annualized_return_percent": annual_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_percent": max_dd * 100,
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def ensure_plots_available():
    if plt is None or sns is None:
        raise RuntimeError("matplotlib and seaborn are required for plotting. Install them or rerun without --plot-only")


def plot_equity_curves(results: Mapping[str, StrategyResult], baseline: Mapping[str, object], output_dir: Path) -> None:
    ensure_plots_available()
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        plt.plot(result.equity_curve.index, result.equity_curve["total_equity"], label=name)
    baseline_series = None
    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curves.png", dpi=150)
    plt.close()


def plot_drawdowns(results: Mapping[str, StrategyResult], output_dir: Path) -> None:
    ensure_plots_available()
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        drawdown = pd.Series(result.metrics["risk"]["drawdown_series"], index=result.equity_curve.index)
        plt.plot(result.equity_curve.index, drawdown * 100, label=name)
    plt.title("Drawdown (%)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "drawdown_curves.png", dpi=150)
    plt.close()


def plot_monthly_heatmap(result: StrategyResult, output_dir: Path) -> None:
    ensure_plots_available()
    monthly = result.metrics["returns"]["monthly_returns"]
    if not monthly:
        return
    df = pd.Series(monthly).rename("Monthly Return (%)").astype(float)
    data = df.to_frame()
    data["Year"] = data.index.str.slice(0, 4)
    data["Month"] = data.index.str.slice(5, 7)
    pivot = data.pivot(index="Year", columns="Month", values="Monthly Return (%)")
    plt.figure(figsize=(10, 4))
    sns.heatmap(pivot.astype(float), annot=True, fmt=".1f", cmap="RdYlGn", center=0)
    plt.title(f"Monthly Returns Heatmap - {result.metrics['returns']['total_return_percent']:.2f}% cumulative")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_returns_heatmap.png", dpi=150)
    plt.close()


def plot_trade_distribution(trades: Sequence[TradeRecord], output_dir: Path) -> None:
    ensure_plots_available()
    if not trades:
        return
    returns = [t.return_pct * 100 for t in trades]
    plt.figure(figsize=(10, 4))
    sns.histplot(returns, bins=40, kde=True)
    plt.title("Distribution of Trade Returns (%)")
    plt.xlabel("Return (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "trade_distribution.png", dpi=150)
    plt.close()


def plot_win_rate_by_symbol(result: StrategyResult, output_dir: Path) -> None:
    ensure_plots_available()
    win_rates = result.metrics["per_symbol"]["win_rate_by_symbol"]
    if not win_rates:
        return
    series = pd.Series(win_rates).sort_values(ascending=False)
    top = series.head(10)
    bottom = series.tail(10)
    combined = pd.concat([top, bottom])
    plt.figure(figsize=(12, 5))
    sns.barplot(x=combined.values, y=combined.index)
    plt.title("Top & Bottom Win Rates by Symbol")
    plt.xlabel("Win Rate (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "win_rate_by_symbol.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------


def print_console_summary(results: Mapping[str, StrategyResult], baseline: Mapping[str, object], campaign_meta: Mapping[str, object]) -> None:
    banner = "=" * 80
    print(banner)
    print("BACKTESTING CAMPAIGN - HPO PRODUCTION MODELS")
    print(banner)
    print()
    print(f"Period: {campaign_meta['start']} to {campaign_meta['end']}")
    print(f"Symbols: {campaign_meta['symbols']}")
    print(f"Initial Capital: ${campaign_meta['initial_capital']:,.2f}")
    print("Position Sizing: Equal weight, max 20 positions, 10% per position")
    print("Transaction Costs: 0.1% commission + 5bps slippage")
    print()
    for name, result in results.items():
        summary = result.console_summary
        print("-" * 80)
        print(f"MODEL: {name}")
        print("-" * 80)
        print(f"Total Return:        {summary['Total Return']:.2f}%")
        print(f"Annualized Return:   {summary['Annualized Return']:.2f}%")
        print(f"Sharpe Ratio:        {summary['Sharpe']:.2f}")
        print(f"Sortino Ratio:       {summary['Sortino']:.2f}")
        print(f"Max Drawdown:        {summary['Max Drawdown']:.2f}%")
        print(f"Calmar Ratio:        {summary['Calmar']:.2f}")
        print()
        print(f"Total Trades:        {summary['Total Trades']}")
        print(f"Win Rate:            {summary['Win Rate']:.2f}%")
        print(f"Profit Factor:       {summary['Profit Factor']:.2f}")
        print(f"Avg Profit/Trade:    ${summary['Avg Profit']:.2f}")
        print(f"Avg Loss/Trade:      ${summary['Avg Loss']:.2f}")
        print(f"Avg Trade Duration:  {summary['Avg Duration']:.2f} hours")
        print(f"Best Trade:          {summary['Best Trade']:.2f}%")
        print(f"Worst Trade:         {summary['Worst Trade']:.2f}%")
        print()
    print("-" * 80)
    print("Baseline (SPY Buy-Hold)")
    print("-" * 80)
    print(f"Total Return:        {baseline['total_return_percent']:.2f}%")
    print(f"Annualized Return:   {baseline['annualized_return_percent']:.2f}%")
    print(f"Sharpe Ratio:        {baseline['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:        {baseline['max_drawdown_percent']:.2f}%")
    print()

    ranking = sorted(
        ((name, result.metrics["risk"]["sharpe_ratio"]) for name, result in results.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    print("FINAL RANKING (by Sharpe Ratio):")
    for idx, (name, sharpe) in enumerate(ranking, start=1):
        print(f"{idx}. {name}: {sharpe:.2f}")
    print()
    if ranking:
        print(f"RECOMMENDATION: Deploy {ranking[0][0]} as production model")
    print(banner)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def build_strategy_specs(args: argparse.Namespace) -> Dict[str, StrategySpec]:
    threshold_override = args.threshold
    default_thresholds = {
        model: float(MODEL_DEFAULTS[model]["threshold"])
        for model in ("mlp", "lstm", "gru")
    }
    if threshold_override is not None:
        for key in default_thresholds:
            default_thresholds[key] = threshold_override

    specs: Dict[str, StrategySpec] = {}

    if "mlp" in args.models:
        specs["MLP Trial 72"] = StrategySpec(
            name="MLP Trial 72",
            label="MLP Trial 72",
            strategy_type="single",
            models=("mlp",),
            threshold=default_thresholds["mlp"],
            thresholds_map={"mlp": default_thresholds["mlp"]},
            color=MODEL_DEFAULTS["mlp"]["color"],
        )
    if "lstm" in args.models:
        specs["LSTM Trial 62"] = StrategySpec(
            name="LSTM Trial 62",
            label="LSTM Trial 62",
            strategy_type="single",
            models=("lstm",),
            threshold=default_thresholds["lstm"],
            thresholds_map={"lstm": default_thresholds["lstm"]},
            color=MODEL_DEFAULTS["lstm"]["color"],
        )
    if "gru" in args.models:
        specs["GRU Trial 93"] = StrategySpec(
            name="GRU Trial 93",
            label="GRU Trial 93",
            strategy_type="single",
            models=("gru",),
            threshold=default_thresholds["gru"],
            thresholds_map={"gru": default_thresholds["gru"]},
            color=MODEL_DEFAULTS["gru"]["color"],
        )

    if "ensemble_majority" in args.models:
        specs["Ensemble Majority (2/3)"] = StrategySpec(
            name="Ensemble Majority (2/3)",
            label="Ensemble Majority 2/3",
            strategy_type="majority",
            models=("mlp", "lstm", "gru"),
            threshold=float(np.mean(list(default_thresholds.values()))),
            thresholds_map=default_thresholds.copy(),
            majority_required=2,
            color="#9467bd",
        )
    if "ensemble_majority_strict" in args.models:
        specs["Ensemble Majority (3/3)"] = StrategySpec(
            name="Ensemble Majority (3/3)",
            label="Ensemble Majority 3/3",
            strategy_type="majority",
            models=("mlp", "lstm", "gru"),
            threshold=float(np.mean(list(default_thresholds.values()))),
            thresholds_map=default_thresholds.copy(),
            majority_required=3,
            color="#8c564b",
        )
    if "ensemble_weighted_val" in args.models:
        weights = {model: float(MODEL_DEFAULTS[model]["val_weight"]) for model in ("mlp", "lstm", "gru")}
        specs["Ensemble Weighted (Val F1)"] = StrategySpec(
            name="Ensemble Weighted (Val F1)",
            label="Ensemble Weighted Val",
            strategy_type="weighted",
            models=("mlp", "lstm", "gru"),
            threshold=default_thresholds["mlp" if threshold_override is None else "mlp"],
            weights=weights,
            color="#e377c2",
        )
    if "ensemble_weighted_test" in args.models:
        weights = {model: float(MODEL_DEFAULTS[model]["test_weight"]) for model in ("mlp", "lstm", "gru")}
        specs["Ensemble Weighted (Test F1)"] = StrategySpec(
            name="Ensemble Weighted (Test F1)",
            label="Ensemble Weighted Test",
            strategy_type="weighted",
            models=("mlp", "lstm", "gru"),
            threshold=default_thresholds["mlp" if threshold_override is None else "mlp"],
            weights=weights,
            color="#7f7f7f",
        )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest HPO production models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "mlp",
            "lstm",
            "gru",
            "ensemble_majority",
            "ensemble_weighted_val",
            "ensemble_weighted_test",
        ],
        choices=[
            "mlp",
            "lstm",
            "gru",
            "ensemble_majority",
            "ensemble_majority_strict",
            "ensemble_weighted_val",
            "ensemble_weighted_test",
        ],
        help="Which strategies to backtest",
    )
    parser.add_argument("--start-date", type=str, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, help="Override threshold for all models")
    parser.add_argument("--position-strategy", choices=["equal_weight", "kelly_criterion"], help="Position sizing strategy override")
    parser.add_argument("--quick-test", action="store_true", help="Run a shortened campaign for smoke testing")
    parser.add_argument("--quick-symbols", nargs="+", help="Symbols to use in quick test mode")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data preparation")
    parser.add_argument("--output-dir", type=str, default="backtesting/results", help="Directory for output artifacts")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate plots from results file")
    parser.add_argument("--results-file", type=str, help="Existing JSON results file for --plot-only")
    return parser.parse_args()


def load_or_plot_only(args: argparse.Namespace) -> Optional[Dict[str, object]]:
    if not args.plot_only:
        return None
    if not args.results_file:
        raise ValueError("--plot-only requires --results-file")
    results_path = _resolve_path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with results_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_setup(model_paths: Mapping[str, Path], symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> None:
    for model, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint for {model} not found: {path}")
    missing_symbols = []
    for symbol in symbols:
        parquet_path = PROJECT_ROOT / "data" / "historical" / symbol / "1Hour" / "data.parquet"
        if not parquet_path.exists():
            missing_symbols.append(symbol)
    if missing_symbols:
        raise FileNotFoundError(f"Historical data missing for symbols: {missing_symbols[:10]}")
    LOGGER.info("Validation complete: checkpoints and historical data available")


def run_campaign(args: argparse.Namespace) -> Dict[str, object]:
    if args.plot_only:
        raise RuntimeError("run_campaign should not be called in plot-only mode")

    output_dir = _resolve_path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    asset_map = _load_asset_mapping(None)
    available_symbols = list(asset_map.keys())
    historical_root = PROJECT_ROOT / "data" / "historical"
    filtered_symbols: List[str] = []
    missing_symbols: List[str] = []
    for sym in available_symbols:
        parquet_path = historical_root / sym / "1Hour" / "data.parquet"
        if parquet_path.exists():
            filtered_symbols.append(sym)
        else:
            missing_symbols.append(sym)
    if missing_symbols:
        LOGGER.warning(
            "Skipping symbols without complete historical data (showing up to 10): %s",
            missing_symbols[:10],
        )
    available_symbols = filtered_symbols

    end_date = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp("2025-10-01", tz="UTC")
    if end_date.tzinfo is None:
        end_date = end_date.tz_localize("UTC")
    start_date = pd.Timestamp(args.start_date) if args.start_date else (end_date - pd.Timedelta(days=730))
    if start_date.tzinfo is None:
        start_date = start_date.tz_localize("UTC")

    if args.quick_test:
        if args.quick_symbols:
            symbols = args.quick_symbols
        else:
            symbols = available_symbols[: min(6, len(available_symbols))]
        end_date = pd.Timestamp(end_date)
        if end_date.tzinfo is None:
            end_date = end_date.tz_localize("UTC")
        start_date = end_date - pd.Timedelta(days=30)
        LOGGER.info("Quick test mode: %s symbols, %s to %s", len(symbols), start_date.date(), end_date.date())
    else:
        symbols = available_symbols
        LOGGER.info("Full campaign: %s symbols", len(symbols))

    model_paths = {name: _resolve_path(info["path"]) for name, info in MODEL_DEFAULTS.items()}
    validate_setup(model_paths, symbols, start_date, end_date)

    if args.position_strategy:
        BACKTEST_CONFIG["position_sizing"]["strategy"] = args.position_strategy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    runtimes: Dict[str, ModelRuntime] = {}
    for name, info in MODEL_DEFAULTS.items():
        bundle = load_checkpoint_bundle(name, model_paths[name], asset_map)
        runtimes[name] = instantiate_model(bundle, device)
        LOGGER.info("Loaded %s model from %s", name.upper(), model_paths[name])

    thresholds_map = {name: float(info["threshold"]) for name, info in MODEL_DEFAULTS.items()}
    if args.threshold is not None:
        thresholds_map = {key: args.threshold for key in thresholds_map}

    specs = build_strategy_specs(args)
    if not specs:
        raise ValueError("No strategies selected. Use --models to specify at least one.")

    LOGGER.info("Generating predictions across %s symbols...", len(symbols))
    market_frames: List[pd.DataFrame] = []

    def _worker(symbol: str) -> pd.DataFrame:
        return process_symbol(symbol, runtimes, thresholds_map, start_date, end_date)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_worker, symbol): symbol for symbol in symbols}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Symbols"):
            symbol = futures[future]
            try:
                df_symbol = future.result()
                market_frames.append(df_symbol)
            except Exception as exc:  # pragma: no cover - logging path
                LOGGER.error("Error processing %s: %s", symbol, exc)

    if not market_frames:
        raise RuntimeError("No symbol data produced; aborting")

    market_df = pd.concat(market_frames)
    market_df.set_index([market_df.index, "symbol"], inplace=True)
    market_df.index.names = ["timestamp", "symbol"]

    LOGGER.info("Running strategies: %s", list(specs.keys()))
    results: Dict[str, StrategyResult] = {}
    for label, spec in specs.items():
        result = run_strategy(spec, market_df, BACKTEST_CONFIG)
        results[label] = result

    baseline = compute_buy_and_hold_baseline(start_date, end_date)

    campaign_metadata = {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "symbols": len(symbols),
        "initial_capital": BACKTEST_CONFIG["initial_capital"],
    }

    print_console_summary(results, baseline, campaign_metadata)

    plot_equity_curves(results, baseline, plots_dir)
    plot_drawdowns(results, plots_dir)
    best_strategy_name = max(results.items(), key=lambda item: item[1].metrics["risk"]["sharpe_ratio"])[0]
    plot_monthly_heatmap(results[best_strategy_name], plots_dir)
    all_trades = [trade for result in results.values() for trade in result.trades]
    plot_trade_distribution(all_trades, plots_dir)
    plot_win_rate_by_symbol(results[best_strategy_name], plots_dir)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"backtest_campaign_{timestamp}.json"
    csv_path = output_dir / f"trades_{timestamp}.csv"

    def _serialize_trade(trade: TradeRecord) -> Dict[str, object]:
        return {
            "timestamp_open": trade.timestamp_open.isoformat(),
            "timestamp_close": trade.timestamp_close.isoformat(),
            "symbol": trade.symbol,
            "strategy": trade.strategy,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl": trade.pnl,
            "return_pct": trade.return_pct,
            "duration_hours": trade.duration_hours,
            "reason": trade.reason,
            "trigger_prob": trade.trigger_prob,
            "threshold": trade.threshold,
        }

    payload = {
        "campaign_metadata": campaign_metadata,
        "models": {
            name: {
                "returns": result.metrics["returns"],
                "risk": result.metrics["risk"],
                "trading": result.metrics["trading"],
                "per_symbol": result.metrics["per_symbol"],
                "trades": [_serialize_trade(trade) for trade in result.trades],
                "equity_curve": [
                    {
                        "timestamp": ts.isoformat(),
                        "cash": float(row["cash"]),
                        "positions_value": float(row["positions_value"]),
                        "total_equity": float(row["total_equity"]),
                    }
                    for ts, row in result.equity_curve.iterrows()
                ],
            }
            for name, result in results.items()
        },
        "comparison": {
            "ranking_by_sharpe": sorted(
                ((name, result.metrics["risk"]["sharpe_ratio"]) for name, result in results.items()),
                key=lambda x: x[1],
                reverse=True,
            ),
            "ranking_by_returns": sorted(
                ((name, result.metrics["returns"]["total_return_percent"]) for name, result in results.items()),
                key=lambda x: x[1],
                reverse=True,
            ),
            "ranking_by_win_rate": sorted(
                ((name, result.metrics["trading"]["win_rate_percent"]) for name, result in results.items()),
                key=lambda x: x[1],
                reverse=True,
            ),
            "recommendation": max(results.items(), key=lambda item: item[1].metrics["risk"]["sharpe_ratio"])[0],
            "baseline": baseline,
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("Saved JSON report to %s", json_path)

    trades_records = []
    for name, result in results.items():
        for trade in result.trades:
            record = trade.__dict__.copy()
            record["strategy"] = name
            trades_records.append(record)
    trades_df = pd.DataFrame(trades_records)
    trades_df.to_csv(csv_path, index=False)
    LOGGER.info("Saved trades log to %s", csv_path)

    return payload


def main() -> None:
    args = parse_args()
    preload = load_or_plot_only(args)
    if args.plot_only:
        if preload is None:
            raise RuntimeError("Failed to load results for plotting")
        output_dir = _resolve_path(args.output_dir)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        dummy_results: Dict[str, StrategyResult] = {}
        for name, data in preload["models"].items():
            equity_df = pd.DataFrame(data["equity_curve"])
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True)
            equity_df.set_index("timestamp", inplace=True)
            trades = [
                TradeRecord(
                    timestamp_open=pd.Timestamp(trade["timestamp_open"]),
                    timestamp_close=pd.Timestamp(trade["timestamp_close"]),
                    symbol=trade["symbol"],
                    strategy=trade["strategy"],
                    entry_price=trade["entry_price"],
                    exit_price=trade["exit_price"],
                    pnl=trade["pnl"],
                    return_pct=trade["return_pct"],
                    duration_hours=trade["duration_hours"],
                    reason=trade["reason"],
                    trigger_prob=trade["trigger_prob"],
                    threshold=trade["threshold"],
                )
                for trade in data.get("trades", [])
            ]
            result = StrategyResult(
                equity_curve=equity_df,
                trades=trades,
                metrics={
                    "returns": data["returns"],
                    "risk": data["risk"],
                    "trading": data["trading"],
                    "per_symbol": data["per_symbol"],
                },
                console_summary={},
            )
            dummy_results[name] = result
        plot_equity_curves(dummy_results, preload["comparison"]["baseline"], plots_dir)
        plot_drawdowns(dummy_results, plots_dir)
        best_name = preload["comparison"]["ranking_by_sharpe"][0][0]
        plot_monthly_heatmap(dummy_results[best_name], plots_dir)
        all_trades = [trade for result in dummy_results.values() for trade in result.trades]
        plot_trade_distribution(all_trades, plots_dir)
        plot_win_rate_by_symbol(dummy_results[best_name], plots_dir)
        LOGGER.info("Regenerated plots from %s", args.results_file)
        return

    run_campaign(args)


if __name__ == "__main__":
    main()
