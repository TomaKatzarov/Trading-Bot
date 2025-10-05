**Comprehensive Forward‑Looking Plan: Custom Neural Network & Reinforcement Learning for Autonomous Trading**

*(Last updated: 2025-05-23)*


## 1  Introduction & Rationale for Pivot
The project initially aimed to leverage Large Language Models (LLMs) with LoRA adaptation for trading‑signal prediction.  While that path yielded **60 % actionable decisions in the latest pilot**, persistent high loss and modest F1 scores revealed a fundamental mismatch between generic text transformers and noisy, non‑stationary financial time‑series.  **Custom neural networks**—specifically designed for multi‑variate sequences—and **direct policy learning via Reinforcement Learning (RL)** align better with the domain.

Key drivers of the pivot:

* **Tailored architectures.**  An LSTM/GRU or attention‑augmented network can encode temporal dependencies and indicator interactions that an LLM never sees.  
* **Policy optimisation.**  RL lets us optimise *returns* directly rather than classify labels that are only proxies for profit.  
* **Adaptability.**  Advanced RL (e.g., PPO, SAC) can continually adapt to regime shifts.  
* **Computational fit.**  Purpose‑built local NNs are lighter than fine‑tuned 8 B‑parameter language models, easing on‑device inference.  
* **Interpretability.**  Attention maps and value estimates can be audited more easily than a giant transformer’s latent states.

---

## 2  Proposed Neural Network Architecture(s)

> *Primary candidates for initial implementation (Phase 1) are MLP, LSTM/GRU, and CNN-LSTM Hybrid.*

| Candidate | Strengths | Weaknesses |
|-----------|-----------|------------|
| **Baseline MLP** | Fast, few parameters, sanity‑check for pipeline. | No temporal memory; weak on long patterns. |
| **LSTM / GRU** | Proven on finance data; gated memory for long dependencies; easy to regularise. | Sequential compute; may overfit if too deep; forgets distant context without help. |
| **LSTM/GRU + Attention** *(primary target)* | Highlights salient bars; long‑range context without huge hidden state. | Slightly more compute; extra hyper‑params. |
| **CNN → LSTM Hybrid** | Local feature extraction then temporal modelling. | Added complexity; conv layer helped little in pilot. |
| **Small Transformer Encoder** | Parallelisable; self‑attention captures arbitrary lag relationships. | Memory O(n²); prone to overfit small data; heavier GPU load. |
| **Advanced GNN / TCN** *[(reconstructed)]* | Useful if modelling inter‑asset graphs or very long context. | Requires extra data engineering; slated for Phase 3 if needed. |

*Implementation Notes*

* Use **PyTorch 2.2**; FP‑32 for training
* **LayerNorm** inside recurrent blocks for stability; **Dropout** 0.2–0.5; optional **BatchNorm1d** after conv layers.  
* Add an **asset‑ID embedding** (one‑hot or learned) concatenated to each timestep vector when training on multiple symbols.

---

## 3  Features & Data Strategy

The finalized feature set, including technical indicators, sentiment features, contextual features, and timeframe considerations, is documented in detail in [`memory-bank/feature_set_NN.md`](memory-bank/feature_set_NN.md).

1.  **Key Aspects from `feature_set_NN.md`:**
    *   **Data Granularity:** 1-hour bars.
    *   **Lookback Window:** 24-48 hours (configurable, starting with 24).
    *   **Prediction Horizon:** 8 hours (for +5% profit / -2% stop-loss target).
    *   **Core Features:** ~18 base features including SMAs, MACD, RSI, Stochastic Oscillator, ADX, ATR, Bollinger Bandwidth, OBV, Volume SMA, 1-hour Return, Daily FinBERT Sentiment, Asset ID Embedding, and Day of Week cyclical encoding.

2.  **Data Preparation Module:**
    *   A new module, `core/data_preparation_nn.py`, will be created to handle all data processing, feature engineering, and sequence generation specifically for the Neural Network models.

3.  **Pre‑processing (as detailed in `feature_set_NN.md` and to be implemented in `core/data_preparation_nn.py`):**
    *   StandardScaler fitted on **training only**; RobustScaler for skewed volume.
    *   Windowed sequences of defined length; label = 1 if **+5 % profit before −2 % stop within 8 h**.
    *   **Class imbalance** (e.g., ≈ 1 : 50) to be addressed with weighted loss **plus** oversampling & augmentation (e.g., Gaussian jitter ±0.5 %, ±1‑bar shift).

4.  **Data versioning:**
    *   All Parquet feature stores tracked under DVC; each training run logs commit‑hash + scaler artefact.

---

## 4  High‑Fidelity Simulation & Back‑testing Environment

*Core requirements*

* Event‑driven bar playback (1 h default, switchable).  
* Realistic cost model: $0.00 commission, 2 bp slippage, 0.5 bp spread.  
* Portfolio tracker with position sizing and cash drag.  
* Configurable **max holding period** equal to prediction horizon.  
* **Walk‑forward** evaluation: rolling windows, leave‑one‑symbol‑out, regime slicing (trend, volatility terciles).  
* Pluggable strategy classes: baseline NN classifier, later RL agent.

*Implementation*

* Use existing `core/backtesting` engine; extend to Gym‑compatible API (`step/ reset / render`).  
* Persist trade logs to SQLite for audit; summarise via dagster ETL into dashboard.

---

## 5  Reinforcement Learning Approach

### 5.1  Agent Class
* **Actor‑Critic (PPO) preferred**: stable on continuous rewards, supports discrete or continuous actions.  
* **Policy network** initialised from *supervised NN weights*; extra head predicts state‑value \(V(s)\).  
* Action space v1 = {*Long*, *Flat*}.  Short‑selling reserved for Phase 3.

### 5.2  State Representation
* Latest tensor (same as supervised training)  
* Position flag (+1 if long, 0 if flat)  
* Optional market regime embedding (volatility tercile)

### 5.3  Reward Function
\[
R_t=\text{ΔPortfolioValue}_t - \lambda \cdot \text{RiskPenalty}_t
\]
where λ ≈ 0.1 penalises draw‑down; trade cost deducted per step.

### 5.4  Training Loop
* Parallel environments (n = 4) with different symbol streams.  
* **Curriculum**: start with single symbol (AAPL), expand to basket.  
* Stop when 100‑episode rolling Sharpe > 1 and max draw‑down < 10 %.

### 5.5  Safety & Interpretability
* Action‑clipping guard rails (max 1 position).  
* Log *λ‑gradients* of reward → feature for explainability.  
* Auto‑disable agent if live PnL < −3 σ rolling expectation.

---

## 6  Phased Development Road‑map

| Phase | Goal | Duration | Exit Criteria |
|-------|------|----------|---------------|
| **1** | Build supervised NN baseline | Apr – Sep 2025 | Val F1 ≥ 0.25 **and** profitable walk‑forward back‑test |
| **2** | Implement RL env + seed agent | Oct 2025 – Jan 2026 | PPO agent Sharpe > 1 vs baseline on 2 symbols |
| **3** | Advanced RL / Ensemble / Regime adaptation | Q1–Q2 2026 | Paper‑trading 3‑month PnL > 5 %, max DD < 8 % |
| **4** | Graphical interface and toolset to be devloped |
| **5** | Paper‑trading deployment & monitoring | Continuous | 12‑month live paper Return > 10 %, Sharpe > 1.5 |

---

## 7  Risk Mitigation Strategies  *[(reconstructed)]*

* **Model over‑fit** → k‑fold walk‑forward, dropout, weight‑decay.  
* **Regime shift** → schedule re‑training every 30 days; volatility embedding fed into agent.  
* **Data outages** → local Parquet cache; fallback to last known NN policy with tighter stop.  
* **Compute limits** on local PC → smaller batch, gradient accumulation; nightly long‑run training.  
* **Catastrophic trading error** → hard kill‑switch if live cumulative loss > 2 % of virtual equity.

---

## 8  Success Metrics & Evaluation Criteria  *[(reconstructed)]*

| Category | Metric | Target (Phase 1) | Target (Phase 4) |
|----------|--------|------------------|------------------|
| Classification | Positive‑class F1 | ≥ 0.25 | n/a |
| Trading Sim | Walk‑forward PF (profit / loss) | > 1.3 | > 1.6 |
| Risk | Max draw‑down | < 12 % | < 8 % |
| Return | CAGR (paper) | > 5 % | > 10 % |
| Stability | Monthly Sharpe | > 0.8 | > 1.5 |
| Ops | Inference latency | < 50 ms | < 50 ms |

---

### Change‑Log (abridged)

* 23‑May‑2025  – Finalized NN feature set (see `feature_set_NN.md`); noted creation of `core/data_preparation_nn.py`; confirmed primary NN architectures for Phase 1.
* 22‑May‑2025  – Added Attention‑based architecture plan; updated Phase 1 timeline.
* 11‑May‑2025  – Initial pivot from LLM‑LoRA → NN + RL.
* 14‑May‑2025  – Documented data pipeline & system patterns.
* 16‑May‑2025  – Finalised initial NN architectures & supervised target definition.

---

*End of document.*
