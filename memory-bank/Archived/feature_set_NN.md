# Finalized Feature Set for Custom Neural Network (NN) Approach

This document outlines the finalized set of features and timeframes to be used for the custom Neural Network (NN) model development, as decided on 2025-05-23. This decision is part of Task 1.1 in the project's strategic pivot to NN and Reinforcement Learning.

## I. Timeframe Features

1.  **Data Granularity:** 1-hour bars.
    *   *Rationale:* Aligns with existing data loading capabilities and is suitable for the 8-hour prediction horizon.
2.  **Lookback Window (Input Sequence Length):** 24 to 48 hours (configurable, starting with 24 hours).
    *   *Rationale:* Provides sufficient context for predicting an 8-hour event horizon, balancing historical information against model complexity.
3.  **Prediction Horizon:** 8 hours (for the +5% profit / -2% stop-loss target).
    *   *Rationale:* Defined by the project's trading signal criteria.

## II. Input Features for the Neural Network

The following features will be calculated based on 1-hour bar data and used as input to the NN models.

### A. Technical Indicators (Derived from 1-hour bars)

1.  **SMA (10-hour):** Simple Moving Average over 10 hours.
    *   *Purpose:* Captures short-term trend.
2.  **SMA (20-hour):** Simple Moving Average over 20 hours.
    *   *Purpose:* Captures medium-term trend.
3.  **MACD Line (12-hour EMA, 26-hour EMA):** Moving Average Convergence Divergence line.
    *   *Purpose:* Trend and momentum identification.
4.  **MACD Signal Line (9-hour EMA of MACD Line):** Signal line for MACD.
    *   *Purpose:* Generates crossover signals with MACD Line.
5.  **MACD Histogram (MACD Line - MACD Signal Line):** Difference between MACD and its signal line.
    *   *Purpose:* Visualizes MACD momentum and divergence.
6.  **RSI (14-hour period):** Relative Strength Index.
    *   *Purpose:* Measures momentum and overbought/oversold conditions.
7.  **Stochastic Oscillator %K (14-hour period, 3-hour smoothing):** Stochastic %K line.
    *   *Purpose:* Momentum indicator.
8.  **Stochastic Oscillator %D (3-hour SMA of %K):** Stochastic %D line (smoothed %K).
    *   *Purpose:* Signal line for Stochastic Oscillator.
9.  **ADX (14-hour period):** Average Directional Index.
    *   *Purpose:* Measures trend strength (not direction).
10. **ATR (14-hour period):** Average True Range.
    *   *Purpose:* Measures market volatility.
11. **Bollinger Bandwidth (20-hour period, 2 standard deviations):** (Upper Band - Lower Band) / Middle Band.
    *   *Purpose:* Measures relative volatility, identifying squeezes and expansions.
12. **OBV (On-Balance Volume):**
    *   *Purpose:* Accumulates volume on up vs. down periods, indicating buying/selling pressure.
13. **Volume SMA (20-hour SMA of Volume):** Simple Moving Average of trading volume.
    *   *Purpose:* Provides a baseline for current volume activity, helps identify unusual volume.
14. **1-hour Return (Percentage Change):** Price change from the previous hour.
    *   *Purpose:* Basic price movement information.

### B. Sentiment Feature

15. **Daily FinBERT Sentiment Score:** Normalized score [0, 1], forward-filled to match hourly technical data.
    *   *Purpose:* Captures external market sentiment not present in price/technical data.

### C. Contextual Features

16. **Asset ID Embedding:** A learnable embedding vector for each traded symbol.
    *   *Purpose:* Allows a single model to learn asset-specific behaviors when trained on multiple symbols.
17. **Day of Week (Sine Component):** `sin(2 * pi * day_of_week / 7)` where day_of_week is 0-6.
    *   *Purpose:* Cyclical encoding to capture weekly seasonality.
18. **Day of Week (Cosine Component):** `cos(2 * pi * day_of_week / 7)` where day_of_week is 0-6.
    *   *Purpose:* Cyclical encoding to capture weekly seasonality.

**Total Base Features (excluding Asset ID embedding dimensions):** Approximately 18 numerical features.

## III. Data Preparation Module

A new Python module, `core/data_preparation_nn.py`, will be created. This module will be responsible for:
*   Loading raw historical data (via `core/hist_data_loader.py`).
*   Calculating all the technical indicators listed above.
*   Integrating the sentiment scores.
*   Generating the contextual features (Asset ID, Day of Week).
*   Creating windowed sequences of these features.
*   Generating target labels based on the +5% profit / -2% stop-loss within an 8-hour horizon.
*   Handling data scaling (e.g., StandardScaler, RobustScaler).
*   Splitting data into training, validation, and test sets, respecting temporal order.
*   Implementing strategies for class imbalance (e.g., weighted loss, oversampling, augmentation).

This dedicated module will separate the data preparation logic for the new NN approach from any previous data preparation scripts used for the LoRA-based models.

## IV. Primary Neural Network Architectures for Focus

Based on the literature review and project goals, the primary NN architectures to be prototyped and evaluated are:
1.  **MLP (Multi-Layer Perceptron):** As a baseline.
2.  **LSTM (Long Short-Term Memory) / GRU (Gated Recurrent Unit):** For sequence modeling.
3.  **CNN-LSTM Hybrid:** Combining convolutional layers for local feature extraction with LSTMs for temporal modeling.

Further architectural refinements (e.g., adding Attention mechanisms) will be considered based on the performance of these initial candidates, as outlined in the `Diagnostic Report and Remediation Plan`.