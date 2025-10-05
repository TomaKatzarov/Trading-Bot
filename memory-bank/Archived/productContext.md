# Product Context Documentation (Updated 2025-04-14)

## Project Purpose
An AI-powered trading system that operates exclusively on a local machine, directly connected to a paper trading account via the Alpaca API. The system leverages a Large Language Model (LLM) with LoRA adaptation to monitor live market data (including technical indicators and sentiment) and predict a trading signal class. Based on the predicted class, the system generates one of three decisions:

*   **Buy:** When the predicted class indicates a strong buy (4) or buy (3) signal.
*   **Hold:** When the predicted class indicates a neutral (2) signal.
*   **Sell:** When the predicted class indicates a sell (1) or strong sell (0) signal.

The comprehensive dashboard displays all trade activities, reasoning (including predicted class), performance metrics, and allows users to select symbols and input account details. **The primary goal is to optimize the system to generate profitable trading decisions.**

## Core Objectives
*   **Local System Operation:** Ensure all operations are performed locally for security, privacy, and control. Eliminate reliance on external computational resources except for API data feeds.
*   **Profit-Optimized Automated Trading:** Automatically initiate trades for selected symbols based on the generated trade decision derived from the LLM's predicted class label, with the **explicit goal of maximizing profitability**. Base trade strategies on the user's inputted account balance and implement basic risk management (stop-loss, take-profit).
*   **Account and Trade Feedback Management:** Maintain and update a persistent account balance and track open positions. Log every trade (including PnL) to a local database and provide aggregated trade feedback for model adaptation. Allow interactive updates from the user via a dedicated module.
*   **Comprehensive Data Integration:** Gather realtime market data, load historical data, incorporate sentiment analysis, and generate a unified 10-feature context vector (9 technical + 1 sentiment) for the LLM.
*   **Comprehensive Dashboard Integration:** Provide real-time visualization of live data, executed trades, open positions, performance metrics, predicted class labels, and final decisions.
*   **User Notifications:** Implement alerting mechanisms (e.g., email) to notify users of executed trades along with the decision rationale.
*   **Adaptive Learning and Specialization:** Continuously adapt the LoRA specialization based on trade outcomes (**PnL**), aiming to improve the prediction of profitable trading signals.
*   **Risk Management:** Implement basic stop-loss/take-profit rules (+5%/-2%) and dynamic position sizing. Use the three-tier decision system (Buy, Hold, Sell), potentially weighted by signal strength, to manage exposure based on user settings and available capital. *(Note: Decision Engine parameters need alignment)*.
*   **Backtesting & Evaluation:** **Planned:** Implement a backtesting engine to rigorously evaluate strategy performance on historical data before live paper trading.

## Enhancements and Optimizations
*   **Alpaca API Integration:** Fully integrate Alpaca's API for market data, news feeds, and paper trading, leveraging streaming capabilities where possible.
*   **Exclusive Paper Trading Focus:** Emphasize the educational and testing nature of the platform, operating solely with paper trading accounts.
*   **Advanced Sentiment Analysis:** Continue using sentiment as an input feature, potentially exploring more advanced NLP techniques in the future.
*   **User-Centric Design:** Enable users to customize symbols and settings via the dashboard.
*   **Secure Notification System:** Implement secure email notifications.
*   **Real-Time Dashboard Performance:** Optimize dashboard responsiveness.
*   **Enhanced Learning Algorithms:** Develop robust algorithms that learn from trade outcomes (PnL) to improve future predictions. This is the **current major focus**.
*   **Error Handling and Logging:** Implement comprehensive error handling and detailed logs.

## Core Problems to Solve
*   **Profit-Optimized Prediction:** Training the LLM/LoRA to reliably predict signals that lead to profitability based on market context (currently in progress).
*   Dynamic market adaptation without full model retraining (potential future enhancement).
*   Real-time decision making with efficient context integration.
*   Effective risk management based on model output and user settings (parameter alignment needed).

## Updated Operational Workflow
```mermaid
graph TD
    A[User Selects Symbols via Dashboard] --> B[System Monitors Live Market Data];
    B --> C[Sentiment Analysis];
    B --> D[Calculate Technical Indicators];
    C --> E{Generate Context Vector (10 Features)};
    D --> E;
    E -- Context Vector --> F[LLM Handler + LoRA];
    F -- Predicted Class (0-4) --> G{Class-to-Decision Mapping};
    G -- Decision ("BUY"/"SELL"/"HOLD") --> H{Decision Engine};
    H --> |Check Risk Rules (+5%/-2%)| H1{Execute Trade?};
    H1 -->|BUY or SELL| I[Execute Trade in Paper Account];
    H1 -->|HOLD or Risk Fail| J[No Action / Log Hold];
    I --> K[Update Dashboard with Trade Details];
    K --> L[Notify User via Secure Email/Alert];
    L --> M[Log Trade Outcome (PnL)];
    M --> N((Retraining Pool / Feedback));
```

## Key Features
*   **Automated Trade Execution:** Executes trades based on LLM-predicted class mapped to Buy/Sell/Hold decisions, subject to risk management rules.
*   **Dashboard Monitoring:** Real-time dashboard showing trades, PnL, live data, predicted class, and final decision. User settings for symbols and account details.
*   **User Notifications:** Secure alerts for executed trades.
*   **Adaptive Learning:** **Current Focus:** Model improvement based on trade outcomes (PnL).
*   **Local Data Processing:** Ensures security and privacy.
*   **Risk Management Tools:** Basic stop-loss/take-profit (+5%/-2%) and dynamic position sizing. Decision system tailored to user's risk tolerance and capital. *(Note: Implementation needs parameter alignment)*.
*   **Backtesting:** **Planned:** Capability to simulate strategy performance on historical data.