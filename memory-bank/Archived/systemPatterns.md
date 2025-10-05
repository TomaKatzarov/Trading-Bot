# System Patterns & Architecture (Updated 2025-05-23)

## Directory Structure

```mermaid
graph TD
    A[Project Root: TradingBotAI] --> B[core/]
    A --> C[models1/]
    A --> D[data/]
    A --> E[tests/]
    A --> F[config/]
    A --> G[utils/]
    A --> H[training/]
    A --> I[cline_docs/]
    A --> J[models/]

    B --> B1[decision_engine.py]
    B --> B2[web_interface.py]
    B --> B3[realtime_handler.py]
    B --> B4[hist_data_loader.py]
    B --> B5[news_sentiment.py]
    B --> B6[historical_context.py]
    B --> B7[data_preparation.py] # (Legacy for LoRA)
    B --> B_new[data_preparation_nn.py] # (New for NN/RL) - IMPLEMENTED
    B --> B8[db_models.py]
    B --> B_nn_arch[nn_architectures.py] # (New for NN/RL) - IMPLEMENTED
    B --> B9[...]

    C --> C1[Base Model Files (e.g., deepseek...)]
    C --> C2[adapter_config.json (Base Model Related)]
    C --> C3[tokenizer.json etc.]

    D --> D1[historical/]
    D --> D2[realtime/]
    D --> D3[training data/ (training_data_pnl_v1.jsonl)]

    E --> E1[test_llm_integration.py]
    E --> E2[test_historical.py]
    E --> E3[initial_population.py]
    E --> E4[purge_db.py]
    E --> E5[verify_db_data.py]
    E --> E6[...]

    F --> F1[symbols.json]

    G --> G1[db_setup.py]
    G --> G2[performance_monitor.py]
    G --> G3[account_manager.py]
    G --> G4[audit_models.py]
    G --> G5[gpu_utils.py] # Added

    H --> H1[lora_trainer1.py]
    H --> H2[adapter_runs/ (Trained Adapters - Timestamped)] # Updated

    J --> J1[llm_handler.py]
    J --> J2[adapters/ (Optional, for manually promoted adapters)]
    J --> J3[inspect_llm.py]

    A --> K[run_training.py]
    A --> L[Credential.env]
    A --> M[trading_bot.db]
    A --> N[scripts/] # Added
    N --> N1[check_data_distribution.py]
    N --> N2[verify_sentiment_processing.py] # Added
    N --> N3[...]
```

**Text Outline:**

```
TradingBotAI/
├── core/                 # Core application logic
│   ├── data_preparation.py  # Legacy: Data prep for LoRA (10 features, PnL target, balancing)
│   ├── data_preparation_nn.py # New: Data prep for custom NN/RL (see feature_set_NN.md) - IMPLEMENTED
│   ├── decision_engine.py # Strategy execution, uses LLMHandler output (to be adapted for NN)
│   ├── hist_data_loader.py # Historical data fetching (Parquet)
│   ├── historical_context.py # Indicator calculation
│   ├── news_sentiment.py  # Sentiment analysis logic (FinBERT) - Enhanced with concurrent processing
│   ├── realtime_handler.py # Real-time data retrieval
│   ├── db_models.py       # SQLAlchemy DB models
│   ├── web_interface.py   # Dashboard backend (Flask/Gunicorn)
│   ├── models/            # New subdirectory for NN model definitions
│   │   └── nn_architectures.py # MLP, LSTM+Attention, GRU+Attention, CNN-LSTM
│   ├── experiment_management/ # New subdirectory for experiment management modules
│   │   ├── config_manager.py # Configuration loading, validation, history
│   │   ├── enhanced_logging.py # Comprehensive MLflow logging
│   │   ├── experiment_organizer.py # Structured experiment naming, tagging, HPO hierarchy
│   │   └── reporting.py # Automated report generation
│   └── ...                # Other core components
│
├── models/               # AI/ML component handling (Legacy LLM/LoRA specific)
│   ├── llm_handler.py     # Loads base model & LATEST LoRA, handles inference, maps class->decision
│   ├── adapters/          # Optional: Directory for manually promoted production adapters
│   └── inspect_llm.py     # Utility for model inspection
│
├── models1/              # Base model storage (Legacy LLM)
│   ├── # Files for unsloth/deepseek-r1-distill-llama-8b-unsloth-bnb-4bit
│   └── ...
│
├── training/             # Training scripts and outputs
│   ├── lora_trainer1.py   # LoRA training script (uses Unsloth, PEFT, HF Trainer)
│   └── adapter_runs/      # Output dir for timestamped trained adapters # Updated
│
├── data/                 # Data storage
│   ├── historical/       # Processed historical market data (Parquet)
│   ├── realtime/         # Streaming data buffers (if used)
│   └── training data/    # Output of data_preparation.py (e.g., training_data_pnl_v1.jsonl)
│
├── tests/                # Testing framework
│   ├── initial_population.py # Populates DB with historical data & sentiment
│   ├── purge_db.py        # Clears database tables
│   ├── verify_db_data.py  # Verifies DB data integrity
│   ├── test_llm_integration.py # Tests LLMHandler+LoRA integration
│   └── ...                # Other unit/integration tests
│
├── cline_docs/           # System documentation (MD files)
│   └── ...               # All documentation files
│
├── utils/                # Utility modules
│   ├── db_setup.py       # Database connection and initialization
│   ├── account_manager.py # Tracks account balance, positions
│   ├── gpu_utils.py      # GPU detection and optimization helpers # Added
│   └── ...               # Other helpers
│
├── scripts/              # Utility and helper scripts # Added
│   ├── check_data_distribution.py # Analyzes training data class balance
│   ├── verify_sentiment_processing.py # Tests enhanced sentiment pipeline # Added
│   ├── attach_sentiment_to_hourly.py # Primary sentiment attachment implementation # Added
│   ├── verify_sentiment_attachment.py # Sentiment attachment verification and testing # Added
│   ├── demo_sentiment_attachment.py # Sentiment attachment demonstration workflow # Added
│   └── ...
│
├── config/               # Configuration files
│   └── symbols.json      # Tradable instruments list
│
├── run_training.py       # Wrapper script to run lora_trainer1.py with env setup
├── Credential.env        # Environment variables (API keys, etc.)
├── trading_bot.db        # SQLite database file
└── ...                   # Other project files (.gitignore, requirements.txt, etc.)
```

---

## System Architecture & Key Technical Decisions

### High-Level Components
*   **Data Acquisition:** `hist_data_loader.py` (historical), `realtime_handler.py` (live), `news_sentiment.py` (sentiment with FinBERT, concurrent processing, Parquet storage).
*   **Data Processing & Storage:** `historical_context.py` (indicators), `db_models.py` / `db_setup.py` (SQLite DB via SQLAlchemy), `data_preparation.py` (Legacy: PnL-based training file generation for LoRA), [`core/data_preparation_nn.py`](core/data_preparation_nn.py) (New: Feature engineering and sequence generation for NN/RL, as per [`memory-bank/feature_set_NN.md`](memory-bank/feature_set_NN.md)).
*   **AI/ML Core:**
    *   **Custom NN Models:** [`core/models/nn_architectures.py`](core/models/nn_architectures.py) (New: Contains MLP, LSTM/GRU+Attention, CNN-LSTM models).
    *   **Experiment Management System:**
        *   **`core/experiment_management/config_manager.py (ConfigurationManager)`**: Centralized configuration management (YAML, CLI overrides, validation, history).
        *   **`core/experiment_management/enhanced_logging.py (EnhancedMLflowLogger)`**: Comprehensive MLflow logging (model architecture, environment, data info, scalers, plots).
        *   **`core/experiment_management/experiment_organizer.py (ExperimentOrganizer)`**: Structured experiment organization (naming, tagging, HPO hierarchy).
        *   **`core/experiment_management/reporting.py (ExperimentReporter)`**: Automated report generation (training, HPO, comparison, HTML).
        *   These modules are integrated into `training/train_nn_model.py` and `training/run_hpo.py`.
        *   For detailed information on the experiment management system, see `docs/experiment_management.md`.
*   **Hyperparameter Optimization (HPO) Framework:**
        *   **`training/run_hpo.py`**:
            *   **Role**: Main orchestration script for Hyperparameter Optimization studies.
            *   **Technology**: Uses the Optuna library.
            *   **Configuration**: Via YAML files (e.g., `training/config_templates/hpo_example.yaml`) and CLI arguments.
            *   **Functionality**: Manages HPO trials, defines search spaces, calls the core training logic in `train_nn_model.py`, logs results to MLflow and SQLite.
            *   **Key Features**: Supports various samplers (TPE, Random) and pruners (Median, Hyperband), configurable optimization metric and direction.
        *   **`training/run_hpo_quick_start.py`**:
            *   **Role**: Simplified, user-friendly wrapper script for quickly launching HPO studies.
            *   **Functionality**: Provides sensible defaults and a streamlined interface for common HPO tasks or initial exploration. Calls `run_hpo.py` internally.
        *   **`training/config_templates/hpo_example.yaml`**:
            *   **Role**: Example configuration file demonstrating how to set up parameters for an HPO study using `run_hpo.py`.
        *   **`docs/hpo_usage_guide.md`**:
            *   **Role**: Primary documentation resource detailing the HPO framework, how to configure and run HPO studies, and how to interpret results.
    *   **Legacy LLM/LoRA:**
        *   `models1/`: Stores the base LLM (`unsloth/deepseek...`).
        *   `training/`: Contains `lora_trainer1.py` (using Unsloth/PEFT/HF Trainer) and `adapter_runs/` (timestamped saved adapters).
        *   `models/llm_handler.py`: Loads base+Latest LoRA, takes 10-feature `context_vector`, generates predicted class (0-4), maps class to string decision ("BUY"/"SELL"/"HOLD").
*   **Decision & Execution:**
    *   `core/decision_engine.py`: Orchestrates analysis; gets data, prepares feature vector, calls model (NN or legacy LLM), receives signal/decision, applies risk rules, potentially places orders via Alpaca API. (Will need adaptation for NN model outputs).
    *   `utils/account_manager.py`: Tracks balance and positions.
*   **Testing:** `tests/` directory contains scripts for DB management, population, verification, and integration testing.
*   **UI:** `core/web_interface.py` (details TBD).
*   **Utilities:** `utils/` contains DB setup, account management, and GPU optimization helpers.

### Key Design Patterns & Logic
*   **LoRA Fine-Tuning:** Uses PEFT and Unsloth libraries for efficient LoRA training. Training script (`lora_trainer1.py`) and execution wrapper (`run_training.py`) are established.
*   **Classification-Based Decisions:**
    *   **Training Target:** LoRA trained to predict a class label (0-4) based on simulated PnL outcomes (8h window, +5%/-2% TP/SL).
    *   **LLM Handler Mapping:** `LLMHandler` translates the predicted class (0-4) into a string decision ("BUY", "SELL", "HOLD").
    *   **Decision Engine Usage:** `DecisionEngine` consumes the string decision.
*   **Context Generation (Legacy LoRA):** `data_preparation.py` generates a 10-feature vector.
*   **Context Generation (New NN/RL):** `core/data_preparation_nn.py` will generate a richer feature set (approx. 18 base features + embeddings) as defined in `memory-bank/feature_set_NN.md`. The `ContextAwareAttention` module is currently **not used**.
*   **Database Persistence:** Uses a file-based SQLite DB (`trading_bot.db`) with WAL mode and isolated connections for stability. Absolute path used.
*   **Configuration:** Uses `config/symbols.json` and `Credential.env`.
*   **Risk Management:** Basic stop-loss, take-profit, and dynamic position sizing rules exist in `DecisionEngine` but **require parameter alignment** (+5%/-2%).
*   **Backtesting (Planned):** A backtesting engine will be developed for offline strategy evaluation.
*   **Dynamic Adapter Loading:** `LLMHandler` automatically loads the most recent adapter from `training/adapter_runs/`.
*   **GPU Optimization:** `utils/gpu_utils.py` integrated to optimize performance based on detected hardware.

### Outdated/Removed Patterns
*   **Confidence Score Logic:** The previous system based on the LLM outputting a numerical confidence score (1-10) is **no longer used**.
*   **Context-Aware Attention Usage:** The attention mechanism for fusing features is **not currently used** in the PnL data preparation or decision pipeline. Context vector is a direct concatenation of scaled technical features and sentiment.

---

## Pattern: HPO Early Stopping Optimality

**Context**: Hyperparameter optimization with early stopping

**Observation**: HPO trials that stop "prematurely" (1-4 epochs) may actually achieve better generalization than full training (50-100 epochs)

**Explanation**:
- Optuna optimizes validation metrics, not training duration
- Early stopping prevents overfitting when train/val distribution differs
- Very low learning rates + short training = slow, stable generalization
- Extended training memorizes training-specific patterns

**When It Applies**:
- Train/val temporal split (distribution shift)
- Class imbalance problems
- Complex time series patterns

**Best Practice**:
1. Trust validation metrics over training epoch count
2. Don't assume "more training is better"
3. If HPO finds low-epoch solutions, investigate WHY (often reveals data issues)
4. Use HPO checkpoints directly, avoid "fixing" short training

**Anti-Pattern**:
- Dismissing good-performing models as "undertrained" based on epoch count alone
- Retraining HPO winners with "better" configurations that ignore validation signals

**Evidence**: Phase 3 HPO trials (epochs 1-4) outperformed full retraining (epochs 56-76) by 3-14x on validation recall
