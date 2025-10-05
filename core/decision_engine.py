# decision_engine.py
import os
import logging
import numpy as np
import torch
from models.llm_handler import LLMHandler, ContextAwareAttention
from core.hist_data_loader import HistoricalDataLoader
from core.db_models import TradeJournal
from core.news_sentiment import NewsSentimentAnalyzer
import alpaca_trade_api as tradeap
import datetime
from datetime import timezone # Import timezone directly
from typing import Dict, Optional
from utils.db_setup import SessionLocal
from utils.audit_models import AuditLog
import uuid
import json
from dotenv import load_dotenv
import pytz
from pathlib import Path
from utils.account_manager import AccountManager
import sys # Import sys

# Load environment variables using an absolute path.
env_path = Path(__file__).resolve().parent.parent / "Credential.env"
load_dotenv(dotenv_path=env_path, override=True)

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# Update risk management parameters to match data preparation settings
TAKE_PROFIT_PERCENTAGE = 5.0  # Updated to 5% to match data preparation
STOP_LOSS_PERCENTAGE = 2.0    # Updated to 2% to match data preparation
MAX_POSITION_SIZE_PERCENT = 10.0  # Keep existing or adjust as needed

class TradingSuspendedError(Exception):
    """Base class for account-related trading halts"""
    pass

class InsufficientFundsError(TradingSuspendedError):
    """Raised when balance below trading threshold"""
    pass

class ExistingPositionError(TradingSuspendedError):
    """Raised when duplicate position detected"""
    pass

class DecisionEngine:
    # --- Risk Management Configuration ---
    POSITION_SIZE_PCT = 0.01 # 1% of capital per trade
    INITIAL_CAPITAL = 2000.0 # Starting capital
    # ------------------------------------

    def __init__(self, starting_balance: Optional[float] = None):
        self.logger = logging.getLogger(__name__)
        # Use starting_balance if provided, otherwise use the class constant
        initial_balance = starting_balance if starting_balance is not None else self.INITIAL_CAPITAL
        self.account_manager = AccountManager(starting_balance=initial_balance)
        self.logger.info(f"DecisionEngine initialized. Initial Capital: ${initial_balance:.2f}")

        # Initialize components
        self.llm_handler = LLMHandler(use_lora=True)
        self.context_aggregator = ContextAwareAttention(realtime_dim=9, historical_dim=9, sentiment_dim=1) # Match features
        self.historical_loader = HistoricalDataLoader()
        self.sentiment_analyzer = NewsSentimentAnalyzer()

        # Setup Alpaca Client
        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")
        base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        if not api_key or not secret_key:
            self.logger.error("Alpaca API Key/Secret not found. Trading disabled.")
            self.alpaca_client = None
        else:
            try:
                self.alpaca_client = tradeap.REST(api_key, secret_key, base_url, api_version="v2")
                account_info = self.alpaca_client.get_account()
                self.logger.info(f"Alpaca client initialized. Account Status: {account_info.status}")
                # Sync initial capital if possible? Or rely on internal tracking for now.
                # self.account_manager.set_current_balance(float(account_info.equity))
            except Exception as e:
                self.logger.error(f"Failed to initialize Alpaca client or get account info: {e}")
                self.alpaca_client = None

    def _validate_feature_vector(self, feature_vector):
        """Validate that the feature vector has the expected dimensions and structure."""
        if feature_vector is None:
            logger.error("Feature vector is None")
            return False
            
        # Expect exactly 10 features: 9 technical indicators + 1 sentiment score
        expected_features = 10
        if len(feature_vector) != expected_features:
            logger.error(f"Feature vector has {len(feature_vector)} features, expected {expected_features}")
            return False
            
        return True

    def predict_action(self, symbol: str, timestamp, feature_vector):
        """Predict trading action based on feature vector.
        
        Args:
            symbol: The trading symbol
            timestamp: Current timestamp
            feature_vector: Vector of technical indicators and sentiment
            
        Returns:
            Tuple of (action, confidence)
        """
        # Validate feature vector first
        if not self._validate_feature_vector(feature_vector):
            self.logger.warning(f"Invalid feature vector for {symbol} at {timestamp}, defaulting to HOLD")
            return "HOLD", 0.0
        
        # Prepare input for LLM handler
        market_data = {
            "context_vector": feature_vector.tolist() if isinstance(feature_vector, np.ndarray) else list(feature_vector),
            "sentiment_score": feature_vector[-1],  # Last element is sentiment score
            "symbol": symbol
        }
        
        # Get prediction from LLM
        prediction = self.llm_handler.analyze_market(market_data)
        
        if not prediction or "decision" not in prediction:
            self.logger.warning(f"LLM prediction failed for {symbol}, defaulting to HOLD")
            return "HOLD", 0.0
        
        action = prediction.get("decision", "HOLD")
        confidence = prediction.get("confidence", 0.5)
        
        self.logger.info(f"Predicted action for {symbol}: {action} (confidence: {confidence:.2f})")
        return action, confidence

    def analyze_market(self, symbol: str) -> Optional[Dict]:
        """
        Gathers data, generates context vector, gets LLM prediction.
        Returns prediction dict or None on error.
        """
        expected_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'VWAP', 'RSI', 'volume_pct', 'returns_1h'
        ]
        self.logger.debug(f"Analyzing market for {symbol}...")

        try:
            # 1. Get Realtime Data (using latest historical as placeholder)
            realtime_dict = self._get_realtime_data(symbol)
            if not realtime_dict:
                 self.logger.warning(f"Could not get realtime data for {symbol}.")
                 return None

            # 2. Get Historical Window
            historical_df = self.historical_loader.get_window(symbol)
            if historical_df is None or historical_df.empty:
                self.logger.warning(f"Could not get historical window for {symbol}.")
                return None

            # 3. Get Sentiment
            sentiment = self.sentiment_analyzer.get_normalized_score(symbol)

            # --- Prepare Tensors ---
            # Realtime Tensor
            realtime_values = [float(realtime_dict[feature]) for feature in expected_features]
            realtime_tensor = torch.tensor(realtime_values, dtype=torch.float32).unsqueeze(0)

            # Historical Tensor Summary
            if not all(f in historical_df.columns for f in expected_features):
                missing = [f for f in expected_features if f not in historical_df.columns]
                self.logger.error(f"Missing features {missing} in historical window for {symbol}.")
                return None
            historical_features = historical_df[expected_features].values
            historical_summary = np.nanmean(historical_features, axis=0)
            historical_tensor_summary = torch.tensor(historical_summary, dtype=torch.float32).unsqueeze(0)

            # Sentiment Tensor
            sentiment_tensor = torch.tensor([[sentiment]], dtype=torch.float32) # Shape [1, 1]

            # --- Aggregate Context ---
            context_vector_tensor = self.context_aggregator(realtime_tensor, historical_tensor_summary, sentiment_tensor)
            context_vector = context_vector_tensor.detach().cpu().numpy().flatten().tolist()

            # --- Get LLM Prediction ---
            market_data_for_llm = {
                "context_vector": context_vector,
                "sentiment_score": sentiment,
                "symbol": symbol
            }
            prediction = self.llm_handler.analyze_market(market_data_for_llm)

            if not prediction or "decision" not in prediction:
                self.logger.warning(f"LLM Handler did not return a valid prediction for {symbol}.")
                return None

            # Add current price to prediction dict for later use
            prediction["current_price"] = realtime_dict.get('Close') # Assuming Close is the relevant price
            prediction["symbol"] = symbol # Ensure symbol is present
            self.logger.info(f"LLM Analysis for {symbol}: Decision={prediction['decision']}, Class={prediction.get('predicted_class', 'N/A')}")
            return prediction

        except KeyError as e:
            self.logger.error(f"Missing feature {e} during analysis for {symbol}.")
            return None
        except Exception as e:
             self.logger.error(f"Error during market analysis for {symbol}: {e}", exc_info=True)
             return None

    def evaluate_open_position(self, symbol: str, current_price: float) -> Optional[str]:
        """Checks if an open position should be closed based on SL/TP."""
        pos = self.account_manager.get_position(symbol)
        if pos is None:
            return None # No open position

        entry_price = pos.get("entry_price", 0.0) # Ensure AccountManager stores 'entry_price'
        if entry_price <= 0:
             self.logger.warning(f"Position for {symbol} has invalid entry price {entry_price}. Cannot evaluate.")
             return "HOLD" # Cannot evaluate, hold

        pnl_pct = ((current_price - entry_price) / entry_price) # Calculate as fraction

        self.logger.debug(f"Evaluating open position {symbol}: Entry={entry_price:.2f}, Current={current_price:.2f}, PnL={pnl_pct:.4f}")

        # Use configured SL/TP percentages
        if pnl_pct >= TAKE_PROFIT_PERCENTAGE / 100:
            self.logger.info(f"TP hit for {symbol} at {pnl_pct*100:.2f}% (Target: {TAKE_PROFIT_PERCENTAGE}%)")
            return "SELL" # Signal to close position
        if pnl_pct <= -STOP_LOSS_PERCENTAGE / 100:
            self.logger.info(f"SL hit for {symbol} at {pnl_pct*100:.2f}% (Target: -{STOP_LOSS_PERCENTAGE}%)")
            return "SELL" # Signal to close position

        return "HOLD" # Continue holding

    def validate_trading_readiness(self):
        """Placeholder for checks like API connection, market hours, sufficient funds etc."""
        if self.alpaca_client is None:
             raise TradingSuspendedError("Alpaca client not initialized.")
        # Add more checks as needed
        pass # Keep simple for now

    def check_existing_positions(self, symbol: str) -> bool:
        """Checks if a position for the symbol is already open."""
        return self.account_manager.has_position(symbol)

    def execute_strategy(self, symbol: str):
        """
        Main strategy execution logic:
        1. Check readiness.
        2. Check open positions for SL/TP => Close if needed.
        3. If NO position exists, analyze market for BUY signal.
        4. If BUY signal, calculate size and execute BUY order.
        """
        try:
            self.validate_trading_readiness() # Check API, funds etc.

            # --- Check Existing Position for SL/TP ---
            if self.check_existing_positions(symbol):
                realtime_data = self._get_realtime_data(symbol)
                current_price = realtime_data.get("Close") # Use Close price for evaluation
                if current_price is None:
                    self.logger.warning(f"Cannot evaluate open position for {symbol}: Missing current price.")
                    return # Skip this cycle if price unavailable

                close_decision = self.evaluate_open_position(symbol, current_price)
                if close_decision == "SELL":
                    pos_details = self.account_manager.get_position(symbol)
                    quantity_to_sell = pos_details.get("quantity", 0) # Get quantity from stored position
                    if quantity_to_sell > 0:
                        sell_prediction = {
                            "symbol": symbol,
                            "decision": "SELL", # Signal to close
                            "quantity": quantity_to_sell,
                            "reason": f"SL/TP hit ({'TP' if (current_price / pos_details.get('entry_price', 1) - 1) >= TAKE_PROFIT_PERCENTAGE / 100 else 'SL'})"
                        }
                        self._execute_order(sell_prediction, close_position=True, exit_price=current_price)
                        return # Position closed, end cycle for this symbol
                    else:
                         self.logger.warning(f"Position for {symbol} found but quantity is {quantity_to_sell}. Cannot close.")
                         # Consider removing invalid position from manager here?
                         self.account_manager.close_position(symbol, current_price, manual=True, reason="Invalid quantity found")
                         return # End cycle after attempting cleanup

                else: # Decision is HOLD
                    self.logger.info(f"Holding existing position for {symbol}.")
                    return # No further action needed if holding

            # --- Analyze for New Entry (Only if no position currently held) ---
            # This block is now only reached if check_existing_positions was False OR if a position was just closed above
            # Add explicit check again to be safe after potential close action
            if not self.check_existing_positions(symbol):
                prediction = self.analyze_market(symbol)
                if not prediction:
                    return # Analysis failed or no signal

                decision = prediction.get("decision")
                current_price = prediction.get("current_price")

                if decision == "BUY" and current_price is not None:
                    # Calculate position size
                    current_balance = self.account_manager.get_current_balance()
                    trade_value_usd = current_balance * self.POSITION_SIZE_PCT
                    # Ensure current_price is not zero before division
                    if current_price <= 0:
                         self.logger.warning(f"Invalid current price {current_price} for {symbol}. Cannot calculate quantity.")
                         return
                    quantity = int(trade_value_usd / current_price) # Use integer shares

                    self.logger.info(f"BUY Signal for {symbol}. Balance: ${current_balance:.2f}, Target Size: ${trade_value_usd:.2f}, Price: ${current_price:.2f}, Quantity: {quantity}")

                    if quantity <= 0:
                         self.logger.warning(f"Calculated quantity is {quantity} for {symbol}. Cannot place BUY order.")
                         return

                    # Basic check if trade value exceeds balance (adjust for margin later if needed)
                    if trade_value_usd > current_balance:
                        self.logger.warning(f"Insufficient funds for {symbol} BUY. Need ${trade_value_usd:.2f}, have ${current_balance:.2f}")
                        # raise InsufficientFundsError(f"Insufficient funds for {symbol} BUY.") # Option to raise
                        return # Skip trade if not enough funds

                    # Prepare and execute BUY order
                    buy_prediction = {
                        "symbol": symbol,
                        "decision": "BUY",
                        "quantity": quantity,
                        "estimated_value": trade_value_usd,
                        "predicted_class": prediction.get("predicted_class") # Log class if available
                    }
                    self._execute_order(buy_prediction, close_position=False, entry_price=current_price) # Pass entry price

                elif decision in ["SELL", "HOLD"]:
                    self.logger.info(f"{decision} decision for {symbol} with no open position: No action taken.")
                else:
                    self.logger.warning(f"Unknown decision '{decision}' for symbol {symbol}")

        except TradingSuspendedError as tse:
             self.logger.warning(f"Trading suspended for {symbol}: {tse}")
        except Exception as e:
            self.logger.error(f"Strategy execution failed for {symbol}: {e}", exc_info=True)


    def _execute_order(self, prediction: Dict, close_position: bool = False, entry_price: Optional[float] = None, exit_price: Optional[float] = None):
        """
        Places order via Alpaca, logs to DB, updates AccountManager.
        Requires entry_price for BUY orders to store in AccountManager.
        Requires exit_price for SELL orders to calculate PnL.
        """
        symbol = prediction.get("symbol")
        decision = prediction.get("decision", "").upper()
        quantity = prediction.get("quantity", 0)

        if not symbol or not decision or quantity <= 0:
            self.logger.error(f"Invalid prediction for order execution: {prediction}")
            return

        if self.alpaca_client is None:
            self.logger.error("Alpaca client not available. Cannot execute order.")
            # Simulate execution for testing? Or raise error?
            # For now, just log and return
            return

        order_side = "buy" if decision == "BUY" else "sell"
        log_action = f"MANUAL_{decision}" if prediction.get("manual") else decision # For logging

        try:
            # --- Place Order ---
            order = self.alpaca_client.submit_order(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                type="market",
                time_in_force="day"
            )
            self.logger.info(f"Submitted {order_side} order for {quantity} shares of {symbol}. Order ID: {order.id}")
            # TODO: Add logic to wait for order fill confirmation? For now, assume filled at approx current price.
            filled_price = float(order.filled_avg_price) if order.filled_avg_price else (exit_price if close_position else entry_price)
            if filled_price is None:
                 self.logger.error(f"Could not determine filled price for order {order.id}. Using 0.0 for logging.")
                 filled_price = 0.0

            # --- Update Account Manager ---
            pnl = 0.0
            if close_position:
                # Use the provided exit_price for PnL calculation
                pnl = self.account_manager.close_position(symbol, exit_price if exit_price else filled_price, manual=prediction.get("manual", False), reason=prediction.get("reason"))
            else: # Opening a position
                if entry_price is None:
                     self.logger.error(f"Entry price is required for BUY order {order.id} to update AccountManager. Using filled price as fallback.")
                     entry_price = filled_price
                self.account_manager.add_position({
                    "trade_id": str(uuid.uuid4()), # Generate a unique ID for the position tracking
                    "symbol": symbol,
                    "entry_price": entry_price, # Store the price at decision time
                    "quantity": quantity,
                    "timestamp": datetime.now(timezone.utc) # Store entry time
                })

            # --- Log to TradeJournal ---
            session = SessionLocal()
            try:
                metadata_str = json.dumps(prediction) # Log the original prediction/reasoning
                trade_record = TradeJournal(
                    id=str(uuid.uuid4()), # Unique ID for this specific journal entry
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    action=log_action, # Log BUY/SELL or MANUAL_BUY/MANUAL_SELL
                    quantity=quantity,
                    paper_price=filled_price, # Log the execution price
                    real_price=None, # Placeholder for real execution if different
                    status="closed" if close_position else "open",
                    pnl_unrealized=0.0, # Reset unrealized PnL on open/close
                    pnl_realized=pnl if close_position else 0.0, # Log realized PnL only on close
                    model_metadata=metadata_str,
                    checksum="N/A" # Placeholder
                )
                session.add(trade_record)
                session.commit()
                self.logger.info(f"Trade action {log_action} for {symbol} recorded in TradeJournal.")
            except Exception as db_e:
                 self.logger.error(f"Failed to log trade to database: {db_e}", exc_info=True)
                 session.rollback()
            finally:
                 session.close()

            # --- Add Alert ---
            alert = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "action": log_action,
                "quantity": quantity,
                "price": filled_price,
                "reason": prediction.get("reason", "")
            }
            self.account_manager.add_alert(alert)
            self.logger.info(f"Order execution process complete for {symbol} {log_action}.")

        except TradingSuspendedError as tse:
             self.logger.warning(f"Order execution suspended for {symbol}: {tse}")
             # Don't re-raise, just log and stop this attempt
        except Exception as e:
            self.logger.error(f"Error executing order for {symbol}: {e}", exc_info=True)
            # Don't re-raise here to allow main loop to continue? Or re-raise?
            # For now, just log the error.

    def _get_realtime_data(self, symbol: str) -> Optional[Dict]:
        """Placeholder: Fetches latest data. Replace with actual implementation."""
        # In a real system, this would connect to a streaming source or query latest bar
        # For simulation/testing, we might fetch the latest record from historical data
        self.logger.debug(f"Fetching 'realtime' data for {symbol} (using latest historical for now)")
        try:
            # Ensure historical loader is ready
            if not hasattr(self, 'historical_loader') or self.historical_loader is None:
                 self.historical_loader = HistoricalDataLoader()

            df = self.historical_loader.get_window(symbol, window_size=1) # Get only the latest row
            if df is None or df.empty:
                self.logger.warning(f"No historical data found for {symbol} to use as realtime.")
                return None

            latest_data = df.iloc[-1].to_dict()
            # Ensure keys match expected features (Open, High, Low, Close, Volume, VWAP, RSI, etc.)
            # Rename 'wap' if needed:
            if 'wap' in latest_data:
                 latest_data['VWAP'] = latest_data.pop('wap')
            # Add any missing keys with default values (e.g., 0 or NaN) if necessary
            expected_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'VWAP', 'RSI', 'volume_pct', 'returns_1h'
            ]
            for key in expected_features:
                if key not in latest_data:
                     latest_data[key] = 0.0 # Or np.nan
            return latest_data
        except Exception as e:
            self.logger.error(f"Error fetching latest historical data as 'realtime' for {symbol}: {e}", exc_info=True)
            return None

    def refresh_open_positions(self) -> Dict[str, Dict]:
        """Refreshes internal position state from Alpaca."""
        if self.alpaca_client is None: return {}
        try:
            alpaca_positions = self.alpaca_client.list_positions()
            current_positions = {}
            for pos in alpaca_positions:
                # Store necessary info for SL/TP checks
                current_positions[pos.symbol] = {
                    "entry_price": float(pos.avg_entry_price),
                    "quantity": float(pos.qty),
                    # Add other relevant details if needed, like entry time from Alpaca if available
                }
            self.account_manager.sync_positions(current_positions) # Sync with AccountManager
            self.logger.info(f"Refreshed open positions from Alpaca: {list(current_positions.keys())}")
            return current_positions
        except Exception as e:
            self.logger.error(f"Error refreshing Alpaca positions: {e}")
            return self.account_manager.get_all_positions() # Return potentially stale internal state

    def manual_close_position(self, symbol: str, exit_price: Optional[float] = None, reason: str = "Manual Intervention"):
        """Manually close an open position."""
        pos_details = self.account_manager.get_position(symbol)
        if not pos_details:
            self.logger.warning(f"No internal record of an open position for {symbol} to manually close.")
            # Optionally try to close via Alpaca anyway?
            # For now, only close if internally tracked.
            return

        quantity_to_sell = pos_details.get("quantity", 0)
        if quantity_to_sell <= 0:
            self.logger.warning(f"Internal position record for {symbol} has invalid quantity {quantity_to_sell}. Cannot close.")
            return

        # Get current price if exit_price not provided
        if exit_price is None:
             realtime_data = self._get_realtime_data(symbol)
             if realtime_data:
                 exit_price = realtime_data.get("Close")
             if exit_price is None:
                  self.logger.error(f"Cannot manually close {symbol}: Failed to get current price.")
                  return

        sell_prediction = {
            "symbol": symbol,
            "decision": "SELL",
            "quantity": quantity_to_sell,
            "reason": reason,
            "manual": True # Flag as manual close
        }
        try:
            self._execute_order(sell_prediction, close_position=True, exit_price=exit_price)
        except Exception as e:
             self.logger.error(f"Manual close failed for {symbol}: {e}")


    def get_metrics(self):
        """Returns current performance metrics."""
        # Example: Return balance and number of open positions
        return {
            "current_balance": self.account_manager.get_current_balance(),
            "open_positions_count": self.account_manager.get_open_position_count(),
            "total_pnl": self.account_manager.get_total_pnl(),
             # Add lora status check
            "lora_status": {"status": "Loaded" if self.llm_handler.adapter_loaded else "Not Loaded", "healthy": self.llm_handler.adapter_loaded}
        }

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Initializing Decision Engine for test...")
    de = DecisionEngine(starting_balance=2000.0) # Use $2k initial capital

    # --- Test Scenario ---
    test_symbol = "AAPL" # Example symbol

    # 1. Refresh positions (should be empty initially)
    print("\n--- Initial State ---")
    de.refresh_open_positions() # Sync with Alpaca (likely empty in paper)
    print(f"Account Manager State: {de.account_manager.get_all_positions()}")
    print(f"Current Balance: ${de.account_manager.get_current_balance():.2f}")

    # 2. Simulate receiving a BUY signal
    print(f"\n--- Simulating BUY for {test_symbol} ---")
    # Manually create a prediction dict as if from LLM
    # Need a realistic current price for quantity calculation
    rt_data = de._get_realtime_data(test_symbol)
    if rt_data and rt_data.get("Close"):
        current_test_price = rt_data["Close"]
        print(f"Using current price: ${current_test_price:.2f}")

        # Simulate BUY using execute_strategy logic
        de.execute_strategy(test_symbol) # Let the strategy handle the BUY if signal matches

        print(f"Account Manager State after BUY attempt: {de.account_manager.get_all_positions()}")
        print(f"Current Balance after BUY attempt (approx): ${de.account_manager.get_current_balance():.2f}")

        # 3. Simulate price moving up (TP hit)
        print(f"\n--- Simulating TP Hit for {test_symbol} ---")
        # Check if position was actually opened
        pos_after_buy = de.account_manager.get_position(test_symbol)
        if pos_after_buy:
            entry_price_sim = pos_after_buy.get("entry_price")
            if entry_price_sim:
                tp_price = entry_price_sim * (1 + TAKE_PROFIT_PERCENTAGE / 100)
                print(f"Simulating price reaching TP: ${tp_price:.2f}")
                # Manually trigger evaluation and potential close
                de.execute_strategy(test_symbol) # This should now evaluate and close if TP hit
            else:
                print("Position opened but entry price missing, cannot simulate TP.")
        else:
            print("No position was opened, skipping TP simulation.")


        print(f"Account Manager State after TP attempt: {de.account_manager.get_all_positions()}")
        print(f"Current Balance after TP attempt: ${de.account_manager.get_current_balance():.2f}")

        # 4. Simulate another BUY
        print(f"\n--- Simulating second BUY for {test_symbol} ---")
        de.execute_strategy(test_symbol) # Attempt BUY again

        print(f"Account Manager State after BUY 2 attempt: {de.account_manager.get_all_positions()}")
        print(f"Current Balance after BUY 2 attempt (approx): ${de.account_manager.get_current_balance():.2f}")

        # 5. Simulate price moving down (SL hit)
        print(f"\n--- Simulating SL Hit for {test_symbol} ---")
        pos_after_buy2 = de.account_manager.get_position(test_symbol)
        if pos_after_buy2:
            entry_price_sim2 = pos_after_buy2.get("entry_price")
            if entry_price_sim2:
                 sl_price = entry_price_sim2 * (1 - STOP_LOSS_PERCENTAGE / 100) # SL is negative
                 print(f"Simulating price reaching SL: ${sl_price:.2f}")
                 # Manually trigger evaluation and potential close
                 de.execute_strategy(test_symbol) # This should evaluate and close if SL hit
            else:
                 print("Position opened but entry price missing, cannot simulate SL.")
        else:
             print("No position was opened for second BUY, skipping SL simulation.")


        print(f"Account Manager State after SL attempt: {de.account_manager.get_all_positions()}")
        print(f"Current Balance after SL attempt: ${de.account_manager.get_current_balance():.2f}")

    else:
        print(f"Could not get initial realtime data for {test_symbol} to run test scenario.")