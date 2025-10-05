import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceMetrics:
    """
    Calculates and stores various performance metrics for a backtest.
    Implements advanced validation techniques including comprehensive metrics
    and threshold optimization for SupervisedNNStrategy.
    """
    def __init__(self, initial_capital, closed_trades=None):
        self.initial_capital = initial_capital
        self.equity_curve = None
        self.closed_trades = closed_trades if closed_trades is not None else []

    def create_equity_curve_dataframe(self, equity_curve_df):
        """
        Assigns the pre-calculated equity curve DataFrame from the Portfolio.
        """
        self.equity_curve = equity_curve_df

    def _calculate_drawdowns(self):
        """
        Calculates the largest peak-to-trough drawdown of the equity curve.
        Returns max_drawdown, peak_date, trough_date.
        """
        if self.equity_curve is None or self.equity_curve.empty:
            return 0.0, None, None

        # Calculate the running maximum
        running_max = np.maximum.accumulate(self.equity_curve['equity_curve'])
        # Calculate the drawdown
        drawdown = (running_max - self.equity_curve['equity_curve']) / running_max
        
        max_drawdown = drawdown.max()
        
        # Find the peak and trough for the maximum drawdown
        if max_drawdown > 0:
            trough_index = drawdown.idxmax()
            peak_index = self.equity_curve['equity_curve'].loc[:trough_index].idxmax()
            return max_drawdown, peak_index, trough_index
        else:
            return 0.0, None, None

    def _calculate_trade_metrics(self):
        """
        Calculates comprehensive trade-level performance metrics.
        """
        total_trades = len(self.closed_trades)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'loss_rate': 0.0,
                'total_pnl': 0.0,
                'total_pnl_percentage': 0.0,
                'avg_pnl_per_trade': 0.0,
                'avg_winning_pnl': 0.0,
                'avg_losing_pnl': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'profit_factor': 0.0,
                'avg_holding_period_hours': 0.0
            }

        winning_trades = [t for t in self.closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in self.closed_trades if t['pnl'] < 0]

        num_winning_trades = len(winning_trades)
        num_losing_trades = len(losing_trades)

        win_rate = (num_winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        loss_rate = (num_losing_trades / total_trades) * 100 if total_trades > 0 else 0.0

        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        total_pnl_percentage = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0

        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0
        avg_winning_pnl = sum(t['pnl'] for t in winning_trades) / num_winning_trades if num_winning_trades > 0 else 0.0
        avg_losing_pnl = sum(t['pnl'] for t in losing_trades) / num_losing_trades if num_losing_trades > 0 else 0.0

        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))  # Absolute value of total losses

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        # Calculate average holding period
        total_holding_seconds = sum(t['holding_period'].total_seconds() for t in self.closed_trades)
        avg_holding_period_hours = (total_holding_seconds / total_trades) / 3600 if total_trades > 0 else 0.0

        return {
            'total_trades': total_trades,
            'winning_trades': num_winning_trades,
            'losing_trades': num_losing_trades,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'total_pnl': total_pnl,
            'total_pnl_percentage': total_pnl_percentage,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'avg_winning_pnl': avg_winning_pnl,
            'avg_losing_pnl': avg_losing_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_holding_period_hours': avg_holding_period_hours
        }

    def _calculate_precision_recall_f1(self, strategy_signals, true_labels):
        """
        Calculates Precision, Recall, and F1-score for BUY signals.
        
        Args:
            strategy_signals (pd.Series): Series of 1 (BUY) or 0 (NO BUY) from the strategy.
            true_labels (pd.Series): Series of 1 (True BUY) or 0 (False BUY) from actual outcomes.
        """
        if strategy_signals is None or true_labels is None or len(strategy_signals) == 0 or len(true_labels) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

        # Convert to pandas Series if they aren't already
        if not isinstance(strategy_signals, pd.Series):
            strategy_signals = pd.Series(strategy_signals)
        if not isinstance(true_labels, pd.Series):
            true_labels = pd.Series(true_labels)

        # Align indices to ensure correct comparison
        aligned_data = pd.DataFrame({'signals': strategy_signals, 'labels': true_labels}).dropna()
        
        if aligned_data.empty:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

        true_positives = ((aligned_data['signals'] == 1) & (aligned_data['labels'] == 1)).sum()
        false_positives = ((aligned_data['signals'] == 1) & (aligned_data['labels'] == 0)).sum()
        false_negatives = ((aligned_data['signals'] == 0) & (aligned_data['labels'] == 1)).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def _calculate_sharpe_ratio(self, annual_return, annual_volatility, risk_free_rate=0.0):
        """
        Calculates the Sharpe Ratio (annualized).
        
        Args:
            annual_return (float): Annualized return percentage
            annual_volatility (float): Annualized volatility percentage
            risk_free_rate (float): Risk-free rate (default 0%)
        """
        if annual_volatility == 0:
            return 0.0
        return (annual_return - risk_free_rate) / annual_volatility

    def _calculate_sortino_ratio(self, annual_return, target_return=0.0):
        """
        Calculates the Sortino Ratio (annualized).
        
        Args:
            annual_return (float): Annualized return percentage
            target_return (float): Target return (default 0%)
        """
        if self.equity_curve is None or self.equity_curve.empty:
            return 0.0

        # Calculate daily returns
        returns = self.equity_curve['returns']

        # Calculate downside returns (returns below target)
        downside_returns = returns[returns < (target_return / 252)]  # Daily target return

        if len(downside_returns) == 0:
            return float('inf')  # No downside volatility

        # Calculate downside deviation (annualized)
        downside_deviation = downside_returns.std() * np.sqrt(252)

        if downside_deviation == 0:
            return float('inf')  # Avoid division by zero

        # Sortino Ratio
        sortino_ratio = (annual_return - target_return) / (downside_deviation * 100)
        return sortino_ratio

    def _calculate_calmar_ratio(self, annual_return, max_drawdown):
        """
        Calculates the Calmar Ratio.
        
        Args:
            annual_return (float): Annualized return percentage
            max_drawdown (float): Maximum drawdown as decimal (e.g., 0.1 for 10%)
        """
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0
        return annual_return / (max_drawdown * 100)

    def optimize_signal_threshold(self, prediction_probabilities, true_labels, thresholds=None):
        """
        Determines an optimal signal_threshold for SupervisedNNStrategy based on F1-score.

        Args:
            prediction_probabilities (pd.Series or array-like): Prediction probabilities from the NN model.
            true_labels (pd.Series or array-like): True labels (1 for BUY, 0 for NO BUY).
            thresholds (list, optional): List of thresholds to test. Defaults to 0.1 to 0.9 in steps of 0.05.

        Returns:
            dict: Contains 'optimal_threshold', 'max_f1_score', and 'metrics_at_optimal_threshold'.
        """
        if prediction_probabilities is None or true_labels is None or len(prediction_probabilities) == 0 or len(true_labels) == 0:
            print("Warning: Prediction probabilities or true labels are empty. Cannot optimize threshold.")
            return {'optimal_threshold': None, 'max_f1_score': 0.0, 'metrics_at_optimal_threshold': {}}

        # Convert to pandas Series if they aren't already
        if not isinstance(prediction_probabilities, pd.Series):
            prediction_probabilities = pd.Series(prediction_probabilities)
        if not isinstance(true_labels, pd.Series):
            true_labels = pd.Series(true_labels)

        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)

        best_f1_score = -1.0
        optimal_threshold = None
        metrics_at_optimal_threshold = {}

        print("\n--- Optimizing Signal Threshold ---")
        for threshold in thresholds:
            # Simulate strategy signals based on the current threshold
            simulated_signals = (prediction_probabilities >= threshold).astype(int)
            
            # Calculate precision, recall, f1-score for the simulated signals
            current_metrics = self._calculate_precision_recall_f1(simulated_signals, true_labels)
            current_f1 = current_metrics['f1_score']

            print(f"  Threshold: {threshold:.2f}, F1-Score: {current_f1:.4f}, Precision: {current_metrics['precision']:.4f}, Recall: {current_metrics['recall']:.4f}")

            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                optimal_threshold = threshold
                metrics_at_optimal_threshold = current_metrics
        
        if optimal_threshold is not None:
            print(f"\nOptimal Threshold Found: {optimal_threshold:.2f} with F1-Score: {best_f1_score:.4f}")
        else:
            print("\nNo optimal threshold found. All F1-scores were 0.")
            
        return {
            'optimal_threshold': optimal_threshold,
            'max_f1_score': best_f1_score,
            'metrics_at_optimal_threshold': metrics_at_optimal_threshold
        }

    def output_summary_stats(self, strategy_signals=None, true_labels=None):
        """
        Outputs comprehensive summary statistics for the equity curve and trade metrics.
        Includes advanced validation metrics: Sharpe, Sortino, Calmar ratios, and signal quality metrics.
        
        Args:
            strategy_signals (pd.Series, optional): Series of strategy signals for precision/recall calculation
            true_labels (pd.Series, optional): Series of true labels for precision/recall calculation
            
        Returns:
            dict: Dictionary of all calculated metrics
        """
        metrics = {}

        if self.equity_curve is None or self.equity_curve.empty:
            print("Equity curve not generated. Run create_equity_curve_dataframe first.")
            return metrics
    
        # Equity Curve Metrics
        total_return = (self.equity_curve['equity_curve'].iloc[-1] - 1.0) * 100.0
        annual_return = (self.equity_curve['returns'].mean() * 252) * 100.0  # Assuming 252 trading days
        annual_volatility = self.equity_curve['returns'].std() * np.sqrt(252) * 100.0
        
        max_drawdown, peak_date, trough_date = self._calculate_drawdowns()
        
        # Advanced Risk-Adjusted Metrics
        sharpe_ratio = self._calculate_sharpe_ratio(annual_return, annual_volatility)
        sortino_ratio = self._calculate_sortino_ratio(annual_return)
        calmar_ratio = self._calculate_calmar_ratio(annual_return, max_drawdown)

        metrics.update({
            'initial_capital': self.initial_capital,
            'final_portfolio_value': self.equity_curve['total'].iloc[-1],
            'total_return_percent': total_return,
            'annualized_return_percent': annual_return,
            'annualized_volatility_percent': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown_percent': max_drawdown * 100
        })

        # Trade Metrics
        trade_metrics = self._calculate_trade_metrics()
        metrics.update(trade_metrics)

        # Signal Quality Metrics (Precision, Recall, F1-score)
        signal_metrics = self._calculate_precision_recall_f1(strategy_signals, true_labels)
        metrics.update({
            'signal_precision': signal_metrics['precision'],
            'signal_recall': signal_metrics['recall'],
            'signal_f1_score': signal_metrics['f1_score']
        })

        # Print comprehensive summary
        print("\n" + "=" * 60)
        print("ADVANCED VALIDATION METRICS - STRATEGY SUMMARY STATISTICS")
        print("=" * 60)
        
        print(f"\n--- Portfolio Performance ---")
        print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
        print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
        print(f"Total Return: {metrics['total_return_percent']:.2f}%")
        print(f"Annualized Return: {metrics['annualized_return_percent']:.2f}%")
        print(f"Annualized Volatility: {metrics['annualized_volatility_percent']:.2f}%")
        
        print(f"\n--- Advanced Risk-Adjusted Metrics ---")
        print(f"Sharpe Ratio (Rf=0%): {metrics['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio (Target=0%): {metrics['sortino_ratio']:.4f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown_percent']:.2f}%")
        if peak_date and trough_date:
            print(f"  (Peak: {peak_date.strftime('%Y-%m-%d')}, Trough: {trough_date.strftime('%Y-%m-%d')})")
        
        print(f"\n--- Trade Performance Metrics ---")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Loss Rate: {metrics['loss_rate']:.2f}%")
        print(f"Total PnL: ${metrics['total_pnl']:,.2f}")
        print(f"Total PnL (% of Initial Capital): {metrics['total_pnl_percentage']:.2f}%")
        print(f"Average PnL per Trade: ${metrics['avg_pnl_per_trade']:.2f}")
        print(f"Average Winning PnL: ${metrics['avg_winning_pnl']:.2f}")
        print(f"Average Losing PnL: ${metrics['avg_losing_pnl']:.2f}")
        print(f"Gross Profit: ${metrics['gross_profit']:,.2f}")
        print(f"Gross Loss: ${metrics['gross_loss']:,.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.4f}")
        print(f"Average Holding Period: {metrics['avg_holding_period_hours']:.2f} hours")

        if strategy_signals is not None and true_labels is not None:
            print(f"\n--- Signal Quality Metrics (BUY Signals) ---")
            print(f"Precision: {metrics['signal_precision']:.4f}")
            print(f"Recall: {metrics['signal_recall']:.4f}")
            print(f"F1-Score: {metrics['signal_f1_score']:.4f}")
        else:
            print(f"\n--- Signal Quality Metrics ---")
            print("Signal quality metrics not available (no strategy signals or true labels provided)")

        # Create equity curve plot
        try:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=self.equity_curve.index, y=self.equity_curve['equity_curve'])
            plt.title('Equity Curve - Advanced Validation Results')
            plt.xlabel('Date')
            plt.ylabel('Equity Value')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Note: Could not display equity curve plot: {e}")

        print("=" * 60)
        return metrics