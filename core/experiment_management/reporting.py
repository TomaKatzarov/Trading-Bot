"""
Reporting and Visualization Enhancements for Experiment Analysis

This module provides automated plotting, summary reports, and comprehensive
analysis capabilities for neural network training experiments.

Part of Activity 6.4: Reporting and Visualization Enhancements
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class ExperimentReporter:
    """
    Comprehensive experiment reporting and visualization system.
    
    Provides automated plotting, summary reports, and analysis capabilities
    for neural network training experiments and HPO studies.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize experiment reporter.
        
        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotly for offline use
        pio.renderers.default = "browser"
        
    def generate_training_report(self, run_id: str, 
                               save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive training report for a single run.
        
        Args:
            run_id: MLflow run ID
            save_artifacts: Whether to save report artifacts to MLflow
            
        Returns:
            Report summary dictionary
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available for report generation")
        
        try:
            # Get run data
            run = mlflow.get_run(run_id)
            metrics = run.data.metrics
            params = run.data.params
            tags = run.data.tags
            
            # Get run history for metrics
            client = mlflow.tracking.MlflowClient()
            metric_history = {}
            for metric_name in metrics.keys():
                history = client.get_metric_history(run_id, metric_name)
                metric_history[metric_name] = [(h.step, h.value) for h in history]
            
            # Generate report components
            report = {
                'run_id': run_id,
                'run_name': run.info.run_name,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                'end_time': datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                'duration': None,
                'final_metrics': metrics,
                'parameters': params,
                'tags': tags,
                'metric_history': metric_history
            }
            
            # Calculate duration
            if report['end_time']:
                report['duration'] = report['end_time'] - report['start_time']
            
            # Generate visualizations
            viz_paths = self._create_training_visualizations(report)
            report['visualizations'] = viz_paths
            
            # Generate summary statistics
            report['summary'] = self._generate_training_summary(report)
            
            # Save report
            report_path = self._save_training_report(report)
            report['report_path'] = report_path
            
            # Upload artifacts to MLflow if requested
            if save_artifacts:
                self._upload_training_artifacts(run_id, report, viz_paths)
            
            logger.info(f"Generated training report for run {run_id}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate training report: {e}")
            raise
    
    def generate_hpo_study_report(self, experiment_id: str, 
                                parent_run_id: Optional[str] = None,
                                save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive HPO study report.
        
        Args:
            experiment_id: MLflow experiment ID
            parent_run_id: Optional parent run ID for HPO study
            save_artifacts: Whether to save report artifacts to MLflow
            
        Returns:
            HPO study report dictionary
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available for HPO report generation")
        
        try:
            # Get all runs in experiment
            runs_df = mlflow.search_runs(experiment_ids=[experiment_id])
            
            # Filter HPO trial runs
            if parent_run_id:
                hpo_runs = runs_df[runs_df['tags.parent_run_id'] == parent_run_id]
            else:
                hpo_runs = runs_df[runs_df['tags.is_hpo_child'] == 'true']
            
            if hpo_runs.empty:
                raise ValueError("No HPO trial runs found")
            
            # Analyze HPO study
            report = {
                'experiment_id': experiment_id,
                'parent_run_id': parent_run_id,
                'study_name': hpo_runs['tags.study_name'].iloc[0] if 'tags.study_name' in hpo_runs.columns else 'Unknown',
                'total_trials': len(hpo_runs),
                'completed_trials': len(hpo_runs[hpo_runs['status'] == 'FINISHED']),
                'failed_trials': len(hpo_runs[hpo_runs['status'] == 'FAILED']),
                'best_trial': None,
                'parameter_importance': {},
                'convergence_analysis': {},
                'trial_summaries': []
            }
            
            # Find best trial
            if not hpo_runs.empty:
                # Assume optimization metric is validation F1 score
                metric_col = 'metrics.val_f1'
                if metric_col in hpo_runs.columns:
                    best_idx = hpo_runs[metric_col].idxmax()
                    best_trial = hpo_runs.loc[best_idx]
                    report['best_trial'] = {
                        'run_id': best_trial['run_id'],
                        'run_name': best_trial['run_name'],
                        'val_f1': best_trial[metric_col],
                        'parameters': self._extract_trial_params(best_trial)
                    }
            
            # Analyze parameter importance
            report['parameter_importance'] = self._analyze_parameter_importance(hpo_runs)
            
            # Analyze convergence
            report['convergence_analysis'] = self._analyze_hpo_convergence(hpo_runs)
            
            # Create trial summaries
            for _, trial in hpo_runs.iterrows():
                trial_summary = {
                    'run_id': trial['run_id'],
                    'run_name': trial['run_name'],
                    'status': trial['status'],
                    'trial_number': trial.get('tags.trial_number', ''),
                    'parameters': self._extract_trial_params(trial),
                    'final_metrics': self._extract_trial_metrics(trial)
                }
                report['trial_summaries'].append(trial_summary)
            
            # Generate visualizations
            viz_paths = self._create_hpo_visualizations(report, hpo_runs)
            report['visualizations'] = viz_paths
            
            # Save report
            report_path = self._save_hpo_report(report)
            report['report_path'] = report_path
            
            # Upload artifacts to MLflow if requested
            if save_artifacts and parent_run_id:
                self._upload_hpo_artifacts(parent_run_id, report, viz_paths)
            
            logger.info(f"Generated HPO study report for experiment {experiment_id}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate HPO study report: {e}")
            raise
    
    def generate_comparison_report(self, run_ids: List[str],
                                 comparison_name: str = "Model Comparison",
                                 save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Generate model comparison report.
        
        Args:
            run_ids: List of MLflow run IDs to compare
            comparison_name: Name for the comparison report
            save_artifacts: Whether to save report artifacts
            
        Returns:
            Comparison report dictionary
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available for comparison report")
        
        try:
            runs_data = []
            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                runs_data.append({
                    'run_id': run_id,
                    'run_name': run.info.run_name,
                    'experiment_id': run.info.experiment_id,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'tags': run.data.tags
                })
            
            # Create comparison analysis
            report = {
                'comparison_name': comparison_name,
                'runs': runs_data,
                'metrics_comparison': self._compare_metrics(runs_data),
                'parameter_comparison': self._compare_parameters(runs_data),
                'performance_ranking': self._rank_performance(runs_data),
                'statistical_analysis': self._statistical_comparison(runs_data)
            }
            
            # Generate visualizations
            viz_paths = self._create_comparison_visualizations(report)
            report['visualizations'] = viz_paths
            
            # Save report
            report_path = self._save_comparison_report(report)
            report['report_path'] = report_path
            
            logger.info(f"Generated comparison report for {len(run_ids)} runs")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            raise
    
    def generate_backtest_report(self, metrics: Dict[str, Any], 
                                results: Dict[str, Any],
                                strategy_config: Dict[str, Any],
                                report_name: str = "Backtest Report",
                                save_artifacts: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive backtesting report.
        
        Implements Activity 7.4: Backtesting Metrics and Reporting.
        Creates detailed analysis of backtest results including equity curves,
        trade summaries, and performance metrics.
        
        Args:
            metrics: Dictionary of backtesting performance metrics
            results: Raw backtest results including trades and equity curve
            strategy_config: Strategy configuration used for backtest
            report_name: Name for the report
            save_artifacts: Whether to save report artifacts
            
        Returns:
            Comprehensive backtest report dictionary
        """
        try:
            logger.info(f"Generating backtest report: {report_name}")
            
            # Create report structure
            report = {
                'report_name': report_name,
                'timestamp': datetime.now().isoformat(),
                'strategy_type': 'SupervisedNN',
                'strategy_config': strategy_config,
                'performance_metrics': metrics,
                'trade_summary': self._create_trade_summary(results.get('closed_trades', [])),
                'visualizations': []
            }
            
            # Generate visualizations
            viz_paths = self._create_backtest_visualizations(report, results)
            report['visualizations'] = viz_paths
            
            # Generate HTML report
            html_report = self._generate_backtest_html_report(report)
            report['html_report'] = html_report
            
            # Save report
            report_path = self._save_backtest_report(report)
            report['report_path'] = report_path
            
            # Upload artifacts to MLflow if requested
            if save_artifacts and MLFLOW_AVAILABLE and mlflow.active_run():
                self._upload_backtest_artifacts(report, viz_paths, html_report)
            
            logger.info(f"Generated backtest report: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate backtest report: {e}")
            raise
    
    def _create_trade_summary(self, closed_trades: List[Dict]) -> Dict[str, Any]:
        """Create summary statistics for closed trades."""
        if not closed_trades:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'unprofitable_trades': 0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_duration_hours': 0.0
            }
        
        profits = [t['pnl'] for t in closed_trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in closed_trades if t['pnl'] < 0]
        durations = [t['holding_period'].total_seconds() / 3600 for t in closed_trades]
        
        return {
            'total_trades': len(closed_trades),
            'profitable_trades': len(profits),
            'unprofitable_trades': len(losses),
            'avg_profit': np.mean(profits) if profits else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'largest_win': max(profits) if profits else 0.0,
            'largest_loss': min(losses) if losses else 0.0,
            'avg_duration_hours': np.mean(durations) if durations else 0.0
        }
    
    def _create_backtest_visualizations(self, report: Dict[str, Any], 
                                       results: Dict[str, Any]) -> List[str]:
        """Create visualizations for backtest report."""
        viz_paths = []
        
        try:
            # Equity curve plot
            if 'equity_curve' in results and results['equity_curve'] is not None:
                equity_path = self._plot_equity_curve(
                    results['equity_curve'], report['report_name']
                )
                viz_paths.append(equity_path)
            
            # Drawdown plot
            if 'equity_curve' in results and results['equity_curve'] is not None:
                drawdown_path = self._plot_drawdown_curve(
                    results['equity_curve'], report['report_name']
                )
                viz_paths.append(drawdown_path)
            
            # Trade distribution plot
            if results.get('closed_trades'):
                trade_dist_path = self._plot_trade_distribution(
                    results['closed_trades'], report['report_name']
                )
                viz_paths.append(trade_dist_path)
            
            # Monthly returns heatmap
            if 'equity_curve' in results and results['equity_curve'] is not None:
                monthly_returns_path = self._plot_monthly_returns(
                    results['equity_curve'], report['report_name']
                )
                viz_paths.append(monthly_returns_path)
            
        except Exception as e:
            logger.warning(f"Failed to create some backtest visualizations: {e}")
        
        return viz_paths
    
    def _plot_equity_curve(self, equity_df: pd.DataFrame, report_name: str) -> str:
        """Plot equity curve over time."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        equity_df['equity_curve'].plot(ax=ax, linewidth=2, color='#2E86AB')
        ax.fill_between(equity_df.index, equity_df['equity_curve'], 
                        alpha=0.3, color='#2E86AB')
        
        ax.set_title(f'Equity Curve - {report_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(['Equity Curve'], loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = str(self.output_dir / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_drawdown_curve(self, equity_df: pd.DataFrame, report_name: str) -> str:
        """Plot drawdown curve over time."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Calculate drawdown
        running_max = equity_df['equity_curve'].cummax()
        drawdown = (running_max - equity_df['equity_curve']) / running_max * 100
        
        ax.fill_between(equity_df.index, 0, drawdown, alpha=0.5, color='#A23B72')
        ax.plot(equity_df.index, drawdown, linewidth=2, color='#A23B72')
        
        ax.set_title(f'Drawdown - {report_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = str(self.output_dir / f"drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_trade_distribution(self, closed_trades: List[Dict], report_name: str) -> str:
        """Plot distribution of trade PnL."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        pnls = [t['pnl'] for t in closed_trades]
        
        # Histogram
        ax1.hist(pnls, bins=30, alpha=0.7, color='#F18F01', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.set_title('Trade PnL Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('PnL ($)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(pnls, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#C73E1D', alpha=0.7))
        ax2.axhline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Trade PnL Box Plot', fontsize=12, fontweight='bold')
        ax2.set_ylabel('PnL ($)', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Trade Analysis - {report_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save plot
        plot_path = str(self.output_dir / f"trade_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_monthly_returns(self, equity_df: pd.DataFrame, report_name: str) -> str:
        """Plot monthly returns as heatmap."""
        # Calculate monthly returns
        monthly_returns = equity_df['returns'].resample('M').sum() * 100
        
        if len(monthly_returns) == 0:
            return None
        
        # Create year-month matrix
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_pivot = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).sum().unstack(fill_value=0)
        
        if monthly_pivot.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(monthly_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, ax=ax, cbar_kws={'label': 'Return (%)'})
        
        ax.set_title(f'Monthly Returns Heatmap - {report_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels[:len(monthly_pivot.columns)])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = str(self.output_dir / f"monthly_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _generate_backtest_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report for backtest results."""
        metrics = report.get('performance_metrics', {})
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report['report_name']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2E86AB;
            color: white;
            padding: 20px;
            border-radius: 5px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2E86AB;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #2E86AB;
            color: white;
        }}
        .viz-container {{
            margin: 20px 0;
        }}
        .viz-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report['report_name']}</h1>
        <p>Generated: {report['timestamp']}</p>
        <p>Strategy: {report['strategy_type']}</p>
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">
                    {metrics.get('total_return', 0):.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{metrics.get('sharpe_ratio', 0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{metrics.get('max_drawdown', 0)*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{metrics.get('win_rate', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{metrics.get('profit_factor', 0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{metrics.get('total_trades', 0)}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Risk-Adjusted Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td>{metrics.get('sortino_ratio', 0):.4f}</td>
            </tr>
            <tr>
                <td>Calmar Ratio</td>
                <td>{metrics.get('calmar_ratio', 0):.4f}</td>
            </tr>
            <tr>
                <td>Annualized Return</td>
                <td>{metrics.get('annual_return', 0):.2f}%</td>
            </tr>
            <tr>
                <td>Annualized Volatility</td>
                <td>{metrics.get('annual_volatility', 0):.2f}%</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Trade Statistics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Winning Trades</div>
                <div class="metric-value positive">{metrics.get('winning_trades', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Losing Trades</div>
                <div class="metric-value negative">{metrics.get('losing_trades', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Win</div>
                <div class="metric-value positive">${report['trade_summary'].get('avg_profit', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Loss</div>
                <div class="metric-value negative">${report['trade_summary'].get('avg_loss', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Largest Win</div>
                <div class="metric-value">${report['trade_summary'].get('largest_win', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Largest Loss</div>
                <div class="metric-value">${report['trade_summary'].get('largest_loss', 0):.2f}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Strategy Configuration</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
"""
        
        # Add strategy config rows
        for key, value in report['strategy_config'].items():
            html_content += f"""
            <tr>
                <td>{key}</td>
                <td>{value}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
"""
        
        # Add visualization images
        for viz_path in report.get('visualizations', []):
            if Path(viz_path).exists():
                html_content += f"""
        <div class="viz-container">
            <img src="{Path(viz_path).name}" alt="Visualization">
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        return html_content
    
    def _save_backtest_report(self, report: Dict[str, Any]) -> str:
        """Save backtest report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"backtest_report_{timestamp}.json"
        report_path = self.output_dir / report_filename
        
        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save HTML report
        html_filename = f"backtest_report_{timestamp}.html"
        html_path = self.output_dir / html_filename
        with open(html_path, 'w') as f:
            f.write(report['html_report'])
        
        logger.info(f"Backtest report saved to: {report_path}")
        logger.info(f"HTML report saved to: {html_path}")
        
        return str(report_path)
    
    def _upload_backtest_artifacts(self, report: Dict[str, Any], 
                                   viz_paths: List[str], html_report: str):
        """Upload backtest report artifacts to MLflow."""
        try:
            # Log metrics as MLflow metrics
            for metric_name, metric_value in report['performance_metrics'].items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"backtest_{metric_name}", metric_value)
            
            # Log visualizations
            for viz_path in viz_paths:
                if Path(viz_path).exists():
                    mlflow.log_artifact(viz_path, "backtest_visualizations")
            
            # Log HTML report
            html_path = self.output_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(html_path, 'w') as f:
                f.write(html_report)
            mlflow.log_artifact(str(html_path), "backtest_reports")
            
            # Log JSON report
            if 'report_path' in report:
                mlflow.log_artifact(report['report_path'], "backtest_reports")
            
            logger.info("Backtest artifacts uploaded to MLflow")
            
        except Exception as e:
            logger.warning(f"Failed to upload some backtest artifacts to MLflow: {e}")
    
    def _create_training_visualizations(self, report: Dict[str, Any]) -> List[str]:
        """Create visualizations for training report."""
        viz_paths = []
        
        try:
            # Learning curves
            if report['metric_history']:
                learning_curves_path = self._plot_learning_curves(
                    report['metric_history'], report['run_name']
                )
                viz_paths.append(learning_curves_path)
            
            # Metrics summary
            if report['final_metrics']:
                metrics_summary_path = self._plot_metrics_summary(
                    report['final_metrics'], report['run_name']
                )
                viz_paths.append(metrics_summary_path)
            
            # Training timeline
            timeline_path = self._plot_training_timeline(report)
            if timeline_path:
                viz_paths.append(timeline_path)
            
        except Exception as e:
            logger.warning(f"Failed to create some training visualizations: {e}")
        
        return viz_paths
    
    def _create_hpo_visualizations(self, report: Dict[str, Any], 
                                 hpo_runs: pd.DataFrame) -> List[str]:
        """Create visualizations for HPO study report."""
        viz_paths = []
        
        try:
            # Optimization history
            opt_history_path = self._plot_optimization_history(hpo_runs, report['study_name'])
            if opt_history_path:
                viz_paths.append(opt_history_path)
            
            # Parameter importance
            if report['parameter_importance']:
                param_importance_path = self._plot_parameter_importance(
                    report['parameter_importance'], report['study_name']
                )
                viz_paths.append(param_importance_path)
            
            # Parallel coordinates plot
            parallel_coords_path = self._plot_parallel_coordinates(hpo_runs, report['study_name'])
            if parallel_coords_path:
                viz_paths.append(parallel_coords_path)
            
            # Performance distribution
            perf_dist_path = self._plot_performance_distribution(hpo_runs, report['study_name'])
            if perf_dist_path:
                viz_paths.append(perf_dist_path)
            
            # Trial convergence
            convergence_path = self._plot_trial_convergence(hpo_runs, report['study_name'])
            if convergence_path:
                viz_paths.append(convergence_path)
            
        except Exception as e:
            logger.warning(f"Failed to create some HPO visualizations: {e}")
        
        return viz_paths
    
    def _create_comparison_visualizations(self, report: Dict[str, Any]) -> List[str]:
        """Create visualizations for comparison report."""
        viz_paths = []
        
        try:
            # Metrics comparison bar chart
            metrics_comparison_path = self._plot_metrics_comparison(
                report['metrics_comparison'], report['comparison_name']
            )
            if metrics_comparison_path:
                viz_paths.append(metrics_comparison_path)
            
            # Performance radar chart
            radar_path = self._plot_performance_radar(
                report['runs'], report['comparison_name']
            )
            if radar_path:
                viz_paths.append(radar_path)
            
            # Parameter comparison heatmap
            if report['parameter_comparison']:
                param_heatmap_path = self._plot_parameter_heatmap(
                    report['parameter_comparison'], report['comparison_name']
                )
                viz_paths.append(param_heatmap_path)
            
        except Exception as e:
            logger.warning(f"Failed to create some comparison visualizations: {e}")
        
        return viz_paths
    
    def _plot_learning_curves(self, metric_history: Dict[str, List[Tuple[int, float]]], 
                            run_name: str) -> str:
        """Plot learning curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, (metric_name, history) in enumerate(metric_history.items()):
            if i >= len(axes):
                break
            
            if history:
                steps, values = zip(*history)
                axes[i].plot(steps, values, marker='o', linewidth=2, markersize=4)
                axes[i].set_title(f'{metric_name.title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric_name.replace('_', ' ').title())
                axes[i].grid(True, alpha=0.3)
                axes[i].legend([metric_name])
        
        # Hide unused subplots
        for i in range(len(metric_history), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Learning Curves - {run_name}', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        filename = f"learning_curves_{run_name.replace(' ', '_')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_optimization_history(self, hpo_runs: pd.DataFrame, study_name: str) -> Optional[str]:
        """Plot HPO optimization history."""
        try:
            metric_col = 'metrics.val_f1'
            if metric_col not in hpo_runs.columns:
                return None
            
            # Sort by trial number
            if 'tags.trial_number' in hpo_runs.columns:
                hpo_runs_sorted = hpo_runs.sort_values('tags.trial_number')
            else:
                hpo_runs_sorted = hpo_runs.sort_values('start_time')
            
            # Create optimization history plot
            fig = go.Figure()
            
            # Add trial values
            fig.add_trace(go.Scatter(
                x=list(range(len(hpo_runs_sorted))),
                y=hpo_runs_sorted[metric_col].values,
                mode='markers+lines',
                name='Trial Values',
                marker=dict(size=8, color='blue', opacity=0.7),
                line=dict(color='lightblue', width=1)
            ))
            
            # Add best value so far
            best_so_far = hpo_runs_sorted[metric_col].cummax()
            fig.add_trace(go.Scatter(
                x=list(range(len(hpo_runs_sorted))),
                y=best_so_far.values,
                mode='lines',
                name='Best Value So Far',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title=f'Optimization History - {study_name}',
                xaxis_title='Trial Number',
                yaxis_title='Validation F1 Score',
                showlegend=True,
                template='plotly_white'
            )
            
            # Save as HTML
            filename = f"optimization_history_{study_name.replace(' ', '_')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            
            return str(filepath)
            
        except Exception as e:
            logger.warning(f"Failed to create optimization history plot: {e}")
            return None
    
    def _plot_parameter_importance(self, param_importance: Dict[str, float], 
                                 study_name: str) -> str:
        """Plot parameter importance."""
        if not param_importance:
            return ""
        
        params = list(param_importance.keys())
        importance = list(param_importance.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        params = [params[i] for i in sorted_indices]
        importance = [importance[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(params, importance, color='skyblue', alpha=0.8)
        
        # Add value labels on bars
        for bar, imp in zip(bars, importance):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{imp:.3f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Parameter Importance - {study_name}')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"parameter_importance_{study_name.replace(' ', '_')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _analyze_parameter_importance(self, hpo_runs: pd.DataFrame) -> Dict[str, float]:
        """Analyze parameter importance using correlation with target metric."""
        try:
            metric_col = 'metrics.val_f1'
            if metric_col not in hpo_runs.columns:
                return {}
            
            # Get parameter columns
            param_cols = [col for col in hpo_runs.columns if col.startswith('params.')]
            
            importance = {}
            for param_col in param_cols:
                try:
                    # Convert to numeric if possible
                    param_values = pd.to_numeric(hpo_runs[param_col], errors='coerce')
                    metric_values = hpo_runs[metric_col]
                    
                    # Calculate correlation
                    correlation = param_values.corr(metric_values)
                    if not np.isnan(correlation):
                        param_name = param_col.replace('params.', '')
                        importance[param_name] = abs(correlation)
                        
                except Exception:
                    continue
            
            return importance
            
        except Exception as e:
            logger.warning(f"Failed to analyze parameter importance: {e}")
            return {}
    
    def _analyze_hpo_convergence(self, hpo_runs: pd.DataFrame) -> Dict[str, Any]:
        """Analyze HPO convergence characteristics."""
        try:
            metric_col = 'metrics.val_f1'
            if metric_col not in hpo_runs.columns:
                return {}
            
            # Sort by trial number or start time
            if 'tags.trial_number' in hpo_runs.columns:
                hpo_runs_sorted = hpo_runs.sort_values('tags.trial_number')
            else:
                hpo_runs_sorted = hpo_runs.sort_values('start_time')
            
            values = hpo_runs_sorted[metric_col].values
            best_so_far = np.maximum.accumulate(values)
            
            # Calculate convergence metrics
            convergence_analysis = {
                'improvement_ratio': np.sum(np.diff(best_so_far) > 0) / len(values),
                'final_improvement': (best_so_far[-1] - best_so_far[0]) / best_so_far[0] if best_so_far[0] > 0 else 0,
                'convergence_trial': np.argmax(best_so_far) + 1,
                'plateau_length': len(values) - np.argmax(best_so_far) - 1
            }
            
            return convergence_analysis
            
        except Exception as e:
            logger.warning(f"Failed to analyze HPO convergence: {e}")
            return {}
    
    def _generate_training_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training summary statistics."""
        summary = {
            'status': report['status'],
            'duration_minutes': report['duration'].total_seconds() / 60 if report['duration'] else None,
            'converged': False,
            'best_epoch': None,
            'final_performance': {}
        }
        
        # Extract key metrics
        if report['final_metrics']:
            for metric, value in report['final_metrics'].items():
                if 'val' in metric:
                    summary['final_performance'][metric] = value
        
        # Analyze convergence from metric history
        if report['metric_history']:
            val_loss_history = report['metric_history'].get('val_loss', [])
            if val_loss_history:
                # Find best epoch (lowest validation loss)
                losses = [loss for _, loss in val_loss_history]
                summary['best_epoch'] = np.argmin(losses) + 1
                
                # Check convergence (improvement in last 10% of training)
                if len(losses) > 10:
                    last_10_percent = int(len(losses) * 0.1)
                    recent_improvement = min(losses[-last_10_percent:]) < min(losses[:-last_10_percent])
                    summary['converged'] = not recent_improvement
        
        return summary
    
    def _save_training_report(self, report: Dict[str, Any]) -> str:
        """Save training report as JSON and generate PDF summary."""
        # Save JSON report
        json_filename = f"training_report_{report['run_name'].replace(' ', '_')}.json"
        json_filepath = self.output_dir / json_filename
        
        # Convert datetime objects to strings for JSON serialization
        json_report = report.copy()
        for key in ['start_time', 'end_time']:
            if json_report.get(key):
                json_report[key] = json_report[key].isoformat()
        if json_report.get('duration'):
            json_report['duration'] = str(json_report['duration'])
        
        with open(json_filepath, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # Generate HTML report
        html_filepath = self._generate_html_training_report(report)
        
        return str(html_filepath)
    
    def _generate_html_training_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML training report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Report - {report['run_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .param {{ background-color: #f8f8f8; padding: 5px; margin: 2px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Training Report: {report['run_name']}</h1>
                <p><strong>Run ID:</strong> {report['run_id']}</p>
                <p><strong>Status:</strong> {report['status']}</p>
                <p><strong>Duration:</strong> {report.get('duration', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>Final Metrics</h2>
                {self._format_metrics_html(report['final_metrics'])}
            </div>
            
            <div class="section">
                <h2>Parameters</h2>
                {self._format_params_html(report['parameters'])}
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                {self._format_summary_html(report['summary'])}
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                {self._format_visualizations_html(report.get('visualizations', []))}
            </div>
        </body>
        </html>
        """
        
        html_filename = f"training_report_{report['run_name'].replace(' ', '_')}.html"
        html_filepath = self.output_dir / html_filename
        
        with open(html_filepath, 'w') as f:
            f.write(html_content)
        
        return str(html_filepath)
    
    def _format_metrics_html(self, metrics: Dict[str, float]) -> str:
        """Format metrics for HTML display."""
        if not metrics:
            return "<p>No metrics available</p>"
        
        html = "<table><tr><th>Metric</th><th>Value</th></tr>"
        for metric, value in sorted(metrics.items()):
            html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
        html += "</table>"
        return html
    
    def _format_params_html(self, params: Dict[str, Any]) -> str:
        """Format parameters for HTML display."""
        if not params:
            return "<p>No parameters available</p>"
        
        html = "<table><tr><th>Parameter</th><th>Value</th></tr>"
        for param, value in sorted(params.items()):
            html += f"<tr><td>{param}</td><td>{value}</td></tr>"
        html += "</table>"
        return html
    
    def _format_summary_html(self, summary: Dict[str, Any]) -> str:
        """Format summary for HTML display."""
        html = "<table><tr><th>Metric</th><th>Value</th></tr>"
        for key, value in summary.items():
            html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        html += "</table>"
        return html
    
    def _format_visualizations_html(self, viz_paths: List[str]) -> str:
        """Format visualizations for HTML display."""
        if not viz_paths:
            return "<p>No visualizations available</p>"
        
        html = ""
        for viz_path in viz_paths:
            filename = Path(viz_path).name
            if viz_path.endswith('.html'):
                html += f'<p><a href="{filename}" target="_blank">{filename}</a></p>'
            else:
                html += f'<img src="{filename}" alt="{filename}" style="max-width: 100%; margin: 10px 0;">'
        
        return html
    
    def _extract_trial_params(self, trial_row: pd.Series) -> Dict[str, Any]:
        """Extract trial parameters from DataFrame row."""
        params = {}
        for col in trial_row.index:
            if col.startswith('params.'):
                param_name = col.replace('params.', '')
                params[param_name] = trial_row[col]
        return params
    
    def _extract_trial_metrics(self, trial_row: pd.Series) -> Dict[str, float]:
        """Extract trial metrics from DataFrame row."""
        metrics = {}
        for col in trial_row.index:
            if col.startswith('metrics.'):
                metric_name = col.replace('metrics.', '')
                metrics[metric_name] = trial_row[col]
        return metrics
    
    def _compare_metrics(self, runs_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compare metrics across runs."""
        all_metrics = set()
        for run in runs_data:
            all_metrics.update(run['metrics'].keys())
        
        comparison = {}
        for metric in all_metrics:
            comparison[metric] = {}
            for run in runs_data:
                run_name = run['run_name']
                comparison[metric][run_name] = run['metrics'].get(metric, np.nan)
        
        return comparison
    
    def _compare_parameters(self, runs_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Compare parameters across runs."""
        all_params = set()
        for run in runs_data:
            all_params.update(run['params'].keys())
        
        comparison = {}
        for param in all_params:
            comparison[param] = {}
            for run in runs_data:
                run_name = run['run_name']
                comparison[param][run_name] = run['params'].get(param, 'N/A')
        
        return comparison
    
    def _rank_performance(self, runs_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank runs by performance."""
        # Use validation F1 score as primary ranking metric
        ranking = []
        for run in runs_data:
            val_f1 = run['metrics'].get('val_f1', 0)
            ranking.append({
                'run_name': run['run_name'],
                'run_id': run['run_id'],
                'val_f1': val_f1,
                'rank': 0  # Will be filled after sorting
            })
        
        # Sort by validation F1 score (descending)
        ranking.sort(key=lambda x: x['val_f1'], reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(ranking):
            entry['rank'] = i + 1
        
        return ranking
    
    def _statistical_comparison(self, runs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical comparison of runs."""
        # Extract key metrics for statistical analysis
        val_f1_scores = [run['metrics'].get('val_f1', np.nan) for run in runs_data]
        val_f1_scores = [score for score in val_f1_scores if not np.isnan(score)]
        
        if len(val_f1_scores) < 2:
            return {'error': 'Insufficient data for statistical comparison'}
        
        stats = {
            'mean_val_f1': np.mean(val_f1_scores),
            'std_val_f1': np.std(val_f1_scores),
            'min_val_f1': np.min(val_f1_scores),
            'max_val_f1': np.max(val_f1_scores),
            'range_val_f1': np.max(val_f1_scores) - np.min(val_f1_scores),
            'coefficient_of_variation': np.std(val_f1_scores) / np.mean(val_f1_scores) if np.mean(val_f1_scores) > 0 else np.inf
        }
        
        return stats
    
    def _save_hpo_report(self, report: Dict[str, Any]) -> str:
        """Save HPO study report."""
        # Save JSON report
        json_filename = f"hpo_study_report_{report['study_name'].replace(' ', '_')}.json"
        json_filepath = self.output_dir / json_filename
        
        with open(json_filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate HTML report
        html_filepath = self._generate_html_hpo_report(report)
        
        return str(html_filepath)
    
    def _generate_html_hpo_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML HPO study report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HPO Study Report - {report['study_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .trial {{ background-color: #f8f8f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>HPO Study Report: {report['study_name']}</h1>
                <p><strong>Total Trials:</strong> {report['total_trials']}</p>
                <p><strong>Completed Trials:</strong> {report['completed_trials']}</p>
                <p><strong>Failed Trials:</strong> {report['failed_trials']}</p>
            </div>
            
            <div class="section">
                <h2>Best Trial</h2>
                {self._format_best_trial_html(report.get('best_trial'))}
            </div>
            
            <div class="section">
                <h2>Parameter Importance</h2>
                {self._format_param_importance_html(report.get('parameter_importance', {}))}
            </div>
            
            <div class="section">
                <h2>Convergence Analysis</h2>
                {self._format_convergence_html(report.get('convergence_analysis', {}))}
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                {self._format_visualizations_html(report.get('visualizations', []))}
            </div>
        </body>
        </html>
        """
        
        html_filename = f"hpo_study_report_{report['study_name'].replace(' ', '_')}.html"
        html_filepath = self.output_dir / html_filename
        
        with open(html_filepath, 'w') as f:
            f.write(html_content)
        
        return str(html_filepath)
    
    def _format_best_trial_html(self, best_trial: Optional[Dict[str, Any]]) -> str:
        """Format best trial information for HTML."""
        if not best_trial:
            return "<p>No best trial information available</p>"
        
        html = f"""
        <p><strong>Run Name:</strong> {best_trial['run_name']}</p>
        <p><strong>Validation F1:</strong> {best_trial['val_f1']:.4f}</p>
        <h3>Parameters:</h3>
        {self._format_params_html(best_trial['parameters'])}
        """
        return html
    
    def _format_param_importance_html(self, param_importance: Dict[str, float]) -> str:
        """Format parameter importance for HTML."""
        if not param_importance:
            return "<p>No parameter importance data available</p>"
        
        html = "<table><tr><th>Parameter</th><th>Importance Score</th></tr>"
        for param, importance in sorted(param_importance.items(), key=lambda x: x[1], reverse=True):
            html += f"<tr><td>{param}</td><td>{importance:.4f}</td></tr>"
        html += "</table>"
        return html
    
    def _format_convergence_html(self, convergence: Dict[str, Any]) -> str:
        """Format convergence analysis for HTML."""
        if not convergence:
            return "<p>No convergence analysis available</p>"
        
        html = "<table><tr><th>Metric</th><th>Value</th></tr>"
        for metric, value in convergence.items():
            html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        html += "</table>"
        return html
    
    def _save_comparison_report(self, report: Dict[str, Any]) -> str:
        """Save comparison report."""
        # Save JSON report
        json_filename = f"comparison_report_{report['comparison_name'].replace(' ', '_')}.json"
        json_filepath = self.output_dir / json_filename
        
        with open(json_filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(json_filepath)
    
    def _upload_training_artifacts(self, run_id: str, report: Dict[str, Any], 
                                 viz_paths: List[str]) -> None:
        """Upload training report artifacts to MLflow."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            with mlflow.start_run(run_id=run_id):
                # Upload visualizations
                for viz_path in viz_paths:
                    mlflow.log_artifact(viz_path, "visualizations")
                
                # Upload report
                if report.get('report_path'):
                    mlflow.log_artifact(report['report_path'], "reports")
                
        except Exception as e:
            logger.warning(f"Failed to upload training artifacts: {e}")
    
    def _upload_hpo_artifacts(self, parent_run_id: str, report: Dict[str, Any],
                            viz_paths: List[str]) -> None:
        """Upload HPO report artifacts to MLflow."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            with mlflow.start_run(run_id=parent_run_id):
                # Upload visualizations
                for viz_path in viz_paths:
                    mlflow.log_artifact(viz_path, "hpo_analysis")
                
                # Upload report
                if report.get('report_path'):
                    mlflow.log_artifact(report['report_path'], "reports")
                
        except Exception as e:
            logger.warning(f"Failed to upload HPO artifacts: {e}")


def create_reporter(output_dir: str = "reports") -> ExperimentReporter:
    """
    Create experiment reporter instance.
    
    Args:
        output_dir: Directory to save reports
        
    Returns:
        ExperimentReporter instance
    """
    return ExperimentReporter(output_dir)
