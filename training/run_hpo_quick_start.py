#!/usr/bin/env python3
"""
Quick Start Script for Hyperparameter Optimization

This script provides a simplified interface for running HPO studies with
sensible defaults. It's designed for quick experimentation and testing.

Usage Examples:
    # Run HPO for LSTM model with 20 trials
    python training/run_hpo_quick_start.py --model lstm --trials 20
    
    # Run HPO for all models with custom study name
    python training/run_hpo_quick_start.py --study-name "my_hpo_study" --trials 30
    
    # Run HPO with pre-generated training data
    python training/run_hpo_quick_start.py --model gru --use-pregenerated --trials 15

Author: Flow-Code
Date: 2025-05-28
Version: 1.0
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.run_hpo import HPOManager, create_hpo_config, check_pregenerated_data
from training.train_nn_model import create_default_config

# Progress bar imports
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_pregenerated_data_fixed() -> bool:
    """
    Check if pre-generated training data exists (fixed path).
    
    Returns:
        bool: True if pre-generated data exists, False otherwise
    """
    # Use absolute path from project root
    data_dir = project_root / "data" / "training_data"
    required_files = [
        "train_X.npy", "train_y.npy", "train_asset_ids.npy",
        "val_X.npy", "val_y.npy", "val_asset_ids.npy",
        "test_X.npy", "test_y.npy", "test_asset_ids.npy",
        "scalers.joblib", "metadata.json", "asset_id_mapping.json"
    ]
    
    if not data_dir.exists():
        logger.info(f"Training data directory does not exist: {data_dir}")
        return False
    
    missing_files = []
    for file_name in required_files:
        if not (data_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.info(f"Missing training data files: {missing_files}")
        return False
    
    logger.info(f"All required training data files found in: {data_dir}")
    return True

def generate_training_data():
    """
    Generate training data using the established pipeline when no pre-generated data is found.
    """
    try:
        # Import data preparation module
        from core.data_preparation_nn import NNDataPreparer
        import json
        import numpy as np
        import joblib
        from pathlib import Path
        
        if RICH_AVAILABLE:
            console.print(Panel(
                "[yellow]⚠ No pre-generated training data found[/yellow]\n"
                "[cyan]Generating training data using established pipeline...[/cyan]",
                title="[bold yellow]Data Generation[/bold yellow]",
                border_style="yellow"
            ))
        else:
            logger.info("⚠ No pre-generated training data found")
            logger.info("Generating training data using established pipeline...")
        
        # Create data preparation configuration
        config = {
            'feature_list': [
                'open', 'high', 'low', 'close', 'volume',
                'SMA_20', 'EMA_12', 'RSI_14', 'MACD_line', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_middle', 'BB_lower', 'BB_bandwidth',
                'sentiment_score_hourly_ffill',
                'DayOfWeek_sin', 'DayOfWeek_cos'
            ],
            'lookback_window': 24,
            'target_profit_pct': 5.0,
            'target_stop_loss_pct': 2.0,
            'target_horizon_hours': 8,
            'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
            'min_samples_per_symbol': 100,
            'nan_handling': 'ffill',
            'use_robust_scaler_for': ['volume'],
            'multi_symbol_training': True
        }
        
        # Initialize data preparer
        data_preparer = NNDataPreparer(config)
        
        # Generate training data
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Preparing training data...", total=None)
                
                try:
                    result = data_preparer.get_prepared_data_for_training()
                    progress.update(task, description="[green]✓ Training data prepared successfully!")
                except Exception as e:
                    progress.update(task, description=f"[red]✗ Failed to prepare data: {e}")
                    raise
        else:
            logger.info("Preparing training data...")
            result = data_preparer.get_prepared_data_for_training()
            logger.info("✓ Training data prepared successfully!")
        
        # Extract data components
        (train_X, train_y, train_asset_ids), (val_X, val_y, val_asset_ids), (test_X, test_y, test_asset_ids), scalers, metadata = result
        
        # Create output directory
        output_dir = project_root / "data" / "training_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data files
        np.save(output_dir / "train_X.npy", train_X)
        np.save(output_dir / "train_y.npy", train_y)
        np.save(output_dir / "train_asset_ids.npy", train_asset_ids)
        np.save(output_dir / "val_X.npy", val_X)
        np.save(output_dir / "val_y.npy", val_y)
        np.save(output_dir / "val_asset_ids.npy", val_asset_ids)
        np.save(output_dir / "test_X.npy", test_X)
        np.save(output_dir / "test_y.npy", test_y)
        np.save(output_dir / "test_asset_ids.npy", test_asset_ids)
        
        # Save scalers and metadata
        joblib.dump(scalers, output_dir / "scalers.joblib")
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save asset ID mapping
        asset_id_mapping_path = project_root / "config" / "asset_id_mapping.json"
        if asset_id_mapping_path.exists():
            with open(asset_id_mapping_path, 'r') as f:
                asset_id_mapping = json.load(f)
            with open(output_dir / "asset_id_mapping.json", 'w') as f:
                json.dump(asset_id_mapping, f, indent=2)
        
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[green]✓ Training data generated successfully![/green]\n"
                f"[cyan]Location:[/cyan] {output_dir}\n"
                f"[cyan]Train samples:[/cyan] {len(train_X):,}\n"
                f"[cyan]Validation samples:[/cyan] {len(val_X):,}\n"
                f"[cyan]Test samples:[/cyan] {len(test_X):,}",
                title="[bold green]Data Generation Complete[/bold green]",
                border_style="green"
            ))
        else:
            logger.info(f"✓ Training data generated successfully at: {output_dir}")
            logger.info(f"Train samples: {len(train_X):,}")
            logger.info(f"Validation samples: {len(val_X):,}")
            logger.info(f"Test samples: {len(test_X):,}")
        
        return True
        
    except ImportError as e:
        error_msg = f"Failed to import data preparation module: {e}"
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[red]✗ {error_msg}[/red]\n"
                "[yellow]Please ensure core.data_preparation_nn is available[/yellow]",
                title="[bold red]Data Generation Failed[/bold red]",
                border_style="red"
            ))
        else:
            logger.error(error_msg)
        return False
        
    except Exception as e:
        error_msg = f"Failed to generate training data: {e}"
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[red]✗ {error_msg}[/red]",
                title="[bold red]Data Generation Failed[/bold red]",
                border_style="red"
            ))
        else:
            logger.error(error_msg)
        return False

def display_data_status():
    """Display training data status with rich formatting and generate if missing."""
    data_exists = check_pregenerated_data_fixed()
    
    if data_exists:
        if RICH_AVAILABLE:
            console.print(Panel(
                "[green]✓ Pre-generated training data found![/green]\n"
                f"[cyan]Location:[/cyan] {project_root}/data/training_data/\n"
                "[cyan]Status:[/cyan] Ready for HPO",
                title="[bold green]Training Data Status[/bold green]",
                border_style="green"
            ))
        else:
            logger.info("✓ Pre-generated training data found - ready for HPO")
    else:
        if RICH_AVAILABLE:
            console.print(Panel(
                "[yellow]⚠ No pre-generated training data found[/yellow]\n"
                "[cyan]Will attempt to generate training data...[/cyan]",
                title="[bold yellow]Training Data Status[/bold yellow]",
                border_style="yellow"
            ))
        else:
            logger.info("⚠ No pre-generated training data found - will attempt to generate")
        
        # Attempt to generate training data
        generation_success = generate_training_data()
        if generation_success:
            data_exists = True
        else:
            if RICH_AVAILABLE:
                console.print(Panel(
                    "[red]✗ Failed to generate training data[/red]\n"
                    "[yellow]HPO will generate data during training (slower)[/yellow]",
                    title="[bold red]Data Generation Failed[/bold red]",
                    border_style="red"
                ))
            else:
                logger.error("✗ Failed to generate training data - HPO will generate data during training")
    
    return data_exists

def get_user_confirmation(message: str) -> bool:
    """Get user confirmation for interactive mode."""
    if RICH_AVAILABLE:
        console.print(f"[yellow]{message}[/yellow]")
    else:
        print(message)
    
    try:
        response = input("Continue? (y/n): ").lower().strip()
        return response in ['y', 'yes']
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[red]Operation cancelled by user[/red]")
        else:
            print("\nOperation cancelled by user")
        return False

def run_quick_hpo(model_type: str = None, trials: int = 50, study_name: str = None,
                  use_pregenerated: bool = None, auto_confirm: bool = False):
    """
    Run a quick HPO study with sensible defaults.
    
    Args:
        model_type: Model type to optimize ('mlp', 'lstm', 'gru', 'cnn_lstm', 'all')
        trials: Number of trials to run
        study_name: Custom study name
        use_pregenerated: Whether to use pre-generated data
        auto_confirm: Whether to automatically confirm prompts
    """
    
    # Display welcome message
    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold blue]🚀 Quick Start HPO for Neural Network Models[/bold blue]\n"
            "[cyan]Simplified interface for rapid hyperparameter optimization[/cyan]",
            title="[bold green]HPO Quick Start[/bold green]",
            border_style="blue"
        ))
    else:
        logger.info("🚀 Quick Start HPO for Neural Network Models")
    
    # Check and handle training data
    data_available = display_data_status()
    
    # Auto-detect pre-generated data usage if not specified
    if use_pregenerated is None:
        use_pregenerated = data_available
    
    # Create HPO configuration with defaults
    hpo_config = create_hpo_config()
    
    # Override with user parameters
    if study_name:
        hpo_config['study_name'] = study_name
    else:
        hpo_config['study_name'] = f"quick_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    hpo_config['n_trials'] = trials
    
    # Set models to optimize
    if model_type and model_type != 'all':
        if model_type not in ['mlp', 'lstm', 'gru', 'cnn_lstm']:
            raise ValueError(f"Invalid model type: {model_type}. Choose from: mlp, lstm, gru, cnn_lstm, all")
        hpo_config['models_to_optimize'] = [model_type]
    else:
        hpo_config['models_to_optimize'] = ['mlp', 'lstm', 'gru', 'cnn_lstm']
    
    # Display configuration
    if RICH_AVAILABLE:
        models_str = ", ".join(hpo_config['models_to_optimize'])
        console.print(Panel(
            f"[cyan]Study Name:[/cyan] {hpo_config['study_name']}\n"
            f"[cyan]Models:[/cyan] {models_str}\n"
            f"[cyan]Trials per Model:[/cyan] {trials}\n"
            f"[cyan]Target Metric:[/cyan] {hpo_config['target_metric']}\n"
            f"[cyan]Use Pre-generated Data:[/cyan] {use_pregenerated}",
            title="[bold cyan]HPO Configuration[/bold cyan]",
            border_style="cyan"
        ))
    else:
        logger.info(f"Study: {hpo_config['study_name']}")
        logger.info(f"Models: {', '.join(hpo_config['models_to_optimize'])}")
        logger.info(f"Trials per model: {trials}")
        logger.info(f"Use pre-generated data: {use_pregenerated}")
    
    # Get user confirmation
    total_trials = trials * len(hpo_config['models_to_optimize'])
    if not auto_confirm and not get_user_confirmation(f"This will run {total_trials} total trials. "):
        return
    
    # Create base training configuration with HPO optimizations
    logger.info("Creating optimized base configuration for fast HPO")
    # Load the fast HPO config template
    fast_config_path = project_root / "training" / "config_templates" / "hpo_fast.yaml"
    if fast_config_path.exists():
        from training.train_nn_model import load_config
        base_config = load_config(str(fast_config_path))
        logger.info("Loaded hpo_fast.yaml configuration")
    else:
        base_config = create_default_config()
        logger.warning("hpo_fast.yaml not found, using default config with optimizations")
    
    # Apply critical HPO optimizations to base_config
    training_cfg = base_config.setdefault('training_config', {})
    training_cfg['epochs'] = 15  # Fast trials - down from 100
    training_cfg['batch_size'] = 256  # Maximize GPU utilization
    training_cfg['early_stopping_patience'] = 5  # Aggressive early stopping
    training_cfg['scheduler_patience'] = 3  # Faster LR adaptation
    training_cfg['use_amp'] = True  # Mixed precision for 2x speedup
    training_cfg['use_torch_compile'] = False  # Disable compile overhead for HPO
    base_config['num_workers'] = 8  # Optimize for i5-13600K (14 cores)
    
    logger.info(f"HPO Config: epochs={training_cfg['epochs']}, batch_size={training_cfg['batch_size']}, "
                f"use_amp={training_cfg['use_amp']}, use_torch_compile={training_cfg['use_torch_compile']}")
    
    # Setup GPU if available
    try:
        import torch
        from utils.gpu_utils import setup_gpu
        if torch.cuda.is_available():
            gpu_info = setup_gpu()
            logger.info(f"GPU setup: {gpu_info}")
    except ImportError:
        logger.warning("GPU utilities not available")
    
    # Create and run HPO manager
    hpo_manager = HPOManager(hpo_config)
      # Run optimization for each model
    all_results = {}
    start_time = time.time()
    
    for i, model_type in enumerate(hpo_config['models_to_optimize'], 1):
        if RICH_AVAILABLE:
            console.print(f"\n[bold blue]{'='*60}[/bold blue]")
            console.print(f"[bold blue]HPO Progress: {i}/{len(hpo_config['models_to_optimize'])} - {model_type.upper()}[/bold blue]")
            console.print(f"[bold blue]{'='*60}[/bold blue]")
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"HPO Progress: {i}/{len(hpo_config['models_to_optimize'])} - {model_type.upper()}")
            logger.info(f"{'='*60}")
        
        try:
            # Clean up any active MLflow runs before starting new model optimization
            try:
                import mlflow
                if mlflow.active_run() is not None:
                    logger.info(f"Ending active MLflow run before starting {model_type} optimization")
                    mlflow.end_run()
            except ImportError:
                pass  # MLflow not available
            except Exception as e:
                logger.warning(f"Error cleaning up MLflow runs: {e}")
            
            # Run optimization
            study = hpo_manager.run_optimization(
                model_type=model_type,
                base_config=base_config,
                n_trials=trials,
                target_metric=hpo_config['target_metric'],
                use_pregenerated=use_pregenerated
            )
              # Analyze results
            results = hpo_manager.analyze_results(study, model_type)
            all_results[model_type] = results
            
            # Save individual results
            hpo_manager.save_results(results, hpo_config['output_dir'])
            
            # Clean up MLflow runs after model completion
            try:
                import mlflow
                if mlflow.active_run() is not None:
                    logger.info(f"Ending MLflow run after {model_type} optimization")
                    mlflow.end_run()
            except ImportError:
                pass  # MLflow not available
            except Exception as e:
                logger.warning(f"Error cleaning up MLflow run after {model_type}: {e}")
            
        except KeyboardInterrupt:
            # Clean up on interruption
            try:
                import mlflow
                if mlflow.active_run() is not None:
                    mlflow.end_run()
            except:
                pass
            if RICH_AVAILABLE:
                console.print(f"\n[yellow]HPO interrupted for {model_type}[/yellow]")
            else:
                logger.info(f"HPO interrupted for {model_type}")
            break
        except Exception as e:
            # Clean up on error
            try:
                import mlflow
                if mlflow.active_run() is not None:
                    mlflow.end_run()
            except:
                pass
            if RICH_AVAILABLE:
                console.print(f"[red]HPO failed for {model_type}: {e}[/red]")
            else:
                logger.error(f"HPO failed for {model_type}: {e}")
            continue
      # Display final results
    elapsed_time = time.time() - start_time
    
    # Final MLflow cleanup
    try:
        import mlflow
        if mlflow.active_run() is not None:
            logger.info("Ending any remaining active MLflow runs")
            mlflow.end_run()
    except ImportError:
        pass  # MLflow not available
    except Exception as e:
        logger.warning(f"Error in final MLflow cleanup: {e}")
    
    if all_results:
        if RICH_AVAILABLE:
            console.print(f"\n[bold green]{'='*60}[/bold green]")
            console.print(f"[bold green]HPO QUICK START COMPLETED[/bold green]")
            console.print(f"[bold green]{'='*60}[/bold green]")
            console.print(f"[cyan]Total Time:[/cyan] {elapsed_time/3600:.1f} hours")
            console.print(f"[cyan]Results saved to:[/cyan] {hpo_config['output_dir']}")
            
            # Results table
            from rich.table import Table
            table = Table(title="Best Results Summary")
            table.add_column("Model", style="cyan")
            table.add_column("Best Value", style="green")
            table.add_column("Trial #", style="yellow")
            table.add_column("Completed", style="blue")
            
            for model_type, results in all_results.items():
                table.add_row(
                    model_type.upper(),
                    f"{results['best_value']:.4f}",
                    str(results['best_trial_number']),
                    f"{results['n_complete']}/{results['n_trials']}"
                )
            
            console.print(table)
            
            # Find overall best
            best_model = max(all_results.items(), key=lambda x: x[1]['best_value'])
            console.print(f"\n[bold green]🏆 Overall Best Model: {best_model[0].upper()}[/bold green]")
            console.print(f"[green]Best Value: {best_model[1]['best_value']:.4f}[/green]")
            
        else:
            logger.info(f"\n{'='*60}")
            logger.info("HPO QUICK START COMPLETED")
            logger.info(f"{'='*60}")
            logger.info(f"Total time: {elapsed_time/3600:.1f} hours")
            
            for model_type, results in all_results.items():
                logger.info(f"{model_type.upper()}: {results['best_value']:.4f} "
                           f"(Trial {results['best_trial_number']}, "
                           f"{results['n_complete']}/{results['n_trials']} completed)")
            
            best_model = max(all_results.items(), key=lambda x: x[1]['best_value'])
            logger.info(f"\n🏆 Overall best: {best_model[0].upper()} "
                       f"(value: {best_model[1]['best_value']:.4f})")
    else:
        if RICH_AVAILABLE:
            console.print("[red]No successful HPO runs completed[/red]")
        else:
            logger.error("No successful HPO runs completed")

def main():
    """Main function for quick start HPO."""
    parser = argparse.ArgumentParser(description='Quick Start HPO for Neural Network Models')
    parser.add_argument('--model', type=str, choices=['mlp', 'lstm', 'gru', 'cnn_lstm', 'all'],
                       default='all', help='Model type to optimize (default: all)')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of trials per model (default: 50)')
    parser.add_argument('--study-name', type=str,
                       help='Custom study name (auto-generated if not provided)')
    parser.add_argument('--use-pregenerated', action='store_true',
                       help='Force use of pre-generated data')
    parser.add_argument('--no-pregenerated', action='store_true',
                       help='Force data generation during training')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Automatically answer yes to prompts and skip confirmation')
    
    args = parser.parse_args()
    
    # Determine data usage preference
    use_pregenerated = None
    if args.use_pregenerated:
        use_pregenerated = True
    elif args.no_pregenerated:
        use_pregenerated = False
    
    try:
        run_quick_hpo(
            model_type=args.model,
            trials=args.trials,
            study_name=args.study_name,
            use_pregenerated=use_pregenerated,
            auto_confirm=args.yes
        )
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[red]Operation cancelled by user[/red]")
        else:
            print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Error: {e}[/red]")
        else:
            logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()