#!/usr/bin/env python
"""
LoRA Diagnostic Tool

This script analyzes a LoRA adapter to provide insights into its structure,
weight distribution, and other diagnostic information.

Usage:
  python tools/lora_diagnostic.py [--lora_path=PATH] [--report_dir=DIR]
"""

import os
import sys
import logging
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project modules
try:
    from models.llm_handler import LLMHandler
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def analyze_adapter_weights(adapter_path):
    """Analyze adapter weights and generate statistics."""
    logger.info(f"Analyzing adapter weights from: {adapter_path}")
    
    try:
        # Get adapter configuration first
        config_path = Path(adapter_path) / "adapter_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        logger.info(f"Adapter config loaded with r={config.get('r', 'N/A')} and alpha={config.get('lora_alpha', 'N/A')}")
        
        # Determine which file to load (safetensors or pytorch_model.bin)
        adapter_file_safetensors = Path(adapter_path) / "adapter_model.safetensors"
        adapter_file_pytorch = Path(adapter_path) / "pytorch_model.bin"
        
        if adapter_file_safetensors.exists():
            logger.info(f"Loading safetensors model: {adapter_file_safetensors}")
            try:
                from safetensors import safe_open
                with safe_open(str(adapter_file_safetensors), framework="pt") as f:
                    tensors = {key: f.get_tensor(key) for key in f.keys()}
            except ImportError:
                logger.warning("safetensors not available. Falling back to PyTorch loading.")
                tensors = torch.load(adapter_file_pytorch, map_location="cpu")
        elif adapter_file_pytorch.exists():
            logger.info(f"Loading PyTorch model: {adapter_file_pytorch}")
            tensors = torch.load(adapter_file_pytorch, map_location="cpu")
        else:
            raise FileNotFoundError(f"No adapter weights found in {adapter_path}")
        
        # Analyze weights
        lora_weights = {}
        a_weights = []
        b_weights = []
        
        for key, tensor in tensors.items():
            if 'lora' in key.lower():
                stats = {
                    "shape": tuple(tensor.shape),
                    "mean": float(tensor.mean().item()),
                    "std": float(tensor.std().item()),
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                    "abs_mean": float(tensor.abs().mean().item())
                }
                lora_weights[key] = stats
                
                # Categorize A and B matrices
                if 'lora_a' in key.lower():
                    a_weights.append(stats)
                elif 'lora_b' in key.lower():
                    b_weights.append(stats)
        
        # Calculate overall statistics
        overall_stats = {
            "total_weights": len(lora_weights),
            "a_weights": len(a_weights),
            "b_weights": len(b_weights),
            "avg_mean": np.mean([w["mean"] for w in lora_weights.values()]),
            "avg_std": np.mean([w["std"] for w in lora_weights.values()]),
            "max_abs": max([max(abs(w["min"]), abs(w["max"])) for w in lora_weights.values()]),
            "avg_abs_mean": np.mean([w["abs_mean"] for w in lora_weights.values()])
        }
        
        # Add A/B matrix stats
        if a_weights and b_weights:
            overall_stats["a_mean"] = np.mean([w["mean"] for w in a_weights]) 
            overall_stats["b_mean"] = np.mean([w["mean"] for w in b_weights])
            overall_stats["a_std"] = np.mean([w["std"] for w in a_weights])
            overall_stats["b_std"] = np.mean([w["std"] for w in b_weights])
        
        return {
            "status": "success", 
            "adapter_path": str(adapter_path),
            "lora_config": config,
            "weight_stats": lora_weights,
            "overall_stats": overall_stats
        }
    
    except Exception as e:
        logger.error(f"Failed to analyze adapter weights: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }

def generate_diagnostic_report(results, output_dir=None):
    """Generate a diagnostic report based on the analysis results."""
    if output_dir is None:
        output_dir = project_root / "analysis" / "reports" / f"lora_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    report_path = output_dir / "lora_diagnostic.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate weight distribution plots if analysis was successful
    if results["status"] == "success":
        try:
            # Generate weight distribution plot
            plot_path = output_dir / "weight_distributions.png"
            plt.figure(figsize=(14, 10))
            
            # Get overall stats for title
            overall = results["overall_stats"]
            lora_config = results["lora_config"]
            r = lora_config.get("r", "N/A")
            alpha = lora_config.get("lora_alpha", "N/A")
            
            # Plot weight statistics
            stats = results["weight_stats"]
            means = [v["mean"] for v in stats.values()]
            stds = [v["std"] for v in stats.values()]
            abs_means = [v["abs_mean"] for v in stats.values()]
            
            plt.subplot(2, 2, 1)
            plt.hist(means, bins=30)
            plt.title("Distribution of Weight Means")
            plt.xlabel("Mean Value")
            plt.ylabel("Count")
            
            plt.subplot(2, 2, 2)
            plt.hist(stds, bins=30)
            plt.title("Distribution of Weight Standard Deviations")
            plt.xlabel("Standard Deviation")
            plt.ylabel("Count")
            
            plt.subplot(2, 2, 3)
            plt.hist(abs_means, bins=30)
            plt.title("Distribution of Absolute Mean Values")
            plt.xlabel("Absolute Mean")
            plt.ylabel("Count")
            
            # Add a text summary in the 4th subplot
            plt.subplot(2, 2, 4)
            plt.axis('off')
            summary_text = f"""
            LoRA Adapter Summary
            -------------------
            r = {r}, alpha = {alpha}
            
            Total weights: {overall['total_weights']}
            A matrices: {overall.get('a_weights', 'N/A')}
            B matrices: {overall.get('b_weights', 'N/A')}
            
            Mean of means: {overall['avg_mean']:.6f}
            Mean of stds: {overall['avg_std']:.6f}
            Max absolute value: {overall['max_abs']:.6f}
            Mean of absolute means: {overall['avg_abs_mean']:.6f}
            """
            plt.text(0.05, 0.95, summary_text, fontsize=12, va='top', ha='left')
            
            plt.suptitle(f"LoRA Adapter Weights Analysis (r={r}, alpha={alpha})")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Generated plots and saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
    
    logger.info(f"Diagnostic report saved to {output_dir}")
    return output_dir

def run_lora_diagnostic(lora_path=None):
    """Run the complete diagnostic process."""
    logger.info("Starting LoRA diagnostic")
    
    try:
        # First, try to get adapter path from LLMHandler if not specified
        if lora_path is None:
            llm_handler = LLMHandler(use_lora=True)
            if llm_handler.adapter_loaded:
                lora_path = llm_handler.adapter_path_used
                logger.info(f"Using currently loaded adapter: {lora_path}")
            else:
                logger.error("No LoRA adapter loaded and no path specified")
                return {
                    "status": "error",
                    "message": "No LoRA adapter available for diagnostics"
                }
        
        # Analyze adapter weights
        results = analyze_adapter_weights(lora_path)
        if results["status"] != "success":
            return results
        
        # Generate report
        output_dir = generate_diagnostic_report(results)
        
        return {
            "status": "success",
            "adapter_path": lora_path,
            "report_dir": str(output_dir)
        }
        
    except Exception as e:
        logger.error(f"LoRA diagnostic failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run diagnostic analysis on LoRA adapter")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to specific LoRA adapter")
    parser.add_argument("--report_dir", type=str, default=None, help="Directory to save the report")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    report_dir = args.report_dir
    if report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_lora_diagnostic(args.lora_path)
    
    if results["status"] == "success":
        logger.info(f"LoRA diagnostic completed successfully. Report saved to: {results.get('report_dir', 'unknown')}")
    else:
        logger.error(f"LoRA diagnostic failed: {results['message']}")
        sys.exit(1)
