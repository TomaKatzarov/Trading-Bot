#!/usr/bin/env python
"""
Qualitative Evaluation Tool for LoRA Models

This script performs qualitative evaluation of LoRA models by 
running predictions on sample inputs and analyzing the results.

Usage:
  python tools/qualitative_evaluation.py [--lora_path=PATH]
"""

import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

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
    try:
        from core.decision_engine import DecisionEngine, TAKE_PROFIT_PERCENTAGE, STOP_LOSS_PERCENTAGE
    except ImportError:
        logger.warning("Decision Engine not available. Using only LLMHandler for evaluation.")
        DecisionEngine = None
        TAKE_PROFIT_PERCENTAGE = 5.0
        STOP_LOSS_PERCENTAGE = 2.0
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def generate_test_scenarios():
    """Generate test scenarios for evaluation."""
    return {
        "bullish": {
            "vector": np.array([0.75, 0.85, 0.73, 0.84, 1.2, 0.82, 0.7, 1.1, 0.3, 0.8]),
            "expected_action": "BUY",
            "description": "Strong bullish setup with high close, increased volume, and positive sentiment"
        },
        "bearish": {
            "vector": np.array([-0.7, -0.65, -0.75, -0.68, 0.9, -0.55, 0.3, 0.8, -0.5, 0.2]),
            "expected_action": "SELL",
            "description": "Bearish setup with lower close, negative returns, and negative sentiment"
        },
        "neutral": {
            "vector": np.array([0.1, 0.05, -0.02, 0.02, 0.5, 0.1, 0.5, 0.5, 0.0, 0.5]),
            "expected_action": "HOLD",
            "description": "Neutral setup with flat indicators and neutral sentiment"
        },
        "extreme_bullish": {
            "vector": np.array([0.95, 0.98, 0.92, 0.97, 2.0, 0.95, 0.9, 1.5, 0.8, 0.95]),
            "expected_action": "BUY",
            "description": "Extremely bullish setup with very high values and strong positive sentiment"
        },
        "extreme_bearish": {
            "vector": np.array([-0.95, -0.85, -0.97, -0.92, 1.5, -0.9, 0.1, 0.9, -0.8, 0.05]),
            "expected_action": "SELL",
            "description": "Extremely bearish setup with very low values and strong negative sentiment"
        }
    }

def run_qualitative_evaluation(lora_path=None):
    """Run qualitative evaluation with sample inputs."""
    logger.info("Starting qualitative evaluation")
    
    try:
        # Initialize LLMHandler with the specified LoRA adapter
        if lora_path:
            llm_handler = LLMHandler()
            llm_handler.load_adapter(lora_path)
            logger.info(f"Loaded specific adapter from: {lora_path}")
        else:
            llm_handler = LLMHandler(use_lora=True)
            logger.info("Using default/latest adapter")
        
        if not llm_handler.adapter_loaded:
            logger.warning("No LoRA adapter loaded. Using base model.")
            
        # Initialize Decision Engine if available
        decision_engine = None
        if DecisionEngine is not None:
            try:
                decision_engine = DecisionEngine(starting_balance=10000.0)
                logger.info(f"Decision Engine initialized with risk parameters: TP={TAKE_PROFIT_PERCENTAGE}%, SL={STOP_LOSS_PERCENTAGE}%")
            except Exception as de_error:
                logger.error(f"Failed to initialize Decision Engine: {de_error}")
        
        # Test scenarios
        scenarios = generate_test_scenarios()
        symbol = "TEST"
        timestamp = datetime.now(timezone.utc)
        
        logger.info("\n" + "="*50)
        logger.info("QUALITATIVE EVALUATION RESULTS")
        logger.info("="*50)
        
        results = []
        
        for name, scenario in scenarios.items():
            logger.info(f"\nScenario: {name}")
            logger.info(f"Description: {scenario['description']}")
            
            # Direct LLM classification
            market_data = {
                "context_vector": scenario['vector'].tolist(),
                "sentiment_score": scenario['vector'][-1],
                "symbol": symbol
            }
            
            analysis_result = llm_handler.analyze_market(market_data)
            class_prediction = analysis_result.get("predicted_class", None)
            confidence = analysis_result.get("confidence", 0.0)
            decision = analysis_result.get("decision", "UNKNOWN")
            
            logger.info(f"LLM Classification: Class {class_prediction}, Decision: {decision}, Confidence: {confidence:.2f}")
            
            # Decision Engine action (if available)
            action = "N/A"
            action_confidence = 0.0
            if decision_engine:
                action, action_confidence = decision_engine.predict_action(
                    symbol, timestamp, scenario['vector']
                )
                logger.info(f"Decision Engine Action: {action}, Confidence: {action_confidence:.2f}")
            
            # Evaluation
            expected = scenario['expected_action']
            llm_result = "✅ MATCH" if decision == expected else "❌ MISMATCH"
            de_result = "✅ MATCH" if action == expected else "❌ MISMATCH"
            logger.info(f"LLM Evaluation: {llm_result} (Expected: {expected}, Got: {decision})")
            if decision_engine:
                logger.info(f"DE Evaluation: {de_result} (Expected: {expected}, Got: {action})")
            
            # Store result for return value
            results.append({
                "case": name,
                "prediction": analysis_result,
                "expected": expected,
                "match": decision == expected
            })
        
        logger.info("\n" + "="*50)
        logger.info("Qualitative evaluation completed")
        
        return {
            "status": "success",
            "adapter_path": llm_handler.adapter_path_used if llm_handler.adapter_loaded else "No adapter",
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Qualitative evaluation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run qualitative evaluation on LoRA model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to specific LoRA adapter")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    results = run_qualitative_evaluation(args.lora_path)
    
    if results["status"] == "success":
        logger.info(f"Qualitative evaluation completed successfully using adapter: {results['adapter_path']}")
        success_rate = sum(1 for r in results["results"] if r["match"]) / len(results["results"])
        logger.info(f"Match rate: {success_rate:.2%}")
    else:
        logger.error(f"Qualitative evaluation failed: {results['message']}")
        sys.exit(1)
