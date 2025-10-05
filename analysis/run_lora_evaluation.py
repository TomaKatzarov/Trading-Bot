#!/usr/bin/env python
"""
LORA Evaluation Runner

This script provides a unified interface to run all LORA evaluation tools:
- Qualitative Evaluation (sample predictions on test inputs)
- LORA Diagnostic (adapter inspection and weight analysis)
- Test Scenario Generation (comprehensive model performance evaluation)

It generates a consolidated report combining insights from all evaluations.

Usage:
  python tools/run_lora_evaluation.py

The script will interactively prompt for configuration options.
"""

import os
import sys
import argparse
import logging
import subprocess
import time
import json
import re
from pathlib import Path
from datetime import datetime
import importlib.util
import textwrap
from typing import Dict, Any, List, Optional, Union, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    from models.llm_handler import LLMHandler
except ImportError as e:
    logger.error(f"Failed to import LLMHandler: {e}")
    logger.error("This script requires the LLMHandler module to run properly.")
    sys.exit(1)

class EvaluationRunner:
    """Manages the execution of all LORA evaluation tools."""

    def __init__(self):
        """Initialize the evaluation runner."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reports_dir = project_root / "analysis" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.lora_adapter_path = None
        self.lora_name = None
        self.test_results = {}
        self.user_config = {}
        self.report_paths = {}

    def check_modules_exist(self) -> bool:
        """Check if all required modules exist."""
        modules = {
            "create_test_scenarios.py": project_root / "tools" / "create_test_scenarios.py",
            "qualitative_evaluation.py": project_root / "tools" / "qualitative_evaluation.py",
            "lora_diagnostic.py": project_root / "tools" / "lora_diagnostic.py"
        }
        
        missing_modules = []
        for name, path in modules.items():
            if not path.exists():
                missing_modules.append(name)
        
        if missing_modules:
            logger.error(f"Missing required modules: {', '.join(missing_modules)}")
            logger.error("Some tests will be skipped.")
            
            # Create missing modules with placeholder code
            for module in missing_modules:
                self.create_placeholder_module(module)
            
            return False
        return True

    def create_placeholder_module(self, module_name: str):
        """Create a placeholder module with skeleton code."""
        module_path = project_root / "tools" / module_name
        
        if module_name == "qualitative_evaluation.py":
            content = textwrap.dedent("""
            #!/usr/bin/env python
            \"\"\"
            Qualitative Evaluation Tool for LoRA Models
            
            This script performs qualitative evaluation of LoRA models by 
            running predictions on sample inputs and analyzing the results.
            
            Usage:
              python tools/qualitative_evaluation.py [--lora_path=PATH]
            \"\"\"
            
            import os
            import sys
            import logging
            import argparse
            from pathlib import Path
            
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
            
            # ...existing code...
            """)
            
        elif module_name == "lora_diagnostic.py":
            content = textwrap.dedent("""
            #!/usr/bin/env python
            \"\"\"
            LoRA Diagnostic Tool
            
            This script analyzes a LoRA adapter to provide insights into its structure,
            weight distribution, and other diagnostic information.
            
            Usage:
              python tools/lora_diagnostic.py [--lora_path=PATH] [--report_dir=DIR]
            \"\"\"
            
            # ...existing code...
            """)
        
        # Write the placeholder module
        with open(module_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created placeholder module: {module_path}")

    def detect_lora_adapter(self) -> Tuple[str, str]:
        """Detect the currently active LORA adapter."""
        logger.info("Detecting active LORA adapter...")
        try:
            adapters_dir = project_root / "training" / "adapter_runs"
            if adapters_dir.exists():
                candidates = [d for d in adapters_dir.iterdir() if d.is_dir()]
                if candidates:
                    latest = max(candidates, key=lambda d: d.stat().st_mtime)
                    logger.info(f"Detected active LoRA adapter folder: {latest.name}")
                    return str(latest), latest.name
            logger.warning("No LoRA adapters found in training/adapter_runs")
            return None, None
        except Exception as e:
            logger.error(f"Error detecting LORA adapter: {e}", exc_info=True)
            return None, None

    def get_user_config(self):
        """Get configuration parameters from user."""
        print("\n" + "="*80)
        print("LORA Evaluation Suite - Configuration")
        print("="*80)
        
        # Detect current LORA adapter
        self.lora_adapter_path, self.lora_name = self.detect_lora_adapter()
        
        if self.lora_adapter_path:
            print(f"\nCurrent active LORA adapter: {self.lora_name}")
            use_current = input("Use this adapter for evaluation? (Y/n): ").strip().lower()
            if use_current != 'n':
                self.user_config['lora_path'] = self.lora_adapter_path
            else:
                self.user_config['lora_path'] = input("Enter path to LORA adapter to evaluate: ").strip()
                if not self.user_config['lora_path']:
                    self.user_config['lora_path'] = self.lora_adapter_path
        else:
            print("\nNo active LORA adapter detected.")
            self.user_config['lora_path'] = input("Enter path to LORA adapter to evaluate: ").strip()
            if not self.user_config['lora_path']:
                logger.error("No LORA adapter specified. Cannot continue.")
                sys.exit(1)
        
        # Get test scenario count
        try:
            num_scenarios = int(input("\nEnter number of test scenarios to generate (default: 100): ").strip() or "100")
            self.user_config['num_scenarios'] = max(10, min(500, num_scenarios))  # Limit between 10-500
        except ValueError:
            print("Invalid input. Using default value of 100.")
            self.user_config['num_scenarios'] = 100
        
        # Create report directory name (sanitize to avoid spaces/special chars)
        raw_name = input("\nEnter name for the evaluation report (optional): ").strip()
        if not raw_name:
            base = self.lora_name or "unknown_adapter"
            raw_name = f"{base}_evaluation_{self.timestamp}"
        # replace any non-alphanumeric or hyphen/underscore with underscore
        safe_name = re.sub(r'[^\w\-]+', '_', raw_name)
        self.user_config['report_name'] = safe_name

        self.user_config['report_dir'] = str(self.reports_dir / self.user_config['report_name'])
        
        # Generate execution plan
        print("\n" + "="*80)
        print("Execution Plan:")
        print(f"  - LORA Adapter: {self.user_config['lora_path']}")
        print(f"  - Test Scenarios: {self.user_config['num_scenarios']}")
        print(f"  - Report Directory: {self.user_config['report_dir']}")
        print("="*80)
        
        confirm = input("\nProceed with evaluation? (Y/n): ").strip().lower()
        if confirm == 'n':
            logger.info("Evaluation cancelled by user.")
            sys.exit(0)
        
        print("\n")
        return self.user_config

    def run_qualitative_evaluation(self):
        """Run qualitative evaluation module."""
        logger.info("Running qualitative evaluation...")
        module_path = project_root / "tools" / "qualitative_evaluation.py"
        
        if not module_path.exists():
            logger.warning("qualitative_evaluation.py not found, skipping this step")
            return {"status": "skipped", "message": "Module not found"}
        
        try:
            spec = importlib.util.spec_from_file_location("qualitative_evaluation", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Call the module's function directly
            if hasattr(module, 'run_qualitative_evaluation'):
                results = module.run_qualitative_evaluation(self.user_config['lora_path'])
                self.report_paths['qualitative_evaluation'] = None  # No separate report file
                logger.info("Qualitative evaluation completed")
                return results
            else:
                logger.error("Qualitative evaluation module missing required function")
                return {"status": "error", "message": "Module missing required function"}
        except Exception as e:
            logger.error(f"Error running qualitative evaluation: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def run_lora_diagnostic(self):
        """Run LORA diagnostic module."""
        logger.info("Running LORA diagnostic...")
        module_path = project_root / "tools" / "lora_diagnostic.py"
        
        if not module_path.exists():
            logger.warning("lora_diagnostic.py not found, skipping this step")
            return {"status": "skipped", "message": "Module not found"}
        
        try:
            # Create subdirectory for diagnostic report
            diagnostic_dir = Path(self.user_config['report_dir']) / "diagnostic"
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            
            spec = importlib.util.spec_from_file_location("lora_diagnostic", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Call the module's function directly
            if hasattr(module, 'run_lora_diagnostic'):
                results = module.run_lora_diagnostic(self.user_config['lora_path'])
                if results["status"] == "success":
                    # Copy or generate the report in our directory
                    report_data = module.analyze_adapter_weights(self.user_config['lora_path'])
                    self.report_paths['lora_diagnostic'] = module.generate_diagnostic_report(report_data, diagnostic_dir)
                    logger.info(f"LORA diagnostic report saved to {self.report_paths['lora_diagnostic']}")
                return results
            else:
                logger.error("LORA diagnostic module missing required function")
                return {"status": "error", "message": "Module missing required function"}
        except Exception as e:
            logger.error(f"Error running LORA diagnostic: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def run_test_scenarios(self):
        """Run test scenario generation module."""
        logger.info("Running test scenario generation...")
        module_path = project_root / "tools" / "create_test_scenarios.py"
        
        if not module_path.exists():
            logger.warning("create_test_scenarios.py not found, skipping this step")
            return {"status": "skipped", "message": "Module not found"}
        
        try:
            # Create command with arguments
            cmd = [
                sys.executable,
                str(module_path),
                f"--scenarios={self.user_config['num_scenarios']}",
                f"--report-name={self.user_config['report_name']}_scenarios"
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Get the path to the generated report
            report_dir = self.reports_dir / f"{self.user_config['report_name']}_scenarios"
            self.report_paths['test_scenarios'] = report_dir
            
            # Parse output to extract key metrics
            output = result.stdout
            metrics = {}
            
            # Look for accuracy in output
            acc_match = re.search(r'Overall Accuracy:\s+([\d.]+)', output)
            if acc_match:
                metrics['accuracy'] = float(acc_match.group(1))
            
            return {
                "status": "success",
                "report_dir": str(report_dir),
                "metrics": metrics,
                "stdout": output,
                "stderr": result.stderr
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Test scenario execution failed: {e}")
            logger.error(f"Subprocess stderr:\n{e.stderr}")
            return {"status": "error", "message": str(e), "stderr": e.stderr}
        except Exception as e:
            logger.error(f"Error running test scenarios: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def generate_comprehensive_report(self):
        """Generate a comprehensive report combining all evaluation results."""
        logger.info("Generating comprehensive evaluation report...")
        
        report_dir = Path(self.user_config['report_dir'])
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "lora_adapter": {
                "path": self.user_config['lora_path'],
                "name": Path(self.user_config['lora_path']).name if self.user_config['lora_path'] else None
            },
            "configuration": self.user_config,
            "qualitative_evaluation": self.test_results.get('qualitative_evaluation', {"status": "not_run"}),
            "lora_diagnostic": self.test_results.get('lora_diagnostic', {"status": "not_run"}),
            "test_scenarios": self.test_results.get('test_scenarios', {"status": "not_run"})
        }
        
        # Save JSON report
        json_report_path = report_dir / "comprehensive_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Create HTML report
        html_report_path = report_dir / "comprehensive_report.html"
        self.generate_html_report(report_data, html_report_path)
        
        logger.info(f"Comprehensive report generated at {report_dir}")
        return str(report_dir)

    def generate_html_report(self, report_data, output_path):
        """Generate an HTML report from the evaluation data."""
        # Extract some key data
        lora_name = report_data["lora_adapter"]["name"] or "Unknown Adapter"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract test scenario results if available
        scenario_results = {}
        test_scenarios = report_data.get("test_scenarios", {})
        if test_scenarios.get("status") == "success" and self.report_paths.get('test_scenarios'):
            try:
                scenario_json = self.report_paths['test_scenarios'] / "evaluation_report.json"
                if scenario_json.exists():
                    with open(scenario_json, 'r') as f:
                        scenario_results = json.load(f)
            except Exception as e:
                logger.error(f"Error loading scenario results: {e}")
        
        # Extract qualitative evaluation results
        qual_eval = report_data.get("qualitative_evaluation", {})
        qual_results = qual_eval.get("results", []) if qual_eval.get("status") == "success" else []
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive LORA Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ margin-bottom: 30px; }}
                .metric {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
                .metric h3 {{ margin-top: 0; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .visualization {{ margin-bottom: 30px; }}
                .visualization img {{ max-width: 100%; border: 1px solid #ddd; }}
                pre {{ background-color: #f5f5f5; padding: 15px; overflow-x: auto; }}
                .summary {{ background-color: #e6f7ff; padding: 15px; border-left: 5px solid #1890ff; margin-bottom: 20px; }}
                .status-success {{ color: green; }}
                .status-error {{ color: red; }}
                .status-skipped {{ color: orange; }}
            </style>
        </head>
        <body>
            <h1>Comprehensive LORA Evaluation Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="summary">
                <h2>Evaluation Summary</h2>
                <p>
                    <strong>LORA Adapter:</strong> {lora_name}<br>
                    <strong>Path:</strong> {report_data["lora_adapter"]["path"]}<br>
                    <strong>Test Scenarios:</strong> {report_data["configuration"]["num_scenarios"]}<br>
                </p>
                <p>
                    <strong>Module Status:</strong><br>
                    Qualitative Evaluation: <span class="status-{report_data['qualitative_evaluation']['status']}">{report_data['qualitative_evaluation']['status'].upper()}</span><br>
                    LORA Diagnostic: <span class="status-{report_data['lora_diagnostic']['status']}">{report_data['lora_diagnostic']['status'].upper()}</span><br>
                    Test Scenarios: <span class="status-{report_data['test_scenarios']['status']}">{report_data['test_scenarios']['status'].upper()}</span><br>
                </p>
        """
        
        # Add scenario metrics if available
        if scenario_results:
            overall_metrics = scenario_results.get("overall_metrics", {})
            html_content += f"""
                <p>
                    <strong>Performance Metrics:</strong><br>
                    Overall Accuracy: <strong>{overall_metrics.get("accuracy", 0)*100:.2f}%</strong><br>
                    Decision Accuracy: <strong>{overall_metrics.get("decision_accuracy", 0)*100:.2f}%</strong><br>
                    Average Confidence: <strong>{overall_metrics.get("avg_confidence", 0)*100:.2f}%</strong><br>
                </p>
            """
            
            # Check for bias
            bias_detected = False
            class_distribution = scenario_results.get("class_distribution", {}).get("predicted", {})
            if class_distribution:
                total_predictions = sum(int(v) for v in class_distribution.values())
                for cls, count in class_distribution.items():
                    if int(count) / total_predictions > 0.8:  # 80% threshold for bias
                        bias_detected = True
                        html_content += f"""
                        <p class="status-error">
                            <strong>BIAS ALERT:</strong> Model shows strong bias toward class {cls} 
                            ({int(count)/total_predictions*100:.1f}% of predictions)
                        </p>
                        """
            
            if not bias_detected:
                html_content += """
                <p class="status-success">
                    <strong>No significant bias detected</strong> in prediction distribution.
                </p>
                """
        
        html_content += """
            </div>
        """
        
        # Add qualitative evaluation section if available
        if qual_results:
            html_content += """
            <div class="container">
                <h2>Qualitative Evaluation</h2>
                <table>
                    <tr>
                        <th>Test Case</th>
                        <th>Predicted Class</th>
                        <th>Decision</th>
                        <th>Confidence</th>
                    </tr>
            """
            
            for result in qual_results:
                case = result["case"]
                prediction = result["prediction"]
                pred_class = prediction.get("predicted_class", "N/A")
                decision = prediction.get("decision", "N/A")
                confidence = prediction.get("confidence", 0) * 100
                
                html_content += f"""
                    <tr>
                        <td>{case}</td>
                        <td>{pred_class}</td>
                        <td>{decision}</td>
                        <td>{confidence:.2f}%</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        # Add links to detailed reports
        html_content += """
            <div class="container">
                <h2>Detailed Reports</h2>
                <p>The following detailed reports are available:</p>
                <ul>
        """
        
        if self.report_paths.get('test_scenarios'):
            html_content += f"""
                <li><a href="../{self.user_config['report_name']}_scenarios/evaluation_report.html" target="_blank">Test Scenarios Evaluation Report</a></li>
            """
        
        if self.report_paths.get('lora_diagnostic'):
            lora_report = Path(self.report_paths['lora_diagnostic']) / "lora_diagnostic.json"
            if lora_report.exists():
                html_content += f"""
                <li><a href="diagnostic/lora_diagnostic.json" target="_blank">LORA Diagnostic Report (JSON)</a></li>
                """
        
        html_content += """
                </ul>
            </div>
        """
        
        # Add recommendations section
        html_content += """
            <div class="container">
                <h2>Recommendations</h2>
        """
        
        if scenario_results:
            scenario_metrics = scenario_results.get("scenario_type_metrics", [])
            overall_accuracy = scenario_results.get("overall_metrics", {}).get("accuracy", 0)
            
            # Find worst-performing scenario type
            worst_type = None
            worst_accuracy = 1.0
            for metric in scenario_metrics:
                acc = float(metric.get("is_correct_class", 0))
                if acc < worst_accuracy:
                    worst_accuracy = acc
                    worst_type = metric.get("scenario_type")
            
            extreme_metrics = next((m for m in scenario_metrics if m.get("scenario_type") == "extreme_buy"), None)
            extreme_accuracy = extreme_metrics.get("is_correct_class", 0) if extreme_metrics else 0
            
            if overall_accuracy < 0.5:
                html_content += """
                <p class="status-error">
                    <strong>Model Performance is Poor:</strong> The model shows very low overall accuracy.
                    Consider complete retraining with a larger dataset or different parameters.
                </p>
                """
            
            if worst_type and worst_accuracy < 0.5:
                html_content += f"""
                <p>
                    <strong>Scenario Type Weakness:</strong> The model performs particularly poorly on '{worst_type}' scenarios,
                    with only {worst_accuracy*100:.1f}% accuracy. Consider adding more training examples for this scenario type.
                </p>
                """
            
            if extreme_accuracy < 0.2:
                html_content += """
                <p>
                    <strong>Poor Extreme Case Handling:</strong> The model struggles with extreme buy/sell signals.
                    This suggests it might be biased toward neutral predictions. Consider adjusting the training data
                    balance or training parameters to improve responsiveness to strong signals.
                </p>
                """
            
            html_content += f"""
            <p>
                <strong>Next Steps:</strong>
            </p>
            <ul>
                <li>{"Retrain with a more balanced dataset" if bias_detected else "Continue monitoring model performance"}</li>
                <li>{"Increase LoRA rank and alpha for better adaptability" if overall_accuracy < 0.5 else "Current LoRA parameters seem adequate"}</li>
                <li>{"Add more extreme case examples to the training data" if extreme_accuracy < 0.3 else "Extreme case handling is acceptable"}</li>
            </ul>
            """
        else:
            html_content += """
            <p>
                No test scenario results available for generating recommendations.
                Please run the test scenario evaluation for detailed recommendations.
            </p>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write the HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return str(output_path)

    def run_all_tests(self):
        """Run all evaluation tests in sequence."""
        # 1. Qualitative evaluation
        self.test_results['qualitative_evaluation'] = self.run_qualitative_evaluation()
        
        # 2. LORA diagnostic
        self.test_results['lora_diagnostic'] = self.run_lora_diagnostic()
        
        # 3. Test scenario generation
        self.test_results['test_scenarios'] = self.run_test_scenarios()
        
        # 4. Generate comprehensive report
        report_path = self.generate_comprehensive_report()
        
        return report_path

def main():
    """Main entry point for the LORA evaluation runner."""
    print("\n" + "="*80)
    print(" "*28 + "LORA EVALUATION SUITE")
    print("="*80)
    print("\nThis tool runs a comprehensive evaluation of the current LORA model.")
    print("It combines qualitative evaluation, diagnostic analysis, and test scenarios.")
    
    runner = EvaluationRunner()
    runner.check_modules_exist()
    runner.get_user_config()
    
    try:
        report_path = runner.run_all_tests()
        print("\n" + "="*80)
        print(f"Evaluation complete! Comprehensive report saved to:")
        print(f"{report_path}/comprehensive_report.html")
        print("="*80 + "\n")
        return 0
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        print("\nEvaluation failed. See log for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
