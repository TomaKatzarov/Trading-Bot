"""Master test runner for comprehensive training pipeline validation.

This script runs all test suites and generates a detailed report of test coverage.
Execute before training to ensure surgical precision in all components.

Usage:
    python tests/run_comprehensive_tests.py
    python tests/run_comprehensive_tests.py --verbose
    python tests/run_comprehensive_tests.py --report-file test_report.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest


# Test suites in execution order
TEST_SUITES = [
    {
        "name": "Production Config Validation",
        "file": "tests/test_production_config_validation.py",
        "description": "Validates actual training config files and data paths are ready for production",
    },
    {
        "name": "Data Validation & Configuration",
        "file": "tests/test_data_validation_comprehensive.py",
        "description": "Validates data format, OHLC constraints, technical indicators, and configuration loading",
    },
    {
        "name": "Reward Calculation Accuracy",
        "file": "tests/test_reward_calculation_accuracy.py",
        "description": "Validates PnL rewards, position sizing, transaction costs, diversity bonuses, and aggregation",
    },
    {
        "name": "Action Space Behavior",
        "file": "tests/test_action_space_behavior.py",
        "description": "Validates continuous action mapping, masking, multi-position support, and trade execution",
    },
    {
        "name": "End-to-End Training Pipeline",
        "file": "tests/test_comprehensive_training_pipeline.py",
        "description": "Validates environment initialization, episode simulation, vectorization, and SAC integration",
    },
    {
        "name": "Reward Infrastructure (Phase A2)",
        "file": "tests/test_reward_infrastructure_e2e.py",
        "description": "Validates reward config matches YAML, component weights, and Sharpe gate behavior",
    },
    {
        "name": "Reward Shaper Stage 2",
        "file": "tests/test_reward_shaper_stage2.py",
        "description": "Validates Sharpe gate, forced exit penalties, and time decay mechanisms",
    },
]


def print_header(text: str, char: str = "=") -> None:
    """Print formatted header."""
    width = 80
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width + "\n")


def print_section(text: str) -> None:
    """Print formatted section."""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}\n")


def run_test_suite(test_file: str, verbose: bool = False) -> Dict[str, any]:
    """Run a single test suite and return results."""
    
    if not Path(test_file).exists():
        return {
            "passed": False,
            "skipped": True,
            "error": f"Test file not found: {test_file}",
            "duration": 0.0,
        }
    
    args = [test_file, "-v"] if verbose else [test_file]
    args.extend(["--tb=short", "--no-header"])
    
    start_time = time.time()
    exit_code = pytest.main(args)
    duration = time.time() - start_time
    
    return {
        "passed": exit_code == 0,
        "skipped": False,
        "error": None if exit_code == 0 else f"Tests failed with exit code {exit_code}",
        "duration": duration,
    }


def generate_report(results: List[Dict], output_file: str = None) -> str:
    """Generate test report."""
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE TRAINING PIPELINE TEST REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary
    total_suites = len(results)
    passed_suites = sum(1 for r in results if r["result"]["passed"])
    skipped_suites = sum(1 for r in results if r["result"]["skipped"])
    failed_suites = total_suites - passed_suites - skipped_suites
    total_duration = sum(r["result"]["duration"] for r in results)
    
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Test Suites:   {total_suites}")
    report_lines.append(f"Passed:              {passed_suites} ✓")
    report_lines.append(f"Failed:              {failed_suites} ✗")
    report_lines.append(f"Skipped:             {skipped_suites} ⊘")
    report_lines.append(f"Total Duration:      {total_duration:.2f}s")
    report_lines.append("")
    
    # Individual results
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 80)
    
    for i, result_info in enumerate(results, 1):
        suite = result_info["suite"]
        result = result_info["result"]
        
        status = "✓ PASSED" if result["passed"] else ("⊘ SKIPPED" if result["skipped"] else "✗ FAILED")
        
        report_lines.append(f"\n{i}. {suite['name']}")
        report_lines.append(f"   Status:      {status}")
        report_lines.append(f"   File:        {suite['file']}")
        report_lines.append(f"   Description: {suite['description']}")
        report_lines.append(f"   Duration:    {result['duration']:.2f}s")
        
        if result["error"]:
            report_lines.append(f"   Error:       {result['error']}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Overall verdict
    if failed_suites == 0 and skipped_suites == 0:
        report_lines.append("✓ ALL TESTS PASSED - TRAINING PIPELINE READY")
    elif failed_suites == 0:
        report_lines.append(f"⚠ PARTIAL SUCCESS - {skipped_suites} suite(s) skipped")
    else:
        report_lines.append(f"✗ TESTS FAILED - {failed_suites} suite(s) failed")
    
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # Save to file if requested
    if output_file:
        Path(output_file).write_text(report_text, encoding="utf-8")
        print(f"\n📄 Report saved to: {output_file}")
    
    return report_text


def main() -> int:
    """Main test runner."""
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for training pipeline"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose test output"
    )
    parser.add_argument(
        "--report-file", "-r",
        type=str,
        default=None,
        help="Save report to file"
    )
    parser.add_argument(
        "--suite", "-s",
        type=str,
        default=None,
        help="Run specific test suite by name (partial match)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only essential tests (skip lengthy integration tests)"
    )
    
    args = parser.parse_args()
    
    # Filter suites if requested
    suites_to_run = TEST_SUITES
    if args.suite:
        suites_to_run = [
            s for s in TEST_SUITES
            if args.suite.lower() in s["name"].lower()
        ]
        if not suites_to_run:
            print(f"❌ No test suites match: {args.suite}")
            return 1
    
    if args.quick:
        # Skip the lengthy end-to-end tests
        suites_to_run = [
            s for s in suites_to_run
            if "End-to-End" not in s["name"]
        ]
    
    # Print header
    print_header("COMPREHENSIVE TRAINING PIPELINE TEST SUITE")
    
    print("This test suite validates:")
    print("  • Data format and integrity")
    print("  • Configuration loading and propagation")
    print("  • Reward calculation accuracy")
    print("  • Action space behavior")
    print("  • Environment initialization")
    print("  • Training pipeline integration")
    print("")
    print(f"Running {len(suites_to_run)} test suite(s)...")
    
    # Run tests
    results = []
    
    for i, suite in enumerate(suites_to_run, 1):
        print_section(f"[{i}/{len(suites_to_run)}] {suite['name']}")
        print(f"Description: {suite['description']}")
        print(f"File: {suite['file']}")
        print("")
        
        result = run_test_suite(suite["file"], verbose=args.verbose)
        
        results.append({
            "suite": suite,
            "result": result,
        })
        
        if result["passed"]:
            print(f"\n✓ Suite passed in {result['duration']:.2f}s")
        elif result["skipped"]:
            print(f"\n⊘ Suite skipped: {result['error']}")
        else:
            print(f"\n✗ Suite failed: {result['error']}")
    
    # Generate report
    print_header("TEST REPORT")
    report = generate_report(results, output_file=args.report_file)
    print(report)
    
    # Return exit code
    failed_count = sum(1 for r in results if not r["result"]["passed"] and not r["result"]["skipped"])
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
