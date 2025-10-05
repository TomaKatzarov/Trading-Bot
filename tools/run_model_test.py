#!/usr/bin/env python
"""
Model Test Execution Script

This script provides a simple wrapper to run the test scenario generation and evaluation.
It sets up the environment, ensures necessary directories exist, and runs the tests.

Usage:
  python tools/run_model_test.py [--scenarios=100] [--report-name=my_report]
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def ensure_directories():
    """Ensure necessary directories exist."""
    reports_dir = project_root / "analysis" / "reports"
    if not reports_dir.exists():
        os.makedirs(reports_dir)
        print(f"Created reports directory: {reports_dir}")
    else:
        print(f"Reports directory exists: {reports_dir}")

def main():
    """Run the test scenario generation and evaluation."""
    print("Setting up environment for model testing...")
    ensure_directories()
    
    # Prepare command to run test scenarios
    cmd = [sys.executable, str(project_root / "tools" / "create_test_scenarios.py")]
    
    # Pass through any arguments
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
