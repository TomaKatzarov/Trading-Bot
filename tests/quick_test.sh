#!/bin/bash
# Quick test runner for Linux/Mac
# Usage: ./tests/quick_test.sh

echo "=============================================="
echo "Running Comprehensive Training Pipeline Tests"
echo "=============================================="
echo ""

# Activate virtual environment if exists
if [ -f "activate_rl_env.sh" ]; then
    echo "Activating virtual environment..."
    source activate_rl_env.sh
fi

# Run tests
python tests/run_comprehensive_tests.py "$@"

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed! Training pipeline is ready."
else
    echo "❌ Some tests failed. Review output above."
fi

exit $EXIT_CODE
