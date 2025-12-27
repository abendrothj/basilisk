#!/bin/bash

echo "🧪 Running Project Basilisk Test Suite"
echo "========================================"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Install test dependencies
echo "Installing test dependencies..."
pip install -q -r tests/requirements.txt

echo ""
echo "Running tests..."
echo ""

# Run tests with different verbosity options
if [ "$1" == "verbose" ] || [ "$1" == "-v" ]; then
    pytest tests/ -v --tb=long
elif [ "$1" == "coverage" ] || [ "$1" == "-c" ]; then
    echo "Running with coverage..."
    pytest tests/ --cov=poison-core --cov=api --cov-report=html --cov-report=term-missing
    echo ""
    echo "📊 Coverage report generated in htmlcov/index.html"
elif [ "$1" == "quick" ] || [ "$1" == "-q" ]; then
    pytest tests/ -q
elif [ "$1" == "unit" ]; then
    echo "Running unit tests only..."
    pytest tests/test_radioactive_poison.py -v
elif [ "$1" == "api" ]; then
    echo "Running API tests only..."
    pytest tests/test_api.py -v
elif [ "$1" == "cli" ]; then
    echo "Running CLI tests only..."
    pytest tests/test_cli.py -v
elif [ "$1" == "failed" ] || [ "$1" == "-f" ]; then
    echo "Re-running failed tests..."
    pytest tests/ --lf -v
elif [ "$1" == "help" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./run_tests.sh [option]"
    echo ""
    echo "Options:"
    echo "  (none)      - Run all tests with standard output"
    echo "  verbose/-v  - Run with verbose output and full tracebacks"
    echo "  coverage/-c - Run with code coverage report"
    echo "  quick/-q    - Run with minimal output"
    echo "  unit        - Run only unit tests (radioactive_poison)"
    echo "  api         - Run only API tests"
    echo "  cli         - Run only CLI tests"
    echo "  failed/-f   - Re-run only failed tests from last run"
    echo "  help/-h     - Show this help message"
    exit 0
else
    pytest tests/ -v
fi

echo ""
echo "========================================"
echo "✅ Test run complete!"
