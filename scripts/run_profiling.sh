#!/bin/bash

# Profiling script for 5D Regression model training
# This script runs performance profiling and generates result plots

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROFILING_DIR="$PROJECT_ROOT/profiling"

cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run ./scripts/simple_setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

echo "Running profiling for 5D Regression..."
echo "Project root: $PROJECT_ROOT"
echo "Profiling directory: $PROFILING_DIR"
echo ""

# Check if profiling directory exists
if [ ! -d "$PROFILING_DIR" ]; then
    echo "Error: Profiling directory not found at $PROFILING_DIR"
    exit 1
fi

cd "$PROFILING_DIR"

# Run the profiling script
echo "Running profiling benchmarks..."
python run_profiling.py

# Run the plotting script
echo ""
echo "Generating result plots..."
python plot_profiling_results.py

echo ""
echo "Profiling complete!"
echo "Results are saved in $PROFILING_DIR/results/"
