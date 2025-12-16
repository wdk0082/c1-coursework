#!/bin/bash

# Test runner script for 5D Regression backend
# This script activates the virtual environment and runs pytest

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run ./scripts/simple_setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

echo "Running tests..."
echo ""

# Run pytest with verbose output
cd backend
pytest -v "$@"
