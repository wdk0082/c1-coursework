#!/bin/bash

# Sphinx documentation build script for fivedreg
# This script generates HTML documentation from docstrings

set -e

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_ROOT/backend/docs"

cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run ./scripts/simple_setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

echo "Building documentation for fivedreg..."
echo "Project root: $PROJECT_ROOT"
echo "Docs directory: $DOCS_DIR"
echo ""

# Check if docs directory exists
if [ ! -d "$DOCS_DIR" ]; then
    echo "Error: Documentation directory not found at $DOCS_DIR"
    exit 1
fi

cd "$DOCS_DIR"

# Clean previous build
make clean

# Build HTML documentation
make html

echo ""
echo "Documentation built successfully!"
echo "Open $DOCS_DIR/_build/html/index.html to view."
