#!/bin/bash

# Simple setup script for c1-coursework
# This script creates a virtual environment and installs all dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up c1-coursework..."
cd "$PROJECT_ROOT"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install backend dependencies (including docs and test)
echo "Installing backend dependencies..."
cd backend
pip install -e ".[docs,test]"
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Setup frontend environment if not exists
if [ ! -f "frontend/.env.local" ] && [ -f "frontend/.env.example" ]; then
    echo "Creating frontend .env.local from .env.example..."
    cp frontend/.env.example frontend/.env.local
fi

# Build documentation
echo "Building documentation..."
./scripts/build_docs.sh

echo ""
echo "Setup complete!"
echo "To activate the virtual environment, run: source .venv/bin/activate"
echo "To start the application, see the Quick Start Guide in README.md"
