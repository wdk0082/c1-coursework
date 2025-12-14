#!/bin/bash

# Docker startup script for 5D Regression application
# This script builds and starts both backend and frontend containers

set -e

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Starting 5D Regression application..."
echo "Project root: $PROJECT_ROOT"
echo ""

# Build and start containers
docker-compose up --build

echo ""
echo "Application stopped."
