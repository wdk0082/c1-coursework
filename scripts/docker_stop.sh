#!/bin/bash

# Docker shutdown script for 5D Regression application
# This script stops and removes the containers

set -e

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Stopping 5D Regression application..."
echo "Project root: $PROJECT_ROOT"
echo ""

# Stop and remove containers
docker compose down

echo ""
echo "Application stopped."
