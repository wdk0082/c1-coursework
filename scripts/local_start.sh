#!/bin/bash

# Local startup script for 5D Regression application
# This script starts both backend and frontend servers locally

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

echo "Starting 5D Regression application (local)..."
echo ""

# Start backend in background
echo "Starting backend on http://localhost:8000..."
cd "$PROJECT_ROOT/backend"
python api.py &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Save PID for stop script
echo $BACKEND_PID > "$PROJECT_ROOT/.backend.pid"

# Wait a moment for backend to start
sleep 2

# Start frontend in background
echo "Starting frontend on http://localhost:3000..."
cd "$PROJECT_ROOT/frontend"
npm run dev &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Save PID for stop script
echo $FRONTEND_PID > "$PROJECT_ROOT/.frontend.pid"

echo ""
echo "Application started!"
echo "- Backend API: http://localhost:8000"
echo "- Frontend: http://localhost:3000"
echo "- API Documentation: http://localhost:8000/docs"
echo ""
echo "To stop the application, run: ./scripts/local_stop.sh"
echo "Or press Ctrl+C"

# Wait for both processes
wait
