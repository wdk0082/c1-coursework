#!/bin/bash

# Local stop script for 5D Regression application
# This script stops both backend and frontend servers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Stopping 5D Regression application..."

# Kill backend if PID file exists
if [ -f "$PROJECT_ROOT/.backend.pid" ]; then
    BACKEND_PID=$(cat "$PROJECT_ROOT/.backend.pid")
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo "Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
    fi
    rm -f "$PROJECT_ROOT/.backend.pid"
fi

# Kill frontend if PID file exists
if [ -f "$PROJECT_ROOT/.frontend.pid" ]; then
    FRONTEND_PID=$(cat "$PROJECT_ROOT/.frontend.pid")
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    rm -f "$PROJECT_ROOT/.frontend.pid"
fi

# Also kill any processes on the ports as fallback
if command -v lsof &> /dev/null; then
    # Kill process on port 8000 (backend)
    BACKEND_PORT_PID=$(lsof -ti:8000 2>/dev/null || true)
    if [ -n "$BACKEND_PORT_PID" ]; then
        echo "Stopping process on port 8000..."
        kill $BACKEND_PORT_PID 2>/dev/null || true
    fi

    # Kill process on port 3000 (frontend)
    FRONTEND_PORT_PID=$(lsof -ti:3000 2>/dev/null || true)
    if [ -n "$FRONTEND_PORT_PID" ]; then
        echo "Stopping process on port 3000..."
        kill $FRONTEND_PORT_PID 2>/dev/null || true
    fi
fi

echo "Application stopped."
