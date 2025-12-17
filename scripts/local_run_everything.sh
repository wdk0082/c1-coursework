#!/bin/bash

# One-stop script to setup and run the application locally
# Runs simple_setup.sh followed by local_start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setup
"$SCRIPT_DIR/simple_setup.sh"

# Start
"$SCRIPT_DIR/local_start.sh"
