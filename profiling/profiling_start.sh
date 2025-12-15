#!/bin/bash

# Navigate to the profiling directory
cd "$(dirname "$0")"

# Run the profiling script
python run_profiling.py

# Run the plotting script
python plot_profiling_results.py