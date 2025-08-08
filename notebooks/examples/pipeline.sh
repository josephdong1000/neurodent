#!/bin/bash

# Exit on error
set -e

# Load Python module
module load Python/3.10.8-GCCcore-12.2.0.lua

# Set working directory
cd /mnt/isilon/marsh_single_unit/PythonEEG

# Activate virtual environment
if [ ! -f .venv/bin/activate ]; then
    echo "Error: Virtual environment not found at .venv/bin/activate"
    exit 1
fi
source .venv/bin/activate

# Check if script argument is provided
if [ -z "$1" ]; then
    echo "Error: No Python script provided"
    echo "Usage: $0 <python_script.py>"
    exit 1
fi

# Run the Python script with unbuffered output
echo "Starting pipeline with script: $1"
python -u "$1"

echo "Pipeline finished successfully."