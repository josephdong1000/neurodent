#!/bin/sh
module load Python/3.10.8-GCCcore-12.2.0.lua
cd ~/source-code/PyEEG
source .venv/bin/activate

python -u ./notebooks/examples/pipeline-war-nspike.py

echo "Pipeline finished."