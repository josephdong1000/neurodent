#!/bin/sh
module load Python/3.10.8-GCCcore-12.2.0.lua
cd /mnt/isilon/marsh_single_unit/PythonEEG
source .venv-linux-3.10/bin/activate

python -u /mnt/isilon/marsh_single_unit/PythonEEG/notebooks/examples/pipeline-batch/pipeline-parallel-testing.py

echo "Pipeline finished."