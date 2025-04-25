#!/bin/sh
module load Python/3.10.8-GCCcore-12.2.0.lua
# cd ~/source-code/PyEEG
cd /mnt/isilon/marsh_single_unit/PythonEEG
source .venv/bin/activate

python -u $1

echo "Pipeline finished."