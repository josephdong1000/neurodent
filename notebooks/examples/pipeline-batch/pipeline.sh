#!/bin/sh
module load Python/3.10.8-GCCcore-12.2.0.lua
cd ~/source-code/PyEEG
source .venv/bin/activate

python -u ./notebooks/examples/pipeline-batch/pipeline.py
# python -m cProfile -o pipeline2.prof ./notebooks/examples/pipeline-batch/pipeline-daskspeedup.py
# snakeviz pipeline.prof

echo "Pipeline finished."