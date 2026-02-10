#!/bin/bash
#SBATCH --job-name=check_rhd_202_all
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=defq

source .venv/bin/activate

BASE_DIR="/mnt/isilon/marsh_single_unit/PythonEEG Data/AP3B2/Intan recordings/Corrupted file_PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251216_143420"

python check_rhd_integrity.py "$BASE_DIR"
