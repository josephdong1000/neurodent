#!/bin/bash
#SBATCH --job-name=check_rhd_202_select
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=defq

source .venv/bin/activate

BASE_DIR="/mnt/isilon/marsh_single_unit/PythonEEG Data/AP3B2/Intan recordings/Corrupted file_PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251216_143420"

# First 5 files
F1="${BASE_DIR}/PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251216_143420.rhd"
F2="${BASE_DIR}/PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251216_150420.rhd"
F3="${BASE_DIR}/PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251216_153420.rhd"
F4="${BASE_DIR}/PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251216_160420.rhd"
F5="${BASE_DIR}/PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251216_163420.rhd"

# Last 5 files
L1="${BASE_DIR}/PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251217_000421.rhd"
L2="${BASE_DIR}/PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251217_003421.rhd"
L3="${BASE_DIR}/PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251217_010421.rhd"
L4="${BASE_DIR}/PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251217_013421.rhd"
L5="${BASE_DIR}/PortA-AP3B2wt-202-F-PortB-AP3B2homo-203-F-PortC-AP3B2het-237-F-PortD-dead-standardEEG 12-16-25_251217_020422.rhd"

python check_rhd_integrity.py "$F1" "$F2" "$F3" "$F4" "$F5" "$L1" "$L2" "$L3" "$L4" "$L5"
