#!/bin/bash
#SBATCH --job-name=cohort_bundle
#SBATCH --output=slurm-%x-%j.out          # detailed log in the submit dir (repo root)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --nodes=1
# Add --partition=... / --account=... below (or on the sbatch command line) if your cluster needs them.
##SBATCH --partition=
##SBATCH --account=

# Submit blinded human-labeling cohort bundles on the cluster.
#
# Usage (submit from the REPO ROOT so dataset extract_func paths resolve):
#     sbatch scripts/labeling/run_cohort_bundle.sbatch --dataset arx_parv --animal 29 --n-per-animal 40
#     sbatch scripts/labeling/run_cohort_bundle.sbatch --dataset arx_parv --n-per-animal 40   # whole strain
#
# Every argument after the script name is forwarded to build_cohort_bundle.py, so all its flags
# (--dataset --animal --n-per-animal --seed --blind-seed --out --dry-run --no-bundle) work here.
#
# Override resources per submission without editing this file, e.g.:
#     sbatch --mem=64G --time=12:00:00 scripts/labeling/run_cohort_bundle.sbatch --dataset sox5_bin --n-per-animal 30
#
# Tip: run the cheap discovery check on the dev box first (no SLURM needed):
#     uv run python scripts/labeling/build_cohort_bundle.py --dataset arx_parv --dry-run

set -euo pipefail

# sbatch starts in the submission directory; assert it is the repo root.
if [[ ! -f config/config.yaml ]]; then
    echo "ERROR: submit from the repo root (config/config.yaml not found in $(pwd))." >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-?}  started=$(date -Is)"
echo "args: $*"

# Capture the exit code so a partial run (some animals skipped/failed -> rc=1) still logs a clean
# finish; the driver writes bundles + keymap regardless.
rc=0
uv run python scripts/labeling/build_cohort_bundle.py "$@" || rc=$?

echo "finished=$(date -Is)  exit=$rc"
exit $rc
