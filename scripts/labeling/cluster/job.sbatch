#!/bin/bash
#SBATCH --job-name=autoreject
#SBATCH --output=results/labeling/logs/%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00

# Run one command of the autoreject evaluation in the port or the July tree (scripts/labeling/CLUSTER_RUNBOOK.md)
# and record where and with which code it ran. Submit from a shell that sourced env.sh, which changes to $PORT
# and exports what sbatch and this script read:
#
#     sbatch --parsable --job-name=<name> --cpus-per-task=<n> --mem=<m> --time=<t> --output="$LOGS/%x_%j.log" \
#         "$JOB" port|july <command> [<argument> ...]
#
# The command's words arrive as the submitting shell expanded them. The job sources env.sh again, changes to the
# tree, unsets VIRTUAL_ENV, sets PYTHONPATH to the tree's src, prints the host, the date, the CPUs, the
# allocation, the tree's git HEAD and status and the neurodent it imports, runs the command, and prints and
# returns its exit code. The resources above are defaults; the runbook gives each job its own.

set -uo pipefail   # no -e: the command's exit code is printed and returned below

if [ "$#" -lt 2 ]; then
    echo "job.sbatch: usage: job.sbatch port|july <command> [<argument> ...]" >&2
    exit 2
fi
tree=$1
shift

env_file=${AUTOREJECT_ENV:-${SLURM_SUBMIT_DIR:-$PWD}/scripts/labeling/cluster/env.sh}
# shellcheck source=env.sh
if ! source "$env_file"; then
    echo "job.sbatch: could not source $env_file" >&2
    exit 2
fi

case $tree in
    port) dir=$PORT ;;
    july) dir=$JULY ;;
    *) echo "job.sbatch: the tree is port or july, not '$tree'" >&2; exit 2 ;;
esac
cd "$dir" || exit 2
unset VIRTUAL_ENV
export PYTHONPATH=$dir/src
export PYTHONUNBUFFERED=1

echo "host: $(hostname)"
echo "date: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "nproc: $(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"
echo "allocation: job ${SLURM_JOB_ID:-none} (${SLURM_JOB_NAME:-}), partition ${SLURM_JOB_PARTITION:-?}," \
     "account ${SLURM_JOB_ACCOUNT:-?}, cpus per task ${SLURM_CPUS_PER_TASK:-?}," \
     "memory per node ${SLURM_MEM_PER_NODE:-?} MB, nodes ${SLURM_JOB_NODELIST:-?}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    squeue -h -j "$SLURM_JOB_ID" -o 'time limit %l, cpus %C, memory %m' 2>/dev/null || true
fi
echo "tree: $tree $dir"
echo "git HEAD: $(git -C "$dir" rev-parse HEAD 2>&1)"
echo "git status (the lines up to 'git status end'; none means clean):"
git -C "$dir" status --porcelain 2>&1
echo "git status end"
if [ -x "$PY" ]; then
    "$PY" -c 'import neurodent; print("neurodent:", neurodent.__file__)' 2>&1
else
    echo "neurodent: $PY does not exist yet"
fi
printf 'command:'
printf ' %q' "$@"
printf '\n----\n'

"$@"
rc=$?

printf -- '----\n'
echo "date: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "exit code: $rc"
exit "$rc"
