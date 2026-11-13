#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=80000
#SBATCH --time=05:00:00
# One instrumented reproduction run: the node monitor in the background, then gather_hang.py.
#
#     sbatch --job-name=NAME [--nodelist=NODE] --output=LOG \
#         --export=ALL,GATHER_PY=/path/to/python \
#         gather_hang.sbatch TREE OUT_ROOT MODE [gather_hang.py options]
#
# TREE is the worktree whose src and scripts run (PYTHONPATH is set to its src), OUT_ROOT the
# folder for this run's directory, NAME_JOBID. GATHER_PY is the interpreter, by default
# TREE/.venv/bin/python. The log ends with the counts of the worker messages that mark the hang,
# then "exit code: N", where 3 means the watchdog ended a stuck run.

set -u
TREE=${1:?tree}; OUT_ROOT=${2:?output root}; shift 2
OUT=$OUT_ROOT/${SLURM_JOB_NAME:-gather}_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
unset VIRTUAL_ENV
PY=${GATHER_PY:-$TREE/.venv/bin/python}
export PYTHONPATH=$TREE/src

echo "host: $(hostname)  start: $(date -Is)  nproc: $(nproc)"
echo "tree: $TREE  commit: $(git -C "$TREE" rev-parse --short HEAD)  out: $OUT"
"$PY" -c 'import neurodent; print("neurodent from", neurodent.__file__)'

bash "$TREE/scripts/diagnostics/node_monitor.sh" "$OUT" &
MONITOR=$!
"$PY" "$TREE/scripts/diagnostics/gather_hang.py" "$@" --out "$OUT"
CODE=$?
kill "$MONITOR" 2>/dev/null; wait "$MONITOR" 2>/dev/null

LOG=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/.*StdOut=//p')
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    for pattern in "failed during get data" "Couldn't gather" "unresponsive" "Removing worker" "Lost connection"; do
        echo "log lines with '$pattern': $(grep -c "$pattern" "$LOG")"
    done
fi
echo "end: $(date -Is)"
echo "exit code: $CODE"
exit "$CODE"
