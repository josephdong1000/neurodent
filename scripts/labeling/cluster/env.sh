# shellcheck shell=bash
# Settings of the autoreject evaluation on the cluster, for scripts/labeling/CLUSTER_RUNBOOK.md.
#
# Every runbook block and every job sources this file:
#
#     source <port worktree>/scripts/labeling/cluster/env.sh || exit 1
#
# It takes the port tree from its own location and the values found on the cluster (MAIN, TIP, JULY, TMP and the
# sbatch partition and account) from $OUT/site.sh, which runbook step 0 writes. It exports what sbatch and the
# jobs read (TMP not among them), sets PYTHONPATH to the port's src and unsets VIRTUAL_ENV for commands on the
# login node, defines the pre-registered inputs as bash arrays and a few helpers, creates $LOGS and changes to
# $PORT. It returns 1, after defining everything it can, when site.sh is missing or leaves MAIN, TIP, JULY or TMP
# empty.

if [ -z "${BASH_VERSION:-}" ]; then
    echo "env.sh: source this file from bash" >&2
    return 1 2>/dev/null || exit 1
fi

AUTOREJECT_ENV=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/env.sh
PORT=$(cd "$(dirname "$AUTOREJECT_ENV")/../../.." && pwd -P)
OUT=$PORT/results/labeling          # gitignored with results/
LOGS=$OUT/logs                      # one log per job, <job name>_<job id>.log
SITE=$OUT/site.sh
PY=$PORT/.venv/bin/python           # the port's venv; the July tree runs it with PYTHONPATH=$JULY/src
JOB=$PORT/scripts/labeling/cluster/job.sbatch
JULY_COMMIT=413535429de06a92341471495678e8c8dab7ee14

# The rater files of config/labeling/preregistration.yaml, relative to $PORT, where every script runs.
DEV=(JD=scripts/labeling/labels/labels_cohort4strains_JD_dev.csv
     JL=scripts/labeling/labels/labels_cohort4strains_JL_dev.csv
     Yong=scripts/labeling/labels/labels_cohort4strains_Yong_dev.csv)
TEST=(JD=scripts/labeling/labels/labels_cohort4strains_JD_test.csv
      JL=scripts/labeling/labels/labels_cohort4strains_JL_test.csv
      Yong=scripts/labeling/labels/labels_cohort4strains_Yong_test.csv)
# The 6 named configurations of the label-free checks (adr.CONFIG_NAMES, spelled as grid names).
NAMED=(autoreject/self autoreject/self+CMR welch/v2/self@nf40_na100 welch/v2/self+CMR@nf40_na100
       welch/v3/self@nf40_na100 welch/v3/self+CMR@nf40_na100)
# The label-free draw: 2 animals per dataset, seed 0.
LABEL_FREE_DATASETS=(arx_parv arx_rosa sox5_bin ap3b2_rhd iqsec2)

MAIN=
TIP=
JULY=
TMP=
SBATCH_PARTITION=
SBATCH_ACCOUNT=
if [ -f "$SITE" ]; then
    # shellcheck source=/dev/null
    source "$SITE"
fi
VALIDATED=${TMP:+$TMP/hash_port}    # the port hash run's temp directory, read by every later step

export AUTOREJECT_ENV PORT OUT LOGS PY JOB JULY_COMMIT MAIN TIP JULY VALIDATED
# TMP stays a shell variable, also when the login profile exported one: in the environment, Python's tempfile
# would read it and put dask's spill directory and pytest's directories under the evaluation's scratch.
export -n TMP
export PYTHONPATH=$PORT/src         # for the login node; job.sbatch sets the tree's own
unset VIRTUAL_ENV
export SBATCH_EXPORT=ALL            # jobs inherit this environment, PATH and AUTOREJECT_ENV included
if [ -n "$SBATCH_PARTITION" ]; then export SBATCH_PARTITION; else unset SBATCH_PARTITION; fi
if [ -n "$SBATCH_ACCOUNT" ]; then export SBATCH_ACCOUNT; else unset SBATCH_ACCOUNT; fi

# jobwait ID: wait until the job has left the queue, then print its accounting. Run it in the background.
jobwait() {
    while squeue -h -j "$1" 2>/dev/null | grep -q .; do
        sleep 60
    done
    jobacct "$1"
}

# jobacct ID [ID ...]: state, exit code, elapsed time, CPUs, requested and peak memory of jobs.
jobacct() {
    if [ "$#" -eq 0 ]; then
        echo "jobacct: give job ids" >&2
        return 2
    fi
    sacct -j "$(IFS=,; echo "$*")" \
        --format=JobID,JobName%20,Partition,State,ExitCode,Elapsed,Timelimit,NCPUS,ReqMem,MaxRSS,NodeList%20
}

# joblog ID: the path of the job's log.
joblog() {
    ls "$LOGS"/*_"$1".log
}

# intermediates: the MNE intermediates in $VALIDATED, one line each: name, size in bytes, modification time.
intermediates() {
    find "$VALIDATED" -maxdepth 1 -type f -name '*_mne-to-rec*' ! -name '*.tmp' -printf '%f\t%s\t%T@\n' \
        | LC_ALL=C sort
}

# check_intermediates: compare $VALIDATED with the listing saved in step 2. Returns 1 when a validated file is
# missing or changed; files added since (the label-free animals' intermediates in step 8) are only counted.
check_intermediates() {
    local ref=$OUT/validated_intermediates.txt now gone
    if [ ! -f "$ref" ]; then
        echo "check_intermediates: no $ref; runbook step 2 writes it" >&2
        return 2
    fi
    now=$(mktemp) || return 2
    intermediates > "$now"
    gone=$(LC_ALL=C comm -23 "$ref" "$now")
    if [ -n "$gone" ]; then
        echo "validated intermediates missing or changed (name, size, modification time as validated):"
        echo "$gone"
        rm -f "$now"
        return 1
    fi
    echo "validated intermediates unchanged: $(wc -l < "$ref" | tr -d ' ') files; lines added since step 2:" \
         "$(LC_ALL=C comm -13 "$ref" "$now" | wc -l | tr -d ' ')"
    rm -f "$now"
}

mkdir -p "$LOGS"
cd "$PORT" || return 1

_env_missing=
for _env_name in MAIN TIP JULY TMP; do
    [ -n "${!_env_name}" ] || _env_missing="$_env_missing $_env_name"
done
unset _env_name
if [ -n "$_env_missing" ]; then
    echo "env.sh: $SITE does not set:$_env_missing. Write it as runbook step 0 says." >&2
    unset _env_missing
    return 1
fi
unset _env_missing
