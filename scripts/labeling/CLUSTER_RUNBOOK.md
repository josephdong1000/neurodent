# Cluster runbook: autoreject evaluation

This is the procedure for a Claude Code session on the SLURM cluster that holds the recordings, picking up the
autoreject evaluation from a handoff. Follow it in order and literally. The spec is
`config/labeling/preregistration.yaml` with its amendment of 2026-11-07; read it in full before step 0, and if
it has no amendment of that date, ask the owner. `scripts/labeling/README.md` gives the evaluation order and the
combinations, and the design write-up is https://claude.ai/artifact/Q5NSyG1BjMbJvPJQTYdPc7. Until step 0.3 has
made the port tree, read this file and the pre-registration with `git -C <MAIN> fetch origin feat/autoreject-port`
and then `git -C <MAIN> show <TIP>:scripts/labeling/CLUSTER_RUNBOOK.md` (MAIN and TIP are explained in steps 0.1
and 0.2).

The owner is asked at two stops: Stop A after the hash check (step 2) and Stop B after the selection (step 5).
Step 7, the test score that can run only once, waits for the owner's go after Stop B. A check-in before step 2
is needed only when step 1 finds a problem. When something fails in a way a step does not cover, stop and ask
the owner rather than work around it.

## How the blocks run

- Every block is bash. Your Bash tool keeps no variables between calls, so every block after step 0.6 starts
  by sourcing `env.sh`, which also changes to `$PORT`. If `echo "$BASH_VERSION"` prints nothing in your tool,
  write each block to a file and run it with `bash <file>`.
- A foreground command must finish within 10 minutes. Anything heavier is a job: submit it with `sbatch`
  through `job.sbatch`, never with `srun` and never on the login node, so that its log can be read afterwards.
  The login node runs only git, listings, the comparison in step 2 and short Python checks that read small
  files.
- The blocks show PORT at its default path, `/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject`.
  If the owner gives another path, use it in every `source` line.
- Replace every `<...>` with its value before running a block; nothing in angle brackets may remain.
- Every `--cpus-per-task`, `--mem` and `--time` below is a guess, not a measurement. After each job, read its
  elapsed time and peak memory with `jobacct` and report them. When a job needs a larger limit, raise it, say
  so in the report, and keep the command itself unchanged.

## Files

- `scripts/labeling/cluster/env.sh`: sourced by every block and every job. It takes `PORT` from its own
  location, sets `OUT=$PORT/results/labeling` (gitignored), `LOGS=$OUT/logs`, `PY=$PORT/.venv/bin/python`,
  `JOB` (the path of `job.sbatch`) and `VALIDATED=$TMP/hash_port`, reads `MAIN`, `TIP`, `JULY`, `TMP`,
  `SBATCH_PARTITION` and `SBATCH_ACCOUNT` from `$OUT/site.sh`, and defines the bash arrays `DEV`, `TEST`
  (the rater files, `rater=path`), `NAMED` (the 6 named configurations) and `LABEL_FREE_DATASETS`. It exports
  `SBATCH_EXPORT=ALL`, `SBATCH_PARTITION` and `SBATCH_ACCOUNT`, which sbatch reads from the environment, sets
  `PYTHONPATH=$PORT/src` and unsets `VIRTUAL_ENV` for the login node, and defines the helpers below. `TMP`
  stays a shell variable, not exported, so that Python's own temporary files (dask's spill directory, pytest's
  directories) go to the node's `TMPDIR` or `/tmp` rather than under `TMP`.
- `scripts/labeling/cluster/job.sbatch`: the one job wrapper. `"$JOB" port|july <command>` sources env.sh,
  changes to the tree, unsets `VIRTUAL_ENV`, sets `PYTHONPATH` to the tree's `src`, prints the host, date,
  `nproc`, the allocation, the tree's git HEAD and status and `neurodent.__file__`, runs the command, then
  prints `exit code: N` and returns N.
- `scripts/labeling/cluster/bootstrap.sh`: creates or verifies the two worktrees (step 0.3).
- `$OUT/site.sh`: the values found on the cluster, written in step 0.6.
- `$OUT/jobs.txt`: the id of every job submitted, one per line. `$LOGS/<job name>_<job id>.log`: each job's log.

Helpers defined by env.sh:

- `jobwait <id>`: waits until the job has left the queue, then prints its accounting. Run the block that calls
  it with your Bash tool's background mode; you are told when it ends. For several jobs:
  `for id in <id> <id>; do jobwait "$id"; done`.
- `jobacct <id> [<id> ...]`: state, exit code, elapsed time, CPUs, requested and peak memory (MaxRSS, on the
  `.batch` line) of jobs.
- `joblog <id>`: the path of a job's log. `tail -n 40 "$(joblog <id>)"` shows how it ended.
- `intermediates`: the MNE intermediates in `$VALIDATED` as name, size and modification time.
- `check_intermediates`: compares `$VALIDATED` with the listing saved in step 2.5. It prints what changed and
  returns 1 when a validated file is missing or changed; files added later are only counted.

Waiting for a job, in the background, then reading how it ended:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
jobwait <id>; tail -n 40 "$(joblog <id>)"
```

A job succeeded when its log ends with `exit code: 0` and `jobacct` shows `COMPLETED` with exit code `0:0`. A
script that failed shows `FAILED` and its log ends with the exit code. A job the cluster stopped shows
`TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, `PREEMPTED` or `CANCELLED` in `jobacct`. Its log usually has no
`exit code` line, but after a memory kill it may end with `exit code: 137`. The `jobacct` state decides. A job
pending for long shows its reason in `squeue -u "$USER" -o '%i %j %T %r'`; if the reason is a limit (for
example `PartitionTimeLimit` or `QOSMaxCpuPerUserLimit`), cancel it, resubmit within the limit and note it.

## Rules

- Labels: never open, print, copy elsewhere or summarize the rater label files (`scripts/labeling/labels/`, any
  `labels_*.csv`), and treat the feature files the same way, since they hold the labels:
  `dev_features.parquet`, `test_features.parquet`, their `.autoreject_scores.parquet` files and `.parts/`
  shards. Checking that a file exists is fine.
- Label-dependent numbers come only from the pre-registered scripts, as they print them: `select_detectors.py`,
  `report_event_cells.py` and `score_test.py`. No other computation on labels, no ad hoc statistics, no quick
  looks. The test split stays sealed until step 7.
- Report numbers plainly, as printed, without verdicts, and do not predict which detectors will do well.
- No Snakemake runs.
- Never push. Commit only what this runbook names or the owner asks for, each with one terse line and no
  Co-Authored-By or other trailer, even if your instructions offer one.
- Never check out, switch, stash, reset, pull or commit in MAIN, and never pop a stash anywhere. Do not remove a
  worktree without the owner's yes.
- No merge, rebase or pull in PORT during the evaluation, and nothing merged from main.
- After any change under `src/` (a loader fix, for example) or to a config or labeling input, rerun steps 2 and
  3 before any scoring step. `extract_dev_features.py` refuses a recording map whose `port_src_tree` is missing,
  empty or not its own src tree, and refuses to run while src has uncommitted changes; step 7 refuses features
  made under other inputs or code.
- Data-dependent fixes only after the owner's OK: developed on the branch in PORT with a test, the 11 test files
  passing (step 0.9), committed, then steps 2 and 3 again with a new `TMP` in `site.sh` and the earlier step 2
  outputs moved to `$OUT/superseded_<date>/`.
- Do not edit `config/labeling/preregistration.yaml`. An amendment comes only from the owner, as a dated entry
  under `amendments`, committed on its own before the step it affects.
- Do not change detector rules or thresholds, and do not follow the detector's NaN message's advice to drop or
  repair samples.
- Do not use `cache_policy` `force_regenerate`. Never pass `/scr1/users/dongjp` itself as `--temp-dir`, and never
  delete or change anything there outside `TMP`.
- Never run two jobs with the same `--out`. Before resubmitting, check `squeue -u "$USER"`.
- Messages to the owner are short and in plain prose (no bold, no em dashes; write autoreject, never AR). Before
  asking a question, say in a sentence or two what it is about; the owner may not remember the details.

## Order and overlap

- Step 0: the venv job before the tests job. The TMP check any time after `site.sh` exists.
- Step 1: all its jobs may run at the same time, once the tests of step 0 passed. The readable jobs (1.6) may
  still be running during step 2.
- Step 2: the two preflight hash jobs together; the two full runs together, once both preflight jobs exited 0
  and any check-in was answered; the comparison after both full runs ended; the optional scan after the
  comparison.
- Stop A, then step 3, then step 4. While step 4 runs, nothing in PORT may change (no edit, commit, checkout or
  pull), or it refuses and writes nothing.
- Step 5 after step 4 succeeded. After Stop B's report, step 6 and step 8 may start; step 7 waits for the owner's
  go and for step 6.
- Step 8: rejection rates first; the injection jobs after it finished, in parallel with each other. Step 8 may
  overlap steps 6 and 7.
- Step 9 last.

## Scratch

- `VALIDATED` holds the intermediates every scoring step reads: never edit or delete its `*_mne-to-rec*` files
  until the owner signs off after step 9.
- After a job was killed, and only while `squeue -u "$USER"` lists none of your jobs, delete the leftovers of
  partial writes directly in that run's temp directory: `temp_*.zarr` directories and `*.tmp` files (list them
  with `find "$VALIDATED" -maxdepth 1 \( -name 'temp_*.zarr' -o -name '*.tmp' \)` first).
- `$TMP/hash_july`, `$TMP/prehash_july`, `$TMP/prehash_port` and `$TMP/scan_port` are not read after Stop A;
  delete them only when the owner says so.

## 0. Setup

### 0.1 Shell, tools and the tip

```bash
echo "bash ${BASH_VERSION:-missing}"; git --version; command -v sbatch squeue sacct sinfo uv
```

Expect a bash version, git 2.17 or later (step 0.9's tests run `git worktree remove`; if it is older, ask the
owner) and a path for each of the five commands. If uv is missing, ask the owner. TIP is the full 40-character
hash of the commit of `origin/feat/autoreject-port` that the handoff names.

### 0.2 MAIN

MAIN is the owner's clone that holds the `neurodent_wt` worktrees. Find it from an existing worktree (only
`rev-parse` runs there):

```bash
WT=/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/aggregate
ls /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt
GCD=$(cd "$WT" && cd "$(git rev-parse --git-common-dir)" && pwd -P) \
    && echo "git directory: $GCD" && echo "MAIN: ${GCD%/.git}"
```

If `aggregate` is missing, use another worktree from the listing. MAIN is the git directory without its trailing
`/.git`; if it does not end in `/.git`, ask the owner. Then:

```bash
MAIN=<MAIN>
git -C "$MAIN" config --get remote.origin.url; git -C "$MAIN" worktree list
```

`origin` must point to github.com/josephdong1000/neurodent (https or ssh); otherwise ask the owner.

### 0.3 Worktrees

PORT becomes a worktree of `feat/autoreject-port` at TIP and JULY a detached worktree at 41353542 (the July
tree). The bootstrap fetches into remote-tracking refs only, creates only the branch `feat/autoreject-port`,
checked out in PORT, and changes no other branch, nor MAIN's index or files:

```bash
MAIN=<MAIN>; TIP=<TIP>
git -C "$MAIN" fetch origin "+refs/heads/feat/autoreject-port:refs/remotes/origin/feat/autoreject-port" \
    && B=$(mktemp) && git -C "$MAIN" show "$TIP:scripts/labeling/cluster/bootstrap.sh" > "$B" \
    && bash "$B" "$MAIN" "$TIP"; echo "exit code: $?"; rm -f "$B"
```

The defaults are PORT `/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject` and JULY
`/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/july`; other paths, only from the owner, go as third and
fourth arguments. Success: a line starting with `ready:`, then the worktree list, and `exit code: 0`. Anything
else, a line starting with `bootstrap:` that names what it did not expect (an existing path or branch, another
commit, a dirty tree, a `config.local.yaml`) or a git error such as a failed fetch or a TIP that has no
`scripts/labeling/cluster/bootstrap.sh`: report it to the owner and wait. Never work around it with `--force`,
`-B`, a reset or by deleting a path.

A session that resumes the evaluation later, once `site.sh` exists, verifies the trees again. The bootstrap
accepts PORT at TIP or at TIP plus the recording map commit of step 3; anything else (an amendment or a fix the
owner approved, for example) makes it stop, and the owner confirms the tree:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
bash "$PORT/scripts/labeling/cluster/bootstrap.sh" "$MAIN" "$TIP" "$PORT" "$JULY"; echo "exit code: $?"
```

Between step 2.4 and the map commit of step 3, the bootstrap stops on one status line,
`?? config/labeling/recording_map.csv`, before it checks JULY. That stop alone is expected and needs no report;
check the rest by hand:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
sha256sum -c "$OUT/recording_map.sha256"; echo "PORT at $(git -C "$PORT" rev-parse HEAD), TIP $TIP"
echo "JULY at $(git -C "$JULY" rev-parse HEAD)"; git -C "$JULY" status --porcelain
```

`sha256sum -c` must print `OK`, PORT must be at TIP, JULY at 413535429de06a92341471495678e8c8dab7ee14, and
JULY's status must print nothing. Then go on from where the evaluation stands, usually waiting at Stop A. Report
anything else to the owner.

### 0.4 Partition, account and limits

```bash
sinfo -s; sinfo -o '%P %a %l %D %c %m'; sacctmgr -n -P show assoc user="$USER" format=account,partition,qos
```

Use the default partition (marked `*`) and the one account listed, unless the owner says otherwise. The jobs
below need one node with 32 CPUs and 128 GB (memory column, in MB, at least 131072, since sbatch reads 128G as
131072 MB) and wall times up to 48:00:00, or 72:00:00 for step 8's rejection rates. If several accounts are
listed, or the limits are lower, ask the owner.

### 0.5 TMP

TMP holds every run's temp directory and must be visible from every node and not purged during the evaluation,
which may last weeks.

```bash
df -hT /scr1/users/dongjp; findmnt -T /scr1/users/dongjp; ls -ld /scr1/users/dongjp/autoreject_eval 2>&1
```

A network or parallel filesystem type (nfs, nfs4, gpfs, lustre, beegfs, cifs) suggests shared storage; xfs, ext4
or tmpfs on a local device suggests storage local to the node. Step 0.7 decides. Use
`TMP=/scr1/users/dongjp/autoreject_eval`, which must not exist yet or be empty; if it holds files, ask the owner
and do not delete them.

### 0.6 site.sh

Write the values, with every `<...>` replaced (the account and partition may stay empty when the cluster needs
none):

```bash
mkdir -p /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/results/labeling
cat > /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/results/labeling/site.sh <<'EOF'
# Values found on the cluster in runbook step 0; env.sh reads them.
MAIN=<MAIN>
TIP=<TIP>
JULY=/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/july
TMP=/scr1/users/dongjp/autoreject_eval
SBATCH_PARTITION=<partition>
SBATCH_ACCOUNT=<account>
EOF
```

Check it, and that the six rater files exist (without opening them):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
grep -n '<' "$OUT/site.sh"; echo "PORT=$PORT OUT=$OUT VALIDATED=$VALIDATED PY=$PY"
echo "partition=${SBATCH_PARTITION:-default} account=${SBATCH_ACCOUNT:-none}"
for r in "${DEV[@]}" "${TEST[@]}"; do [ -f "${r#*=}" ] && echo "present: ${r#*=}" || echo "missing: ${r#*=}"; done
```

The `grep` must print nothing and all six files must be present.

### 0.7 TMP from a compute node

A short job checks that a compute node sees the same TMP (resources unmeasured):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
mkdir -p "$TMP" && hostname > "$TMP/.seen_from_login"
sbatch --parsable --job-name=tmp_check --cpus-per-task=1 --mem=1G --time=0:10:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port bash -c 'df -hT "$1"
        findmnt -T "$1"
        echo "login marker: $(cat "$1/.seen_from_login")"
        hostname > "$1/.seen_from_node"' tmp_check "$TMP" | tee -a "$OUT/jobs.txt"
```

When it has ended, `cat "$TMP/.seen_from_node"` on the login node must print the compute node's name, and the log
must show `login marker: <the login node's name>` and the same filesystem as step 0.5. If either is missing, TMP
is not shared: stop and ask the owner for a shared path that is not purged. Then
`rm -f "$TMP/.seen_from_login" "$TMP/.seen_from_node"`.

### 0.8 The venv

The venv is built from `uv.lock` with Python 3.10, which `.python-version` names:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
uv --version; uv python find 3.10
```

If uv finds no Python 3.10, ask the owner how to provide one (for example `uv python install 3.10`, or a module);
do not load a module without the owner's word. Then (resources unmeasured):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
sbatch --parsable --job-name=venv --cpus-per-task=4 --mem=16G --time=1:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port uv sync --all-extras --frozen | tee -a "$OUT/jobs.txt"
```

When it has ended with exit code 0:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
"$PY" -c 'import sys, neurodent; print(sys.version.split()[0], neurodent.__file__)'; git -C "$PORT" status --porcelain
```

Expect 3.10.x and `$PORT/src/neurodent/__init__.py`, and no status lines. On a failure (no interpreter, no
network on the node, a disk quota), report the log's last lines to the owner.

### 0.9 Tests

The 11 labeling test files, slow tests included (resources unmeasured):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
sbatch --parsable --job-name=tests --cpus-per-task=4 --mem=16G --time=2:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" -m pytest -q -rs -p no:cacheprovider tests/test_scoring.py tests/test_labeling_bundle.py \
    tests/test_cohort_labeling.py tests/test_detector_selection.py tests/test_evaluation_protocol.py \
    tests/test_hash_check.py tests/test_label_free_checks.py tests/test_autoreject_detector.py \
    tests/test_combinations.py tests/test_config_settings.py tests/test_extract_features.py | tee -a "$OUT/jobs.txt"
```

Expected: the summary line reads `N passed, 3 skipped` (warnings aside), with no `failed` or `error`; the 3 skips
are the browser tests in `tests/test_labeling_bundle.py`, which need playwright chromium (none are skipped if
chromium is installed). The log's header names `$PORT/src/neurodent/__init__.py`. Afterwards
`git -C "$PORT" status --porcelain` must print nothing. Any failure: stop and send the owner the failing tests
and their errors.

For each of two tests, `tests/test_hash_check.py` adds a detached worktree of the repository under pytest's temp
directory, registered in MAIN's git directory, and removes it afterwards. If the tests job fails or is killed,
report any line of `git -C "$MAIN" worktree list` whose path holds `port-worktree`, and do not prune it without
the owner's yes.

## 1. Preflight checks (data, no labels)

### 1.1 Discovery dry run

The loader's discovery and datetime checks for the 20 labeled animals (the July draw: 5 per dataset, seed 42),
without loading samples (resources unmeasured):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
mkdir -p "$OUT/preflight"
sbatch --parsable --job-name=dryrun --cpus-per-task=2 --mem=8G --time=1:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" scripts/labeling/build_cohort_bundle.py --all --limit-per-dataset 5 --seed 42 --dry-run \
    | tee -a "$OUT/jobs.txt"
```

Success: exit code 0 and four tables, `arx_parv  (5 animals)`, `arx_rosa  (5 animals)`, `sox5_bin  (5 animals)`
and `ap3b2_rhd  (5 animals)`, whose 20 ids are the labeled animals that this prints:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
"$PY" -c 'import json; print(*json.load(open("config/labeling/cohort_manifest.json"))["rendered_animals"], sep="\n")'
```

No line may say `with no files, unresolved channels or a loader validation error`. Otherwise: check-in.

### 1.2 Channel names of the Intan animals

AP3B2het-207-M and ArxRosa-2064 load Intan files split by channel. If channel names repeat across ports, the
current loader refuses them. Resources unmeasured:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
mkdir -p "$OUT/preflight"
cat > "$OUT/preflight/channel_names.py" <<'EOF'
"""Print the channel names of each recording of two Intan animals, as the loader reads them."""
import sys

sys.path.insert(0, "scripts/labeling")
import build_cohort_bundle as C

failed = False
for dataset, animal in [("ap3b2_rhd", "AP3B2het-207-M"), ("arx_rosa", "ArxRosa-2064")]:
    try:
        config, samples_config = C._prepare(dataset)
        ao = C.load_cohort_animal(samples_config, config, animal)
        for i, lro in enumerate(ao.long_recordings):
            names = [str(n) for n in lro.channel_names]
            print(f"{animal} recording {i} ({ao.animaldays[i]}): {len(names)} channels, {len(set(names))} distinct: "
                  f"{names}", flush=True)
    except Exception as e:
        failed = True
        print(f"{animal}: {type(e).__name__}: {e}", flush=True)
sys.exit(1 if failed else 0)
EOF
sbatch --parsable --job-name=channels --cpus-per-task=4 --mem=16G --time=1:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" "$OUT/preflight/channel_names.py" | tee -a "$OUT/jobs.txt"
```

Success: exit code 0, one line per recording, and on each line the two counts are equal. An error about repeated
or missing channel names means the five ap3b2 animals (and ArxRosa-2064, if it fails) need a loader fix in
`src/neurodent/loading/lro_merge.py`, a later code round: check-in.

### 1.3 config.local.yaml in MAIN

`load_dataset_config` merges a gitignored `config/config.local.yaml` between `config.yaml` and the dataset
config, so the July bundle may have loaded with one. The fresh worktrees have none.

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
ls -l "$MAIN/config/config.local.yaml" 2>&1
[ -f "$MAIN/config/config.local.yaml" ] && "$PY" -c 'import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
for k, v in d.items():
    print(k, sorted(v) if isinstance(v, dict) else v)' "$MAIN/config/config.local.yaml"
```

If it exists, its keys go to the owner at the check-in. Never copy it into PORT or JULY without the owner's word;
if the owner wants it, it goes into both trees before step 2.

### 1.4 July-era cache

July rendered from MNE intermediates cached in `/scr1/users/dongjp`, possibly written by older code. List them,
read-only:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
mkdir -p "$OUT/preflight"
find /scr1/users/dongjp -maxdepth 1 -type f -name '*_mne-to-rec*' -printf '%TY-%Tm-%Td %TH:%TM %s %f\n' \
    | sort > "$OUT/preflight/july_era_cache.txt"
wc -l < "$OUT/preflight/july_era_cache.txt"
head -n 3 "$OUT/preflight/july_era_cache.txt"; tail -n 3 "$OUT/preflight/july_era_cache.txt"
awk '{s += $3} END {printf "%.1f GB\n", s / 1e9}' "$OUT/preflight/july_era_cache.txt"
```

The count, the date range and the total size go to the owner at Stop A (the keymap was locked on 2026-07-24 at
18:17 -0400). Replaying these files with `--reuse-temp-dir` is the owner's call, not part of this runbook.

### 1.5 Scratch space and purge policy

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
df -hT "$TMP"; ls /scr1; grep -i -n -e purge -e scratch /etc/motd 2>/dev/null
```

Look for the purge policy of this filesystem in the cluster's documentation, a README under `/scr1` or the
message of the day. `VALIDATED` must survive until step 8 has finished. The space needed is unmeasured; as a
guide, two hash runs each write about the July-era total from 1.4, and step 8 adds the label-free animals'
sessions. If the policy is not documented or the free space is below that guide: check-in.

### 1.6 Readable files of the label-free draw

Step 8 loads 10 unlabeled animals for the first time, and one unreadable file stops its single job. The
validator opens every file a load would open, reading headers only, for every animal of a dataset (resources
unmeasured):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
mkdir -p "$OUT/preflight"
for ds in "${LABEL_FREE_DATASETS[@]}"; do
    sbatch --parsable --job-name="readable_$ds" --cpus-per-task=2 --mem=8G --time=8:00:00 \
        --output="$LOGS/%x_%j.log" "$JOB" port "$PY" scripts/iqsec2/validate_config.py "$ds" --readable \
        --json "$OUT/preflight/readable_$ds.json" | tee -a "$OUT/jobs.txt"
done
```

A job may exit 1 because some animal of its dataset fails; only the drawn animals matter here. When all five have
ended, print the draw (computed from the configs as `rejection_rates.py --draw 2 --seed 0` computes it) and each
drawn animal's result, on the login node:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
"$PY" - <<'EOF' 2>&1 | tee "$OUT/preflight/label_free_draw.txt"
import json
import sys

sys.path.insert(0, "scripts/labeling")
import rejection_rates as RR

datasets = ["arx_parv", "arx_rosa", "sox5_bin", "ap3b2_rhd", "iqsec2"]
plan, _ = RR.select_animals(datasets, draw=2, seed=0, labeled=RR.labeled_animals())
print("\n".join(RR.plan_lines(plan, draw=2, seed=0)))
for ds, animals in plan:
    with open(f"results/labeling/preflight/readable_{ds}.json") as fh:
        report = json.load(fh)
    found = {str(r["animal"]): r for r in report["discovery"]}
    opened = {str(r["animal"]): r for r in report["readable"] or []}
    for animal, _ in animals:
        d, r = found.get(str(animal)), opened.get(str(animal))
        if d is None or r is None:
            print(f"{ds} {animal}: not in the report")
        elif not d["ok"]:
            print(f"{ds} {animal}: discovery failed: {d['error']}")
        elif not r["checked"]:
            print(f"{ds} {animal}: not checked: {r.get('reason')}")
        else:
            issues = r["aborts"] + r["problems"] + r["drops"]
            print(f"{ds} {animal}: {'failed' if r['failed'] else 'ok'}, {r['n_files']} files, "
                  f"{len(r['aborts'])} would stop the load, {len(r['problems'])} inconsistent, "
                  f"{len(r['drops'])} would be dropped" + (f"; first: {issues[0]}" if issues else ""))
EOF
```

Success: each of the 10 drawn animals is `ok` with 0 that would stop the load and 0 inconsistent. This result goes
to the owner at Stop A whatever it shows; it concerns step 8 only and does not hold up step 2.

### Check-in before step 2 (only if needed)

Send the owner a short message and wait for the answer if any of these holds: the dry run (1.1) did not list
exactly the 20 labeled animals or flagged one; the channel check (1.2) failed; MAIN has a `config.local.yaml`
(1.3); TMP's purge policy is unknown or its free space is short (1.5). Otherwise go on to step 2.

## 2. Hash check

The labels name recordings by their July position. This step hashes the samples behind every labeled cell under
the July tree and the port, and writes `config/labeling/recording_map.csv`, which ties each labeled recording to
a port recording and records the port's src tree (`port_src_tree`). Each run has its own empty temp directory.

### 2.1 Preflight hash runs

Three animals that load through SpikeInterface, so no MNE conversion: two Intan recordings split by channel and
one bin/csv animal. Both trees; the two jobs may run at the same time (resources unmeasured):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
sbatch --parsable --job-name=prehash_july --cpus-per-task=4 --mem=32G --time=6:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" july "$PY" "$PORT/scripts/labeling/hash_labeled_windows.py" --tree july \
    --animal ArxRosa-2064 AP3B2het-207-M AM2 --out "$OUT/preflight/hashes_july_3.csv" --temp-dir "$TMP/prehash_july" \
    | tee -a "$OUT/jobs.txt"
sbatch --parsable --job-name=prehash_port --cpus-per-task=4 --mem=32G --time=6:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" scripts/labeling/hash_labeled_windows.py --tree port \
    --animal ArxRosa-2064 AP3B2het-207-M AM2 --out "$OUT/preflight/hashes_port_3.csv" --temp-dir "$TMP/prehash_port" \
    | tee -a "$OUT/jobs.txt"
```

Success: both exit 0, each log's script output ends with
`wrote N rows (M hashed, 0 read errors) for 3 of 3 animal(s) to ...`, where M is below N only when a channel is
absent or duplicated at some position, and then the job's `exit code: 0`, and neither log has an `ERROR` line.
The July log's header shows git HEAD 413535429de06a92341471495678e8c8dab7ee14 and neurodent from `$JULY/src`;
the port log shows TIP and `$PORT/src`. If either exits non-zero, send the owner the `ERROR` lines of both logs
and wait before 2.2.

### 2.2 Full runs

Two separate jobs, which may run at the same time; never put both runs in one job under `set -e`, since a run
that exits 1 would stop the other (resources unmeasured):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
sbatch --parsable --job-name=hash_july --cpus-per-task=4 --mem=64G --time=24:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" july "$PY" "$PORT/scripts/labeling/hash_labeled_windows.py" --tree july \
    --out "$OUT/hashes_july.csv" --temp-dir "$TMP/hash_july" | tee -a "$OUT/jobs.txt"
sbatch --parsable --job-name=hash_port --cpus-per-task=4 --mem=64G --time=24:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" scripts/labeling/hash_labeled_windows.py --tree port \
    --out "$OUT/hashes_port.csv" --temp-dir "$VALIDATED" | tee -a "$OUT/jobs.txt"
```

Each run writes `<out>.partial` as it goes and renames it to `<out>` when every animal has been tried, so a
leftover `.partial` means the run did not finish. Success: exit 0 and, as the script's last line,
`wrote N rows (M hashed, 0 read errors) for 20 of 20 animal(s) to ...`, with M as in 2.1. Exit 1 means an
animal failed to load or some cells could not be read (`ERROR` lines name them); the run is finished, so do not
rerun it: go on to the comparison, which counts those cells as not hashed, and report them at Stop A.

### 2.3 Retry rule

A run the cluster stopped (`TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, `PREEMPTED` or `CANCELLED` in `jobacct`,
whatever the log's last line, `<out>.partial` left) is resubmitted with the same command plus `--reuse-temp-dir`,
on its own temp directory. Raise `--mem` after `OUT_OF_MEMORY` and `--time` after `TIMEOUT` (the figures in 2.2
are unmeasured). Never give a retry the other tree's directory or a new one; `VALIDATED` stays `$TMP/hash_port`.
For the port run:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
squeue -u "$USER"
sbatch --parsable --job-name=hash_port --cpus-per-task=4 --mem=<64G or more> --time=<24:00:00 or more> \
    --output="$LOGS/%x_%j.log" "$JOB" port "$PY" scripts/labeling/hash_labeled_windows.py --tree port \
    --out "$OUT/hashes_port.csv" --temp-dir "$VALIDATED" --reuse-temp-dir | tee -a "$OUT/jobs.txt"
```

The July run likewise with its own line from 2.2 plus `--reuse-temp-dir`. Note every retry (job ids and reason)
in the Stop A report.

### 2.4 Comparison

On the login node, after both full runs ended:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
set -o pipefail
"$PY" scripts/labeling/compare_window_hashes.py --july "$OUT/hashes_july.csv" --port "$OUT/hashes_port.csv" \
    --out "$OUT/hash_comparison.csv" 2>&1 | tee "$LOGS/compare_window_hashes.log"
echo "exit code: $?"
sha256sum config/labeling/recording_map.csv | tee "$OUT/recording_map.sha256"
```

It writes `$OUT/hash_comparison.csv` (one row per labeled cell) and `config/labeling/recording_map.csv`, and
prints the counts per class and per animal, the git commit and src tree of each run, the July fragment-count
check, and every July recording with its status and, when unmapped, its reasons and the file and fragment counts
of the port recordings of that animal on that date (different counts point to files added or dropped). Exit 0:
every recording is mapped. Exit 1: some cells are not identical or remapped, or some recordings are unmapped; go
to Stop A either way. Check that the `july hashes:` line names git commit
413535429de06a92341471495678e8c8dab7ee14 and the `port hashes:` line names TIP, both with `src dirty False` and a
src tree rather than `not written`. Check that no indented warning follows them: several commits or src trees,
src differed from the commit, rows that do not all record a src tree, the July and port hashes record the same
commit or src tree, or `port_src_tree names no single src tree`. Report any such line at Stop A. A traceback or
exit 2: report it.

### 2.5 Intermediates listing

Save what the port run validated; steps 4, 6 and 8 compare against it:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
intermediates > "$OUT/validated_intermediates.txt"; wc -l < "$OUT/validated_intermediates.txt"
check_intermediates; du -sh "$VALIDATED" "$TMP/hash_july"
```

Expect files for the 9 MNE-loaded animals (5 arx_parv, 4 arx_rosa), an intermediate and a `.meta.json` per
session.

### 2.6 Mapping rules to report

A recording maps when every cell is identical (same samples at the same position) or remapped (same samples at
another position of the animal). Two more rules settle samples found at several positions: a recording held whole
at several port positions maps to its July position, and a port position claimed by several July recordings stays
with the one whose July position it is. List every recording where a rule other than identical fired:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
"$PY" - <<'EOF' | tee "$OUT/mapping_rules.txt"
import csv

with open("config/labeling/recording_map.csv") as fh:
    rows = list(csv.DictReader(fh))
with open("results/labeling/hash_comparison.csv") as fh:
    several = {c["july_recording"] for c in csv.DictReader(fh) if ";" in c["hash_positions"]}
print(f"{sum(r['status'] == 'mapped' for r in rows)} of {len(rows)} July recordings mapped")
for r in rows:
    notes = []
    if r["status"] != "mapped":
        notes.append(f"unmapped: {r['reason']}")
    else:
        if r["port_position"] != r["july_position"]:
            notes.append(f"mapped to port position {r['port_position']}, not its July position")
        if r["n_remapped"] not in ("", "0"):
            notes.append(f"{r['n_remapped']} of {r['n_cells']} cells remapped")
        if r["july_recording"] in several:
            notes.append("some cells are held at several port positions")
    if notes:
        print(f"{r['july_recording']}: " + "; ".join(notes))
EOF
```

Report every line at Stop A. Reasons to name explicitly: `port position <q> also holds the cells of ...` (two July
recordings on one port position), `every cell is held at port positions ...` (ambiguous, unmapped), `no single
port position holds every cell`, and, after a scan, `shifted ...`. Do not edit the map, remap by hand or amend
anything; the owner decides at Stop A.

### 2.7 Optional scan of unmapped animals

A scan hashes every window of every port recording of some animals, so the comparison can tell samples that moved
by whole windows (class `shifted`) from samples that changed. Run it when some recordings are unmapped for
`different` or `missing` cells of animals that loaded, and at most 5 animals are involved; with more, ask at Stop A
instead. It reads every sample of each animal and writes about 100 MB per recorded day (resources unmeasured).
Quote each id in single quotes:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
sbatch --parsable --job-name=scan_port --cpus-per-task=4 --mem=64G --time=24:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" scripts/labeling/hash_labeled_windows.py --tree port --all-windows \
    --animal '<id>' '<id>' --out "$OUT/scan_port.csv" --temp-dir "$TMP/scan_port" | tee -a "$OUT/jobs.txt"
```

Then, on the login node, compare into scratch paths; the script refuses to write the committed map with a scan:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
set -o pipefail
"$PY" scripts/labeling/compare_window_hashes.py --july "$OUT/hashes_july.csv" --port "$OUT/hashes_port.csv" \
    --port-scan "$OUT/scan_port.csv" --out "$OUT/hash_comparison_scan.csv" \
    --recording-map "$OUT/recording_map_scan.csv" 2>&1 | tee "$LOGS/compare_window_hashes_scan.log"
echo "exit code: $?"; sha256sum -c "$OUT/recording_map.sha256"
```

`sha256sum -c` must print `OK`: the committed map is as 2.4 wrote it. A shifted recording stays unmapped; its
reason names the port position and window offset.

## Stop A

Send the owner, as plain text with numbers and no verdicts:

- the comparison printout (`$LOGS/compare_window_hashes.log`) and, if run, the scan comparison's;
- the map's statuses and reasons (in that printout), `$OUT/mapping_rules.txt` and the sha256 from
  `$OUT/recording_map.sha256`;
- the preflight outputs: the test summary line (0.9), the dry run table (1.1), the channel names (1.2), the
  `config.local.yaml` finding (1.3), the July-era cache count, date range and size (1.4), TMP's filesystem, free
  space and purge policy (0.5, 0.7, 1.5), the draw and its readable check (1.6) and the preflight hash results
  (2.1);
- the output of `jobacct $(cut -d';' -f1 "$OUT/jobs.txt")` (after sourcing env.sh), which covers every job so
  far, and any retry with its reason.

Ask: may the map be committed as it is (step 3)? If any recording is unmapped, should steps 4 and 6 pass
`--allow-unmapped`, which leaves unmapped recordings out as the pre-registration says? Is an amendment needed (the
owner writes it)? May `$TMP/hash_july`, `$TMP/prehash_july`, `$TMP/prehash_port` and `$TMP/scan_port` be deleted?
Then wait.

## 3. Commit the recording map

Only after the owner's go. If the owner gives an amendment, add it to `config/labeling/preregistration.yaml` as
written, as a dated entry under `amendments`, and commit it on its own with a one-line message before step 4.

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
sha256sum -c "$OUT/recording_map.sha256"; git -C "$PORT" status --porcelain
PC=$("$PY" -c 'import csv
rows = csv.DictReader(open("config/labeling/recording_map.csv"))
print(";".join(sorted({r["port_commit"] for r in rows})))')
[ "$PC" = "$TIP" ] && echo "hashed port commit $PC is TIP" || echo "hashed port commit '$PC' is not TIP"
```

`sha256sum -c` must print `OK`, the status must list only `?? config/labeling/recording_map.csv`, and the hashed
port commit must be TIP. Otherwise ask the owner. Then:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
git -C "$PORT" add config/labeling/recording_map.csv && git -C "$PORT" diff --cached --name-only
git -C "$PORT" commit -q -m "Add the recording map from the hash check of ${TIP:0:12}"
git -C "$PORT" log -1 --format='%H %s'; git -C "$PORT" status --porcelain
```

The `diff --cached` line must print only `config/labeling/recording_map.csv`, the commit message is that one line
with no trailer, and the status prints nothing afterwards. Do not push. Committing the map changes no file under
`src/`, so the map's `port_src_tree` still matches the tree that steps 4 and 6 run.

## 4. Dev features

Before submitting: the tree is clean, the last commit is step 3's, and the validated intermediates are unchanged.
Add `--allow-unmapped` right after `--which dev` only if the owner said so at Stop A (resources unmeasured;
`--workers` equals `--cpus-per-task`):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
git -C "$PORT" status --porcelain; git -C "$PORT" log -1 --format='%H %s'
[ -z "$(git -C "$PORT" status --porcelain)" ] \
    || { echo "PORT is not clean; report the lines above to the owner"; exit 1; }
check_intermediates || exit 1
sbatch --parsable --job-name=dev_features --cpus-per-task=32 --mem=128G --time=48:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" scripts/labeling/extract_dev_features.py --which dev --workers 32 --temp-dir "$VALIDATED" \
    --out "$OUT/dev_features.parquet" "${DEV[@]}" | tee -a "$OUT/jobs.txt"
```

Outputs, all under `$OUT`: `dev_features.parquet` (one row per labeled cell), `dev_features.autoreject_scores.parquet`
(each cell's score and cutoff per configuration), `dev_features.parquet.provenance.json` (inputs, code, src tree,
parameters, the channels of each recording and the config lists by recording), and `dev_features.parquet.parts/`
(one shard per finished animal, kept). When an animal fails, the run writes `dev_features.parquet.errors.json`
instead of the parquet and exits 1. When extraction starts, an earlier run's outputs at these names are renamed
to `<name>.stale`.

While it runs, change nothing in PORT: no edit, commit, checkout or pull of code, `src`, configs, labels, keymap,
split or recording map. The run hashes them at the start, after each animal and before writing, and refuses on
any change.

Success: exit code 0 and, as the script's last line, `wrote N cells over A animals, R recordings, 38 autoreject
configurations to .../dev_features.parquet, with the scores in ...`, with A = 14 unless animals were left out
with `--allow-unmapped`. Then:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
check_intermediates; grep -c -e 'Exporting raw to' -e 'regenerating' "$(joblog <id>)"
```

`check_intermediates` must report them unchanged with 0 lines added, and the count must be 0: the run read the
validated intermediates and wrote none. Otherwise report it to the owner before step 5.

Then save how the config bad_channels lists matched the recordings. The block prints only these keys of the
provenance, which hold config entries, recordings and channel names, no label values:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
"$PY" -c 'import json, sys; p = json.load(open(sys.argv[1]))
keys = ("config_list_by_recording", "config_list_matches", "config_list_unmatched", "config_list_uncovered",
        "config_list_unknown_channels")
print(json.dumps({k: p.get(k) for k in keys}, indent=1))' "$OUT/dev_features.parquet.provenance.json" \
    | tee "$OUT/dev_config_lists.txt"
```

Send `dev_config_lists.txt` with the Stop B report, and name every entry under `config_list_unmatched`,
`config_list_uncovered` and `config_list_unknown_channels`.

On failure:

- A refusal within minutes, before anything loads: unmapped recordings without `--allow-unmapped`; a map whose
  `port_src_tree` differs from the run's src tree, or lacks it (src changed since the hash check; the fix is to
  rerun steps 2 and 3 with the current code, after the owner's go); src with uncommitted changes, or git unable
  to tell (the fix is removing the changes); a map that repeats a july_recording; or another input problem.
  Report the message and wait.
- `N of M animal(s) failed: [...]` with exit 1: report each entry of `dev_features.parquet.errors.json` (animal,
  error type, message) and wait. This includes the detector refusing NaN or inf samples: do not drop or repair
  samples and do not change any rule.
- `... changed during the run; refusing`: something changed PORT. Report what `git -C "$PORT" status` and
  `git -C "$PORT" log -3` show and wait.
- `mapped labelled recordings with no extracted cells`: report and wait.
- Stopped by the cluster (`TIMEOUT` or `OUT_OF_MEMORY` in `jobacct`, whatever the log's last line): resubmit the
  same line plus `--resume`, with more `--mem` after a memory kill, and note it. Finished animals are read back
  from their shards.

A rerun after the owner resolved a failure is the same line plus `--resume`; shards made under other parameters,
inputs or code are recomputed. Read `jobacct <id>` (elapsed, MaxRSS) and report it; adjust step 6's `--mem` and
`--time` from it if needed, never its flags.

## 5. Selection and event cells

Resources unmeasured:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
sbatch --parsable --job-name=selection --cpus-per-task=4 --mem=16G --time=2:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" scripts/labeling/select_detectors.py --features "$OUT/dev_features.parquet" \
    --out "$OUT/dev_selection.json" | tee -a "$OUT/jobs.txt"
```

It writes `$OUT/dev_selection.json` and prints: the dev cells, animals, cv groups and datasets; the F, L and M
settings it read from `config/config.yaml`; the pick, k, with its replace(k) F1 to 4 decimals (and whether a tie
was broken by grid order); a table of the combinations (today, replace, add, lof_branch, autoreject_alone,
filters_alone, config_lists_alone) under the majority rule, which decides, with F1 and its 95 percent interval,
precision, recall, kappa, flag rate and cells; the paired F1 differences replace minus today and add minus today
with their intervals; the same F1s and differences side by side under the majority, any and unanimous rules; the
nested leave-one-group-out and leave-one-dataset-out runs of the choice, with each held-out part's k; the
diagnostic ranking of single detectors, which decides nothing; the raters' pairwise agreement; and finally
`frozen pick: <k>, written to ...`. A refusal (features that are not dev; a dirty or missing src tree; a parquet
that differs from its provenance; features without the recorded channels or config lists; config files that
differ from those the extraction recorded, are missing, or are read without a record; filter settings overridden
by a `config.local.yaml` or a dataset config; a y that is not the majority consensus of the rater categories)
writes nothing: report it and wait.

When it has ended with exit code 0:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
sbatch --parsable --job-name=event_cells --cpus-per-task=2 --mem=8G --time=1:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" scripts/labeling/report_event_cells.py --features "$OUT/dev_features.parquet" \
    --selection "$OUT/dev_selection.json" --out "$OUT/dev_event_cells.csv" | tee -a "$OUT/jobs.txt"
```

It writes `$OUT/dev_event_cells.csv`, each combination's and detector's rejection rate per rater-category pattern,
and prints `wrote N rows for 7 combinations (k = <k>) and 43 detectors to ...`. The pick is now frozen. Editing
`select_detectors.py`, `report_event_cells.py` or `score_test.py` after this means rerunning step 5, since step 7
refuses scoring code the selection did not record.

## Stop B

Send the owner the selection printout and the event-cell line as printed, plain numbers without verdicts, with
`$OUT/dev_config_lists.txt` from step 4 and the jobacct lines of steps 4 and 5. Step 6 and step 8 may start now.
Step 7 waits for the owner's go.

## 6. Test features

The same line as step 4 with `--which test`, the test raters and the test output, and every other flag as in step
4, `--allow-unmapped` included if step 4 had it; `score_test.py` refuses features made with other parameters,
inputs or code. `extract_dev_features.py --which test` reads the test labels only to attach the consensus; it
scores nothing and prints no cell count. Resources as step 4, unmeasured:

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
git -C "$PORT" status --porcelain
[ -z "$(git -C "$PORT" status --porcelain)" ] \
    || { echo "PORT is not clean; report the lines above to the owner"; exit 1; }
check_intermediates || exit 1
sbatch --parsable --job-name=test_features --cpus-per-task=32 --mem=128G --time=48:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" scripts/labeling/extract_dev_features.py --which test --workers 32 --temp-dir "$VALIDATED" \
    --out "$OUT/test_features.parquet" "${TEST[@]}" | tee -a "$OUT/jobs.txt"
```

Outputs as in step 4, under `test_features`. Success: exit code 0 and, as the script's last line,
`wrote A animals, R recordings, 38 autoreject configurations to ...`, with A = 6 unless animals were left out.
The checks afterwards, the rules while it runs, the failures and `--resume` are as in step 4.

## 7. Test score, once

Only after the owner's go. All three files the `ls` names must be absent (resources unmeasured; the time limit is
generous because a run stopped after its claim cannot simply be repeated):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
ls -l "$OUT/test_score.json" "$OUT/dev_selection.json.test_scored" "$OUT/test_features.parquet.test_scored" 2>&1
git -C "$PORT" status --porcelain
[ -z "$(git -C "$PORT" status --porcelain)" ] \
    || { echo "PORT is not clean; report the lines above to the owner"; exit 1; }
sbatch --parsable --job-name=test_score --cpus-per-task=4 --mem=16G --time=4:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" scripts/labeling/score_test.py --features "$OUT/test_features.parquet" \
    --pick "$OUT/dev_selection.json" --out "$OUT/test_score.json" | tee -a "$OUT/jobs.txt"
```

Before it compares any label, it refuses when the dev and test features were produced differently (parameters,
partitions, every input but the label files, the code files, the src tree; the list is in `score_test.py`'s
docstring), when the scoring code or the F, L and M settings differ from what the selection recorded, when the
config files differ from those the test extraction recorded, are missing, or are read without a record, when y
is not the majority consensus of the rater categories, or when the features lack a column or record it needs. A
refusal leaves nothing behind: report the message and wait. When every check has passed it claims the run,
creating an empty `test_score.json` and the markers `dev_selection.json.test_scored` and
`test_features.parquet.test_scored`, then scores today, replace(k), add(k), lof_branch, autoreject_alone,
filters_alone and config_lists_alone once, under the majority, any and unanimous rules, with intervals, the
paired differences, the breakdowns, the event-cell patterns and the raters' agreement on test cells. Success:
exit code 0, a printout starting `test split, N cells over 6 animals, k = <k>:` and ending
`written to .../test_score.json`.

A run that fails after the claim (a traceback, or a job the cluster stopped) leaves the empty `test_score.json`
and both markers. Do not delete them and do not retry: report to the owner, who alone decides whether the three
files are deleted and the run repeated, and the report says so.

## 8. Label-free checks

After step 5. These read no labels. The configurations are the 6 named ones and the frozen k, read from
`pick.autoreject_config` (a repeat of a named one is dropped). Rejection rates are one job for the whole draw,
because the pre-registration records one draw in its provenance; it writes nothing unless every animal finishes
(resources unmeasured; `--workers` equals `--cpus-per-task`):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
PICK=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["pick"]["autoreject_config"])' \
    "$OUT/dev_selection.json") && echo "PICK=$PICK" || exit 1
check_intermediates || exit 1
sbatch --parsable --job-name=rejection_rates --cpus-per-task=16 --mem=64G --time=72:00:00 --output="$LOGS/%x_%j.log" \
    "$JOB" port "$PY" scripts/labeling/rejection_rates.py --dataset "${LABEL_FREE_DATASETS[@]}" --draw 2 --seed 0 \
    --configs "${NAMED[@]}" "$PICK" --workers 16 --temp-dir "$VALIDATED" --out "$OUT/rejection_rates.csv" \
    --evidence-out "$OUT/rejection_evidence.parquet" | tee -a "$OUT/jobs.txt"
```

Its log shows the draw after the selection's INFO lines, ids quoted for the shell where needed; it must equal the
draw in `$OUT/preflight/label_free_draw.txt`. Each recording logs `fit on N fragments, scored ... in S s`.
Success: exit code 0 and the lines `wrote N rate rows over R recordings of 10 animals to ...`, the cutoffs, the
provenance (`rejection_rates.csv.provenance.json`, which records the draw) and the per-fragment scores.
Afterwards `check_intermediates` must report the validated files unchanged; the files it counts as added are the
drawn MNE-loaded animals' intermediates. Any load or scan error stops the run with nothing written: report it and
wait (a NaN refusal as in step 4). A run stopped by the time limit wrote nothing: resubmit with a longer `--time`
if the partition allows, or ask the owner, since running animals separately would lose the single recorded draw.

Then one injection job per drawn animal, in parallel. The loop reads the ids from the provenance, so nothing is
quoted by hand; an id typed by hand is pasted exactly as the draw printed it, quotes included (resources
unmeasured):

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
PICK=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["pick"]["autoreject_config"])' \
    "$OUT/dev_selection.json") && echo "PICK=$PICK" || exit 1
"$PY" -c 'import json, sys
selection = json.load(open(sys.argv[1]))["params"]["selection"]
for dataset, record in selection.items():
    for animal in record["drawn"]:
        print(dataset, animal, sep="\t")' "$OUT/rejection_rates.csv.provenance.json" > "$OUT/label_free_draw.tsv"
cat "$OUT/label_free_draw.tsv"
while IFS=$'\t' read -r ds id; do
    sbatch --parsable --job-name=injection --cpus-per-task=16 --mem=64G --time=24:00:00 --output="$LOGS/%x_%j.log" \
        "$JOB" port "$PY" scripts/labeling/inject_artifacts.py --dataset "$ds" --animal "$id" --seed 0 \
        --configs "${NAMED[@]}" "$PICK" --workers 16 --temp-dir "$VALIDATED" \
        --out "$OUT/injection_${id//[^A-Za-z0-9_.-]/_}.csv" < /dev/null | tee -a "$OUT/jobs.txt"
done < "$OUT/label_free_draw.tsv"
```

The kinds (clip, flatline, step, broadband, line60, drift), amplitudes (2, 5, 10 and 20 times channel RMS) and
fraction (0.05) are the script's defaults, as pre-registered. Each job's success: exit code 0, `wrote N rows to
.../injection_<id>.csv` and `wrote the provenance to ...`. A failure: report it and wait.

## 9. Bring back

```bash
source /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject/scripts/labeling/cluster/env.sh || exit 1
jobacct $(cut -d';' -f1 "$OUT/jobs.txt") | tee "$OUT/sacct.txt"
check_intermediates | tee "$OUT/intermediates_final.txt"
du -sh "$TMP"/* 2>&1 | tee "$OUT/scratch_usage.txt"
git -C "$PORT" log --format='%H %s' "$TIP"..HEAD | tee "$OUT/commits_since_tip.txt"
A=$OUT/autoreject_evaluation_$(date +%Y%m%d).tar.gz
tar -czf "$A" --exclude='*.parquet' --exclude='*.parts' --exclude='*.stale' --exclude='scan_port.csv' \
    --exclude='*.tar.gz' results/labeling config/labeling/recording_map.csv && tar -tzf "$A" && ls -l "$A"
```

The archive holds every JSON and CSV under `$OUT` (selection, test score, event cells, hash tables and comparison,
rejection rates and cutoffs, injection tables), every provenance file, `config/labeling/recording_map.csv`, the
logs, `jobs.txt`, `sacct.txt`, `site.sh`, the preflight outputs and the intermediates listings. The parquets (they
hold labels or are large) and the scan stay on the cluster under `$OUT`. Tell the owner the archive's path, and
paste into the reply, as printed and without verdicts: the selection printout, the event-cell line, the test score
printout, the `wrote` lines of rejection rates and of each injection job, and the sacct table. Then ask whether the
scratch under `TMP` may be deleted.
