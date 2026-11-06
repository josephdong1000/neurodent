# Cluster runbook: autoreject evaluation

The steps run in this order. Each lists its command, what it writes, and a resource estimate. Commands are
plain `python` calls from the port worktree; heavy steps (2, 4, 6, 8) should run as batch jobs with the
resources noted. Paths below assume:

```bash
PORT=/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/autoreject     # branch feat/autoreject-port
JULY=/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt/july           # detached at 41353542
TMP=/scr1/users/dongjp/autoreject                                       # scratch for intermediates
PY=$PORT/.venv/bin/python
OUT=$PORT/results/labeling                                              # gitignored outputs
DEV="JD=scripts/labeling/labels/labels_cohort4strains_JD_dev.csv JL=scripts/labeling/labels/labels_cohort4strains_JL_dev.csv Yong=scripts/labeling/labels/labels_cohort4strains_Yong_dev.csv"
TEST="JD=scripts/labeling/labels/labels_cohort4strains_JD_test.csv JL=scripts/labeling/labels/labels_cohort4strains_JL_test.csv Yong=scripts/labeling/labels/labels_cohort4strains_Yong_test.csv"
```

Work from `$PORT` unless a step says otherwise, with `unset VIRTUAL_ENV`. Nothing here needs the label
contents outside the scripts; do not open the label files or summarize them by hand.

## 0. Environment

```bash
cd $PORT && unset VIRTUAL_ENV
uv sync --all-extras --frozen
$PY -c "import neurodent; print(neurodent.__file__)"    # must print $PORT/src/neurodent/__init__.py
$PY -m pytest -m "not slow" -q tests/test_autoreject_detector.py tests/test_evaluation_protocol.py
```

## 1. Loader checks (no labels)

Discovery for the 20 labeled animals (the July draw: 5 per dataset, seed 42):

```bash
$PY scripts/labeling/build_cohort_bundle.py --all --limit-per-dataset 5 --seed 42 --dry-run
```

It must list 20 animals and none "with no files, unresolved channels or a loader validation error".

ap3b2 channel identity. The July commit that split Intan files by port id was not ported; if the .rhd
channel names repeat across ports, the current loader refuses the animal. Load one labeled ap3b2 animal:

```bash
$PY - <<'EOF'
import sys; sys.path.insert(0, "scripts/labeling")
import build_cohort_bundle as C
config, samples_config = C._prepare("ap3b2_rhd")
ao = C.load_cohort_animal(samples_config, config, "AP3B2het-207-M")
print([list(lro.channel_names) for lro in ao.long_recordings])
EOF
```

If it raises about repeated or missing channel names, stop: the five ap3b2 animals need a channel-identity
fix in `loading/lro_merge.py` before they can be scored.

## 2. Sample-hash check

The labels name recordings by July position. This step checks where each labeled cell lands now and
writes `config/labeling/recording_map.csv`. Each run needs its own empty temp directory.

```bash
git -C $PORT worktree add --detach $JULY 41353542        # once
cd $JULY && PYTHONPATH=$JULY/src $PY $PORT/scripts/labeling/hash_labeled_windows.py --tree july \
    --out $OUT/hashes_july.csv --temp-dir $TMP/hash_july
cd $PORT && PYTHONPATH=$PORT/src $PY scripts/labeling/hash_labeled_windows.py --tree port \
    --out $OUT/hashes_port.csv --temp-dir $TMP/hash_port
$PY scripts/labeling/compare_window_hashes.py --july $OUT/hashes_july.csv --port $OUT/hashes_port.csv \
    --out $OUT/hash_comparison.csv
```

Writes the two hash tables, the per-cell comparison and `config/labeling/recording_map.csv`. A hash run
writes `<out>.partial` as it goes and renames it at the end, so a leftover `.partial` file means the run did
not finish. A hash run exits 1 when an animal failed to load or some cells could not be read (status
`read_error`); the comparison counts those cells as not hashed. The comparison exits 0 only when every
labeled recording is mapped; otherwise its printed reasons say which are not and why. Resources: both
runs load all recordings of 20 animals, about 4 CPUs, 32 GB and 6 h each, as for the original bundle.

`$TMP/hash_port` now holds the current code's validated intermediates. Every later step passes it as
`--temp-dir`.

## 3. Lock the pre-registration

Review `config/labeling/preregistration.yaml`, adjust while it is a draft, then set `status: locked` and
`locked_on`, and commit it together with `config/labeling/recording_map.csv`. Nothing below runs before this
commit. If some recordings are unmapped, the pre-registration says they are left out; steps 4 and 6 then
need `--allow-unmapped`.

## 4. Dev features

```bash
$PY scripts/labeling/extract_dev_features.py --which dev --temp-dir $TMP/hash_port \
    --out $OUT/dev_features.parquet $DEV
```

Writes the parquet and `dev_features.parquet.provenance.json`. Resources: 43 recordings, about 652,000
fragments through WAR and LOF, plus the 38-configuration grid on up to 5000 fragments per recording. The
July notes suggest 64 CPUs, 64 GB and up to 24 h; this grid is smaller than July's.

## 5. Selection (minutes)

```bash
$PY scripts/labeling/select_detectors.py --features $OUT/dev_features.parquet --out $OUT/dev_selection.json
$PY scripts/labeling/report_event_cells.py --features $OUT/dev_features.parquet \
    --selection $OUT/dev_selection.json --out $OUT/dev_event_cells.csv
```

The first prints the ranking, the frozen pick, the nested and leave-one-dataset-out checks and rater
agreement, and writes `dev_selection.json`. The pick is now frozen.

Optional reference, the detectors at their config settings on dev:

```bash
$PY scripts/labeling/score_detectors.py --all --lof --autoreject --temp-dir $TMP/hash_port \
    --out $OUT/dev_config_scores.csv $DEV
```

## 6. Test features

```bash
$PY scripts/labeling/extract_dev_features.py --which test --temp-dir $TMP/hash_port \
    --out $OUT/test_features.parquet $TEST
```

Same flags as step 4 (the test scorer refuses features built differently). Resources: 21 recordings, about
324,000 fragments.

## 7. Test score, once

```bash
$PY scripts/labeling/score_test.py --features $OUT/test_features.parquet --pick $OUT/dev_selection.json \
    --out $OUT/test_score.json
```

It refuses a second run. An empty `test_score.json` means a run started and failed; delete it deliberately
before retrying, and note that in the report.

## 8. Label-free checks

Rejection rates on whole animals outside the labeled set, for the frozen pick (if it is an autoreject
configuration) and the 6 named configurations:

```bash
NAMED="autoreject/self autoreject/self+CMR welch/v2/self@nf40_na100 welch/v2/self+CMR@nf40_na100 welch/v3/self@nf40_na100 welch/v3/self+CMR@nf40_na100"
PICK=<the frozen pick from dev_selection.json, if it is an autoreject configuration not in NAMED>
$PY scripts/labeling/rejection_rates.py --dataset arx_parv arx_rosa sox5_bin ap3b2_rhd iqsec2 --draw 2 --seed 0 \
    --configs $NAMED $PICK --temp-dir $TMP/hash_port --out $OUT/rejection_rates.csv \
    --evidence-out $OUT/rejection_evidence.parquet
```

It prints the draw (quoted ids) and records it in `rejection_rates.csv.provenance.json`. Then artifact
injection on each drawn animal, one run per animal:

```bash
$PY scripts/labeling/inject_artifacts.py --dataset <dataset> --animal '<id>' --configs $NAMED $PICK \
    --temp-dir $TMP/hash_port --out $OUT/injection_<id>.csv
```

Resources: 10 animals, each a full recording through the fitted configurations; size jobs like step 4,
per animal.

## 9. Bring back

Copy back `$OUT/*.json`, `$OUT/*.csv`, every `*.provenance.json`, `config/labeling/recording_map.csv` and the
job logs. The parquets are optional.
