# Artifact labeling and detector evaluation

Human ground-truth labels per (channel, window), then detectors scored against them. See each function's
docstring for arguments.

- `render_context.py`: `render_lro()` renders one figure per window (all channels) into `images/`, plus
  `manifest.csv` and `geometry.json`. Works for any format the loader reads (rhd, EDF, NWB, bin).
- `build_rater_bundle.py`: packs a rendered set into a self-contained HTML rating zip.
- `build_cohort_bundle.py`: renders a blinded multi-dataset cohort and writes the unblinding keymap.
- `neurodent.results.scoring`: `ingest`, `unblind`, `consensus`, `interrater`, `pairwise_agreement`,
  `score_mask`. It lives in the library because it scores `FILTER_REGISTRY` detectors and owns the manifest
  schema.

## Labeling flow

1. Render each recording's chosen windows with `render_lro(..., append=True)` to collect many animals in one
   bundle, or build a cohort with `build_cohort_bundle.py`.
2. Bundle: `uv run python scripts/labeling/build_rater_bundle.py <render_dir>` gives `rating_bundle_*.zip`.
3. Rate: send the zip. The rater opens `index.html` (offline, no install), types their name, and labels.
   Every channel defaults to `clean`, so they touch only exceptions, press Next, and finally Export CSV. All
   raters label every window, each in a different order.
4. Split: `partition_labels.py` writes `_dev` and `_test` copies of each rater CSV by `split.json`.

`FILTER_REGISTRY` masks mean keep where True, while `score_mask` takes True as reject; use
`score_keep_mask` for a filter's output. Windows are selected at random only (selecting on a detector's
output would flatter it), and tuning and testing are split by animal.

## Evaluation order

The evaluation is pre-registered in `config/labeling/preregistration.yaml`; lock it before step 2.
`CLUSTER_RUNBOOK.md` has the commands.

1. Sample check: `hash_labeled_windows.py` under the July code and the current code, each with its own empty
   `--temp-dir`, then `compare_window_hashes.py`, which writes `config/labeling/recording_map.csv`. Labels name
   recordings by their July position; the map ties each to a current recording by identity. The comparison
   also checks each July-tree fragment count against `config/labeling/july_n_fragments.csv` (recovered from
   the keymap by `july_fragment_counts.py`) and exits 0 only when every labeled recording is mapped. Later
   scoring runs pass the current-code run's directory as `--temp-dir`.
2. Dev features: `extract_dev_features.py --which dev`, one row per labeled cell with every detector's score.
3. Selection: `select_detectors.py` freezes one detector and reports nested and leave-one-dataset-out
   checks and rater agreement.
4. Event cells: `report_event_cells.py` on dev.
5. Test: `extract_dev_features.py --which test`, then `score_test.py`, which runs once.
6. Label-free checks: `rejection_rates.py` and `inject_artifacts.py` on animals outside the labeled set that
   the pipeline's quality filter keeps.

`score_detectors.py` scores detectors at their config settings on dev, without tuning. Every script that
reads labels refuses animals outside its partition, and nothing but `score_test.py` reads test labels
against a detector.

Tests: `uv run pytest tests/test_scoring.py tests/test_labeling_bundle.py tests/test_cohort_labeling.py
tests/test_detector_selection.py tests/test_evaluation_protocol.py tests/test_hash_check.py
tests/test_label_free_checks.py tests/test_autoreject_detector.py`.

## The rubric (this lives only here)

- `clean`: normal EEG for that channel. Keep. The default.
- `bad`: artifact, not brain: movement, muscle, electrode pop or step, flatline, saturation. Not plain
  60 Hz hum (already notched); only gross residual mains (120 or 180 Hz, visible in the PSD panel) counts.
- `event`: real but dramatic brain activity (epileptiform discharge, seizure). Keep it: an event builds,
  sustains and stops, while an artifact usually does not. Without this category a seizure has no answer but
  "bad", and the ground truth would call seizures artifacts.
- `unsure`: genuinely cannot tell; use it rather than guess.

Scoring maps reject to `bad` and keep to `clean` and `event` (`event` is tagged so over-rejection can be
measured); `unsure` drops out.
