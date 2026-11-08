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

The evaluation is pre-registered in `config/labeling/preregistration.yaml`, locked 2026-11-06 and amended
2026-11-07. A later change is a dated entry under its `amendments`, approved by the owner before the step it
affects. `CLUSTER_RUNBOOK.md` is the procedure on the cluster, where the recordings are, with its settings, job
wrapper and worktree bootstrap in `cluster/`. The design write-up is
https://claude.ai/artifact/Q5NSyG1BjMbJvPJQTYdPc7.

Detectors run together in the pipeline, so the decision is which combination to run. The members are per-cell
reject masks: F, the fragment filters at their settings in `config/config.yaml`, applied as the pipeline applies
them; L, LOF at `channel_filter_config.lof.reject_lof_threshold`; M, the dataset configs' `bad_channels` lists;
and A(k), configuration k of the 38-configuration autoreject grid. A combination rejects a cell when any member
does. The combinations, under the names every output JSON uses:

- `today`: F or M, the manual branch that zeitgeber, relfreq and the figures use;
- `replace`: F or A(k);
- `add`: F or M or A(k);
- `lof_branch`: F or L, a reference;
- `autoreject_alone`: A(k); `filters_alone`: F; `config_lists_alone`: M, on the cells whose recording has a list.

Each is scored against the majority consensus, which decides, and against the any and unanimous rules alongside
it, on dev and on test, pooled and per dataset, cv_group, animal, isday and genotype.

1. Sample check: `hash_labeled_windows.py` under the July tree (41353542) and the port, each with its own empty
   `--temp-dir`, then `compare_window_hashes.py`, which writes `config/labeling/recording_map.csv`. Labels name
   recordings by their July position; the map ties each to a port recording by the hashes of its samples and
   records the port's src tree (`port_src_tree`), written only when the port run's src had no uncommitted
   changes. The comparison also checks each July-tree fragment count against
   `config/labeling/july_n_fragments.csv` (recovered from the keymap by `july_fragment_counts.py`) and exits 0
   only when every labeled recording is mapped. An optional scan (`--all-windows`, then `--port-scan`) tells
   samples that moved by whole windows from samples that changed. Every later run passes the port run's temp
   directory as `--temp-dir`.
2. Dev features: `extract_dev_features.py --which dev`, one row per labeled cell with the filter, LOF, config list
   and autoreject values. It refuses a recording map whose `port_src_tree` is missing, empty or not its own src
   tree, and refuses to run while src has uncommitted changes.
3. Selection: `select_detectors.py` freezes k, the configuration with the highest dev F1 of replace(k)
   (`pick.autoreject_config`), and reports every combination with its interval, the paired differences replace
   minus today and add minus today, nested leave-one-group-out and leave-one-dataset-out runs of the choice, the
   ranking of the 43 single detectors as a diagnostic, and rater agreement.
4. Event cells: `report_event_cells.py` on dev, for the combinations and the single detectors.
5. Test: `extract_dev_features.py --which test` with the dev run's flags, then `score_test.py`, which runs once.
6. Label-free checks: `rejection_rates.py` and `inject_artifacts.py` for the 6 named configurations and k, on two
   unlabeled animals per labeled dataset and two iqsec2 animals that the pipeline's quality filter keeps, drawn
   with seed 0.

`score_detectors.py` scores detectors at their config settings on dev, without tuning; it is not part of the
pre-registered evaluation. Every script that reads labels refuses animals outside its partition, and nothing but
`score_test.py` scores test labels against a detector; `extract_dev_features.py --which test` reads them only to
attach the consensus.

Tests, 11 files; `-m "not slow"` would skip their runs on the mini_real recordings, and the 3 browser tests in
`test_labeling_bundle.py` skip without playwright chromium:

    uv run pytest tests/test_scoring.py tests/test_labeling_bundle.py tests/test_cohort_labeling.py \
        tests/test_detector_selection.py tests/test_evaluation_protocol.py tests/test_hash_check.py \
        tests/test_label_free_checks.py tests/test_autoreject_detector.py tests/test_combinations.py \
        tests/test_config_settings.py tests/test_extract_features.py

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
