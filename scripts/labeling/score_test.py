#!/usr/bin/env python
"""Score the combinations of the frozen dev pick on the test split, once.

This is the only script that compares test labels with a detector. It needs:

- the test parquet from ``extract_dev_features.py --which test`` and its provenance file;
- the dev selection JSON from ``select_detectors.py``, whose autoreject configuration k
  (``pick.autoreject_config``) and F, L and M settings are applied unchanged.

It refuses to run when:

- the output exists, or ``<pick>.test_scored`` or ``<features>.test_scored`` does: the test split is scored
  once, and a run claims all three before it compares any label. An empty output means a run started;
- the selection JSON is missing, is not JSON, or lacks ``pick.autoreject_config`` naming a grid
  configuration, the F, L and M settings, or the dev features' provenance;
- the sha256 of a file in ``select_detectors.SCORING_CODE`` differs from the selection's record of it
  (``provenance.code``): select_detectors.py and report_event_cells.py as imported here, and this file;
- the test provenance records ``outputs.features.sha256`` and it differs from the test parquet's;
- the dev and test features were produced differently. Compared: the parameters rule, raters,
  ar_max_fit_fragments, seed, configs, positional_recordings, temp_dir, datasets, lof_chunk_duration_s and
  chunk_duration_s; the partitions (the dev features say dev, the test features test); the sha256 of every
  input either provenance records except the rater label files (``labels_<rater>``), that is the keymap, the
  split, the recording map and the config files: ``config/config.yaml``, ``config/config.local.yaml`` when
  present, the dataset configs and their samples files; the sha256 of every file in
  ``score_detectors.CODE_FILES``; and ``src_tree``, which both must record and share, with ``src_dirty`` false
  in each. The git commit is not compared;
- the F, L or M settings in ``config/config.yaml`` (``--config``) differ from those the selection recorded,
  each dataset's canonical channels included, or the config files differ from those the test extraction
  recorded (``select_detectors.config_problem``): by sha256, ``--config`` against ``config/config.yaml`` and the
  dataset configs, ``config.local.yaml`` and samples files beside it against their ``config/`` entries, a
  recorded file now missing, and a file the settings read without a record;
- the test provenance lacks ``recording_channels`` or ``config_list_by_recording``, or they do not describe the
  cells (``select_detectors.recorded_channels``);
- the parquet holds an animal outside ``split.json``'s test list or a column of unknown role; lacks the
  columns F, L and M read, y, a breakdown column or a rater category column; holds a y other than 0 and 1, or
  a y that is not the majority consensus of its ``cat_<rater>`` columns.

Every check runs, and F, L, M and the combinations' rejections are computed, before the claim. Scored once on
the test cells: today (F or M), replace(k) (F or A(k)), add(k) (F or M or A(k)), lof_branch (F or L), A(k)
alone, F alone and M alone (on the cells whose recording has a config list), each with pooled precision,
recall, F1, Cohen's kappa, tp, fp, fn, tn and flag rate, a 95 percent F1 interval from a bootstrap over test
``cv_group``, and the same metrics per dataset, cv_group, animal, isday and genotype; the paired F1 differences
replace(k) minus today and add(k) minus today with their intervals. All of it is scored against the majority
consensus, the primary rule, and again against the any and unanimous consensus of the cells' rater categories
(``select_detectors.rule_labels``), under each combination's ``alongside``. Also reported: rejection rates per
rater-category pattern for every combination, and the raters' pairwise agreement on the test cells. The output
records the sha256 of the selection it applied and, in ``provenance.code``, of the scoring code it ran.

    uv run python scripts/labeling/score_test.py --features results/labeling/test_features.parquet \\
        --pick results/labeling/dev_selection.json --out results/labeling/test_score.json
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))
import report_event_cells as EV  # noqa: E402
import select_detectors as SD  # noqa: E402
from neurodent.results import autoreject_detector as adr  # noqa: E402

# Provenance parameters that must agree between the dev and the test features. The two chunk durations set how
# the LOF distances and the WAR are computed, so the extraction keys its shards on them too.
MATCHED_PARAMS = ["rule", "raters", "ar_max_fit_fragments", "seed", "configs", "positional_recordings", "temp_dir",
                  "datasets", "lof_chunk_duration_s", "chunk_duration_s"]
# Inputs named with this prefix are the rater label files, which differ between the partitions by design;
# every other input's sha256 must agree.
LABEL_INPUT_PREFIX = "labels_"
# A run claims ``<path><suffix>`` next to the selection and next to the test parquet.
SCORED_SUFFIX = ".test_scored"


def matched_code():
    """Return the code files whose sha256 must agree between the extractions, ``score_detectors.CODE_FILES``."""
    import score_detectors

    return [str(p.relative_to(score_detectors.REPO)) for p in score_detectors.CODE_FILES]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _input_sha256(prov, key):
    entry = (prov.get("inputs") or {}).get(key)
    return entry.get("sha256") if isinstance(entry, dict) else None


def provenance_mismatches(dev_prov, test_prov):
    """Return a list of differences between the dev and test feature provenance that matter for scoring."""
    out = []
    dp, tp = dev_prov.get("params", {}), test_prov.get("params", {})
    for k in MATCHED_PARAMS:
        if dp.get(k) != tp.get(k):
            out.append(f"param {k}: dev {dp.get(k)!r}, test {tp.get(k)!r}")
    if dp.get("which") != "dev" or tp.get("which") != "test":
        out.append(f"partitions: dev features say {dp.get('which')!r}, test features say {tp.get('which')!r}")
    inputs = {k for p in (dev_prov, test_prov) for k in (p.get("inputs") or {}) if not k.startswith(LABEL_INPUT_PREFIX)}
    for key in sorted(inputs):
        if _input_sha256(dev_prov, key) != _input_sha256(test_prov, key):
            out.append(f"input {key}: sha256 dev {_input_sha256(dev_prov, key)!r}, "
                       f"test {_input_sha256(test_prov, key)!r}")
    dc, tc = dev_prov.get("code", {}), test_prov.get("code", {})
    for f in matched_code():
        if dc.get(f) != tc.get(f):
            out.append(f"code {f} differs between the two extractions")
    for name, prov in (("dev", dev_prov), ("test", test_prov)):
        out += SD.src_problems(prov, name)
    d_tree, t_tree = dev_prov.get("src_tree"), test_prov.get("src_tree")
    if d_tree and t_tree and d_tree != t_tree:
        out.append(f"src_tree: dev {d_tree}, test {t_tree}")
    return out


def running_scoring_code():
    """Return ``{repo-relative path: sha256}`` of the scoring code this run executes.

    That is this file and the select_detectors and report_event_cells modules it imported.
    """
    files = (Path(__file__), Path(SD.__file__), Path(EV.__file__))
    return {str(p.resolve().relative_to(REPO)): _sha256(p) for p in files}


def scoring_code_mismatches(selection):
    """Return each file of ``SD.SCORING_CODE`` whose sha256 differs between the selection's record and this run."""
    prov = selection.get("provenance")
    recorded = prov.get("code") if isinstance(prov, dict) else None
    recorded = recorded if isinstance(recorded, dict) else {}
    running = running_scoring_code()
    return [f"{rel}: the selection recorded sha256 {recorded.get(rel)!r}, this run has {running.get(rel)!r}"
            for rel in SD.SCORING_CODE if recorded.get(rel) != running.get(rel)]


def read_pick(path):
    """Return the dev selection JSON, refusing a missing or malformed one.

    Raises:
        SystemExit: If the file is missing or not JSON, or lacks ``pick.autoreject_config`` naming a grid
            configuration, the F, L and M settings, or the dev features' provenance.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no frozen pick at {path}; run select_detectors.py on the dev features first")
    try:
        selection = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError) as e:
        raise SystemExit(f"{path} is not a selection JSON ({e}); refusing") from e
    if not isinstance(selection, dict):
        raise SystemExit(f"{path} is not a selection JSON; refusing")
    pick = selection.get("pick")
    k = pick.get("autoreject_config") if isinstance(pick, dict) else None
    if k not in adr.grid_config_names():
        raise SystemExit(f"{path}: pick.autoreject_config is {k!r}, not a grid configuration; refusing")
    settings = selection.get("settings")
    missing = [key for key in SD.SETTINGS_KEYS if not isinstance(settings, dict) or key not in settings]
    if missing:
        raise SystemExit(f"{path} lacks the settings {missing}; refusing")
    features = selection.get("features")
    if not isinstance(features, dict) or not features.get("provenance"):
        raise SystemExit(f"{path} lacks the dev features' provenance; refusing")
    return selection


def scored_markers(pick, features):
    """Return the files a run claims next to the selection and next to the test parquet."""
    return [Path(str(pick) + SCORED_SUFFIX), Path(str(features) + SCORED_SUFFIX)]


def refuse_scored(out, markers):
    """Refuse when the output or a marker exists: the test split has been scored, or a run started.

    Raises:
        SystemExit: Naming the first file that exists and, for a marker, the output its run claimed.
    """
    if out.exists():
        raise SystemExit(f"{out} exists: the test split has been scored. It is scored once; refusing.")
    for marker in markers:
        if marker.exists():
            try:
                claimed = json.loads(marker.read_text()).get("out")
            except (ValueError, AttributeError):
                claimed = None
            raise SystemExit(f"{marker} exists: a run with this file has scored the test split"
                             f"{f' into {claimed}' if claimed else ''}. It is scored once; refusing.")


def claim(out, markers, record):
    """Create the output, empty, and each marker, holding ``record``; refuse if any of them exists."""
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, text in ((out, ""), *((m, record) for m in markers)):
        try:
            with open(path, "x") as f:
                f.write(text)
        except FileExistsError:
            raise SystemExit(f"{path} exists: the test split has been scored. It is scored once; refusing.") from None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score the combinations of the frozen dev pick on the test split, once.")
    ap.add_argument("--features", type=Path, required=True, help="test parquet from extract_dev_features.py")
    ap.add_argument("--pick", type=Path, required=True, help="dev selection JSON from select_detectors.py")
    ap.add_argument("--split", type=Path, default=REPO / "config/labeling/split.json")
    ap.add_argument("--config", type=Path, default=SD.CONFIG_PATH,
                    help="pipeline config whose F, L and M settings must equal the selection's "
                         "(default: config/config.yaml)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)

    from score_detectors import check_split, provenance

    markers = scored_markers(args.pick, args.features)
    refuse_scored(args.out, markers)
    selection = read_pick(args.pick)
    drift = scoring_code_mismatches(selection)
    if drift:
        raise SystemExit("the selection was made with other scoring code; rerun select_detectors.py with this code "
                         "or restore the files:\n  " + "\n  ".join(drift))
    k = selection["pick"]["autoreject_config"]
    prov_path = Path(str(args.features) + ".provenance.json")
    if not prov_path.exists():
        raise SystemExit(f"missing {prov_path}; test features without provenance cannot be scored")
    test_prov = json.loads(prov_path.read_text())
    features_sha = SD.check_parquet(args.features, test_prov)
    problems = provenance_mismatches(selection["features"]["provenance"], test_prov)
    if problems:
        raise SystemExit("dev and test features were produced differently:\n  " + "\n  ".join(problems))
    SD.require_recording_records(test_prov, "test")

    df = pd.read_parquet(args.features)
    check_split(set(df["animal"]), args.split, "test")
    SD.classify_columns(df)
    SD.require_member_columns(df)
    SD.require_scoring_columns(df)
    EV.category_columns(df)
    labels = SD.rule_labels(df)
    SD.require_majority(df, labels, test_prov)
    datasets = SD.settings_datasets(df, test_prov)
    settings = SD.load_settings(args.config, datasets=datasets)
    changed = SD.settings_differences(selection["settings"], settings)
    if changed:
        raise SystemExit("the filter settings differ from those the selection recorded:\n  " + "\n  ".join(changed))
    problem = SD.config_problem(args.config, test_prov, "test", datasets)
    if problem:
        raise SystemExit(problem)
    members, audits = SD.member_masks(df, settings, test_prov)
    preds = SD.combination_preds(members, SD.mask_pred(df, k))
    cells = SD.combination_cells(df)

    # Claim the output and the markers before any label is compared, so a second run cannot start.
    pick_sha = _sha256(args.pick)
    claim(args.out, markers, json.dumps({"out": str(args.out), "pick": str(args.pick), "pick_sha256": pick_sha,
                                         "features": str(args.features), "features_sha256": features_sha}))

    combos = SD.score_combinations(df, preds, boot=args.boot,
                                   alongside={rule: labels[rule].to_numpy() for rule in SD.ALONGSIDE})
    prov = provenance({"features": args.features, "pick": args.pick, "split": args.split, "config": args.config},
                      script="score_test", boot=args.boot)
    prov["code"] = {**(prov.get("code") or {}), **running_scoring_code()}  # the scoring code checked above
    result = {
        "pick": {"path": str(args.pick), "sha256": pick_sha, "autoreject_config": k,
                 "dev_replace_f1": selection["pick"].get("replace_f1"),
                 "tie_broken": selection["pick"].get("tie_broken")},
        "settings": settings,
        "member_masks": audits,
        "combinations": combos,
        "event_cells": EV.event_table(df, preds, cells).to_dict("records"),
        "rater_agreement": SD.rater_agreement(df),
        "features": {"path": str(args.features), "sha256": features_sha, "provenance": test_prov},
        "provenance": prov,
    }
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=SD.json_default))
    os.replace(tmp, args.out)
    print(f"test split, {len(df)} cells over {df['animal'].nunique()} animals, k = {k}:")
    SD.print_combinations(combos)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
