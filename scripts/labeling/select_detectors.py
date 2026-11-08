#!/usr/bin/env python
"""Choose the autoreject configuration for the pipeline's detector combination on the dev features.

Reads the dev parquet written by ``extract_dev_features.py`` (no signal reload), its provenance file and the
filter settings in ``config/config.yaml``. Detectors run together in the pipeline, so the decision is which
combination to run. Its members are per-cell reject masks:

- F: the fragment filters in ``fragment_filter_config`` at their settings, applied as the pipeline applies
  them. high_rms rejects ``rms > max_rms``, low_rms ``rms < min_rms``, logrms_range ``logrmsz_abs > z_range``
  and high_beta ``beta_wmax > max_beta_prop``, where ``beta_wmax`` is the window's largest beta proportion, so
  high_beta rejects the window on every channel. A window left with fewer than 3 valid channels is then
  rejected on every channel, as ``workflow/scripts/filter_wars_fragments.py`` calls ``apply_filters`` with
  ``min_valid_channels=3``. A missing score rejects nothing, as a NaN comparison is false in the filters.
- L: LOF at ``channel_filter_config.lof.reject_lof_threshold``. A channel is rejected for its whole recording
  when ``lof >= threshold`` (the pipeline keeps a channel only when its score is below the threshold), and a
  recording left with fewer than ``min_valid_channels`` valid channels is rejected whole. A cell without a LOF
  score is kept.
- M: the dataset configs' bad_channels lists, with the same recording rule at
  ``channel_filter_config.manual.min_valid_channels``. It rejects nothing else where no list applies.
- A(k): the 0/1 column of autoreject configuration k; a cell outside the fit pool (NaN) is kept.

The pipeline filters the standardized WAR, which ``standardize_wars.py`` pads with NaN to its dataset's
standardized channels (``canonical_channels``, read from the config as that script reads them), and its minimum
valid channel rules count every one of them. The features' provenance records, per recording, its own channels
among those (``recording_channels``) and its config list (``config_list_by_recording``: whether a list applies,
the listed channels the recording has and the listed names it lacks). The run refuses features without both
records, and a recording whose channels are not standardized channels. Every standardized channel counts in the
rules, whether or not the features hold a cell on it. M counts a listed standardized channel as listed, as the
manual branch rejects it, padded or not. F counts a channel without a cell in a window as kept and L counts a
channel without a cell in a recording as unflagged: exact for a padded channel, whose values are NaN, and
assumed for a channel of the recording every rater abstained on, whose scores the features lack; the audits
count the decisions that rest on the assumption.

A combination rejects a cell when any member does: today = F or M, the manual branch; replace(k) = F or A(k);
add(k) = F or M or A(k); lof_branch = F or L, the LOF branch, as a reference; A(k) alone; F alone; and M
alone, scored only on cells whose recording has a list.

Ground truth: the majority consensus y is the primary rule and the only one the selection uses. The any and
unanimous consensus of each cell are computed from its ``cat_<rater>`` columns with the vote mapping and the
rules of ``neurodent.results.scoring`` (bad 1; clean and event 0; unsure and blank abstain), and every
combination is scored against them alongside. The run refuses a y that is not the majority consensus of those
columns.

Selection: k is the configuration with the highest replace(k) F1 over all dev cells, to 4 decimals; ties go
to the earlier configuration in ``adr.grid_config_names()`` order. Nothing else is tuned. Nested
leave-one-group-out (``cv_group``) and leave-one-dataset-out runs repeat the choice inside each outer training
part, predict the held-out part with replace(k) and pool the held-out predictions. A run the cells cannot
support (one dataset, or one cv_group) is recorded as not run, with its reason under ``not_run``.

Every combination gets, under each rule, pooled precision, recall, F1, Cohen's kappa, tp, fp, fn, tn and flag
rate; a 95 percent F1 interval from a bootstrap over ``cv_group`` (BCa, percentile when BCa is undefined); and
the same metrics per dataset, cv_group, animal, isday and genotype. replace(k) and add(k) also get their paired
F1 difference from today with its interval, under each rule.

The ranking of the 43 single detectors (5 filters with thresholds tuned on dev labels, then the grid's masks)
and its nested runs are kept under ``diagnostic``; they decide nothing. The ranking needs two cv_groups, and
its nested runs two cv_groups in every training part. Rater agreement on the dev cells (pairwise Cohen's
kappa, overall and per dataset) is the ceiling.

Before anything is computed, the run refuses features that ``score_test.py`` would refuse (a provenance that
does not say ``which`` dev, lacks ``src_tree`` or whose ``src_dirty`` is not false, or whose
``outputs.features.sha256`` differs from the parquet's), features whose provenance lacks the recording records,
and config files other than those the dev extraction recorded (``config_problem``): by sha256, ``--config``
against ``config/config.yaml``, and the dataset configs, ``config.local.yaml`` and samples files beside it against
their ``config/`` entries, a recorded file now missing, and a file the settings read without a record. The
canonical channels the rules count come from those files.

The output JSON holds ``pick`` (``autoreject_config``, ``replace_f1``, ``tie_broken``), ``settings`` (F, L and M
settings, each dataset's canonical channels and the config's sha256), ``member_masks`` (label-free counts of
what F, L and M reject, their audits, the standardized channels each recording lacks and its own channels it
has no cell on),
``combinations`` (majority metrics at the top of each entry, the any and unanimous ones under ``alongside``),
``nested_replace_logo``, ``nested_replace_lodo``, ``diagnostic``, ``rater_agreement``, ``features`` (the features'
path, sha256 and provenance) and ``provenance``, whose ``code`` also records the sha256 of every file in
``SCORING_CODE``, which ``score_test.py`` requires to be unchanged.

    uv run python scripts/labeling/select_detectors.py --features results/labeling/dev_features.parquet \\
        --out results/labeling/dev_selection.json
"""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import bootstrap
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import LeaveOneGroupOut

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))
from neurodent.results import autoreject_detector as adr  # noqa: E402
from neurodent.results.filters import FILTER_REGISTRY  # noqa: E402
from neurodent.results.scoring import LABEL_MAP, RATER_CATEGORY_PREFIX, consensus, pairwise_agreement  # noqa: E402
from neurodent.workflow.utils.config import load_dataset_config, resolve_samples_config  # noqa: E402

# Filter or LOF detector: (feature column, reject direction). "hi" rejects scores above tau, "lo" below.
# Autoreject configurations come from the parquet's grid columns.
DETECTORS = {
    "high_rms": ("rms", "hi"),
    "low_rms": ("rms", "lo"),
    "logrms_range": ("logrmsz_abs", "hi"),
    "high_beta": ("beta_wmax", "hi"),
    "lof": ("lof", "hi"),
}
META_COLS = {"recording", "window", "channel", "t_start_s", "y", "animal", "animalday", "dataset", "cv_group",
             "genotype", "timestamp", "isday", "n_raters", "any_event", "config_list", "config_list_defined"}
FILTER_COLS = {"rms", "logrmsz_abs", "beta_wmax", "lof"}
GROUP_COL = "cv_group"
# Rater category to binary vote, from scoring.LABEL_MAP: artifact 1, clean and event 0, unsure and unlabeled NaN
# (an abstention).
CATEGORY_VOTE = {category: vote for vote, category in LABEL_MAP.values()}
# The consensus rules: majority is primary and the only one the selection uses; the others are reported alongside.
PRIMARY_RULE = "majority"
ALONGSIDE = ("any", "unanimous")

CONFIG_PATH = REPO / "config/config.yaml"
# Provenance inputs under this prefix are config files, compared with the files beside --config (config_problem).
CONFIG_INPUT_PREFIX = "config/"
# The threshold parameter of each fragment filter the per-cell scores can reproduce.
FRAGMENT_FILTER_PARAMS = {"high_rms": "max_rms", "low_rms": "min_rms", "logrms_range": "z_range",
                          "high_beta": "max_beta_prop"}
# workflow/scripts/filter_wars_fragments.py calls apply_filters(..., min_valid_channels=3); the config has no key.
FRAGMENT_MIN_VALID_CHANNELS = 3
# The settings score_test.py requires to be unchanged between selection and test.
SETTINGS_KEYS = ("fragment_filters", "fragment_min_valid_channels", "lof", "config_lists", "canonical_channels")
# Columns the members read.
MEMBER_COLS = ["recording", "window", "channel", "dataset", "rms", "logrmsz_abs", "beta_wmax", "lof", "config_list",
               "config_list_defined"]
# The code that turns a pick into scores. The selection records the sha256 of each file, and score_test.py
# refuses to run when one differs.
SCORING_CODE = ["scripts/labeling/select_detectors.py", "scripts/labeling/score_test.py",
                "scripts/labeling/report_event_cells.py"]

# Combination: members. "A" is the chosen autoreject configuration. today is the pipeline's manual branch and
# lof_branch its LOF branch, scored as a reference.
COMBINATIONS = {
    "today": ("F", "M"),
    "replace": ("F", "A"),
    "add": ("F", "M", "A"),
    "lof_branch": ("F", "L"),
    "autoreject_alone": ("A",),
    "filters_alone": ("F",),
    "config_lists_alone": ("M",),
}
# The features' provenance records the extraction writes for each recording, which F, L and M read.
RECORDING_RECORDS = ("recording_channels", "config_list_by_recording")
# Combination: the combination its paired F1 difference is taken from.
PAIRED = {"replace": "today", "add": "today"}
# Output key: column the scores are broken down by.
BREAKDOWNS = {"per_dataset": "dataset", "per_cv_group": GROUP_COL, "per_animal": "animal", "per_isday": "isday",
              "per_genotype": "genotype"}


def score_of(df, col, direction):
    """Return per-cell scores in precision-recall convention (higher means more likely an artifact)."""
    s = df[col].to_numpy(float)
    return s if direction == "hi" else -s


def best_tau(y, s):
    """Return ``(F1, tau)`` for the F1-optimal threshold on ``s`` (reject where s >= tau), over all rows."""
    m = np.isfinite(s)
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return 0.0, np.inf
    p, r, thr = precision_recall_curve(y[m], s[m])
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)[:-1]  # aligned to thr
    if not len(f1):
        return 0.0, np.inf
    k = int(np.argmax(f1))
    return float(f1[k]), float(thr[k])


def predict(s, tau):
    """Return 0/1 rejections, s >= tau; non-finite scores are kept."""
    out = np.zeros(len(s), int)
    m = np.isfinite(s)
    out[m] = (s[m] >= tau).astype(int)
    return out


def _groups(groups):
    uniq = np.array(sorted(pd.unique(groups)))
    return uniq, {a: np.where(groups == a)[0] for a in uniq}


def _cluster_ci(stat, n_groups, n=2000, seed=0):
    """Return a 95 percent interval from ``scipy.stats.bootstrap`` with the group as the resampling unit.

    Whole groups are resampled with replacement and their cells kept together, so the interval respects
    correlation within a group. BCa is tried first, then the percentile method when BCa is undefined (a
    statistic with no jackknife spread). With fewer than 2 groups there is nothing to resample and the
    interval is NaN.
    """
    if n_groups < 2:
        return np.nan, np.nan
    gidx = np.arange(n_groups)
    lo = hi = np.nan
    for method in ("BCa", "percentile"):
        ci = bootstrap((gidx,), stat, n_resamples=n, vectorized=False, method=method,
                       random_state=np.random.default_rng(seed)).confidence_interval
        lo, hi = float(ci.low), float(ci.high)
        if np.isfinite(lo) and np.isfinite(hi):
            break
    return lo, hi


def cluster_boot_ci(y, yhat, groups, n=2000, seed=0):
    """Return the 95 percent interval of pooled F1 under a bootstrap over groups."""
    uniq, by = _groups(groups)

    def stat(sampled):
        cells = np.concatenate([by[uniq[int(i)]] for i in np.asarray(sampled).ravel()])
        return f1_score(y[cells], yhat[cells], zero_division=0)

    return _cluster_ci(stat, len(uniq), n, seed)


def paired_delta_ci(y, yhat_a, yhat_b, groups, n=2000, seed=0):
    """Return the 95 percent interval of F1(A) minus F1(B) under a bootstrap over groups."""
    uniq, by = _groups(groups)

    def stat(sampled):
        c = np.concatenate([by[uniq[int(i)]] for i in np.asarray(sampled).ravel()])
        return f1_score(y[c], yhat_a[c], zero_division=0) - f1_score(y[c], yhat_b[c], zero_division=0)

    return _cluster_ci(stat, len(uniq), n, seed)


class _AllInOneFold:
    """A splitter with one fold whose train and test rows are all rows.

    It lets :func:`evaluate` compute the in-sample ``all_dev_f1`` through the same loop that
    ``LeaveOneGroupOut`` uses for ``logo_f1``.
    """

    def split(self, X, y=None, groups=None):
        idx = np.arange(len(X))
        yield idx, idx


def evaluate(s, y, groups, splitter):
    """Fit tau per fold on the training rows, predict the held-out rows, and pool the predictions.

    Args:
        s: Scores, higher meaning more likely an artifact.
        y: 0/1 labels.
        groups: Group of each row, passed to the splitter.
        splitter: ``_AllInOneFold()`` for the in-sample F1, or ``LeaveOneGroupOut()``.

    Returns:
        ``(F1, out-of-fold predictions, {group: F1}, taus per fold)``; with ``_AllInOneFold`` the only tau is
        the deployable one.
    """
    oof = np.zeros(len(s), int)
    per, taus = {}, []
    for tr, te in splitter.split(s, y, groups=groups):
        _, tau = best_tau(y[tr], s[tr])
        oof[te] = predict(s[te], tau)
        taus.append(tau)
        for a in np.unique(groups[te]):
            m = groups[te] == a
            per[str(a)] = round(float(f1_score(y[te][m], oof[te][m], zero_division=0)), 3)
    return float(f1_score(y, oof, zero_division=0)), oof, per, taus


def per_animal_f1(y, yhat, groups):
    """Return ``{group: F1}`` of a fixed prediction."""
    return {str(a): round(float(f1_score(y[groups == a], yhat[groups == a], zero_division=0)), 3)
            for a in sorted(pd.unique(groups))}


def counts(y, yhat):
    """Return tp, fp, fn, tn, n, flag rate and F1 (NaN when there are neither positives nor flags)."""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else np.nan
    return {"n": int(len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "flag_rate": float(yhat.mean()) if len(y) else np.nan, "f1": f1}


def metrics(y, yhat):
    """Return :func:`counts` plus precision, recall and Cohen's kappa of a 0/1 prediction.

    Precision is NaN without flags and recall NaN without positives. Kappa is Cohen's, as sklearn's
    ``cohen_kappa_score``, computed from the counts in integer arithmetic; it is NaN when chance agreement is
    1, that is when the labels and the prediction hold one and the same class.
    """
    c = counts(y, yhat)
    tp, fp, fn, tn, n = c["tp"], c["fp"], c["fn"], c["tn"], c["n"]
    chance = (tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)  # n squared times the chance agreement
    kappa = (n * (tp + tn) - chance) / (n * n - chance) if n and chance != n * n else np.nan
    return {**c, "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan, "cohen_kappa": kappa}


def breakdown(df, yhat, col, y=None):
    """Return ``{value of col: metrics}`` of a prediction, against ``y`` (default: the column y)."""
    y = df["y"].to_numpy() if y is None else np.asarray(y)
    yhat = np.asarray(yhat)
    vals = df[col].astype(str).to_numpy()
    return {v: metrics(y[vals == v], yhat[vals == v]) for v in sorted(pd.unique(vals))}


def mask_pred(df, name):
    """Return an autoreject configuration's 0/1 rejections; a cell outside the fit pool (NaN) is kept."""
    return np.nan_to_num(df[name].to_numpy(dtype=float), nan=0.0).astype(int)


def classify_columns(df):
    """Return the autoreject columns of the parquet in grid order, refusing any column of unknown role.

    ``config_list`` and ``config_list_defined`` are metadata: they define M and are never ranked.

    Raises:
        SystemExit: If a column is neither metadata, a filter score, a rater category nor a grid name.
    """
    grid = adr.grid_config_names()
    ar_cols = [c for c in grid if c in df.columns]
    known = META_COLS | FILTER_COLS | set(grid)
    unknown = [c for c in df.columns if c not in known and not c.startswith(RATER_CATEGORY_PREFIX)]
    if unknown:
        raise SystemExit(f"columns of unknown role: {unknown}. Add them to META_COLS or the grid; refusing to "
                         "rank them as detectors.")
    missing = [c for c in grid if c not in df.columns]
    if missing:
        raise SystemExit(f"grid configurations missing from the features: {missing}")
    return ar_cols


def require_member_columns(df):
    """Refuse features that cannot give F, L and M.

    Raises:
        SystemExit: If a column the members read is missing, ``config_list`` holds anything but 0 and 1 or
            ``config_list_defined`` anything but true and false, a cell is listed where no list is defined, or a
            (recording, window, channel) cell appears more than once.
    """
    missing = [c for c in MEMBER_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"the features lack {missing}; extract them with the current extract_dev_features.py")
    listed, defined = df["config_list"], df["config_list_defined"]
    if listed.isna().any() or not listed.isin([0, 1]).all():
        raise SystemExit("config_list holds values other than 0 and 1")
    if defined.isna().any() or not defined.isin([True, False]).all():
        raise SystemExit("config_list_defined holds values other than true and false")
    if ((listed == 1) & ~defined.astype(bool)).any():
        raise SystemExit("config_list marks cells whose config_list_defined is false")
    repeated = df[df.duplicated(["recording", "window", "channel"], keep=False)]
    if len(repeated):
        cells = repeated[["recording", "window", "channel"]].drop_duplicates().head(5).to_records(index=False)
        raise SystemExit(f"cells that appear more than once: {[tuple(c) for c in cells]}")


def require_recording_records(prov, name=None):
    """Refuse a features provenance without the recording records F, L and M read.

    Raises:
        SystemExit: If ``recording_channels`` or ``config_list_by_recording`` is missing or not a mapping.
    """
    missing = [key for key in RECORDING_RECORDS if not isinstance(prov.get(key), dict)]
    if missing:
        raise SystemExit(f"the {name + ' ' if name else ''}features' provenance lacks {missing}, which F, L and M "
                         "read; extract them with the current extract_dev_features.py")


def _channel_list(value):
    """Return ``value`` as a list of distinct channel names, or None when it is not one."""
    if not isinstance(value, list) or not all(isinstance(c, str) for c in value) or len(set(value)) != len(value):
        return None
    return value


def recorded_channels(df, prov, settings):
    """Return each recording's channels and config list, from the features' provenance and the config.

    The pipeline standardizes every WAR of a dataset to the same channels (``settings["canonical_channels"]``),
    padding the ones a recording lacks with NaN, and its minimum valid channel rules count all of them.
    ``recording_channels`` names the recording's own channels among those, the ones it is not padded on, and
    ``config_list_by_recording`` gives whether a list applies (``defined``), the listed channels the recording
    has (``channels``) and the listed names it lacks (``absent``). A listed name counts for M when it is a
    standardized channel, as the manual branch rejects it there, padded or not, and ignores any other.

    Args:
        df: Cells with recording, channel, dataset, config_list and config_list_defined.
        prov: The features' provenance.
        settings: Output of :func:`load_settings`, with the canonical channels of every dataset of ``df``.

    Returns:
        ``{recording: {"standardized": [channel], "channels": [channel], "listed": [channel], "defined": bool}}``
        for the recordings of ``df``: the standardized channels, the recording's own, and the standardized
        channels its list names, each in the standardized order.

    Raises:
        SystemExit: If the provenance lacks either record, or an entry for a recording of ``df``; if a
            recording's channels are not distinct standardized channels, or it has cells on another channel or
            in several datasets; or if a config list entry is malformed, names channels while no list applies,
            puts a channel on the wrong side of the recording's channels, or disagrees with the parquet's
            ``config_list`` and ``config_list_defined``.
    """
    require_recording_records(prov)
    by_rec, lists = prov["recording_channels"], prov["config_list_by_recording"]
    canonical = settings.get("canonical_channels") or {}
    cells = pd.DataFrame({"recording": df["recording"].astype(str).to_numpy(),
                          "dataset": df["dataset"].astype(str).to_numpy(),
                          "channel": df["channel"].astype(str).to_numpy()})
    unrecorded = sorted(r for r in cells["recording"].unique() if r not in by_rec or r not in lists)
    if unrecorded:
        raise SystemExit(f"the features' provenance has no recording_channels or config_list_by_recording entry "
                         f"for {unrecorded}; extract them with the current extract_dev_features.py")
    unknown = sorted(set(cells["dataset"]) - set(canonical))
    if unknown:
        raise SystemExit(f"the settings hold no canonical channels for the datasets {unknown}")
    problems, out = [], {}
    for rec, sub in cells.groupby("recording", sort=True):
        datasets = sorted(set(sub["dataset"]))
        channels = _channel_list(by_rec[rec])
        if len(datasets) > 1:
            problems.append(f"{rec}: cells in several datasets, {datasets}")
            continue
        if channels is None:
            problems.append(f"{rec}: recording_channels is not a list of distinct channel names: {by_rec[rec]!r}")
            continue
        standardized = list(canonical[datasets[0]])
        dropped = [c for c in channels if c not in standardized]
        if dropped:
            problems.append(f"{rec}: recording_channels names {dropped}, which are not among the standardized "
                            f"channels of {datasets[0]}, {standardized}")
            continue
        outside = sorted(set(sub["channel"]) - set(channels))
        if outside:
            problems.append(f"{rec}: cells on channels outside its recording_channels: {outside}")
        entry = lists[rec]
        has = _channel_list(entry.get("channels")) if isinstance(entry, dict) else None
        lacks = _channel_list(entry.get("absent")) if isinstance(entry, dict) else None
        defined = entry.get("defined") if isinstance(entry, dict) else None
        if not isinstance(defined, bool) or has is None or lacks is None:
            problems.append(f"{rec}: config_list_by_recording is not {{'defined': bool, 'channels': [names], "
                            f"'absent': [names]}}: {entry!r}")
            continue
        if not defined and (has or lacks):
            problems.append(f"{rec}: config_list_by_recording names {has + lacks} while no list applies")
        wrong = sorted(set(has) - set(channels)) + sorted(set(lacks) & set(channels))
        if wrong:
            problems.append(f"{rec}: config_list_by_recording puts {wrong} on the wrong side of its "
                            f"recording_channels {channels}")
        named = set(has) | set(lacks)
        out[rec] = {"standardized": standardized, "channels": [c for c in standardized if c in channels],
                    "listed": [c for c in standardized if c in named], "defined": defined, "has": set(has)}
    if problems:
        raise SystemExit("the features' recording records do not describe their cells:\n  " + "\n  ".join(problems))
    rec = cells["recording"]
    defined = rec.map({r: v["defined"] for r, v in out.items()}).to_numpy(bool)
    listed = np.array([c in out[r]["has"] for r, c in zip(rec, cells["channel"])], bool)
    differ = (defined != df["config_list_defined"].astype(bool).to_numpy()) | (listed != (df["config_list"].to_numpy() == 1))
    if differ.any():
        examples = cells.loc[differ, ["recording", "channel"]].drop_duplicates().head(5).to_records(index=False)
        raise SystemExit(f"config_list and config_list_defined disagree with config_list_by_recording on "
                         f"{int(differ.sum())} cells, for example {[tuple(e) for e in examples]}")
    return {r: {k: v for k, v in entry.items() if k != "has"} for r, entry in out.items()}


def rule_labels(df):
    """Return each cell's consensus under each rule, from its ``cat_<rater>`` columns.

    A rater's category becomes a vote as ``scoring.LABEL_MAP`` maps it (artifact 1; clean and event 0; unsure and
    unlabeled abstain), a missing category abstains, and ``scoring.consensus`` applies each rule to the votes.

    Args:
        df: Cells with recording, window, channel and the ``cat_<rater>`` columns.

    Returns:
        DataFrame aligned to ``df`` with one 0/1 column per rule: majority (``PRIMARY_RULE``), then ``ALONGSIDE``.

    Raises:
        SystemExit: If ``df`` has no ``cat_<rater>`` column or a category outside ``CATEGORY_VOTE``, holds a cell
            twice, or holds a cell no rater voted on.
    """
    cat_cols = sorted(c for c in df.columns if c.startswith(RATER_CATEGORY_PREFIX))
    if not cat_cols:
        raise SystemExit("the features hold no rater category columns (cat_<rater>)")
    keys = ["recording", "window", "channel"]
    cells = pd.DataFrame({"recording": df["recording"].astype(str).to_numpy(),
                          "window": df["window"].to_numpy(), "channel": df["channel"].astype(str).to_numpy()})
    long = pd.concat([cells.assign(rater=c[len(RATER_CATEGORY_PREFIX):], category=df[c].to_numpy())
                      for c in cat_cols], ignore_index=True)
    long["category"] = long["category"].where(long["category"].notna(), "unlabeled").astype(str)
    unknown = sorted(set(long["category"]) - set(CATEGORY_VOTE))
    if unknown:
        raise SystemExit(f"rater categories {unknown} are not among {sorted(CATEGORY_VOTE)}")
    long["y"] = long["category"].map(CATEGORY_VOTE)
    long["t_start_s"] = np.nan  # consensus carries a start time through; none is needed here
    out = {}
    for rule in (PRIMARY_RULE, *ALONGSIDE):
        try:
            voted = consensus(long, rule=rule)
        except ValueError as e:
            raise SystemExit(f"the {rule} consensus of the rater categories: {e}") from e
        y = cells.merge(voted[keys + ["y_true"]], on=keys, how="left")["y_true"]
        if y.isna().any():
            raise SystemExit(f"{int(y.isna().sum())} cells hold no rater vote, so no consensus")
        out[rule] = y.astype(int).to_numpy()
    return pd.DataFrame(out, index=df.index)


def require_majority(df, labels, prov):
    """Refuse a y other than the majority consensus of the cells' rater categories (``rule_labels``).

    Raises:
        SystemExit: Naming how many cells differ and the rule the extraction recorded.
    """
    differ = df["y"].to_numpy(int) != labels[PRIMARY_RULE].to_numpy()
    if differ.any():
        rule = (prov.get("params") or {}).get("rule")
        raise SystemExit(f"y differs on {int(differ.sum())} cells from the {PRIMARY_RULE} consensus of the "
                         f"cat_<rater> columns (the extraction recorded rule {rule!r}); the pre-registered primary "
                         f"rule is {PRIMARY_RULE}. Refusing.")


def recorded_sha256(prov, key):
    """Return the sha256 a provenance records for input ``key``, or None."""
    entry = (prov.get("inputs") or {}).get(key)
    return entry.get("sha256") if isinstance(entry, dict) else None


def config_input_path(config_path, key):
    """Return the file beside ``config_path`` that a provenance input key under ``config/`` names.

    ``config/config.yaml`` is ``config_path`` itself. Any other key is taken relative to its folder, as the
    settings read the files there: ``config/datasets/ds1.yaml`` is ``<folder>/datasets/ds1.yaml``.
    """
    path = Path(config_path)
    rel = key[len(CONFIG_INPUT_PREFIX):]
    return path if rel == "config.yaml" else path.parent / rel


def config_problem(config_path, prov, name, datasets=()):
    """Return why the config files the settings come from are not those the extraction recorded, or None.

    Every input the provenance records under ``config/`` is compared by sha256 with the file
    :func:`config_input_path` maps it to: ``config_path`` for ``config/config.yaml``, and ``config.local.yaml``,
    the dataset configs and samples files beside it for the others. A recorded file that is now missing is a
    problem, and so is a file the settings read that the extraction did not record: ``config_path``,
    ``config.local.yaml`` beside it when it exists, and the dataset config of each of ``datasets``.

    Args:
        config_path: The pipeline config the settings are read from (``--config``).
        prov: The features' provenance.
        name: The partition the features hold, for the message.
        datasets: Dataset names whose configs the settings read (:func:`settings_datasets`).

    Returns:
        A message naming every file that differs, or None when all agree.
    """
    path = Path(config_path)
    if not path.exists():
        return f"pipeline config not found: {path}"
    recorded = {key: recorded_sha256(prov, key) for key in (prov.get("inputs") or {})
                if key.startswith(CONFIG_INPUT_PREFIX)}
    read = {"config/config.yaml", *(f"config/datasets/{d}.yaml" for d in datasets)}
    if path.with_name("config.local.yaml").exists():
        read.add("config/config.local.yaml")
    problems = []
    for key in sorted(read | set(recorded)):
        local = config_input_path(path, key)
        if key not in recorded:
            problems.append(f"{local} is read for the settings, and the {name} extraction recorded None for {key}")
            continue
        sha = recorded[key]
        if not local.exists():
            problems.append(f"{local} is missing, and the {name} extraction recorded {sha!r} for {key}")
            continue
        actual = _sha256(local)
        if actual != sha:
            problems.append(f"{local} has sha256 {actual}, and the {name} extraction recorded {sha!r} for {key}")
    if not problems:
        return None
    return (f"the config files differ from those the {name} features were extracted under, and the settings must "
            "come from those. Restore them, or extract the features again:\n  " + "\n  ".join(problems))


def require_scoring_columns(df):
    """Refuse cells that cannot be scored against their labels.

    Raises:
        SystemExit: If y or a breakdown column is missing, or y holds values other than 0 and 1.
    """
    missing = [c for c in ("y", *BREAKDOWNS.values()) if c not in df.columns]
    if missing:
        raise SystemExit(f"the features lack {missing}")
    if df["y"].isna().any() or not df["y"].isin([0, 1]).all():
        raise SystemExit("y holds values other than 0 and 1")


def src_problems(prov, name):
    """Return why features cannot be scored for the src they were extracted with.

    ``src_tree`` must be recorded and ``src_dirty`` must be false.
    """
    out = []
    if not prov.get("src_tree"):
        out.append(f"{name} features: src_tree not recorded ({prov.get('src_tree')!r})")
    if prov.get("src_dirty") is not False:
        out.append(f"{name} features: src_dirty is {prov.get('src_dirty')!r}; src must be committed and clean")
    return out


def check_parquet(path, prov):
    """Return the parquet's sha256, refusing a parquet other than the one its provenance describes.

    Raises:
        SystemExit: If the parquet is missing, or the provenance records ``outputs.features.sha256`` and it
            differs from the file's.
    """
    if not Path(path).exists():
        raise SystemExit(f"{path} not found beside its provenance. An extraction may have stopped between writing "
                         "the two; refusing.")
    actual = _sha256(path)
    recorded = ((prov.get("outputs") or {}).get("features") or {}).get("sha256")
    if recorded and recorded != actual:
        raise SystemExit(f"{path} is not the parquet its provenance describes: sha256 {actual}, provenance {recorded}. "
                         "An extraction may have stopped between writing the two; refusing.")
    return actual


def scoring_code_hashes():
    """Return ``{repo-relative path: sha256}`` of every file in :data:`SCORING_CODE`."""
    return {rel: _sha256(REPO / rel) for rel in SCORING_CODE}


def _filter_default(name, param):
    """Return the registry function's default for a fragment filter's threshold parameter."""
    return inspect.signature(FILTER_REGISTRY[name].apply).parameters[param].default


def _display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def canonical_channels(config_path, dataset):
    """Return the channels the pipeline standardizes a dataset's WARs to, in order.

    ``standardize_wars.py`` pads each WAR with NaN to ``analysis.standardization.channel_reorder`` when the
    merged config sets it, and otherwise to the keys of the samples config's ``channels`` map
    (``constants.CHANNEL_ABBREVS``). The config is merged as the pipeline merges it: ``config_path``, then
    ``config.local.yaml`` beside it, then ``datasets/<dataset>.yaml``.

    Raises:
        SystemExit: If the dataset config cannot be read or names no channels.
    """
    path = Path(config_path)
    try:
        merged = load_dataset_config(dataset, config_path=path, local_path=path.with_name("config.local.yaml"),
                                     datasets_dir=path.parent / "datasets")
        reorder = ((merged.get("analysis") or {}).get("standardization") or {}).get("channel_reorder")
        channels = list(resolve_samples_config(merged).get("channels") or {}) if reorder is None else list(reorder)
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise SystemExit(f"cannot read the channels of dataset {dataset}: {e}") from e
    if not channels:
        raise SystemExit(f"dataset {dataset} names no channels, so its standardized channels are unknown")
    return channels


def load_settings(config_path=CONFIG_PATH, datasets=()):
    """Return the F, L and M settings the pipeline reads from its config.

    The fragment filters are those listed in ``analysis.fragment_filter_config``, each with its threshold or,
    when the entry leaves it out, the filter function's default, since ``apply_filters`` passes the entry's
    keys through. LOF reads ``channel_filter_config.lof`` and the config lists ``channel_filter_config.manual``.
    Each dataset's canonical channels come from :func:`canonical_channels`.

    Args:
        config_path: ``config/config.yaml``.
        datasets: Dataset names whose ``datasets/<name>.yaml`` beside the config is checked for overrides and
            read for its canonical channels.

    Returns:
        ``{"config": {"path", "sha256"}, "fragment_filters": {filter: {parameter: value}},
        "fragment_min_valid_channels": 3, "lof": {"reject_lof_threshold", "min_valid_channels"},
        "config_lists": {"min_valid_channels"}, "canonical_channels": {dataset: [channel]}}``.

    Raises:
        SystemExit: If the config lists a fragment filter the per-cell scores cannot reproduce, sets a global
            ``reject_channels`` list or lacks a setting, if ``config.local.yaml`` or a dataset config
            overrides the filter settings (the pipeline merges both over ``config.yaml``), or if a dataset's
            channels cannot be read.
    """
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"pipeline config not found: {path}")
    analysis = (yaml.safe_load(path.read_text()) or {}).get("analysis") or {}
    overriders = [path.with_name("config.local.yaml")] + [path.parent / "datasets" / f"{d}.yaml" for d in datasets]
    for other in overriders:
        if not other.exists():
            continue
        other_analysis = (yaml.safe_load(other.read_text()) or {}).get("analysis") or {}
        touched = [k for k in ("fragment_filter_config", "channel_filter_config") if k in other_analysis]
        if touched:
            raise SystemExit(f"{other} overrides analysis.{touched[0]}; the evaluation scores the settings in "
                             f"{path}. Refusing.")
    frag = analysis.get("fragment_filter_config")
    if not frag:
        raise SystemExit(f"{path} has no analysis.fragment_filter_config")
    unknown = [name for name in frag if name not in FRAGMENT_FILTER_PARAMS]
    if unknown:
        raise SystemExit(f"fragment_filter_config lists {unknown}, which the per-cell scores cannot reproduce; "
                         f"F covers {sorted(FRAGMENT_FILTER_PARAMS)}")
    fragment = {}
    for name, params in frag.items():
        param = FRAGMENT_FILTER_PARAMS[name]
        fragment[name] = {param: (params or {}).get(param, _filter_default(name, param))}
    channel = analysis.get("channel_filter_config") or {}
    try:
        lof, manual = channel["lof"], channel["manual"]
        out = {"config": {"path": _display_path(path), "sha256": _sha256(path)},
               "fragment_filters": fragment,
               "fragment_min_valid_channels": FRAGMENT_MIN_VALID_CHANNELS,
               "lof": {"reject_lof_threshold": lof["reject_lof_threshold"],
                       "min_valid_channels": lof["min_valid_channels"]},
               "config_lists": {"min_valid_channels": manual["min_valid_channels"]}}
    except (KeyError, TypeError) as e:
        raise SystemExit(f"{path}: channel_filter_config lacks {e}") from e
    for branch in ("lof", "manual"):
        if channel[branch].get("reject_channels"):
            raise SystemExit(f"channel_filter_config.{branch}.reject_channels lists "
                             f"{channel[branch]['reject_channels']}; no member covers a global channel list")
    out["canonical_channels"] = {d: canonical_channels(path, d) for d in datasets}
    return out


def settings_differences(recorded, current):
    """Return a description of each F, L or M setting that differs between two settings dicts."""
    return [f"{key}: recorded {recorded.get(key)!r}, config now {current.get(key)!r}"
            for key in SETTINGS_KEYS if recorded.get(key) != current.get(key)]


def settings_datasets(df, prov):
    """Return the datasets whose settings to read: those of the cells and those the extraction scanned.

    Both partitions' extractions scan the same datasets (``params.datasets``), so the dev and test runs read the
    same canonical channels even when a partition has no cell in some dataset.
    """
    return sorted(set(df["dataset"].astype(str)) | set((prov.get("params") or {}).get("datasets") or []))


def _per_cell(df, recordings, value):
    """Return ``value(entry)`` of each cell's recording entry from :func:`recorded_channels`, as an array."""
    return df["recording"].astype(str).map({r: value(entry) for r, entry in recordings.items()}).to_numpy()


def _channels_with_cells(df):
    """Return ``{recording: set of channels the features hold a cell on}``."""
    return df["channel"].astype(str).groupby(df["recording"].astype(str)).agg(set).to_dict()


def fragment_reject(df, settings, recordings):
    """Return F, the fragment filters' 0/1 rejections, and an audit.

    The window rule counts every standardized channel of the cell's recording: the window's cells the filters
    keep, its padded channels, which are NaN and which every fragment filter keeps, and its own channels without
    a cell in the window (every rater abstained), counted as kept since the features lack their scores.

    Args:
        df: Cells with rms, logrmsz_abs, beta_wmax, recording, window and channel.
        settings: Output of :func:`load_settings`.
        recordings: Output of :func:`recorded_channels` for the recordings of ``df``.

    Returns:
        ``(0/1 per cell, audit)``. The audit counts the cells rejected; those rejected only by the window rule;
        the windows where the recording has a channel without a cell; and the cells, and their windows, that the
        window rule keeps only because it counts those channels as kept.
    """
    fs = settings["fragment_filters"]
    rms = df["rms"].to_numpy(float)
    rej = np.zeros(len(df), bool)
    with np.errstate(invalid="ignore"):  # a NaN comparison is false: a missing score rejects nothing
        if "high_rms" in fs:
            rej |= rms > fs["high_rms"]["max_rms"]
        if "low_rms" in fs:
            rej |= rms < fs["low_rms"]["min_rms"]
        if "logrms_range" in fs:
            rej |= df["logrmsz_abs"].to_numpy(float) > abs(fs["logrms_range"]["z_range"])
        if "high_beta" in fs:
            rej |= df["beta_wmax"].to_numpy(float) > fs["high_beta"]["max_beta_prop"]
    min_valid = settings["fragment_min_valid_channels"]
    n_standardized = _per_cell(df, recordings, lambda e: len(e["standardized"]))
    n_own = _per_cell(df, recordings, lambda e: len(e["channels"]))
    windows = [df["recording"].astype(str), df["window"]]
    n_kept = pd.Series(~rej, index=df.index).groupby(windows).transform("sum").to_numpy()
    n_cells = pd.Series(1, index=df.index).groupby(windows).transform("sum").to_numpy()
    rule = n_kept + (n_standardized - n_cells) < min_valid
    leaning = ~rule & ~rej & (n_kept + (n_standardized - n_own) < min_valid)
    out = rej | rule
    audit = {"n_cells": int(len(df)), "n_rejected": int(out.sum()), "n_window_rule_only": int((rule & ~rej).sum()),
             "n_windows_with_channels_without_cells": _n_windows(df, n_cells < n_own),
             "n_kept_by_channels_without_cells": int(leaning.sum()),
             "n_windows_kept_by_channels_without_cells": _n_windows(df, leaning)}
    return out.astype(int), audit


def _n_windows(df, cells):
    """Return the number of (recording, window) windows of the marked cells."""
    return int(len(df.loc[cells, ["recording", "window"]].drop_duplicates()))


def _recordings(df, cells):
    """Return the sorted recordings of the marked cells."""
    return sorted(df.loc[cells, "recording"].astype(str).unique())


def lof_reject(df, settings, recordings):
    """Return L, LOF's 0/1 rejections at the configured threshold, and an audit.

    The recording rule counts every standardized channel of the recording as valid unless LOF flags it: a padded
    channel has no LOF score, so the pipeline never flags it, and a channel of the recording without a cell (every
    rater abstained on it in every labeled window) counts as unflagged, since the features lack its score.

    Args:
        df: Cells with lof, recording and channel.
        settings: Output of :func:`load_settings`.
        recordings: Output of :func:`recorded_channels` for the recordings of ``df``.

    Returns:
        ``(0/1 per cell, audit)``: cells rejected, those rejected only by the recording rule, cells without a
        LOF score, and the cells, and their recordings, that the recording rule keeps only because it counts the
        recording's channels without a cell as unflagged. Cells without a score are kept; the features cannot
        tell a missing score from a NaN one, which the pipeline would reject.
    """
    lof = df["lof"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        flagged = lof >= settings["lof"]["reject_lof_threshold"]
    rec = df["recording"].astype(str)
    by_channel = pd.Series(flagged, index=df.index).groupby([rec, df["channel"].astype(str)]).any()
    n_flagged = rec.map(by_channel.groupby(level=0).sum()).to_numpy()
    n_with_cells = rec.map(by_channel.groupby(level=0).size()).to_numpy()
    n_standardized = _per_cell(df, recordings, lambda e: len(e["standardized"]))
    n_own = _per_cell(df, recordings, lambda e: len(e["channels"]))
    min_valid = settings["lof"]["min_valid_channels"]
    rule = n_standardized - n_flagged < min_valid
    leaning = ~rule & ~flagged & (n_standardized - n_flagged - (n_own - n_with_cells) < min_valid)
    out = flagged | rule
    audit = {"n_cells": int(len(df)), "n_rejected": int(out.sum()), "n_recording_rule_only": int((rule & ~flagged).sum()),
             "n_score_missing": int(np.isnan(lof).sum()), "n_kept_by_channels_without_cells": int(leaning.sum()),
             "recordings_kept_by_channels_without_cells": _recordings(df, leaning)}
    return out.astype(int), audit


def list_reject(df, settings, recordings):
    """Return M, the config lists' 0/1 rejections, and an audit.

    A cell is listed when its recording's list names its channel. The recording rule counts the standardized
    channels the list does not name (:func:`recorded_channels`), so a listed channel counts against it whether
    the recording has a cell on it or not, as the manual branch rejects it on the standardized WAR, padded or
    not.

    Args:
        df: Cells with recording and channel.
        settings: Output of :func:`load_settings`.
        recordings: Output of :func:`recorded_channels` for the recordings of ``df``.

    Returns:
        ``(0/1 per cell, audit)``: cells with a list defined, cells listed, cells rejected, those rejected only by
        the recording rule, and the listed channels each recording has no cell on (``listed_without_cells``),
        which the rule counts.
    """
    rec, channel = df["recording"].astype(str), df["channel"].astype(str)
    listed = np.array([c in recordings[r]["listed"] for r, c in zip(rec, channel)], bool)
    defined = _per_cell(df, recordings, lambda e: e["defined"]).astype(bool)
    n_unlisted = _per_cell(df, recordings, lambda e: len(e["standardized"]) - len(e["listed"]))
    rule = n_unlisted < settings["config_lists"]["min_valid_channels"]
    out = listed | rule
    have = _channels_with_cells(df)
    without = {r: [c for c in e["listed"] if c not in have[r]] for r, e in recordings.items() if r in have}
    audit = {"n_cells": int(len(df)), "n_defined": int(defined.sum()), "n_listed": int(listed.sum()),
             "n_rejected": int(out.sum()), "n_recording_rule_only": int((rule & ~listed).sum()),
             "listed_without_cells": {r: v for r, v in without.items() if v}}
    return out.astype(int), audit


def member_masks(df, settings, prov):
    """Return F, L and M per cell and their audits.

    Args:
        df: Cells with the member columns.
        settings: Output of :func:`load_settings`.
        prov: The features' provenance, with the recording records :func:`recorded_channels` reads.

    Returns:
        ``({"F", "L", "M": 0/1 per cell}, {"F", "L", "M": audit, "padded_channels": {recording: [channel]},
        "channels_without_cells": {recording: [channel]}})``: the standardized channels each recording lacks,
        and the recording's own channels it has no cell on, whose F and L decisions the rules take as kept.
    """
    recordings = recorded_channels(df, prov, settings)
    preds, audits = {}, {}
    preds["F"], audits["F"] = fragment_reject(df, settings, recordings)
    preds["L"], audits["L"] = lof_reject(df, settings, recordings)
    preds["M"], audits["M"] = list_reject(df, settings, recordings)
    have = _channels_with_cells(df)
    padded = {r: [c for c in e["standardized"] if c not in e["channels"]] for r, e in recordings.items()}
    without = {r: [c for c in e["channels"] if c not in have[r]] for r, e in recordings.items()}
    audits["padded_channels"] = {r: v for r, v in padded.items() if v}
    audits["channels_without_cells"] = {r: v for r, v in without.items() if v}
    return preds, audits


def combination_preds(members, a_pred):
    """Return ``{combination: 0/1 per cell}``; a combination rejects a cell when any member does."""
    parts = {**members, "A": np.asarray(a_pred)}
    return {name: np.logical_or.reduce([np.asarray(parts[m]).astype(bool) for m in ms]).astype(int)
            for name, ms in COMBINATIONS.items()}


def combination_cells(df):
    """Return ``{combination: bool per cell}``, the cells each combination is scored on."""
    defined = df["config_list_defined"].astype(bool).to_numpy()
    return {name: (defined if name == "config_lists_alone" else np.ones(len(df), bool)) for name in COMBINATIONS}


def _f1(y, yhat):
    return float(f1_score(y, yhat, zero_division=0))


def score_rule(df, y, preds, boot=2000, seed=0):
    """Return every combination's metrics against one set of labels, with the paired differences.

    Args:
        df: Cells with cv_group, config_list_defined and the breakdown columns.
        y: 0/1 labels per cell under one consensus rule.
        preds: Output of :func:`combination_preds`.
        boot: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        ``{combination: {"pooled", "ci95", per_* breakdowns}}``; replace and add also hold ``minus_today``,
        ``{"f1_difference", "ci95"}``.
    """
    y = np.asarray(y, int)
    groups = df[GROUP_COL].astype(str).to_numpy()
    cells = combination_cells(df)
    out = {}
    for name in COMBINATIONS:
        c = cells[name]
        yhat = preds[name][c]
        lo, hi = cluster_boot_ci(y[c], yhat, groups[c], n=boot, seed=seed) if c.any() else (np.nan, np.nan)
        out[name] = {"pooled": metrics(y[c], yhat), "ci95": [lo, hi],
                     **{key: breakdown(df[c], yhat, col, y=y[c]) for key, col in BREAKDOWNS.items()}}
    for name, base in PAIRED.items():
        lo, hi = paired_delta_ci(y, preds[name], preds[base], groups, n=boot, seed=seed)
        out[name][f"minus_{base}"] = {"f1_difference": _f1(y, preds[name]) - _f1(y, preds[base]), "ci95": [lo, hi]}
    return out


def score_combinations(df, preds, boot=2000, seed=0, alongside=None):
    """Return every combination's metrics under the primary rule and under the rules reported alongside.

    Args:
        df: Cells with y (the majority consensus), cv_group and the breakdown columns.
        preds: Output of :func:`combination_preds`.
        boot: Bootstrap resamples.
        seed: Bootstrap seed.
        alongside: ``{rule: 0/1 labels per cell}`` for the rules reported alongside majority (the ``ALONGSIDE``
            columns of :func:`rule_labels`), or None to score majority only.

    Returns:
        ``{combination: {"members", "definition", "cells", "rule", "pooled", "ci95", per_* breakdowns}}``, where
        ``rule`` names the primary rule the top-level metrics use; replace and add also hold ``minus_today``,
        ``{"f1_difference", "ci95"}``. With ``alongside``, each entry also holds ``alongside``, ``{rule: the same
        metrics, breakdowns and paired difference under that rule}``.
    """
    primary = score_rule(df, df["y"].to_numpy(int), preds, boot=boot, seed=seed)
    others = {rule: score_rule(df, labels, preds, boot=boot, seed=seed) for rule, labels in (alongside or {}).items()}
    out = {}
    for name, members in COMBINATIONS.items():
        out[name] = {"members": list(members),
                     "definition": " or ".join(f"{m}(k)" if m == "A" else m for m in members),
                     "cells": "config_list_defined" if name == "config_lists_alone" else "all",
                     "rule": PRIMARY_RULE, **primary[name]}
        if alongside is not None:
            out[name]["alongside"] = {rule: scored[name] for rule, scored in others.items()}
    return out


def choose_config(df, ar_cols, f_pred):
    """Return the autoreject configuration whose replace(k) F1 is highest on ``df``.

    Args:
        df: Cells with y and the grid columns.
        ar_cols: Grid columns in ``adr.grid_config_names()`` order.
        f_pred: F on the same cells.

    Returns:
        ``(k, replace(k) F1, tie_broken)``. F1 is compared to 4 decimals; ties go to the earlier column, and
        ``tie_broken`` says whether another configuration shared the best value.
    """
    y = df["y"].to_numpy(int)
    f = np.asarray(f_pred).astype(bool)
    f1s = [_f1(y, (f | mask_pred(df, k).astype(bool)).astype(int)) for k in ar_cols]
    rounded = [round(v, 4) for v in f1s]
    tied = [i for i, v in enumerate(rounded) if v == max(rounded)]
    return ar_cols[tied[0]], f1s[tied[0]], len(tied) > 1


def too_few(df, outer_col, inner_col=None):
    """Return why leaving one ``outer_col`` value out cannot run on ``df``, or None when it can.

    Leaving one value out needs two values of ``outer_col``. A ranking by leaving one ``inner_col`` value out
    inside each training part also needs two values of ``inner_col`` in every part.
    """
    outer = df[outer_col].astype(str).to_numpy()
    values = sorted(pd.unique(outer))
    if len(values) < 2:
        return f"leaving one {outer_col} out needs at least 2, and the cells hold {len(values)}"
    if inner_col is not None:
        inner = df[inner_col].astype(str).to_numpy()
        for value in values:
            n = len(pd.unique(inner[outer != value]))
            if n < 2:
                return (f"ranking by leaving one {inner_col} out needs at least 2 in each training part, and the "
                        f"part without {outer_col} {value} holds {n}")
    return None


def skipped(reason):
    """Return the record of a nested run the cells cannot support, with the reason under ``not_run``."""
    return {"f1": np.nan, "ci95": [np.nan, np.nan], "pooled": None, "folds": [], "per_dataset": {},
            "not_run": reason}


def nested_replace(df, ar_cols, f_pred, outer_col):
    """Choose k inside each outer training part and predict the held-out part with replace(k).

    Callers check :func:`too_few` first: ``LeaveOneGroupOut`` refuses fewer than two values of ``outer_col``.

    Args:
        df: The dev cells.
        ar_cols: Grid columns in grid order.
        f_pred: F on ``df``.
        outer_col: Column defining the outer folds (``cv_group`` or ``dataset``).

    Returns:
        ``(pooled held-out predictions, per-fold records)``; a record holds the held-out unit, k, its training
        replace F1, whether a tie was broken, and the held-out metrics.
    """
    y = df["y"].to_numpy(int)
    f = np.asarray(f_pred).astype(bool)
    outer = df[outer_col].astype(str).to_numpy()
    pred = np.zeros(len(df), int)
    folds = []
    for tr, te in LeaveOneGroupOut().split(df, y, groups=outer):
        k, train_f1, tie = choose_config(df.iloc[tr], ar_cols, f[tr])
        p = (f[te] | mask_pred(df.iloc[te], k).astype(bool)).astype(int)
        pred[te] = p
        folds.append({"held_out": str(outer[te][0]), "autoreject_config": k, "train_replace_f1": round(train_f1, 4),
                      "tie_broken": tie, **metrics(y[te], p)})
    return pred, folds


def rank(df, ar_cols, group_col=GROUP_COL):
    """Rank every single detector on ``df`` by leave-one-group-out F1 (the diagnostic ranking).

    Returns:
        ``(order, {name: F1})``, ordered by F1 to 4 decimals, ties in the fixed detector order.
    """
    y = df["y"].to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    scores = {}
    for name, (col, direction) in DETECTORS.items():
        scores[name] = evaluate(score_of(df, col, direction), y, groups, LeaveOneGroupOut())[0]
    for name in ar_cols:
        scores[name] = float(f1_score(y, mask_pred(df, name), zero_division=0))
    order = sorted(scores, key=lambda k: -round(scores[k], 4))  # stable: filters first, then grid order
    return order, scores


def fit_and_predict(train, test, name):
    """Predict ``test`` with one detector: a filter's tau is tuned on ``train``; a mask is used as is.

    Returns:
        ``(0/1 predictions, tau in feature units or None)``.
    """
    if name in DETECTORS:
        col, direction = DETECTORS[name]
        _, tau = best_tau(train["y"].to_numpy(), score_of(train, col, direction))
        return predict(score_of(test, col, direction), tau), (tau if direction == "hi" else -tau)
    return mask_pred(test, name), None


def nested(df, ar_cols, outer_col, inner_col=GROUP_COL):
    """Repeat the single-detector ranking inside each outer training part; score its first on the held-out part.

    Callers check ``too_few(df, outer_col, inner_col)`` first.

    Args:
        df: The dev cells.
        ar_cols: Autoreject columns.
        outer_col: Column defining the outer folds (``cv_group`` or ``dataset``).
        inner_col: Column defining the inner folds used to rank.

    Returns:
        ``(pooled predictions, per-fold records)``.
    """
    y = df["y"].to_numpy()
    outer = df[outer_col].astype(str).to_numpy()
    pred = np.zeros(len(df), int)
    folds = []
    for tr, te in LeaveOneGroupOut().split(df, y, groups=outer):
        inner, held = df.iloc[tr], df.iloc[te]
        order, scores = rank(inner, ar_cols, inner_col)
        pick = order[0]
        p, tau = fit_and_predict(inner, held, pick)
        pred[te] = p
        folds.append({"held_out": str(outer[te][0]), "pick": pick, "inner_logo_f1": round(scores[pick], 4),
                      "tau": tau, **metrics(y[te], p)})
    return pred, folds


def diagnostic(df, ar_cols, boot=2000):
    """Return the ranking of the 43 single detectors with its nested runs; it decides nothing.

    Filters get a threshold tuned on the dev labels (``tau``, deployable), a leave-one-group-out F1 with the
    threshold refit without the held-out group, and a leave-one-dataset-out F1; autoreject masks are used as
    they are. Each entry holds pooled metrics (held-out predictions for filters), an F1 interval, and
    per-group and per-dataset metrics.

    With fewer than two cv_groups nothing is ranked, and ``not_run`` says why. With one dataset a filter's
    leave-one-dataset-out F1 is NaN, and a nested run the cells cannot support (:func:`too_few`) is
    :func:`skipped`.
    """
    note = "single detectors ranked one at a time; this ranking decides nothing"
    problem = too_few(df, GROUP_COL)
    if problem:
        return {"note": note, "not_run": problem, "ranked_first": None, "order": [], "leaderboard": {},
                "nested_logo": skipped(problem), "nested_lodo": skipped(problem)}
    y = df["y"].to_numpy()
    groups = df[GROUP_COL].astype(str).to_numpy()
    datasets = df["dataset"].astype(str).to_numpy()
    one_dataset = too_few(df, "dataset") is not None
    res = {}
    for name, (col, direction) in DETECTORS.items():
        s = score_of(df, col, direction)
        all_f1, _, _, taus_all = evaluate(s, y, groups, _AllInOneFold())
        logo_f1, oof, _, _ = evaluate(s, y, groups, LeaveOneGroupOut())
        lodo_f1 = np.nan if one_dataset else evaluate(s, y, datasets, LeaveOneGroupOut())[0]
        lo, hi = cluster_boot_ci(y, oof, groups, n=boot)
        res[name] = {"kind": "filter", "col": col, "direction": direction,
                     "tau": (taus_all[0] if direction == "hi" else -taus_all[0]),
                     "all_dev_f1": round(all_f1, 4), "logo_f1": round(logo_f1, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "lodo_f1": round(lodo_f1, 4), "n_score_missing": int((~np.isfinite(s)).sum()),
                     "pooled": metrics(y, oof),
                     "per_group": breakdown(df, oof, GROUP_COL), "per_dataset": breakdown(df, oof, "dataset"),
                     "_yhat": oof}
    for name in ar_cols:
        yhat = mask_pred(df, name)
        f1 = _f1(y, yhat)
        lo, hi = cluster_boot_ci(y, yhat, groups, n=boot)
        res[name] = {"kind": "autoreject", "col": name, "direction": "reject", "tau": None,
                     "all_dev_f1": round(f1, 4), "logo_f1": round(f1, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "lodo_f1": round(f1, 4), "n_score_missing": int(df[name].isna().sum()),
                     "pooled": metrics(y, yhat),
                     "per_group": breakdown(df, yhat, GROUP_COL), "per_dataset": breakdown(df, yhat, "dataset"),
                     "_yhat": yhat}
    order, _ = rank(df, ar_cols)
    first, runner = order[0], order[1]
    dlo, dhi = paired_delta_ci(y, res[first]["_yhat"], res[runner]["_yhat"], groups, n=boot)
    problem = too_few(df, GROUP_COL, GROUP_COL)
    if problem:
        nested_logo = skipped(problem)
    else:
        nested_pred, nested_folds = nested(df, ar_cols, GROUP_COL)
        n_lo, n_hi = cluster_boot_ci(y, nested_pred, groups, n=boot)
        nested_logo = {"f1": round(_f1(y, nested_pred), 4), "ci95": [round(n_lo, 4), round(n_hi, 4)],
                       "pooled": metrics(y, nested_pred), "folds": nested_folds}
    problem = too_few(df, "dataset", GROUP_COL)
    if problem:
        nested_lodo = skipped(problem)
    else:
        lodo_pred, lodo_folds = nested(df, ar_cols, "dataset")
        nested_lodo = {"f1": round(_f1(y, lodo_pred), 4), "pooled": metrics(y, lodo_pred), "folds": lodo_folds,
                       "per_dataset": breakdown(df, lodo_pred, "dataset")}
    return {
        "note": note,
        "ranked_first": {"detector": first, "kind": res[first]["kind"], "tau": res[first]["tau"],
                         "logo_f1": res[first]["logo_f1"], "runner_up": runner,
                         "delta_ci95": [round(dlo, 4), round(dhi, 4)]},
        "order": order,
        "leaderboard": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()},
        "nested_logo": nested_logo,
        "nested_lodo": nested_lodo,
    }


def rater_agreement(df):
    """Return pairwise rater agreement on the cells, overall and per dataset, from the category columns."""
    cat_cols = [c for c in df.columns if c.startswith(RATER_CATEGORY_PREFIX)]
    if len(cat_cols) < 2:
        return {"overall": [], "per_dataset": []}
    long = df.melt(id_vars=["recording", "window", "channel", "dataset"], value_vars=cat_cols,
                   var_name="rater", value_name="category")
    long["rater"] = long["rater"].str[len(RATER_CATEGORY_PREFIX):]
    long["y"] = long["category"].map(CATEGORY_VOTE)
    overall = pairwise_agreement(long.drop(columns="dataset"))
    per_ds = pairwise_agreement(long, by="dataset")
    return {"overall": overall.to_dict("records"), "per_dataset": per_ds.to_dict("records")}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(o):
    """Serialize numpy scalars for ``json.dumps``."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def describe_settings(settings):
    """Return the F and L settings as one plain-text line."""
    fs = settings["fragment_filters"]
    parts = []
    if "high_rms" in fs:
        parts.append(f"rms > {fs['high_rms']['max_rms']}")
    if "low_rms" in fs:
        parts.append(f"rms < {fs['low_rms']['min_rms']}")
    if "logrms_range" in fs:
        parts.append(f"log RMS |z| > {abs(fs['logrms_range']['z_range'])}")
    if "high_beta" in fs:
        parts.append(f"window beta proportion > {fs['high_beta']['max_beta_prop']}")
    canonical = ", ".join(f"{d} {len(c)}" for d, c in (settings.get("canonical_channels") or {}).items())
    return (f"F rejects {', '.join(parts)}, then windows with fewer than {settings['fragment_min_valid_channels']} "
            f"valid channels; L rejects LOF >= {settings['lof']['reject_lof_threshold']}; canonical channels: "
            f"{canonical or 'none'}")


def _f1_with_ci(f1, ci):
    return f"{f1:6.3f} [{ci[0]:6.3f},{ci[1]:6.3f}]"


def print_combinations(combos):
    """Print each combination's majority metrics with the F1 interval and the paired differences, then F1 with
    its interval under every rule, the paired differences included."""
    print(f"{'combination':20} {'F1':>6} {'95% CI':>15}  {'prec':>6} {'recall':>6} {'kappa':>6} {'flags':>6} "
          f"{'cells':>6}   ({PRIMARY_RULE})")
    for name, c in combos.items():
        p = c["pooled"]
        print(f"{name:20} {_f1_with_ci(p['f1'], c['ci95'])}  {p['precision']:6.3f} {p['recall']:6.3f} "
              f"{p['cohen_kappa']:6.3f} {p['flag_rate']:6.3f} {p['n']:6d}")
    for name, base in PAIRED.items():
        d = combos[name][f"minus_{base}"]
        lo, hi = d["ci95"]
        print(f"{name} minus {base}: F1 difference {d['f1_difference']:.3f} [{lo:.3f}, {hi:.3f}]")
    rules = [r for r in ALONGSIDE if all(r in (c.get("alongside") or {}) for c in combos.values())]
    if not rules:
        return
    print(f"\nF1 with its 95% CI under each rule:\n{'':20} " + " ".join(f"{r:>22}" for r in (PRIMARY_RULE, *rules)))
    for name, c in combos.items():
        entries = [c, *(c["alongside"][r] for r in rules)]
        print(f"{name:20} " + " ".join(_f1_with_ci(e["pooled"]["f1"], e["ci95"]) for e in entries))
    for name, base in PAIRED.items():
        key = f"minus_{base}"
        entries = [combos[name][key], *(combos[name]["alongside"][r][key] for r in rules)]
        print(f"{name + ' minus ' + base:20} " + " ".join(_f1_with_ci(e["f1_difference"], e["ci95"]) for e in entries))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Choose the autoreject configuration for the pipeline's combination on the dev features.")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=REPO / "config/labeling/split.json")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH,
                    help="pipeline config whose filter settings define F, L and M (default: config/config.yaml)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--top", type=int, default=10, help="how many diagnostic single detectors to print")
    args = ap.parse_args(argv)

    from score_detectors import check_split, provenance

    prov_path = Path(str(args.features) + ".provenance.json")
    if not prov_path.exists():
        raise SystemExit(f"missing {prov_path}; features without provenance cannot be selected on")
    features_prov = json.loads(prov_path.read_text())
    features_sha = check_parquet(args.features, features_prov)
    which = (features_prov.get("params") or {}).get("which")
    problems = [] if which == "dev" else [f"dev features: the provenance says which {which!r}"]
    problems += src_problems(features_prov, "dev")
    if problems:
        raise SystemExit("score_test.py would refuse these features, so they are not selected on:\n  "
                         + "\n  ".join(problems))
    require_recording_records(features_prov, "dev")
    df = pd.read_parquet(args.features)
    check_split(set(df["animal"]), args.split, "dev")
    ar_cols = classify_columns(df)
    require_member_columns(df)
    require_scoring_columns(df)
    datasets = settings_datasets(df, features_prov)
    problem = config_problem(args.config, features_prov, "dev", datasets)
    if problem:
        raise SystemExit(problem)
    labels = rule_labels(df)
    require_majority(df, labels, features_prov)
    settings = load_settings(args.config, datasets=datasets)
    y = df["y"].to_numpy(int)
    groups = df[GROUP_COL].astype(str).to_numpy()
    print(f"{len(df)} cells over {df['animal'].nunique()} animals in {len(pd.unique(groups))} groups and "
          f"{df['dataset'].nunique()} datasets; {len(ar_cols)} autoreject configurations")
    print(f"settings from {settings['config']['path']} (sha256 {settings['config']['sha256'][:12]}): "
          f"{describe_settings(settings)}\n")

    members, audits = member_masks(df, settings, features_prov)
    k, k_f1, tie = choose_config(df, ar_cols, members["F"])
    preds = combination_preds(members, mask_pred(df, k))
    combos = score_combinations(df, preds, boot=args.boot, alongside={r: labels[r].to_numpy() for r in ALONGSIDE})
    print(f"pick: {k}, replace F1 {k_f1:.4f}{', tie broken by grid order' if tie else ''}\n")
    print_combinations(combos)

    nested_out = {}
    for key, outer in (("nested_replace_logo", GROUP_COL), ("nested_replace_lodo", "dataset")):
        problem = too_few(df, outer)
        if problem:
            nested_out[key] = skipped(problem)
            print(f"\n{key}: not run; {problem}")
            continue
        pred, folds = nested_replace(df, ar_cols, members["F"], outer)
        lo, hi = cluster_boot_ci(y, pred, groups, n=args.boot)
        nested_out[key] = {"f1": round(_f1(y, pred), 4), "ci95": [round(lo, 4), round(hi, 4)],
                           "pooled": metrics(y, pred), "folds": folds, "per_dataset": breakdown(df, pred, "dataset")}
        print(f"\n{key}: replace F1 of the whole choice {nested_out[key]['f1']:.3f} [{lo:.3f}, {hi:.3f}]")
        for f in folds:
            print(f"  held out {f['held_out']}: k {f['autoreject_config']}, train F1 {f['train_replace_f1']:.3f}, "
                  f"held-out F1 {f['f1']:.3f}")

    diag = diagnostic(df, ar_cols, boot=args.boot)
    if diag.get("not_run"):
        print(f"\ndiagnostic ranking of single detectors not run; {diag['not_run']}")
    else:
        print(f"\ndiagnostic ranking of single detectors (decides nothing), first {args.top}:")
    for name in diag["order"][:args.top]:
        r = diag["leaderboard"][name]
        print(f"  {name:34} logo F1 {r['logo_f1']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  {r['kind']}")
    for key in ("nested_logo", "nested_lodo"):
        if diag[key].get("not_run") and not diag.get("not_run"):
            print(f"diagnostic {key} not run; {diag[key]['not_run']}")

    agreement = rater_agreement(df)
    for row in agreement["overall"]:
        print(f"raters {row['rater_a']} and {row['rater_b']}: kappa {row['cohen_kappa']:.3f} "
              f"over {row['n_cells']} cells")

    prov = provenance({"features": args.features, "split": args.split, "config": args.config},
                      script="select_detectors", boot=args.boot, group_col=GROUP_COL)
    prov["code"] = {**(prov.get("code") or {}), **scoring_code_hashes()}
    out = {
        "pick": {"autoreject_config": k, "replace_f1": round(k_f1, 4), "tie_broken": tie},
        "settings": settings,
        "member_masks": audits,
        "combinations": combos,
        **nested_out,
        "diagnostic": diag,
        "rater_agreement": agreement,
        "features": {"path": str(args.features), "sha256": features_sha, "provenance": features_prov},
        "provenance": prov,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=json_default))
    print(f"\nfrozen pick: {k}, written to {args.out}")


if __name__ == "__main__":
    main()
