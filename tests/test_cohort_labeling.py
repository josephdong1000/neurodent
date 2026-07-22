"""Tests for the cross-animal, blinded labeling cohort bundler.

Two layers:

* Fast, pure-function tests for the channel blinding (``render_context.blind_channels`` /
  ``neutral_labels``) and de-scramble (``scoring.unblind``).
* Integration tests that drive the real ``config/datasets/mini_real.yaml`` committed dataset through
  ``load_animal_recordings`` -> ``render_lro`` -> ``build`` -- the first coverage of the
  AnimalOrganizer -> LRO -> render path on real data (previously only a synthetic in-memory LRO was
  tested). Marked ``integration`` + ``slow`` since they load recordings via SpikeInterface.
"""
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "labeling"))
import build_cohort_bundle as C  # noqa: E402
import build_rater_bundle as B  # noqa: E402
import render_context as R  # noqa: E402
import score_detectors as SD  # noqa: E402

from neurodent.results.scoring import ingest, unblind  # noqa: E402

MINI_REAL_ABBREVS = {"LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"}


# --------------------------------------------------------------------------- pure-function tests

def test_blind_channels_deterministic_and_reversible():
    names = ["LMot", "RMot", "LHip", "RHip"]
    perm, disp, keymap = R.blind_channels(names, "7:A10__0")
    perm2, _, _ = R.blind_channels(names, "7:A10__0")
    perm3, _, _ = R.blind_channels(names, "7:F22__0")

    assert list(perm) == list(perm2), "same seed_key must reproduce the permutation"
    assert list(perm) != list(perm3), "different recording must scramble differently"
    assert disp == ["Ch A", "Ch B", "Ch C", "Ch D"], "neutral labels blind anatomy"
    # slot i shows the channel at original index perm[i]; the keymap records exactly that.
    assert [k["channel"] for k in keymap] == [names[i] for i in perm]
    assert [k["slot"] for k in keymap] == disp


def test_neutral_labels_two_letter_fallback():
    labels = R.neutral_labels(28)
    assert labels[25] == "Ch Z"
    assert labels[26] == "Ch AA"
    assert labels[27] == "Ch AB"
    assert len(set(labels)) == 28, "labels must be unique so they can key manifest columns"


def test_unblind_round_trip_and_missing_raises():
    long_df = pd.DataFrame({
        "recording": ["A10__0", "A10__0"], "window": [12, 12], "t_start_s": [60.0, 60.0],
        "channel": ["Ch A", "Ch B"], "rater": ["r1", "r1"], "y": [0, 1],
        "category": ["clean", "artifact"],
    })
    keymap = pd.DataFrame({
        "recording": ["A10__0", "A10__0"], "window": [12, 12], "slot": ["Ch A", "Ch B"],
        "channel": ["LHip", "RMot"],
    })
    out = unblind(long_df, keymap)
    assert out["channel"].tolist() == ["LHip", "RMot"], "slots resolve to true channels"
    assert out["slot"].tolist() == ["Ch A", "Ch B"], "neutral slot is preserved"

    orphan = long_df.assign(channel=["Ch A", "Ch Z"])  # 'Ch Z' has no keymap entry
    with pytest.raises(ValueError, match="no keymap entry"):
        unblind(orphan, keymap)


def test_mixed_bundle_union_columns_and_unblind(tmp_path):
    """One mixed bundle holding an 8-channel and a 10-channel recording, PER-WINDOW blinded: the manifest
    shares a UNION of neutral slots, geometry stays per-recording (neutral), each window scrambles
    independently, and ingest -> unblind (keyed on recording, window, slot) drops the blank padding while
    recovering true channels (and still raises on a LABELLED unmapped cell)."""
    fs = 250
    N = 200 * fs                                   # long enough for mid-recording windows + context flanks
    rng = np.random.RandomState(0)
    union = R.neutral_labels(10)                   # Ch A .. Ch J
    wins = [16, 30]                                # TWO windows (both fit the 200s span), for per-window scramble
    true8 = ["T8_" + c for c in "ABCDEFGH"]
    true10 = ["T10_" + c for c in "ABCDEFGHIJ"]

    # 8-channel then 10-channel, PER-WINDOW blinded, appended into ONE manifest sharing the 10-slot union.
    _, km8 = R.render_windows((50e-6 * rng.randn(8, N)).astype("float32"), fs, true8,
                              tmp_path, wins, "rec8", dpi=50, all_channels=union, append=False, blind_seed=1)
    _, km10 = R.render_windows((50e-6 * rng.randn(10, N)).astype("float32"), fs, true10,
                               tmp_path, wins, "rec10", dpi=50, all_channels=union, append=True, blind_seed=1)
    km = pd.DataFrame(km8 + km10)                  # columns: recording, window, slot, channel

    # Manifest header is the union; geometry lists each recording's NEUTRAL slots (no true names).
    with open(tmp_path / "manifest.csv") as fh:
        header = next(csv.reader(fh))
    assert [c[len("label_"):] for c in header if c.startswith("label_")] == union
    geom = json.loads((tmp_path / "geometry.json").read_text())
    assert [r["channel"] for r in geom["rec8"]["rows"]] == R.neutral_labels(8)
    assert [r["channel"] for r in geom["rec10"]["rows"]] == union

    # PER-WINDOW: rec10's two windows must have DIFFERENT slot->true maps.
    m = {(k["window"], k["slot"]): k["channel"] for k in km10}
    assert any(m[(wins[0], s)] != m[(wins[1], s)] for s in union), "windows must scramble independently"

    # build() -> one zip that opens, bakes the union, and leaks no true channel name.
    zpath = B.build(tmp_path, name="mixed")
    with zipfile.ZipFile(zpath) as zf:
        blob = b"".join(zf.read(n) for n in zf.namelist())
    assert b'"Ch J"' in blob                        # the 10th union slot is present in the baked CHANNELS
    for t in true8 + true10:
        assert t.encode() not in blob, f"true channel {t!r} leaked into the bundle"

    # Simulate a rater: label each recording's REAL slots, blank the padding (as the HTML export does).
    filled = pd.read_csv(tmp_path / "manifest.csv", dtype=str).fillna("")
    for i, r in filled.iterrows():
        real = set(R.neutral_labels(8) if r["recording"] == "rec8" else union)
        for c in union:
            filled.at[i, "label_" + c] = "bad" if c in real else ""
    rater_csv = tmp_path / "r1.csv"
    filled.to_csv(rater_csv, index=False)

    long_df = ingest({"r1": rater_csv})
    out = unblind(long_df, km)
    assert not out["channel"].str.startswith("Ch ").any(), "every kept cell mapped to a true channel"
    assert set(out.loc[out["recording"] == "rec8", "channel"]) == set(true8)     # padding dropped, true recovered
    assert set(out.loc[out["recording"] == "rec10", "channel"]) == set(true10)

    # A LABELLED cell in a padding slot the recording lacks is a real mismatch -> must raise.
    bad = long_df.copy()
    bad.loc[(bad["recording"] == "rec8") & (bad["channel"] == "Ch I"), "y"] = 1
    with pytest.raises(ValueError, match="keymap entry"):
        unblind(bad, km)


# --------------------------------------------------------------------------- integration on mini_real

@pytest.fixture
def _at_repo_root(monkeypatch):
    """Dataset extract_func paths resolve relative to the CWD, so run from the repo root."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_load_animal_recordings_mini_real(_at_repo_root):
    """The shared loader reconstructs a real animal's recordings with a canonical montage."""
    pytest.importorskip("spikeinterface")
    from neurodent.core.utils import resolve_channels
    from neurodent.workflow.utils import load_animal_recordings

    config, samples_config = C._prepare("mini_real")
    ao = load_animal_recordings(samples_config, config, [("", "A10", "")], "A10")
    assert ao.long_recordings, "expected at least one loaded recording for A10"
    abbrevs = set(resolve_channels(list(ao.long_recordings[0].channel_names)))
    assert abbrevs <= MINI_REAL_ABBREVS, f"unexpected channels: {abbrevs - MINI_REAL_ABBREVS}"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_build_cohort_blinded_bundle_mini_real(_at_repo_root, tmp_path):
    """End-to-end: build a blinded cohort bundle, prove blinding holds, and round-trip the labels.

    mini_real recordings are 120 s (only one window survives the context-flank margins), so one window
    per animal is drawn -- enough to exercise cross-animal append (A10 + F22 -> one 10-ch bundle),
    the keymap, blinding integrity, and ingest -> unblind.
    """
    pytest.importorskip("spikeinterface")

    rc = C.build_cohort(["mini_real"], tmp_path, n_per_animal=1, seed=0, blind_seed=1)
    assert rc == 0, "no animal should have failed"

    # Keymap lives OUTSIDE any bundle dir and maps neutral slots to canonical channels.
    keymap_csv = tmp_path / "_unblind" / "keymap.csv"
    km = pd.read_csv(keymap_csv)
    assert not km.empty
    assert km["slot"].str.startswith("Ch ").all(), "display slots must be neutral"
    assert set(km["channel"]) <= MINI_REAL_ABBREVS, "keymap stores true canonical channels"
    assert set(km["recording"]) == {"A10__0", "F22__0"}, "both animals appended into the cohort"

    # A bundle zip was built, and its manifest label columns are neutral slots (not anatomy).
    zips = list(tmp_path.rglob("rating_bundle_*.zip"))
    assert zips, "expected a bundle zip"
    manifest = next(tmp_path.rglob("manifest.csv"))
    with open(manifest) as fh:
        header = next(csv.reader(fh))
    label_cols = [c[len("label_"):] for c in header if c.startswith("label_")]
    assert label_cols and all(c.startswith("Ch ") for c in label_cols), label_cols
    # Cross-animal append: both recordings share one manifest.
    recs = {row["recording"] for row in csv.DictReader(open(manifest))}
    assert recs == {"A10__0", "F22__0"}

    # Blinding integrity: no true channel name may appear anywhere inside the shipped zip.
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            blob = b"".join(zf.read(n) for n in zf.namelist())
        for true_name in set(km["channel"]):
            assert true_name.encode() not in blob, f"true channel {true_name!r} leaked into {z.name}"

    # Round-trip: a rater labels every neutral slot; ingest -> unblind recovers the true channels.
    filled = pd.read_csv(manifest)
    for c in [c for c in filled.columns if c.startswith("label_")]:
        filled[c] = "clean"
    rater_csv = tmp_path / "rater1.csv"
    filled.to_csv(rater_csv, index=False)

    long_df = ingest({"r1": rater_csv})
    assert set(long_df["channel"]) == set(label_cols), "ingest keys on neutral slots pre-unblind"
    restored = unblind(long_df, km)
    assert set(restored["channel"]) <= MINI_REAL_ABBREVS, "unblind restores true channels"
    # Every (recording, slot) resolved to the exact channel the keymap recorded.
    check = restored.merge(km, left_on=["recording", "slot"], right_on=["recording", "slot"],
                           suffixes=("", "_km"))
    assert (check["channel"] == check["channel_km"]).all()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_score_detectors_mini_real(_at_repo_root, tmp_path):
    """End-to-end DETECTOR scoring on real (tiny) data, proving no drift between the bundle builder and
    the scorer: build the blinded bundle, fake a rater, then score the fragment detectors via the SHARED
    ``load_cohort_animal`` seam + the WAR feature path. The key assertion is ``n_uncovered == 0`` -- every
    labelled window matched a 5 s detector fragment, i.e. the scorer's ``window_s=FRAG_S`` re-extraction
    lines up 1:1 with what the rater labelled.
    """
    pytest.importorskip("spikeinterface")
    from neurodent.analysis import AnimalAnalyzer

    rc = C.build_cohort(["mini_real"], tmp_path, n_per_animal=1, seed=0, blind_seed=1)
    assert rc == 0
    keymap_csv = tmp_path / "_unblind" / "keymap.csv"
    manifest = next(tmp_path.rglob("manifest.csv"))

    # Fake a rater: label every real slot "bad" so there are reject cells (recall is then defined).
    filled = pd.read_csv(manifest)
    for c in [c for c in filled.columns if c.startswith("label_")]:
        filled[c] = "bad"
    rater_csv = tmp_path / "r1.csv"
    filled.to_csv(rater_csv, index=False)

    # Consensus via the exact score_bundle round-trip the scorer reuses.
    cons = SD.build_consensus({"r1": rater_csv}, keymap_csv, "majority")
    assert not cons.empty

    # Score every labelled animal through the SHARED loader + WAR-gen feature path (window_s=FRAG_S).
    config, samples_config = C._prepare("mini_real")
    animals = {r.rsplit("__", 1)[0] for r in cons["recording"].unique()}
    rows, ar_rows = [], []
    for animal_id in sorted(animals):
        ao = C.load_cohort_animal(samples_config, config, animal_id)
        war = AnimalAnalyzer(ao).compute_windowed_analysis(
            SD.FEATURES, window_s=R.FRAG_S, apply_notch_filter=True, multiprocess_mode="serial")
        rows += SD.score_animal(war, ao, animal_id, cons)
        # autoreject: the 6 adapted arms, fit per-LRO on the whole (tiny) recording, scored via score_mask.
        ar_rows += SD.score_animal_autoreject(ao, animal_id, cons, max_fit_fragments=None, seed=0)
    assert rows, "expected per-(recording, detector) scores"

    df = pd.DataFrame(rows)
    assert set(df["detector"]) == set(SD.FRAGMENT_DETECTORS), "every fragment detector scored"
    assert df["n_uncovered"].fillna(0).sum() == 0, "every labelled window aligned to a 5 s detector fragment"
    assert df["n"].fillna(0).sum() > 0, "some labelled cells were scored"
    scored = df[df["n"].fillna(0) > 0]
    for metric in ("precision", "recall", "f1"):
        assert scored[metric].between(0.0, 1.0).all(), f"{metric} out of [0,1]"

    # Autoreject arms: all 6 configs scored, aligned (n_uncovered==0), metrics well-formed.
    adf = pd.DataFrame(ar_rows)
    from neurodent.results import autoreject_detector as adr
    assert set(adf["detector"]) == set(adr.CONFIG_NAMES), "all 6 autoreject arms scored"
    assert adf["n_uncovered"].fillna(0).sum() == 0, "autoreject grids aligned to the labelled 5 s fragments"
    ascored = adf[adf["n"].fillna(0) > 0]
    for metric in ("precision", "recall", "f1"):
        assert ascored[metric].between(0.0, 1.0).all(), f"autoreject {metric} out of [0,1]"
