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
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "labeling"))
import build_cohort_bundle as C  # noqa: E402
import render_context as R  # noqa: E402

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
        "recording": ["A10__0", "A10__0"], "slot": ["Ch A", "Ch B"], "channel": ["LHip", "RMot"],
    })
    out = unblind(long_df, keymap)
    assert out["channel"].tolist() == ["LHip", "RMot"], "slots resolve to true channels"
    assert out["slot"].tolist() == ["Ch A", "Ch B"], "neutral slot is preserved"

    orphan = long_df.assign(channel=["Ch A", "Ch Z"])  # 'Ch Z' has no keymap entry
    with pytest.raises(ValueError, match="no keymap entry"):
        unblind(orphan, keymap)


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

    rc = C.build_cohort("mini_real", tmp_path, n_per_animal=1, seed=0, blind_seed=1)
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
