"""Tests for the labeling tooling: rendering, geometry, and the rater bundle (issue #208).

The scorer these feed is tested in tests/test_scoring.py.

The failure worth testing for is emitting labels attached to the wrong channel or window, which no
eyeballing of the output would reveal. So geometry is checked against the rendered pixels rather than
against the arithmetic that produced it, and the end-to-end test drives a real browser, since raters
see windows in a shuffled order.
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

import build_rater_bundle as B  # noqa: E402
import render_context as R  # noqa: E402

from neurodent.results.scoring import consensus, ingest, score_mask

FS = 250
CHS = ["ch0", "ch1", "ch2", "ch3"]


def synth(n_ch=4, secs=200, seed=0):
    return np.random.RandomState(seed).randn(n_ch, secs * FS) * 50e-6


# --------------------------------------------------------------------------- geometry

def test_geometry_is_stable_across_windows(tmp_path):
    """One geometry is stored per recording; that is only sound if every window lays out the same."""
    d = synth()
    R.render_windows(d, FS, CHS, tmp_path, [1, 20, 39], "A", dpi=60)   # raises if rows move
    rows = json.loads((tmp_path / "geometry.json").read_text())["A"]["rows"]
    assert [r["channel"] for r in rows] == CHS
    assert all(0.0 < r["top"] < r["bottom"] < 1.0 for r in rows)


def test_geometry_drift_is_caught(tmp_path, monkeypatch):
    """The guard must actually fire -- otherwise it is decoration."""
    d = synth()
    calls = {"n": 0}
    real = R.plt.figure

    def wobble(*a, **k):                       # make the 2nd window lay out differently
        calls["n"] += 1
        if calls["n"] == 2:
            k["figsize"] = (16, 10)
        return real(*a, **k)

    monkeypatch.setattr(R.plt, "figure", wobble)
    with pytest.raises(AssertionError, match="rows moved"):
        R.render_windows(d, FS, CHS, tmp_path, [1, 20], "A", dpi=60)


def test_geometry_rows_land_on_the_right_channel_in_pixels(tmp_path):
    """Check the recorded rows against the rendered image, not against the same arithmetic.

    Channel 2 gets dense noise (lots of ink), the others a slow sine (little ink). Per-channel
    y-scaling equalises amplitude, so ink density is what distinguishes them. If ch2's geometry does
    not land on the inky panel, the buttons point at the wrong trace.
    """
    from PIL import Image

    t = np.arange(200 * FS) / FS
    d = np.stack([np.sin(2 * np.pi * 2 * t) * 50e-6] * 4)
    d[2] = np.random.RandomState(1).randn(t.size) * 50e-6     # the inky one

    R.render_windows(d, FS, CHS, tmp_path, [20], "A", dpi=90)
    img = np.asarray(Image.open(next((tmp_path / "images").glob("*.png"))).convert("L"))
    H, W = img.shape
    rows = json.loads((tmp_path / "geometry.json").read_text())["A"]["rows"]

    ink = []
    for r in rows:
        band = img[int(r["top"] * H):int(r["bottom"] * H), int(0.12 * W):int(0.70 * W)]
        ink.append(float((band < 100).mean()))

    assert int(np.argmax(ink)) == 2, f"geometry points at the wrong channel; ink per row = {ink}"
    assert ink[2] > 2 * max(ink[i] for i in (0, 1, 3)), f"not a decisive margin: {ink}"


# --------------------------------------------------------------------------- dead channel

def _live_axis_occupancy(psd_lim, data, fs, win):
    """What fraction of the shared PSD axis the LIVE channels actually occupy, given limits.

    Deliberately NOT a copy of psd_limits' internals -- it measures the limits the code returns, so a
    regression to a naive min()..max() makes this shrink and the test fail.
    """
    from scipy.signal import welch
    f, p = welch(data, fs=fs, nperseg=min(win, 2000), axis=1)
    m = (f >= R.FMIN) & (f <= R.FMAX)
    pdb = 10 * np.log10(p[:, m] + 1e-20)
    live = np.ptp(data, axis=1) > 0
    lo, hi = psd_lim
    return float(pdb[live].max() - pdb[live].min()) / (hi - lo)


def test_dead_channel_still_renders_and_does_not_wreck_the_shared_psd_axis(tmp_path):
    """Flatline is a `bad` category, so it will appear. Read naively it is -200 dB and stretches the
    shared PSD axis until the live channels occupy a few percent of it, ruining every PSD panel."""
    d = synth()
    d[2] = 0.0                                       # disconnected electrode
    win = int(R.FRAG_S * FS)

    R.render_windows(d, FS, CHS, tmp_path, [5], "A", dpi=60)
    assert list((tmp_path / "images").glob("*.png"))

    used = _live_axis_occupancy(R.psd_limits(d, FS, win), d, FS, win)
    assert used > 0.15, f"live channels use only {used:.1%} of the shared PSD axis"

    # ...and the naive implementation this guards against must actually fail that bar, or the test
    # above is measuring nothing.
    from scipy.signal import welch
    f, p = welch(d, fs=FS, nperseg=min(win, 2000), axis=1)
    m = (f >= R.FMIN) & (f <= R.FMAX)
    naive = 10 * np.log10(p[:, m] + 1e-20)
    naive_used = _live_axis_occupancy((naive.min() - 15, naive.max() + 15), d, FS, win)
    assert naive_used < 0.05, "the naive axis is not actually broken; this test proves nothing"

    # The dead channel's own trace panel: p99.5 is 0, so without a floor it gets a zero-height axis.
    assert R.trace_ylim(d)[2] >= R.YLIM_FLOOR_UV
    assert np.percentile(np.abs(d[2]), 99.5) == 0.0


# --------------------------------------------------------------------------- cross-animal append

def test_append_with_different_channels_is_refused(tmp_path):
    """The multi-animal path. Silently writing animal 2's labels under animal 1's channel names would
    corrupt the ground truth across the whole campaign."""
    R.render_windows(synth(4), FS, ["a", "b", "c", "d"], tmp_path, [5], "R1", dpi=60)
    with pytest.raises(ValueError, match="do not match the existing manifest"):
        R.render_windows(synth(3), FS, ["x", "y", "z"], tmp_path, [5], "R2", append=True, dpi=60)


def test_append_with_same_channels_accumulates(tmp_path):
    R.render_windows(synth(), FS, CHS, tmp_path, [5], "R1", dpi=60)
    R.render_windows(synth(seed=2), FS, CHS, tmp_path, [5], "R2", append=True, dpi=60)
    rows = list(csv.DictReader(open(tmp_path / "manifest.csv")))
    assert [r["recording"] for r in rows] == ["R1", "R2"]
    assert len({r["image"] for r in rows}) == 2                    # recording-prefixed, no collision
    assert set(json.loads((tmp_path / "geometry.json").read_text())) == {"R1", "R2"}


# --------------------------------------------------------------------------- end-to-end

@pytest.mark.slow
@pytest.mark.browser
def test_end_to_end_known_artifacts_survive_the_round_trip(tmp_path):
    """Inject artifacts at known (window, channel) cells, render, bundle, drive the real page, click
    exactly those cells, export, ingest, score.

    Raters see windows in a per-rater shuffled order, so this is what proves cell identity survives
    image -> manifest -> bundle -> shuffled display -> CSV -> consensus -> score. A perfect detector
    must come back at F1 = 1.0; less means the plumbing moved a label.
    """
    pw = pytest.importorskip("playwright.sync_api")

    windows = [10, 11, 12, 13]
    truth = {(11, "ch2"), (13, "ch0"), (13, "ch1")}      # the cells we will call `bad`

    d = synth(secs=200, seed=3)
    for w, ch in truth:                                   # make them look the part
        i = CHS.index(ch)
        s = int(w * R.FRAG_S * FS)
        d[i, s:s + int(R.FRAG_S * FS)] += np.random.RandomState(w).randn(int(R.FRAG_S * FS)) * 900e-6

    R.render_windows(d, FS, CHS, tmp_path, windows, "A", dpi=70)
    zip_path = B.build(tmp_path, name="e2e")

    work = tmp_path / "unzipped"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(work)

    with pw.sync_playwright() as p:
        br = p.chromium.launch()
        page = br.new_page(viewport={"width": 1680, "height": 1000})
        page.on("dialog", lambda dl: dl.accept("tester"))
        page.goto(f"file://{(work / 'index.html').resolve()}")
        page.wait_for_selector(".rowctl")

        # Walk every window IN THE ORDER THE RATER SEES IT (shuffled), clicking `bad` on the cells we
        # injected. We locate the window by what the page reports, never by display position.
        seen = set()
        for _ in range(len(windows)):
            # the header is deliberately opaque to raters, so read the true window id from JS state
            shown = int(page.evaluate("() => WINDOWS[order[pos]].window"))
            seen.add(shown)
            for ch in CHS:
                if (shown, ch) in truth:
                    page.evaluate(
                        """([ch]) => {
                          const rc = [...document.querySelectorAll('.rowctl')]
                            .find(r => r.querySelector('.chn').title === ch);
                          [...rc.querySelectorAll('.opt')].find(o => o.dataset.v === 'bad').click();
                        }""", [ch])
            page.click("#next")
            page.wait_for_timeout(60)

        assert seen == set(windows), "the page did not show every window exactly once"

        with page.expect_download() as dl:
            page.click("#expBtn")
        out = tmp_path / "labels.csv"
        dl.value.save_as(out)
        br.close()

    long = ingest({"r1": str(out)})
    cons = consensus(long)

    got = {(int(r.window), r.channel) for r in cons.itertuples() if r.y_true == 1}
    assert got == truth, f"labels moved in the round trip: expected {truth}, got {got}"

    perfect = np.zeros((max(windows) + 1, len(CHS)), dtype=bool)
    for w, ch in truth:
        perfect[w, CHS.index(ch)] = True
    assert score_mask(perfect, CHS, cons, "A")["f1"] == 1.0

    blind = np.zeros_like(perfect)                        # a detector that finds nothing
    assert score_mask(blind, CHS, cons, "A")["recall"] == 0.0


@pytest.mark.slow
@pytest.mark.browser
def test_seen_tracking_and_portable_resume(tmp_path):
    """The CSV is a complete record and the rater never loses their place: every window is exported
    with a `seen` flag (unseen rows blank, so 'judged clean' != 'never seen'); importing on a fresh
    machine restores labels and resumes at the first unseen window."""
    import csv as _csv
    pw = pytest.importorskip("playwright.sync_api")

    R.render_windows(synth(secs=200, seed=5), FS, CHS, tmp_path, [10, 11, 12, 13, 14, 15], "A", dpi=70)
    zip_path = B.build(tmp_path, name="resume")
    work = tmp_path / "unz"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(work)
    url = f"file://{(work / 'index.html').resolve()}"

    with pw.sync_playwright() as p:
        br = p.chromium.launch()

        ctx = br.new_context()
        pg = ctx.new_page()
        pg.on("dialog", lambda dl: dl.accept("rater1"))
        pg.goto(url); pg.wait_for_selector(".rowctl")
        n = pg.evaluate("() => WINDOWS.length")
        for _ in range(2):                                    # see 3 windows (start + 2 Next)
            pg.click("#next")
        pg.evaluate("""() => { const rc=document.querySelectorAll('.rowctl')[0];
          [...rc.querySelectorAll('.opt')].find(o=>o.dataset.v==='bad').click(); }""")
        assert "seen 3/" in pg.inner_text("#seen")
        with pg.expect_download() as dl:
            pg.click("#expBtn")
        out = tmp_path / "partial.csv"
        dl.value.save_as(out)
        ctx.close()

        rows = list(_csv.DictReader(open(out)))
        label_cols = [c for c in rows[0] if c.startswith("label_")]
        assert len(rows) == n, "every window must be in the CSV, not just the touched ones"
        assert "seen" in rows[0] and "display_order" in rows[0]
        seen_rows = [r for r in rows if r["seen"] == "1"]
        unseen_rows = [r for r in rows if r["seen"] == "0"]
        assert len(seen_rows) == 3 and len(unseen_rows) == n - 3
        assert all(all(r[c] == "" for c in label_cols) for r in unseen_rows), "unseen rows must be blank"

        ctx2 = br.new_context()                               # FRESH machine: empty localStorage
        pg2 = ctx2.new_page()
        pg2.on("dialog", lambda dl: dl.accept("rater1"))
        pg2.goto(url); pg2.wait_for_selector(".rowctl")
        assert pg2.inner_text("#prog").startswith("1 /")      # starts at window 1
        pg2.set_input_files("#impFile", str(out))
        pg2.wait_for_timeout(200)
        assert pg2.inner_text("#prog").startswith("4 /"), "import must resume at the first unseen window"
        flagged = pg2.evaluate("""() => Object.values(state).filter(
            l => Object.values(l).some(v => v && v !== 'clean')).length""")
        assert flagged == 1, "the flagged label must survive the import"
        ctx2.close()
        br.close()

    # and the completed CSV scores cleanly: unseen (blank) cells drop out, seen cells form the truth
    long = ingest({"rater1": str(out)})
    cons = consensus(long)
    assert len(cons.window.unique()) == 3, "only the 3 seen windows form the truth set (unseen drop out)"
    assert set(cons.window.unique()) <= {10, 11, 12, 13, 14, 15}


@pytest.mark.slow
@pytest.mark.browser
def test_export_import_export_is_idempotent(tmp_path):
    """Import must not fabricate a judgment. The airtightness fuzz caught that rendering the resume
    window marked it seen + all-clean, so export->import->export drifted (an unreviewed window became
    a 'clean' reviewed one, polluting ground truth). A window is 'seen' only when the rater ENGAGES it.
    """
    import csv as _csv
    pw = pytest.importorskip("playwright.sync_api")

    R.render_windows(synth(secs=200, seed=9), FS, CHS, tmp_path, [10, 11, 12, 13, 14, 15], "A", dpi=70)
    zip_path = B.build(tmp_path, name="idem")
    work = tmp_path / "unz"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(work)
    url = f"file://{(work / 'index.html').resolve()}"

    def export_from(br, do_labels, out_name):
        ctx = br.new_context()
        pg = ctx.new_page()
        pg.on("dialog", lambda dl: dl.accept("rater1"))
        pg.goto(url); pg.wait_for_selector(".rowctl")
        do_labels(pg)
        with pg.expect_download() as dl:
            pg.click("#expBtn")
        p = tmp_path / out_name
        dl.value.save_as(p)
        ctx.close()
        return list(_csv.DictReader(open(p))), p

    def label_some(pg):                                   # engage 3 windows, flag one, then advance off
        pg.click("#next")
        pg.evaluate("""() => { const rc=document.querySelectorAll('.rowctl')[0];
          [...rc.querySelectorAll('.opt')].find(o=>o.dataset.v==='bad').click(); }""")
        pg.click("#next")

    with pw.sync_playwright() as p:
        br = p.chromium.launch()
        rows1, csv1 = export_from(br, label_some, "csv1.csv")
        # fresh context: import csv1, DO NOT engage anything, re-export
        rows2, _ = export_from(br, lambda pg: (pg.set_input_files("#impFile", str(csv1)), pg.wait_for_timeout(200)), "csv2.csv")
        br.close()

    def key(rows):
        return {r["window"]: (r["seen"], tuple(r[c] for c in rows[0] if c.startswith("label_"))) for r in rows}
    k1, k2 = key(rows1), key(rows2)
    seen1 = {w for w, (s, _) in k1.items() if s == "1"}
    seen2 = {w for w, (s, _) in k2.items() if s == "1"}
    assert seen2 == seen1, f"import fabricated seen windows: {seen2 - seen1}"
    assert k2 == k1, "export -> import -> export is not a fixed point (labels/seen drifted)"


# --------------------------------------------------------------------------- window selection

def test_select_random_spans_recording_no_dupes_margins_respected():
    """Ground truth is only representative if windows are drawn across the whole recording, not from a
    couple of autocorrelated stretches. The margin keeps every window's context on screen."""
    n_win = 1440                                             # 7200 s / 5 s
    flank = int(np.ceil(max(R.CONTEXT_S, R.OVERVIEW_S or 0) / R.FRAG_S))
    sel = R.select_random(n_win, 50, seed=0)

    assert len(sel) == 50 == len(set(sel))                  # distinct, right count
    assert sel == sorted(sel)
    assert min(sel) >= flank and max(sel) <= n_win - flank - 1   # context never runs off either end
    assert min(sel) < n_win * 0.25 and max(sel) > n_win * 0.75   # actually spans, not one clump
    assert sel == R.select_random(n_win, 50, seed=0)        # deterministic
    assert sel != R.select_random(n_win, 50, seed=1)        # and the seed matters


def test_select_random_refuses_an_impossible_draw():
    """Asking for more distinct windows than the margin leaves must raise, not silently return fewer:
    a short bundle is a biased bundle."""
    with pytest.raises(ValueError, match="cannot draw"):
        R.select_random(30, 50)


# --------------------------------------------------------------------------- render via the loader

def test_render_lro_uses_absolute_windows_and_correct_units(tmp_path):
    """Two traps render_lro must not fall into:

    - get_fragment_np gives (n_samples, n_channels) in uV; render_windows wants (n_channels,
      n_samples) in volts. A missed transpose or 1e-6 makes every y-scale nonsense.
    - only a span is loaded per stretch, but the window numbers the rater sees, and the scorer keys
      on, must stay absolute.
    """
    mne_ = pytest.importorskip("mne")
    pytest.importorskip("spikeinterface")
    from datetime import datetime

    from neurodent.loading.long_recording_organizer import LongRecordingOrganizer

    fs, secs = 250, 400
    t = np.arange(secs * fs) / fs
    data_v = np.stack([100e-6 * np.sin(2 * np.pi * 7 * t)] * 4)          # 100 µV, held in volts
    info = mne_.create_info([f"ch{i}" for i in range(4)], fs, ch_types="eeg")
    src = tmp_path / "src_raw.fif"
    mne_.io.RawArray(data_v, info, verbose=False).save(src, overwrite=True, verbose=False)

    lro = LongRecordingOrganizer(
        src, mode="mne", extract_func=mne_.io.read_raw_fif,
        intermediate="bin", intermediate_dir=str(tmp_path / "im"),
        manual_datetimes=datetime(2024, 1, 1), cache_policy="force_regenerate",
    )

    windows = [30, 31, 32]                                   # deep in the recording, so sample0 != 0
    rows, _ = R.render_lro(lro, tmp_path / "out", windows, "A", dpi=60)

    assert [r["window"] for r in rows] == windows            # absolute, not 0,1,2
    got_t = [r["t_start_s"] for r in rows]
    assert got_t == [pytest.approx(w * R.FRAG_S) for w in windows], (
        f"window times are span-relative, not absolute: {got_t}")

    # The y-scale must reflect a ~100 µV signal. If the µV->V conversion were skipped the scale would
    # be ~1e6 times larger; if applied twice, ~1e6 times smaller.
    from neurodent.analysis.long_recording_analyzer import LongRecordingAnalyzer

    an = LongRecordingAnalyzer(lro, fragment_len_s=R.FRAG_S, apply_notch_filter=True)
    ylim, _ = R.lro_scales(an)
    assert 50 < float(np.median(ylim)) < 500, f"y-scale is not µV-plausible: {ylim}"
