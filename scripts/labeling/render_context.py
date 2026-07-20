"""Render context figures for an artifact-labeling round.

One figure per WINDOW showing ALL channels. Each channel gets its own subplot row with its own FIXED
y-scale (held constant across windows, and printed on the row, so raters calibrate "normal" amplitude
and artifacts clip visibly), sharing a time axis, with the target window shaded straight down the
column and +/- CONTEXT_S of surrounding context, plus a PSD panel of the target beside each trace.

Above them sits one compact all-channel overview strip spanning +/- OVERVIEW_S (set OVERVIEW_S = None
to drop it). The zoom panels show morphology; the strip shows whether an event is evolving, which is
the only reason context exists at all.

The rater labels each channel for that window (clean / bad / event / unsure), so one image and one
manifest row cover all channels.

Call render_windows once per recording with append=True to accumulate many animals into one bundle.
"""
import csv
import hashlib
import json
import re
import string
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch

from neurodent.results.scoring import LABEL_COL_PREFIX

FRAG_S = 5.0
CONTEXT_S = 15.0        # flank on each side of the target window (the zoom panels)
OVERVIEW_S = 60.0       # flank for the compact all-channel overview strip; None to disable
FMIN, FMAX = 1.0, 200.0
YLIM_FLOOR_UV = 10.0        # so a dead channel gets a panel, not a zero-height axis
PSD_DYNAMIC_RANGE_DB = 80   # so a dead channel cannot stretch the shared axis into uselessness
AMP_FLAG_MULT = 4.0         # flag a channel (red label) when its scale exceeds this x the median channel

MANIFEST = "manifest.csv"
GEOMETRY = "geometry.json"      # where each channel's row landed, so the rater page can put that
                                # channel's buttons beside it instead of in a table to cross-reference


def slug(s):
    """Filesystem-safe recording id, used to keep image names unique ACROSS recordings."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")


def trace_ylim(data):
    """(C,) fixed per-channel uV half-scale for the trace rows.

    Floored, because a dead channel's p99.5 is 0 and would otherwise get a zero-height axis. It is a
    `bad` category the rater must be able to see, so it has to render as a flat line, not degenerate.
    """
    return np.maximum(1.2 * np.percentile(np.abs(data), 99.5, axis=1) * 1e6, YLIM_FLOOR_UV)


def psd_limits(data, fs, win):
    """(lo, hi) dB for the one shared PSD axis across all channels.

    Shared because the per-channel time scales necessarily hide the ~20x differences in channel
    baseline, so this is the only place a uniformly-huge channel is visible.

    Limits come from the LIVE channels, with the dynamic range capped: a dead channel sits at -200 dB
    and a naive min()..max() would squash the live ones into a few percent of the axis, ruining the
    panel for every window. A dead channel instead clips off the bottom, which is the right signal.
    """
    f, p = welch(data, fs=fs, nperseg=min(win, 2000), axis=1)
    m = (f >= FMIN) & (f <= FMAX)
    pdb = 10 * np.log10(p[:, m] + 1e-20)
    live = np.ptp(data, axis=1) > 0
    ref = pdb[live] if live.any() else pdb
    hi = float(ref.max())
    lo = max(float(ref.min()), hi - PSD_DYNAMIC_RANGE_DB)
    return lo - 10, hi + 10


def load_fif(fif, crop_s=7200, notch=60.0):
    raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
    raw.crop(0, min(crop_s, raw.times[-1])).load_data(verbose=False)
    if notch:
        raw.notch_filter(notch, verbose=False)
    return raw


def select_random(n_windows, n_select, seed=0):
    """Random, detector-independent, distinct single windows drawn across the whole recording.

    Adjacent 5 s windows are autocorrelated, so contiguous stretches oversample a few moments;
    independent draws across the full span give a representative ground-truth sample. A context-flank
    margin is reserved at both ends so every drawn window keeps its full +/- context on screen.
    """
    rng = np.random.RandomState(seed)
    flank = int(np.ceil(max(CONTEXT_S, OVERVIEW_S or 0) / FRAG_S))   # widest context reaches this far
    pool = range(flank, n_windows - flank)
    if len(pool) < n_select:
        raise ValueError(
            f"cannot draw {n_select} distinct windows: only {len(pool)} eligible "
            f"({n_windows} windows, {flank} reserved as flank at each end)"
        )
    return sorted(int(w) for w in rng.choice(pool, size=n_select, replace=False))


def render_windows(data, fs, ch_names, out, windows, recording, dpi=110, append=False,
                   sample0=0, ylim=None, psd_lim=None, all_channels=None, blind_seed=None):
    """data: (C, n_samples) volts for ONE recording, in TRUE channel order. windows: flat list.

    Writes one PNG per window into out/images/ and appends rows to out/manifest.csv.
    Image names are prefixed with the recording slug so multiple recordings never collide.

    Args:
        sample0: absolute sample index of ``data[:, 0]``, non-zero when ``data`` is only a span of the
            recording. Window numbers stay absolute: the number a rater sees and the scorer keys on
            must be the true one, not an index into whatever slice was loaded.
        ylim, psd_lim: OPTIONAL fixed scales. When omitted (the default) the amplitude scale is computed
            PER WINDOW from that window's visible +/- CONTEXT_S zoom context, so a blow-out ELSEWHERE in
            the recording cannot crush an otherwise-normal window. Pass explicit scales only to force one
            fixed scale across windows.
        all_channels: the manifest label-column set (a superset of this recording's display channels).
            Lets ONE mixed bundle hold recordings with DIFFERENT channel counts; a shorter recording
            leaves the extra slots blank. Defaults to this recording's own display channels.
        blind_seed: when not None, EACH window is INDEPENDENTLY channel-blinded — its traces are permuted
            by a seed derived from ``(blind_seed, recording, window)``, rows are shown under neutral slots
            (``Ch A`` ...), and a per-``(recording, window, slot)`` keymap is returned. ``ch_names`` is
            then the TRUE channel names. When None, channels render in order (no keymap).

    Returns:
        tuple[list[dict], list[dict]]: ``(manifest_rows, keymap)``. ``keymap`` is empty unless blinding;
        each entry is ``{"recording", "window", "slot", "channel"}``.
    """
    out = Path(out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    C = len(ch_names)
    win = int(FRAG_S * fs)
    rec = slug(recording)

    # Identity vs display: when blinding, the passed ``ch_names`` are the TRUE channels and rows are
    # shown under neutral slots; the true<->slot mapping is recorded PER WINDOW in the keymap.
    blinding = blind_seed is not None
    true_names = list(ch_names)
    display = neutral_labels(C) if blinding else list(ch_names)
    fixed_ylim = None if ylim is None else np.asarray(ylim)
    fixed_psd = None if psd_lim is None else tuple(psd_lim)

    # Manifest label columns use the DISPLAY names. `all_channels` (a superset) lets a mixed bundle share
    # ONE column set across recordings with different channel counts; a shorter recording leaves the
    # extra slots blank. Defaults to this recording's own display channels.
    label_channels = list(all_channels) if all_channels is not None else list(display)
    missing = [c for c in display if c not in label_channels]
    if missing:
        raise ValueError(f"render_windows: channels {missing} of {recording!r} not in all_channels={label_channels}")

    # Appending a recording with different columns would write its labels under the first recording's
    # column names: labels attributed to the wrong channels, across animals, silently.
    fields = ["image", "recording", "window", "t_start_s"] + [f"{LABEL_COL_PREFIX}{c}" for c in label_channels]
    path = out / MANIFEST
    if append and path.exists():
        with open(path) as fh:
            existing = next(csv.reader(fh), [])
        if existing and existing != fields:
            raise ValueError(
                f"cannot append {recording!r}: its channels do not match the existing manifest.\n"
                f"  manifest: {[c[len(LABEL_COL_PREFIX):] for c in existing if c.startswith(LABEL_COL_PREFIX)]}\n"
                f"  this one: {list(display)}\n"
                "Bundle recordings with identical channel sets, or render them separately.")

    keymap = []
    rows, geom_rows = [], None
    for w in windows:
        # `w` is absolute; `data` may only be a span starting at sample0.
        t0, t1 = w * win - sample0, (w + 1) * win - sample0
        if t0 < 0 or t1 > data.shape[1]:
            raise IndexError(
                f"window {w} (samples {w*win}..{(w+1)*win}) is outside the loaded span "
                f"[{sample0}, {sample0 + data.shape[1]}). Load a span that covers it.")
        c0 = max(0, t0 - int(CONTEXT_S * fs))
        c1 = min(data.shape[1], t1 + int(CONTEXT_S * fs))
        ts0, ts1 = (t0 + sample0) / fs, (t1 + sample0) / fs   # absolute bounds -> the manifest only
        # The x-axis is RELATIVE to the window start (0 = window start), so a rater never sees the
        # absolute time-of-day (which could bias) and the figure stays blind. Identity lives in the CSV.
        tt = (np.arange(c0, c1) + sample0) / fs - ts0
        wlo, whi = 0.0, ts1 - ts0                             # the labelled window, in relative seconds

        # Per-WINDOW channel blinding: a fresh permutation per (recording, window), so no display slot
        # can be learned to a true channel across a recording's windows. Row i shows true_names[perm[i]].
        perm = _channel_perm(f"{blind_seed}:{recording}:{w}", C) if blinding else np.arange(C)
        wdata = data[perm]                                    # (C, N): row i is channel perm[i]
        if blinding:
            keymap += [{"recording": recording, "window": int(w), "slot": display[i],
                        "channel": true_names[int(perm[i])]} for i in range(C)]

        # Per-window amplitude scale from the VISIBLE +/- CONTEXT_S zoom context (not the whole
        # recording), reordered to the blinded rows: a distant blow-out no longer crushes this window,
        # while an in-window hot channel still scales high and still trips the red flag below.
        ylim = (fixed_ylim if fixed_ylim is not None else trace_ylim(data[:, c0:c1]))[perm]
        psd_lim = fixed_psd if fixed_psd is not None else psd_limits(data[:, c0:c1], fs, win)

        # 16:9 so the figure fits one screen; a rater who scrolls every window is spending attention
        # on the scrollbar instead of the trace.
        n_extra = 1 if OVERVIEW_S else 0
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(C + n_extra, 2, width_ratios=[5, 1],
                              height_ratios=[2.5] * n_extra + [1] * C)

        if OVERVIEW_S:
            # Evolution strip: per-channel smoothed ENVELOPE, not the raw trace. At this time scale a
            # raw trace is an unreadable smear; the envelope shows whether something builds up,
            # sustains and stops (an event) or is a one-off spike (an artifact). Morphology and
            # rhythmicity are the zoom panels' job below.
            o0 = max(0, t0 - int(OVERVIEW_S * fs))
            o1 = min(data.shape[1], t1 + int(OVERVIEW_S * fs))
            k = max(1, int(0.25 * fs))                       # 250 ms smoothing kernel
            ker = np.ones(k) / k
            ax = fig.add_subplot(gs[0, :])
            for ci in range(C):
                env = np.convolve(np.abs(wdata[ci, o0:o1]) * 1e6, ker, mode="same") / ylim[ci]
                step = max(1, len(env) // 3000)
                ax.fill_between((np.arange(o0, o1)[::step] + sample0) / fs - ts0, (C - 1 - ci),
                                (C - 1 - ci) + 3.0 * env[::step], lw=0, color="0.35")
            ax.axvspan(wlo, whi, color="orange", alpha=0.35)
            ax.set_yticks(np.arange(C) + 0.25)
            ax.set_yticklabels(list(reversed(display)), fontsize=7)
            ax.set_ylim(-0.2, C + 0.2)
            ax.set_xlim((o0 + sample0) / fs - ts0, (o1 + sample0) / fs - ts0)
            ax.tick_params(labelsize=7, pad=1)
            ax.set_title(f"evolution  +/-{OVERVIEW_S:.0f}s   envelope per channel, scaled to that "
                         f"channel   (shaded = the window you are labelling)", fontsize=8)

        med_scale = np.median(ylim)                          # a hot channel's per-channel scale (which
        hot = ylim > AMP_FLAG_MULT * med_scale               # hides its amplitude) shows up red-flagged
        trace_axes = []
        for ci, ch in enumerate(display):
            a0 = fig.add_subplot(gs[ci + n_extra, 0])
            a1 = fig.add_subplot(gs[ci + n_extra, 1])
            trace_axes.append((ch, a0))
            a0.plot(tt, wdata[ci, c0:c1] * 1e6, lw=0.3, color="0.15")
            a0.axvspan(wlo, whi, color="orange", alpha=0.25)
            a0.set_ylim(-ylim[ci], ylim[ci])
            lbl = f"{ch}\n+/-{ylim[ci]:.0f}uV" + (f"  {ylim[ci]/med_scale:.0f}x" if hot[ci] else "")
            a0.set_ylabel(lbl, fontsize=7, rotation=0, ha="right", va="center",
                          color="red" if hot[ci] else "black", fontweight="bold" if hot[ci] else "normal")
            a0.tick_params(labelsize=7, pad=1)
            if ci < C - 1:
                a0.set_xticklabels([])
            f, p = welch(wdata[ci, t0:t1], fs=fs, nperseg=min(win, 2000))
            m = (f >= FMIN) & (f <= FMAX)
            a1.semilogx(f[m], 10 * np.log10(p[m] + 1e-20), lw=0.9, color="C0")
            a1.set_ylim(*psd_lim)                     # SHARED across channels: height = absolute power
            a1.tick_params(labelsize=7, pad=1)
            if ci < C - 1:
                a1.set_xticklabels([])
            if ci == 0:
                a1.set_title("PSD of target (dB, shared axis)", fontsize=8)
            if ci == C - 1:
                a0.set_xlabel("s", fontsize=8)
                a1.set_xlabel("Hz", fontsize=8)

        # No recording / window / absolute time in the title: raters label blind (identity is in the CSV).
        fig.suptitle("shaded = the window to label   •   time scales are PER-CHANNEL; the PSD axis is "
                     "shared   •   a red channel label is unusually high-amplitude", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=0.15)

        # After tight_layout, so these are final. Fractions from the top, which is what CSS wants.
        # One geometry per recording is only valid if every window lays out identically; if not, the
        # rater's buttons drift onto the wrong trace with no visible sign. So check, do not assume.
        this_geom = [{"channel": ch,
                      "top": float(1.0 - a.get_position().y1),
                      "bottom": float(1.0 - a.get_position().y0)}
                     for ch, a in trace_axes]
        if geom_rows is None:
            geom_rows = this_geom
        else:
            drift = max(abs(a[k] - b[k]) for a, b in zip(this_geom, geom_rows) for k in ("top", "bottom"))
            if drift > 1e-6:
                raise AssertionError(
                    f"{recording} window {w}: channel rows moved by {drift:.2e} of image height "
                    "relative to the first window. One geometry per recording is then wrong and the "
                    "rater's buttons would not line up with the traces.")

        name = f"{rec}_w{w:06d}.png"                       # recording-prefixed: no cross-animal collision
        fig.savefig(img_dir / name, dpi=dpi)
        # Shared-scale twin for the rater's Space toggle: same figure, one global y-scale (the loudest
        # channel's), so a uniformly-hot channel is obvious. Geometry is unchanged (only y-limits move).
        sh = float(np.max(ylim))
        for ch, a0 in trace_axes:
            a0.set_ylim(-sh, sh)
            a0.set_ylabel(f"{ch}\n+/-{sh:.0f}uV", fontsize=7, rotation=0, ha="right", va="center")
        fig.suptitle("shaded = the window to label   •   time scales are SHARED across channels (one "
                     "global amplitude)   •   the PSD axis is shared", fontsize=10)
        fig.savefig(img_dir / name.replace(".png", "__shared.png"), dpi=dpi)
        plt.close(fig)

        row = {"image": name, "recording": recording, "window": int(w), "t_start_s": round(ts0, 1)}
        row.update({f"{LABEL_COL_PREFIX}{c}": "" for c in label_channels})
        rows.append(row)

    write_header = not (append and path.exists())
    with open(path, "a" if append else "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            wtr.writeheader()
        wtr.writerows(rows)

    gpath = out / GEOMETRY
    geom = json.loads(gpath.read_text()) if (append and gpath.exists()) else {}
    geom[recording] = {"rows": geom_rows}
    gpath.write_text(json.dumps(geom, indent=1))
    return rows, keymap


def _contiguous_runs(windows):
    """[1,2,3, 9,10] -> [[1,2,3], [9,10]]. Each run is loaded as one span."""
    runs = []
    for w in sorted(set(int(x) for x in windows)):
        if runs and w == runs[-1][-1] + 1:
            runs[-1].append(w)
        else:
            runs.append([w])
    return runs


def lro_scales(lan, n_probe=24):
    """Per-channel y-scale and shared PSD axis, estimated across the whole recording.

    Estimated once, not per span: the fixed scale only works as a calibration aid if it does not shift
    between stretches.
    """
    idxs = np.unique(np.linspace(0, lan.n_fragments - 1, n_probe).astype(int))
    probe = np.concatenate([lan.get_fragment_np(int(i)) for i in idxs], axis=0).T * 1e-6  # (C, N) volts
    fs = int(lan.f_s)
    return trace_ylim(probe), psd_limits(probe, fs, int(FRAG_S * fs))


def neutral_labels(n):
    """Neutral channel display names (``Ch A``, ``Ch B``, ...), which blind anatomy.

    Raters must not be able to read a channel's anatomy off its label (a montage
    aid becomes a bias), so trace rows are named by opaque slot. Falls back to
    two-letter slots (``Ch AA``) past 26 channels.
    """
    out = []
    for i in range(n):
        s, j = "", i
        while True:
            s = string.ascii_uppercase[j % 26] + s
            j = j // 26 - 1
            if j < 0:
                break
        out.append(f"Ch {s}")
    return out


def _channel_perm(seed_key, n):
    """Deterministic permutation of ``n`` channel rows from a hashed ``seed_key``.

    Used both for the (legacy) per-recording :func:`blind_channels` and — keyed on
    ``f"{blind_seed}:{recording}:{window}"`` — for the per-WINDOW blinding in
    :func:`render_windows`, so no display slot can be learned to a true channel across a
    recording's windows.
    """
    digest = hashlib.sha256(str(seed_key).encode()).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "big"))
    return rng.permutation(n)


def blind_channels(channel_names, seed_key):
    """Deterministic channel blinding for a single ``seed_key`` (one permutation).

    Draws a permutation of the channel axis from a seed derived from ``seed_key`` and names the
    permuted slots with :func:`neutral_labels`. (The cohort bundler now blinds per WINDOW inside
    :func:`render_windows`; this remains the single-permutation primitive/utility.)

    Args:
        channel_names (list[str]): True channel names, in recording order.
        seed_key: Any value; hashed to seed the permutation deterministically.

    Returns:
        tuple[np.ndarray, list[str], list[dict]]: ``(perm, display_names, keymap)``
        where ``perm`` reorders the channel axis (row ``i`` shows ``channel_names[perm[i]]``),
        ``display_names`` are the neutral labels, and ``keymap`` is the de-scramble record —
        one ``{"slot": <neutral>, "channel": <true name>}`` per slot.
    """
    n = len(channel_names)
    perm = _channel_perm(seed_key, n)
    names = neutral_labels(n)
    keymap = [{"slot": names[i], "channel": channel_names[int(perm[i])]} for i in range(n)]
    return perm, names, keymap


def render_lro(lrec, out, windows=None, recording=None, append=False, dpi=110, notch=True,
               n_select=None, seed=0, blind_seed=None, all_channels=None, true_names=None):
    """Render from a LongRecording, i.e. any format the loader reads (rhd/EDF/NWB/bin).

    The loader knows each format's import parameters, so this is how a real campaign renders; nothing
    needs converting to fif. Pass explicit `windows` (e.g. a cross-animal draw) or let it draw random
    windows across this recording.

    Units: get_fragment_np gives (n_samples, n_channels) in uV, render_windows wants
    (n_channels, n_samples) in volts. Notch is applied per fragment by LongRecordingAnalyzer, so the
    rater and the detector see the same signal as the WAR features.

    Channel blinding: pass ``blind_seed`` to blind each window INDEPENDENTLY (see
    :func:`render_windows`) — traces permuted per ``(recording, window)``, rows shown under neutral
    slots so anatomy cannot bias the rater, and a per-``(recording, window, slot)`` keymap returned for
    the caller to unblind the labels. Default ``None`` renders true channels in order. Amplitude scale is
    computed per window over its visible +/- CONTEXT_S context.

    Returns:
        tuple[list[dict], list[dict]]: ``(manifest_rows, keymap)`` (keymap empty when not blinding).
    """
    from neurodent.analysis.long_recording_analyzer import LongRecordingAnalyzer

    if windows is None and n_select is None:
        raise ValueError("pass windows=[...] or n_select=<how many windows to draw at random>")
    lan = LongRecordingAnalyzer(lrec, fragment_len_s=FRAG_S, apply_notch_filter=notch)
    fs = int(lan.f_s)
    win = int(FRAG_S * fs)
    if windows is None:
        windows = select_random(lan.n_fragments, n_select, seed=seed)
    # TRUE channel names for the keymap/labels. Callers pass canonical abbrevs (resolve_channels) so the
    # de-scramble key stores anatomy, not raw ids; default to the recording's own channel names.
    ch_names = list(true_names) if true_names is not None else list(lan.channel_names)
    if len(ch_names) != len(lan.channel_names):
        raise ValueError(f"true_names has {len(ch_names)} entries, recording has {len(lan.channel_names)} channels")

    # The context flanks reach beyond the target window, so one fragment is not enough.
    flank_frags = int(np.ceil(max(CONTEXT_S, OVERVIEW_S or 0) / FRAG_S))

    rows, keymap = [], []
    for run in _contiguous_runs(windows):
        f0 = max(0, run[0] - flank_frags)
        f1 = min(lan.n_fragments, run[-1] + 1 + flank_frags)
        span = np.concatenate([lan.get_fragment_np(int(i)) for i in range(f0, f1)], axis=0)
        data = span.T * 1e-6                                    # (C, N) µV -> volts, TRUE channel order
        r, k = render_windows(data, fs, ch_names, out, run, recording,
                              dpi=dpi, append=append or bool(rows), sample0=f0 * win,
                              all_channels=all_channels, blind_seed=blind_seed)
        rows += r
        keymap += k
    return rows, keymap


def render_fif(fif, out, windows=None, n_select=None,
               crop_s=7200, seed=0, append=False, recording=None):
    """Render one .fif. Pass explicit `windows` (your own selection) or `n_select` to draw at random."""
    if windows is None and n_select is None:
        raise ValueError("pass windows=[...] or n_select=<how many windows to draw at random>")
    raw = load_fif(fif, crop_s=crop_s)
    fs = int(raw.info["sfreq"])
    data = raw.get_data()
    if windows is None:
        n_win = int(data.shape[1] // (FRAG_S * fs))
        windows = select_random(n_win, n_select, seed=seed)
    rec = recording or Path(fif).stem
    rows, _ = render_windows(data, fs, raw.ch_names, out, windows, rec, append=append)
    return rows


if __name__ == "__main__":
    # 16 is just a quick local pilot count; a real round sizes n_select by how many labelled cells
    # the prevalence/agreement estimate needs, so it is chosen per campaign, not defaulted in the API.
    rows = render_fif("443-wt-443-wt-jul-18-2012-raw-1.fif", "results/labeling_pilot", n_select=16)
    print(f"rendered {len(rows)} window images to results/labeling_pilot/images/ ({MANIFEST})")
