#!/usr/bin/env python3
"""Inspect raw EDF + RHD recording traces for channel-health assessment.

For every recording under ``DATA_ROOT``, plot a 60s window (starting 60s
into the file to skip onset transients), apply the pipeline's 60 Hz notch
(``iirnotch(60, 30, fs)``, matches ``core/analyze_frag.py``), and save:

  edf_examination/<folder>/<file-stem>.png           -- one PNG per source file
  edf_examination/<folder>/_combined_<marker>.png    -- group of files, joined

The "marker" groups files within a folder by their per-recording session token:
  - EDF: ``_``, ``_1_``, ``_2_`` (1017-style) or `` ``, ``-``, `` 1 ``, `` 2 ``
    (1199-style) — the substring right before ``SelectionN`` / ``SectionN``.
  - RHD: 6-digit ``YYMMDD`` date token at end of stem (e.g. ``200804``).

For RHD, only the *first* file per (folder, date) is plotted — each .rhd is
already 1-30 min of data, and folders contain hundreds-to-thousands.

Run on HPC:
    sbatch --job-name=inspect_traces --cpus-per-task=2 --mem=8G --time=1:00:00 \\
      --wrap="cd /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_Joseph_devtree \\
              && uv run python scripts/inspect_recording_traces.py"
"""
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import spikeinterface.extractors as se
from mne import create_info
from mne.io import BaseRaw, RawArray, read_raw_edf
from scipy.signal import filtfilt, iirnotch

DATA_ROOT = "/mnt/isilon/marsh_single_unit/PythonEEG Data/Arx Rosa"
SAVE_DIR = Path("/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_Joseph_devtree/edf_examination")
# Matches the pipeline's notch in core/analyze_frag.py: iirnotch(60, 30, fs)
NOTCH_FREQ = 60.0
NOTCH_Q = 30.0
FILT_BUFFER_S = 5.0
PLOT_KWARGS = dict(scalings={"eeg": 1000e-6}, clipping=0.5, show=False, show_scrollbars=False)

# EDF marker: token before SelectionN / SectionN ('_', '_1_', ' ', ' 1 ', '-').
_SECTION_SPLIT = re.compile(r"(?:Selection|Section)\d")
_EDF_MARKER = re.compile(r"[_\-\s]+(?:\d+[_\-\s]+)?$")
# RHD marker: YYMMDD date token, followed by HHMMSS time at end of stem.
_RHD_DATE = re.compile(r"_(\d{6})_\d{6}$")


def session_marker(stem: str) -> str:
    """Extract the day/group token from an EDF or RHD stem."""
    parts = _SECTION_SPLIT.split(stem, maxsplit=1)
    if len(parts) > 1:
        m = _EDF_MARKER.search(parts[0])
        return m.group(0) if m else "?"
    m = _RHD_DATE.search(stem)
    return m.group(1) if m else "?"


def short_label(stem: str) -> str:
    """Compact per-file label ('Selection1', 'Section3', 't084859')."""
    m = re.search(r"(?:Selection|Section)\d+", stem)
    if m:
        return m.group(0)
    m = re.search(r"_\d{6}_(\d{6})$", stem)
    if m:
        return f"t{m.group(1)}"
    return stem[-15:]


def safe_marker(marker: str) -> str:
    """Filesystem-safe rendering of a session marker (spaces → underscores)."""
    return marker.replace(" ", "_").replace("/", "_") or "default"


def _plot_window(total_s: float) -> tuple[float, float, float, float]:
    """Return (plot_start, duration, left_buf, right_buf) given total recording length."""
    plot_start = 60.0 if total_s >= 120.0 else 0.0
    duration = min(60.0, total_s - plot_start)
    left_buf = min(FILT_BUFFER_S, plot_start)
    right_buf = min(FILT_BUFFER_S, total_s - (plot_start + duration))
    return plot_start, duration, left_buf, right_buf


def _apply_notch(data: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase 60 Hz notch (Q=30) along the time axis, preserving dtype."""
    b, a = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=fs)
    return filtfilt(b, a, data, axis=-1).astype(data.dtype, copy=False)


def _load_edf_window(file: str) -> tuple[BaseRaw, float]:
    raw = read_raw_edf(file, preload=False, verbose="error")
    total_s = raw.n_times / raw.info["sfreq"]
    plot_start, duration, left_buf, right_buf = _plot_window(total_s)
    raw.crop(tmin=plot_start - left_buf, tmax=plot_start + duration + right_buf)
    raw.load_data(verbose="error")
    raw.apply_function(lambda x: _apply_notch(x, raw.info["sfreq"]), verbose="error")
    raw.crop(tmin=left_buf, tmax=left_buf + duration)
    return raw, duration


def _load_rhd_window(file: str) -> tuple[BaseRaw, float]:
    """Read 60s window from an Intan RHD via SpikeInterface, return MNE RawArray."""
    rec = se.read_intan(file, stream_id="0", ignore_integrity_checks=True)
    fs = rec.get_sampling_frequency()
    total_s = rec.get_num_samples() / fs
    plot_start, duration, left_buf, right_buf = _plot_window(total_s)

    start_frame = int(round((plot_start - left_buf) * fs))
    end_frame = int(round((plot_start + duration + right_buf) * fs))
    rec_slice = rec.frame_slice(start_frame, end_frame)
    # return_scaled=True → µV per pipeline convention.
    data_uv = rec_slice.get_traces(return_scaled=True).T.astype("float32", copy=False)
    data_uv = _apply_notch(data_uv, fs)

    buf_samples = int(round(left_buf * fs))
    plot_samples = int(round(duration * fs))
    data_uv = data_uv[:, buf_samples:buf_samples + plot_samples]

    info = create_info(
        ch_names=[str(c) for c in rec_slice.channel_ids],
        sfreq=fs,
        ch_types="eeg",
    )
    return RawArray(data_uv * 1e-6, info, verbose="error"), duration  # µV → V for MNE


def load_filtered_window(file: str) -> tuple[BaseRaw, float]:
    """Return (Raw, duration) for the 60s plot window, notch-filtered."""
    suffix = Path(file).suffix.lower()
    if suffix == ".edf":
        return _load_edf_window(file)
    if suffix == ".rhd":
        return _load_rhd_window(file)
    raise ValueError(f"Unsupported file extension: {file}")


def save_single_file_plot(raw: BaseRaw, file: str, duration: float) -> None:
    fig = raw.plot(
        start=0,
        duration=duration,
        n_channels=raw.info["nchan"],
        **PLOT_KWARGS,
    )
    # Scale height to channel count so RHD (128 channels) doesn't squash into a tiny strip.
    fig.set_size_inches(15, max(8, raw.info["nchan"] * 0.3))
    fig.suptitle(
        f"{Path(file).stem}  |  scale: 2000 µV / channel  |  60 Hz notch (Q=30)",
        fontsize=10,
        y=0.995,
    )
    session_dir = SAVE_DIR / Path(file).parent.name
    session_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(session_dir / f"{Path(file).stem}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_session_combined_plot(folder, marker, raws, file_list, durations):
    """Concatenate the per-file 60s windows and plot them side-by-side."""
    # concatenate_raws mutates the first arg; copy defensively.
    combined = mne.concatenate_raws([r.copy() for r in raws])
    n_channels = combined.info["nchan"]
    total_s = sum(durations)

    fig = combined.plot(
        start=0,
        duration=total_s,
        n_channels=n_channels,
        **PLOT_KWARGS,
    )
    # 4 inches per file horizontally; tall enough for all channels.
    fig.set_size_inches(max(20, len(file_list) * 4), max(10, n_channels * 0.3))

    # File boundaries (vertical red lines) + short labels at the top of the plot.
    cumulative = 0.0
    file_axis = fig.axes[0]
    for i, (file, d) in enumerate(zip(file_list, durations)):
        if i > 0:
            for ax in fig.axes[:n_channels]:
                ax.axvline(cumulative, color="red", linewidth=0.5, alpha=0.5)
        file_axis.text(
            cumulative + d / 2,
            1.05,
            short_label(Path(file).stem),
            transform=file_axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )
        cumulative += d

    fig.suptitle(
        f"{folder}  |  marker={marker!r}  |  scale: 2000 µV/channel  |  "
        f"60 Hz notch  |  {len(file_list)} files × 60s",
        fontsize=11,
        y=0.998,
    )
    out_dir = SAVE_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"_combined_{safe_marker(marker)}.png", dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Restrict glob to one level deep so orphan files in DATA_ROOT itself are skipped.
    edf_files = sorted(glob(f"{DATA_ROOT}/*/*.EDF"))
    rhd_files = sorted(glob(f"{DATA_ROOT}/*/*.rhd"))
    print(f"Found {len(edf_files)} EDF + {len(rhd_files)} RHD files", flush=True)

    # Group by (folder, marker).
    groups = defaultdict(list)
    for f in edf_files + rhd_files:
        folder = Path(f).parent.name
        marker = session_marker(Path(f).stem)
        groups[(folder, marker)].append(f)

    # For RHD groups, keep only the first file per (folder, date): each .rhd is
    # already 1-30 min of recording, and folders contain hundreds-to-thousands.
    for key, file_list in list(groups.items()):
        if all(Path(f).suffix.lower() == ".rhd" for f in file_list):
            groups[key] = [sorted(file_list)[0]]

    for (folder, marker), file_list in sorted(groups.items()):
        file_list = sorted(file_list)
        print(f"\n=== {folder} | marker={marker!r} ({len(file_list)} files) ===", flush=True)

        group_raws = []
        durations = []
        for i, file in enumerate(file_list, 1):
            print(f"  [{i}/{len(file_list)}] {Path(file).name}", flush=True)
            raw, duration = load_filtered_window(file)
            save_single_file_plot(raw, file, duration)
            group_raws.append(raw)
            durations.append(duration)

        if len(group_raws) > 1:
            print(f"  -> combined plot ({len(group_raws)} files)", flush=True)
            save_session_combined_plot(folder, marker, group_raws, file_list, durations)
        del group_raws


if __name__ == "__main__":
    main()
