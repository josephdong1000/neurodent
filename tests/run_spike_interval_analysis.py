#!/usr/bin/env python3
"""
Spike Interval Analysis Script

Processes all .fif recordings under the fdsars directory, computes spike
statistics (count, rate, median ISI) per channel, generates ISI histogram
plots, and saves a summary CSV.

Converted from test_Yong_spikeInterval.ipynb to avoid Jupyter OOM crashes.

Usage:
    python run_spike_interval_analysis.py
"""

import gc
import json
import sys

import matplotlib
matplotlib.use("Agg")  # Headless backend — no GUI, lower memory footprint

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root so neurodent is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neurodent import core

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of *values* with corresponding *weights*.

    NaN values are dropped before computation. Returns NaN if no valid data.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & ~np.isnan(weights)
    v, w = values[mask], weights[mask]
    if len(v) == 0:
        return np.nan
    if len(v) == 1:
        return float(v[0])
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumw = np.cumsum(w)
    half = cumw[-1] / 2.0
    idx = np.searchsorted(cumw, half)
    return float(v[min(idx, len(v) - 1)])


def process_one_recording(fif_path: Path, hist_dir: Path) -> tuple[dict, dict] | None:
    """Process a single .fif recording.

    Returns:
        (row_data, isi_arrays) where isi_arrays maps channel abbreviation
        to the raw ISI numpy array for that session, or None on failure.
    """

    # --- Load metadata from companion JSON ---
    json_name = fif_path.name.replace("-raw.fif", ".json")
    json_path = fif_path.with_name(json_name)

    animal_id, full_genotype, animal_day = "Unknown", "Unknown", "Unknown"
    sex, genotype_status = "Unknown", "Unknown"

    if json_path.is_file():
        with open(json_path, "r") as f:
            eeg_metadata = json.load(f)
            animal_id = eeg_metadata.get("animal_id", "Unknown")
            full_genotype = eeg_metadata.get("genotype", "Unknown")
            animal_day = eeg_metadata.get("animal_day", "Unknown")

            if full_genotype != "Unknown" and len(full_genotype) > 1:
                first_char = full_genotype[0].upper()
                if first_char == "M":
                    sex, genotype_status = "Male", full_genotype[1:]
                elif first_char == "F":
                    sex, genotype_status = "Female", full_genotype[1:]
                else:
                    genotype_status = full_genotype
            else:
                genotype_status = full_genotype

    # --- Read raw ---
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
    total_duration = raw.n_times / raw.info["sfreq"]
    all_events, all_event_id = mne.events_from_annotations(raw, verbose=False)
    recording_name = f"{animal_id}_{genotype_status}_{animal_day}"

    # --- Build per-channel stats and histogram subplots ---
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes_flat = axes.flatten()

    row_data = {
        "animal_id": animal_id,
        "sex": sex,
        "genotype": genotype_status,
        "animal_day": animal_day,
        "total_duration_sec": total_duration,
        "total_duration_hours": total_duration / 3600,
    }

    isi_arrays = {}  # ch_abbrev -> raw ISI array for this session

    for ch_idx, ch_name in enumerate(raw.ch_names):
        if ch_idx >= 10:
            break

        label = f"Spike_Ch{ch_idx}"
        ch_name_abbrev = core.parse_chname_to_abbrev(ch_name)
        display_name = f"{ch_name_abbrev} ({ch_idx + 1})"
        ax = axes_flat[ch_idx]

        if label not in all_event_id:
            row_data[f"{ch_name_abbrev}_spike_count"] = 0
            row_data[f"{ch_name_abbrev}_spike_rate"] = 0.0
            row_data[f"{ch_name_abbrev}_spike_rate_per_hour"] = 0.0
            row_data[f"{ch_name_abbrev}_median_isi"] = np.nan
            row_data[f"{ch_name_abbrev}_cv_isi"] = np.nan
            isi_arrays[ch_name_abbrev] = np.array([])
            ax.set_title(f"{display_name}\nNo Spikes")
            continue

        ch_event_id = all_event_id[label]
        ch_events = all_events[all_events[:, 2] == ch_event_id]
        spike_times = ch_events[:, 0] / raw.info["sfreq"]

        spike_count = len(spike_times)
        spike_rate = spike_count / total_duration if total_duration > 0 else 0.0
        spike_rate_per_hour = spike_rate * 3600

        if spike_count > 1:
            isi = np.diff(spike_times)
            isi_arrays[ch_name_abbrev] = isi
            median_isi = np.median(isi)
            mean_isi = np.mean(isi)
            std_isi = np.std(isi)
            cv_isi = std_isi / mean_isi if mean_isi > 0 else np.nan
            bins = np.logspace(
                np.log10(max(isi.min(), 0.001)), np.log10(isi.max()), 50
            )

            ax.hist(isi, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
            ax.set_xscale("log")
            ax.axvline(
                median_isi,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Med: {median_isi:.2f}s",
            )

            ax.set_title(
                f"{display_name}\nCnt: {spike_count}, Rate: {spike_rate_per_hour:.1f} /hr",
                fontsize=10,
            )
            ax.set_xlabel("ISI (s) (Log Scale)", fontsize=8)
            ax.legend(fontsize=7)
            row_data[f"{ch_name_abbrev}_median_isi"] = median_isi
            row_data[f"{ch_name_abbrev}_cv_isi"] = cv_isi
        else:
            ax.set_title(f"{display_name}\n{spike_count} Spike (No ISI)")
            row_data[f"{ch_name_abbrev}_median_isi"] = np.nan
            row_data[f"{ch_name_abbrev}_cv_isi"] = np.nan
            isi_arrays[ch_name_abbrev] = np.array([])

        row_data[f"{ch_name_abbrev}_spike_count"] = spike_count
        row_data[f"{ch_name_abbrev}_spike_rate"] = spike_rate
        row_data[f"{ch_name_abbrev}_spike_rate_per_hour"] = spike_rate_per_hour

    # --- Consolidated event plot ---
    for i in [10, 11]:
        axes_flat[i].remove()

    gs = axes[0, 0].get_gridspec()
    ax_events = fig.add_subplot(gs[2, 2:])

    if len(all_events) > 0:
        mne.viz.plot_events(
            all_events,
            sfreq=raw.info["sfreq"],
            event_id=all_event_id,
            axes=ax_events,
            show=False,
        )

        if ax_events.get_legend():
            ax_events.get_legend().remove()

        for collection in ax_events.collections:
            collection.set_sizes([10])
            collection.set_alpha(0.7)

        ax_events.set_title("Temporal Event Distribution", fontsize=12, pad=15)
        ax_events.tick_params(axis="both", labelsize=8)
        ax_events.set_xlabel("Time (s)", fontsize=10)
    else:
        ax_events.text(0.5, 0.5, "No Events to Plot", ha="center")

    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92, bottom=0.08)
    fig.suptitle(f"Summary: {recording_name}", fontsize=18)
    fig.savefig(hist_dir / f"{recording_name}_combined_summary.png")
    plt.close(fig)

    # --- Explicit cleanup ---
    del raw, all_events, fig, axes
    gc.collect()

    return row_data, isi_arrays


# Channels arranged so L/R pairs share the same column
#   Row 0 (Left):  LAud  LVis  LHip  LBar  LMot
#   Row 1 (Right): RAud  RVis  RHip  RBar  RMot
CHANNEL_GRID = [
    ["LAud", "LVis", "LHip", "LBar", "LMot"],
    ["RAud", "RVis", "RHip", "RBar", "RMot"],
]
GENOTYPE_ORDER = ["WT", "Het", "Mut"]
GENOTYPE_COLORS = {"WT": "#4CAF50", "Het": "#2196F3", "Mut": "#F44336"}
SD_THRESHOLD = 3  # Exclude values beyond this many SDs from the column mean


def _filter_outliers(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove values more than SD_THRESHOLD standard deviations from the mean.

    Returns (filtered_array, n_excluded).
    """
    if len(values) < 3:
        return values, 0
    mean, std = np.mean(values), np.std(values)
    mask = np.abs(values - mean) <= SD_THRESHOLD * std
    return values[mask], int(np.sum(~mask))


def _violin_subplot(ax, data_by_geno: list[np.ndarray], channel_name: str,
                    ylabel: str):
    """Draw a single violin subplot for one channel, return total outliers excluded."""
    total_excluded = 0
    filtered = []
    for arr in data_by_geno:
        filt, n_exc = _filter_outliers(arr)
        filtered.append(filt)
        total_excluded += n_exc

    # Need at least 2 data points per group for a violin; fall back to scatter
    positions = list(range(1, len(GENOTYPE_ORDER) + 1))

    for pos, filt, geno in zip(positions, filtered, GENOTYPE_ORDER):
        color = GENOTYPE_COLORS[geno]
        if len(filt) >= 2:
            vp = ax.violinplot(filt, positions=[pos], showmedians=True,
                               showextrema=False, widths=0.7)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.6)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.5)
        # Overlay individual points
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(filt))
        ax.scatter(pos + jitter, filt, s=10, alpha=0.4, color=color, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(GENOTYPE_ORDER)

    title = channel_name
    if total_excluded > 0:
        title += f"\n({total_excluded} outlier{'s' if total_excluded > 1 else ''} excluded)"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=9)

    return total_excluded


def plot_genotype_boxplots(csv_path: Path, output_dir: Path):
    """
    Load the spike summary CSV and create violin-plot figures comparing
    ISI and spike rate across WT, Het, and Mut genotypes for each channel.

    Layout: 2 rows × 5 columns, Left channels on top, Right on bottom,
    so each column represents one brain region (Aud, Vis, Hip, Bar, Mot).

    Outliers beyond 3 SD from the per-column mean are excluded, with
    the count noted in each subplot title.

    Can be called independently of the main analysis pipeline.
    """
    df = pd.read_csv(csv_path)

    # Keep only WT / Het / Mut
    df = df[df["genotype"].isin(GENOTYPE_ORDER)].copy()
    print(f"Loaded {len(df)} recordings (WT={len(df[df.genotype=='WT'])}, "
          f"Het={len(df[df.genotype=='Het'])}, Mut={len(df[df.genotype=='Mut'])})")

    # ---- Figure 1: Median ISI per channel ----
    fig_isi, axes_isi = plt.subplots(2, 5, figsize=(26, 10))
    total_isi_excluded = 0

    for row_idx, row_channels in enumerate(CHANNEL_GRID):
        for col_idx, ch in enumerate(row_channels):
            ax = axes_isi[row_idx, col_idx]
            col = f"{ch}_median_isi"
            if col not in df.columns:
                ax.set_title(f"{ch}\n(no data)")
                continue
            data_by_geno = [
                df.loc[df["genotype"] == g, col].dropna().values
                for g in GENOTYPE_ORDER
            ]
            total_isi_excluded += _violin_subplot(ax, data_by_geno, ch,
                                                  "Median ISI (s)")

    fig_isi.suptitle(
        f"Median Inter-Spike Interval by Genotype  (>{SD_THRESHOLD} SD outliers removed)",
        fontsize=16, y=0.98,
    )
    fig_isi.tight_layout(rect=[0, 0, 1, 0.95])
    isi_path = output_dir / "violin_ISI_by_genotype.png"
    fig_isi.savefig(isi_path, dpi=150)
    plt.close(fig_isi)
    print(f"Saved ISI violin:   {isi_path}  ({total_isi_excluded} total outliers removed)")

    # ---- Figure 2: Spike rate per channel ----
    fig_rate, axes_rate = plt.subplots(2, 5, figsize=(26, 10))
    total_rate_excluded = 0

    for row_idx, row_channels in enumerate(CHANNEL_GRID):
        for col_idx, ch in enumerate(row_channels):
            ax = axes_rate[row_idx, col_idx]
            col = f"{ch}_spike_rate_per_hour"
            # Fall back to Hz column if per-hour not yet in CSV
            if col not in df.columns:
                col = f"{ch}_spike_rate"
                if col not in df.columns:
                    ax.set_title(f"{ch}\n(no data)")
                    continue
                # Convert Hz → counts/hr on the fly
                scale = 3600
            else:
                scale = 1
            data_by_geno = [
                df.loc[df["genotype"] == g, col].dropna().values * scale
                for g in GENOTYPE_ORDER
            ]
            total_rate_excluded += _violin_subplot(ax, data_by_geno, ch,
                                                   "Spike Rate (counts/hr)")

    fig_rate.suptitle(
        f"Spike Rate (counts/hr) by Genotype  (>{SD_THRESHOLD} SD outliers removed)",
        fontsize=16, y=0.98,
    )
    fig_rate.tight_layout(rect=[0, 0, 1, 0.95])
    rate_path = output_dir / "violin_spike_rate_by_genotype.png"
    fig_rate.savefig(rate_path, dpi=150)
    plt.close(fig_rate)
    print(f"Saved rate violin:  {rate_path}  ({total_rate_excluded} total outliers removed)")

    # ---- Figure 3: CV of ISI per channel ----
    fig_cv, axes_cv = plt.subplots(2, 5, figsize=(26, 10))
    total_cv_excluded = 0

    for row_idx, row_channels in enumerate(CHANNEL_GRID):
        for col_idx, ch in enumerate(row_channels):
            ax = axes_cv[row_idx, col_idx]
            col = f"{ch}_cv_isi"
            if col not in df.columns:
                ax.set_title(f"{ch}\n(no data)")
                continue
            data_by_geno = [
                df.loc[df["genotype"] == g, col].dropna().values
                for g in GENOTYPE_ORDER
            ]
            total_cv_excluded += _violin_subplot(ax, data_by_geno, ch,
                                                 "CV of ISI")
            # Add reference line at CV=1 (Poisson firing)
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    fig_cv.suptitle(
        f"Coefficient of Variation of ISI by Genotype  (>{SD_THRESHOLD} SD outliers removed)\n"
        r"CV<1: regular | CV=1: Poisson | CV>1: bursty",
        fontsize=14, y=0.99,
    )
    fig_cv.tight_layout(rect=[0, 0, 1, 0.93])
    cv_path = output_dir / "violin_CV_ISI_by_genotype.png"
    fig_cv.savefig(cv_path, dpi=150)
    plt.close(fig_cv)
    print(f"Saved CV violin:    {cv_path}  ({total_cv_excluded} total outliers removed)")


def main():
    # --- Paths ---
    data_path = Path(
        "/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_Yastika/results/fdsars"
    )
    output_dir = data_path / "summary_output"
    hist_dir = output_dir / "histogram_ISI"
    hist_dir.mkdir(parents=True, exist_ok=True)
    session_csv_path = output_dir / "spike_summary_per_session.csv"
    animal_csv_path = output_dir / "spike_summary.csv"

    # Level-1 folders = animals; exclude the output directory
    animal_folders = sorted(
        p for p in data_path.iterdir()
        if p.is_dir() and p != output_dir
    )
    print(f"Found {len(animal_folders)} animal folders.")

    # --- Process every recording session, grouped by animal ---
    all_session_rows = []   # one row per session
    all_animal_rows = []    # one row per animal (averaged)

    for animal_idx, animal_dir in enumerate(animal_folders, start=1):
        # Find all recording-session subfolders with .fif files
        session_fifs = []
        for session_dir in sorted(animal_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            fif_search = list(session_dir.rglob("*raw.fif"))
            if fif_search:
                session_fifs.append(fif_search[0])

        if not session_fifs:
            continue

        print(f"[{animal_idx}/{len(animal_folders)}] {animal_dir.name}  "
              f"({len(session_fifs)} session{'s' if len(session_fifs) > 1 else ''})")

        # Process each session for this animal
        animal_session_rows = []
        animal_isi = {}  # ch_abbrev -> list of ISI arrays
        for fif_path in session_fifs:
            try:
                result = process_one_recording(fif_path, hist_dir)
                if result is not None:
                    row, isi_arrays = result
                    animal_session_rows.append(row)
                    all_session_rows.append(row)
                    # Accumulate raw ISI arrays per channel
                    for ch, arr in isi_arrays.items():
                        animal_isi.setdefault(ch, []).append(arr)
            except Exception as e:
                print(f"  ERROR ({fif_path.parent.name}): {e}")

        if not animal_session_rows:
            continue

        # Aggregate per-animal
        df_sessions = pd.DataFrame(animal_session_rows)
        animal_row = {}

        # Use folder name as animal_id (not JSON metadata)
        animal_row["animal_id"] = animal_dir.name

        # Keep identity columns from the first session
        for col in ["sex", "genotype"]:
            animal_row[col] = df_sessions[col].iloc[0]

        animal_row["num_sessions"] = len(df_sessions)

        # Duration-weighted median for spike count, rate, etc.
        weights = df_sessions["total_duration_sec"].values

        # Total duration = sum across sessions (not weighted median)
        animal_row["total_duration_sec"] = float(weights.sum())
        animal_row["total_duration_hours"] = float(weights.sum() / 3600)

        numeric_cols = df_sessions.select_dtypes(include="number").columns
        for col in numeric_cols:
            # Skip ISI/CV columns — computed from pooled raw ISIs below
            if col.endswith("_median_isi") or col.endswith("_cv_isi"):
                continue
            # Skip total_duration — already summed above
            if col == "total_duration_sec" or col == "total_duration_hours":
                continue
            vals = df_sessions[col].values
            animal_row[col] = _weighted_median(vals, weights)

        # Compute median ISI and CV from pooled raw ISI values
        for ch, arrays in animal_isi.items():
            pooled = np.concatenate(arrays)
            if len(pooled) > 0:
                animal_row[f"{ch}_median_isi"] = float(np.median(pooled))
                mean_p = np.mean(pooled)
                animal_row[f"{ch}_cv_isi"] = float(np.std(pooled) / mean_p) if mean_p > 0 else np.nan
            else:
                animal_row[f"{ch}_median_isi"] = np.nan
                animal_row[f"{ch}_cv_isi"] = np.nan

        all_animal_rows.append(animal_row)

    # --- Save per-animal CSV first (most important output) ---
    df_animals = pd.DataFrame(all_animal_rows)
    df_animals.to_csv(animal_csv_path, index=False)
    print(f"\nSaved {len(all_animal_rows)} animals  to {animal_csv_path}")

    # --- Save per-session CSV ---
    df_sessions_all = pd.DataFrame(all_session_rows)
    df_sessions_all.to_csv(session_csv_path, index=False)
    print(f"Saved {len(all_session_rows)} sessions to {session_csv_path}")

    # Free memory before generating plots
    del all_session_rows, all_animal_rows, df_sessions_all
    gc.collect()

    # --- Generate genotype comparison violin plots (per-animal data) ---
    plot_genotype_boxplots(animal_csv_path, output_dir)


if __name__ == "__main__":
    if "--boxplots-only" in sys.argv:
        # Skip the full analysis, just regenerate boxplots from existing CSV
        data_path = Path(
            "/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_Yastika/results/fdsars"
        )
        output_dir = data_path / "summary_output"
        csv_path = output_dir / "spike_summary.csv"

        if not csv_path.exists():
            print(f"ERROR: CSV not found at {csv_path}")
            print("Run the full analysis first (without --boxplots-only).")
            sys.exit(1)

        plot_genotype_boxplots(csv_path, output_dir)
    else:
        main()
