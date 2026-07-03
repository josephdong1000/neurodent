#!/usr/bin/env python3
"""
Validate that dask and serial multiprocess modes produce identical spike detection results on real data.

This script loads a real recording, runs spike detection in both serial and dask modes,
and compares the results at sample-level precision to ensure they are EXACTLY identical.

IMPORTANT:
- Run this script ONLY on the cluster, not locally
- Uses RHD file loading parameters from config.local.yaml
- Dask will automatically use all available CPU cores (request cores with: srun -c 10 --pty bash)
- Validates exact spike indices (sample-level precision), not just counts

Usage:
    # Request 10 cores for dask processing
    srun -c 10 --pty bash
    cd /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_Joseph_devtree

    # Run validation (default: 30 minutes of data)
    uv run python scripts/validate_dask_serial_spike_consistency.py \\
        --recording-folder "/path/to/recording/folder"

Example:
    srun -c 10 --pty bash
    cd /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_Joseph_devtree
    uv run python scripts/validate_dask_serial_spike_consistency.py \\
        --recording-folder "/mnt/isilon/marsh_single_unit/PythonEEG Data/AP3B2/Intan recordings/PortA-AP3B2wt-247-F-PortB-AP3B2homo-275-F-PortC-AP3B2wt-246-F-standardEEG 11-21-25_251121_122901" \\
        --duration 1800 \\
        --verbose
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from neurodent.analysis.spike_detection import FrequencyDomainSpikeDetector
from neurodent.loading.long_recording_organizer import LongRecordingOrganizer


def load_detection_params(config_path=None):
    """Load detection parameters from config file."""
    if config_path is None:
        config_path = Path("config/config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract FDSAR parameters (same path as pipeline: analysis.frequency_domain_spike_detection.default_params)
    fdsar_config = config.get("analysis", {}).get("frequency_domain_spike_detection", {})
    detection_params = fdsar_config.get("default_params", {})

    # Provide defaults if not in config
    default_params = {
        "bp": [3.0, 40.0],
        "notch": 60.0,
        "notch_q": 30.0,
        "freq_slices": [10.0, 20.0],
        "sneo_percentile": 99.99,
        "cluster_gap_ms": 80.0,
        "search_ms": 160.0,
        "baseline_ms": 500.0,
        "k_sigma": 3.0,
        "smooth_window": 7,
        "vote_k": 2,
        "smooth_len": 5,
    }

    # Merge with defaults
    for key, value in default_params.items():
        if key not in detection_params:
            detection_params[key] = value

    return detection_params


def load_recording(recording_folder, duration=None):
    """
    Load a recording using LongRecordingOrganizer (direct approach).

    Args:
        recording_folder: Path to recording folder containing .rhd files
        duration: Optional duration in seconds to load (loads full recording if None)

    Returns:
        SpikeInterface recording object
    """
    from datetime import datetime

    print(f"Loading recording from: {recording_folder}")

    # Load using LRO directly (same as pipeline uses internally for .rhd files)
    # Parameters from config.local.yaml for RHD processing
    lro = LongRecordingOrganizer(
        base_folder_path=Path(recording_folder),
        mode="si",  # SpikeInterface mode
        manual_datetimes=datetime(2025, 1, 1, 12, 0),  # Dummy date for validation
        file_pattern="*.rhd",  # Match all RHD files in folder
        extract_func="read_intan",  # Use Intan reader
        input_type="files",  # Concatenate multiple RHD files
        stream_id="0",  # Select amplifier channels
    )

    recording = lro.LongRecording

    # Optionally trim to specified duration
    if duration is not None:
        fs = recording.get_sampling_frequency()
        n_frames = int(duration * fs)
        recording = recording.frame_slice(start_frame=0, end_frame=n_frames)
        print(f"Trimmed to {duration}s ({n_frames} frames)")

    n_channels = recording.get_num_channels()
    n_samples = recording.get_num_frames()
    duration_actual = n_samples / recording.get_sampling_frequency()

    print(f"Recording loaded: {n_channels} channels, {duration_actual:.1f}s")

    return recording


def run_spike_detection(recording, detection_params, mode="serial"):
    """
    Run spike detection in specified mode.

    Args:
        recording: SpikeInterface recording
        detection_params: Detection parameters dict
        mode: "serial" or "dask"

    Returns:
        tuple: (spike_indices, runtime_seconds)
    """
    print(f"\n--- Running spike detection ({mode.upper()} mode) ---")

    # Report CPU cores available for dask
    if mode == "dask":
        cpu_count = os.cpu_count()
        print(f"CPU cores available for dask: {cpu_count}")

    start_time = time.time()

    spike_indices = FrequencyDomainSpikeDetector.detect_spikes_recording(
        recording, detection_params, multiprocess_mode=mode
    )

    runtime = time.time() - start_time

    # Count total spikes
    total_spikes = sum(len(spikes) for spikes in spike_indices)

    print(f"Runtime: {runtime:.2f}s")
    print(f"Total spikes detected: {total_spikes:,}")

    return spike_indices, runtime


def compare_results(spike_indices_serial, spike_indices_dask, verbose=False):
    """
    Compare spike detection results from serial and dask modes.

    Args:
        spike_indices_serial: Spike indices from serial mode
        spike_indices_dask: Spike indices from dask mode
        verbose: Print detailed channel-by-channel comparison

    Returns:
        bool: True if results are identical, False otherwise
    """
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS (Sample-Level Precision)")
    print("=" * 70)

    n_channels = len(spike_indices_serial)
    all_match = True
    mismatches = []

    for ch_idx in range(n_channels):
        serial_spikes = spike_indices_serial[ch_idx]
        dask_spikes = spike_indices_dask[ch_idx]

        channel_match = True

        if len(serial_spikes) != len(dask_spikes):
            msg = f"❌ Channel {ch_idx}: Different spike counts (serial={len(serial_spikes)}, dask={len(dask_spikes)})"
            print(msg)
            mismatches.append({
                "channel": ch_idx,
                "type": "count_mismatch",
                "serial_count": len(serial_spikes),
                "dask_count": len(dask_spikes),
            })
            all_match = False
            channel_match = False
        elif not np.array_equal(serial_spikes, dask_spikes):
            # Find first difference
            diff_mask = serial_spikes != dask_spikes
            first_diff_idx = np.where(diff_mask)[0][0]
            msg = f"❌ Channel {ch_idx}: Spike times differ ({len(serial_spikes)} spikes)"
            print(msg)
            print(f"   First difference at spike index {first_diff_idx}: "
                  f"serial={serial_spikes[first_diff_idx]}, dask={dask_spikes[first_diff_idx]}")
            mismatches.append({
                "channel": ch_idx,
                "type": "timing_mismatch",
                "first_diff_index": first_diff_idx,
                "serial_value": int(serial_spikes[first_diff_idx]),
                "dask_value": int(dask_spikes[first_diff_idx]),
            })
            all_match = False
            channel_match = False
        elif verbose or not channel_match:
            print(f"✅ Channel {ch_idx}: {len(serial_spikes):,} spikes (EXACT match)")

    print("=" * 70)

    if all_match:
        print("✅ SUCCESS: All channels match perfectly at sample-level precision!")
        print("   Spike counts AND exact spike indices are identical.")
        print("=" * 70)
        return True
    else:
        print(f"❌ FAILURE: Spike detection results differ between modes!")
        print(f"   {len(mismatches)} channels with mismatches")
        print("=" * 70)
        return False


def save_comparison_csv(spike_indices_serial, spike_indices_dask, output_path):
    """Save detailed spike comparison to CSV."""
    import pandas as pd

    rows = []
    for ch_idx in range(len(spike_indices_serial)):
        serial_spikes = spike_indices_serial[ch_idx]
        dask_spikes = spike_indices_dask[ch_idx]

        rows.append({
            "channel": ch_idx,
            "serial_count": len(serial_spikes),
            "dask_count": len(dask_spikes),
            "match": np.array_equal(serial_spikes, dask_spikes),
            "serial_first_10": str(serial_spikes[:10].tolist()) if len(serial_spikes) > 0 else "[]",
            "dask_first_10": str(dask_spikes[:10].tolist()) if len(dask_spikes) > 0 else "[]",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate dask vs serial spike detection consistency on real data"
    )
    parser.add_argument(
        "--recording-folder",
        type=Path,
        required=True,
        help="Path to recording folder (e.g., Intan recording folder)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1800,  # 30 minutes default
        help="Duration in seconds to load (default: 1800 = 30 minutes)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Save comparison to CSV file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed channel-by-channel comparison",
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("SPIKE DETECTION CONSISTENCY VALIDATION")
    print("Dask vs Serial Mode Comparison")
    print("Sample-Level Precision Required")
    print("=" * 70)

    # Load detection parameters
    detection_params = load_detection_params(args.config)
    print(f"\nDetection parameters:")
    for key, value in sorted(detection_params.items()):
        print(f"  {key}: {value}")

    # Load recording
    recording = load_recording(args.recording_folder, args.duration)

    # Run serial mode
    spike_indices_serial, runtime_serial = run_spike_detection(
        recording, detection_params, mode="serial"
    )

    # Run dask mode
    spike_indices_dask, runtime_dask = run_spike_detection(
        recording, detection_params, mode="dask"
    )

    # Compare results
    results_match = compare_results(
        spike_indices_serial, spike_indices_dask, verbose=args.verbose
    )

    # Print performance comparison
    speedup = runtime_serial / runtime_dask if runtime_dask > 0 else 0
    print(f"\nPerformance: Dask was {speedup:.2f}x faster than serial")
    print(f"  Serial: {runtime_serial:.2f}s")
    print(f"  Dask:   {runtime_dask:.2f}s")

    # Save CSV if requested
    if args.save_csv:
        save_comparison_csv(spike_indices_serial, spike_indices_dask, args.save_csv)

    # Exit with appropriate code
    sys.exit(0 if results_match else 1)


if __name__ == "__main__":
    main()
