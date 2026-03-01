"""
Generate a minimal synthetic dataset for integration testing.

This module creates a tiny but structurally valid dataset that mirrors the
``sox5_bin`` (nest-mode) layout so that the full Snakemake pipeline can be
exercised end-to-end without requiring real EEG recordings.

The generated data lives entirely inside a temporary directory and is
suitable for use in pytest fixtures.

Usage from tests::

    from tests.example_data.generate import create_synthetic_dataset

    def test_pipeline(tmp_path):
        ds = create_synthetic_dataset(tmp_path)
        # ds["data_root"]        → Path to raw data root
        # ds["samples_config"]   → dict suitable for samples_example.json
        # ds["animals"]          → list of animal id strings
"""

import json
import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Defaults matching the real pipeline constants
# ---------------------------------------------------------------------------
SAMPLING_RATE = 1000          # Hz  (neurodent.constants.GLOBAL_SAMPLING_RATE)
N_CHANNELS = 8
DURATION_SECONDS = 5          # tiny: just enough for one analysis window
DTYPE = np.float32            # neurodent.constants.GLOBAL_DTYPE
CHANNEL_NAMES = [
    "C-001", "C-002", "C-003", "C-004",
    "C-005", "C-006", "C-007", "C-008",
]


def _write_bin_and_meta(
    folder: Path,
    *,
    file_stem: str = "rec",
    n_channels: int = N_CHANNELS,
    sampling_rate: int = SAMPLING_RATE,
    duration_s: float = DURATION_SECONDS,
    channel_names: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Write a column-major .bin file and companion _Meta.csv.

    This produces the same format consumed by neurodent's ``"bin"`` loader
    (``LongRecordingOrganizer(item, mode="bin")``).

    Args:
        folder: Directory to write into (created if absent).
        file_stem: Base filename (files will be ``{stem}_ColMajor.bin`` and
            ``{stem}_Meta.csv``).
        n_channels: Number of EEG channels.
        sampling_rate: Sampling rate in Hz.
        duration_s: Recording duration in seconds.
        channel_names: Channel label list.  Defaults to ``C-001`` … ``C-008``.
        seed: NumPy RNG seed for reproducibility.

    Returns:
        Dict with ``"bin_path"``, ``"meta_path"``, ``"n_samples"``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    channel_names = channel_names or CHANNEL_NAMES[:n_channels]
    rng = np.random.default_rng(seed)

    n_samples = int(duration_s * sampling_rate)

    # Realistic EEG: sum of sine waves + noise (µV scale)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    data = np.zeros((n_channels, n_samples), dtype=DTYPE)
    for ch in range(n_channels):
        freq = 2 + ch * 3  # spread across delta → beta
        data[ch] = (
            50 * np.sin(2 * np.pi * freq * t)
            + 20 * np.sin(2 * np.pi * (freq + 8) * t)
            + 5 * rng.standard_normal(n_samples)
        ).astype(DTYPE)

    # Column-major binary: shape (n_samples, n_channels) in Fortran order
    bin_path = folder / f"{file_stem}_ColMajor.bin"
    col_major = np.asfortranarray(data.T)  # (n_samples, n_channels)
    col_major.tofile(str(bin_path))

    # Companion metadata CSV
    meta_path = folder / f"{file_stem}_Meta.csv"
    timestamp = "2025-01-15T10:00:00"
    lines = [
        "Entity,BinColumn,Label,ProbeInfo,SampleRate,Units,Precision,LastEdit"
    ]
    for i, ch_name in enumerate(channel_names, start=1):
        probe = f"Probe/{ch_name}"
        lines.append(
            f"{i},{i},{ch_name},{probe},{sampling_rate},µV,float32,{timestamp}"
        )
    meta_path.write_text("\n".join(lines) + "\n")

    return {"bin_path": bin_path, "meta_path": meta_path, "n_samples": n_samples}


def create_synthetic_dataset(
    root: Path,
    *,
    animals: list[dict] | None = None,
    n_sessions: int = 1,
    duration_s: float = DURATION_SECONDS,
) -> dict:
    """Create a full synthetic dataset under *root*.

    The layout follows the **nest** convention expected by ``sox5_bin``::

        root/
        └── example_session/
            ├── ExWT/
            │   └── day1/
            │       ├── rec_ColMajor.bin
            │       └── rec_Meta.csv
            └── ExKO/
                └── day1/
                    ├── rec_ColMajor.bin
                    └── rec_Meta.csv

    A matching ``samples_config`` dict (equivalent to ``samples_example.json``)
    is also returned so that tests can feed it directly into the Snakemake
    pipeline or into ``inject_config_aliases``.

    Args:
        root: Top-level directory (typically ``tmp_path``).
        animals: List of ``{"id": str, "sex": str, "gene": str}`` dicts.
            Defaults to ``ExWT`` (WT) and ``ExKO`` (KO).
        n_sessions: Number of day-sessions per animal.
        duration_s: Recording duration per session in seconds.

    Returns:
        Dict with keys:
        - ``"data_root"``      — Path to the raw data directory
        - ``"samples_config"`` — dict ready for ``json.dump`` / ``inject_config_aliases``
        - ``"animals"``        — list of animal ID strings
        - ``"session_folder"`` — name of the session folder (e.g. ``"example_session"``)
    """
    if animals is None:
        animals = [
            {"id": "ExWT", "sex": "M", "gene": "WT"},
            {"id": "ExKO", "sex": "M", "gene": "KO"},
        ]

    data_root = root / "raw"
    session_folder = "example_session"
    animal_ids = [a["id"] for a in animals]

    for animal in animals:
        for day_idx in range(1, n_sessions + 1):
            day_folder = data_root / session_folder / animal["id"] / f"day{day_idx}"
            _write_bin_and_meta(
                day_folder,
                duration_s=duration_s,
                seed=hash(animal["id"]) % (2**31) + day_idx,
            )

    samples_config = {
        "data_parent_folder": str(data_root),
        "GENOTYPE_ALIASES": {
            "WT": ["WT", "ExWT"],
            "KO": ["KO", "ExKO"],
        },
        "ANIMAL_METADATA": animals,
        "data_folders_to_animal_ids": {
            session_folder: animal_ids,
        },
    }

    return {
        "data_root": data_root,
        "samples_config": samples_config,
        "animals": animal_ids,
        "session_folder": session_folder,
    }
