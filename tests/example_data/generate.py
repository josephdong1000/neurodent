"""
Generate a minimal synthetic dataset for integration testing.

This module creates a tiny but structurally valid NWB dataset so that the
full Snakemake pipeline can be exercised end-to-end without requiring real
EEG recordings.

The generated data uses the SpikeInterface-compatible NWB format: each
session segment is stored as a single ``.nwb`` file.  The directory layout
uses ``{animal}/{session}/{index}.nwb`` placeholders, matching the
``example`` dataset config.

Usage from tests::

    from tests.example_data.generate import create_synthetic_dataset

    def test_pipeline(tmp_path):
        ds = create_synthetic_dataset(tmp_path)
        # ds["data_root"]        → Path to raw data root
        # ds["samples_config"]   → dict suitable for samples_example.json
        # ds["animals"]          → list of animal id strings
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Defaults matching the real pipeline constants
# ---------------------------------------------------------------------------
SAMPLING_RATE = 1000          # Hz  (neurodent.constants.GLOBAL_SAMPLING_RATE)
N_CHANNELS = 8
DURATION_SECONDS = 5          # tiny: just enough for one analysis window
DTYPE = np.float32            # neurodent.constants.GLOBAL_DTYPE
# Use standard abbreviations recognised by parse_chname_to_abbrev / DEFAULT_ID_TO_NAME.
# IDs correspond to 8 of the 10 default channels (drop LMot/RMot for brevity).
CHANNEL_NAMES = [
    "LAud", "LVis", "LHip", "LBar",
    "RBar", "RHip", "RVis", "RAud",
]
# NWB electrode table IDs matching the pipeline's DEFAULT_ID_TO_NAME keys.
CHANNEL_IDS = [9, 10, 12, 14, 17, 19, 21, 22]


def _write_nwb_file(
    filepath: Path,
    *,
    n_channels: int = N_CHANNELS,
    sampling_rate: int = SAMPLING_RATE,
    duration_s: float = DURATION_SECONDS,
    channel_names: list[str] | None = None,
    channel_ids: list[int] | None = None,
    seed: int = 42,
) -> dict:
    """Write a single NWB file with synthetic EEG data.

    The file can be loaded back with ``spikeinterface.extractors.read_nwb_recording``
    or any NWB-compatible reader.

    Args:
        filepath: Output ``.nwb`` path (parent directories created if absent).
        n_channels: Number of EEG channels.
        sampling_rate: Sampling rate in Hz.
        duration_s: Recording duration in seconds.
        channel_names: Channel label list.  Defaults to standard abbreviations.
        channel_ids: Integer electrode IDs for the NWB electrode table.
            Defaults to ``CHANNEL_IDS`` so that SpikeInterface reads them
            as IDs matching ``DEFAULT_ID_TO_NAME``.
        seed: NumPy RNG seed for reproducibility.

    Returns:
        Dict with ``"nwb_path"`` and ``"n_samples"``.
    """
    import pynwb
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.ecephys import ElectricalSeries
    from datetime import datetime
    from dateutil.tz import tzlocal

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    channel_names = channel_names or CHANNEL_NAMES[:n_channels]
    channel_ids = channel_ids or CHANNEL_IDS[:n_channels]
    rng = np.random.default_rng(seed)

    n_samples = int(duration_s * sampling_rate)

    # Realistic EEG: sum of sine waves + noise (µV scale)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    data = np.zeros((n_samples, n_channels), dtype=DTYPE)
    for ch in range(n_channels):
        freq = 2 + ch * 3  # spread across delta → beta
        data[:, ch] = (
            50 * np.sin(2 * np.pi * freq * t)
            + 20 * np.sin(2 * np.pi * (freq + 8) * t)
            + 5 * rng.standard_normal(n_samples)
        ).astype(DTYPE)

    # Build NWB file
    nwbfile = NWBFile(
        session_description="synthetic EEG for integration testing",
        identifier=f"test_{filepath.stem}_{seed}",
        session_start_time=datetime(2025, 1, 15, 10, 0, 0, tzinfo=tzlocal()),
    )

    device = nwbfile.create_device(name="test_device")
    group = nwbfile.create_electrode_group(
        name="test_group", description="synthetic channels",
        device=device, location="brain",
    )

    for i, ch_name in enumerate(channel_names):
        nwbfile.add_electrode(
            id=channel_ids[i],
            x=0.0, y=0.0, z=float(i),
            imp=0.0, filtering="none",
            group=group, location="brain",
        )

    electrode_region = nwbfile.create_electrode_table_region(
        list(range(n_channels)), "all channels",
    )

    es = ElectricalSeries(
        name="ElectricalSeries",
        data=data,
        electrodes=electrode_region,
        rate=float(sampling_rate),
    )
    nwbfile.add_acquisition(es)

    with NWBHDF5IO(str(filepath), "w") as io:
        io.write(nwbfile)

    return {"nwb_path": filepath, "n_samples": n_samples}


def create_synthetic_dataset(
    root: Path,
    *,
    animals: list[dict] | None = None,
    n_sessions: int = 1,
    duration_s: float = DURATION_SECONDS,
) -> dict:
    """Create a full synthetic NWB dataset under *root*.

    The layout uses ``{animal}/{session}/{index}.nwb`` placeholders::

        root/
        └── example_session/
            ├── ExWT/
            │   └── day1/
            │       └── recording.nwb
            └── ExKO/
                └── day1/
                    └── recording.nwb

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
            {"id": "ExKO", "sex": "F", "gene": "KO"},
        ]

    data_root = root / "raw"
    session_folder = "example_session"
    animal_ids = [a["id"] for a in animals]

    for animal in animals:
        for day_idx in range(1, n_sessions + 1):
            nwb_path = (
                data_root / session_folder / animal["id"]
                / f"day{day_idx}" / "recording.nwb"
            )
            _write_nwb_file(
                nwb_path,
                duration_s=duration_s,
                # Modulo 2^31 ensures the hash fits within NumPy's RNG seed range
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

