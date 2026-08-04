"""Validate the shipped dataset configs' channel maps.

Channel resolution is exact: ``resolve_channel`` looks a raw name up in
``CHANNEL_ABBREV_BY_RAW`` and never infers one from a substring. A raw spelling that
the data presents but the map omits is therefore a hard failure at load, not a warning.

Every other channel test in this suite installs a map inside the test, so none of them
can catch a shipped config that is missing a spelling its own data uses. These do.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "config" / "datasets"


def _dataset_configs():
    """Yield ``(name, parsed_yaml)`` for every shipped dataset config."""
    for path in sorted(DATASETS_DIR.glob("*.yaml")):
        with path.open() as f:
            yield path.stem, yaml.safe_load(f) or {}


def _channels_of(config):
    """Return a dataset config's ``channels`` map, or None if it declares none."""
    return (config.get("samples_data") or {}).get("channels")


DATASETS = list(_dataset_configs())
NAMES = [name for name, _ in DATASETS]


@pytest.mark.parametrize("name", NAMES)
def test_channel_map_is_well_formed(name):
    """Each abbreviation maps to a non-empty list of non-empty strings."""
    config = dict(DATASETS)[name]
    channels = _channels_of(config)
    if channels is None:
        pytest.skip(f"{name} declares no channels map")

    assert isinstance(channels, dict), f"{name}: channels must be a mapping"
    for abbrev, raws in channels.items():
        assert isinstance(abbrev, str) and abbrev, f"{name}: bad abbreviation {abbrev!r}"
        assert isinstance(raws, list) and raws, f"{name}: {abbrev} has no raw names"
        for raw in raws:
            assert isinstance(raw, str) and raw.strip(), f"{name}: {abbrev} has bad raw name {raw!r}"


@pytest.mark.parametrize("name", NAMES)
def test_no_raw_name_maps_to_two_abbreviations(name):
    """A raw name under two abbreviations is ambiguous and raises at set_channel_map.

    Guarding it here names the offending config and spelling, rather than surfacing as a
    load-time error with no indication of which dataset is at fault.
    """
    config = dict(DATASETS)[name]
    channels = _channels_of(config)
    if channels is None:
        pytest.skip(f"{name} declares no channels map")

    owner = {}
    for abbrev, raws in channels.items():
        for raw in raws:
            assert raw not in owner, f"{name}: raw name {raw!r} maps to both {owner[raw]!r} and {abbrev!r}"
            owner[raw] = abbrev


def _bin_reader_datasets():
    """Names of datasets whose reader takes channel names from a Meta.csv sidecar."""
    out = []
    for name, config in DATASETS:
        kwargs = ((config.get("analysis") or {}).get("war_generation") or {}).get("lro_kwargs") or {}
        if "read_bin_csv_pair" in str(kwargs.get("extract_func", "")):
            out.append(name)
    return out


BIN_DATASETS = _bin_reader_datasets()


@pytest.mark.eeg_data
@pytest.mark.slow
@pytest.mark.parametrize("name", BIN_DATASETS)
def test_shipped_map_covers_every_raw_name_on_disk(name):
    """Every channel name the dataset's own data presents resolves under its own map.

    This is the only test that compares a shipped config against the bytes it will
    actually load. It caught sox5 shipping a map without the suffixless spellings of
    four regions, which appear in 689 of its 9700 non-empty sidecars.
    """
    import csv

    config = dict(DATASETS)[name]
    channels = _channels_of(config)
    if channels is None:
        pytest.skip(f"{name} declares no channels map")

    data_root = Path(str((config.get("samples_data") or {}).get("data_root", "")))
    if not data_root.is_dir():
        pytest.skip(f"{name}: data_root not available ({data_root})")

    known = {raw for raws in channels.values() for raw in raws}
    unresolved = {}
    scanned = 0
    for meta in data_root.rglob("*_Meta.csv"):
        try:
            rows = list(csv.DictReader(meta.open()))
        except (OSError, UnicodeDecodeError):
            continue
        if not rows:
            continue  # header-only sidecars carry no channel names
        scanned += 1
        for row in rows:
            label = (row.get("Label") or "").strip()
            if label and label not in known:
                unresolved.setdefault(label, 0)
                unresolved[label] += 1

    if scanned == 0:
        pytest.skip(f"{name}: no non-empty Meta.csv found under {data_root}")

    assert not unresolved, (
        f"{name}: {len(unresolved)} raw channel name(s) present on disk but absent from the "
        f"config's channels map, across {scanned} sidecar(s): "
        + ", ".join(f"{k!r} ({v} rows)" for k, v in sorted(unresolved.items(), key=lambda x: -x[1]))
    )
