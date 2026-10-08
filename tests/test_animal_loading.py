"""Integration tests for the shared animal loading path used by the Snakemake pipeline.

These drive the committed ``config/datasets/mini_real.yaml`` fixture through
``load_dataset_config`` and ``load_animal_recordings``, which is the same path
``workflow/scripts/generate_wars.py`` takes. Marked integration and slow because
they load recordings through SpikeInterface.
"""

from pathlib import Path

import pytest

from neurodent.loading import AnimalOrganizer
from neurodent.workflow import apply_samples_config
from neurodent.workflow.utils import (
    expand_animals_config,
    load_animal_recordings,
    load_dataset_config,
    resolve_samples_config,
)

MINI_REAL_ABBREVS = {"LMot", "RMot", "LBar", "RBar", "LHip", "RHip", "LAud", "RAud", "LVis", "RVis"}


@pytest.fixture
def _at_repo_root(monkeypatch):
    """Run from the repo root so dataset extract_func paths resolve."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])


def _prepare(dataset):
    """Assemble config, expand samples, and install the channel map and ANIMAL_METADATA globals.

    ``apply_samples_config`` must run before any ``load_animal_recordings`` or
    ``resolve_channels`` call, so it happens here.

    Args:
        dataset (str): Dataset name (the ``config/datasets/{name}.yaml`` stem).

    Returns:
        tuple[dict, dict]: ``(config, samples_config)``.
    """
    from neurodent.core.utils import set_temp_directory

    config = load_dataset_config(dataset)
    samples_config = expand_animals_config(resolve_samples_config(config))
    set_temp_directory(config["temp_directory"])
    apply_samples_config(samples_config)
    return config, samples_config


def _prepare_with_sources(dataset, animal_id, sources):
    """Prepare a dataset config with `sources` declared on one animal.

    Injected before expand_animals_config, which is how a real config declares it and
    also the only way _animal_overrides gets created for a dataset that otherwise has
    no per-animal overrides.
    """
    from neurodent.core.utils import set_temp_directory

    config = load_dataset_config(dataset)
    raw = resolve_samples_config(config)
    for animal in raw["animals"]:
        if animal["id"] == animal_id:
            animal["sources"] = sources
    samples_config = expand_animals_config(raw)
    set_temp_directory(config["temp_directory"])
    apply_samples_config(samples_config)
    return config, samples_config


def _load_animal(samples_config, config, animal_id):
    """Load one animal exactly as WAR generation does, honoring its channel_subset."""
    channel_subset = samples_config.get("_animal_channel_subsets", {}).get(animal_id)
    return load_animal_recordings(
        samples_config, config, [("", animal_id, "")], animal_id, channel_subset=channel_subset
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_load_animal_recordings_mini_real(_at_repo_root):
    """The shared loader reconstructs a real animal's recordings with a canonical montage."""
    pytest.importorskip("spikeinterface")
    from neurodent.core.utils import resolve_channels

    config, samples_config = _prepare("mini_real")
    ao = load_animal_recordings(samples_config, config, [("", "A10", "")], "A10")

    assert ao.long_recordings, "expected at least one loaded recording for A10"
    abbrevs = set(resolve_channels(list(ao.long_recordings[0].channel_names)))
    assert abbrevs <= MINI_REAL_ABBREVS, f"unexpected channels: {abbrevs - MINI_REAL_ABBREVS}"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_validate_only_discovers_without_loading(_at_repo_root):
    """validate_only returns a discovery summary and agrees with the real load.

    The dry-run runs the same discovery, skip, and manual_datetimes validation the
    real load does, so its session count must match what an actual load produces.
    """
    pytest.importorskip("spikeinterface")

    config, samples_config = _prepare("mini_real")
    summary = load_animal_recordings(
        samples_config, config, [("", "A10", "")], "A10", validate_only=True
    )

    assert isinstance(summary, dict), "validate_only returns a summary dict, not an organizer"
    assert set(summary) >= {"n_sessions", "n_files", "sessions"}
    assert summary["n_sessions"] > 0, "expected the dry-run to discover sessions for A10"

    ao = _load_animal(samples_config, config, "A10")
    assert summary["n_sessions"] == len(ao.long_recordings), (
        "the dry-run session count must match what a real load produces"
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_animalday_single_source_mini_real(_at_repo_root):
    """The animalday has a single source of truth across the AnimalOrganizer and the WAR.

    ``from_lros`` stamps ``lro.animalday`` and the WAR reads it, so
    ``ao.animaldays`` equals the WAR's ``animalday`` column exactly. Previously the
    WAR re-derived it from the raw folder session while the AnimalOrganizer used the
    parsed date, so the two diverged for any dataset whose folder session is not a
    date (for example sox5's "062921" against "Jul-01-2021"). Those sessions were
    then silently dropped from LOF filtering and from detector scoring.
    """
    pytest.importorskip("spikeinterface")
    from neurodent.analysis import AnimalAnalyzer

    config, samples_config = _prepare("mini_real")
    for animal_id in ("A10", "F22"):
        ao = _load_animal(samples_config, config, animal_id)
        stamps = [getattr(lro, "animalday", None) for lro in ao.long_recordings]
        assert all(stamps), "every LRO must be stamped with its canonical animalday"
        assert stamps == list(ao.animaldays), "ao.animaldays must mirror lro.animalday"

        war = AnimalAnalyzer(ao).compute_windowed_analysis(
            ["rms"], window_s=5, apply_notch_filter=True, multiprocess_mode="serial"
        )
        assert set(ao.animaldays) == set(war.result["animalday"].unique()), (
            "the WAR animalday column must equal ao.animaldays so every AO to WAR join lines up"
        )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_per_animal_sources_load_independently(_at_repo_root):
    """An animal may draw from several sources, each with its own pattern and reader.

    IQSEC2's dual-tree cohorts need this: the same animal was recorded sequentially by
    two acquisition systems, so one entry must carry an Intan source whose filenames give
    start times and a DataWave source whose sidecars give end times. Without it the animal
    has to be split into two entries, which counts one mouse twice in cross-animal
    statistics.

    Two identical sources are used here so the expected file count is exactly double the
    single-source baseline, which distinguishes "both sources ran" from "one ran twice".
    """
    pytest.importorskip("spikeinterface")

    config, samples_config = _prepare("mini_real")
    baseline = load_animal_recordings(
        samples_config, config, [("", "A10", "")], "A10", validate_only=True
    )

    pattern = config["analysis"]["war_generation"]["pattern"]
    config, samples_config = _prepare_with_sources("mini_real", "A10", [
        {"pattern": pattern, "lro_kwargs": {"datetimes_are_start": True}},
        {"pattern": pattern, "lro_kwargs": {"datetimes_are_start": True}},
    ])
    doubled = load_animal_recordings(
        samples_config, config, [("", "A10", "")], "A10", validate_only=True
    )

    assert doubled["n_files"] == baseline["n_files"] * 2, (
        "each source must be discovered independently"
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mutates_constants
def test_source_lro_kwargs_win_over_animal_level(_at_repo_root):
    """A source's own lro_kwargs must beat the animal-level ones, not the reverse.

    The per-animal update used to run last and silently clobber the per-source value,
    which matters because the two trees need opposite datetimes_are_start: rhd filename
    stamps are starts, DataWave LastEdit is an end.
    """
    pytest.importorskip("spikeinterface")

    config = load_dataset_config("mini_real")
    pattern = config["analysis"]["war_generation"]["pattern"]
    raw = resolve_samples_config(config)
    for animal in raw["animals"]:
        if animal["id"] == "A10":
            animal["lro_kwargs"] = {"datetimes_are_start": False}
            animal["sources"] = [
                {"pattern": pattern, "lro_kwargs": {"datetimes_are_start": True}},
            ]
    samples_config = expand_animals_config(raw)
    apply_samples_config(samples_config)

    captured = {}
    real_ao = AnimalOrganizer

    class _Spy(real_ao):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs.get("lro_kwargs", {}))
            super().__init__(*args, **kwargs)

    import neurodent.workflow.utils.animal_loading as al

    monkey = al.AnimalOrganizer
    al.AnimalOrganizer = _Spy
    try:
        load_animal_recordings(
            samples_config, config, [("", "A10", "")], "A10", validate_only=True
        )
    finally:
        al.AnimalOrganizer = monkey

    assert captured.get("datetimes_are_start") is True, (
        "the per-source value must survive the per-animal update"
    )
