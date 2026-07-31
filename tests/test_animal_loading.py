"""Integration tests for the shared animal loading path used by the Snakemake pipeline.

These drive the committed ``config/datasets/mini_real.yaml`` fixture through
``load_dataset_config`` and ``load_animal_recordings``, which is the same path
``workflow/scripts/generate_wars.py`` takes. Marked integration and slow because
they load recordings through SpikeInterface.
"""

from pathlib import Path

import pytest

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
