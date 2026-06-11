"""Regression guard for the ``/``-in-genotype path bug.

Real arxrosa genotypes (e.g. ``Arx(F/y); Rosa(+/wt)``) contain ``/`` characters
that are valid display labels but break filesystem paths.  We've hit this bug
class four separate times on `run/arxrosa` so far; these tests construct WARs,
FDSARs, and AnimalPlotters with the trigger pattern and verify every
file-writing entry point produces a file without raising ``FileNotFoundError``
or similar.

The convention they protect is documented in :func:`neurodent.core.utils.slugify`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless

from neurodent.core.utils import slugify
from neurodent.visualization import WindowAnalysisResult
from neurodent.visualization.plotting.animal import AnimalPlotter

ARXROSA_GENOTYPE = "Arx(F/y); Rosa(+/wt)"
ARXROSA_ANIMAL_ID = "ArxRosa-1015"
ARXROSA_ANIMAL_DAY = "Feb-24-2015"


def _slashy_war() -> WindowAnalysisResult:
    """Build a small WAR whose genotype + animaldays carry the trigger pattern."""
    n_rows = 6
    C = 4
    rng = np.random.default_rng(42)
    animaldays = [f"{ARXROSA_ANIMAL_ID} {ARXROSA_GENOTYPE} {ARXROSA_ANIMAL_DAY}"] * n_rows
    data = {
        "animal": [ARXROSA_ANIMAL_ID] * n_rows,
        "animalday": animaldays,
        "genotype": [ARXROSA_GENOTYPE] * n_rows,
        "isday": [True, False] * (n_rows // 2),
        "timestamp": pd.date_range("2015-02-24", periods=n_rows, freq="1h"),
        "duration": [3600.0] * n_rows,
        "rms": [rng.random(C).tolist() for _ in range(n_rows)],
    }
    return WindowAnalysisResult(
        result=pd.DataFrame(data),
        animal_id=ARXROSA_ANIMAL_ID,
        genotype=ARXROSA_GENOTYPE,
        channel_names=["LMot", "RMot", "LBar", "RBar"],
        suppress_short_interval_error=True,
    )


class TestSlashInGenotypePaths:
    """Cross-cutting regression suite for the path-safety convention."""

    def test_path_safe_properties_strip_slashes(self):
        """Layer C1: ``path_safe_*`` accessors return path-component-safe strings."""
        war = _slashy_war()

        # Display attrs stay verbatim (these are correct domain notation).
        assert war.genotype == ARXROSA_GENOTYPE
        assert "/" in war.animaldays[0]

        # Path-safe forms have no path-breaking characters.
        assert "/" not in war.path_safe_animal_id
        assert war.path_safe_animal_id == slugify(ARXROSA_ANIMAL_ID)
        for ad in war.path_safe_animaldays:
            assert "/" not in ad
            assert ";" not in ad

    def test_war_save_then_reload_with_slashy_genotype(self):
        """The library can save + reload a WAR with the trigger pattern intact."""
        war = _slashy_war()
        with tempfile.TemporaryDirectory() as tmp:
            war.save_parquet_and_json(tmp, filename="war")
            reloaded = WindowAnalysisResult.load_parquet_and_json(folder_path=tmp)
            # Genotype + animaldays round-trip unchanged in their display form.
            assert reloaded.genotype == ARXROSA_GENOTYPE
            assert reloaded.animaldays[0] == war.animaldays[0]

    def test_animal_plotter_handles_slashy_title(self):
        """Layer A: ``_handle_figure`` defensive-slugifies the title."""
        with tempfile.TemporaryDirectory() as tmp:
            war = _slashy_war()
            save_path = Path(tmp) / war.path_safe_animal_id
            plotter = AnimalPlotter(war, save_fig=True, save_path=str(save_path))
            import matplotlib.pyplot as plt

            fig, _ = plt.subplots()
            # Title contains the trigger pattern.  Pre-fix, this would raise
            # FileNotFoundError because matplotlib treats '/' as a path separator.
            plotter._handle_figure(
                fig,
                title=f"coherecorr_spectral_{ARXROSA_ANIMAL_ID} {ARXROSA_GENOTYPE} {ARXROSA_ANIMAL_DAY}",
            )
            # At least one PNG ended up next to save_path.
            written = list(Path(tmp).glob("*.png"))
            assert written, "AnimalPlotter._handle_figure did not write any PNG"
            for f in written:
                assert "/" not in f.name, f"Path-breaking '/' in filename: {f}"

    def test_animal_plotter_handles_none_title(self):
        """Regression: Layer A must still produce ``<save_path>.png`` when title is None."""
        with tempfile.TemporaryDirectory() as tmp:
            war = _slashy_war()
            save_path = Path(tmp) / war.path_safe_animal_id
            plotter = AnimalPlotter(war, save_fig=True, save_path=str(save_path))
            import matplotlib.pyplot as plt

            fig, _ = plt.subplots()
            plotter._handle_figure(fig, title=None)
            assert (save_path.parent / f"{save_path.name}.png").exists()

    def test_fdsar_path_safe_save_stem_strips_slashes(self):
        """Layer C1: ``FrequencyDomainSpikeAnalysisResult.path_safe_save_stem`` is safe."""
        import mne
        from neurodent.visualization.frequency_domain_results import (
            FrequencyDomainSpikeAnalysisResult,
        )

        # Minimal RawArray so the constructor's exactly-one-of guard is satisfied.
        info = mne.create_info(ch_names=["LMot", "RMot"], sfreq=1000.0, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((2, 100), dtype=float), info, verbose=False)
        fdsar = FrequencyDomainSpikeAnalysisResult(
            result_sas=None,
            result_mne=raw,
            spike_indices=[],
            detection_params={},
            animal_id=ARXROSA_ANIMAL_ID,
            genotype=ARXROSA_GENOTYPE,
            animal_day=ARXROSA_ANIMAL_DAY,
            bin_folder_name="",
            metadata=None,
            channel_names=["LMot", "RMot"],
            assume_from_number=False,
        )
        stem = fdsar.path_safe_save_stem
        assert "/" not in stem
        assert ";" not in stem
        # Stays informative: contains traces of all three input fields.
        assert "arxrosa-1015" in stem
        assert "feb-24-2015" in stem
