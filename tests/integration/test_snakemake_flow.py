"""
Integration Tests for Snakemake Workflow
========================================

Tests that validate the pipeline's data-loading path using a minimal
synthetic NWB dataset generated on the fly.  These tests exercise the real
``FileDiscoverer``, ``AnimalOrganizer``, and analysis pipeline against actual
files on disk, without requiring production-scale recordings.

Running
-------
Run only integration tests::

    uv run pytest tests/integration/ -v -m integration

Or include them in the full suite::

    uv run pytest tests/ -v
"""

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Custom extractor for dual .bin/.csv tests
# ---------------------------------------------------------------------------


def _bin_csv_extractor(discovered_file, **kwargs):
    """Custom extractor that reads paired .bin + .csv files into a recording.

    This is the kind of function a user would write when their data
    consists of multiple files per recording segment (e.g. sox5 format).
    ``AnimalOrganizer`` passes the multi-file ``DiscoveredFile`` object
    directly to this callable.
    """
    import csv
    import os
    import spikeinterface.core as si_core

    bin_path = [p for p in discovered_file.paths if p.endswith(".bin")][0]
    csv_path = [p for p in discovered_file.paths if p.endswith(".csv")][0]

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_channels = len(rows)
    sampling_rate = float(rows[0]["sampling_rate"])
    file_size = os.path.getsize(bin_path)
    bytes_per_frame = np.dtype(np.float32).itemsize * n_channels
    remainder = file_size % bytes_per_frame
    if remainder != 0:
        raise ValueError(
            f"Binary file size ({file_size} bytes) is not divisible by the "
            f"expected frame size ({bytes_per_frame} bytes = "
            f"{np.dtype(np.float32).itemsize} bytes/float32 × {n_channels} channels). "
            f"Remainder: {remainder} bytes. "
            f"This usually means either:\n"
            f"  1. The number of channels in the CSV metadata ({n_channels}) "
            f"does not match the binary — check '{csv_path}'.\n"
            f"  2. The binary file is corrupt or was truncated during "
            f"transfer — re-export or re-copy '{bin_path}'.\n"
            f"  3. The binary uses a different dtype (e.g. float64 or int16) "
            f"instead of float32."
        )
    n_samples = file_size // bytes_per_frame
    data = np.memmap(bin_path, dtype=np.float32, mode="r", shape=(n_samples, n_channels), order="F")

    return si_core.NumpyRecording(
        traces_list=[data],
        sampling_frequency=sampling_rate,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def example_pipeline_env(tmp_path_factory):
    """Create a complete, tiny pipeline environment under tmp_path.

    Returns a dict with ``data_root``, ``samples_config``, ``animals``,
    ``session_folder``, and the full ``config`` dict that would normally
    come from Snakemake.
    """
    from tests.integration.generate import create_synthetic_dataset

    tmp_path = tmp_path_factory.mktemp("pipeline")
    ds = create_synthetic_dataset(tmp_path, n_sessions=2, duration_s=3)

    # Build a minimal pipeline config (mirrors config/config.yaml + example.yaml)
    pipeline_config = {
        "temp_directory": str(tmp_path / "tmp"),
        "analysis": {
            "war_generation": {
                "pattern": "{animal}/{session}/{index}.nwb",
                "assume_from_number": True,
                "skip_sessions": [],
                "lro_kwargs": {
                    "mode": "si",
                    "extract_func": "read_nwb_recording",
                    "multiprocess_mode": "serial",
                },
            },
        },
        "cluster": {
            "war_generation": {"interface": None},
        },
        "overrides": {},
    }

    ds["config"] = pipeline_config
    return ds


# ---------------------------------------------------------------------------
# Tests — Dataset Generation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestExampleDatasetGeneration:
    """Verify that the synthetic dataset generator produces valid NWB files."""

    def test_creates_directory_tree(self, example_dataset):
        """The generated dataset has the expected directory structure."""
        root = example_dataset["data_root"]
        session = example_dataset["session_folder"]

        for animal_id in example_dataset["animals"]:
            day_dir = root / session / animal_id / "day1"
            assert day_dir.is_dir(), f"Missing directory: {day_dir}"

            nwb_files = list(day_dir.glob("*.nwb"))
            assert len(nwb_files) == 1, f"Expected 1 NWB file, got {nwb_files}"

    def test_nwb_file_readable_by_spikeinterface(self, example_dataset):
        """NWB file can be loaded by SpikeInterface."""
        import spikeinterface.extractors as se

        root = example_dataset["data_root"]
        session = example_dataset["session_folder"]
        animal_id = example_dataset["animals"][0]

        nwb_file = next((root / session / animal_id / "day1").glob("*.nwb"))
        rec = se.read_nwb_recording(str(nwb_file))

        assert rec.get_num_channels() == 8
        # example_dataset fixture uses default duration_s=5, sr=1000
        assert rec.get_num_samples() == 5 * 1000

    def test_samples_config_structure(self, example_dataset):
        """samples_config contains all required keys."""
        sc = example_dataset["samples_config"]
        assert "data_root" in sc
        assert "ANIMAL_METADATA" in sc
        assert "data_folders_to_animal_ids" in sc
        assert "GENOTYPE_ALIASES" in sc


# ---------------------------------------------------------------------------
# Tests — File Discovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFileDiscoveryWithExampleData:
    """Test that FileDiscoverer works with the generated synthetic data."""

    def test_discovers_all_animals(self, example_pipeline_env):
        """Pattern discovers files for all animals in the dataset."""
        from neurodent.core.discovery import FileDiscoverer

        ds = example_pipeline_env
        cfg = ds["config"]["analysis"]["war_generation"]

        # Build absolute pattern the same way generate_wars.py does
        base_path = str(ds["data_root"] / ds["session_folder"])
        pattern = f"{base_path}/{cfg['pattern']}"

        discoverer = FileDiscoverer(pattern)
        all_files = discoverer.discover()

        # 2 animals × 2 sessions = at least 4 discovered items
        assert len(all_files) >= 4

        found_animals = {f.metadata["animal"] for f in all_files}
        for animal_id in ds["animals"]:
            assert animal_id in found_animals, f"{animal_id} not discovered"

    def test_discovers_sessions_per_animal(self, example_pipeline_env):
        """Pattern discovers multiple sessions for a single animal."""
        from neurodent.core.discovery import FileDiscoverer

        ds = example_pipeline_env
        cfg = ds["config"]["analysis"]["war_generation"]

        base_path = str(ds["data_root"] / ds["session_folder"])
        pattern = f"{base_path}/{cfg['pattern']}"

        animal_id = ds["animals"][0]
        discoverer = FileDiscoverer(pattern)
        animal_files = discoverer.discover(animal=animal_id)

        sessions = {f.metadata["session"] for f in animal_files}
        assert "day1" in sessions
        assert "day2" in sessions

    def test_filter_by_animal_id(self, example_pipeline_env):
        """Filtering by animal_id returns only that animal's files."""
        from neurodent.core.discovery import FileDiscoverer

        ds = example_pipeline_env
        cfg = ds["config"]["analysis"]["war_generation"]

        base_path = str(ds["data_root"] / ds["session_folder"])
        pattern = f"{base_path}/{cfg['pattern']}"

        discoverer = FileDiscoverer(pattern)
        for animal_id in ds["animals"]:
            filtered = discoverer.discover(animal=animal_id)
            for f in filtered:
                assert f.metadata["animal"] == animal_id


# ---------------------------------------------------------------------------
# Tests — Config Integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSamplesConfigIntegration:
    """Test that the generated samples_config works with neurodent utilities."""

    @pytest.mark.mutates_constants
    def test_inject_config_aliases(self, example_dataset):
        """inject_config_aliases succeeds with the synthetic config."""
        from neurodent.workflow import inject_config_aliases
        from neurodent import constants

        # Save original state to restore after test
        orig_genotype_aliases = constants.GENOTYPE_ALIASES
        orig_animal_metadata = constants.ANIMAL_METADATA

        try:
            sc = example_dataset["samples_config"]
            inject_config_aliases(sc)

            # Verify metadata was injected
            assert "ExWT" in constants.ANIMAL_METADATA
            assert constants.ANIMAL_METADATA["ExWT"]["gene"] == "WT"
            assert "ExKO" in constants.ANIMAL_METADATA
            assert constants.ANIMAL_METADATA["ExKO"]["gene"] == "KO"
        finally:
            # Restore original global state to avoid leaking into other tests
            constants.GENOTYPE_ALIASES = orig_genotype_aliases
            constants.ANIMAL_METADATA = orig_animal_metadata

    def test_samples_config_serializable(self, example_dataset):
        """samples_config can be serialized to JSON (for writing to disk)."""
        sc = example_dataset["samples_config"]
        dumped = json.dumps(sc, indent=2)
        reloaded = json.loads(dumped)
        assert reloaded["ANIMAL_METADATA"] == sc["ANIMAL_METADATA"]


# ---------------------------------------------------------------------------
# Tests — Pipeline Steps (data loading → analysis)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.mutates_constants
class TestPipelineSteps:
    """Test actual pipeline stages end-to-end with synthetic NWB data.

    These tests exercise the real AnimalOrganizer → LongRecordingOrganizer →
    analysis pipeline, not just file discovery.
    """

    def test_animal_organizer_loads_data(self, example_pipeline_env):
        """AnimalOrganizer successfully loads NWB data for a single animal."""
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.visualization import AnimalOrganizer

        ds = example_pipeline_env
        cfg = ds["config"]["analysis"]["war_generation"]

        # Inject metadata so genotype resolution works
        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            inject_config_aliases(ds["samples_config"])

            base_path = str(ds["data_root"] / ds["session_folder"])
            pattern = f"{base_path}/{cfg['pattern']}"
            animal_id = ds["animals"][0]

            ao = AnimalOrganizer(
                pattern,
                animal_id=animal_id,
                skip_sessions=cfg.get("skip_sessions", []),
                assume_from_number=cfg["assume_from_number"],
                lro_kwargs=cfg["lro_kwargs"],
            )

            assert ao.animal_id == animal_id
            assert len(ao.long_recordings) >= 1
            # Verify sessions were created
            assert len(ao.unique_animaldays) >= 1

            # Verify the underlying SI recording was loaded correctly
            for lro in ao.long_recordings:
                assert hasattr(lro, "LongRecording")
                assert lro.LongRecording is not None
                assert lro.LongRecording.get_num_channels() == 8
        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases

    def test_war_generation(self, example_pipeline_env):
        """compute_windowed_analysis produces a WindowAnalysisResult.

        Uses a single-session dataset with an explicit manual_datetimes
        timestamp so that the timeline / fragment-metadata path works.
        """
        from datetime import datetime
        from dateutil.tz import tzlocal
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.visualization import AnimalOrganizer

        ds = example_pipeline_env
        cfg = ds["config"]["analysis"]["war_generation"]

        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            inject_config_aliases(ds["samples_config"])

            base_path = str(ds["data_root"] / ds["session_folder"])
            pattern = f"{base_path}/{cfg['pattern']}"
            animal_id = ds["animals"][0]

            # Provide manual_datetimes so the SI-loaded recording has
            # file_end_datetimes populated (required for WAR fragment timestamps).
            lro_kwargs = dict(cfg["lro_kwargs"])
            lro_kwargs["manual_datetimes"] = datetime(2025, 1, 15, 10, 0, 0, tzinfo=tzlocal())

            ao = AnimalOrganizer(
                pattern,
                animal_id=animal_id,
                skip_sessions=["day2"],  # use only 1 session
                assume_from_number=cfg["assume_from_number"],
                lro_kwargs=lro_kwargs,
            )

            # base_folder_path is normally set during persist_recording in the
            # full pipeline; set it here so compute_windowed_analysis logging works
            for lro in ao.long_recordings:
                if not hasattr(lro, "base_folder_path"):
                    lro.base_folder_path = "synthetic_test"

            war = ao.compute_windowed_analysis(
                ["all"],
                multiprocess_mode="serial",
                window_s=1,  # small windows for tiny data
                apply_notch_filter=False,  # skip filtering for speed
            )

            assert war is not None
            # WAR should have a result DataFrame
            assert hasattr(war, "result") and war.result is not None
        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases


# ---------------------------------------------------------------------------
# Tests — Pipeline Continuation (WAR → Plotters / FDSAR)
# ---------------------------------------------------------------------------


def _build_war(ds):
    """Helper: build a WAR for a single animal from a pipeline environment.

    Shared by continuation tests that start from an already-computed WAR.
    """
    from datetime import datetime
    from dateutil.tz import tzlocal
    from neurodent.visualization import AnimalOrganizer

    cfg = ds["config"]["analysis"]["war_generation"]
    base_path = str(ds["data_root"] / ds["session_folder"])
    pattern = f"{base_path}/{cfg['pattern']}"
    animal_id = ds["animals"][0]

    lro_kwargs = dict(cfg["lro_kwargs"])
    lro_kwargs["manual_datetimes"] = datetime(2025, 1, 15, 10, 0, 0, tzinfo=tzlocal())

    ao = AnimalOrganizer(
        pattern,
        animal_id=animal_id,
        skip_sessions=["day2"],
        assume_from_number=cfg["assume_from_number"],
        lro_kwargs=lro_kwargs,
    )

    for lro in ao.long_recordings:
        if not hasattr(lro, "base_folder_path"):
            lro.base_folder_path = "synthetic_test"

    war = ao.compute_windowed_analysis(
        ["all"],
        multiprocess_mode="serial",
        window_s=1,
        apply_notch_filter=False,
    )
    return ao, war


@pytest.mark.integration
@pytest.mark.mutates_constants
class TestPipelineContinuation:
    """Test downstream pipeline stages that consume a WAR.

    Validates that WAR output can be fed into AnimalPlotter,
    ExperimentPlotter, and that FDSAR generation runs on the
    same synthetic data.
    """

    @pytest.fixture(scope="class")
    def war_env(self, example_pipeline_env):
        """Return (ao, war, ds) with constants injected for the test scope."""
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases

        ds = example_pipeline_env
        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        inject_config_aliases(ds["samples_config"])

        ao, war = _build_war(ds)
        yield ao, war, ds

        constants.ANIMAL_METADATA = orig_metadata
        constants.GENOTYPE_ALIASES = orig_aliases

    def test_animal_plotter_instantiation(self, war_env):
        """AnimalPlotter can be created from the generated WAR."""
        from neurodent.visualization.plotting import AnimalPlotter

        _ao, war, _ds = war_env
        ap = AnimalPlotter(war)

        assert ap.window_result is war
        assert ap.genotype is not None
        assert ap.n_channels == 8

    def test_experiment_plotter_instantiation(self, war_env):
        """ExperimentPlotter can be created from the generated WAR."""
        from neurodent.visualization.plotting import ExperimentPlotter

        _ao, war, _ds = war_env
        ep = ExperimentPlotter(war, features=["all"])

        assert len(ep.results) == 1
        assert ep.results[0] is war
        assert ep.concat_df_wars is not None
        assert not ep.concat_df_wars.empty

    def test_experiment_plotter_multiple_wars(self, example_pipeline_env):
        """ExperimentPlotter accepts WARs from multiple animals."""
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.visualization.plotting import ExperimentPlotter

        ds = example_pipeline_env
        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            inject_config_aliases(ds["samples_config"])
            _ao1, war1 = _build_war(ds)

            # Build a second WAR for the other animal
            from datetime import datetime
            from dateutil.tz import tzlocal
            from neurodent.visualization import AnimalOrganizer

            cfg = ds["config"]["analysis"]["war_generation"]
            base_path = str(ds["data_root"] / ds["session_folder"])
            pattern = f"{base_path}/{cfg['pattern']}"
            lro_kwargs = dict(cfg["lro_kwargs"])
            lro_kwargs["manual_datetimes"] = datetime(2025, 1, 15, 10, 0, 0, tzinfo=tzlocal())

            ao2 = AnimalOrganizer(
                pattern,
                animal_id=ds["animals"][1],
                skip_sessions=["day2"],
                assume_from_number=cfg["assume_from_number"],
                lro_kwargs=lro_kwargs,
            )
            for lro in ao2.long_recordings:
                if not hasattr(lro, "base_folder_path"):
                    lro.base_folder_path = "synthetic_test"

            war2 = ao2.compute_windowed_analysis(
                ["all"],
                multiprocess_mode="serial",
                window_s=1,
                apply_notch_filter=False,
            )

            ep = ExperimentPlotter([war1, war2], features=["all"])
            assert len(ep.results) == 2
            assert not ep.concat_df_wars.empty
        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases

    def test_fdsar_generation(self, war_env):
        """Frequency-domain spike analysis runs on the same AO data."""
        ao, _war, _ds = war_env

        fdsars = ao.compute_frequency_domain_spike_analysis(
            multiprocess_mode="serial",
        )

        assert len(fdsars) >= 1
        for fdsar in fdsars:
            assert fdsar.animal_id == ao.animal_id
            assert fdsar.genotype is not None

    def test_pipeline_war_save_load_round_trip(self, war_env):
        """Pipeline-generated WAR survives the save→load round trip losslessly.

        Closes the gap that the synthetic-fixture round-trip tests in
        tests/test_visualization.py only cover by topology — this exercises
        the EXACT shapes produced by AnimalOrganizer.compute_windowed_analysis
        (real LINEAR_2D / BAND / MATRIX / BANDED_MATRIX / HIST cells), through
        the production save_parquet_and_json / load_parquet_and_json entry
        points used by every snakemake rule.

        Compares cell-by-cell via JSON normalisation (tolerates the
        ndarray→list / tuple→list normalisations the load path performs).
        """
        import json
        import tempfile
        from pathlib import Path

        import numpy as np
        import pandas as pd

        from neurodent.visualization import WindowAnalysisResult

        _ao, war, _ds = war_env

        def _default(obj):
            # Unified handler: numpy types first (json's default= would
            # override _NumpyEncoder.default), then anything else as str.
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return str(obj)

        def _norm(v) -> str:
            if v is None:
                return "null"
            if isinstance(v, float) and np.isnan(v):
                return "nan"
            # Save path normalises tz-aware datetime columns to UTC; the
            # absolute moment is preserved.  Convert any tz-aware Timestamp
            # to UTC (then naive for canonical string form) so equivalence
            # is on the moment in time, not on the tz label.
            if isinstance(v, pd.Timestamp) and v.tz is not None:
                v = v.tz_convert("UTC").tz_localize(None)
            return json.dumps(v, sort_keys=True, default=_default)

        with tempfile.TemporaryDirectory() as tmpdir:
            war.save_parquet_and_json(tmpdir, filename="war")
            loaded = WindowAnalysisResult.load_parquet_and_json(folder_path=Path(tmpdir))

        # Structural equivalence
        assert len(loaded.result) == len(war.result)
        assert set(loaded.result.columns) == set(war.result.columns)
        assert loaded.animal_id == war.animal_id
        assert loaded.genotype == war.genotype
        assert loaded.channel_names == war.channel_names

        # Cell-by-cell normalised equivalence (every row, every column).
        for col in war.result.columns:
            for i in range(len(war.result)):
                assert _norm(war.result[col].iloc[i]) == _norm(loaded.result[col].iloc[i]), (
                    f"col={col!r} row={i}\n  before: {war.result[col].iloc[i]!r}\n  "
                    f"after:  {loaded.result[col].iloc[i]!r}"
                )

    def test_pipeline_war_streaming_equivalent_to_eager(self, war_env):
        """Pipeline-generated WAR: streaming reorder_and_pad matches the eager path.

        Same scope as test_pipeline_war_save_load_round_trip but exercises
        the streaming entry point used by standardize_wars.py in production.
        """
        import json
        import tempfile
        from pathlib import Path

        import numpy as np
        import pandas as pd

        from neurodent.visualization import WindowAnalysisResult

        _ao, war, _ds = war_env

        # WAR generation produces 8 channels (LMot..RVis); reorder is a no-op
        # for matching channels but still exercises every code path.
        target_channels = war.channel_abbrevs or war.channel_names

        def _default(obj):
            # Unified handler: numpy types first (json's default= would
            # override _NumpyEncoder.default), then anything else as str.
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return str(obj)

        def _norm(v) -> str:
            if v is None:
                return "null"
            if isinstance(v, float) and np.isnan(v):
                return "nan"
            # Save path normalises tz-aware datetime columns to UTC; the
            # absolute moment is preserved.  Convert any tz-aware Timestamp
            # to UTC (then naive for canonical string form) so equivalence
            # is on the moment in time, not on the tz label.
            if isinstance(v, pd.Timestamp) and v.tz is not None:
                v = v.tz_convert("UTC").tz_localize(None)
            return json.dumps(v, sort_keys=True, default=_default)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "src"
            src.mkdir()
            war.save_parquet_and_json(src, filename="war")

            # Eager: load, reorder, save.
            eager_dst = tmp / "eager"
            eager_dst.mkdir()
            eager_war = WindowAnalysisResult.load_parquet_and_json(folder_path=src)
            eager_war.reorder_and_pad_channels(target_channels, use_abbrevs=True)
            eager_war.save_parquet_and_json(eager_dst, filename="war")

            # Streaming: do it via the production entry point in one shot.
            stream_dst = tmp / "stream"
            WindowAnalysisResult.stream_reorder_and_pad_to_parquet(
                src_folder=src,
                dst_folder=stream_dst,
                target_channels=target_channels,
                use_abbrevs=True,
                batch_size=4,  # multi-batch on a small WAR
            )

            eager_re = WindowAnalysisResult.load_parquet_and_json(folder_path=eager_dst)
            stream_re = WindowAnalysisResult.load_parquet_and_json(folder_path=stream_dst)

        assert len(eager_re.result) == len(stream_re.result)
        assert set(eager_re.result.columns) == set(stream_re.result.columns)
        assert eager_re.channel_names == stream_re.channel_names

        for col in eager_re.result.columns:
            for i in range(len(eager_re.result)):
                assert _norm(eager_re.result[col].iloc[i]) == _norm(stream_re.result[col].iloc[i]), (
                    f"col={col!r} row={i}\n  eager:  {eager_re.result[col].iloc[i]!r}\n  "
                    f"stream: {stream_re.result[col].iloc[i]!r}"
                )


# ---------------------------------------------------------------------------
# Tests — Dual .bin/.csv format (sox5-style with multi-pattern discovery)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBinCsvMultiPatternDiscovery:
    """Test multi-pattern discovery with paired .bin/.csv files (sox5-style format).

    This validates that ``FileDiscoverer`` correctly groups dual-file
    recordings when given a list of patterns, and that a custom extractor
    can load the paired files through the pipeline.
    """

    @pytest.fixture
    def bin_csv_env(self, tmp_path):
        """Create a dual .bin/.csv dataset under tmp_path."""
        from tests.integration.generate import create_synthetic_bin_csv_dataset

        return create_synthetic_bin_csv_dataset(
            tmp_path, n_sessions=2, duration_s=3,
        )

    def test_discovers_paired_files(self, bin_csv_env):
        """Multi-pattern discovers grouped .bin/.csv pairs."""
        from neurodent.core.discovery import FileDiscoverer

        ds = bin_csv_env
        base_path = str(ds["data_root"] / ds["session_folder"])
        patterns = [f"{base_path}/{p}" for p in ds["pattern"]]

        discoverer = FileDiscoverer(patterns)
        groups = discoverer.discover()

        # 2 animals × 2 sessions = 4 groups
        assert len(groups) == 4
        for g in groups:
            assert g.is_multi_file
            assert len(g.paths) == 2
            assert any(p.endswith("_ColMajor.bin") for p in g.paths)
            assert any(p.endswith("_Meta.csv") for p in g.paths)

    def test_filter_by_animal(self, bin_csv_env):
        """Multi-pattern discovery filters correctly by animal."""
        from neurodent.core.discovery import FileDiscoverer

        ds = bin_csv_env
        base_path = str(ds["data_root"] / ds["session_folder"])
        patterns = [f"{base_path}/{p}" for p in ds["pattern"]]

        discoverer = FileDiscoverer(patterns)
        filtered = discoverer.discover(animal="ExWT")

        # 1 animal × 2 sessions = 2 groups
        assert len(filtered) == 2
        for g in filtered:
            assert g.metadata["animal"] == "ExWT"

    @pytest.mark.mutates_constants
    def test_custom_extractor_via_animal_organizer(self, bin_csv_env):
        """AnimalOrganizer loads paired .bin/.csv files via a custom extract_func.

        This verifies that the pipeline correctly passes multi-file
        ``DiscoveredFile`` objects to a user-defined extractor function
        when using a list of patterns.
        """
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.visualization import AnimalOrganizer

        ds = bin_csv_env
        base_path = str(ds["data_root"] / ds["session_folder"])
        patterns = [f"{base_path}/{p}" for p in ds["pattern"]]

        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            inject_config_aliases(ds["samples_config"])

            ao = AnimalOrganizer(
                patterns,
                animal_id="ExWT",
                assume_from_number=True,
                lro_kwargs={
                    "mode": "si",
                    "extract_func": _bin_csv_extractor,
                    "multiprocess_mode": "serial",
                },
            )

            assert ao.animal_id == "ExWT"
            assert len(ao.long_recordings) >= 1

            for lro in ao.long_recordings:
                rec = lro.LongRecording
                assert rec is not None
                assert rec.get_num_channels() == 8
                assert rec.get_num_samples() == int(3 * 1000)  # 3s @ 1kHz
        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases


# ---------------------------------------------------------------------------
# Per-Animal Pattern Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPerAnimalPatternDict:
    """Test per-animal pattern dict in generate_wars config.

    When ``pattern`` is a dict mapping ``animal_id → pattern(s)``, each animal
    gets its own discovery pattern.  This supports heterogeneous file structures
    across animals in the same dataset.
    """

    @pytest.fixture
    def bin_csv_env(self, tmp_path):
        """Create a dual .bin/.csv dataset under tmp_path."""
        from tests.integration.generate import create_synthetic_bin_csv_dataset

        return create_synthetic_bin_csv_dataset(
            tmp_path, n_sessions=2, duration_s=3,
        )

    def test_per_animal_dict_discovery(self, bin_csv_env):
        """Per-animal pattern dict resolves to correct patterns per animal."""
        from neurodent.core.discovery import FileDiscoverer

        ds = bin_csv_env
        base_path = str(ds["data_root"] / ds["session_folder"])

        # Build a per-animal pattern dict (both animals share the same patterns here,
        # but the dict structure is the key thing being tested)
        pattern_config = {
            "ExWT": ds["pattern"],
            "ExKO": ds["pattern"],
        }

        for animal_id in ["ExWT", "ExKO"]:
            animal_pattern = pattern_config[animal_id]
            if isinstance(animal_pattern, list):
                patterns = [f"{base_path}/{p}" for p in animal_pattern]
            else:
                patterns = f"{base_path}/{animal_pattern}"

            discoverer = FileDiscoverer(patterns)
            filtered = discoverer.discover(animal=animal_id)

            # Each animal has 2 sessions
            assert len(filtered) == 2
            for g in filtered:
                assert g.metadata["animal"] == animal_id

    def test_per_animal_dict_with_heterogeneous_patterns(self, tmp_path):
        """Animals can have different pattern structures (e.g. NWB vs bin/csv)."""
        from neurodent.workflow.utils import resolve_animal_pattern

        # Per-animal pattern dict with heterogeneous patterns
        pattern_config = {
            "A10": "{data_root}/{animal}/{session}/{index}.rhd",
            "B5": [
                "{data_root}/{animal}/{session}/{index}.bin",
                "{data_root}/{animal}/{session}/{index}.csv",
            ],
            "C9": "{data_root}/{animal}/{session}/{index}.rhd",
        }

        # Verify string pattern resolves to string
        result = resolve_animal_pattern(pattern_config, "A10", "/data")
        assert isinstance(result, str)
        assert result == "/data/{animal}/{session}/{index}.rhd"

        # Verify list pattern resolves to list
        result = resolve_animal_pattern(pattern_config, "B5", "/data")
        assert isinstance(result, list)
        assert len(result) == 2

        # Verify another string pattern
        result = resolve_animal_pattern(pattern_config, "C9", "/data")
        assert isinstance(result, str)

    def test_per_animal_dict_missing_animal_raises(self):
        """Accessing a missing animal in the pattern dict raises KeyError."""
        from neurodent.workflow.utils import resolve_animal_pattern

        pattern_config = {
            "A10": "{data_root}/{animal}/{session}/{index}.rhd",
            "B5": ["{data_root}/{animal}/{session}/{index}.bin", "{data_root}/{animal}/{session}/{index}.csv"],
        }

        with pytest.raises(KeyError, match="C9"):
            resolve_animal_pattern(pattern_config, "C9", "/data")

    def test_shared_pattern_with_data_root(self):
        """Shared string/list pattern with {data_root} substitution."""
        from neurodent.workflow.utils import resolve_animal_pattern

        # String pattern
        result = resolve_animal_pattern("{data_root}/{animal}/{index}.nwb", "A10", "/data")
        assert result == "/data/{animal}/{index}.nwb"

        # List pattern
        result = resolve_animal_pattern(
            ["{data_root}/{animal}/{index}.bin", "{data_root}/{animal}/{index}.csv"],
            "A10",
            "/data",
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == "/data/{animal}/{index}.bin"

    def test_pattern_without_data_root_unchanged(self):
        """Patterns without {data_root} are returned as-is."""
        from neurodent.workflow.utils import resolve_animal_pattern

        result = resolve_animal_pattern("{animal}/{index}.nwb", "A10", "/data")
        assert result == "{animal}/{index}.nwb"


# ---------------------------------------------------------------------------
# Mini Real Dataset Tests
# ---------------------------------------------------------------------------

def _mini_real_extractor(discovered_file, **kwargs):
    """Custom extractor that reads paired .bin + .csv files (real mini data).

    The mini real dataset uses ``SampleRate`` (capitalized) in the CSV
    header, which differs from the synthetic ``sampling_rate`` key.
    """
    import csv
    import spikeinterface.core as si_core

    bin_path = [p for p in discovered_file.paths if p.endswith(".bin")][0]
    csv_path = [p for p in discovered_file.paths if p.endswith(".csv")][0]

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_channels = len(rows)
    sampling_rate = float(rows[0]["SampleRate"])
    data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, n_channels)

    return si_core.NumpyRecording(
        traces_list=[data],
        sampling_frequency=sampling_rate,
    )


@pytest.mark.integration
class TestMiniRealDataset:
    """Integration tests using committed mini real bin/csv recordings.

    These tests exercise file discovery and data loading against the small
    real recordings committed in ``.tests/integration/data/``.  They validate
    that the ``mini_real`` dataset config works end-to-end with pattern-based
    discovery including the ``{animal}`` placeholder.
    """

    @pytest.fixture
    def mini_real_config(self):
        """Load mini real dataset configuration."""
        import yaml
        from neurodent.workflow.utils import expand_animals_config

        config_path = Path(__file__).resolve().parents[2] / "config" / "datasets" / "mini_real.yaml"
        with open(config_path) as f:
            ds_config = yaml.safe_load(f)

        samples_path = Path(__file__).resolve().parents[2] / "config" / "samples_mini_real.json"
        with open(samples_path) as f:
            samples_config = json.load(f)

        # Expand unified animals config if present
        samples_config = expand_animals_config(samples_config)

        return {
            "ds_config": ds_config,
            "samples_config": samples_config,
            "data_root": Path(__file__).resolve().parents[2] / samples_config["data_root"],
        }

    def test_mini_real_data_files_exist(self, mini_real_config):
        """Verify that the committed mini real data files are present."""
        data_dir = mini_real_config["data_root"]

        for animal_dir in ["A10", "F22"]:
            animal_data_dir = data_dir / animal_dir
            assert animal_data_dir.is_dir(), f"Missing animal directory: {animal_data_dir}"
            bin_files = list(animal_data_dir.glob("*_ColMajor.bin"))
            csv_files = list(animal_data_dir.glob("*_Meta.csv"))
            assert len(bin_files) >= 1, f"No .bin files in {animal_data_dir}"
            assert len(csv_files) >= 1, f"No .csv files in {animal_data_dir}"

    def test_mini_real_discovery_with_animal_placeholder(self, mini_real_config):
        """FileDiscoverer finds mini real files using {animal}/{index} pattern."""
        from neurodent.core.discovery import FileDiscoverer
        from neurodent.workflow.utils import resolve_animal_pattern

        cfg = mini_real_config
        ds = cfg["ds_config"]

        # Build absolute patterns the same way generate_wars.py does
        patterns = resolve_animal_pattern(
            ds["analysis"]["war_generation"]["pattern"],
            "",
            data_root=str(cfg["data_root"]),
        )

        discoverer = FileDiscoverer(patterns)
        groups = discoverer.discover()

        # Should find 2 groups (one per animal)
        assert len(groups) == 2
        for g in groups:
            assert g.is_multi_file
            assert len(g.paths) == 2
            assert any(p.endswith("_ColMajor.bin") for p in g.paths)
            assert any(p.endswith("_Meta.csv") for p in g.paths)

    def test_mini_real_filter_by_animal(self, mini_real_config):
        """FileDiscoverer correctly filters by animal_id."""
        from neurodent.core.discovery import FileDiscoverer
        from neurodent.workflow.utils import resolve_animal_pattern

        cfg = mini_real_config
        ds = cfg["ds_config"]
        patterns = resolve_animal_pattern(
            ds["analysis"]["war_generation"]["pattern"],
            "",
            data_root=str(cfg["data_root"]),
        )

        discoverer = FileDiscoverer(patterns)

        for animal_id in ["A10", "F22"]:
            filtered = discoverer.discover(animal=animal_id)
            assert len(filtered) == 1, f"Expected 1 group for {animal_id}, got {len(filtered)}"
            assert filtered[0].metadata["animal"] == animal_id

    @pytest.mark.mutates_constants
    def test_mini_real_animal_organizer_loads_data(self, mini_real_config):
        """AnimalOrganizer loads mini real bin/csv data via custom extractor."""
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.workflow.utils import resolve_animal_pattern
        from neurodent.visualization import AnimalOrganizer

        cfg = mini_real_config
        ds = cfg["ds_config"]
        patterns = resolve_animal_pattern(
            ds["analysis"]["war_generation"]["pattern"],
            "A10",
            data_root=str(cfg["data_root"]),
        )

        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            inject_config_aliases(cfg["samples_config"])

            ao = AnimalOrganizer(
                patterns,
                animal_id="A10",
                assume_from_number=ds["analysis"]["war_generation"]["assume_from_number"],
                lro_kwargs={
                    "mode": "si",
                    "extract_func": _mini_real_extractor,
                    "multiprocess_mode": "serial",
                },
            )

            assert ao.animal_id == "A10"
            assert len(ao.long_recordings) >= 1

            for lro in ao.long_recordings:
                rec = lro.LongRecording
                assert rec is not None
                assert rec.get_num_channels() == 10  # 10 channels in real data
                assert rec.get_sampling_frequency() == 1000.0
                assert rec.get_total_duration() > 0
        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases

    @pytest.mark.mutates_constants
    def test_mini_real_both_animals_loadable(self, mini_real_config):
        """Both animals (A10, F22) can be loaded from mini real data."""
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.workflow.utils import resolve_animal_pattern
        from neurodent.visualization import AnimalOrganizer

        cfg = mini_real_config
        ds = cfg["ds_config"]

        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            inject_config_aliases(cfg["samples_config"])

            for animal_id in ["A10", "F22"]:
                patterns = resolve_animal_pattern(
                    ds["analysis"]["war_generation"]["pattern"],
                    animal_id,
                    data_root=str(cfg["data_root"]),
                )
                ao = AnimalOrganizer(
                    patterns,
                    animal_id=animal_id,
                    assume_from_number=True,
                    lro_kwargs={
                        "mode": "si",
                        "extract_func": _mini_real_extractor,
                        "multiprocess_mode": "serial",
                    },
                )

                assert ao.animal_id == animal_id
                assert len(ao.long_recordings) == 1
                assert ao.long_recordings[0].LongRecording is not None
        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases

    @pytest.mark.mutates_constants
    def test_mini_real_loads_via_dotted_extract_func(self, mini_real_config):
        """AnimalOrganizer resolves a dotted extract_func string from config."""
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.workflow.utils import resolve_animal_pattern
        from neurodent.visualization import AnimalOrganizer

        cfg = mini_real_config
        ds = cfg["ds_config"]
        patterns = resolve_animal_pattern(
            ds["analysis"]["war_generation"]["pattern"],
            "A10",
            data_root=str(cfg["data_root"]),
        )

        # Use the extract_func string from the dataset config
        lro_kwargs = dict(ds["analysis"]["war_generation"]["lro_kwargs"])

        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            inject_config_aliases(cfg["samples_config"])

            ao = AnimalOrganizer(
                patterns,
                animal_id="A10",
                assume_from_number=ds["analysis"]["war_generation"]["assume_from_number"],
                lro_kwargs=lro_kwargs,
            )

            assert ao.animal_id == "A10"
            assert len(ao.long_recordings) >= 1
            rec = ao.long_recordings[0].LongRecording
            assert rec is not None
            assert rec.get_num_channels() == 10
            assert rec.get_sampling_frequency() == 1000.0
        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases


@pytest.mark.integration
class TestJointRecordingSplit:
    """Test joint recording split functionality where animals share a recording file."""

    def test_joint_recording_split(self):
        """Test splitting A10's 10 channels into two virtual animals (A10-1 and A10-2)."""
        from neurodent import constants
        from neurodent.workflow import inject_config_aliases
        from neurodent.workflow.utils import expand_animals_config
        from neurodent.visualization import AnimalOrganizer

        # Use existing A10 test data but configure it as a joint recording split
        # A10-1 gets channels 0-4, A10-2 gets channels 5-9
        mini_real_data = Path(__file__).resolve().parents[2] / ".tests" / "integration" / "data"
        config_dict = {
            "data_root": str(mini_real_data),
            "animals": [
                {
                    "id": "A10-1",
                    "sex": "M",
                    "gene": "WT",
                    "channels": ["0", "1", "2", "3", "4"],
                    "group": "A10",  # Both animals share the A10 group for discovery
                    "manual_datetime": "2025-05-10 10:00:00",
                },
                {
                    "id": "A10-2",
                    "sex": "M",
                    "gene": "WT",
                    "channels": ["5", "6", "7", "8", "9"],
                    "group": "A10",  # Same group, different channels
                    "manual_datetime": "2025-05-10 10:00:00",
                },
            ],
        }

        orig_metadata = constants.ANIMAL_METADATA
        orig_aliases = constants.GENOTYPE_ALIASES
        try:
            # Test config expansion
            expanded = expand_animals_config(config_dict)

            # Inject the config so metadata is available
            inject_config_aliases(expanded)

            # Verify both animals are in the expanded config
            animal_ids = {a["id"] for a in expanded["animals"]}
            assert "A10-1" in animal_ids
            assert "A10-2" in animal_ids

            # Verify group mapping
            assert expanded["_animal_groups"]["A10-1"] == "A10"
            assert expanded["_animal_groups"]["A10-2"] == "A10"

            # Verify channel mapping
            assert expanded["_animal_channels"]["A10-1"] == ["0", "1", "2", "3", "4"]
            assert expanded["_animal_channels"]["A10-2"] == ["5", "6", "7", "8", "9"]

            # Test channel validation - no overlaps within group
            channels_a10_1 = set(expanded["_animal_channels"]["A10-1"])
            channels_a10_2 = set(expanded["_animal_channels"]["A10-2"])
            assert len(channels_a10_1 & channels_a10_2) == 0, "Channels should not overlap within same group"

            # Test AnimalOrganizer.split() functionality with real mini-real data
            # Use the group name "A10" in the pattern to discover files for both animals
            # The pattern matches "Cage 2 A10-0" as the {index} placeholder
            patterns = [
                f"{mini_real_data}/A10/{{index}}_ColMajor.bin",
                f"{mini_real_data}/A10/{{index}}_Meta.csv",
            ]

            # Test A10-1 (first 5 channels)
            ao1 = AnimalOrganizer(
                patterns,
                animal_id="A10",  # Use the group name for discovery
                assume_from_number=True,
                lro_kwargs={
                    "mode": "si",
                    "extract_func": _mini_real_extractor,
                    "multiprocess_mode": "serial",
                    "manual_datetimes": "2025-05-10 10:00:00",
                },
            )
            assert ao1.animal_id == "A10"

            # Split into two animals with different channel subsets
            split_groups = {
                "A10-1": ["0", "1", "2", "3", "4"],
                "A10-2": ["5", "6", "7", "8", "9"],
            }
            split_aos = ao1.split(split_groups)

            # Verify A10-1 split
            assert "A10-1" in split_aos
            ao1_split = split_aos["A10-1"]
            assert ao1_split.animal_id == "A10-1"
            assert len(ao1_split.long_recordings) >= 1
            rec1 = ao1_split.long_recordings[0].LongRecording
            assert rec1 is not None
            assert rec1.get_num_channels() == 5  # Only 5 channels after split
            assert rec1.get_sampling_frequency() == 1000.0

            # Verify channel IDs match what we requested
            channel_ids_1 = rec1.get_channel_ids()
            assert len(channel_ids_1) == 5
            expected_channels_1 = {"0", "1", "2", "3", "4"}
            assert set(channel_ids_1) == expected_channels_1

            # Verify A10-2 split
            assert "A10-2" in split_aos
            ao2_split = split_aos["A10-2"]
            assert ao2_split.animal_id == "A10-2"
            rec2 = ao2_split.long_recordings[0].LongRecording
            assert rec2 is not None
            assert rec2.get_num_channels() == 5
            channel_ids_2 = rec2.get_channel_ids()
            expected_channels_2 = {"5", "6", "7", "8", "9"}
            assert set(channel_ids_2) == expected_channels_2

        finally:
            constants.ANIMAL_METADATA = orig_metadata
            constants.GENOTYPE_ALIASES = orig_aliases



    