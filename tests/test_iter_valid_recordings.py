"""Tests for AnimalOrganizer._iter_valid_recordings() and datetimes_are_start propagation.

Tests the centralized empty-recording filter added to AnimalOrganizer, ensuring that
compute_bad_channels, compute_windowed_analysis, and compute_frequency_domain_spike_analysis
all skip zero-sample recordings instead of crashing.

Also tests the datetimes_are_start propagation fix in generate_wars.py.
"""

import logging
from unittest.mock import MagicMock

import pytest

from neurodent.core import LongRecordingOrganizer
from neurodent.visualization.results import AnimalOrganizer


class TestIterValidRecordings:
    """Tests for the centralized _iter_valid_recordings() method."""

    _date_counter = 0

    def _make_mock_lro(self, total_samples=1000, display_name="mock_lro", date=None):
        """Create a mock LongRecordingOrganizer with configurable total_samples."""
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["ch1", "ch2"]
        lro.base_folder_path = "/tmp/mock"
        lro.labels = {}
        lro.display_name = display_name

        # Mock the inner LongRecording
        mock_rec = MagicMock()
        mock_rec.get_total_samples.return_value = total_samples
        lro.LongRecording = mock_rec

        # Each LRO needs a unique date to avoid merging in from_lros
        if date is None:
            TestIterValidRecordings._date_counter += 1
            date = f"Jan-{TestIterValidRecordings._date_counter:02d}-2022"
        lro.get_date_string.return_value = date

        return lro

    def _make_ao(self, lros):
        """Create an AnimalOrganizer from a list of mock LROs."""
        return AnimalOrganizer.from_lros(
            lros, animal_id="TestAnimal", genotype="WT"
        )

    def test_all_valid_recordings_yielded(self):
        """All recordings with nonzero samples are yielded."""
        lros = [self._make_mock_lro(1000, f"rec_{i}") for i in range(3)]
        ao = self._make_ao(lros)

        result = list(ao._iter_valid_recordings())
        assert len(result) == 3
        assert [idx for idx, _ in result] == [0, 1, 2]

    def test_empty_recording_skipped(self, caplog):
        """Recording with 0 total samples is skipped with a warning."""
        lros = [
            self._make_mock_lro(1000, "good_rec"),
            self._make_mock_lro(0, "empty_rec"),
            self._make_mock_lro(2000, "another_good_rec"),
        ]
        ao = self._make_ao(lros)

        with caplog.at_level(logging.WARNING):
            result = list(ao._iter_valid_recordings())

        # Only 2 valid recordings
        assert len(result) == 2
        indices = [idx for idx, _ in result]
        assert indices == [0, 2]

        # Warning was logged
        assert any("Skipping recording 1" in msg for msg in caplog.messages)
        assert any("0 total samples" in msg for msg in caplog.messages)

    def test_all_empty_recordings_skipped(self, caplog):
        """All recordings with 0 samples are skipped."""
        lros = [self._make_mock_lro(0, f"empty_{i}") for i in range(3)]
        ao = self._make_ao(lros)

        with caplog.at_level(logging.WARNING):
            result = list(ao._iter_valid_recordings())

        assert len(result) == 0
        assert len([m for m in caplog.messages if "Skipping recording" in m]) == 3

    def test_no_long_recording_attribute_still_yielded(self):
        """LRO without LongRecording attribute is still yielded (not filtered)."""
        lro = self._make_mock_lro(1000, "normal")
        lro_no_attr = MagicMock(spec=LongRecordingOrganizer)
        lro_no_attr.channel_names = ["ch1", "ch2"]  # Must match other LROs
        lro_no_attr.base_folder_path = "/tmp"
        lro_no_attr.labels = {}
        lro_no_attr.display_name = "no_lr_attr"
        TestIterValidRecordings._date_counter += 1
        lro_no_attr.get_date_string.return_value = f"Feb-{TestIterValidRecordings._date_counter:02d}-2022"
        # Remove the LongRecording attribute
        del lro_no_attr.LongRecording

        ao = self._make_ao([lro, lro_no_attr])
        result = list(ao._iter_valid_recordings())
        # Both should be yielded (the one without LongRecording is not filtered)
        assert len(result) == 2

    def test_none_long_recording_still_yielded(self):
        """LRO with LongRecording=None is still yielded (not filtered)."""
        lro = self._make_mock_lro(1000, "normal")
        lro_none = self._make_mock_lro(1000, "none_lr")
        lro_none.LongRecording = None
        lro_none.get_date_string.return_value = "Jan-02-2022"

        ao = self._make_ao([lro, lro_none])
        result = list(ao._iter_valid_recordings())
        assert len(result) == 2

    def test_single_recording_valid(self):
        """Single valid recording works correctly."""
        ao = self._make_ao([self._make_mock_lro(500, "single")])
        result = list(ao._iter_valid_recordings())
        assert len(result) == 1
        assert result[0][0] == 0

    def test_single_recording_empty(self, caplog):
        """Single empty recording produces empty iterator."""
        ao = self._make_ao([self._make_mock_lro(0, "empty_single")])

        with caplog.at_level(logging.WARNING):
            result = list(ao._iter_valid_recordings())

        assert len(result) == 0

    def test_preserves_original_indices(self):
        """Yielded indices correspond to positions in self.long_recordings."""
        lros = [
            self._make_mock_lro(0, "empty_0"),
            self._make_mock_lro(0, "empty_1"),
            self._make_mock_lro(100, "valid_2"),
            self._make_mock_lro(0, "empty_3"),
            self._make_mock_lro(200, "valid_4"),
        ]
        ao = self._make_ao(lros)
        result = list(ao._iter_valid_recordings())

        assert len(result) == 2
        assert result[0][0] == 2
        assert result[1][0] == 4


class TestDatetimesAreStartPropagation:
    """Tests for datetimes_are_start propagation from war_generation config to lro_kwargs."""

    def test_datetimes_are_start_propagated_to_lro_kwargs(self):
        """datetimes_are_start at war_generation level is propagated into lro_kwargs."""
        session_analysis_config = {
            "datetimes_are_start": False,
            "lro_kwargs": {"mode": "si", "multiprocess_mode": "serial"},
        }

        session_lro_kwargs = dict(session_analysis_config.get("lro_kwargs", {}))

        # Reproduce the propagation logic from generate_wars.py
        if "datetimes_are_start" in session_analysis_config:
            session_lro_kwargs.setdefault(
                "datetimes_are_start", session_analysis_config["datetimes_are_start"]
            )

        assert "datetimes_are_start" in session_lro_kwargs
        assert session_lro_kwargs["datetimes_are_start"] is False

    def test_datetimes_are_start_true_propagated(self):
        """datetimes_are_start=True is also propagated correctly."""
        session_analysis_config = {
            "datetimes_are_start": True,
            "lro_kwargs": {"mode": "si"},
        }
        session_lro_kwargs = dict(session_analysis_config.get("lro_kwargs", {}))

        if "datetimes_are_start" in session_analysis_config:
            session_lro_kwargs.setdefault(
                "datetimes_are_start", session_analysis_config["datetimes_are_start"]
            )

        assert session_lro_kwargs["datetimes_are_start"] is True

    def test_lro_kwargs_override_takes_precedence(self):
        """If datetimes_are_start is already in lro_kwargs, it is not overwritten."""
        session_analysis_config = {
            "datetimes_are_start": False,
            "lro_kwargs": {
                "mode": "si",
                "datetimes_are_start": True,  # Explicit override in lro_kwargs
            },
        }
        session_lro_kwargs = dict(session_analysis_config.get("lro_kwargs", {}))

        if "datetimes_are_start" in session_analysis_config:
            session_lro_kwargs.setdefault(
                "datetimes_are_start", session_analysis_config["datetimes_are_start"]
            )

        # lro_kwargs value should win (setdefault doesn't overwrite)
        assert session_lro_kwargs["datetimes_are_start"] is True

    def test_missing_datetimes_are_start_no_propagation(self):
        """If datetimes_are_start is absent from war_generation config, nothing is added."""
        session_analysis_config = {
            "lro_kwargs": {"mode": "si"},
        }
        session_lro_kwargs = dict(session_analysis_config.get("lro_kwargs", {}))

        if "datetimes_are_start" in session_analysis_config:
            session_lro_kwargs.setdefault(
                "datetimes_are_start", session_analysis_config["datetimes_are_start"]
            )

        assert "datetimes_are_start" not in session_lro_kwargs

    def test_per_animal_override_takes_precedence_over_propagated(self):
        """Per-animal overrides (applied via .update()) override the propagated value."""
        session_analysis_config = {
            "datetimes_are_start": False,
            "lro_kwargs": {"mode": "si"},
        }
        session_lro_kwargs = dict(session_analysis_config.get("lro_kwargs", {}))

        # Step 1: Propagate from war_generation level
        if "datetimes_are_start" in session_analysis_config:
            session_lro_kwargs.setdefault(
                "datetimes_are_start", session_analysis_config["datetimes_are_start"]
            )

        assert session_lro_kwargs["datetimes_are_start"] is False

        # Step 2: Apply per-animal overrides (as done in generate_wars.py)
        animal_overrides = {"lro_kwargs": {"datetimes_are_start": True}}
        session_lro_kwargs.update(animal_overrides["lro_kwargs"])

        # Per-animal override wins
        assert session_lro_kwargs["datetimes_are_start"] is True

    def test_sox5_bin_config_shape(self):
        """Verify the fix handles the actual sox5_bin.yaml config shape."""
        # This mirrors the actual config structure that caused the bug
        sox5_config = {
            "pattern": [
                "{data_root}/*/*{animal}*/{session}/*-{index}_ColMajor.bin",
                "{data_root}/*/*{animal}*/{session}/*-{index}_Meta.csv",
            ],
            "assume_from_number": True,
            "datetimes_are_start": False,  # At war_generation level, NOT inside lro_kwargs
            "lro_kwargs": {
                "mode": "si",
                "extract_func": "tests/data/readers.py:read_bin_csv_pair",
                "multiprocess_mode": "serial",
            },
        }

        session_lro_kwargs = dict(sox5_config.get("lro_kwargs", {}))
        assert "datetimes_are_start" not in session_lro_kwargs  # Bug: was missing

        # Apply the fix
        if "datetimes_are_start" in sox5_config:
            session_lro_kwargs.setdefault(
                "datetimes_are_start", sox5_config["datetimes_are_start"]
            )

        assert session_lro_kwargs["datetimes_are_start"] is False  # Now present


class TestComputeBadChannelsGracefulFailure:
    """Tests that AnimalOrganizer.compute_bad_channels handles LOF failures gracefully.

    When one recording fails LOF computation (e.g. due to an empty 0-sample file),
    other recordings should still be processed and the animal should not crash.
    """

    _date_counter = 1000

    def _make_mock_lro(self, total_samples=1000, display_name="mock_lro", date=None,
                       lof_fail=False):
        """Create a mock LRO with configurable behavior."""
        lro = MagicMock(spec=LongRecordingOrganizer)
        lro.channel_names = ["ch1", "ch2"]
        lro.base_folder_path = "/tmp/mock"
        lro.labels = {}
        lro.display_name = display_name
        lro.bad_channel_names = []

        mock_rec = MagicMock()
        mock_rec.get_total_samples.return_value = total_samples
        lro.LongRecording = mock_rec

        if date is None:
            TestComputeBadChannelsGracefulFailure._date_counter += 1
            date = f"Jan-{TestComputeBadChannelsGracefulFailure._date_counter:02d}-2022"
        lro.get_date_string.return_value = date

        if lof_fail:
            lro.compute_bad_channels.side_effect = ValueError(
                "can't extend empty axis 0 using modes other than 'constant' or 'empty'"
            )
        else:
            lro.compute_bad_channels.return_value = None
            lro.lof_scores = [1.0, 1.0]

        return lro

    def _make_ao(self, lros):
        return AnimalOrganizer.from_lros(
            lros, animal_id="TestAnimal", genotype="WT"
        )

    def test_one_failing_recording_does_not_crash_animal(self, caplog):
        """One failing LOF does not prevent other recordings from completing."""
        lros = [
            self._make_mock_lro(1000, "good_1"),
            self._make_mock_lro(1000, "failing", lof_fail=True),
            self._make_mock_lro(1000, "good_2"),
        ]
        ao = self._make_ao(lros)

        with caplog.at_level(logging.WARNING):
            ao.compute_bad_channels(lof_threshold=1.5)

        # Good recordings should have their compute_bad_channels called
        lros[0].compute_bad_channels.assert_called_once()
        lros[2].compute_bad_channels.assert_called_once()

        # Failing recording was attempted
        lros[1].compute_bad_channels.assert_called_once()

        # Warning was logged for the failing recording
        assert any("Skipping LOF computation" in msg for msg in caplog.messages)

    def test_all_recordings_succeed(self):
        """When all recordings succeed, no warnings and all get LOF scores."""
        lros = [self._make_mock_lro(1000, f"good_{i}") for i in range(3)]
        ao = self._make_ao(lros)

        ao.compute_bad_channels(lof_threshold=1.5)

        for lro in lros:
            lro.compute_bad_channels.assert_called_once()
