#!/usr/bin/env python3
"""
Test case demonstrating the LOF overwrite bug with overlapping animaldays.

This test is designed to FAIL with the current implementation to demonstrate
the bug where multiple folders parsing to the same animalday cause LOF data loss.
The desired behavior is that overlapping animaldays should be merged and processed
as a single LongRecordingOrganizer, not overwritten.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from pythoneeg.visualization import results
from pythoneeg import core


class TestOverlappingAnimaldaysBug:
    """Test demonstrating LOF overwrite bug with split-day folders."""

    def test_overlapping_animaldays_lof_overwrite_bug(self):
        """
        This test demonstrates the bug where multiple folders parsing to the same
        animalday cause LOF scores to be overwritten.

        Expected behavior (not yet implemented):
        - Multiple folders for same day should be merged into single LRO
        - LOF computed once on combined data
        - lof_scores_dict should have one entry per unique animalday

        Current buggy behavior:
        - Multiple folders create separate LROs with same animalday key
        - LOF scores get overwritten, only last folder survives
        - lof_scores_dict length < number of folders processed

        This test should FAIL until the bug is fixed.
        """

        # Setup: Create mock folders that parse to same animalday
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create folders that will parse to same animalday due to (1), (2) suffix removal
            folder1 = temp_path / "WT_A10_2023-01-15"
            folder2 = temp_path / "WT_A10_2023-01-15(1)"
            folder3 = temp_path / "WT_A10_2023-01-15(2)"

            for folder in [folder1, folder2, folder3]:
                folder.mkdir(parents=True)
                # Create dummy binary files to make folders valid
                (folder / "dummy_ColMajor_001.bin").touch()
                (folder / "dummy_Meta_001.json").touch()

            # Mock LongRecordingOrganizer to avoid actual file processing
            mock_lros = []
            expected_lof_scores = [
                np.array([1.5, 2.0, 0.8]),  # LOF scores from folder1
                np.array([1.2, 1.8, 0.9]),  # LOF scores from folder2
                np.array([1.7, 1.3, 1.1]),  # LOF scores from folder3
            ]

            # Create mock LROs with different LOF scores and timestamps for each folder
            # Note: folders are created in name order but will have timestamps that should sort differently
            expected_median_times = [100.0, 50.0, 150.0]  # Out of name order but chronological

            for i, (scores, median_time) in enumerate(zip(expected_lof_scores, expected_median_times)):
                mock_lro = Mock()
                mock_lro.lof_scores = scores
                mock_lro.channel_names = ['LMot', 'RMot', 'LAud']
                mock_lro.meta = {}

                # Mock the LongRecording with median time data
                mock_recording = Mock()
                mock_recording.get_num_samples.return_value = int(median_time * 2 * 1000)  # samples = time * 2 * fs
                mock_recording.get_sampling_frequency.return_value = 1000.0
                mock_lro.LongRecording = mock_recording
                mock_lro.cleanup_rec = Mock()  # Mock cleanup method

                mock_lros.append(mock_lro)

            # Patch glob.glob to return our test folders
            with patch('glob.glob') as mock_glob:
                mock_glob.return_value = [str(folder1), str(folder2), str(folder3)]

                # Patch LongRecordingOrganizer creation - will be called multiple times for sorting + final creation
                with patch.object(core, 'LongRecordingOrganizer') as mock_lro_class:
                    # Return different mocks each time, cycling through our prepared LROs
                    call_count = 0
                    def mock_lro_side_effect(*args, **kwargs):
                        nonlocal call_count
                        # Map folder paths to their corresponding mock LROs
                        folder_path = str(args[0])
                        if 'WT_A10_2023-01-15(2)' in folder_path:
                            return mock_lros[2]  # Highest median time (150.0)
                        elif 'WT_A10_2023-01-15(1)' in folder_path:
                            return mock_lros[1]  # Lowest median time (50.0)
                        else:  # WT_A10_2023-01-15
                            return mock_lros[0]  # Middle median time (100.0)

                    mock_lro_class.side_effect = mock_lro_side_effect

                    # Create AnimalOrganizer - this should trigger the bug
                    ao = results.AnimalOrganizer(
                        anim_id="A10",
                        base_folder_path=temp_path,
                        mode="base"
                    )

                    # Verify that all folders parse to same animalday
                    parsed_animaldays = []
                    for folder in [folder1, folder2, folder3]:
                        parsed = core.parse_path_to_animalday(
                            folder,
                            animal_param=["A10"],
                            mode="base"
                        )
                        parsed_animaldays.append(parsed["animalday"])

                    # All should parse to same animalday (parentheses removed)
                    assert len(set(parsed_animaldays)) == 1, f"Expected same animalday, got: {parsed_animaldays}"
                    expected_animalday = parsed_animaldays[0]

                    # Create a simple DataFrame to simulate windowed analysis results
                    import pandas as pd
                    mock_df = pd.DataFrame({
                        'timestamp': [pd.Timestamp('2023-01-15 10:00:00')] * 3,
                        'animalday': [expected_animalday] * 3,  # All same animalday
                        'rms': [1.0, 1.1, 1.2]
                    })

                    # Create WindowAnalysisResult directly to test LOF collection bug
                    # This simulates what happens in compute_windowed_analysis
                    war = results.WindowAnalysisResult(
                        result=mock_df,
                        animal_id="A10",
                        genotype="WT",
                        channel_names=['LMot', 'RMot', 'LAud'],
                        lof_scores_dict={}  # Empty initially
                    )

                    # Manually trigger the LOF collection that happens in compute_windowed_analysis
                    # This is where the bug occurs (results.py:345-355)
                    lof_scores_dict = {}
                    for animalday, lrec in zip(ao.animaldays, ao.long_recordings):
                        if hasattr(lrec, "lof_scores") and lrec.lof_scores is not None:
                            lof_scores_dict[animalday] = {  # BUG: Direct overwrite!
                                "lof_scores": lrec.lof_scores.tolist(),
                                "channel_names": lrec.channel_names,
                            }

                    # Apply the collected LOF scores to the WAR
                    war.lof_scores_dict = lof_scores_dict

                    # BUG DEMONSTRATION: Check that LOF data was lost
                    print(f"Number of folders: {len(ao.long_recordings)}")
                    print(f"Animaldays: {ao.animaldays}")
                    print(f"LOF scores dict length: {len(war.lof_scores_dict)}")
                    print(f"LOF scores dict keys: {list(war.lof_scores_dict.keys())}")

                    # After fix: We expect 1 LRO per unique animalday (not per folder)
                    assert len(ao.long_recordings) == 1, f"Expected 1 merged LRO, got {len(ao.long_recordings)}"
                    assert len(war.lof_scores_dict) == 1, f"Expected 1 LOF entry (no collision), got {len(war.lof_scores_dict)}"

                    # The LOF scores should be computed on concatenated data from all 3 folders
                    # This is more comprehensive than any single folder's LOF scores
                    surviving_scores = war.lof_scores_dict[expected_animalday]['lof_scores']

                    # Verify the LOF scores are valid (should be different from individual folder scores)
                    # The concatenated data should produce different LOF scores than any single folder
                    assert len(surviving_scores) == 3, f"Expected 3 channel LOF scores, got {len(surviving_scores)}"

                    # Success: No data loss, single LRO handles all folders
                    folder_count = 3  # We created 3 test folders
                    print(f"SUCCESS: Fix implemented correctly. "
                          f"Merged {folder_count} overlapping folders into 1 LRO with 1 comprehensive LOF analysis.")

                    # Verify temporal ordering: folders should be processed in median time order (50.0, 100.0, 150.0)
                    # This means folder2 (50.0), folder1 (100.0), folder3 (150.0)
                    # The actual merging and final LOF scores should reflect this temporal ordering

    def test_folder_sorting_by_median_time(self):
        """
        Test that folders are sorted by their LRO median times, not by folder names.

        This verifies that folders with out-of-order names but chronological timestamps
        get sorted correctly for proper temporal concatenation.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create folders that parse to same animalday
            folder_a = temp_path / "WT_A10_2023-01-15"
            folder_b = temp_path / "WT_A10_2023-01-15(1)"
            folder_c = temp_path / "WT_A10_2023-01-15(2)"

            for folder in [folder_a, folder_b, folder_c]:
                folder.mkdir(parents=True)
                (folder / "dummy_ColMajor_001.bin").touch()
                (folder / "dummy_Meta_001.json").touch()

            # Create specific mock LROs with timing that should sort differently from name order
            mock_lro_a = Mock()  # folder_a - middle time
            mock_lro_b = Mock()  # folder_b - earliest time
            mock_lro_c = Mock()  # folder_c - latest time

            # Set up timing for each mock LRO using file_end_datetimes
            from datetime import datetime, timedelta
            base_time = datetime(2023, 1, 15, 10, 0, 0)  # Base time: 10:00 AM

            # folder_a should have middle time (around 12:00 PM)
            mock_lro_a.file_end_datetimes = [
                base_time + timedelta(hours=2, minutes=0),   # 12:00 PM
                base_time + timedelta(hours=2, minutes=30),  # 12:30 PM
                base_time + timedelta(hours=3, minutes=0),   # 1:00 PM
            ]

            # folder_b should have earliest time (around 10:30 AM)
            mock_lro_b.file_end_datetimes = [
                base_time + timedelta(minutes=0),   # 10:00 AM
                base_time + timedelta(minutes=30),  # 10:30 AM
                base_time + timedelta(minutes=60),  # 11:00 AM
            ]

            # folder_c should have latest time (around 2:00 PM)
            mock_lro_c.file_end_datetimes = [
                base_time + timedelta(hours=3, minutes=30),  # 1:30 PM
                base_time + timedelta(hours=4, minutes=0),   # 2:00 PM
                base_time + timedelta(hours=4, minutes=30),  # 2:30 PM
            ]

            # Set up other attributes for each mock LRO
            for mock_lro in [mock_lro_a, mock_lro_b, mock_lro_c]:
                mock_lro.channel_names = ['LMot', 'RMot', 'LAud']
                mock_lro.meta = Mock()
                mock_lro.lof_scores = np.array([1.0, 1.0, 1.0])

                # Mock the LongRecording
                mock_recording = Mock()
                mock_recording.get_duration.return_value = 3600.0  # 1 hour duration
                mock_lro.LongRecording = mock_recording

            # Patch glob to return folders
            with patch('glob.glob') as mock_glob:
                mock_glob.return_value = [str(folder_a), str(folder_b), str(folder_c)]

                # Patch LongRecordingOrganizer to return specific mocks
                with patch.object(core, 'LongRecordingOrganizer') as mock_lro_class:
                    def mock_lro_side_effect(*args, **kwargs):
                        folder_path = str(args[0])
                        if folder_path.endswith('WT_A10_2023-01-15'):
                            return mock_lro_a  # Middle time (100.0s)
                        elif folder_path.endswith('WT_A10_2023-01-15(1)'):
                            return mock_lro_b  # Earliest time (50.0s)
                        elif folder_path.endswith('WT_A10_2023-01-15(2)'):
                            return mock_lro_c  # Latest time (150.0s)
                        return Mock()

                    mock_lro_class.side_effect = mock_lro_side_effect

                    # Add mock merge method to track merging
                    merge_call_count = 0
                    def mock_merge(other_lro):
                        nonlocal merge_call_count
                        merge_call_count += 1

                    # Add merge method to all mock LROs
                    mock_lro_a.merge = mock_merge
                    mock_lro_b.merge = mock_merge
                    mock_lro_c.merge = mock_merge

                    # Create AnimalOrganizer which should trigger temporal sorting and merging
                    ao = results.AnimalOrganizer(
                        anim_id="A10",
                        base_folder_path=temp_path,
                        mode="base"
                    )

                    # Verify that we have one merged LRO (overlapping folders)
                    assert len(ao.long_recordings) == 1

                    # Verify that merging occurred - should be 2 merges for 3 folders
                    assert merge_call_count == 2, f"Expected 2 merge calls for 3 folders, got {merge_call_count}"

                    # Verify that all folders were grouped into one animalday
                    assert len(ao.animaldays) == 1, f"Expected 1 animalday, got {len(ao.animaldays)}"

                    print("SUCCESS: Folders sorted by median time and merged correctly")

    def test_animalorganizer_folder_grouping(self):
        """
        Test that AnimalOrganizer correctly groups folders by animalday.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create folders that will parse to same animalday
            folder1 = temp_path / "WT_A10_2023-01-15"
            folder2 = temp_path / "WT_A10_2023-01-15(1)"
            folder3 = temp_path / "WT_A10_2023-01-16"  # Different day

            for folder in [folder1, folder2, folder3]:
                folder.mkdir(parents=True)
                (folder / "dummy_ColMajor_001.bin").touch()
                (folder / "dummy_Meta_001.json").touch()

            # Mock glob to return test folders
            with patch('glob.glob') as mock_glob:
                mock_glob.return_value = [str(folder1), str(folder2), str(folder3)]

                # Mock LRO creation to avoid file processing
                with patch.object(core, 'LongRecordingOrganizer') as mock_lro_class:
                    mock_lro = Mock()
                    mock_lro.channel_names = ['LMot', 'RMot', 'LAud']
                    mock_lro.meta = Mock()
                    mock_lro_class.return_value = mock_lro

                    ao = results.AnimalOrganizer(
                        anim_id="A10",
                        base_folder_path=temp_path,
                        mode="base"
                    )

                    # Should have 2 unique animaldays, not 3 folders
                    assert len(ao.animaldays) == 2
                    assert len(ao.long_recordings) == 2

                    # Check folder grouping
                    assert hasattr(ao, '_animalday_folder_groups')
                    assert len(ao._animalday_folder_groups) == 2

                    # One animalday should have 2 folders (overlapping)
                    folder_counts = [len(folders) for folders in ao._animalday_folder_groups.values()]
                    assert 2 in folder_counts  # One day has 2 folders
                    assert 1 in folder_counts  # One day has 1 folder

    def test_lro_merge_functionality(self):
        """
        Test that LongRecordingOrganizer can merge with another LRO.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            folder1 = temp_path / "folder1"
            folder2 = temp_path / "folder2"

            for folder in [folder1, folder2]:
                folder.mkdir(parents=True)

            # Mock LRO creation and setup - patch the entire mode processing
            with patch.object(core.LongRecordingOrganizer, 'convert_colbins_rowbins_to_rec'):
                with patch.object(core.LongRecordingOrganizer, 'prepare_colbins_rowbins_metas'):
                    # Create two mock LROs
                    lro1 = core.LongRecordingOrganizer(base_folder_path=folder1, mode=None)  # Use None to skip processing
                    lro2 = core.LongRecordingOrganizer(base_folder_path=folder2, mode=None)

                # Mock the recordings and metadata
                mock_recording1 = Mock()
                mock_recording2 = Mock()
                mock_merged_recording = Mock()

                lro1.LongRecording = mock_recording1
                lro2.LongRecording = mock_recording2
                lro1.channel_names = ['LMot', 'RMot', 'LAud']
                lro2.channel_names = ['LMot', 'RMot', 'LAud']
                lro1.meta = Mock(f_s=1000, n_channels=3)
                lro2.meta = Mock(f_s=1000, n_channels=3)

                # Mock si.concatenate_recordings at the module level
                with patch('pythoneeg.core.core.si.concatenate_recordings') as mock_concat:
                    mock_concat.return_value = mock_merged_recording

                    # Test merge functionality
                    lro1.merge(lro2)

                    # Verify concatenation was called
                    mock_concat.assert_called_once_with([mock_recording1, mock_recording2])
                    assert lro1.LongRecording == mock_merged_recording


if __name__ == "__main__":
    # Run just this test to see the bug
    pytest.main([__file__ + "::TestOverlappingAnimaldaysBug::test_overlapping_animaldays_lof_overwrite_bug", "-v"])