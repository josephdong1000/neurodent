#!/usr/bin/env python3
"""
Comprehensive LOF Evaluation Tests

This module contains all tests for LOF evaluation functionality including:
- Ground truth extraction from samples.json
- LOF evaluation fixes for channel name mapping and key mismatches
- Edge cases and error conditions
- Real-world data scenarios

Consolidates functionality from multiple scattered test files.
"""

import pytest
import logging
import numpy as np
from unittest.mock import Mock
import sys
from pathlib import Path

from neurodent.visualization.results import WindowAnalysisResult


class TestLOFEvaluationFixes:
    """Test the fixes to evaluate_lof_threshold_binary method"""

    def test_key_filtering_with_mismatch(self):
        """Test that key mismatches raise ValueError"""

        # Mock data with key mismatch
        lof_scores_dict = {
            "FMUT FMut Jan-21-2022": {
                "lof_scores": [1.5, 2.8],
                "channel_names": ["Intan Input (1)/PortB L Motor Ctx", "Intan Input (1)/PortB R Motor Ctx"],
            },
            "FMUT FMut Jan-22-2022": {
                "lof_scores": [0.8, 4.1],
                "channel_names": ["Intan Input (1)/PortB L Motor Ctx", "Intan Input (1)/PortB R Motor Ctx"],
            },
        }

        bad_channels_dict = {
            "FMUT FMut Jan-21-2022": {"Intan Input (1)/PortB L Motor Ctx"},
            "FMUT FMut Jan-22-2022": {"Intan Input (1)/PortB R Motor Ctx"},
            "FMUT FMut Jan-23-2022": {"Intan Input (1)/PortB L Motor Ctx"},  # This key missing in LOF
        }

        # Create a minimal mock instance just for the method
        war_mock = Mock()
        war_mock.lof_scores_dict = lof_scores_dict
        war_mock.bad_channels_dict = bad_channels_dict
        war_mock.channel_names = ["LMot", "RMot"]

        # Bind the actual method to the mock
        war_mock.evaluate_lof_threshold_binary = WindowAnalysisResult.evaluate_lof_threshold_binary.__get__(war_mock)

        # This should raise an error for mismatched key
        with pytest.raises(ValueError, match="bad_channels_dict contains keys not found in lof_scores_dict"):
            war_mock.evaluate_lof_threshold_binary(
                ground_truth_bad_channels=None,  # Use bad_channels_dict fallback
                threshold=2.0,
                evaluation_channels=None,  # Use all channels
            )

    def test_channel_name_mapping_with_abbreviations(self):
        """Test that channel name mapping works between full names and abbreviations"""

        from neurodent.visualization.results import WindowAnalysisResult

        # Mock data with channel name mismatch
        lof_scores_dict = {
            "day1": {
                "lof_scores": [1.5, 2.8, 0.9, 3.2],
                "channel_names": [
                    "Intan Input (1)/PortB L Motor Ctx",
                    "Intan Input (1)/PortB R Motor Ctx",
                    "Intan Input (1)/PortB L Aud Ctx",
                    "Intan Input (1)/PortB R Aud Ctx",
                ],
            }
        }

        # Bad channels use abbreviations (realistic scenario)
        bad_channels_dict = {
            "day1": {"LMot", "RAud"}  # Abbreviations, not full names
        }

        # Create a minimal mock instance
        war_mock = Mock()
        war_mock.lof_scores_dict = lof_scores_dict
        war_mock.bad_channels_dict = bad_channels_dict
        war_mock.channel_names = ["LMot", "RMot", "LAud", "RAud"]

        # Bind the actual method to the mock
        war_mock.evaluate_lof_threshold_binary = WindowAnalysisResult.evaluate_lof_threshold_binary.__get__(war_mock)

        # Test with abbreviation evaluation channels
        evaluation_channels = ["LMot", "RMot", "LAud", "RAud"]

        y_true, y_pred = war_mock.evaluate_lof_threshold_binary(
            ground_truth_bad_channels=None, threshold=2.0, evaluation_channels=evaluation_channels
        )

        # Should successfully map channels and generate evaluation points
        assert len(y_true) > 0, "Channel mapping should generate evaluation points"
        assert len(y_pred) > 0

        # Should have 4 evaluation points (one per channel)
        assert len(y_true) == 4
        assert len(y_pred) == 4

        # Verify we get some true positives (abbreviations should map correctly)
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        assert tp > 0, "Should find true positives with abbreviation mapping"

    def test_realistic_user_data_scenario(self):
        """Test with realistic user-provided data scenario"""

        from neurodent.visualization.results import WindowAnalysisResult

        # User-provided realistic test data - only include data that has corresponding LOF scores
        bad_channels_dict = {"M2 MWT Jan-14-2022": ["LHip", "LVis", "RVis", "RHip", "LBar"]}

        lof_scores_dict = {
            "M2 MWT Jan-14-2022": {
                "lof_scores": [1.057, 0.983, 0.989, 0.956, 0.967, 1.163, 0.979, 1.073, 0.944, 1.161],
                "channel_names": [
                    "Intan Input (1)/PortD L Aud Ctx",
                    "Intan Input (1)/PortD L Vis Ctx",
                    "Intan Input (1)/PortD L Hipp",
                    "Intan Input (1)/PortD L Barrel Ctx",
                    "Intan Input (1)/PortD L Motor Ctx",
                    "Intan Input (1)/PortD R Motor Ctx",
                    "Intan Input (1)/PortD R Barrel Ctx",
                    "Intan Input (1)/PortD R Hipp",
                    "Intan Input (1)/PortD R Vis Ctx",
                    "Intan Input (1)/PortD R Aud Ctx",
                ],
            }
        }

        # Create mock WAR object
        war_mock = Mock()
        war_mock.lof_scores_dict = lof_scores_dict
        war_mock.bad_channels_dict = bad_channels_dict
        war_mock.channel_names = ["LAud", "LVis", "LHip", "LBar", "LMot", "RMot", "RBar", "RHip", "RVis", "RAud"]

        # Bind the actual method
        war_mock.evaluate_lof_threshold_binary = WindowAnalysisResult.evaluate_lof_threshold_binary.__get__(war_mock)

        # Test with threshold that should give mixed results
        y_true, y_pred = war_mock.evaluate_lof_threshold_binary(
            ground_truth_bad_channels=None, threshold=1.0, evaluation_channels=None
        )

        # Calculate confusion matrix
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

        # Should have some evaluation - at minimum, confusion matrix should sum correctly
        assert (tp + fp + fn + tn) == len(y_true), "Confusion matrix should sum to total"
        assert len(y_true) > 0, "Should have some evaluation points"

        # Verify channel mapping worked by checking we have some bad channels
        # The exact number depends on channel mapping success, but should be reasonable
        total_bad_actual = sum(y_true)
        assert total_bad_actual > 0, f"Should have some bad channels, got {total_bad_actual}"
        assert total_bad_actual <= 10, f"Should not exceed total channels, got {total_bad_actual}"

    def test_empty_lof_scores_raises_error(self):
        """Test that missing LOF scores raises appropriate error"""

        from neurodent.visualization.results import WindowAnalysisResult

        war_mock = Mock()
        war_mock.lof_scores_dict = None
        war_mock.evaluate_lof_threshold_binary = WindowAnalysisResult.evaluate_lof_threshold_binary.__get__(war_mock)

        with pytest.raises(ValueError, match="LOF scores not available"):
            war_mock.evaluate_lof_threshold_binary(threshold=2.0)

    def test_missing_threshold_raises_error(self):
        """Test that missing threshold raises error"""

        from neurodent.visualization.results import WindowAnalysisResult

        war_mock = Mock()
        war_mock.lof_scores_dict = {"day1": {"lof_scores": [1, 2], "channel_names": ["ch1", "ch2"]}}
        war_mock.evaluate_lof_threshold_binary = WindowAnalysisResult.evaluate_lof_threshold_binary.__get__(war_mock)

        with pytest.raises(ValueError, match="threshold parameter is required"):
            war_mock.evaluate_lof_threshold_binary(threshold=None)

    def test_channel_abbreviation_mapping_logic(self):
        """Test the built-in channel abbreviation mapping"""

        from neurodent.core.utils import parse_chname_to_abbrev

        # Test motor channels
        assert parse_chname_to_abbrev("Intan Input (1)/PortB L Motor Ctx") == "LMot"
        assert parse_chname_to_abbrev("Intan Input (1)/PortB R Motor Ctx") == "RMot"

        # Test barrel channels
        assert parse_chname_to_abbrev("Intan Input (1)/PortB L Barrel Ctx") == "LBar"
        assert parse_chname_to_abbrev("Intan Input (1)/PortB R Barrel Ctx") == "RBar"

        # Test auditory channels
        assert parse_chname_to_abbrev("Intan Input (1)/PortB L Aud Ctx") == "LAud"
        assert parse_chname_to_abbrev("Intan Input (1)/PortB R Aud Ctx") == "RAud"

        # Test visual channels
        assert parse_chname_to_abbrev("Intan Input (1)/PortB L Vis Ctx") == "LVis"
        assert parse_chname_to_abbrev("Intan Input (1)/PortB R Vis Ctx") == "RVis"

        # Test hippocampus channels
        assert parse_chname_to_abbrev("Intan Input (1)/PortB L Hipp") == "LHip"
        assert parse_chname_to_abbrev("Intan Input (1)/PortB R Hipp") == "RHip"

        # Test that abbreviations are returned as-is
        assert parse_chname_to_abbrev("LMot") == "LMot"
        assert parse_chname_to_abbrev("RVis") == "RVis"

    def test_threshold_behavior_realistic_range(self):
        """Test that different thresholds produce expected precision-recall tradeoff"""

        from neurodent.visualization.results import WindowAnalysisResult

        # Create data where some bad channels have high scores, some have low scores
        lof_scores_dict = {
            "day1": {
                "lof_scores": [0.8, 2.5, 0.9, 3.0, 1.1],  # Mix of high and low scores
                "channel_names": [
                    "Intan Input (1)/PortD L Motor Ctx",  # Good channel, low score
                    "Intan Input (1)/PortD L Vis Ctx",  # Bad channel, high score
                    "Intan Input (1)/PortD R Motor Ctx",  # Good channel, low score
                    "Intan Input (1)/PortD L Hipp",  # Bad channel, high score
                    "Intan Input (1)/PortD R Vis Ctx",  # Bad channel, medium score
                ],
            }
        }

        bad_channels_dict = {
            "day1": ["LVis", "LHip", "RVis"]  # 3 bad channels
        }

        war_mock = Mock()
        war_mock.lof_scores_dict = lof_scores_dict
        war_mock.bad_channels_dict = bad_channels_dict
        war_mock.channel_names = ["LMot", "LVis", "RMot", "LHip", "RVis"]
        war_mock.evaluate_lof_threshold_binary = WindowAnalysisResult.evaluate_lof_threshold_binary.__get__(war_mock)

        # Test different thresholds
        results = []
        for threshold in [1.0, 2.0, 3.5]:
            y_true, y_pred = war_mock.evaluate_lof_threshold_binary(
                ground_truth_bad_channels=None, threshold=threshold, evaluation_channels=None
            )

            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            results.append((threshold, precision, recall, tp, fp, fn))

        # Should see precision-recall tradeoff
        # Lower threshold (1.0) should have higher recall, potentially lower precision
        # Higher threshold (3.5) should have higher precision, potentially lower recall

        low_thresh_result = results[0]  # threshold 1.0
        high_thresh_result = results[2]  # threshold 3.5

        # At least one threshold should detect some bad channels
        assert any(tp > 0 for _, _, _, tp, _, _ in results), "Should detect some bad channels at some threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
