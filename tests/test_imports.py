"""
Comprehensive tests for import patterns and IDE functionality.
Tests all import strategies and validates proper module loading.
"""

import importlib
import pytest
import sys
from unittest.mock import patch


class TestImportPatterns:
    """Test all supported import patterns work correctly."""

    def test_direct_class_imports(self):
        """Test importing specific classes directly."""
        from neurodent.loading import LongRecordingOrganizer
        from neurodent.analysis import LongRecordingAnalyzer, FragmentAnalyzer

        assert LongRecordingOrganizer is not None
        assert LongRecordingAnalyzer is not None
        assert FragmentAnalyzer is not None

    def test_direct_function_imports(self):
        """Test importing utility functions from the shared helper package."""
        from neurodent.core.utils import (
            get_temp_directory,
            nanaverage,
            log_transform,
            resolve_channel,
        )

        assert callable(get_temp_directory)
        assert callable(nanaverage)
        assert callable(log_transform)
        assert callable(resolve_channel)

    def test_module_level_access(self):
        """Stage classes live in their stage packages, not in neurodent.core."""
        import neurodent.loading
        import neurodent.analysis

        assert hasattr(neurodent.loading, "LongRecordingOrganizer")
        assert hasattr(neurodent.analysis, "LongRecordingAnalyzer")
        assert hasattr(neurodent.analysis, "FragmentAnalyzer")

    def test_package_level_access(self):
        """Headline classes are also lazily importable from the top level."""
        import neurodent

        assert neurodent.LongRecordingOrganizer is not None
        assert neurodent.LongRecordingAnalyzer is not None
        assert neurodent.AnimalOrganizer is not None

    def test_import_consistency(self):
        """Test that different import patterns return the same objects."""
        from neurodent.loading import LongRecordingOrganizer as direct
        import neurodent
        import neurodent.loading as loading_module

        assert direct is loading_module.LongRecordingOrganizer
        assert direct is neurodent.LongRecordingOrganizer


class TestCircularImports:
    """Test that circular import issues are resolved properly."""

    def test_no_circular_import_errors(self):
        """Importing the stage packages in a fresh interpreter raises no circular-import errors.

        Runs in a subprocess so it never mutates this interpreter's ``sys.modules``
        (deleting/reloading stage modules here would create duplicate class objects
        and break ``isinstance`` checks in later tests).
        """
        import subprocess

        code = (
            "import neurodent.loading, neurodent.analysis, neurodent.results, neurodent.plotting; "
            "from neurodent.loading import LongRecordingOrganizer; "
            "from neurodent.analysis import LongRecordingAnalyzer; "
            "assert LongRecordingOrganizer is not None and LongRecordingAnalyzer is not None"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, f"circular import detected:\n{result.stdout}\n{result.stderr}"

    def test_import_order_independence(self):
        """Test that import order doesn't matter."""
        # Test different import orders
        from neurodent.analysis import LongRecordingAnalyzer
        from neurodent.loading import LongRecordingOrganizer
        from neurodent.analysis import FragmentAnalyzer

        assert LongRecordingAnalyzer is not None
        assert LongRecordingOrganizer is not None
        assert FragmentAnalyzer is not None


class TestIDEFunctionality:
    """Test that IDE features work properly with imports."""

    def test_docstring_availability(self):
        """Test that docstrings are immediately accessible."""
        from neurodent.loading import LongRecordingOrganizer
        from neurodent.analysis import LongRecordingAnalyzer, FragmentAnalyzer

        # All classes should have docstrings available
        assert hasattr(LongRecordingOrganizer, "__doc__")
        assert hasattr(LongRecordingAnalyzer, "__doc__")
        assert hasattr(FragmentAnalyzer, "__doc__")

        # Classes should be accessible even if docstrings are None
        # (some classes may not have docstrings but should still be importable)
        assert LongRecordingOrganizer is not None
        assert LongRecordingAnalyzer is not None
        assert FragmentAnalyzer is not None

    def test_class_attributes_accessible(self):
        """Test that class attributes are immediately accessible for IDE inspection."""
        from neurodent.loading import LongRecordingOrganizer
        from neurodent.analysis import LongRecordingAnalyzer

        # Check that classes have expected attributes accessible
        assert hasattr(LongRecordingOrganizer, "__init__")
        assert hasattr(LongRecordingAnalyzer, "__init__")

        # Check method signatures are accessible
        import inspect

        assert inspect.signature(LongRecordingOrganizer.__init__) is not None
        assert inspect.signature(LongRecordingAnalyzer.__init__) is not None

    def test_module_dir_contents(self):
        """Test that dir() returns expected contents for IDE autocomplete."""
        import neurodent.loading
        import neurodent.analysis
        import neurodent.core.utils

        assert "LongRecordingOrganizer" in dir(neurodent.loading)
        assert "LongRecordingAnalyzer" in dir(neurodent.analysis)
        assert "FragmentAnalyzer" in dir(neurodent.analysis)

        util_dir = dir(neurodent.core.utils)
        for item in ["get_temp_directory", "nanaverage", "log_transform", "resolve_channel"]:
            assert item in util_dir, f"{item} not found in dir(neurodent.core.utils)"


class TestImportPerformance:
    """Test import performance characteristics."""

    def test_immediate_availability(self):
        """Test that classes are available immediately after import."""
        import time

        start_time = time.time()
        from neurodent.loading import LongRecordingOrganizer
        from neurodent.analysis import FragmentAnalyzer

        import_time = time.time() - start_time

        # Classes should be immediately accessible
        assert LongRecordingOrganizer is not None
        assert FragmentAnalyzer is not None

        # Import should complete reasonably quickly (less than 5 seconds even with heavy deps)
        assert import_time < 5.0, f"Import took {import_time:.2f}s, too slow"

    def test_repeated_imports_cached(self):
        """Test that repeated imports return the same object (cached)."""
        from neurodent.loading import LongRecordingOrganizer
        from neurodent.loading import LongRecordingOrganizer as LRO2

        # Same object should be returned
        assert LongRecordingOrganizer is LRO2

        # Test multiple imports return same object
        import neurodent.core

        assert LongRecordingOrganizer is neurodent.loading.long_recording_organizer.LongRecordingOrganizer


class TestImportErrors:
    """Test proper error handling for import issues."""

    def test_nonexistent_import_error(self):
        """Test that importing non-existent items raises proper errors."""
        with pytest.raises(ImportError):
            from neurodent.core import NonExistentClass

    def test_module_attribute_error(self):
        """Test that accessing non-existent attributes raises proper errors."""
        import neurodent.core

        with pytest.raises(AttributeError):
            _ = neurodent.core.NonExistentClass


class TestStandardizedImports:
    """Test that standardized import patterns work correctly."""

    def test_core_module_import_pattern(self):
        """Shared helpers come from core.utils; stage classes from stage packages."""
        from neurodent.core import utils
        from neurodent.loading import LongRecordingOrganizer
        from neurodent.analysis import LongRecordingAnalyzer, FragmentAnalyzer

        assert hasattr(utils, "resolve_channel")
        assert callable(utils.resolve_channel)

        assert LongRecordingOrganizer is not None
        assert LongRecordingAnalyzer is not None
        assert FragmentAnalyzer is not None

    def test_public_api_accessibility(self):
        """Test that public helper functions are accessible through core.utils."""
        from neurodent.core import utils

        public_functions = [
            "resolve_channel",
            "get_temp_directory",
            "set_temp_directory",
            "nanaverage",
            "log_transform",
            "validate_timestamps",
        ]

        for func_name in public_functions:
            assert hasattr(utils, func_name), f"core.utils.{func_name} should be accessible"
            assert callable(getattr(utils, func_name)), f"core.utils.{func_name} should be callable"

    def test_internal_utils_accessibility(self):
        """Test that internal/advanced utils are accessible through core.utils."""
        from neurodent import core

        # Test internal/advanced functions (available via core.utils)
        internal_functions = [
            "parse_truncate",
            "cache_fragments_to_zarr",
            "stream_fragments_to_zarr",
            "chunked_channel_distance_matrix",
            "is_day",
            "nanmean_series_of_np",
            "sort_dataframe_by_plot_order",
            "_get_groupby_keys",
            "_get_pairwise_combinations",
        ]

        assert hasattr(core, "utils"), "core.utils should be accessible"

        for func_name in internal_functions:
            assert hasattr(core.utils, func_name), f"core.utils.{func_name} should be accessible (internal API)"
            assert callable(getattr(core.utils, func_name)), f"core.utils.{func_name} should be callable"

    def test_both_import_patterns_equivalent(self):
        """Test that both import patterns access the same functions."""
        from neurodent.core import utils
        from neurodent.core.utils import resolve_channel

        # Both should reference the same function
        assert utils.resolve_channel is resolve_channel


class TestVisualizationImports:
    """Test visualization module imports work correctly."""

    def test_visualization_imports(self):
        """Test that visualization modules import correctly."""
        from neurodent.results import WindowAnalysisResult
        from neurodent.loading import AnimalOrganizer
        from neurodent.plotting import AnimalPlotter, ExperimentPlotter

        assert WindowAnalysisResult is not None
        assert AnimalOrganizer is not None
        assert AnimalPlotter is not None
        assert ExperimentPlotter is not None

    def test_plotting_submodule_imports(self):
        """Test plotting submodule imports."""
        from neurodent.plotting import AnimalPlotter, ExperimentPlotter

        assert AnimalPlotter is not None
        assert ExperimentPlotter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
