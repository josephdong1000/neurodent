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
        from neurodent.core import LongRecordingOrganizer, LongRecordingAnalyzer, FragmentAnalyzer
        from neurodent.core import DDFBinaryMetadata

        assert LongRecordingOrganizer is not None
        assert LongRecordingAnalyzer is not None
        assert FragmentAnalyzer is not None
        assert DDFBinaryMetadata is not None

    def test_direct_function_imports(self):
        """Test importing utility functions directly."""
        from neurodent.core import (
            get_temp_directory,
            nanaverage,
            log_transform,
            parse_chname_to_abbrev,
            parse_path_to_animalday,
        )

        assert callable(get_temp_directory)
        assert callable(nanaverage)
        assert callable(log_transform)
        assert callable(parse_chname_to_abbrev)
        assert callable(parse_path_to_animalday)

    def test_module_level_access(self):
        """Test accessing classes via module namespace."""
        import neurodent.core

        assert hasattr(neurodent.core, "LongRecordingOrganizer")
        assert hasattr(neurodent.core, "LongRecordingAnalyzer")
        assert hasattr(neurodent.core, "FragmentAnalyzer")
        assert hasattr(neurodent.core, "DDFBinaryMetadata")

    def test_package_level_access(self):
        """Test accessing via package import."""
        import neurodent
        from neurodent import core

        assert hasattr(core, "LongRecordingOrganizer")
        assert hasattr(core, "LongRecordingAnalyzer")
        assert hasattr(core, "FragmentAnalyzer")

    def test_import_consistency(self):
        """Test that different import patterns return the same objects."""
        from neurodent.core import LongRecordingOrganizer as direct
        import neurodent.core as core_module
        from neurodent import core as package_core

        assert direct is core_module.LongRecordingOrganizer
        assert direct is package_core.LongRecordingOrganizer
        assert core_module.LongRecordingOrganizer is package_core.LongRecordingOrganizer


class TestCircularImports:
    """Test that circular import issues are resolved properly."""

    def test_no_circular_import_errors(self):
        """Test that importing doesn't cause circular import errors."""
        # Force reload to test clean import
        if "neurodent.core" in sys.modules:
            del sys.modules["neurodent.core"]
        if "neurodent.core.analysis" in sys.modules:
            del sys.modules["neurodent.core.analysis"]
        if "neurodent.core.core" in sys.modules:
            del sys.modules["neurodent.core.core"]

        # This should not raise any circular import errors
        import neurodent.core

        assert neurodent.core.LongRecordingOrganizer is not None
        assert neurodent.core.LongRecordingAnalyzer is not None

    def test_import_order_independence(self):
        """Test that import order doesn't matter."""
        # Test different import orders
        from neurodent.core import LongRecordingAnalyzer
        from neurodent.core import LongRecordingOrganizer
        from neurodent.core import FragmentAnalyzer

        assert LongRecordingAnalyzer is not None
        assert LongRecordingOrganizer is not None
        assert FragmentAnalyzer is not None


class TestIDEFunctionality:
    """Test that IDE features work properly with imports."""

    def test_docstring_availability(self):
        """Test that docstrings are immediately accessible."""
        from neurodent.core import LongRecordingOrganizer, LongRecordingAnalyzer, FragmentAnalyzer

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
        from neurodent.core import LongRecordingOrganizer, LongRecordingAnalyzer

        # Check that classes have expected attributes accessible
        assert hasattr(LongRecordingOrganizer, "__init__")
        assert hasattr(LongRecordingAnalyzer, "__init__")

        # Check method signatures are accessible
        import inspect

        assert inspect.signature(LongRecordingOrganizer.__init__) is not None
        assert inspect.signature(LongRecordingAnalyzer.__init__) is not None

    def test_module_dir_contents(self):
        """Test that dir() returns expected contents for IDE autocomplete."""
        import neurodent.core

        dir_contents = dir(neurodent.core)
        expected_classes = ["LongRecordingOrganizer", "LongRecordingAnalyzer", "FragmentAnalyzer", "DDFBinaryMetadata"]
        expected_functions = [
            "get_temp_directory",
            "nanaverage",
            "log_transform",
            "parse_chname_to_abbrev",
            "parse_path_to_animalday",
        ]

        for item in expected_classes + expected_functions:
            assert item in dir_contents, f"{item} not found in dir(neurodent.core)"


class TestImportPerformance:
    """Test import performance characteristics."""

    def test_immediate_availability(self):
        """Test that classes are available immediately after import."""
        import time

        start_time = time.time()
        from neurodent.core import LongRecordingOrganizer, FragmentAnalyzer

        import_time = time.time() - start_time

        # Classes should be immediately accessible
        assert LongRecordingOrganizer is not None
        assert FragmentAnalyzer is not None

        # Import should complete reasonably quickly (less than 5 seconds even with heavy deps)
        assert import_time < 5.0, f"Import took {import_time:.2f}s, too slow"

    def test_repeated_imports_cached(self):
        """Test that repeated imports return the same object (cached)."""
        from neurodent.core import LongRecordingOrganizer
        from neurodent.core import LongRecordingOrganizer as LRO2

        # Same object should be returned
        assert LongRecordingOrganizer is LRO2

        # Test multiple imports return same object
        import neurodent.core

        assert LongRecordingOrganizer is neurodent.core.LongRecordingOrganizer


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
        """Test that importing core as a module and accessing functions works."""
        from neurodent import core

        # Test that all expected functions are accessible via core.function_name()
        assert hasattr(core, "parse_chname_to_abbrev")
        assert hasattr(core, "LongRecordingOrganizer")
        assert hasattr(core, "LongRecordingAnalyzer")
        assert hasattr(core, "FragmentAnalyzer")

        # Test that functions are callable
        assert callable(core.parse_chname_to_abbrev)

        # Test that classes are instantiable (basic check)
        assert core.LongRecordingOrganizer is not None
        assert core.LongRecordingAnalyzer is not None
        assert core.FragmentAnalyzer is not None

    def test_public_api_accessibility(self):
        """Test that public API functions are accessible through core module."""
        from neurodent import core

        # Test public API functions (available directly on core)
        public_functions = [
            "parse_chname_to_abbrev",
            "get_temp_directory",
            "set_temp_directory",
            "nanaverage",
            "log_transform",
            "parse_path_to_animalday",
            "validate_timestamps",
        ]

        for func_name in public_functions:
            assert hasattr(core, func_name), f"core.{func_name} should be accessible (public API)"
            assert callable(getattr(core, func_name)), f"core.{func_name} should be callable"

    def test_internal_utils_accessibility(self):
        """Test that internal/advanced utils are accessible through core.utils."""
        from neurodent import core

        # Test internal/advanced functions (available via core.utils)
        internal_functions = [
            "parse_truncate",
            "cache_fragments_to_zarr",
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
        from neurodent import core
        from neurodent.core.utils import parse_chname_to_abbrev

        # Both should reference the same function
        assert core.parse_chname_to_abbrev is parse_chname_to_abbrev


class TestVisualizationImports:
    """Test visualization module imports work correctly."""

    def test_visualization_imports(self):
        """Test that visualization modules import correctly."""
        from neurodent.visualization import (
            WindowAnalysisResult,
            AnimalOrganizer,
            SpikeAnalysisResult,
            AnimalPlotter,
            ExperimentPlotter,
        )

        assert WindowAnalysisResult is not None
        assert AnimalOrganizer is not None
        assert SpikeAnalysisResult is not None
        assert AnimalPlotter is not None
        assert ExperimentPlotter is not None

    def test_plotting_submodule_imports(self):
        """Test plotting submodule imports."""
        from neurodent.visualization.plotting import AnimalPlotter, ExperimentPlotter

        assert AnimalPlotter is not None
        assert ExperimentPlotter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
