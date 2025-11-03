"""
Tests for the run_tests.py test runner script.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add tests directory to path to import run_tests module
# This is needed because run_tests.py is a utility script in tests/
sys.path.insert(0, str(Path(__file__).parent))

from run_tests import run_tests, run_specific_test, run_linting, run_type_checking, main


class TestRunTests:
    """Test the run_tests function."""

    @patch("run_tests.subprocess.run")
    def test_run_tests_default_all(self, mock_run):
        """Test running all tests with default options."""
        mock_run.return_value = Mock(returncode=0)

        result = run_tests()

        expected_cmd = [
            "python",
            "-m",
            "pytest",
            "--cov=neurodent",
            "--cov-report=term-missing",
            "--cov-report=html",
            "tests/",
        ]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_tests_unit_only(self, mock_run):
        """Test running unit tests only."""
        mock_run.return_value = Mock(returncode=0)

        result = run_tests(test_type="unit")

        expected_cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "unit",
            "--cov=neurodent",
            "--cov-report=term-missing",
            "--cov-report=html",
            "tests/",
        ]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_tests_integration_only(self, mock_run):
        """Test running integration tests only."""
        mock_run.return_value = Mock(returncode=0)

        result = run_tests(test_type="integration")

        expected_cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "integration",
            "--cov=neurodent",
            "--cov-report=term-missing",
            "--cov-report=html",
            "tests/",
        ]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_tests_slow_only(self, mock_run):
        """Test running slow tests only."""
        mock_run.return_value = Mock(returncode=0)

        result = run_tests(test_type="slow")

        expected_cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "slow",
            "--cov=neurodent",
            "--cov-report=term-missing",
            "--cov-report=html",
            "tests/",
        ]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_tests_fast_only(self, mock_run):
        """Test running fast tests (excluding slow)."""
        mock_run.return_value = Mock(returncode=0)

        result = run_tests(test_type="fast")

        expected_cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "not slow",
            "--cov=neurodent",
            "--cov-report=term-missing",
            "--cov-report=html",
            "tests/",
        ]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_tests_no_coverage(self, mock_run):
        """Test running tests without coverage."""
        mock_run.return_value = Mock(returncode=0)

        result = run_tests(coverage=False)

        expected_cmd = ["python", "-m", "pytest", "tests/"]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_tests_verbose(self, mock_run):
        """Test running tests in verbose mode."""
        mock_run.return_value = Mock(returncode=0)

        result = run_tests(verbose=True)

        expected_cmd = [
            "python",
            "-m",
            "pytest",
            "--cov=neurodent",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-v",
            "tests/",
        ]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_tests_parallel(self, mock_run):
        """Test running tests in parallel."""
        mock_run.return_value = Mock(returncode=0)

        result = run_tests(parallel=True)

        expected_cmd = [
            "python",
            "-m",
            "pytest",
            "--cov=neurodent",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-n",
            "auto",
            "tests/",
        ]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_tests_all_options(self, mock_run):
        """Test running tests with all options enabled."""
        mock_run.return_value = Mock(returncode=0)

        result = run_tests(test_type="unit", coverage=True, verbose=True, parallel=True)

        expected_cmd = [
            "python",
            "-m",
            "pytest",
            "-m",
            "unit",
            "--cov=neurodent",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-v",
            "-n",
            "auto",
            "tests/",
        ]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_tests_failure(self, mock_run):
        """Test handling of test failures."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "pytest")

        result = run_tests()

        assert result is False


class TestRunSpecificTest:
    """Test the run_specific_test function."""

    @patch("run_tests.subprocess.run")
    def test_run_specific_test_success(self, mock_run):
        """Test running a specific test file successfully."""
        mock_run.return_value = Mock(returncode=0)

        result = run_specific_test("tests/test_utils.py")

        expected_cmd = ["python", "-m", "pytest", "tests/test_utils.py"]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_specific_test_verbose(self, mock_run):
        """Test running a specific test file with verbose output."""
        mock_run.return_value = Mock(returncode=0)

        result = run_specific_test("tests/test_utils.py", verbose=True)

        expected_cmd = ["python", "-m", "pytest", "tests/test_utils.py", "-v"]
        mock_run.assert_called_once_with(expected_cmd, check=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_specific_test_failure(self, mock_run):
        """Test handling of specific test failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "pytest")

        result = run_specific_test("tests/test_utils.py")

        assert result is False


class TestRunLinting:
    """Test the run_linting function."""

    @patch("run_tests.subprocess.run")
    def test_run_linting_success(self, mock_run):
        """Test successful linting."""
        mock_run.return_value = Mock(returncode=0, stdout="All good!")

        result = run_linting()

        expected_cmd = ["flake8", "neurodent/", "tests/"]
        mock_run.assert_called_once_with(expected_cmd, capture_output=True, text=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_linting_issues_found(self, mock_run):
        """Test linting with issues found."""
        mock_run.return_value = Mock(returncode=1, stdout="Linting issues...")

        result = run_linting()

        assert result is False

    @patch("run_tests.subprocess.run")
    def test_run_linting_flake8_not_found(self, mock_run):
        """Test linting when flake8 is not installed."""
        mock_run.side_effect = FileNotFoundError()

        result = run_linting()

        assert result is False


class TestRunTypeChecking:
    """Test the run_type_checking function."""

    @patch("run_tests.subprocess.run")
    def test_run_type_checking_success(self, mock_run):
        """Test successful type checking."""
        mock_run.return_value = Mock(returncode=0, stdout="Success: no issues found")

        result = run_type_checking()

        expected_cmd = ["mypy", "neurodent/"]
        mock_run.assert_called_once_with(expected_cmd, capture_output=True, text=True)
        assert result is True

    @patch("run_tests.subprocess.run")
    def test_run_type_checking_issues_found(self, mock_run):
        """Test type checking with issues found."""
        mock_run.return_value = Mock(returncode=1, stdout="Type checking issues...")

        result = run_type_checking()

        assert result is False

    @patch("run_tests.subprocess.run")
    def test_run_type_checking_mypy_not_found(self, mock_run):
        """Test type checking when mypy is not installed."""
        mock_run.side_effect = FileNotFoundError()

        result = run_type_checking()

        assert result is False


class TestMainFunction:
    """Test the main function and argument parsing."""

    @patch("run_tests.run_tests")
    @patch("sys.argv", ["run_tests.py"])
    def test_main_default_args(self, mock_run_tests):
        """Test main function with default arguments."""
        mock_run_tests.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_run_tests.assert_called_once_with("all", True, False, False)

    @patch("run_tests.run_tests")
    @patch("sys.argv", ["run_tests.py", "--type", "unit", "--verbose"])
    def test_main_unit_verbose(self, mock_run_tests):
        """Test main function with unit tests and verbose output."""
        mock_run_tests.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_run_tests.assert_called_once_with("unit", True, True, False)

    @patch("run_tests.run_tests")
    @patch("sys.argv", ["run_tests.py", "--no-coverage", "--parallel"])
    def test_main_no_coverage_parallel(self, mock_run_tests):
        """Test main function without coverage and with parallel execution."""
        mock_run_tests.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_run_tests.assert_called_once_with("all", False, False, True)

    @patch("run_tests.run_specific_test")
    @patch("sys.argv", ["run_tests.py", "--file", "tests/test_utils.py"])
    def test_main_specific_file(self, mock_run_specific):
        """Test main function with specific file."""
        mock_run_specific.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_run_specific.assert_called_once_with("tests/test_utils.py", False)

    @patch("run_tests.run_linting")
    @patch("sys.argv", ["run_tests.py", "--lint"])
    def test_main_lint_only(self, mock_run_linting):
        """Test main function with lint only."""
        mock_run_linting.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_run_linting.assert_called_once()

    @patch("run_tests.run_type_checking")
    @patch("sys.argv", ["run_tests.py", "--type-check"])
    def test_main_type_check_only(self, mock_run_type_checking):
        """Test main function with type checking only."""
        mock_run_type_checking.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_run_type_checking.assert_called_once()

    @patch("run_tests.run_linting")
    @patch("run_tests.run_type_checking")
    @patch("run_tests.run_tests")
    @patch("sys.argv", ["run_tests.py", "--all-checks"])
    def test_main_all_checks(self, mock_run_tests, mock_run_type_checking, mock_run_linting):
        """Test main function with all checks enabled."""
        mock_run_linting.return_value = True
        mock_run_type_checking.return_value = True
        mock_run_tests.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_run_linting.assert_called_once()
        mock_run_type_checking.assert_called_once()
        mock_run_tests.assert_called_once_with("all", True, False, False)

    @patch("run_tests.run_tests")
    @patch("sys.argv", ["run_tests.py"])
    def test_main_failure_exit_code(self, mock_run_tests):
        """Test main function returns failure exit code when tests fail."""
        mock_run_tests.return_value = False

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1


class TestRunTestsIntegration:
    """Integration tests for run_tests.py functionality."""

    def test_help_output(self):
        """Test that help output is generated correctly."""
        result = subprocess.run([sys.executable, "tests/run_tests.py", "--help"], capture_output=True, text=True)

        assert result.returncode == 0
        assert "Run NeuRodent tests" in result.stdout
        assert "--type" in result.stdout
        assert "--no-coverage" in result.stdout
        assert "--verbose" in result.stdout
        assert "--parallel" in result.stdout
        assert "--file" in result.stdout
        assert "--lint" in result.stdout
        assert "--type-check" in result.stdout
        assert "--all-checks" in result.stdout


# All tests in this file are unit tests
