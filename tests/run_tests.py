#!/usr/bin/env python3
"""
Test runner script for PythonEEG package.
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", coverage=True, verbose=False, parallel=False):
    """
    Run tests with specified options.
    
    Args:
        test_type (str): Type of tests to run ('all', 'unit', 'integration', 'slow')
        coverage (bool): Whether to run with coverage
        verbose (bool): Whether to run in verbose mode
        parallel (bool): Whether to run tests in parallel
    """
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test type filters
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    
    # Add coverage options
    if coverage:
        cmd.extend(["--cov=pythoneeg", "--cov-report=term-missing", "--cov-report=html"])
    
    # Add verbose option
    if verbose:
        cmd.append("-v")
    
    # Add parallel option
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add test discovery
    cmd.append("tests/")
    
    print(f"Running tests with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        return False


def run_specific_test(test_file, verbose=False):
    """
    Run a specific test file.
    
    Args:
        test_file (str): Path to test file
        verbose (bool): Whether to run in verbose mode
    """
    cmd = ["python", "-m", "pytest", test_file]
    
    if verbose:
        cmd.append("-v")
    
    print(f"Running specific test: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("Test passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Test failed with exit code {e.returncode}")
        return False


def run_linting():
    """Run code linting checks."""
    print("Running linting checks...")
    
    # Check if flake8 is available
    try:
        result = subprocess.run(["flake8", "pythoneeg/", "tests/"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Linting passed!")
            return True
        else:
            print("Linting issues found:")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print("flake8 not found. Install with: pip install flake8")
        return False


def run_type_checking():
    """Run type checking with mypy."""
    print("Running type checking...")
    
    try:
        result = subprocess.run(["mypy", "pythoneeg/"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Type checking passed!")
            return True
        else:
            print("Type checking issues found:")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print("mypy not found. Install with: pip install mypy")
        return False


def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(description="Run PythonEEG tests")
    parser.add_argument("--type", choices=["all", "unit", "integration", "slow", "fast"],
                       default="all", help="Type of tests to run")
    parser.add_argument("--no-coverage", action="store_true", 
                       help="Disable coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Run in verbose mode")
    parser.add_argument("--parallel", "-p", action="store_true", 
                       help="Run tests in parallel")
    parser.add_argument("--file", "-f", help="Run specific test file")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--all-checks", action="store_true", 
                       help="Run all checks (tests, linting, type checking)")
    
    args = parser.parse_args()
    
    success = True
    
    if args.file:
        success = run_specific_test(args.file, args.verbose)
    elif args.lint:
        success = run_linting()
    elif args.type_check:
        success = run_type_checking()
    elif args.all_checks:
        print("Running all checks...")
        success = run_linting() and run_type_checking() and run_tests(
            args.type, not args.no_coverage, args.verbose, args.parallel
        )
    else:
        success = run_tests(args.type, not args.no_coverage, args.verbose, args.parallel)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 