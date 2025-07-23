#!/usr/bin/env python3
"""
Simple test runner for development.
Usage:
    python test_runner.py                                    # Run all tests in test_core.py
    python test_runner.py TestLongRecordingOrganizer         # Run specific class
    python test_runner.py test_get_datetime_fragment         # Run specific test
    python test_runner.py TestClass::test_method             # Run specific test in class
"""
import sys
import subprocess

def run_tests(test_spec=None):
    """Run tests with development-friendly defaults (no coverage, short output)."""
    cmd = ["python", "-m", "pytest", "-c", "pytest-dev.ini"]
    
    if test_spec:
        if "::" in test_spec:
            # Full specification like TestClass::test_method
            cmd.append(f"tests/test_core.py::{test_spec}")
        elif test_spec.startswith("Test"):
            # Class name like TestLongRecordingOrganizer
            cmd.append(f"tests/test_core.py::{test_spec}")
        elif test_spec.startswith("test_"):
            # Test method name - search across all classes
            cmd.extend(["-k", test_spec])
        else:
            # Partial match or pattern
            cmd.extend(["-k", test_spec])
    else:
        # Run all tests in test_core.py
        cmd.append("tests/test_core.py")
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)

if __name__ == "__main__":
    test_spec = sys.argv[1] if len(sys.argv) > 1 else None
    exit_code = run_tests(test_spec).returncode
    sys.exit(exit_code)