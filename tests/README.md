# PythonEEG Testing Guide

This directory contains the comprehensive test suite for the PythonEEG package.

## Overview

The testing framework is built using pytest and provides:

- **Unit Tests**: Test individual functions and methods in isolation
- **Integration Tests**: Test interactions between components
- **Mock Tests**: Test external dependencies without requiring actual data
- **Performance Tests**: Benchmark critical functions
- **Coverage Reporting**: Track code coverage metrics

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py             # Pytest configuration and shared fixtures
├── test_constants.py       # Tests for constants module
├── test_utils.py           # Tests for utility functions
├── test_analysis.py        # Tests for analysis module
├── test_visualization.py   # Tests for visualization module
├── test_core.py            # Tests for core functionality
├── test_integration.py     # Integration tests
├── run_tests.py            # Test runner script
└── README.md               # This file
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=pythoneeg --cov-report=html

# Run specific test file
python -m pytest tests/test_utils.py

# Run specific test function
python -m pytest tests/test_utils.py::TestConvertUnitsToMultiplier::test_valid_conversions
```

### Using the Test Runner

```bash
# Run all tests with coverage
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py --type unit

# Run fast tests (exclude slow ones)
python tests/run_tests.py --type fast

# Run with verbose output
python tests/run_tests.py --verbose

# Run tests in parallel
python tests/run_tests.py --parallel

# Run all checks (tests + linting + type checking)
python tests/run_tests.py --all-checks
```

### Test Categories

Tests are categorized using pytest markers:

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (slower, test interactions)
- `@pytest.mark.slow`: Slow tests (benchmarks, full pipelines)
- `@pytest.mark.eeg_data`: Tests requiring EEG data files
- `@pytest.mark.spikeinterface`: Tests requiring SpikeInterface
- `@pytest.mark.mne`: Tests requiring MNE
- `@pytest.mark.visualization`: Tests for visualization modules
- `@pytest.mark.core`: Tests for core functionality

```bash
# Run only unit tests
python -m pytest -m unit

# Run integration tests
python -m pytest -m integration

# Skip slow tests
python -m pytest -m "not slow"

# Run tests requiring specific dependencies
python -m pytest -m "spikeinterface or mne"
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

### Data Fixtures

- `sample_eeg_data`: Single-channel EEG data
- `sample_multi_channel_eeg_data`: Multi-channel EEG data
- `sample_metadata`: Sample metadata dictionary
- `sample_dataframe`: Sample DataFrame for testing

### Mock Fixtures

- `mock_spikeinterface`: Mocked SpikeInterface objects
- `mock_mne`: Mocked MNE objects
- `test_file_paths`: Test file paths for various formats

### Environment Fixtures

- `temp_dir`: Temporary directory for test files
- `setup_test_environment`: Automatic test environment setup/cleanup

## Writing Tests

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<function_name>_<scenario>`

### Example Test Structure

```python
import pytest
import numpy as np
from pythoneeg.core import utils

class TestExampleFunction:
    """Test example function."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = utils.example_function(1, 2)
        assert result == 3
        
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="Invalid input"):
            utils.example_function(-1, 0)
            
    def test_edge_cases(self):
        """Test edge cases."""
        result = utils.example_function(0, 0)
        assert result == 0
        
    @pytest.mark.slow
    def test_large_input(self):
        """Test with large input (slow test)."""
        large_data = np.random.randn(10000)
        result = utils.example_function(large_data, 1)
        assert len(result) == len(large_data)
```

### Best Practices

1. **Test Isolation**: Each test should be independent
2. **Descriptive Names**: Use clear, descriptive test names
3. **Arrange-Act-Assert**: Structure tests in three parts
4. **Test Edge Cases**: Include boundary conditions and error cases
5. **Use Fixtures**: Reuse common test data and setup
6. **Mock External Dependencies**: Don't rely on external services
7. **Test Documentation**: Document what each test validates

### Testing Guidelines

#### Unit Tests
- Test individual functions/methods
- Use mocks for external dependencies
- Focus on logic and edge cases
- Keep tests fast (< 1 second each)

#### Integration Tests
- Test component interactions
- Use realistic test data
- May be slower but more comprehensive
- Test end-to-end workflows

#### Performance Tests
- Benchmark critical functions
- Use `@pytest.mark.benchmark`
- Compare against baseline performance
- Run separately from regular tests

## Coverage

### Coverage Goals

- **Overall Coverage**: > 80%
- **Core Modules**: > 90%
- **Critical Paths**: 100%

### Coverage Reports

```bash
# Generate HTML coverage report
python -m pytest --cov=pythoneeg --cov-report=html

# Generate XML coverage report (for CI)
python -m pytest --cov=pythoneeg --cov-report=xml

# View coverage in terminal
python -m pytest --cov=pythoneeg --cov-report=term-missing
```

### Coverage Configuration

Coverage settings are configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["pythoneeg"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/migrations/*",
    "*/venv/*",
    "*/env/*",
    "*/site-packages/*",
]
```

## Continuous Integration

### GitHub Actions

Tests are automatically run on:

- Push to main branch
- Pull requests
- Release tags

### CI Pipeline

1. **Linting**: Check code style with flake8
2. **Type Checking**: Verify type hints with mypy
3. **Unit Tests**: Run fast unit tests
4. **Integration Tests**: Run integration tests
5. **Coverage**: Generate coverage reports
6. **Performance Tests**: Run benchmarks (nightly)

### Local CI Simulation

```bash
# Run all CI checks locally
python tests/run_tests.py --all-checks
```

## Debugging Tests

### Verbose Output

```bash
# Run with verbose output
python -m pytest -v

# Show print statements
python -m pytest -s

# Show local variables on failure
python -m pytest -l
```

### Test Debugging

```bash
# Run single test with debugger
python -m pytest tests/test_utils.py::TestExample::test_function -s --pdb

# Run with maximum verbosity
python -m pytest -vvv --tb=long
```

### Coverage Debugging

```bash
# Show which lines are not covered
python -m pytest --cov=pythoneeg --cov-report=term-missing

# Generate detailed coverage report
python -m pytest --cov=pythoneeg --cov-report=html --cov-report=term-missing
```

## Performance Testing

### Benchmark Tests

```python
import pytest

@pytest.mark.benchmark
def test_function_performance(benchmark):
    """Benchmark function performance."""
    result = benchmark(utils.expensive_function, large_data)
    assert result is not None
```

### Running Benchmarks

```bash
# Run benchmark tests
python -m pytest --benchmark-only

# Compare with previous runs
python -m pytest --benchmark-compare

# Generate benchmark report
python -m pytest --benchmark-json=benchmark_results.json
```

## Test Data Management

### Synthetic Test Data

- Use `numpy.random` for synthetic data
- Create realistic EEG signals with known properties
- Use fixtures for consistent test data

### External Data

- Store test data in `tests/data/` directory
- Use small, representative datasets
- Document data sources and formats
- Include data validation tests

### Data Cleanup

- Use temporary directories for test files
- Clean up after tests automatically
- Use `@pytest.fixture(scope="session")` for expensive setup

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure package is installed in development mode
2. **Missing Dependencies**: Install test dependencies with `pip install -e ".[dev]"`
3. **Path Issues**: Use `pathlib.Path` for cross-platform compatibility
4. **Mock Issues**: Ensure mocks are properly configured

### Getting Help

- Check pytest documentation: https://docs.pytest.org/
- Review existing tests for examples
- Use `python -m pytest --help` for command options
- Check coverage reports for untested code

## Contributing

When adding new functionality:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Add appropriate test markers
5. Update this documentation if needed

### Adding New Tests

1. Create test file following naming convention
2. Use existing fixtures when possible
3. Add appropriate markers
4. Include docstrings explaining test purpose
5. Ensure tests are fast and reliable 