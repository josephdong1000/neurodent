# PyEEG Development Guide

Python scientific computing package for extracting features from mouse EEG recordings. Uses numpy/scipy, SpikeInterface (and sometimes MNE), matplotlib/seaborn, pytest.

## Project Essentials

**Two main workflows:**
1. **WAR** - Windowed Analysis Results: Feature extraction from continuous EEG
2. **SAR** - Spike Analysis Results: Spike detection, integrates with WAR via nspike/lognspike

**Key features extracted:**
- **Linear features**: RMS amplitude, amplitude variance, PSD power, spike counts (+ log values)
- **Band features**: PSD across frequency bands, fractional PSD power (+ log values)
- **Connectivity features**: Coherence, imaginary coherence (imcoh), Pearson correlation pairwise (+ Fisher Z-transforms: zcohere, zimcoh, zpcorr)
- **Advanced features**: Full power spectral density

**Frequency bands:** Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-25Hz), Gamma (25-40Hz)
*Note: Delta changed from 0.1-4Hz to 1-4Hz for coherence estimation*

## Commands

**Testing (Always run before commits):**
- `python -m pytest tests/ -v` - All tests
- `python -m pytest tests/test_analysis.py -v` - Specific module
- `python -m pytest tests/ --cov=pythoneeg --cov-report=term-missing` - Coverage
- `python tests/run_tests.py --type unit` - Fast unit tests only
- `python tests/run_tests.py --verbose` - Full coverage report

**Package management:**
- `pipenv install --dev` - Install dependencies
- `pipenv requirements > requirements.txt` - Update requirements
- `pip install -e .` - Development install

## Critical Requirements

**Scientific accuracy:** 
- Use appropriate data types (float64 for calculations)
- Validate input data shapes, ranges, and types before processing
- Handle sampling rates and frequency domains carefully
- Coherence (real number) = magnitude of coherency (complex value)

**Design:**
- Research the best practices for codebase design decisions
- Make minimal changes unless instructed otherwise
- Avoid overdocumenting code that is self-explanatory, particularly inline comments
- Typehints are great -- use them
- **Imports**: Prefer immediate imports for IDE support; use lazy imports only for heavy dependencies (MNE/SpikeInterface classes)
- **API Design**: When parameters create confusing interactions, unify into single clear parameter with descriptive values
- **Code Cleanup**: Remove unused functions, consolidate redundant code, clean up imports when requested
- **Variable Scope**: Trace all code paths to ensure variables are properly defined in scope before implementing parameter changes

**Error handling:** 
- Return `np.nan` for math failures, insufficient data, undefined results
- Raise exceptions for invalid input types/shapes, critical parameter violations

**Testing:** 
- 80% minimum coverage required
- Use `np.testing.assert_allclose()` for floating-point comparisons, but examine large tolerances carefully
- Mock external dependencies (SpikeInterface, MNE) in unit tests
- Test edge cases: empty arrays, NaN values, extreme values
- **Import changes**: Test all import patterns, circular imports, and IDE functionality
- **Test Organization**: Consolidate scattered test files covering same functionality into comprehensive suites

**Test markers:**
```python
@pytest.mark.unit        # Fast, isolated tests
@pytest.mark.integration # Component interaction tests  
@pytest.mark.slow        # Performance/benchmark tests
```

**Expected warnings and handling:**
- **MNE cycle warnings** (`fmin=X.X Hz corresponds to Y.Y < 5 cycles`): Expected for short test signals, acceptable in tests with `warnings.simplefilter("ignore")`
- **Divide by zero in arctanh**: Expected when coherence = 1.0 exactly, handled by z_epsilon parameter
- **SciPy nperseg warnings**: Expected when test signals shorter than default window, typically harmless

## Key Modules

- `pythoneeg/core/analysis.py` - `LongRecordingAnalyzer` main analysis orchestrator
- `pythoneeg/core/analyze_frag.py` - `FragmentAnalyzer` core feature extraction  
- `pythoneeg/core/analyze_sort.py` - Spike sorting integration
- `pythoneeg/core/core.py` - Data loading and organization classes
- `pythoneeg/visualization/results.py` - `AnimalOrganizer` organize files by animal, `WindowAnalysisResult` and `SpikeAnalysisResult` hold analysis output
- `pythoneeg/visualization/plotting/animal.py` - `AnimalPlotter` at the animal level
- `pythoneeg/visualization/plotting/experiment.py` - `ExperimentPlotter` across several animals
- `pythoneeg/constants.py` - Frequency bands, feature definitions (don't modify without scientific justification)

**Task tracking:** Use TodoWrite for multi-step implementations, mark in_progress before starting
**Branch workflow:** feature/ → main via PR  
**Code references:** Use `file_path:line_number` format (e.g., `pythoneeg/core/analysis.py:245`)

Remember: This is scientific software researchers depend on for accurate results. Prioritize correctness over performance.

**Performance vs. Accuracy Trade-offs:**
- Accept slower computations for better numerical accuracy 
- Prefer robust statistical methods even if computationally expensive
- Short epochs may trigger warnings but produce usable results for testing purposes
- Real research data should use proper epoch lengths (≥5 cycles for lowest frequency of interest)