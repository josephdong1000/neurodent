# PythonEEG

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15780751.svg)](https://doi.org/10.5281/zenodo.15780751)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/josephdong1000/PyEEG/HEAD) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/josephdong1000/PyEEG/)

> Presented at [USRSE'25](https://doi.org/10.5281/zenodo.17274681)!

A Python package for standardizing rodent EEG analysis and figure generation. Various EEG formats are loadable and features are extracted in parallel. Also includes a Snakemake workflow for automated analysis.

## Overview

PyEEG provides two main analysis workflows:

1. **Windowed Analysis Results (WAR)** - Extracts features from continuous EEG data divided into time windows
2. **Spike Analysis Results (SAR)** - Analyzes spike-sorted neural data and integrates it with EEG features

The library supports multiple data formats (binary files, SpikeInterface recordings, MNE objects) and includes parallel processing capabilities using Dask for large datasets.

### Features Extracted

#### Linear Features (single values per channel)
- **RMS amplitude** - Root mean square of the signal
- **Log RMS amplitude** - Logarithm of RMS amplitude  
- **Amplitude variance** - Variance of signal amplitude
- **Log amplitude variance** - Logarithm of amplitude variance
- **PSD total power** - Total power spectral density across frequency band
- **Log PSD total power** - Logarithm of total PSD power
- **PSD slope** - Slope of power spectral density on log-log scale
- **Spike count** - Number of detected spikes
- **Log spike count** - Logarithm of spike count

#### Band Features (values per frequency band)
- **PSD band power** - Power spectral density for each frequency band
- **Log PSD band power** - Logarithm of PSD band power
- **PSD fractional power** - PSD band power as fraction of total power
- **Log PSD fractional power** - Logarithm of PSD fractional power

#### Connectivity Features
- **Coherence** - Spectral coherence between channels
- **Pearson correlation** - Pearson correlation coefficient between channels

#### Frequency Domain
- **Power Spectral Density** - Full power spectral density with frequency coordinates

#### Frequency Bands
- **Delta**: 0.1-4 Hz, **Theta**: 4-8 Hz, **Alpha**: 8-13 Hz, **Beta**: 13-25 Hz, **Gamma**: 25-40 Hz

### How to Use PyEEG

Documentation can be found at https://josephdong1000.github.io/PyEEG/

#### Basic Workflow
1. **Load Data**: Use `LongRecordingOrganizer` to load EEG recordings from various formats
2. **Windowed Analysis**: Create `AnimalOrganizer` to compute features across time windows
3. **Spike Analysis**: Integrate spike-sorted data from `MountainSortAnalyzer`
4. **Visualization**: Generate plots using `ExperimentPlotter` and `AnimalPlotter`

#### Example Usage
```python
# Load and organize recordings
lro = LongRecordingOrganizer(data_path, mode="bin")
ao = AnimalOrganizer(lro)

# Compute windowed analysis
war = ao.compute_windowed_analysis(features=["rms", "psdband", "cohere"])

# Generate plots
ep = ExperimentPlotter([war])
ep.plot_feature("rms", groupby="genotype")
```

#### Advanced Features
- **Flexible Data Loading**: PyEEG uses MNE and SpikeInterface loaders in Python and custom loaders for proprietary formats using MATLAB, including:
  - Neuroscope/Neuralynx (.dat, .eeg)
  - Open Ephys (.continuous)
  - NWB (.nwb) neurophysiology format
  - Binary (.bin) files
- **Bad Channel Detection**: Automatic identification of bad channels using Local Outlier Factor
- **Multi-processing**: Parallel/distributed processing with Dask for large datasets
- **Data Filtering**: Built-in filtering for artifacts and outliers
- **Flexible Grouping**: Group analysis by genotype, time of day, recording session, etc.

## Snakemake Workflow

A companion Snakemake workflow is provided for building automated PyEEG analysis pipelines.

The workflow processes multiple animals in parallel through WAR generation, quality filtering, fragment/channel filtering, and statistical analysis with SLURM cluster integration.

To run Snakemake on a specific computing environment, first create a [Snakemake profile](https://github.com/snakemake-profiles/doc), then run Snakemake:

```bash
# Run the complete workflow
snakemake
```

## Setup (for developing)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate
# Install requirements
pip install -r requirements.txt
```

### Planned features
- Cross-frequency coupling (CFC)
  - Phase-Locking Value (PLV)
    - [phase-phase]
  - Phase-Lag Index (PLI / wPLI / dPLI)
    - [phase-phase]
    - wPLI is resistant to volume conduction
    - dPLI determines direction (leading/lagging)
  - Phase-Amplitude Coupling (PAC)
    - [phase-amplitude]
    - Driven auto-regressive (DAR) model from `pactools`
  - Amplitude-Envelope Coupling (AEC)
    - [amplitude-amplitude]
- Spike-LFP coupling
  <!-- - **Caveat**: is this legitimate for population spiking?
    - I only see papers for single unit spiking
    - Maybe useful for examining interictal cortical discharges -->
  - Pairwise-Phase Consistency (PPC)
    - [spike-phase]
    - Corrected PPC (Vinck 2011)
- Canonical coherence
  - [channels-channels]
- Peri-spike EEG
