# PythonEEG

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/josephdong1000/PyEEG/HEAD) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/josephdong1000/PyEEG/)

**Extracts features from mouse EEGs and generates figures**

## Setup (for developing)

- Install Microsoft Visual C++ 14.0 or greater (to get SpikeInterface to work)
  - https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - Minimal install required, Visual Studio Build Tools 2022 should be sufficient
- Have Python installed
  - https://www.python.org/downloads/
- Set up the Python environment
  - Create environment: `python -m venv .venv`
  - Activate environment: `source .venv/bin/activate`
    - Check that `which python` returns the .venv environment
  - Install requirements: `pip install -r requirements.txt`
- Install VSCode, or other appropriate development software
  - https://code.visualstudio.com/
- Program demo is in `notebooks/examples/pipeline.ipynb`. Set the notebook kernel to `.venv`

A module version of this library will be released for production

## Features

EEGs are loaded from companion Matlab code that converts .DDF to .BIN. Linux and Mac are supported (Python 3.10). Windows was supported in an earlier version but is currently untested.

- RMS amplitude
- Amplitude variance
- PSD band power
- PSD slope
- Coherence
- Pearson correlation

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
  - **Caveat**: is this legitimate for population spiking?
    - I only see papers for single unit spiking
    - Maybe useful for examining interictal cortical discharges
  - Pairwise-Phase Consistency (PPC)
    - [spike-phase]
    - Corrected PPC (Vinck 2011)
- Canonical coherence
  - [channels-channels]
- Peri-spike EEG
- GUI

## Tasks

- [x] Proper git/github repo setup
- [x] Summary box plots
- [x] Parallelized feature computation
- [ ] Test run over all Marsh dataset
- [ ] Spike sorting
- [ ] Dimensionality reduction
  - [ ] PCA
  - [ ] UMAP / qUMAP
- [ ] Cross-frequency coherence
- [ ] Canonical coherence
- [ ] Peri-spike EEG (with MNE)
- [ ] GUI

