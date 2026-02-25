# NeuRodent 🐁

[![CI](https://github.com/josephdong1000/neurodent/actions/workflows/test-build-docs.yml/badge.svg)](https://github.com/josephdong1000/neurodent/actions/workflows/test-build-docs.yml)
<!-- [![Coverage](https://codecov.io/gh/josephdong1000/neurodent/branch/main/graph/badge.svg)](https://codecov.io/gh/josephdong1000/neurodent) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/neurodent)](https://pypi.org/project/neurodent/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15780751.svg)](https://doi.org/10.5281/zenodo.15780751)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/josephdong1000/neurodent/HEAD)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/josephdong1000/neurodent/)


> Presented at [USRSE'25](https://doi.org/10.5281/zenodo.17274681)!

A Python package for standardizing rodent EEG analysis and figure generation. Various EEG formats are loadable and features are extracted in parallel. Also includes a Snakemake workflow for automated analysis.

## Installation

NeuRodent can be installed via `pip`:

```bash
pip install neurodent
```

For pipeline support, development setup, and other installation options, check out the [full installation guide](https://josephdong1000.github.io/neurodent/main/installation/index.html).

## Usage

> **Visit the full documentation** for more how-tos and examples:
> https://josephdong1000.github.io/neurodent

- [Quickstart](https://josephdong1000.github.io/neurodent/main/quickstart/index.html)
- [Tutorials](https://josephdong1000.github.io/neurodent/main/tutorials/index.html)  
- [API documentation](https://josephdong1000.github.io/neurodent/main/api/index.html)

## Overview

NeuRodent loads multi-format EEG data (`LongRecordingAnalyzer` → `AnimalOrganizer`) and computes features over windows (`WindowAnalysisResult`) and population spiking (`FrequencyDomainSpikeAnalysisResult`). Results feed into `AnimalOrganizer` and `ExperimentPlotter` for multi-animal comparison by genotype, session, or circadian cycle.

```python
lro = LongRecordingOrganizer(data_path)
ao = AnimalOrganizer(lro)
war = ao.compute_windowed_analysis(features=["rms", "psdband", "cohere"])
ep = ExperimentPlotter([war])
ep.plot_feature("rms", groupby="genotype")
```

A companion [Snakemake workflow](https://josephdong1000.github.io/neurodent/main/tutorials/index.html) automates the full pipeline with cluster support.

## Acknowledgements

This project benefited from insights and best practices described in Peter K. G. Williams’s [One Good Tutorial](https://onegoodtutorial.org/).

## Citation

If you find NeuRodent useful, please cite our work!

```bibtex
@misc{https://doi.org/10.5281/zenodo.17051374,
  doi = {10.5281/ZENODO.17051374},
  url = {https://zenodo.org/doi/10.5281/zenodo.17051374},
  author = {Dong,  Joseph and Yongtaek Oh,   and Marsh,  Eric},
  title = {josephdong1000/PyEEG: 0.1.1},
  publisher = {Zenodo},
  year = {2025},
  copyright = {MIT License}
}
```