# NeuRodent 🐁

[![CI](https://github.com/josephdong1000/neurodent/actions/workflows/test-build-docs.yml/badge.svg)](https://github.com/josephdong1000/neurodent/actions/workflows/test-build-docs.yml)
[![Coverage](https://codecov.io/gh/josephdong1000/neurodent/branch/main/graph/badge.svg)](https://codecov.io/gh/josephdong1000/neurodent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/neurodent)](https://pypi.org/project/neurodent/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15780751.svg)](https://doi.org/10.5281/zenodo.15780751)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/josephdong1000/neurodent/HEAD)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/josephdong1000/neurodent/)

A Python package for standardizing rodent EEG analysis and figure generation. Various EEG formats are loadable and features are extracted in parallel. Also includes a Snakemake workflow for automated analysis.

> Presented at [USRSE'25](https://doi.org/10.5281/zenodo.17274681)!

### What it does

NeuRodent loads EEG/LFP recordings and computes features across recording channels and time. Features include signal amplitude, power in standard brainwave frequency bands, power spectrum slope, coherence/correlation between pairs of channels, and voltage spikes that occur when many neurons are activated at once. Results are saved as tables and can be visualized for single animal or multiple animal recordings, enabling comparison across experimental groups.

### Why NeuRodent

Most laboratories have developed quantitative methods independently, and several free and commercial software packages exist for rodent EEG analysis. However, a significant lack of interoperability remains between these tools in handling variability in data type, experimental design, and statistical analysis. This fragmentation forces researchers to expend excessive time and productivity on technical troubleshooting rather than scientific discovery. NeuRodent addresses this by providing a modular, scalable, and interoperable Python-based framework for neuroscience researchers working with large-scale rodent cohorts, so that analyses can be compared and shared across laboratories.

### Design

NeuRodent is organized as two components: a core Python analysis library, and a Snakemake pipeline that orchestrates the library over large datasets and is installed with the `neurodent[pipeline]` extra. Per-session or per-animal analyses can use the core library alone from scripts or notebooks, while multi-animal and multi-session processing can use the pipeline for cluster-level orchestration on SLURM. Within the library, computation is structured around a hierarchy of organizer classes that mirror the stages of a rodent EEG experiment:

- `LongRecordingOrganizer`: one recording session, many channels
- `AnimalOrganizer`: one animal, many recording sessions
- `WindowAnalysisResult`, `FrequencyDomainSpikeAnalysisResult`, `ZeitgeberAnalysisResult`: analysis results of one animal
- `AnimalPlotter`: plots from one animal
- `ExperimentPlotter`, `ZeitgeberPlotter`: plots from many animals

Rather than implementing its own file readers, NeuRodent delegates data loading to SpikeInterface and MNE-Python, and a custom reader function can be supplied for novel formats.

### Who it is for

Rodent EEG researchers who need reproducible and standardized signal analyses across their studies, whether that is a handful of pilot recordings on a laptop or a cohort recorded over months on a cluster.

## Installation

NeuRodent can be installed via `pip` or `conda`:

```bash
pip install neurodent
```

or

```bash
conda install -c conda-forge neurodent
```

For pipeline support, development setup, and other installation options, check out the [full installation guide](https://josephdong1000.github.io/neurodent/main/installation/index.html).

## Usage

> **Visit the full documentation** for more how-tos and examples:
> https://josephdong1000.github.io/neurodent

- [Quickstart](https://josephdong1000.github.io/neurodent/main/quickstart/index.html)
- [Tutorials](https://josephdong1000.github.io/neurodent/main/tutorials/index.html)  
- [API documentation](https://josephdong1000.github.io/neurodent/main/api/index.html)

## Overview

NeuRodent loads multi-format EEG data (`LongRecordingOrganizer` → `AnimalOrganizer`) and computes features over windows (`WindowAnalysisResult`) and population spiking (`FrequencyDomainSpikeAnalysisResult`). Results feed into `AnimalPlotter` and `ExperimentPlotter` for multi-animal comparison by genotype, session, or circadian cycle.

```python
from neurodent import AnimalOrganizer, AnimalAnalyzer, ExperimentPlotter

features = ["rms", "psdband", "cohere"]

# AnimalOrganizer discovers recordings from a placeholder pattern
ao = AnimalOrganizer(
    pattern="data/{animal}/*.edf",
    animal_id="A10",
    lro_kwargs={"mode": "si", "extract_func": "read_edf"},
)
war = AnimalAnalyzer(ao).compute_windowed_analysis(features=features)

ep = ExperimentPlotter([war], features=features)
ep.plot_catplot("rms", groupby="genotype")
```

A companion [Snakemake workflow](https://josephdong1000.github.io/neurodent/main/tutorials/index.html) automates the full pipeline with cluster support.

## Snakemake Workflow

The pipeline follows the [Snakemake Workflow Catalog](https://snakemake.github.io/snakemake-workflow-catalog/) standardized layout, with `workflow/Snakefile` as the single entry point.

```bash
# Deploy via snakedeploy
pip install snakedeploy
snakedeploy deploy-workflow https://github.com/josephdong1000/neurodent . --tag <version>

# Run manually
snakemake --snakefile workflow/Snakefile --configfile config/config.yaml
```

## Acknowledgements

This project benefited from insights and best practices described in Peter K. G. Williams’s [One Good Tutorial](https://onegoodtutorial.org/).

## Citation

If you find NeuRodent useful, please cite our work!

```bibtex
@software{dong2026neurodent,
  author    = {Dong, Joseph P. and Oh, Yongtaek and Marsh, Eric D.},
  title     = {NeuRodent},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15780751},
  url       = {https://doi.org/10.5281/zenodo.15780751},
  license   = {MIT}
}
```

Machine-readable citation metadata is also available in [CITATION.cff](./CITATION.cff).

## Community & Governance

- Contributing: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Support: [SUPPORT.md](./SUPPORT.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)