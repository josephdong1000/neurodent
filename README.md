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