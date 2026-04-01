---
title: 'NeuRodent: a Python Package and Pipeline for Unifying Rodent EEG Analyses'
tags:
- neuroscience
- electroencephalography
- local field potential
- pipeline
- python
authors:
- name: Joseph P. Dong
  orchid: 0009-0001-8636-6534
  affiliation: 1
- name: Oh Yongtaek
  orchid: 0000-0002-1723-0553
  affiliation: 1
- name: Eric D. Marsh
  orchid: 0000-0003-3264-0902
  affiliation: "1, 2"
affiliations:
- name: Children's Hospital of Philadelphia, United States
  index: 1
  ror: 01z7r7q48
- name: University of Pennslyvania, United States
  index: 2
  ror: 00b30xv10
date: 31 March 2026
bibliography: paper.bib
---

# Statement of Need

Electroencephalography (EEG) and its invasive counterpart, local field potential (LFP) recording, are cornerstone neuroscience techniques to measure cortical brain activity across species from humans to rodents. EEG has had over a century of use in clinical settings ([Hans Berger (1873-1941)--the history of electroencephalography] - PubMed) and is widely used today to diagnose and localize epilepsy, as well as in animal models to validate and develop mechanistic understandings of neurological diseases (Modelling epilepsy in the mouse: challenges and solutions | Disease Models & Mechanisms | The Company of Biologists ). Historically, EEG interpretation has been qualitative and “expert” based, but quantitative analysis approaches have gained acceptance over the last 30 years (REFS). Most laboratories have developed quantitative methods independently and several free and commercial software packages exist for rodent EEG analysis. However, a significant lack of interoperability remains between these tools in handling variability in data type, experimental design, and statistical analysis. This fragmentation forces researchers to expend excessive time and productivity on technical troubleshooting rather than scientific discovery. 

`NeuRodent` was developed to address these challenges by providing a modular, scalable, and interoperable Python-based framework specifically designed for neuroscience researchers working with large-scale rodent cohorts. By unifying analysis pipelines, `NeuRodent` enables seamless collaboration and analysis comparisons across different laboratories. Key technical features include: 

- **Modular Architecture**: A generalized framework for feature calculation allows contributors to easily extend the library of available metrics. 
- **Data organization**: A dedicated scheme designed for rodent study analysis, including genotype and experimental day, rather than individual subject sessions. 
- **High Interoperability**: Integration with SpikeInterface and MNE-Python ensures support for a wide array of neuroscience file formats with syntax frequently used in the field. 
- **Scalability**: To address the challenge of analyzing large EEG datasets efficiently, the package integrates dataset caching and uses Dask (Rocklin, 2015) and Snakemake (citation) to parallelize computations across high-performance computing (HPC) clusters. 
- **Reproducibility**: Development follows Continuous Integration (CI) practices, and intermediate results are saved to prevent redundant computations following pipeline errors. 

# State of the Field 

The current landscape of electrophysiology and neuroimaging software includes several mature, high-level tools such as SpikeInterface (Buccino et al., 2020) and MNE-Python (Gramfort  et al., 2013) in Python, as well as EEGLAB (Delorme & Makeig, 2004), FieldTrip (Oostenveld et al., 2011), Brainstorm (Tadel et al., 2011), and SPM (Litvak  et al., 2011) in MATLAB. These platforms provide powerful building blocks for neurophysiology analysis but are primarily oriented toward human-centric research, high-frequency spike sorting, or local single-session computation. As a result, most rodent EEG analyses remain ad-hoc and designed for local, single-session use rather than maintained and scalable applications. `NeuRodent` provides a unique scholarly contribution and addresses this gap by serving as an orchestration layer that coodinates data loading, analysis, and visualization into reproducible and rodent-specific EEG workflows. 

# Software Design 

`NeuRodent` is organized as two independently installable components: a core Python analysis library, and a Snakemake pipeline that orchestrates the library over large datasets. This layered design balances ease of adoption with scalable deployment: per-session or per-animal analyses can use the core library alone and call it from scripts or notebooks, while multi-animal and multi-session processing can use the Snakemake pipeline to gain cluster-level orchestration using SLURM or Kubernetes. The two components are decoupled, ensuring that changes to the pipeline logic do not force changes on users of the core library, and vice versa. 

Rather than implementing its own file readers, `NeuRodent` delegates data loading to SpikeInterface and MNE-Python [citations], which together cover most electrophysiology formats in use. Users may also supply a custom reader function for novel formats. This deferred approach avoids duplicating format-support effort that is already well maintained by those communities and ensures that `NeuRodent` inherits new format support automatically. 

Within the core library, computation is structured around a hierarchy of organizer classes that mirror the stages of a rodent EEG experiment (figure): 

- **LongRecordingOrganizer**: one recording session, many channels 
- **AnimalOrganizer**: one animal, many recording sessions 
- **WindowAnalysisResult**, **FrequencyDomainSpikeAnalysisResult**: analysis results of one animal 
- **AnimalPlotter**: plots from one animal 
- **ExperimentPlotter**: plots from many animals 

Each of these classes naturally encapsulates a specific scope of rodent EEG/LFP analysis. A nested class hierarchy was chosen over a flat library to make the hierarchy of EEG analysis explicit, with lower level objects composing higher level ones. A practical consequence of this design is that analysis can be embarrassingly parallelized by processing each channel and time window independently. `NeuRodent` uses Dask to enable configurable parallel processing of channels and windows, either locally or on a distributed cluster. Adjustable in-memory chunk sizes let users trade throughput for RAM, an important consideration given that EEG recordings can span days or weeks. 

`NeuRodent` enables contributors to add new features to compute in windowed analyses by discovering feature computation functions at runtime. This greatly reduces the barrier to contribution for domain scientists who may not be familiar with the broader structure of `NeuRodent`. Artifact rejection is done in a similar fashion, where users can write additional filters and apply them with minimal changes. All computed features are outputted as Pandas DataFrames saved in Parquet files, which enables downstream workflows in Excel, R, or other analysis tools to interoperate with `NeuRodent` outputs without needing format conversion. 

`NeuRodent` is distributed via PyPI and conda-forge for ease of installation and is published on the Snakemake workflow catalog. Continuous integration tests cover more than 90% of the core library, and the full Snakemake workflow is additionally tested end-to-end on a miniature example dataset, ensuring that changes to the library do not silently break the Snakemake pipeline. Clear explanations of installation instructions and tutorials are provided on the `NeuRodent` documentation website. Contributing guidelines and a Makefile-based development setup are provided so that new contributors can get started with a few commands. 

# Research Impact Statement 

Originally designed for EEG analyses for the lead developer, `NeuRodent` is being developed by a team of six code contributors using data across laboratories. The package has been used in a submitted manuscript at the time of writing (preprint) and presented at the US Research Software Engineering conference [citation to USRSE proceedings]. `NeuRodent` and its accompanying Snakemake pipeline will be of great use to neuroscientists performing intracranial LFP analyses on large recorded datasets and distributing analyses across laboratories. 

# AI Usage Disclosure 

Claude Opus 4.6 via GitHub Copilot and Claude Code was used to assist with code development and documentation writing. AI-generated code and tests were manually reviewed and edited by human authors, and correctness was validated against large datasets used by our group and collaborating groups. All architectural decisions and scientific interpretations were made by human authors. 

# Acknowledgements 

Funding information 

We acknowledge contributions from Yastika Singh, Yifan Wei, Jessica Lahr, and Ananya Madhira. This project benefited from insights and best practices described in Peter K. G. Williams’s One Good Tutorial. 
 
# References 