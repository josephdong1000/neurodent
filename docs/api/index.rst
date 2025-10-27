API Reference
=============

This is the complete API reference for Neurodent.

Modules
-------

.. toctree::
   :maxdepth: 2

   core/index
   visualization/index
   constants

Quick Links
-----------

**Core Classes:**

* :class:`neurodent.core.LongRecordingOrganizer` - Load and organize EEG recordings
* :class:`neurodent.core.LongRecordingAnalyzer` - Analyze recordings and compute features
* :class:`neurodent.core.FragmentAnalyzer` - Analyze individual fragments
* :class:`neurodent.core.MountainSortAnalyzer` - Interface with MountainSort spike sorting

**Visualization & Results:**

* :class:`neurodent.visualization.WindowAnalysisResult` - Windowed analysis results (WAR)
* :class:`neurodent.visualization.SpikeAnalysisResult` - Spike analysis results (SAR)
* :class:`neurodent.visualization.AnimalPlotter` - Plot single animal data
* :class:`neurodent.visualization.ExperimentPlotter` - Compare multiple animals

**Constants:**

* :mod:`neurodent.constants` - Feature names and frequency bands
