"""**Visualization module** for organizing analysis results and creating plots.

This module provides tools to aggregate, organize, and visualize the outputs of
the :mod:`neurodent.core` analysis pipeline across animals and experiments.

**Typical Workflow:**

.. code-block:: python

    from neurodent import visualization as viz
    from neurodent.core import LongRecordingAnalyzer

    # 1. Wrap analysis results for easier access
    war = viz.WindowAnalysisResult.load('analysis_results.pkl')

    # 2. Aggregate across sessions for one animal
    organizer = viz.AnimalOrganizer([war1, war2, war3])
    combined_df = organizer.get_combined_features()

    # 3. Plot temporal heatmaps
    plotter = viz.AnimalPlotter(organizer)
    fig = plotter.plot_temporal_heatmap(feature='psdband', band='theta')

    # 4. Experiment-wide comparisons (multiple animals)
    exp_plotter = viz.ExperimentPlotter([animal1_org, animal2_org])
    fig = exp_plotter.plot_genotype_comparison(feature='cohere')

**Output Types:**

- **Temporal heatmaps**: Feature values over time (x) and channels (y)
- **Band comparisons**: Power across frequency bands
- **Statistical plots**: Genotype/condition comparisons with error bars
- **Export to CSV/TIF**: Publication-ready outputs

**See Also:**

- :doc:`/tutorials/visualization` - Plotting examples
- :doc:`/tutorials/windowed_analysis` - Understanding result structure
- :mod:`neurodent.constants.plotting` - Color palettes and styling
"""

from .results import (
    WindowAnalysisResult,
    AnimalFeatureParser,
    AnimalOrganizer,
)
from .plotting import (
    AnimalPlotter,
    ExperimentPlotter,
    ZeitgeberPlotter,
)
from .frequency_domain_results import FrequencyDomainSpikeAnalysisResult

__all__ = [
    "WindowAnalysisResult",
    "AnimalFeatureParser",
    "AnimalOrganizer",
    "FrequencyDomainSpikeAnalysisResult",
    "AnimalPlotter",
    "ExperimentPlotter",
    "ZeitgeberPlotter",
]
