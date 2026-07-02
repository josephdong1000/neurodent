"""Data loading and analysis orchestration (issues #110, #135, #137).

``AnimalOrganizer`` discovers and loads an animal's recordings across sessions;
``AnalysisPipeline`` runs the LOF / windowed-analysis / spike-detection steps on
them. Co-located with ``LongRecordingOrganizer`` under ``core`` because they are
data-loading, not visualization.
"""

from .animal_organizer import AnimalOrganizer
from .pipeline import AnalysisPipeline

__all__ = ["AnimalOrganizer", "AnalysisPipeline"]
