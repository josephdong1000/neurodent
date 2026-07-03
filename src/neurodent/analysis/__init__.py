"""Analysis stage: compute features and detect spikes from loaded recordings.

Holds the single-recording analyzer (:class:`LongRecordingAnalyzer`), the
per-fragment feature computations (:class:`FragmentAnalyzer`), spike detection
(:class:`FrequencyDomainSpikeDetector`), and the per-animal
:class:`AnalysisPipeline` that turns an ``AnimalOrganizer`` into result objects.
"""

from .long_recording_analyzer import LongRecordingAnalyzer
from .fragment_analyzer import FragmentAnalyzer
from .spike_detection import FrequencyDomainSpikeDetector
from .pipeline import AnalysisPipeline

__all__ = [
    "LongRecordingAnalyzer",
    "FragmentAnalyzer",
    "FrequencyDomainSpikeDetector",
    "AnalysisPipeline",
]
