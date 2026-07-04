"""NeuRodent: Rodent EEG analysis tools."""

import importlib.metadata

__version__ = importlib.metadata.version("neurodent")
__author__ = "Joseph Dong, Yongtaek Oh, Eric Marsh"
__email__ = "dongjp@chop.edu"
__license__ = "MIT"
__title__ = "neurodent"
__summary__ = "Rodent EEG analysis tools"
__uri__ = "https://github.com/josephdong1000/neurodent"

from .constants import set_channel_map

# Stage-based headline classes, resolved lazily on first access (PEP 562) so a
# bare ``import neurodent`` stays cheap and never eager-loads the plotting stack.
_LAZY_EXPORTS = {
    "AnimalOrganizer": "neurodent.loading",
    "LongRecordingOrganizer": "neurodent.loading",
    "LongRecordingAnalyzer": "neurodent.analysis",
    "WindowAnalysisResult": "neurodent.results",
    "FrequencyDomainSpikeAnalysisResult": "neurodent.results",
    "ZeitgeberAnalysisResult": "neurodent.results",
    "AnimalPlotter": "neurodent.plotting",
    "ExperimentPlotter": "neurodent.plotting",
    "ZeitgeberPlotter": "neurodent.plotting",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        import importlib

        return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY_EXPORTS])


__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__title__",
    "__summary__",
    "__uri__",
    "set_channel_map",
    *_LAZY_EXPORTS,
]
