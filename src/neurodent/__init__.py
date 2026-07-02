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
    "AnimalOrganizer": "neurodent.core.loading",
    "LongRecordingOrganizer": "neurodent.core",
    "LongRecordingAnalyzer": "neurodent.core",
    "WindowAnalysisResult": "neurodent.core.results",
    "FrequencyDomainSpikeAnalysisResult": "neurodent.core.results",
    "ZeitgeberAnalysisResult": "neurodent.core.results",
    "AnimalPlotter": "neurodent.visualization",
    "ExperimentPlotter": "neurodent.visualization",
    "ZeitgeberPlotter": "neurodent.visualization",
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
