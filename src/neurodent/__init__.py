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

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__title__",
    "__summary__",
    "__uri__",
    "set_channel_map",
]
