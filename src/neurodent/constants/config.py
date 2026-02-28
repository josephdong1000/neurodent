"""Global configuration constants for neurodent.

Includes sampling rates, data types, and spike sorting parameters.
"""

import numpy as np

GLOBAL_SAMPLING_RATE = 1000
"""Default sampling rate in Hz for all recordings."""

GLOBAL_DTYPE = np.float32
"""Default NumPy data type for signal processing."""

LINE_FREQ = 60
"""Power line frequency in Hz (for notch filtering)."""


