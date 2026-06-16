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

NEURODENT_SIDECAR_NAME = "neurodent_lro.json"
"""Filename of the LRO metadata sidecar written inside a saved recording folder.

Written by :meth:`LongRecordingOrganizer.save_recording` and read back by
:meth:`LongRecordingOrganizer.load_recording` to restore LRO-level metadata
(timestamps, durations, units, bad channels, labels) that the raw
SpikeInterface recording folder does not carry.
"""


