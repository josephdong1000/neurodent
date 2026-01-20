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

SORTING_PARAMS = {
    "notch_freq": LINE_FREQ,
    "common_ref": True,
    "scale": None,
    "whiten": True,
    "freq_min": 0.1,
    "freq_max": 100,
}
"""Default parameters for MountainSort spike sorting."""

SCHEME2_SORTING_PARAMS = {
    "detect_channel_radius": 1,
    "phase1_detect_channel_radius": 1,
    "snippet_T1": 0.1,
    "snippet_T2": 0.1,
}
"""Scheme2 sorting parameters for multi-channel detection."""

WAVEFORM_PARAMS = {
    "notch_freq": LINE_FREQ,
    "common_ref": False,
    "scale": None,
    "whiten": False,
    "freq_min": None,
    "freq_max": None,
}
"""Parameters for raw waveform extraction (no preprocessing)."""
