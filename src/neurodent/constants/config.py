"""Global configuration constants for neurodent.

Includes sampling rates, data types, and spike detection parameters.
"""

import numpy as np

GLOBAL_SAMPLING_RATE = 1000
"""Default sampling rate in Hz for all recordings."""

GLOBAL_DTYPE = np.float32
"""Default NumPy data type for signal processing."""

LINE_FREQ = 60
"""Power line frequency in Hz (for notch filtering)."""

RECORDING_TIME_TOLERANCE_S = 10.0
"""Largest overlap or gap, in seconds, between consecutive files of one animal that counts
as clock jitter rather than as a misplaced timestamp or a hole.

The same value applies to every format and dataset. Measured benign overlaps on the
datasets in use: up to 2.47 s between DataWave files whose starts come from the sidecar
LastEdit (99th percentile 0.91 s over 928 neighbouring pairs), up to 3.5 s at sox5_bin
session boundaries, 0.48 s between whole-second Intan session anchors, and 0.7 s between
EDF header anchors. The smallest real misplacement worth catching is one file, and the
shortest files in use are 60 s. A larger overlap raises; a larger gap is logged.
"""

NEURODENT_SIDECAR_NAME = "neurodent_lro.json"
"""Filename of the LRO metadata sidecar written inside a saved recording folder.

Written by :meth:`LongRecordingOrganizer.save_recording` and read back by
:meth:`LongRecordingOrganizer.load_recording` to restore LRO-level metadata
(timestamps, durations, units, bad channels, labels) that the raw
SpikeInterface recording folder does not carry.
"""


