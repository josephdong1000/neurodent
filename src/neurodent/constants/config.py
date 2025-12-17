import numpy as np

GLOBAL_SAMPLING_RATE = 1000
GLOBAL_DTYPE = np.float32

LINE_FREQ = 60

SORTING_PARAMS = {
    "notch_freq": LINE_FREQ,
    "common_ref": True,
    # 'common_ref' : False,
    "scale": None,
    "whiten": True,
    # 'whiten' : False,
    "freq_min": 0.1,
    "freq_max": 100,
}

SCHEME2_SORTING_PARAMS = {
    "detect_channel_radius": 1,
    "phase1_detect_channel_radius": 1,
    "snippet_T1": 0.1,
    "snippet_T2": 0.1,
}

WAVEFORM_PARAMS = {
    "notch_freq": LINE_FREQ,
    "common_ref": False,
    "scale": None,
    "whiten": False,
    "freq_min": None,
    "freq_max": None,
}
