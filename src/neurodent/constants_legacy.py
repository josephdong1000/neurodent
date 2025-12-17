from datetime import datetime

import numpy as np

DEFAULT_ID_TO_LR = {
    9: "L",
    10: "L",
    12: "L",
    14: "L",
    15: "L",
    16: "R",
    17: "R",
    19: "R",
    21: "R",
    22: "R",
}

GENOTYPE_ALIASES = {"WT": ["WT", "wildtype"], "KO": ["KO", "knockout"]}
CHNAME_ALIASES = {
    "Aud": ["Aud", "aud", "AUD"],
    "Vis": ["Vis", "vis", "VIS"],
    "Hip": ["Hip", "hip", "HIP"],
    "Bar": ["Bar", "bar", "BAR"],
    "Mot": ["Mot", "mot", "MOT"],
    # 'S' : ['Som', 'som']
}
LR_ALIASES = {
    "L": ["left", "Left", "LEFT", "L ", " L"],
    "R": ["right", "Right", "RIGHT", "R ", " R"],
}

DEFAULT_ID_TO_NAME = {
    9: "LAud",
    10: "LVis",
    12: "LHip",
    14: "LBar",
    15: "LMot",
    16: "RMot",
    17: "RBar",
    19: "RHip",
    21: "RVis",
    22: "RAud",
}
DF_SORT_ORDER = {
    "channel": ["average", "all", "LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis", "LHip", "RHip"],
    "genotype": ["WT", "KO"],
    "sex": ["Male", "Female"],
    "isday": [True, False],
    "band": ["delta", "theta", "alpha", "beta", "gamma"],
}

DATEPARSER_PATTERNS_TO_REMOVE = [
    r"[A-Z]+\d+",  # Matches patterns like 'A5', 'G20'
    r"\([0-9]+\)",  # Matches patterns like '(2)', '(15)'
    r"(?:\b\d\s){1,}(\d\b)?",
    r"\s\d$",
    # r'WT',             # Common lab identifiers
    # r'KO',
    # r'Mouse[- ]?',     # Mouse with optional hyphen/space
    # r'test',
    # r'_+',             # Multiple underscores
    # r'\.+',            # Multiple dots
    # r'\s\d\s',         # Single numbers
    # r'^\d\s'
]
DEFAULT_DAY = datetime(2000, 1, 1)

GLOBAL_SAMPLING_RATE = 1000
GLOBAL_DTYPE = np.float32

LINEAR_FEATURES = [
    "rms",
    "ampvar",
    "psdtotal",
    "psdslope",
    "nspike",
    "logrms",
    "logampvar",
    "logpsdtotal",
    "lognspike",
]
BAND_FEATURES = ["psdband", "psdfrac"] + ["logpsdband", "logpsdfrac"]
MATRIX_FEATURES = ["cohere", "zcohere", "imcoh", "zimcoh", "pcorr", "zpcorr"]
HIST_FEATURES = ["psd"]
FEATURES = LINEAR_FEATURES + BAND_FEATURES + MATRIX_FEATURES + HIST_FEATURES
WAR_FEATURES = [f for f in FEATURES if "nspike" not in f]

FEATURE_PLOT_HEIGHT_RATIOS = {
    # Linear features (across channels or channels x bands)
    "rms": 1,  # NOTE add in log features?
    "ampvar": 1,
    "psdtotal": 1,
    "psdslope": 2,
    "psdband": 5,
    "psdfrac": 5,
    "nspike": 1,
    # Matrix features (heatmaps of flattened matrices for spectral analysis)
    "cohere": 5,
    "zcohere": 5,
    "imcoh": 5,
    "zimcoh": 5,
    "pcorr": 1,
    "zpcorr": 1,
}

FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 25),
    "gamma": (25, 40),
}
"""Dictionary of frequency band ranges in Hz.

Delta band adjusted to 1-4 Hz (changed from 0.1-4 Hz) for reliable coherence
estimation with short epochs and to avoid insufficient cycles warnings.
"""
BAND_NAMES = [k for k, _ in FREQ_BANDS.items()]

FREQ_BAND_TOTAL = (1, 40)  # Updated to match new delta band minimum
FREQ_MINS = [v[0] for _, v in FREQ_BANDS.items()]
FREQ_MAXS = [v[1] for _, v in FREQ_BANDS.items()]
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

# Okabe-Ito colorblind-friendly color palette
# Reference: https://easystats.github.io/see/reference/scale_color_okabeito.html
OKABE_ITO_COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "blue": "#0072B2",
    "green": "#009E73",
    "yellow": "#F5C710",
    "lightblue": "#56B4E9",
    "red": "#D55E00",
    "purple": "#CC79A7",
}

# Convenience exports for backwards compatibility with okabeito package
black = OKABE_ITO_COLORS["black"]
orange = OKABE_ITO_COLORS["orange"]
blue = OKABE_ITO_COLORS["blue"]
green = OKABE_ITO_COLORS["green"]
yellow = OKABE_ITO_COLORS["yellow"]
lightblue = OKABE_ITO_COLORS["lightblue"]
red = OKABE_ITO_COLORS["red"]
purple = OKABE_ITO_COLORS["purple"]
