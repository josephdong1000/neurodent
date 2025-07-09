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
    "Aud": ["Aud", "aud"],
    "Vis": ["Vis", "vis"],
    "Hip": ["Hip", "hip"],
    "Bar": ["Bar", "bar"],
    "Mot": ["Mot", "mot"],
    # 'S' : ['Som', 'som']
}
LR_ALIASES = {
    "L": ["left", "Left", "L ", " L"],
    "R": ["right", "Right", "R ", " R"],
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
PLOT_ORDER = {
    "channel": ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis", "LHip", "RHip"],
    "genotype": ["WT", "KO"],
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
MATRIX_FEATURES = ["cohere", "pcorr", "zpcorr"]
HIST_FEATURES = ["psd"]
FEATURES = LINEAR_FEATURES + BAND_FEATURES + MATRIX_FEATURES + HIST_FEATURES
WAR_FEATURES = [f for f in FEATURES if "nspike" not in f]

LINPLOT_HEIGHT_RATIOS = {
    "rms": 1,  # NOTE add in log features?
    "ampvar": 1,
    "psdtotal": 1,
    "psdslope": 2,
    "psdband": 5,
    "psdfrac": 5,
    "nspike": 1,
}

FREQ_BANDS = {
    "delta": (0.1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 25),
    "gamma": (25, 40),
}  # TODO try delta 1-4 instead
BAND_NAMES = [k for k, _ in FREQ_BANDS.items()]

FREQ_BAND_TOTAL = (0.1, 40)
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
