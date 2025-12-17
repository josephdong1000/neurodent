from datetime import datetime

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
]

DEFAULT_DAY = datetime(2000, 1, 1)
