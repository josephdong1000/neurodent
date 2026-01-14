"""Channel and metadata mapping constants.

Includes channel ID mappings, naming aliases, and sort orders.
"""

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
"""Maps Intan channel IDs to hemisphere (L/R)."""

GENOTYPE_ALIASES = {"WT": ["WT", "wildtype"], "KO": ["KO", "knockout"]}
"""Canonical genotype names mapped to their aliases."""

CHNAME_ALIASES = {
    "Aud": ["Aud", "aud", "AUD"],
    "Vis": ["Vis", "vis", "VIS"],
    "Hip": ["Hip", "hip", "HIP"],
    "Bar": ["Bar", "bar", "BAR"],
    "Mot": ["Mot", "mot", "MOT"],
}
"""Canonical brain region names mapped to their aliases."""

LR_ALIASES = {
    "L": ["left", "Left", "LEFT", "L ", " L"],
    "R": ["right", "Right", "RIGHT", "R ", " R"],
}
"""Canonical hemisphere names mapped to their aliases."""

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
"""Maps Intan channel IDs to full channel names (e.g., 'LAud')."""

DF_SORT_ORDER = {
    "channel": ["average", "all", "LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis", "LHip", "RHip"],
    "genotype": ["WT", "KO"],
    "sex": ["Male", "Female"],
    "isday": [True, False],
    "band": ["delta", "theta", "alpha", "beta", "gamma"],
}
"""Defines categorical sort orders for DataFrame columns."""

DATEPARSER_PATTERNS_TO_REMOVE = [
    r"[A-Z]+\d+",
    r"\([0-9]+\)",
    r"(?:\b\d\s){1,}(\d\b)?",
    r"\s\d$",
]
"""Regex patterns to strip from filenames before date parsing."""

DEFAULT_DAY = datetime(2000, 1, 1)
"""Fallback date when parsing fails."""
