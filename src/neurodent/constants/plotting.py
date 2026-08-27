"""Plotting style constants.

Includes color palettes and figure layout settings.
"""

FEATURE_PLOT_HEIGHT_RATIOS = {
    "rms": 1,
    "ampvar": 1,
    "psdtotal": 1,
    "psdslope": 2,
    "psdband": 5,
    "psdfrac": 5,
    "nspike": 1,
    "logrms": 1,
    "logampvar": 1,
    "logpsdtotal": 1,
    "lognspike": 1,
    "logpsdband": 5,
    "logpsdfrac": 5,
    "cohere": 5,
    "zcohere": 5,
    "imcoh": 5,
    "zimcoh": 5,
    "pcorr": 1,
    "zpcorr": 1,
}
"""Height ratios for multi-feature subplot layouts."""

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
"""Okabe-Ito colorblind-friendly palette.

Reference: https://easystats.github.io/see/reference/scale_color_okabeito.html
"""

