"""Channel and metadata mapping constants.

The channel model is a **flat list of channels**: each canonical channel abbreviation
(``"LMot"``, ``"LHip"``, …) is the atomic unit, declared once in :data:`CHANNEL_MAP`
together with the raw channel-name spellings that resolve to it. Everything else about
channels — the canonical order (:data:`CHANNEL_ABBREVS`), the DataFrame sort order
(:data:`DF_SORT_ORDER`), the standardization target, and the LOF evaluation set — is
**derived** from that single source via :func:`neurodent.constants.set_channel_map`.
There is no separate region/hemisphere model; left/right is part of the channel identity.
"""

DEFAULT_GENOTYPE_MAP = {}
"""Module default for :data:`GENOTYPE_MAP` (empty = passthrough). ``apply_samples_config``
resets the live map to this when a dataset omits a ``GENOTYPE_MAP`` block, so applying one
dataset's config never leaks its map into the next."""

GENOTYPE_MAP = dict(DEFAULT_GENOTYPE_MAP)
"""Exact ``{canonical_label: [accepted spellings]}`` map normalizing the per-animal
``genotype`` value to a canonical short label (parallels :data:`CHANNEL_MAP` and
:data:`SEX_MAP`; matched **exactly**, never fuzzily). Empty default = passthrough (the
raw ``genotype`` value is kept as-is, for datasets that don't need normalization).
When populated it is **authoritative**: a value it does not cover raises at load (just
as an unknown raw channel name does), so config typos surface immediately. Populated
per-dataset via ``GENOTYPE_MAP`` in the samples config (reset to
:data:`DEFAULT_GENOTYPE_MAP` when absent)."""

DEFAULT_SEX_MAP = {
    "Male": ["Male", "male", "M", "m"],
    "Female": ["Female", "female", "F", "f"],
}
"""Module default for :data:`SEX_MAP` (the standard M/F spellings). ``apply_samples_config``
resets the live map to a copy of this when a dataset omits a ``SEX_MAP`` block."""

SEX_MAP = {k: list(v) for k, v in DEFAULT_SEX_MAP.items()}
"""Exact ``{canonical_label: [accepted spellings]}`` map normalizing the per-animal
``sex`` value (parallels :data:`GENOTYPE_MAP`; matched **exactly**). Authoritative when
populated — an uncovered value raises at load. Populated per-dataset via ``SEX_MAP`` in
the samples config (reset to :data:`DEFAULT_SEX_MAP` when absent)."""

# --- Channels: the single source of truth --------------------------------------------
CHANNEL_MAP = {
    "LMot": ["LMot"],
    "RMot": ["RMot"],
    "LBar": ["LBar"],
    "RBar": ["RBar"],
    "LHip": ["LHip"],
    "RHip": ["RHip"],
    "LAud": ["LAud"],
    "RAud": ["RAud"],
    "LVis": ["LVis"],
    "RVis": ["RVis"],
}
"""Single source of truth for channels: exact ``{abbrev: [raw names]}`` map, in canonical
order. A channel is the atomic unit (left/right is part of the identity, not a separate
axis); the code never interprets the label — it is matched **exactly** against raw data
channel names. This is a ``_MAP`` (exact key→values, like :data:`GENOTYPE_MAP` and
:data:`SEX_MAP`), not a fuzzy, substring-matched ``_ALIASES`` table. Populated per-dataset
via ``channels`` in the samples config.
:data:`CHANNEL_ABBREVS`,
:data:`CHANNEL_ABBREV_BY_RAW`, and the channel entry of :data:`DF_SORT_ORDER` are derived
from this; never edit those directly."""

# Derived from CHANNEL_MAP; recomputed by set_channel_map()/_recompute_channel_map_derived().
CHANNEL_ABBREVS = list(CHANNEL_MAP)
"""Ordered canonical channel abbreviations. **Derived** from :data:`CHANNEL_MAP` keys."""

CHANNEL_ABBREV_BY_RAW = {raw: abbrev for abbrev, raws in CHANNEL_MAP.items() for raw in raws}
"""Exact reverse lookup ``{raw channel name: canonical abbrev}``. **Derived** from
:data:`CHANNEL_MAP`; the sole resolution table for ``resolve_channel`` (exact
match only — no inference)."""

DF_SORT_ORDER = {
    "channel": ["average", "all", *CHANNEL_ABBREVS],
    "genotype": ["WT", "KO"],
    "sex": ["Male", "Female"],
    "isday": [True, False],
    "band": ["delta", "theta", "alpha", "beta", "gamma"],
}
"""Defines categorical sort orders for DataFrame columns. The ``channel`` entry is
**derived** from :data:`CHANNEL_ABBREVS`."""

FEATURE_LABELS = {
    # Linear features
    "rms": "RMS",
    "logrms": "Log(RMS)",
    "ampvar": "Amplitude Variance",
    "logampvar": "Log(Amplitude Variance)",
    "psdtotal": "Total PSD",
    "logpsdtotal": "Log(Total PSD)",
    "psdslope": "PSD Slope",
    "nspike": "Spike Count",
    "lognspike": "Log(Spike Count)",
    # Band features (base names - bands appended dynamically)
    "psdband": "Band Power",
    "logpsdband": "Log Band Power",
    "psdfrac": "Power Fraction",
    "logpsdfrac": "Log Power Fraction",
    # Matrix features (base names - bands appended dynamically)
    "cohere": "Coherence",
    "zcohere": "Z-Coherence",
    "imcoh": "Imaginary Coherence",
    "zimcoh": "Z-Imaginary Coherence",
    "pcorr": "Pearson Correlation",
    "zpcorr": "Z-Pearson Correlation",
    # Histogram features (frequency-bin spectrum per channel)
    "psd": r"PSD ($\mu V^2/Hz$)",
    # Derived/composite features
    "alphadelta": "Alpha/Delta Ratio",
}
"""Canonical display labels for features."""
