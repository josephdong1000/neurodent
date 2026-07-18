"""Channel/animal/label name resolution and slugification."""

import re
import unicodedata
import warnings

from neurodent import constants


def resolve_channel(channel_name: str) -> str:
    """
    Resolve a raw channel name to its canonical channel abbreviation by **exact lookup**.

    Resolution is explicit and never inferred: (1) the (stripped) name is already a
    canonical abbreviation (:data:`neurodent.constants.CHANNEL_ABBREVS`); (2) it is an
    exact key in :data:`neurodent.constants.CHANNEL_ABBREV_BY_RAW` (the per-dataset
    ``raw name -> abbrev`` map). Anything else **raises loudly** — there is no fuzzy,
    substring, or number-based guessing.

    Args:
        channel_name (str): Raw channel name from the data.

    Returns:
        str: Canonical channel abbreviation.

    Raises:
        ValueError: When the name is not in the configured channel map. Configure the exact
            raw name under its abbreviation (``channels`` in the samples config, or
            :func:`neurodent.set_channel_map`).

    Examples:
        >>> resolve_channel("LMot")          # already canonical
        'LMot'
        >>> resolve_channel("L Motor Ctx")   # configured raw name -> abbrev
        'LMot'
    """
    raw = channel_name.strip()
    if raw in constants.CHANNEL_ABBREVS:
        return raw
    if raw in constants.CHANNEL_ABBREV_BY_RAW:
        return constants.CHANNEL_ABBREV_BY_RAW[raw]
    raise ValueError(
        f"Channel {raw!r} is not in the configured channel map. "
        f"Canonical labels: {constants.CHANNEL_ABBREVS}; configured raw names: "
        f"{sorted(constants.CHANNEL_ABBREV_BY_RAW)}. "
        f"Add the exact raw name under its abbreviation in the samples config "
        f"(channels) or via neurodent.set_channel_map()."
    )


def resolve_channels(names: list[str], ids: list[str] | None = None) -> list[str]:
    """Abbreviate a list of channels via exact lookup.

    A channel carries two identifiers the pipeline must not conflate: its stable **ID**
    (``get_channel_ids()`` — what a dataset's ``channels`` map is keyed on, e.g. an Intan
    hardware port ``D-009``) and its **display name** (the ``channel_name`` property — an
    experimenter's label, possibly non-unique). When *ids* is given, resolution is
    **ID-first**: each channel maps by its ID, deferring to its display name only when the ID
    is not in the configured map. That one rule lets datasets that key on IDs and those that
    key on labels both resolve, with no per-dataset flag. When *ids* is omitted (readers with
    no separate ID space), it is a plain per-name lookup — the historical behaviour.

    Unmappable channels are **warned about loudly** (and kept as their ID/name so callers
    comparing channel sets still get a value) rather than silently swallowed.

    Args:
        names: Display channel name strings.
        ids: Stable channel IDs, same length/order as *names*. Defaults to *names*.

    Returns:
        List of canonical abbreviations (same length as input).
    """
    if ids is None:
        ids = names
    result = []
    for cid, cname in zip(ids, names):
        try:
            result.append(resolve_channel(cid))
            continue
        except (ValueError, KeyError, AttributeError) as e:
            err = e
        if cname != cid:
            try:
                result.append(resolve_channel(cname))
                continue
            except (ValueError, KeyError, AttributeError) as e:
                err = e
        warnings.warn(
            f"Channel {cid!r} could not be mapped to a canonical abbreviation: {err}",
            UserWarning,
            stacklevel=2,
        )
        result.append(cid)
    return result


def parse_str_to_animal(string: str, animal_param: tuple[int, str] | str | list[str] = (0, None)) -> str:
    """
    DEPRECATED: Use FileDiscoverer with {animal} placeholder in pattern instead.

    Parses the filename of a binfolder to get the animal id.

    Args:
        string (str): String to parse.
        animal_param: Parameter specifying how to parse the animal ID:
            tuple[int, str]: (index, separator) for simple split and index. Not recommended for inconsistent naming conventions.
            str: regex pattern to extract ID. Most general use case. If multiple matches are found, returns the first match.
            list[str]: list of possible animal IDs to match against. Returns first match in list order, case-sensitive, ignoring empty strings.

    Returns:
        str: Animal id.

    Examples:
        # Tuple format: (index, separator)
        >>> parse_str_to_animal("WT_A10_2023-01-01_data.bin", (1, "_"))
        'A10'
        >>> parse_str_to_animal("A10_WT_recording.bin", (0, "_"))
        'A10'

        # Regex pattern format
        >>> parse_str_to_animal("WT_A10_2023-01-01_data.bin", r"A\\\\d+")
        'A10'
        >>> parse_str_to_animal("subject_123_data.bin", r"\\\\d+")
        '123'

        # List format: possible IDs to match
        >>> parse_str_to_animal("WT_A10_2023-01-01_data.bin", ["A10", "A11", "A12"])
        'A10'
        >>> parse_str_to_animal("WT_A10_data.bin", ["B15", "C20"])  # No match
        ValueError: No matching ID found in WT_A10_data.bin from possible IDs: ['B15', 'C20']
    """
    warnings.warn(
        "parse_str_to_animal is deprecated. Use FileDiscoverer with {animal} placeholder in pattern instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if isinstance(animal_param, tuple):
        index, sep = animal_param
        animid = string.split(sep)
        return animid[index]
    elif isinstance(animal_param, str):
        pattern = animal_param
        match = re.search(pattern, string)
        if match:
            return match.group()
        raise ValueError(f"No match found for pattern {pattern} in string {string}")
    elif isinstance(animal_param, list):
        possible_ids = animal_param
        for id in possible_ids:
            # Skip empty or whitespace-only strings
            if id and id.strip() and id in string:
                return id
        raise ValueError(f"No matching ID found in {string} from possible IDs: {possible_ids}")
    else:
        raise ValueError(f"Invalid animal_param type: {type(animal_param)}")


def normalize_value_from_aliases(
    value: str,
    alias_dict: dict[str, list[str]],
) -> str | None:
    """Normalize a value to its canonical form using a value map.

    Performs **exact** matching: the value must equal one of the accepted spellings
    listed for a canonical label. Used for normalizing standalone configuration values
    against an exact ``_MAP`` (e.g. :data:`~neurodent.constants.SEX_MAP`,
    :data:`~neurodent.constants.GENOTYPE_MAP`).

    Args:
        value: The raw value to normalize (e.g., ``"M"``, ``"female"``).
        alias_dict: Dictionary of ``{canonical_key: [accepted spellings]}``.

    Returns:
        The canonical key if *value* matches any spelling, or ``None`` if no match.
    """
    for canonical_key, aliases in alias_dict.items():
        if value in aliases:
            return canonical_key
    return None


def rename_mne_channels(mne_obj):
    """Rename an MNE object's channels in place to canonical abbreviations.

    Applies :func:`resolve_channel` (exact lookup) to every entry of
    ``mne_obj.info['ch_names']``. Format-agnostic — works on any MNE object
    whose raw channel names are declared in :data:`~neurodent.constants.CHANNEL_MAP`.

    Args:
        mne_obj: An MNE object exposing ``info['ch_names']`` (e.g. a ``RawArray``).

    Returns:
        The same ``mne_obj``, with channel names replaced by their canonical abbreviations.
    """
    for i in range(len(mne_obj.info['ch_names'])):
        mne_obj.info['ch_names'][i] = resolve_channel(mne_obj.info['ch_names'][i])
    return mne_obj


def slugify(value, allow_unicode=False):
    """Convert a string to a URL-friendly slug.

    Converts to ASCII (unless *allow_unicode* is ``True``), lowercases,
    removes non-alphanumeric characters (except hyphens and underscores),
    and converts spaces and repeated dashes to single dashes.

    Drop-in replacement for ``django.utils.text.slugify`` using only the
    standard library.

    **Path-safety convention.**
        This is the canonical helper for converting display-friendly identifiers
        (animal IDs, animaldays, genotype strings) into filesystem-safe path
        components.  Any code that constructs a ``Path`` or filename from one of
        these strings **must** route the value through ``slugify(...)`` directly,
        or through one of the ``path_safe_*`` accessors on
        :class:`~neurodent.results.WindowAnalysisResult`,
        :class:`~neurodent.results.streaming.LazyWindowAnalysisResult`, or
        :class:`~neurodent.results.FrequencyDomainSpikeAnalysisResult`.

        Display strings — which may contain ``/``, ``;``, parens, spaces, etc.
        (e.g. the real arxrosa genotype ``Arx(F/y); Rosa(+/wt)``) — are correct
        domain notation and stay unchanged on the public attributes
        (``animal_id``, ``animaldays``, ``genotype``).  They're the source of
        truth for what humans see in logs and plot labels.  Only the
        ``path_safe_*`` accessors return the slugified form.

    Args:
        value: The string to slugify.
        allow_unicode: If ``True``, keep Unicode characters instead of
            transliterating to ASCII.

    Returns:
        str: A URL-safe slug string.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def get_feature_label(feature_name: str) -> str:
    """
    Convert a feature column name to a human-readable label.
    
    Handles:
    - Base features: "rms" -> "RMS"
    - Banded features: "logpsdband_delta" -> "Log Band Power - Delta"
    - Baseline-subtracted: "logrms_nobase" -> "Log(RMS) - Baseline"
    
    Args:
        feature_name: Column name (e.g., "logpsdband_delta_nobase")
    
    Returns:
        Human-readable label. Falls back to the original name if not found.
    
    Examples:
        >>> get_feature_label("logpsdband_delta")
        'Log Band Power (Delta)'
        >>> get_feature_label("alphadelta")
        'Alpha/Delta Ratio'
        >>> get_feature_label("logrms_nobase")
        'Log(RMS) - Baseline'
    """
    # Check for _nobase suffix
    is_baseline_subtracted = feature_name.endswith("_nobase")
    if is_baseline_subtracted:
        feature_name = feature_name[:-7]  # Remove "_nobase"
    
    # Check for band suffix (only for banded/matrix features)
    band_name = None
    base_feature = feature_name
    
    # Only check for band suffix if the base is a known banded feature
    for band in constants.BAND_NAMES:
        if feature_name.endswith(f"_{band}"):
            potential_base = feature_name[: -(len(band) + 1)]
            if potential_base in constants.BAND_FEATURES or potential_base in constants.MATRIX_FEATURES:
                base_feature = potential_base
                band_name = band
                break
    
    # Look up base label
    base_label = constants.FEATURE_LABELS.get(base_feature, feature_name)
    
    # Build final label
    label = base_label
    if band_name:
        label = f"{label} ({band_name.capitalize()})"
    if is_baseline_subtracted:
        label = f"{label} - Baseline"
    
    return label
