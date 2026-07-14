"""Unit conversion and MNE unit extraction."""

import logging

import numpy as np
try:
    import mne
    from mne.io.constants import FIFF
except Exception:  # pragma: no cover - optional at import time
    mne = None
    FIFF = None


# Bounds on the median absolute amplitude of a recording claiming to be in µV. Real data (animal 443)
# has per-channel medians of 17..369 µV. Nothing physiological lands outside the HARD bounds, so they
# cannot false-alarm while still catching a 1e6 volts-as-µV slip. The SOFT bounds would catch a 1e3
# slip (mV/nV) but a loud animal can exceed them, so crossing those only warns.
UV_HARD_MIN, UV_HARD_MAX = 1e-2, 1e5
UV_SOFT_MIN, UV_SOFT_MAX = 1e0, 1e3


def assert_microvolts(data, context: str = "recording") -> float | None:
    """Raise if ``data`` cannot plausibly be in microvolts.

    Never rescales: guessing at units is how the error gets baked in. The statistic is the median
    absolute amplitude over finite, non-zero samples; zeros are excluded so a dead channel does not
    drag it to 0 and trip the floor.

    Args:
        data: voltages claimed to be in µV.
        context (str): what is being checked, for the error message.

    Returns:
        float | None: the median absolute amplitude in µV, or None if the data was empty or all-zero.

    Raises:
        ValueError: if the median lies outside the hard bounds.
    """
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    nonzero = np.abs(finite[finite != 0])
    if nonzero.size == 0:
        return None

    med = float(np.median(nonzero))

    if not (UV_HARD_MIN <= med <= UV_HARD_MAX):
        likely = "volts mistaken for µV (a factor of 1e6)" if med < UV_HARD_MIN else (
            "nanovolts mistaken for µV (a factor of 1e3)"
        )
        raise ValueError(
            f"{context}: data claims to be in µV but its median amplitude is {med:.3g} µV, outside "
            f"the physiologically possible range [{UV_HARD_MIN:g}, {UV_HARD_MAX:g}] µV. "
            f"Most likely cause: {likely}. Fix the unit handling at the source; this will not "
            f"rescale for you."
        )

    if not (UV_SOFT_MIN <= med <= UV_SOFT_MAX):
        logging.warning(
            "%s: median amplitude is %.3g µV, outside the usual [%g, %g] µV. Possible for an unusually "
            "quiet or loud recording, but also what a 1e3 unit slip (mV/nV) looks like.",
            context, med, UV_SOFT_MIN, UV_SOFT_MAX,
        )
    return med


def convert_units_to_multiplier(current_units: str, target_units: str = "µV") -> float:
    """
    Convert between different voltage units and return the multiplication factor.

    This function calculates the conversion factor needed to transform values
    from one voltage unit to another (e.g., from mV to µV).

    Args:
        current_units (str): The current unit of the values. Must be one of: 'µV', 'mV', 'V', 'nV'.
        target_units (str, optional): The target unit to convert to. Defaults to 'µV'.
            Must be one of: 'µV', 'mV', 'V', 'nV'.

    Returns:
        float: The multiplication factor to convert from current_units to target_units.
            To convert values, multiply your data by this factor.

    Raises:
        AssertionError: If current_units or target_units are not supported.

    Examples:
        >>> convert_units_to_multiplier("mV", "µV")
        1000.0
        >>> convert_units_to_multiplier("V", "mV")
        1000.0
        >>> convert_units_to_multiplier("µV", "V")
        1e-06
    """
    units_to_mult = {"µV": 1e-6, "mV": 1e-3, "V": 1, "nV": 1e-9}

    assert current_units in units_to_mult.keys(), f"No valid current unit called '{current_units}' found"
    assert target_units in units_to_mult.keys(), f"No valid target unit called '{target_units}' found"

    return units_to_mult[current_units] / units_to_mult[target_units]


def extract_mne_unit_info(raw_info: dict) -> tuple[str | None, float | None]:
    """Extract unit information from MNE Raw info object.

    Args:
        raw_info (dict): MNE Raw.info object containing channel information

    Returns:
        tuple[str | None, float | None]: (unit_name, mult_to_uV) where unit_name
                                        is the consistent unit across all channels
                                        and mult_to_uV is the conversion factor to µV

    Raises:
        ValueError: If channel units are inconsistent across channels
    """
    if mne is None or FIFF is None:
        logging.warning("MNE not available, cannot extract unit information")
        return None, None

    if "chs" not in raw_info or not raw_info["chs"]:
        logging.warning("No channel information found in MNE Raw.info, using default units")
        return None, None

    # Extract unit information from all channels
    channel_units = []
    unit_muls = []

    for ch_info in raw_info["chs"]:
        ch_name = ch_info.get("ch_name", "unknown")
        unit = ch_info.get("unit", None)
        unit_mul = ch_info.get("unit_mul", None)

        if unit is not None and unit_mul is not None:
            channel_units.append((ch_name, unit, unit_mul))
            unit_muls.append(unit_mul)

    if not channel_units:
        logging.warning("No unit information found in any channels, using default units")
        return None, None

    # Check for consistency in unit values
    unique_units = set(unit for _, unit, _ in channel_units)
    unique_unit_muls = set(unit_muls)

    if len(unique_units) > 1:
        unit_details = [(ch, unit, mul) for ch, unit, mul in channel_units]
        raise ValueError(
            f"Inconsistent units across channels. Found different unit values: {unique_units}. "
            f"Channel details: {unit_details}"
        )

    if len(unique_unit_muls) > 1:
        unit_details = [(ch, unit, mul) for ch, unit, mul in channel_units]
        raise ValueError(
            f"Inconsistent unit multipliers across channels. Found different unit_mul values: {unique_unit_muls}. "
            f"Channel details: {unit_details}"
        )

    # Get the consistent unit values
    unit_code = list(unique_units)[0]
    unit_mul = list(unique_unit_muls)[0]

    # Convert MNE unit codes to string representation using FIFF constants
    # Based on MNE FIFF constants documentation
    unit_str = None
    if hasattr(FIFF, "FIFF_UNIT_V") and unit_code == FIFF.FIFF_UNIT_V:
        unit_str = "V"
    elif hasattr(FIFF, "FIFF_UNIT_T") and unit_code == FIFF.FIFF_UNIT_T:
        unit_str = "T"  # Tesla - MEG magnetometer
    elif hasattr(FIFF, "FIFF_UNIT_T_M") and unit_code == FIFF.FIFF_UNIT_T_M:
        unit_str = "T/m"  # Tesla/meter - MEG gradiometer
    else:
        logging.warning(f"Unknown MNE unit code {unit_code}, using default units")
        return None, None

    # Convert unit multipliers using FIFF constants
    multiplier = None
    if hasattr(FIFF, "FIFF_UNITM_NONE") and unit_mul == FIFF.FIFF_UNITM_NONE:
        multiplier = 1.0
    elif hasattr(FIFF, "FIFF_UNITM_MU") and unit_mul == FIFF.FIFF_UNITM_MU:
        multiplier = 1e-6  # micro
    elif hasattr(FIFF, "FIFF_UNITM_M") and unit_mul == FIFF.FIFF_UNITM_M:
        multiplier = 1e-3  # milli
    elif hasattr(FIFF, "FIFF_UNITM_N") and unit_mul == FIFF.FIFF_UNITM_N:
        multiplier = 1e-9  # nano
    elif hasattr(FIFF, "FIFF_UNITM_P") and unit_mul == FIFF.FIFF_UNITM_P:
        multiplier = 1e-12  # pico
    elif hasattr(FIFF, "FIFF_UNITM_F") and unit_mul == FIFF.FIFF_UNITM_F:
        multiplier = 1e-15  # femto
    else:
        # Fallback to numerical interpretation if FIFF constants not available
        mul_mapping = {
            0: 1.0,  # FIFF_UNITM_NONE
            -3: 1e-3,  # FIFF_UNITM_M (milli)
            -6: 1e-6,  # FIFF_UNITM_MU (micro)
            -9: 1e-9,  # FIFF_UNITM_N (nano)
            -12: 1e-12,  # FIFF_UNITM_P (pico)
            -15: 1e-15,  # FIFF_UNITM_F (femto)
        }
        multiplier = mul_mapping.get(unit_mul)
        if multiplier is None:
            logging.warning(f"Unknown MNE unit multiplier {unit_mul}, using default units")
            return None, None

    # For EEG data (voltage units), compute the final unit and conversion factor
    if unit_str == "V":
        # Apply the MNE multiplier to get the actual unit
        if multiplier == 1e-6:
            final_unit = "µV"
        elif multiplier == 1e-3:
            final_unit = "mV"
        elif multiplier == 1.0:
            final_unit = "V"
        elif multiplier == 1e-9:
            final_unit = "nV"
        else:
            logging.warning(f"Unusual voltage unit multiplier {multiplier}, treating as V")
            final_unit = "V"

        # Convert to µV multiplier using existing utility
        try:
            mult_to_uV = convert_units_to_multiplier(final_unit, "µV")
            logging.info(f"Extracted MNE units: {final_unit} -> mult_to_uV = {mult_to_uV}")
            return final_unit, mult_to_uV
        except (ValueError, AssertionError) as e:
            logging.warning(f"Failed to convert units {final_unit}: {e}")
            return None, None
    else:
        # Non-voltage units (MEG, etc.) - don't convert to µV
        logging.info(f"Non-voltage units detected: {unit_str}, not converting to µV")
        return None, None


def log_transform(rec: np.ndarray, **kwargs) -> np.ndarray:
    """Log transform the signal

    Args:
        rec (np.ndarray): The signal to log transform.

    Returns:
        np.ndarray: ln(rec + 1)
    """
    if rec is not None:
        return np.log(rec + 1)
    else:
        return None
