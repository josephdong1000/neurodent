"""Unit conversion and MNE unit extraction."""

import logging

import numpy as np
try:
    import mne
    from mne.io.constants import FIFF
except Exception:  # pragma: no cover - optional at import time
    mne = None
    FIFF = None


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
