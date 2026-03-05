"""
Tests for MNE unit extraction (neurodent.core.utils.extract_mne_unit_info).

Covers: all voltage multipliers (µV/mV/V/nV), Tesla non-voltage path,
inconsistent units/multipliers, and missing channel info.
"""

import pytest

from neurodent.core.utils import extract_mne_unit_info


class TestExtractMneUnitInfo:
    """Tests for MNE unit extraction covering all branches."""

    def _make_raw_info(self, unit, unit_mul, n_channels=2):
        """Helper to create a minimal raw_info dict mimicking MNE info."""
        return {
            "chs": [
                {"ch_name": f"ch{i}", "unit": unit, "unit_mul": unit_mul}
                for i in range(n_channels)
            ]
        }

    def test_no_channels(self):
        result = extract_mne_unit_info({"chs": []})
        assert result == (None, None)

    def test_missing_chs_key(self):
        result = extract_mne_unit_info({})
        assert result == (None, None)

    def test_channels_without_unit(self):
        info = {"chs": [{"ch_name": "ch0"}]}
        result = extract_mne_unit_info(info)
        assert result == (None, None)

    def test_inconsistent_units_raises(self):
        info = {
            "chs": [
                {"ch_name": "ch0", "unit": 107, "unit_mul": 0},
                {"ch_name": "ch1", "unit": 112, "unit_mul": 0},
            ]
        }
        with pytest.raises(ValueError, match="Inconsistent units"):
            extract_mne_unit_info(info)

    def test_inconsistent_unit_muls_raises(self):
        info = {
            "chs": [
                {"ch_name": "ch0", "unit": 107, "unit_mul": 0},
                {"ch_name": "ch1", "unit": 107, "unit_mul": -6},
            ]
        }
        with pytest.raises(ValueError, match="Inconsistent unit multipliers"):
            extract_mne_unit_info(info)

    def test_unknown_unit_code(self):
        info = self._make_raw_info(unit=999, unit_mul=0)
        result = extract_mne_unit_info(info)
        assert result == (None, None)

    def test_voltage_micro(self):
        """FIFF_UNIT_V (107) + FIFF_UNITM_MU (-6) → µV, mult=1.0"""
        try:
            from mne.io.constants import FIFF
            info = self._make_raw_info(unit=FIFF.FIFF_UNIT_V, unit_mul=FIFF.FIFF_UNITM_MU)
            unit_name, mult = extract_mne_unit_info(info)
            assert unit_name == "µV"
            assert mult == pytest.approx(1.0)
        except ImportError:
            pytest.skip("MNE not available")

    def test_voltage_milli(self):
        try:
            from mne.io.constants import FIFF
            info = self._make_raw_info(unit=FIFF.FIFF_UNIT_V, unit_mul=FIFF.FIFF_UNITM_M)
            unit_name, mult = extract_mne_unit_info(info)
            assert unit_name == "mV"
            assert mult == pytest.approx(1e3)
        except ImportError:
            pytest.skip("MNE not available")

    def test_voltage_none_multiplier(self):
        try:
            from mne.io.constants import FIFF
            info = self._make_raw_info(unit=FIFF.FIFF_UNIT_V, unit_mul=FIFF.FIFF_UNITM_NONE)
            unit_name, mult = extract_mne_unit_info(info)
            assert unit_name == "V"
            # V → µV conversion: 1 V = 1e6 µV
            assert mult == pytest.approx(1e6)
        except ImportError:
            pytest.skip("MNE not available")

    def test_voltage_nano(self):
        try:
            from mne.io.constants import FIFF
            info = self._make_raw_info(unit=FIFF.FIFF_UNIT_V, unit_mul=FIFF.FIFF_UNITM_N)
            unit_name, mult = extract_mne_unit_info(info)
            assert unit_name == "nV"
            assert mult == pytest.approx(1e-3)
        except ImportError:
            pytest.skip("MNE not available")

    def test_tesla_units_non_voltage(self):
        """Non-voltage units (Tesla) should return (None, None)."""
        try:
            from mne.io.constants import FIFF
            info = self._make_raw_info(unit=FIFF.FIFF_UNIT_T, unit_mul=FIFF.FIFF_UNITM_NONE)
            result = extract_mne_unit_info(info)
            assert result == (None, None)
        except ImportError:
            pytest.skip("MNE not available")
