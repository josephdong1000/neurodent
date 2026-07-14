"""End-to-end units check: known amplitude in, microvolts out, for every loading path.

`get_fragment_np` promises uV and everything downstream believes it, so a 1e6 slip corrupts filters,
features and plots at once while still producing plausible output.

The units are established differently per format (SpikeInterface gain, EDF physical dimension, or the
binary intermediate's gain), so each path is checked empirically rather than by inspection: push a
known amplitude through the real loader and assert what comes out.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")
si = pytest.importorskip("spikeinterface")

from neurodent.analysis.long_recording_analyzer import LongRecordingAnalyzer
from neurodent.core.utils import assert_microvolts
from neurodent.core.utils.units import UV_HARD_MAX, UV_HARD_MIN
from neurodent.loading.long_recording_organizer import LongRecordingOrganizer

FS = 1000
SECS = 30
N_CH = 4
AMP_UV = 100.0                     # the known truth
FREQ = 10.0


@pytest.fixture(scope="module")
def source_fif(tmp_path_factory):
    """A 100 µV sine, written as a .fif. MNE holds EEG in VOLTS, so on disk this is 100e-6."""
    d = tmp_path_factory.mktemp("units_src")
    t = np.arange(SECS * FS) / FS
    data_v = np.stack([AMP_UV * 1e-6 * np.sin(2 * np.pi * FREQ * t)] * N_CH)
    info = mne.create_info([f"ch{i}" for i in range(N_CH)], FS, ch_types="eeg")
    raw = mne.io.RawArray(data_v, info, verbose=False)
    path = d / "src_raw.fif"
    raw.save(path, overwrite=True, verbose=False)
    return path, float(np.median(np.abs(data_v)) * 1e6)   # expected median, in µV


def _load(source_fif_path, intermediate, tmp_path):
    lro = LongRecordingOrganizer(
        source_fif_path,
        mode="mne",
        extract_func=mne.io.read_raw_fif,
        intermediate=intermediate,
        intermediate_dir=str(tmp_path),
        manual_datetimes=datetime(2024, 1, 1),
        cache_policy="force_regenerate",
    )
    return lro, LongRecordingAnalyzer(lro, fragment_len_s=5, apply_notch_filter=False)


@pytest.mark.parametrize("intermediate", ["edf", "bin"])
def test_mne_path_returns_microvolts(source_fif, intermediate, tmp_path):
    """100 uV in -> ~100 uV out, whichever intermediate is used.

    `bin` writes MNE's native volts to disk, so its gain must convert them; a gain of 1 hands back
    volts while promising uV.
    """
    path, expected_uv = source_fif
    lro, lan = _load(path, intermediate, tmp_path)

    assert lro.meta.mult_to_uV == pytest.approx(1e6)       # volts -> uV
    frag = lan.get_fragment_np(1)
    med = float(np.median(np.abs(frag)))

    assert med == pytest.approx(expected_uv, rel=0.05), (
        f"{intermediate}: expected ~{expected_uv:.1f} µV, got {med:.3g} "
        f"(off by {expected_uv / med:.3g}x -- a unit slip, not noise)"
    )
    assert UV_HARD_MIN < med < UV_HARD_MAX


def test_mne_intermediates_agree(source_fif, tmp_path):
    """edf and bin must not disagree about what the data is."""
    path, _ = source_fif
    _, lan_edf = _load(path, "edf", tmp_path / "e")
    _, lan_bin = _load(path, "bin", tmp_path / "b")

    med_edf = float(np.median(np.abs(lan_edf.get_fragment_np(1))))
    med_bin = float(np.median(np.abs(lan_bin.get_fragment_np(1))))
    assert med_edf == pytest.approx(med_bin, rel=0.05), (
        f"the two MNE intermediates disagree: edf={med_edf:.4g} µV vs bin={med_bin:.4g} µV"
    )


def test_integer_adc_recording_is_scaled_to_microvolts(tmp_path):
    """Raw integer ADC counts + gain_to_uV -> uV via spre.scale_to_uV.

    A real integer recording, not a mock, so the scaling actually runs (test_intan_units_fix.py mocks
    get_traces to hand back uV, and so never exercises it).
    """
    from spikeinterface.core import NumpyRecording

    gain = 0.195                                   # µV per ADC count (a real Intan value)
    counts = AMP_UV / gain                         # ADC counts encoding a 100 µV sine
    t = np.arange(SECS * FS) / FS
    adc = np.round(counts * np.sin(2 * np.pi * FREQ * t)).astype(np.int16)
    traces = np.stack([adc] * N_CH, axis=1)        # (n_samples, n_channels), integer counts

    rec = NumpyRecording([traces], sampling_frequency=FS)
    rec.set_channel_gains(gain)
    rec.set_channel_offsets(0.0)
    assert rec.get_dtype().kind == "i"             # integer: the scale_to_uV branch must fire

    lro = LongRecordingOrganizer(
        tmp_path, mode=None, manual_datetimes=datetime(2024, 1, 1), datetimes_are_start=True
    )
    lro._init_from_recording(rec)

    out = lro.LongRecording.get_traces(return_scaled=True)
    expected = float(np.median(np.abs(adc.astype(float) * gain)))
    assert float(np.median(np.abs(out))) == pytest.approx(expected, rel=0.05)


class TestAssertMicrovolts:
    """Bounds are calibrated on real data (per-channel medians 17..369 uV)."""

    def test_accepts_real_amplitudes(self):
        for med in (17.0, 369.0):                  # the quietest and loudest real channels
            assert assert_microvolts(np.full(100, med)) == pytest.approx(med)

    def test_rejects_volts_mistaken_for_microvolts(self):
        """A 50 uV signal handed over as 5e-05."""
        with pytest.raises(ValueError, match="volts mistaken"):
            assert_microvolts(np.full(100, 5e-5))

    def test_rejects_nanovolts_mistaken_for_microvolts(self):
        with pytest.raises(ValueError, match="nanovolts mistaken"):
            assert_microvolts(np.full(100, 5e5))

    def test_dead_channel_does_not_trip_the_floor(self):
        """A disconnected electrode is all zeros: a `bad` category, not a unit error. Zeros must be
        excluded, or any recording containing one would be rejected as mis-scaled."""
        data = np.concatenate([np.zeros(500), np.full(500, 50.0)])
        assert assert_microvolts(data) == pytest.approx(50.0)

    def test_all_zero_and_empty_are_skipped_not_failed(self):
        assert assert_microvolts(np.zeros(100)) is None
        assert assert_microvolts(np.array([])) is None

    def test_nans_are_ignored(self):
        data = np.concatenate([np.full(100, np.nan), np.full(100, 50.0)])
        assert assert_microvolts(data) == pytest.approx(50.0)

    def test_loud_recording_warns_but_does_not_raise(self, caplog):
        """A burst-heavy channel can exceed the soft bound, which must not block loading."""
        assert assert_microvolts(np.full(100, 5e3)) == pytest.approx(5e3)   # no raise
