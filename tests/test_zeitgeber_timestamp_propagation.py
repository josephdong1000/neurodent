"""End-to-end proof that ``manual_datetimes`` propagate to WAR window timestamps
and onward to the zeitgeber (clock/ZT) axis.

Builds a synthetic recording with a known amplitude **step** at a known offset,
assigns a known start datetime, runs the real ``compute_windowed_analysis``, and
asserts the step lands at the expected window timestamp and Zeitgeber Time. A
second test proves a *gap* in per-file datetimes (e.g. a missing Selection)
propagates as a discontinuity in the WAR timestamps.

This is the regression guard for the timestamp chain
``manual_datetimes -> finalize_file_timestamps -> file_end_datetimes ->
TimestampMapper -> WAR result["timestamp"] -> add_zeitgeber_time_columns ->
shift_to_zeitgeber_reference``.
"""

import datetime

import numpy as np
import spikeinterface.core as si

from neurodent.core import LongRecordingOrganizer
from neurodent.core.zeitgeber import (
    add_zeitgeber_time_columns,
    shift_to_zeitgeber_reference,
)
from neurodent.visualization import AnimalOrganizer

FS = 1000
WINDOW_S = 60  # 1-minute windows keep the window count modest
CH_NAMES = ["L Aud", "R Aud"]  # parse to LAud/RAud via default aliases


def _stepped_recording(hours, step_h, low=1.0, high=6.0):
    """White noise whose amplitude steps from *low* to *high* at *step_h* hours."""
    n = int(hours * 3600 * FS)
    step_idx = int(step_h * 3600 * FS)
    env = np.empty(n, dtype=np.float32)
    env[:step_idx] = low
    env[step_idx:] = high
    data = np.random.randn(n, len(CH_NAMES)).astype(np.float32) * env[:, None]
    rec = si.NumpyRecording(data, sampling_frequency=float(FS))
    return rec.rename_channels(new_channel_ids=CH_NAMES)


def _war_from_recording(rec, manual_datetimes, file_durations):
    """Build a single-LRO AnimalOrganizer from a synthetic recording + manual times."""
    lro = LongRecordingOrganizer(item=None, recording=rec)
    # Durations are floats in production (get_duration()); int durations make
    # TimestampMapper's offset a numpy.int64 that timedelta rejects.
    file_durations = [float(d) for d in file_durations]
    lro.file_durations = file_durations
    lro.cumulative_file_durations = [float(x) for x in np.cumsum(file_durations)]
    lro.manual_datetimes = manual_datetimes
    lro.datetimes_are_start = True
    lro.finalize_file_timestamps()
    ao = AnimalOrganizer.from_lros([lro], animal_id="T", genotype="WT")
    war = ao.compute_windowed_analysis(
        ["rms"], window_s=WINDOW_S, multiprocess_mode="serial"
    )
    df = war.result.sort_values("timestamp").reset_index(drop=True)
    # rms is per-channel per window; reduce to one scalar per window.
    df["rms_s"] = df["rms"].apply(lambda a: float(np.mean(a)))
    return df


def test_step_propagates_to_war_timestamp_and_zt():
    """A +2h amplitude step, started at clock 08:00, lands at the right timestamp and ZT."""
    start = datetime.datetime(2020, 1, 1, 8, 0, 0)
    hours, step_h = 4, 2
    df = _war_from_recording(_stepped_recording(hours, step_h), start, [hours * 3600])

    # (1) First window's timestamp == the assigned start.
    assert df["timestamp"].iloc[0] == start

    # (2) The rms step occurs at the window whose timestamp == start + step_h.
    thresh = (df["rms_s"].min() + df["rms_s"].max()) / 2
    first_high = int(df.index[df["rms_s"] > thresh][0])
    step_ts = df["timestamp"].iloc[first_high]
    expected = start + datetime.timedelta(hours=step_h)
    assert abs((step_ts - expected).total_seconds()) <= WINDOW_S, (step_ts, expected)

    # (3) Zeitgeber mapping: shift_hours=6 => ZT0 = clock 06:00.
    #     Clock 08:00 -> ZT 2:00 (120 min); step at clock 10:00 -> ZT 4:00 (240 min).
    z = add_zeitgeber_time_columns(df.copy(), interval_minutes=60)
    z = shift_to_zeitgeber_reference(z, shift_hours=6)
    hourly = z.groupby("zt_minutes")["rms_s"].mean()
    # Pre-step hour (ZT 120 = clock 08:00) low; post-step hour (ZT 300 = clock
    # 11:00) high. Crossover sits at ZT 240 = clock 10:00 = start+2h.
    assert hourly.loc[120] < hourly.loc[300]
    assert hourly.idxmax() >= 240 and hourly.idxmin() <= 180


def test_per_file_gap_propagates_to_war_timestamps():
    """A 1h gap between two files' start times shows up as a 1h jump in WAR timestamps.

    Mirrors a missing Selection: file A covers [start, start+1h], file B is anchored
    1h later at start+2h (a 1h hole), via a per-file start list. The WAR window
    timestamps must reflect that hole, not treat the recording as contiguous.
    """
    start = datetime.datetime(2020, 1, 1, 8, 0, 0)
    rec_a = _stepped_recording(1, 1, low=1.0, high=1.0)  # flat 1h, low amplitude
    rec_b = _stepped_recording(1, 0, low=6.0, high=6.0)  # flat 1h, high amplitude
    rec = si.concatenate_recordings([rec_a, rec_b])

    start_b = start + datetime.timedelta(hours=2)  # 1h gap after A ends at +1h
    df = _war_from_recording(rec, [start, start_b], [3600, 3600])

    # The injected 1h hole shows up as one large jump between consecutive window
    # timestamps; all other gaps are ~one window. (The single window straddling
    # the file boundary maps to the prior file, so assert on the max jump rather
    # than the low/high-rms split.)
    diffs = df["timestamp"].diff().dropna().dt.total_seconds()
    assert diffs.max() >= 3600 - 2 * WINDOW_S, diffs.max()      # ~1h discontinuity
    assert diffs.median() <= 2 * WINDOW_S                       # rest are ~1 window
    # Total span ~3h (2h data + 1h hole), proving the gap was honored.
    span_h = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600
    assert span_h >= 2.9, span_h


def test_multiday_recording_folds_across_days():
    """Three recordings on consecutive calendar dates fold onto the same ZT axis.

    Proves the chain handles MULTIPLE DAYS: each day is anchored at clock 08:00
    with a step at +1.5h; the zeitgeber fold must overlay all three days into the
    same ZT bins, preserving the within-day day/night pattern.
    """
    base = datetime.datetime(2020, 1, 1, 8, 0, 0)
    lros = []
    for d in range(3):
        rec = _stepped_recording(3, 1.5, low=1.0, high=6.0)  # 3h, step at +1.5h
        lro = LongRecordingOrganizer(item=None, recording=rec)
        lro.file_durations = [float(3 * 3600)]
        lro.cumulative_file_durations = [float(3 * 3600)]
        lro.manual_datetimes = base + datetime.timedelta(days=d)
        lro.datetimes_are_start = True
        lro.finalize_file_timestamps()
        lros.append(lro)

    ao = AnimalOrganizer.from_lros(lros, animal_id="T", genotype="WT")
    war = ao.compute_windowed_analysis(
        ["rms"], window_s=WINDOW_S, multiprocess_mode="serial"
    )
    df = war.result.sort_values("timestamp").reset_index(drop=True)
    df["rms_s"] = df["rms"].apply(lambda a: float(np.mean(a)))

    # Spans three distinct calendar dates.
    assert df["timestamp"].dt.date.nunique() == 3

    z = add_zeitgeber_time_columns(df.copy(), interval_minutes=60)
    z = shift_to_zeitgeber_reference(z, shift_hours=6)

    # At least one ZT bin aggregates all three days (the fold overlays them).
    days_per_bin = z.groupby("zt_minutes")["timestamp"].apply(
        lambda s: s.dt.date.nunique()
    )
    assert days_per_bin.max() == 3

    # Within the covered ZT window (clock 08:00-11:00 = ZT 02:00-05:00), the
    # step at clock 09:30 (ZT 03:30) separates low (ZT2) from high (ZT5).
    hourly = z.groupby("zt_minutes")["rms_s"].mean()
    assert hourly.loc[120] < hourly.loc[300]
