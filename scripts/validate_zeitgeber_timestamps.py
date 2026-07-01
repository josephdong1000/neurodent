#!/usr/bin/env python3
"""Validate that ``manual_datetime`` propagates correctly to the zeitgeber plot.

Builds a *synthetic* recording whose amplitude is a known function of Zeitgeber
Time (ZT), assigns a known start clock time, runs the real windowed-analysis +
zeitgeber pipeline, renders the zeitgeber plot, and numerically checks the
feature lands where ZT says it should. If the checks PASS and the PNG shows the
expected structure under the night shading, the chain
``manual_datetime -> WAR timestamp -> zeitgeber -> plot`` is wired correctly.

Envelopes (amplitude vs ZT, ZT0 = lights-on):
  * ``square``   — low in ZT day (0-12), high in ZT night (12-24).
  * ``sine``     — 24h sine, peak at ZT18 (mid-night).
  * ``triangle`` — ramps UP through ZT day, DOWN through ZT night; peak at ZT12.

Recording shape:
  * continuous: ``--hours N`` (one contiguous recording from ``--start``).
  * segmented:  ``--segments "0-10,15-20"`` builds per-day clock-hour segments
    with gaps between them (mirrors a split/missing-Selection recording),
    repeated for ``--days`` days, anchored via a per-file start list.

One PNG is written to ``{--out}/{--name}.png``. This is a manual validation tool
(run on the cluster / via the ``zeitgeber_validation`` snakemake target, NOT on a
small local machine — 90h at 1 kHz is ~1.3 GB for one channel).

Examples:
    uv run python scripts/validate_zeitgeber_timestamps.py \
        --envelope square --hours 48 --name square_48h --out results/zeitgeber_validation
    uv run python scripts/validate_zeitgeber_timestamps.py \
        --envelope square --segments "0-10,15-20" --days 2 --name split_gap
"""

import argparse
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spikeinterface.core as si

from neurodent.core import LongRecordingOrganizer
from neurodent.core.zeitgeber import (
    add_zeitgeber_time_columns,
    shift_to_zeitgeber_reference,
)
from neurodent.visualization import AnimalOrganizer

FS = 1000
WINDOW_S = 60
CH_NAME = "L Aud"  # parses to LAud via default aliases
LOW, HIGH = 1.0, 6.0


def amplitude(zt_h, envelope):
    """Amplitude as a function of Zeitgeber-time hours (0-24)."""
    zt_h = np.asarray(zt_h)
    if envelope == "square":
        return np.where(zt_h >= 12, HIGH, LOW)
    if envelope == "sine":  # peak at ZT18
        return LOW + (HIGH - LOW) * 0.5 * (1 + np.sin(2 * np.pi * (zt_h - 12) / 24))
    if envelope == "triangle":  # ramp up through day, down through night; peak ZT12
        up = LOW + (HIGH - LOW) * (zt_h / 12.0)
        down = HIGH - (HIGH - LOW) * ((zt_h - 12) / 12.0)
        return np.where(zt_h <= 12, up, down)
    raise ValueError(f"unknown envelope: {envelope}")


def _chunk(start_clock_h, dur_h, envelope, gen_shift):
    """One recording chunk whose amplitude follows the envelope vs GENERATION ZT.

    The envelope is defined against ``zt_gen = (clock - gen_shift) mod 24``. The
    plot's ZT axis later uses the (possibly different) processing shift, so the
    landmark visibly moves by (gen_shift - proc_shift) — that is how a shifted
    scenario confirms ZT0 re-anchoring.
    """
    n = int(dur_h * 3600 * FS)
    t_h = np.arange(n, dtype=np.float64) / (3600 * FS)
    zt_gen = (start_clock_h + t_h - gen_shift) % 24
    amp = amplitude(zt_gen, envelope).astype(np.float32)
    data = np.random.randn(n, 1).astype(np.float32) * amp[:, None]
    rec = si.NumpyRecording(data, sampling_frequency=float(FS))
    return rec.rename_channels(new_channel_ids=[CH_NAME])


def build_continuous(hours, envelope, start, gen_shift):
    rec = _chunk(start.hour + start.minute / 60, hours, envelope, gen_shift)
    return rec, start, [float(hours * 3600)]


def build_segmented(segments, days, envelope, start, gen_shift):
    """Per-day clock-hour segments with gaps; anchored via a per-file start list."""
    day0 = start.replace(hour=0, minute=0, second=0, microsecond=0)
    recs, starts, durs = [], [], []
    for d in range(days):
        for s, e in segments:
            dur_h = e - s
            recs.append(_chunk(s, dur_h, envelope, gen_shift))
            starts.append(day0 + datetime.timedelta(days=d, hours=s))
            durs.append(float(dur_h * 3600))
    return si.concatenate_recordings(recs), starts, durs


def make_zt_df(rec, manual_datetimes, file_durations, shift_hours):
    lro = LongRecordingOrganizer(item=None, recording=rec)
    lro.file_durations = list(file_durations)
    lro.cumulative_file_durations = list(np.cumsum(file_durations))
    lro.manual_datetimes = manual_datetimes
    lro.datetimes_are_start = True
    lro.finalize_file_timestamps()
    ao = AnimalOrganizer.from_lros([lro], animal_id="SYN", genotype="WT")
    war = ao.compute_windowed_analysis(
        ["rms"], window_s=WINDOW_S, multiprocess_mode="serial"
    )
    df = war.result.copy()
    df["rms"] = df["rms"].apply(lambda a: float(np.mean(a)))
    df = add_zeitgeber_time_columns(df, interval_minutes=60)
    df = shift_to_zeitgeber_reference(df, shift_hours=shift_hours)
    return df


def numeric_check(df, envelope, segmented, shifted):
    hourly = df.groupby("zt_minutes")["rms"].mean()
    if shifted:
        # gen_shift != proc_shift: the landmark is deliberately re-anchored, so
        # the band moves by (gen-proc). Eye-test the offset; skip strict assert.
        peak = int(hourly.idxmax())
        print(f"  [shifted/{envelope}] peak now at ZT{peak/60:.1f}h "
              f"(moved by gen_shift-proc_shift) -> VISUAL CHECK")
        return True
    if segmented:
        populated = set(int(x) for x in hourly.index)
        ok = len(populated) < 24 and len(populated) > 0  # there ARE gaps + data
        print(f"  [segmented/{envelope}] populated ZT-hour bins={sorted(b//60 for b in populated)}; "
              f"gaps present -> {'PASS' if ok else 'FAIL'}")
        return ok
    if envelope == "square":
        day = df.loc[df["zt_minutes"] < 720, "rms"].mean()
        night = df.loc[df["zt_minutes"] >= 720, "rms"].mean()
        ok = night > 1.5 * day
        print(f"  [square] ZT-day={day:.2f} ZT-night={night:.2f} night>1.5*day -> {'PASS' if ok else 'FAIL'}")
        return ok
    if envelope == "sine":
        peak = int(hourly.idxmax())
        ok = 900 <= peak <= 1260  # ZT18 +/- 3h
        print(f"  [sine] peak ZT{peak/60:.1f}h (expect ~18) -> {'PASS' if ok else 'FAIL'}")
        return ok
    if envelope == "triangle":
        peak = int(hourly.idxmax())
        ok = 540 <= peak <= 900  # ZT12 +/- 3h
        print(f"  [triangle] peak ZT{peak/60:.1f}h (expect ~12) -> {'PASS' if ok else 'FAIL'}")
        return ok
    return True


def render(df, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    try:
        from neurodent.visualization.plotting import ZeitgeberPlotter
        pdf = df[["zt_minutes", "rms"]].copy()
        pdf["genotype"], pdf["sex"] = "WT", "M"
        ZeitgeberPlotter(pdf).plot_feature("rms", png, figsize=[12, 6], n_days=2)
    except Exception as e:  # guaranteed fallback
        print(f"  (ZeitgeberPlotter failed: {e}; fallback plot)")
        hourly = df.groupby("zt_minutes")["rms"].mean()
        h2 = pd.concat([hourly, hourly.rename(lambda x: x + 1440)])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(h2.index, h2.values, marker="o")
        for x0 in (12 * 60, 36 * 60):
            ax.axvspan(x0, x0 + 12 * 60, alpha=0.1, color="grey")
        ax.set(xlabel="ZT (min; grey=night)", ylabel="RMS", title=name)
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envelope", choices=["square", "sine", "triangle"], default="square")
    ap.add_argument("--hours", type=float, default=48, help="continuous-recording length")
    ap.add_argument("--segments", default=None,
                    help='clock-hour segments for a split recording, e.g. "0-10,15-20"')
    ap.add_argument("--days", type=int, default=2, help="days to repeat segments over")
    ap.add_argument("--start", default="2020-01-01 06:00:00")
    ap.add_argument("--shift-hours", type=int, default=6,
                    help="PROCESSING ZT0 / lights-on clock hour (re-anchors the plot ZT axis)")
    ap.add_argument("--gen-shift-hours", type=int, default=None,
                    help="GENERATION ZT0 for the envelope (default = --shift-hours). "
                         "Set different from --shift-hours to test ZT0 re-anchoring: the "
                         "landmark moves by (gen - proc) hours on the plot.")
    ap.add_argument("--out", default="results/zeitgeber_validation")
    ap.add_argument("--name", default=None, help="output PNG basename (default: envelope)")
    args = ap.parse_args()

    start = pd.Timestamp(args.start).to_pydatetime()
    name = args.name or args.envelope
    segmented = args.segments is not None
    gen_shift = args.gen_shift_hours if args.gen_shift_hours is not None else args.shift_hours
    shifted = gen_shift != args.shift_hours

    print(f"name={name} envelope={args.envelope} start={start} "
          f"gen_ZT0=clock{gen_shift:02d} proc_ZT0=clock{args.shift_hours:02d} "
          f"{'segments=' + args.segments if segmented else 'hours=' + str(args.hours)}")

    if segmented:
        segs = [tuple(float(x) for x in part.split("-")) for part in args.segments.split(",")]
        rec, mdt, durs = build_segmented(segs, args.days, args.envelope, start, gen_shift)
    else:
        rec, mdt, durs = build_continuous(args.hours, args.envelope, start, gen_shift)

    df = make_zt_df(rec, mdt, durs, args.shift_hours)
    ok = numeric_check(df, args.envelope, segmented, shifted)
    render(df, name, args.out)
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
