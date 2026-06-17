#!/usr/bin/env python3
"""Generate a synthetic on-disk EDF dataset to drive the REAL Snakemake pipeline.

Each scenario is a synthetic recording whose amplitude is a known function of
Zeitgeber Time (square/sine/triangle), written to disk as EDF(s) with correct
timings/headers, then run through the *unchanged* pipeline as a separate dataset
(``NEURODENT_DATASET=zeitgeber_test``). Because the pipeline's zeitgeber plots are
grouped by ``gene``x``sex`` (not per-animal), **each scenario is given a distinct
``gene``** so it appears as its own row in ``results/zeitgeber_plots/NN_<feature>.png``.

This materializes what an in-memory check cannot: the files flow through fragment
RMS filtering, channel/LOF filtering, flattening, zeitgeber, diagnostics and EP —
confirming the pipeline processes real on-disk files completely, and that the
shapes come out right (triangle peaks at ZT12, square high under night shading,
``gap_split`` shows the missing-segment ZT gap via a per-session manual_datetime list).

Run on the cluster (writes GBs):
    sbatch --job-name=zt_make --cpus-per-task=4 --mem=24G --time=4:00:00 \
      --wrap="cd <repo> && uv run python scripts/make_zeitgeber_test_dataset.py --replicates 2"

Then:
    NEURODENT_DATASET=zeitgeber_test uv run snakemake --snakefile workflow/Snakefile --profile <slurm>
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

# scripts/ is on sys.path when run as `python scripts/...`; reuse the envelope math.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_zeitgeber_timestamps as vz  # noqa: E402

from neurodent.core import LongRecordingOrganizer  # noqa: E402

FS = vz.FS  # 1000 Hz
# 8 channels matching config.yaml standardization.channel_reorder (no Hip — the
# manual channel filter rejects LHip/RHip). Names parse via CHNAME/LR aliases.
CHANNELS = ["L Mot", "R Mot", "L Bar", "R Bar", "L Aud", "R Aud", "L Vis", "R Vis"]
LOW_UV, HIGH_UV = 80.0, 300.0  # keep window RMS inside fragment filter [min 50, max 500]

# scenario -> dict(envelope, hours, gen_shift, gap). gap uses per-day clock segments.
SCENARIOS = {
    "square_24h":    dict(envelope="square",   hours=24, gen_shift=6, gap=False),
    "square_90h":    dict(envelope="square",   hours=90, gen_shift=6, gap=False),
    "sine_24h":      dict(envelope="sine",     hours=24, gen_shift=6, gap=False),
    "triangle_24h":  dict(envelope="triangle", hours=24, gen_shift=6, gap=False),
    "square_shift0": dict(envelope="square",   hours=24, gen_shift=0, gap=False),
    "gap_split":     dict(envelope="square",   hours=None, gen_shift=6, gap=True,
                          segments=[(0, 10), (15, 20)], days=2),
}
START = datetime.datetime(2020, 1, 1, 6, 0, 0)  # ZT0 = clock 06:00 (pipeline shift=6)


def _envelope_uv(zt_h, envelope):
    """Envelope shape scaled into [LOW_UV, HIGH_UV] microvolts."""
    shape01 = (vz.amplitude(zt_h, envelope) - vz.LOW) / (vz.HIGH - vz.LOW)
    return (LOW_UV + shape01 * (HIGH_UV - LOW_UV)).astype(np.float32)


def _recording(start_clock_h, dur_h, envelope, gen_shift, seed):
    """8-channel NumpyRecording: white noise (uV) shaped by the ZT envelope."""
    import spikeinterface.core as si
    rng = np.random.default_rng(seed)
    n = int(dur_h * 3600 * FS)
    t_h = np.arange(n, dtype=np.float64) / (3600 * FS)
    zt_gen = (start_clock_h + t_h - gen_shift) % 24
    amp = _envelope_uv(zt_gen, envelope)              # (n,)
    data = (rng.standard_normal((n, len(CHANNELS))).astype(np.float32) * amp[:, None])
    rec = si.NumpyRecording(data, sampling_frequency=float(FS))
    return rec.rename_channels(new_channel_ids=CHANNELS)


def _save_edf(rec, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lro = LongRecordingOrganizer(item=None, recording=rec)
    lro.channel_names = CHANNELS
    lro.save_to_edf(path, overwrite=True)


def _manual_dt(spec):
    """Compute the manual_datetime (scalar dict, or per-session list for gap)."""
    if spec["gap"]:
        manual_dt = {}
        day0 = START.replace(hour=0, minute=0, second=0, microsecond=0)
        for d in range(spec["days"]):
            manual_dt[f"_{d}_"] = [
                (day0 + datetime.timedelta(days=d, hours=s)).strftime("%Y-%m-%d %H:%M:%S")
                for s, _ in spec["segments"]
            ]
        return manual_dt
    return {"_0_": START.strftime("%Y-%m-%d %H:%M:%S")}


def _entry(animal, name, manual_dt):
    return {
        "id": animal,
        "gene": name,        # scenario name as genotype -> its own facet row
        "sex": "M",
        "pattern": "{data_root}/{animal}/{session}/{index}.EDF",
        "lro_kwargs": {"mode": "mne", "extract_func": "read_raw_edf"},
        "manual_datetime": manual_dt,
    }


def _entry_only(name, spec, data_root, rep):
    return _entry(f"{name}__r{rep}", name, _manual_dt(spec))


def write_scenario(name, spec, data_root, rep, seed):
    """Write EDF(s) for one (scenario, replicate); return its samples-json entry."""
    animal = f"{name}__r{rep}"
    base = data_root / animal
    if spec["gap"]:
        # Per-day clock segments -> separate EDFs; per-session manual_datetime LIST.
        for d in range(spec["days"]):
            for idx, (s, e) in enumerate(spec["segments"]):
                rec = _recording(s, e - s, spec["envelope"], spec["gen_shift"], seed + d * 10 + idx)
                _save_edf(rec, base / f"_{d}_" / f"{idx}.EDF")
    else:
        rec = _recording(START.hour, spec["hours"], spec["envelope"], spec["gen_shift"], seed)
        _save_edf(rec, base / "_0_" / "0.EDF")
    return _entry(animal, name, _manual_dt(spec))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/scr1/users/dongjp/zeitgeber_test_data")
    ap.add_argument("--replicates", type=int, default=2,
                    help="synthetic replicates per scenario (>=2 lets the EP/stats branch run)")
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS),
                    help="subset of scenario names to generate")
    ap.add_argument("--temp-directory", default="/scr1/users/dongjp/zeitgeber_test_tmp")
    ap.add_argument("--config-only", action="store_true",
                    help="write the samples json + dataset yaml WITHOUT generating EDFs "
                         "(for DAG dry-runs / setup)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    animals = []
    for name in args.scenarios:
        spec = SCENARIOS[name]
        for rep in range(args.replicates):
            if args.config_only:
                animals.append(_entry_only(name, spec, data_root, rep))
                continue
            print(f"[gen] {name} replicate {rep} ...", flush=True)
            animals.append(write_scenario(name, spec, data_root, rep, seed=1000 + rep))

    samples = {
        "data_root": str(data_root),
        "_note": "Synthetic pipeline-validation dataset. gene == scenario (one facet "
                 "row per scenario in zeitgeber plots). Generated by "
                 "scripts/make_zeitgeber_test_dataset.py.",
        "LR_ALIASES": {"L": ["L "], "R": ["R "]},
        "CHNAME_ALIASES": {"Mot": ["Mot"], "Bar": ["Bar"], "Aud": ["Aud"], "Vis": ["Vis"]},
        "animals": animals,
    }
    samples_path = Path("config/samples_zeitgeber_test.json")
    samples_path.write_text(json.dumps(samples, indent=4))
    print(f"[ok] wrote {samples_path} ({len(animals)} animals)")

    dataset_yaml = f"""# Synthetic zeitgeber pipeline-validation dataset (generated).
temp_directory: "{args.temp_directory}"

samples:
  samples_file: "config/samples_zeitgeber_test.json"

analysis:
  war_generation:
    pattern:
      - "{{data_root}}/{{animal}}/{{session}}/{{index}}.EDF"
    assume_from_number: false
    datetimes_are_start: true
    lro_kwargs:
      mode: "mne"
      extract_func: "read_raw_edf"
      multiprocess_mode: "serial"
"""
    dataset_path = Path("config/datasets/zeitgeber_test.yaml")
    dataset_path.write_text(dataset_yaml)
    print(f"[ok] wrote {dataset_path}")
    print("\nNext: NEURODENT_DATASET=zeitgeber_test uv run snakemake "
          "--snakefile workflow/Snakefile --profile <slurm_profile>")


if __name__ == "__main__":
    main()
