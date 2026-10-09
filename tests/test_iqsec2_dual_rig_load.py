"""A two-rig animal loaded end to end, on small synthetic bin/csv pairs.

The seven IQSEC2 dual-rig animals depend on three things in the loader that nothing else
exercises, and getting any of them wrong would only show on the cluster or not at all:

* a source's own ``channel_subset: null`` cancels the split the animal-level subset asks for;
* a split leg and an unsplit leg merge into one animalday with every sample in its channel;
* the per-file start list ``bin_file_starts`` writes is paired with the files the way the
  loader pairs it. The two sort independently, so a change to either sort gives every file
  another file's time, and nothing at load would notice: the times are still all distinct.

The test config test_iqsec2_dual_rig_channel_order.py checks the shipped YAML. This one checks
that the YAML means what it is meant to.
"""

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "iqsec2" / "generate_config.py"
HEADER = "Entity,BinColumn,Label,ProbeInfo,SampleRate,Units,Precision,LastEdit"
RATE = 1000
SECONDS = 2
SUFFIXES = ("_ColMajor.bin", "_Meta.csv")


def _pair(folder, stem, names, values, last_edit):
    """Write one ColMajor bin and its sidecar. Column i holds the constant values[i]."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{i},{i},{name},{name},{RATE},µV,float32,{last_edit:%Y-%m-%dT%H:%M:%S}"
        for i, name in enumerate(names, start=1)
    ]
    (folder / f"{stem}_Meta.csv").write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    data = np.repeat(np.asarray(values, dtype=np.float32)[None, :], RATE * SECONDS, axis=0)
    (folder / f"{stem}_ColMajor.bin").write_bytes(data.tobytes(order="F"))


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("iqsec2_generate_config", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is guarded by __name__, so nothing runs
    return module


@pytest.mark.integration
@pytest.mark.mutates_constants
def test_two_rig_animal_loads_as_one_animalday(tmp_path, generator):
    from neurodent.core.utils import resolve_channels
    from neurodent.workflow import apply_samples_config
    from neurodent.workflow.utils import expand_animals_config, load_animal_recordings

    day = datetime(2020, 2, 3)

    # Intan stand-in: one file holding two animals on ports A and B, its columns not in
    # subset order, so the split has to pick and reorder.
    _pair(tmp_path / "rhd", "S1_0", ["B-012", "A-012", "B-010", "A-010"], [9, 2, 9, 1],
          day + timedelta(seconds=SECONDS))

    # DataWave stand-in: this animal alone. LHip carries 100 plus the file's index, so the
    # merged recording shows which file landed where. The counter wrapped, as animal 78's
    # did: index 10 was recorded first. 1, 2, 10 also sorts differently as text, so a text
    # sort on either side fails here too.
    probe = ["Intan Input (1)/PortA L Vis Ctx", "Intan Input (1)/PortA L Hipp"]
    ends = {10: day.replace(hour=11), 1: day.replace(hour=12), 2: day.replace(hour=13)}
    for index, end in ends.items():
        _pair(tmp_path / "bins" / "Datawave_9", f"rec-{index}", probe, [1, 100 + index], end)

    bin_pattern = [f"{{data_root}}/bins/{{session}}/rec-{{index}}{s}" for s in SUFFIXES]
    starts = generator.bin_file_starts(
        [p.replace("{data_root}", str(tmp_path)) for p in bin_pattern]
    )

    reader = {
        "mode": "si",
        "extract_func": "neurodent.readers:read_bin_csv_pair",
        "datetimes_are_start": True,
    }
    samples = expand_animals_config({
        "data_root": str(tmp_path),
        "channels": {
            "LVis": ["A-010", "B-010", probe[0]],
            "LHip": ["A-012", "B-012", probe[1]],
        },
        "animals": [{
            "id": "X9", "genotype": "WT", "sex": "M",
            "channel_subset": ["A-010", "A-012"],
            "sources": [
                {
                    "pattern": [f"{{data_root}}/rhd/{{session}}_{{index}}{s}" for s in SUFFIXES],
                    "lro_kwargs": dict(reader),
                    "manual_datetime": {"S1": f"{day:%Y-%m-%d %H:%M:%S}"},
                },
                {
                    "pattern": bin_pattern,
                    "channel_subset": None,
                    "lro_kwargs": dict(reader),
                    "manual_datetime": starts,
                },
            ],
        }],
    })
    apply_samples_config(samples)
    config = {"analysis": {"war_generation": {"pattern": "{index}", "lro_kwargs": {"mode": "si"}}}}

    ao = load_animal_recordings(
        samples, config, [("", "X9", "")], "X9",
        channel_subset=samples["_animal_channel_subsets"]["X9"],
    )

    assert len(ao.long_recordings) == 1, "same date, so the two legs merge into one animalday"
    lro = ao.long_recordings[0]
    assert resolve_channels(lro.channel_names) == ["LVis", "LHip"]

    traces = lro.LongRecording.get_traces()
    assert np.all(traces[:, 0] == 1), "an LVis sample from either rig must stay in LVis"

    # Each file's samples must sit at that file's own time. Its sidecar LastEdit is its end.
    placed = list(zip(traces[:: RATE * SECONDS, 1].tolist(), lro.file_end_datetimes))
    assert placed == [
        (2, day + timedelta(seconds=SECONDS)),
        (110, ends[10]),
        (101, ends[1]),
        (102, ends[2]),
    ]
