"""An animal's recording days come out in date order, whatever order their names sort in.

``from_lros`` used to order animaldays by sorting their 'Mon-DD-YYYY' names as text, so
'Feb-01-2020' came before 'Jan-31-2020', 'Jul' before 'Jun' and 'Nov' before 'Oct'. The first
animalday sets the reference channel spelling, and anything that pairs per-day lists by
position inherits the order, which is how spike counts ended up on another day's rows. The
direct constructor kept discovery order instead, so its reference depended on how the files
happened to be listed. Both now order by time.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
import spikeinterface.core as si

from neurodent.loading import AnimalOrganizer, LongRecordingOrganizer

RATE = 1000


def _recording(name, start, seconds=60, channels=("LMot", "RMot")):
    traces = np.zeros((seconds * RATE, len(channels)), dtype=np.float32)
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=RATE, channel_ids=list(channels))
    lro = LongRecordingOrganizer(item=None, recording=rec)
    lro.item = name
    lro.file_end_datetimes = [start + timedelta(seconds=seconds)]
    return lro


def test_from_lros_orders_days_by_date_across_month_and_year_boundaries():
    lros = [
        _recording("feb", datetime(2020, 2, 1, 9)),
        _recording("dec", datetime(2019, 12, 31, 9)),
        _recording("jan", datetime(2020, 1, 31, 9)),
    ]

    ao = AnimalOrganizer.from_lros(lros, animal_id="A1")

    assert [day.split()[-1] for day in ao.animaldays] == ["Dec-31-2019", "Jan-31-2020", "Feb-01-2020"]
    assert [lro.item for lro in ao.long_recordings] == ["dec", "jan", "feb"]
    assert [lro.animalday for lro in ao.long_recordings] == ao.animaldays


@pytest.mark.mutates_constants
def test_the_reference_spelling_comes_from_the_earliest_day():
    from neurodent.workflow import apply_samples_config

    apply_samples_config({"channels": {"LMot": ["L Motor", "L Motor Ctx"], "RMot": ["R Motor", "R Motor Ctx"]}})
    lros = [
        _recording("feb", datetime(2020, 2, 1, 9), channels=("L Motor Ctx", "R Motor Ctx")),
        _recording("jan", datetime(2020, 1, 31, 9), channels=("L Motor", "R Motor")),
    ]

    ao = AnimalOrganizer.from_lros(lros, animal_id="A1")

    assert ao.channel_names == ["L Motor", "R Motor"], "Jan-31 is first in time, Feb-01 first as text"


HEADER = "Entity,BinColumn,Label,ProbeInfo,SampleRate,Units,Precision,LastEdit"


def _pair(folder, stem, names, seconds=30):
    folder.mkdir(parents=True, exist_ok=True)
    rows = [f"{i},{i},{n},{n},{RATE},µV,float32," for i, n in enumerate(names, start=1)]
    (folder / f"{stem}_Meta.csv").write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    (folder / f"{stem}_ColMajor.bin").write_bytes(np.zeros((seconds * RATE, len(names)), np.float32).tobytes())


@pytest.mark.integration
@pytest.mark.mutates_constants
def test_the_constructor_orders_sessions_by_time_not_by_listing(tmp_path):
    """'a_later' is listed first but recorded second, and spells its channels differently."""
    from neurodent.workflow import apply_samples_config

    apply_samples_config({"channels": {"LMot": ["L Motor", "L Motor Ctx"], "RMot": ["R Motor", "R Motor Ctx"]}})
    _pair(tmp_path / "a_later", "rec-1", ["L Motor Ctx", "R Motor Ctx"])
    _pair(tmp_path / "b_earlier", "rec-1", ["L Motor", "R Motor"])

    ao = AnimalOrganizer(
        [str(tmp_path / "{session}" / f"rec-{{index}}{s}") for s in ("_ColMajor.bin", "_Meta.csv")],
        animal_id="X1",
        lro_kwargs={
            "mode": "si", "extract_func": "neurodent.readers:read_bin_csv_pair",
            "datetimes_are_start": True,
            "manual_datetimes": {"a_later": "2020-01-31 12:00:00", "b_earlier": "2020-01-30 12:00:00"},
        },
    )

    assert [day.rsplit("_", 1)[-1] for day in ao.animaldays] == ["earlier", "later"]
    assert ao.channel_names == ["L Motor", "R Motor"]
