"""Seven IQSEC2 animals were recorded on two rigs, and the two legs get merged.

When the loader merges two recordings into one animalday it compares their channel lists
after translating each name to an abbreviation, and it compares them AS ORDERED LISTS, not as
sets (``lro_merge._validate_merge_compatibility``). The two rigs name the same electrode
completely differently: the Intan leg calls it ``A-010`` and the DataWave leg calls it
``Intan Input (1)/PortA L Vis Ctx``. The merge works only because both happen to translate to
the same eight abbreviations in the same sequence.

That coincidence is load bearing and nothing else guards it. Reordering the ``_ports`` anchor
in the dataset config, or reordering ``AMP_TO_REGION`` in the generator, would turn a working
merge into a hard failure hours into a cluster run. These tests make it a pytest failure
instead.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "datasets" / "iqsec2.yaml"

# The order the DataWave sidecars list their rows in, measured from the files themselves:
# every one of the 707 Meta.csv files across Datawave_76 through Datawave_83 lists its eight
# channels in this order, ascending by the BinColumn column. This is a property of the export,
# so a change here means the data changed, not the code.
DATAWAVE_ORDER = ["LVis", "LHip", "LBar", "LMot", "RMot", "RBar", "RHip", "RVis"]

# The Label text each of those rows carries, in the same order. The dual-rig cohorts use this
# one title-case spelling throughout; older exports also used a lowercase and a positional
# form, which is why the channels map holds more than one spelling per electrode.
DATAWAVE_LABELS = [
    "L Vis Ctx", "L Hipp", "L Barrel Ctx", "L Motor Ctx",
    "R Motor Ctx", "R Barrel Ctx", "R Hipp", "R Vis Ctx",
]

# The seven animals with two sources, and the port each one's Intan leg uses.
DUAL_RIG_PORTS = {
    "IQSEC2_76": "a",
    "IQSEC2_77": "c",
    "IQSEC2_78": "d",
    "IQSEC2_80": "a",
    "IQSEC2_81": "b",
    "IQSEC2_82": "c",
    "IQSEC2_83": "d",
}


@pytest.fixture(scope="module")
def samples():
    return yaml.safe_load(CONFIG.read_text())["samples_data"]


@pytest.fixture(scope="module")
def abbrev_by_raw(samples):
    """Reverse the config's channels map, the way constants.set_channel_map does."""
    reverse = {}
    for abbrev, raws in samples["channels"].items():
        for raw in raws:
            assert raw not in reverse or reverse[raw] == abbrev, (
                f"{raw!r} maps to both {reverse.get(raw)!r} and {abbrev!r}"
            )
            reverse[raw] = abbrev
    return reverse


class TestTheTwoKeySpacesAgree:
    @pytest.mark.parametrize("port", ["a", "b", "c", "d"])
    def test_intan_port_resolves_to_the_datawave_order(self, samples, abbrev_by_raw, port):
        """Each port's channel list must translate to the DataWave order, element for element."""
        channels = samples["_ports"][f"port_{port}"]

        resolved = [abbrev_by_raw[name] for name in channels]

        assert resolved == DATAWAVE_ORDER, (
            f"port_{port} resolves to {resolved}, but the DataWave sidecars list "
            f"{DATAWAVE_ORDER}. Merging the two legs of an animal on this port would raise."
        )

    def test_every_dual_rig_animal_uses_a_pinned_port(self, samples):
        """If an animal gains a second source, its port has to be covered above."""
        animals = {a["id"]: a for a in samples["animals"]}
        with_sources = {
            aid for aid, a in animals.items()
            if any("channel_subset" in s for s in a.get("sources") or [])
        }

        assert with_sources == set(DUAL_RIG_PORTS), (
            "the set of animals with a DataWave source changed; update DUAL_RIG_PORTS"
        )

    @pytest.mark.parametrize("animal_id,port", sorted(DUAL_RIG_PORTS.items()))
    def test_animal_level_subset_is_the_expected_port(self, samples, animal_id, port):
        animals = {a["id"]: a for a in samples["animals"]}

        assert animals[animal_id]["channel_subset"] == samples["_ports"][f"port_{port}"]

    @pytest.mark.parametrize("port_letter", list("ABCD"))
    def test_datawave_names_resolve_to_the_same_order(self, abbrev_by_raw, port_letter):
        """The exact strings these animals' sidecars carry must resolve in sidecar order.

        The channels map holds several spellings per electrode on purpose, because older
        exports used lowercase and positional forms. The dual-rig cohorts use only the
        title-case form below, so that is what gets pinned.
        """
        names = [
            f"Intan Input (1)/Port{port_letter} {label}" for label in DATAWAVE_LABELS
        ]

        missing = [name for name in names if name not in abbrev_by_raw]
        assert not missing, f"absent from the channels map, so they fail at load: {missing}"

        resolved = [abbrev_by_raw[name] for name in names]
        assert resolved == DATAWAVE_ORDER, f"Port{port_letter} resolves to {resolved}"


class TestTheDataWaveSourceCancelsTheSplit:
    """Each DataWave source must say channel_subset: null.

    The loader decides whether to split a recording into per-animal channel groups from a key
    on the ANIMAL, not from the source, so both legs inherit one answer. The Intan leg needs
    the split because one file holds four animals. The DataWave leg does not, and without an
    explicit null it is split against names it does not contain and raises.
    """

    @pytest.mark.parametrize("animal_id", sorted(DUAL_RIG_PORTS))
    def test_null_subset_present(self, samples, animal_id):
        animals = {a["id"]: a for a in samples["animals"]}
        sources = animals[animal_id]["sources"]
        datawave = [s for s in sources if "channel_subset" in s]

        assert len(datawave) == 1
        assert datawave[0]["channel_subset"] is None, (
            "an explicit null is what cancels the inherited split; omitting the key "
            "silently inherits it and omitting the source's own value raises at load"
        )

    @pytest.mark.parametrize("animal_id", sorted(DUAL_RIG_PORTS))
    def test_datetimes_are_start_sits_inside_lro_kwargs(self, samples, animal_id):
        """Beside the pattern it is read by nothing, and nothing validates source keys."""
        animals = {a["id"]: a for a in samples["animals"]}

        for source in animals[animal_id]["sources"]:
            assert "datetimes_are_start" not in source
            assert source["lro_kwargs"]["datetimes_are_start"] is True

    @pytest.mark.parametrize("animal_id", sorted(DUAL_RIG_PORTS))
    def test_no_animal_level_pattern_or_times_survive(self, samples, animal_id):
        """A leftover animal-level pattern makes a source load the other leg a second time."""
        animals = {a["id"]: a for a in samples["animals"]}
        animal = animals[animal_id]

        for key in ("pattern", "lro_kwargs", "manual_datetime", "skip_sessions"):
            assert key not in animal, f"{animal_id} still carries an animal-level {key}"

    @pytest.mark.parametrize("animal_id", sorted(DUAL_RIG_PORTS))
    def test_per_file_times_are_one_list_per_session(self, samples, animal_id):
        """One time per file, not one anchor per session.

        Animal 78's filename counter wrapped partway through its recording, so an anchor plus
        cumulative durations would place files up to 14 hours from where they belong.
        """
        animals = {a["id"]: a for a in samples["animals"]}
        datawave = next(
            s for s in animals[animal_id]["sources"] if "channel_subset" in s
        )

        times = datawave["manual_datetime"]
        assert len(times) == 1
        (values,) = times.values()
        assert isinstance(values, list)
        assert len(values) == len(set(values)), "duplicate timestamps are rejected at load"
