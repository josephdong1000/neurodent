"""What an animal's config entry means to the loader, checked without reading any file.

Two kinds of silent misreading are covered here. A key inside a ``sources`` entry that the
loader does not read used to be dropped without a word, so a typo or a key in the wrong
place changed what loaded. And whether a recording was split down to the animal's channels
was decided once per animal, so a source's own ``channel_subset`` was thrown away whenever
the animal had none, and a caller that forgot to pass the animal's list got every channel
of a file holding four animals. Splitting is now decided per source and read from the
config by the loader itself.
"""

from types import SimpleNamespace

import pytest

from neurodent import constants
from neurodent.workflow.utils import animal_loading, expand_animals_config, load_animal_recordings
from neurodent.workflow.utils.config import SOURCE_KEYS

PATTERN = "{data_root}/{animal}/{session}/{index}.bin"


def _animal(animal_id="A1", **extra):
    return {"id": animal_id, "genotype": "WT", "sex": "M", **extra}


def _expand(*animals):
    return expand_animals_config({"data_root": "/data", "animals": list(animals)})


class TestSourceKeys:
    @pytest.mark.parametrize("key", ["datetimes_are_start", "chanel_subset", "group", "bad_channels", "id"])
    def test_unknown_source_key_raises(self, key):
        source = {"pattern": PATTERN, key: None}

        with pytest.raises(ValueError, match=rf"animal 'A1', source 0: unknown key\(s\) \['{key}'\]"):
            _expand(_animal(sources=[source]))

    def test_datetimes_are_start_points_into_lro_kwargs(self):
        with pytest.raises(ValueError, match="belongs inside the source's lro_kwargs"):
            _expand(_animal(sources=[{"pattern": PATTERN, "datetimes_are_start": False}]))

    @pytest.mark.parametrize("sources", [[], None, {"pattern": PATTERN}])
    def test_sources_must_be_a_nonempty_list(self, sources):
        """An empty or null list used to fall back to the animal's own pattern silently."""
        with pytest.raises(ValueError, match="sources must be a non-empty list"):
            _expand(_animal(sources=sources))

    def test_source_must_be_a_mapping(self):
        with pytest.raises(ValueError, match="a source must be a mapping"):
            _expand(_animal(sources=[PATTERN]))

    def test_every_key_the_loader_reads_is_accepted(self):
        source = {
            "pattern": PATTERN,
            "lro_kwargs": {"mode": "si"},
            "manual_datetime": "2020-01-01 00:00:00",
            "skip_sessions": ["bad*"],
            "channel_subset": None,
        }
        assert set(source) == SOURCE_KEYS

        result = _expand(_animal(sources=[source]))

        assert result["_animal_overrides"]["A1"]["sources"] == [source]

    def test_excluded_animal_is_not_checked(self):
        result = _expand(_animal(), _animal("B2", exclude=True, sources=[{"typo": 1}]))

        assert [a["id"] for a in result["animals"]] == ["A1"]


class TestChannelSubsetShape:
    @pytest.mark.parametrize("value", ["A-010", [], ["A-010", 3]])
    @pytest.mark.parametrize("level", ["animal", "source"])
    def test_anything_but_a_list_of_names_raises(self, value, level):
        """A bare string used to be accepted and split as one channel per character."""
        animal = (
            _animal(channel_subset=value) if level == "animal"
            else _animal(sources=[{"pattern": PATTERN, "channel_subset": value}])
        )

        with pytest.raises(ValueError, match="channel_subset must be null or a non-empty list"):
            _expand(animal)

    def test_a_channel_listed_twice_raises(self):
        with pytest.raises(ValueError, match=r"lists \['A-010'\] more than once"):
            _expand(_animal(channel_subset=["A-010", "A-012", "A-010"]))

    def test_null_is_accepted_at_both_levels(self):
        _expand(_animal(channel_subset=None, sources=[{"pattern": PATTERN, "channel_subset": None}]))


class TestGroups:
    def test_group_on_an_animal_that_splits_nothing_raises(self):
        with pytest.raises(ValueError, match="declares group 'G' but no channel_subset"):
            _expand(_animal(group="G"))

    def test_animal_level_null_on_a_grouped_animal_raises_that_error_not_a_type_error(self):
        with pytest.raises(ValueError, match="declares group 'G' but no channel_subset"):
            _expand(_animal(group="G", channel_subset=None))

    def test_a_source_level_list_is_enough_for_a_group(self):
        result = _expand(_animal(group="G", sources=[{"pattern": PATTERN, "channel_subset": ["A-010"]}]))

        assert result["_animal_groups"] == {"A1": "G"}

    def test_source_level_lists_join_the_overlap_check(self):
        with pytest.raises(ValueError, match="'A-010' is assigned to both 'A1' and 'B2' in group 'G'"):
            _expand(
                _animal(group="G", channel_subset=["A-010"]),
                _animal("B2", group="G", sources=[{"pattern": PATTERN, "channel_subset": ["A-010"]}]),
            )

    def test_one_animal_may_claim_a_channel_on_two_of_its_sources(self):
        _expand(_animal(
            group="G", channel_subset=["A-010"],
            sources=[{"pattern": PATTERN}, {"pattern": "{data_root}/{animal}/deeper/{index}.bin"}],
        ))


class _SpyOrganizer:
    """Stands in for AnimalOrganizer and records what each source was asked to do."""

    created = []

    def __init__(self, pattern, animal_id, skip_sessions, lro_kwargs, validate_only):
        self.pattern = pattern
        self.discovered_as = animal_id
        self.split_to = None
        self.long_recordings = [self]
        self._animalday_folder_groups = {"S1": ["file"]}
        _SpyOrganizer.created.append(self)

    def split(self, groups):
        ((name, subset),) = groups.items()
        self.split_to = subset
        return {name: self}

    @classmethod
    def from_lros(cls, lros, animal_id, genotype, sex):
        return SimpleNamespace(long_recordings=lros)


@pytest.fixture
def load(monkeypatch):
    """Load one animal through the real loader with the organizer replaced by a spy."""
    _SpyOrganizer.created = []
    monkeypatch.setattr(animal_loading, "AnimalOrganizer", _SpyOrganizer)

    def run(animal, *others, channel_subset=None, validate_only=False):
        samples = _expand(animal, *others)
        monkeypatch.setattr(constants, "ANIMAL_METADATA", {
            a["id"]: {"genotype": "WT", "sex": "M"} for a in [animal, *others]
        })
        config = {"analysis": {"war_generation": {"pattern": PATTERN, "lro_kwargs": {}}}}
        load_animal_recordings(
            samples, config, [("", animal["id"], "")], animal["id"],
            channel_subset=channel_subset, validate_only=validate_only,
        )
        return [(spy.discovered_as, spy.split_to) for spy in _SpyOrganizer.created]

    return run


class TestSplitIsDecidedPerSource:
    def test_a_source_list_splits_an_animal_that_has_none_of_its_own(self, load):
        """Thrown away before: splitting was decided from the animal-level key alone."""
        animal = _animal(sources=[{"pattern": PATTERN, "channel_subset": ["A-010", "A-012"]}])

        assert load(animal) == [("A1", ["A-010", "A-012"])]

    def test_null_opts_one_source_out_and_its_sibling_inherits(self, load):
        animal = _animal(
            channel_subset=["A-010", "A-012"],
            sources=[{"pattern": PATTERN}, {"pattern": PATTERN, "channel_subset": None}],
        )

        assert load(animal) == [("A1", ["A-010", "A-012"]), ("A1", None)]

    def test_only_a_split_source_looks_in_the_group_folder(self, load):
        """An unsplit source in the shared folder would load every animal of the group as one."""
        animal = _animal(
            group="G", channel_subset=["A-010"],
            sources=[{"pattern": PATTERN}, {"pattern": PATTERN, "channel_subset": None}],
        )

        assert load(animal) == [("G", ["A-010"]), ("A1", None)]

    def test_validate_only_discovers_each_source_the_same_way(self, load):
        animal = _animal(
            group="G", channel_subset=["A-010"],
            sources=[{"pattern": PATTERN}, {"pattern": PATTERN, "channel_subset": None}],
        )

        assert load(animal, validate_only=True) == [("G", None), ("A1", None)]


class TestTheConfigDecidesTheChannels:
    def test_a_caller_that_omits_the_list_still_gets_the_split(self, load):
        """Before, an omitted list loaded all channels of a four-animal file without a word."""
        assert load(_animal(channel_subset=["A-010", "A-012"])) == [("A1", ["A-010", "A-012"])]

    def test_a_caller_passing_the_configured_list_is_accepted(self, load):
        animal = _animal(channel_subset=["A-010", "A-012"])

        assert load(animal, channel_subset=["A-010", "A-012"]) == [("A1", ["A-010", "A-012"])]

    def test_a_caller_passing_a_different_list_raises(self, load):
        with pytest.raises(ValueError, match="the dataset config assigns"):
            load(_animal(channel_subset=["A-010", "A-012"]), channel_subset=["B-010", "B-012"])

    def test_a_caller_passing_a_list_for_an_unsplit_animal_raises(self, load):
        """Silently ignored before."""
        with pytest.raises(ValueError, match="the dataset config assigns None"):
            load(_animal(), channel_subset=["A-010"])

    def test_an_animal_without_a_list_is_not_split(self, load):
        assert load(_animal()) == [("A1", None)]
