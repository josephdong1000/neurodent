"""Unit tests for ``WindowAnalysisResult._enrich_metadata_from_constants``.

This helper is the single chokepoint that re-enriches ``sex``/``genotype`` from
``constants.ANIMAL_METADATA`` at every WAR construction, applied identically to both
fields and to BOTH the object attributes AND the per-row ``result`` columns (the
columns are what downstream renderers read). It is the robust fix for the
``sex="Unknown"`` fragility.
"""
import pandas as pd
import pytest

from neurodent import constants
from neurodent.results.window_analysis_result import WindowAnalysisResult


def _make_war(animal_id="X1", sex="Unknown", genotype="KO"):
    """Build a WAR bypassing ``__init__``, with baked attrs + per-row columns."""
    war = WindowAnalysisResult.__new__(WindowAnalysisResult)
    war.animal_id = animal_id
    war.sex = sex
    war.genotype = genotype
    war.result = pd.DataFrame(
        {
            "animal": [animal_id] * 3,
            "sex": [sex] * 3,
            "genotype": [genotype] * 3,
            "value": [1, 2, 3],
        }
    )
    return war


@pytest.fixture
def reset_metadata():
    original = constants.ANIMAL_METADATA
    yield
    constants.ANIMAL_METADATA = original


@pytest.mark.mutates_constants
class TestReenrichMetadataFromConstants:
    def test_populated_updates_attrs_and_columns(self, reset_metadata):
        """The load-bearing case: both attrs and per-row columns get the config value
        (ANIMAL_METADATA key and WAR canonical are both 'genotype')."""
        war = _make_war(sex="Unknown", genotype="KO")
        constants.ANIMAL_METADATA = {"X1": {"sex": "Male", "genotype": "WT"}}
        war._enrich_metadata_from_constants()
        assert war.sex == "Male"
        assert war.genotype == "WT"
        assert list(war.result["sex"].unique()) == ["Male"]
        assert list(war.result["genotype"].unique()) == ["WT"]

    def test_empty_metadata_keeps_baked(self, reset_metadata):
        """Portability: a standalone load (empty ANIMAL_METADATA) keeps baked values."""
        war = _make_war(sex="Unknown", genotype="KO")
        constants.ANIMAL_METADATA = {}
        war._enrich_metadata_from_constants()
        assert war.sex == "Unknown"
        assert war.genotype == "KO"
        assert list(war.result["sex"].unique()) == ["Unknown"]

    def test_partial_coverage_keeps_baked(self, reset_metadata):
        """An animal absent from ANIMAL_METADATA keeps its baked values."""
        war = _make_war(animal_id="X1", sex="Unknown", genotype="KO")
        constants.ANIMAL_METADATA = {"OTHER": {"sex": "Male", "genotype": "WT"}}
        war._enrich_metadata_from_constants()
        assert war.sex == "Unknown"
        assert war.genotype == "KO"

    def test_missing_field_only_updates_present(self, reset_metadata):
        """A None/missing metadata field must not overwrite a baked value."""
        war = _make_war(sex="Unknown", genotype="KO")
        constants.ANIMAL_METADATA = {"X1": {"genotype": "WT"}}  # no sex
        war._enrich_metadata_from_constants()
        assert war.genotype == "WT"
        assert war.sex == "Unknown"  # not overwritten with None

    def test_idempotent(self, reset_metadata):
        war = _make_war(sex="Unknown", genotype="KO")
        constants.ANIMAL_METADATA = {"X1": {"sex": "Female", "genotype": "Het"}}
        war._enrich_metadata_from_constants()
        war._enrich_metadata_from_constants()
        assert war.sex == "Female"
        assert war.genotype == "Het"
        assert list(war.result["genotype"].unique()) == ["Het"]
