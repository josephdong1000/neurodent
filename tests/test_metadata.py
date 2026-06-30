"""
Tests for neurodent.core.metadata module.
"""

import pytest
import pandas as pd
from neurodent.core import metadata


# Sample config for testing
SAMPLE_CONFIG = {
    "ANIMAL_METADATA": [
        {"id": "M1", "sex": "Male", "genotype": "WT"},
        {"id": "F1", "sex": "Female", "genotype": "Mut"},
        {"id": "M2", "sex": "Male", "genotype": "Het"},
    ]
}


class TestLoadAnimalMetadata:
    """Tests for load_animal_metadata function."""

    def test_basic_load(self):
        """Test loading a valid config."""
        result = metadata.load_animal_metadata(SAMPLE_CONFIG)

        assert "M1" in result
        assert result["M1"]["sex"] == "Male"
        assert result["M1"]["genotype"] == "WT"

        assert "F1" in result
        assert result["F1"]["sex"] == "Female"
        assert result["F1"]["genotype"] == "Mut"

    def test_missing_animal_metadata_key(self):
        """Test error when ANIMAL_METADATA is missing."""
        with pytest.raises(KeyError, match="ANIMAL_METADATA"):
            metadata.load_animal_metadata({})

    def test_missing_id_in_entry(self):
        """Test error when an entry lacks 'id'."""
        bad_config = {"ANIMAL_METADATA": [{"sex": "Male", "genotype": "WT"}]}
        with pytest.raises(ValueError, match="Missing 'id'"):
            metadata.load_animal_metadata(bad_config)


class TestResolveMetadata:
    """Tests for resolve_metadata function."""

    def test_found(self):
        """Test successful lookup."""
        animal_meta = {"M1": {"sex": "Male", "genotype": "WT"}}
        result = metadata.resolve_metadata("M1", animal_meta)
        assert result["sex"] == "Male"
        assert result["genotype"] == "WT"

    def test_not_found(self):
        """Test error for missing animal."""
        animal_meta = {"M1": {"sex": "Male", "genotype": "WT"}}
        with pytest.raises(KeyError, match="not found"):
            metadata.resolve_metadata("UNKNOWN", animal_meta)


class TestEnrichMetadata:
    """Tests for enrich_metadata function."""

    def test_basic_enrichment(self):
        """Test adding sex/genotype columns."""
        animal_meta = {
            "M1": {"sex": "Male", "genotype": "WT"},
            "F1": {"sex": "Female", "genotype": "Mut"},
        }
        df = pd.DataFrame({"animal": ["M1", "F1"], "value": [1, 2]})

        result = metadata.enrich_metadata(df, animal_meta)

        assert list(result["sex"]) == ["Male", "Female"]
        assert list(result["genotype"]) == ["WT", "Mut"]

    def test_missing_animal_raises(self):
        """Test error when animal not in metadata."""
        animal_meta = {"M1": {"sex": "Male", "genotype": "WT"}}
        df = pd.DataFrame({"animal": ["M1", "UNKNOWN"], "value": [1, 2]})

        with pytest.raises(KeyError, match="not found"):
            metadata.enrich_metadata(df, animal_meta)

    def test_no_animal_column(self):
        """Test graceful handling when no 'animal' column."""
        animal_meta = {"M1": {"sex": "Male", "genotype": "WT"}}
        df = pd.DataFrame({"value": [1, 2]})

        # Should return unchanged
        result = metadata.enrich_metadata(df, animal_meta)
        assert "sex" not in result.columns
        assert "genotype" not in result.columns

    def test_empty_dataframe(self):
        """Test enrichment of empty DataFrame."""
        animal_meta = {"M1": {"sex": "Male", "genotype": "WT"}}
        df = pd.DataFrame({"animal": [], "value": []})

        result = metadata.enrich_metadata(df, animal_meta)
        assert "sex" in result.columns
        assert "genotype" in result.columns
        assert len(result) == 0

    def test_missing_sex_field(self):
        """Test handling when entry is missing 'sex' field."""
        animal_meta = {"M1": {"genotype": "WT"}}  # No sex
        df = pd.DataFrame({"animal": ["M1"], "value": [1]})

        result = metadata.enrich_metadata(df, animal_meta)
        assert pd.isna(result["sex"].iloc[0]) or result["sex"].iloc[0] is None

    def test_missing_genotype_field(self):
        """Test handling when entry is missing 'genotype' field."""
        animal_meta = {"M1": {"sex": "Male"}}  # No genotype
        df = pd.DataFrame({"animal": ["M1"], "value": [1]})

        result = metadata.enrich_metadata(df, animal_meta)
        assert pd.isna(result["genotype"].iloc[0]) or result["genotype"].iloc[0] is None


class TestSexNormalization:
    """Tests for sex normalization via SEX_MAP (populated by default, so strict)."""

    @pytest.mark.parametrize("raw,expected", [
        ("M", "Male"),
        ("m", "Male"),
        ("male", "Male"),
        ("Male", "Male"),
        ("F", "Female"),
        ("f", "Female"),
        ("female", "Female"),
        ("Female", "Female"),
    ])
    def test_alias_resolves_to_canonical(self, raw, expected):
        config = {"ANIMAL_METADATA": [{"id": "A1", "sex": raw}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["sex"] == expected

    def test_unknown_raises(self):
        """A value not covered by the populated SEX_MAP raises (strict)."""
        config = {"ANIMAL_METADATA": [{"id": "A1", "sex": "UNKNOWN"}]}
        with pytest.raises(ValueError, match="Unrecognized sex value"):
            metadata.load_animal_metadata(config)

    def test_none_stays_none(self):
        config = {"ANIMAL_METADATA": [{"id": "A1", "sex": None}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["sex"] is None


class TestGenotypeNormalization:
    """Tests for genotype normalization via GENOTYPE_MAP (default empty = passthrough,
    so these need no constant mutation)."""

    def test_empty_map_passthrough(self):
        """Default empty GENOTYPE_MAP keeps the raw genotype string as-is."""
        from neurodent import constants
        assert constants.GENOTYPE_MAP == {}  # default
        config = {"ANIMAL_METADATA": [{"id": "A1", "genotype": "Arx(F/y);Parvcre+"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["genotype"] == "Arx(F/y);Parvcre+"

    def test_none_stays_none(self):
        config = {"ANIMAL_METADATA": [{"id": "A1", "genotype": None}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["genotype"] is None

    def test_missing_genotype_key_defaults_none(self):
        config = {"ANIMAL_METADATA": [{"id": "A1", "sex": "M"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["genotype"] is None

    def test_gene_accepted_as_alias_for_genotype(self):
        """A config may write the legacy 'gene'; it normalizes to the internal 'genotype'."""
        config = {"ANIMAL_METADATA": [{"id": "A1", "sex": "Male", "gene": "WT"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["genotype"] == "WT"
        assert "gene" not in result["A1"]  # collapsed to the canonical key

    def test_genotype_wins_when_both_keys_present(self):
        """If both 'gene' and 'genotype' are given, 'genotype' takes precedence."""
        config = {"ANIMAL_METADATA": [{"id": "A1", "gene": "KO", "genotype": "WT"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["genotype"] == "WT"
        assert "gene" not in result["A1"]


@pytest.mark.mutates_constants
class TestGenotypeNormalizationConfigured:
    """Tests for genotype normalization when GENOTYPE_MAP is populated (mirrors sex)."""

    @pytest.fixture
    def genotype_map(self):
        from neurodent import constants
        original = constants.GENOTYPE_MAP
        constants.GENOTYPE_MAP = {
            "KO": ["KO", "Arx(F/y);Parvcre+"],
            "WT": ["WT", "Arx(wt);Parvcre-"],
        }
        yield
        constants.GENOTYPE_MAP = original

    def test_value_resolves_to_canonical(self, genotype_map):
        config = {"ANIMAL_METADATA": [{"id": "A1", "genotype": "Arx(F/y);Parvcre+"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["genotype"] == "KO"

    def test_self_label_is_idempotent(self, genotype_map):
        config = {"ANIMAL_METADATA": [{"id": "A1", "genotype": "KO"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["genotype"] == "KO"

    def test_unknown_raises(self, genotype_map):
        """A value not covered by the populated GENOTYPE_MAP raises (strict)."""
        config = {"ANIMAL_METADATA": [{"id": "A1", "genotype": "Mut"}]}
        with pytest.raises(ValueError, match="Unrecognized genotype value"):
            metadata.load_animal_metadata(config)


class TestLoadAnimalMetadataEdgeCases:
    """Edge case tests for load_animal_metadata."""

    def test_empty_list(self):
        """Test loading empty ANIMAL_METADATA list."""
        config = {"ANIMAL_METADATA": []}
        result = metadata.load_animal_metadata(config)
        assert result == {}

    def test_duplicate_ids_last_wins(self):
        """Test that duplicate IDs use last occurrence."""
        config = {
            "ANIMAL_METADATA": [
                {"id": "M1", "sex": "Male", "genotype": "WT"},
                {"id": "M1", "sex": "Female", "genotype": "Mut"},  # Duplicate
            ]
        }
        result = metadata.load_animal_metadata(config)
        assert result["M1"]["sex"] == "Female"
        assert result["M1"]["genotype"] == "Mut"

    def test_extra_fields_preserved(self):
        """Test that extra fields in metadata are preserved."""
        config = {
            "ANIMAL_METADATA": [
                {"id": "M1", "sex": "Male", "genotype": "WT", "cohort": 1, "notes": "test"},
            ]
        }
        result = metadata.load_animal_metadata(config)
        assert result["M1"]["cohort"] == 1
        assert result["M1"]["notes"] == "test"

    def test_null_sex_genotype_values(self):
        """Test entries with null/None values."""
        config = {
            "ANIMAL_METADATA": [
                {"id": "M1", "sex": None, "genotype": None},
            ]
        }
        result = metadata.load_animal_metadata(config)
        assert result["M1"]["sex"] is None
        assert result["M1"]["genotype"] is None


@pytest.mark.mutates_constants
class TestInjectConfigAliases:
    """Tests for apply_samples_config function."""

    def test_injects_animal_metadata(self):
        """Test that ANIMAL_METADATA is injected into constants."""
        from neurodent import constants
        from neurodent.workflow.utils import apply_samples_config

        # Save original state
        original = getattr(constants, 'ANIMAL_METADATA', None)

        config = {
            "ANIMAL_METADATA": [
                {"id": "TEST1", "sex": "Male", "genotype": "WT"},
            ]
        }
        apply_samples_config(config)

        assert "TEST1" in constants.ANIMAL_METADATA
        assert constants.ANIMAL_METADATA["TEST1"]["sex"] == "Male"

        # Restore
        if original is not None:
            constants.ANIMAL_METADATA = original

    def test_no_animal_metadata_leaves_empty(self):
        """Test that missing ANIMAL_METADATA doesn't crash."""
        from neurodent import constants
        from neurodent.workflow.utils import apply_samples_config

        # Save original state
        original_meta = getattr(constants, 'ANIMAL_METADATA', None)
        original_map = getattr(constants, 'GENOTYPE_MAP', None)

        # Explicitly clear/set state for validation
        constants.ANIMAL_METADATA = {}

        config = {"GENOTYPE_MAP": {"WT": ["WT"]}}  # No ANIMAL_METADATA
        apply_samples_config(config)

        # Should remain empty (no auto-convert anymore)
        assert constants.ANIMAL_METADATA == {}

        # Restore
        if original_meta is not None:
            constants.ANIMAL_METADATA = original_meta
        if original_map is not None:
            constants.GENOTYPE_MAP = original_map

    def test_injects_genotype_and_sex_maps(self):
        """GENOTYPE_MAP and SEX_MAP are injected into constants for normalization."""
        from neurodent import constants
        from neurodent.workflow.utils import apply_samples_config

        original_genotype = getattr(constants, "GENOTYPE_MAP", None)
        original_sex = getattr(constants, "SEX_MAP", None)
        try:
            config = {
                "GENOTYPE_MAP": {"KO": ["Arx(F/y);Parvcre+"]},
                "SEX_MAP": {"Male": ["dude"]},
            }
            apply_samples_config(config)
            assert constants.GENOTYPE_MAP == {"KO": ["Arx(F/y);Parvcre+"]}
            assert constants.SEX_MAP == {"Male": ["dude"]}
        finally:
            if original_genotype is not None:
                constants.GENOTYPE_MAP = original_genotype
            if original_sex is not None:
                constants.SEX_MAP = original_sex
