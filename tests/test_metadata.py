"""
Tests for neurodent.core.metadata module.
"""

import pytest
import pandas as pd
from neurodent.core import metadata


# Sample config for testing
SAMPLE_CONFIG = {
    "ANIMAL_METADATA": [
        {"id": "M1", "sex": "Male", "gene": "WT"},
        {"id": "F1", "sex": "Female", "gene": "Mut"},
        {"id": "M2", "sex": "Male", "gene": "Het"},
    ]
}


class TestLoadAnimalMetadata:
    """Tests for load_animal_metadata function."""

    def test_basic_load(self):
        """Test loading a valid config."""
        result = metadata.load_animal_metadata(SAMPLE_CONFIG)
        
        assert "M1" in result
        assert result["M1"]["sex"] == "Male"
        assert result["M1"]["gene"] == "WT"
        
        assert "F1" in result
        assert result["F1"]["sex"] == "Female"
        assert result["F1"]["gene"] == "Mut"

    def test_missing_animal_metadata_key(self):
        """Test error when ANIMAL_METADATA is missing."""
        with pytest.raises(KeyError, match="ANIMAL_METADATA"):
            metadata.load_animal_metadata({})

    def test_missing_id_in_entry(self):
        """Test error when an entry lacks 'id'."""
        bad_config = {"ANIMAL_METADATA": [{"sex": "Male", "gene": "WT"}]}
        with pytest.raises(ValueError, match="Missing 'id'"):
            metadata.load_animal_metadata(bad_config)


class TestResolveMetadata:
    """Tests for resolve_metadata function."""

    def test_found(self):
        """Test successful lookup."""
        animal_meta = {"M1": {"sex": "Male", "gene": "WT"}}
        result = metadata.resolve_metadata("M1", animal_meta)
        assert result["sex"] == "Male"
        assert result["gene"] == "WT"

    def test_not_found(self):
        """Test error for missing animal."""
        animal_meta = {"M1": {"sex": "Male", "gene": "WT"}}
        with pytest.raises(KeyError, match="not found"):
            metadata.resolve_metadata("UNKNOWN", animal_meta)


class TestEnrichMetadata:
    """Tests for enrich_metadata function."""

    def test_basic_enrichment(self):
        """Test adding sex/gene columns."""
        animal_meta = {
            "M1": {"sex": "Male", "gene": "WT"},
            "F1": {"sex": "Female", "gene": "Mut"},
        }
        df = pd.DataFrame({"animal": ["M1", "F1"], "value": [1, 2]})
        
        result = metadata.enrich_metadata(df, animal_meta)
        
        assert list(result["sex"]) == ["Male", "Female"]
        assert list(result["genotype"]) == ["WT", "Mut"]

    def test_missing_animal_raises(self):
        """Test error when animal not in metadata."""
        animal_meta = {"M1": {"sex": "Male", "gene": "WT"}}
        df = pd.DataFrame({"animal": ["M1", "UNKNOWN"], "value": [1, 2]})
        
        with pytest.raises(KeyError, match="not found"):
            metadata.enrich_metadata(df, animal_meta)

    def test_no_animal_column(self):
        """Test graceful handling when no 'animal' column."""
        animal_meta = {"M1": {"sex": "Male", "gene": "WT"}}
        df = pd.DataFrame({"value": [1, 2]})
        
        # Should return unchanged
        result = metadata.enrich_metadata(df, animal_meta)
        assert "sex" not in result.columns
        assert "genotype" not in result.columns

    def test_empty_dataframe(self):
        """Test enrichment of empty DataFrame."""
        animal_meta = {"M1": {"sex": "Male", "gene": "WT"}}
        df = pd.DataFrame({"animal": [], "value": []})
        
        result = metadata.enrich_metadata(df, animal_meta)
        assert "sex" in result.columns
        assert "genotype" in result.columns
        assert len(result) == 0

    def test_missing_sex_field(self):
        """Test handling when entry is missing 'sex' field."""
        animal_meta = {"M1": {"gene": "WT"}}  # No sex
        df = pd.DataFrame({"animal": ["M1"], "value": [1]})
        
        result = metadata.enrich_metadata(df, animal_meta)
        assert pd.isna(result["sex"].iloc[0]) or result["sex"].iloc[0] is None

    def test_missing_gene_field(self):
        """Test handling when entry is missing 'gene' field."""
        animal_meta = {"M1": {"sex": "Male"}}  # No gene
        df = pd.DataFrame({"animal": ["M1"], "value": [1]})
        
        result = metadata.enrich_metadata(df, animal_meta)
        assert pd.isna(result["genotype"].iloc[0]) or result["genotype"].iloc[0] is None


class TestSexNormalization:
    """Tests for sex normalization via SEX_ALIASES."""

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

    def test_unknown_passes_through_with_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            config = {"ANIMAL_METADATA": [{"id": "A1", "sex": "UNKNOWN"}]}
            result = metadata.load_animal_metadata(config)
        assert result["A1"]["sex"] == "UNKNOWN"
        assert "Unrecognized sex value" in caplog.text

    def test_none_stays_none(self):
        config = {"ANIMAL_METADATA": [{"id": "A1", "sex": None}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["sex"] is None


class TestGeneNormalization:
    """Tests for gene normalization via GENE_ALIASES (parallels sex). Default empty
    GENE_ALIASES = passthrough, so these need no constant mutation."""

    def test_empty_aliases_passthrough_no_warning(self, caplog):
        """Default empty GENE_ALIASES keeps the raw gene string with no warning."""
        import logging
        from neurodent import constants
        assert constants.GENE_ALIASES == {}  # default
        with caplog.at_level(logging.WARNING):
            config = {"ANIMAL_METADATA": [{"id": "A1", "gene": "Arx(F/y);Parvcre+"}]}
            result = metadata.load_animal_metadata(config)
        assert result["A1"]["gene"] == "Arx(F/y);Parvcre+"
        assert "Unrecognized" not in caplog.text

    def test_none_stays_none(self):
        config = {"ANIMAL_METADATA": [{"id": "A1", "gene": None}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["gene"] is None

    def test_missing_gene_key_defaults_none(self):
        config = {"ANIMAL_METADATA": [{"id": "A1", "sex": "M"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["gene"] is None

    def test_genotype_accepted_as_alias_for_gene(self):
        """A config may write 'genotype' instead of 'gene'; it normalizes to 'gene'."""
        config = {"ANIMAL_METADATA": [{"id": "A1", "sex": "Male", "genotype": "WT"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["gene"] == "WT"
        assert "genotype" not in result["A1"]  # collapsed to the internal key

    def test_gene_wins_when_both_keys_present(self):
        """If both 'gene' and 'genotype' are given, 'gene' takes precedence."""
        config = {"ANIMAL_METADATA": [{"id": "A1", "gene": "WT", "genotype": "KO"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["gene"] == "WT"
        assert "genotype" not in result["A1"]


@pytest.mark.mutates_constants
class TestGeneNormalizationConfigured:
    """Tests for gene normalization when GENE_ALIASES is populated (mirrors sex)."""

    @pytest.fixture
    def gene_aliases(self):
        from neurodent import constants
        original = constants.GENE_ALIASES
        constants.GENE_ALIASES = {
            "KO": ["KO", "Arx(F/y);Parvcre+"],
            "WT": ["WT", "Arx(wt);Parvcre-"],
        }
        yield
        constants.GENE_ALIASES = original

    def test_alias_resolves_to_canonical(self, gene_aliases):
        config = {"ANIMAL_METADATA": [{"id": "A1", "gene": "Arx(F/y);Parvcre+"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["gene"] == "KO"

    def test_self_alias_is_idempotent(self, gene_aliases):
        config = {"ANIMAL_METADATA": [{"id": "A1", "gene": "KO"}]}
        result = metadata.load_animal_metadata(config)
        assert result["A1"]["gene"] == "KO"

    def test_unknown_passes_through_with_warning(self, gene_aliases, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            config = {"ANIMAL_METADATA": [{"id": "A1", "gene": "Mut"}]}
            result = metadata.load_animal_metadata(config)
        assert result["A1"]["gene"] == "Mut"
        assert "Unrecognized gene value" in caplog.text


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
                {"id": "M1", "sex": "Male", "gene": "WT"},
                {"id": "M1", "sex": "Female", "gene": "Mut"},  # Duplicate
            ]
        }
        result = metadata.load_animal_metadata(config)
        assert result["M1"]["sex"] == "Female"
        assert result["M1"]["gene"] == "Mut"

    def test_extra_fields_preserved(self):
        """Test that extra fields in metadata are preserved."""
        config = {
            "ANIMAL_METADATA": [
                {"id": "M1", "sex": "Male", "gene": "WT", "cohort": 1, "notes": "test"},
            ]
        }
        result = metadata.load_animal_metadata(config)
        assert result["M1"]["cohort"] == 1
        assert result["M1"]["notes"] == "test"

    def test_null_sex_gene_values(self):
        """Test entries with null/None values."""
        config = {
            "ANIMAL_METADATA": [
                {"id": "M1", "sex": None, "gene": None},
            ]
        }
        result = metadata.load_animal_metadata(config)
        assert result["M1"]["sex"] is None
        assert result["M1"]["gene"] is None


@pytest.mark.mutates_constants
class TestInjectConfigAliases:
    """Tests for inject_config_aliases function."""

    def test_injects_animal_metadata(self):
        """Test that ANIMAL_METADATA is injected into constants."""
        from neurodent import constants
        from neurodent.workflow.utils import inject_config_aliases
        
        # Save original state
        original = getattr(constants, 'ANIMAL_METADATA', None)
        
        config = {
            "ANIMAL_METADATA": [
                {"id": "TEST1", "sex": "Male", "gene": "WT"},
            ]
        }
        inject_config_aliases(config)
        
        assert "TEST1" in constants.ANIMAL_METADATA
        assert constants.ANIMAL_METADATA["TEST1"]["sex"] == "Male"
        
        # Restore
        if original is not None:
            constants.ANIMAL_METADATA = original

    def test_no_animal_metadata_leaves_empty(self):
        """Test that missing ANIMAL_METADATA doesn't crash."""
        from neurodent import constants
        from neurodent.workflow.utils import inject_config_aliases
        
        # Save original state
        original_meta = getattr(constants, 'ANIMAL_METADATA', None)
        original_aliases = getattr(constants, 'GENOTYPE_ALIASES', None)
        
        # Explicitly clear/set state for validation
        constants.ANIMAL_METADATA = {}
        
        config = {"GENOTYPE_ALIASES": {"MWT": ["M1"]}}  # No ANIMAL_METADATA
        inject_config_aliases(config)
        
        # Should remain empty (no auto-convert anymore)
        assert constants.ANIMAL_METADATA == {}
        
        # Restore
        if original_meta is not None:
            constants.ANIMAL_METADATA = original_meta
        if original_aliases is not None:
            constants.GENOTYPE_ALIASES = original_aliases

    def test_injects_genotype_aliases(self):
        """Test that GENOTYPE_ALIASES is still injected for legacy uses."""
        from neurodent import constants
        from neurodent.workflow.utils import inject_config_aliases
        
        # Save original state
        original = getattr(constants, 'GENOTYPE_ALIASES', None)
        
        config = {"GENOTYPE_ALIASES": {"TestGeno": ["T1", "T2"]}}
        inject_config_aliases(config)

        assert "TestGeno" in constants.GENOTYPE_ALIASES
        assert constants.GENOTYPE_ALIASES["TestGeno"] == ["T1", "T2"]

        # Restore
        if original is not None:
            constants.GENOTYPE_ALIASES = original

    def test_injects_gene_and_sex_aliases(self):
        """GENE_ALIASES and SEX_ALIASES are injected into constants for normalization."""
        from neurodent import constants
        from neurodent.workflow.utils import inject_config_aliases

        original_gene = getattr(constants, "GENE_ALIASES", None)
        original_sex = getattr(constants, "SEX_ALIASES", None)
        try:
            config = {
                "GENE_ALIASES": {"KO": ["Arx(F/y);Parvcre+"]},
                "SEX_ALIASES": {"Male": ["dude"]},
            }
            inject_config_aliases(config)
            assert constants.GENE_ALIASES == {"KO": ["Arx(F/y);Parvcre+"]}
            assert constants.SEX_ALIASES == {"Male": ["dude"]}
        finally:
            if original_gene is not None:
                constants.GENE_ALIASES = original_gene
            if original_sex is not None:
                constants.SEX_ALIASES = original_sex

