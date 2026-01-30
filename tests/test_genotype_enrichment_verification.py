
import pandas as pd
import pytest
import logging
from neurodent.core.zeitgeber import enrich_genotype_metadata

# Mock Genotype Aliases similar to samples.json
MOCK_ALIASES = {
    "MWT": ["M1", "M2_Alias"],
    "FMut": ["F1", "F2_Alias"],
    "MHet": ["M3"],
}

def test_enrich_from_aliases():
    """Test standard lookup from alias map."""
    df = pd.DataFrame({
        "animal": ["M1", "F1", "M3"]
    })
    
    # Run enrichment
    enriched = enrich_genotype_metadata(df, genotype_aliases=MOCK_ALIASES)
    
    # Check Genotype
    assert enriched.loc[0, "genotype"] == "MWT"
    assert enriched.loc[1, "genotype"] == "FMut"
    assert enriched.loc[2, "genotype"] == "MHet"
    
    # Check Sex (inferred from first char of canonical genotype)
    assert enriched.loc[0, "sex"] == "Male"
    assert enriched.loc[1, "sex"] == "Female"
    assert enriched.loc[2, "sex"] == "Male"
    
    # Check Gene (inferred from rest of canonical genotype)
    assert enriched.loc[0, "gene"] == "WT"
    assert enriched.loc[1, "gene"] == "Mut"
    assert enriched.loc[2, "gene"] == "Het"

def test_enrich_fallback_underscore():
    """Test fallback parsing for names with underscores (e.g. M_WT)."""
    df = pd.DataFrame({
        "animal": ["Unknown1"],
        "genotype": ["M_WT"]
    })
    
    enriched = enrich_genotype_metadata(df, genotype_aliases=MOCK_ALIASES)
    
    assert enriched.loc[0, "sex"] == "Male"
    assert enriched.loc[0, "gene"] == "WT"

def test_enrich_fallback_no_underscore():
    """Test fallback parsing for names without underscores (e.g. FWT)."""
    df = pd.DataFrame({
        "animal": ["Unknown2"],
        "genotype": ["FWT"]
    })
    
    enriched = enrich_genotype_metadata(df, genotype_aliases=MOCK_ALIASES)
    
    assert enriched.loc[0, "sex"] == "Female"
    assert enriched.loc[0, "gene"] == "WT"
    # Crucially, check that it didn't slice [2:] (which would give 'T')
    assert enriched.loc[0, "gene"] != "T"

def test_enrich_suffix_sex_inference():
    """Test sex inference from animal name suffix if genotype is totally missing."""
    df = pd.DataFrame({
        "animal": ["Mouse-M", "Mouse-F", "MouseNoSuffix"],
        "genotype": [None, None, None]
    })
    
    # Note: genotype_aliases is empty/irrelevant here
    enriched = enrich_genotype_metadata(df, genotype_aliases={})
    
    assert enriched.loc[0, "sex"] == "Male"
    assert enriched.loc[1, "sex"] == "Female"
    # No suffix, no genotype -> None (or NaN)
    assert pd.isna(enriched.loc[2, "sex"])

def test_mixed_alias_and_fallback():
    """Test a mix of animals in alias map and those relying on fallback."""
    df = pd.DataFrame({
        "animal": ["M1", "Unknown_F"],
        "genotype": ["Garbage", "F_Mut"] # M1 has "Garbage" but is in Alias map (should win?)
    })
    
    enriched = enrich_genotype_metadata(df, genotype_aliases=MOCK_ALIASES)
    
    # M1 is in alias map as MWT. Map should take precedence over "Garbage"
    assert enriched.loc[0, "genotype"] == "MWT"
    assert enriched.loc[0, "gene"] == "WT"
    
    # Unknown_F is not in map. Should fallback to parsing "F_Mut"
    # NOTE: current logic blindly uses fallback if not in map
    assert enriched.loc[1, "sex"] == "Female"
    assert enriched.loc[1, "gene"] == "Mut"

def test_missing_animal_column():
    """Test behavior when animal column is missing (only genotype column)."""
    df = pd.DataFrame({
        "genotype": ["M_WT", "FMut"]
    })
    
    enriched = enrich_genotype_metadata(df, genotype_aliases=MOCK_ALIASES)
    
    assert enriched.loc[0, "gene"] == "WT"
    assert enriched.loc[1, "gene"] == "Mut"

if __name__ == "__main__":
    # verification script manually running tests if called directly
    try:
        test_enrich_from_aliases()
        print("test_enrich_from_aliases PASSED")
        test_enrich_fallback_underscore()
        print("test_enrich_fallback_underscore PASSED")
        test_enrich_fallback_no_underscore()
        print("test_enrich_fallback_no_underscore PASSED")
        test_enrich_suffix_sex_inference()
        print("test_enrich_suffix_sex_inference PASSED")
        test_mixed_alias_and_fallback()
        print("test_mixed_alias_and_fallback PASSED")
        test_missing_animal_column()
        print("test_missing_animal_column PASSED")
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
