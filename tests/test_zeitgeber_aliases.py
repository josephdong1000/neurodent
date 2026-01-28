
import pytest
import pandas as pd
from neurodent.core import zeitgeber

def test_enrich_with_genotype_aliases():
    """
    Test enrich_genotype_metadata using the genotype_aliases dictionary.
    This verifies the logic refactored from generate_relfreq_plots.py.
    """
    # Setup data
    # Animal1: In alias list, Has suffix
    # Animal2: In alias list, No suffix (should fallback or be None if suffix required - code attempts suffix check)
    # Animal3: Not in alias list, should fallback to genotype parsing
    df = pd.DataFrame({
        "animal": ["Anim1-M", "Anim2", "Anim3-F"],
        "genotype": ["legacy_g1", "legacy_g2", "M_WT"]
    })
    
    genotype_aliases = {
        "HOMO": ["Anim1-M"],
        "HET": ["Anim2"]
    }
    
    # Run enrichment
    result = zeitgeber.enrich_genotype_metadata(df, genotype_aliases=genotype_aliases)
    
    # 1. Check Anim1-M
    # Should get gene="HOMO" from alias
    # Should get sex="Male" from suffix
    row1 = result[result["animal"] == "Anim1-M"].iloc[0]
    assert row1["gene"] == "HOMO"
    assert row1["sex"] == "Male"
    
    # 2. Check Anim2
    # Should get gene="HET" from alias
    # Should get sex=None (no suffix) -> Wait, fallback logic might run?
    # Fallback checks genotype "legacy_g2". "l" is not F/M. So likely Sex is NaN or inferred if code allows.
    # Code: df["sex"] = df["genotype"].str[0].map({"F": "Female", "M": "Male"})
    # "l" -> NaN.
    row2 = result[result["animal"] == "Anim2"].iloc[0]
    assert row2["gene"] == "HET"
    # Sex might be NaN/None
    
    # 3. Check Anim3-F
    # Not in alias. Gene should come from genotype "M_WT" -> "WT"
    # Sex should come from suffix "F" -> "Female" (suffix check runs on all animals)
    row3 = result[result["animal"] == "Anim3-F"].iloc[0]
    assert row3["gene"] == "WT"
    assert row3["sex"] == "Female"

def test_enrich_aliases_precedence():
    """Verify alias takes precedence over genotype parsing for Gene."""
    df = pd.DataFrame({
        "animal": ["Mouse-1"],
        "genotype": ["M_WT"] # Parses to gene=WT
    })
    
    # Alias says Mouse-1 is MUTANT
    aliases = {"MUTANT": ["Mouse-1"]}
    
    result = zeitgeber.enrich_genotype_metadata(df, genotype_aliases=aliases)
    
    # Should be MUTANT, not WT
    assert result.iloc[0]["gene"] == "MUTANT"
