
import pytest
import pandas as pd
from neurodent.core import zeitgeber

def test_enrich_with_genotype_aliases():
    """
    Test enrich_genotype_metadata using the genotype_aliases dictionary.
    Verifies the simplified backward compatibility wrapper.
    """
    # Setup data
    df = pd.DataFrame({
        "animal": ["Anim1-M", "Anim2", "Anim3-F"],
        "genotype": ["legacy_g1", "legacy_g2", "M_WT"]
    })
    
    # Use keys that encode metadata explicitly, as 'magic' inference is removed.
    genotype_aliases = {
        "M_HOMO": ["Anim1-M"],  # Encodes Male, HOMO
        "_HET": ["Anim2"],      # Encodes Sex=None, HET
        "F_WT": ["Anim3-F"]     # Encodes Female, WT (Strict requirement: must be in alias)
    }
    
    # Run enrichment
    result = zeitgeber.enrich_genotype_metadata(df, genotype_aliases=genotype_aliases)
    
    # 1. Check Anim1-M
    # Key="M_HOMO" -> Sex=Male, Gene=HOMO
    row1 = result[result["animal"] == "Anim1-M"].iloc[0]
    assert row1["genotype"] == "HOMO"
    assert row1["sex"] == "Male"
    
    # 2. Check Anim2
    # Key="_HET" -> Sex=None, Gene=HET
    row2 = result[result["animal"] == "Anim2"].iloc[0]
    assert row2["genotype"] == "HET"
    # Sex is None
    assert pd.isna(row2["sex"]) or row2["sex"] is None
    
    # 3. Check Anim3-F
    # Key="F_WT" -> Sex=Female, Gene=WT
    row3 = result[result["animal"] == "Anim3-F"].iloc[0]
    assert row3["genotype"] == "WT"
    assert row3["sex"] == "Female"

def test_enrich_aliases_precedence():
    """Verify alias takes precedence over genotype parsing for Gene."""
    df = pd.DataFrame({
        "animal": ["Mouse-1"],
        "genotype": ["M_WT"] # Original data implies Male, WT
    })
    
    # Alias says Mouse-1 is MUTANT. Use _MUTANT to bypass M/F parsing and just get gene.
    aliases = {"_MUTANT": ["Mouse-1"]}
    
    result = zeitgeber.enrich_genotype_metadata(df, genotype_aliases=aliases)
    
    # Should be MUTANT, not WT
    assert result.iloc[0]["genotype"] == "MUTANT"
    # Sex becomes None/NaN because alias "_MUTANT" doesn't provide it
    assert pd.isna(result.iloc[0]["sex"]) or result.iloc[0]["sex"] is None
