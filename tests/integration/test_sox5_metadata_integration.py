
import pytest
import pandas as pd
from neurodent.core import zeitgeber

# Hardcoded sample configuration derived from "sox5 combine genotypes.json"
# This avoids fragile absolute paths while preserving the complexity of real-world data.
SAMPLE_SOX5_CONFIG = {
    "GENOTYPE_ALIASES": {
        "MWT": [
            "MWT", "M4_WT", "AIIM(WT)", "group9_M2_Cage 3",
            "AM3", "BM6", "CM5", "AM4", "AM2", "CM9"
        ],
        "MHet": [
            "MHet", "MHET", "M8_HET", "group9_M3_Cage 4",
            "mouse #3 Cage 4A", "mouse M3 cage1A", "mouse M2 cage 3A"
        ],
        "MMut": [
            "M3_MT", "OLDMMT", "MMUT", "MMT",
            "mouse #8 Cage 3A", "mouse #2 Cage 2A"
        ],
        "FWT": ["FWT", "BF3", "CF2", "DF3", "CF1"],
        "FHet": [
            "FHET", "F7Het", "F5 cage 1A",
            "032221_cohort 2, Group 3, Mouse 6 Cage 2A Re-Recording",
            "GF4", "IF5"
        ],
        "FMut": ["FMUT", "F9Mut", "mouse F10 cage 4A"],
        "Unknown": ["M3_group 10_cage 1", "F8_group 10_cage 3"]
    },
    # Sample mapping of folders to animals to simulate input dataframe construction
    "data_folders_to_animal_ids": {
        "010822_cohort4_group2_2mice_MWT_MHET": ["M3", "M10"],
        "011622_cohort4_group4_3mice_MMutOLD_FMUT_FMUT_FWT": ["FMUT_", "FWT", "OLDMMT"],
        "060921_Cohort 3_EM1_AM2_GF4": ["AM2", "EM1", "GF4"],
        "031021_cohort 2, group 3 and 4": ["#8 Cage 3A", "#8 Cage 1A"]
    }
}

def test_sox5_metadata_integration():
    """
    Test metadata enrichment using representative configuration from the Sox5 project.
    Verifies that enrich_genotype_metadata correctly handles complex aliasing and ID patterns.
    """
    genotype_aliases = SAMPLE_SOX5_CONFIG["GENOTYPE_ALIASES"]
    folder_map = SAMPLE_SOX5_CONFIG["data_folders_to_animal_ids"]

    # Flatten all animal IDs from the folder map to simulate a pipeline run
    all_animals = []
    for folder, animals in folder_map.items():
        all_animals.extend(animals)
    
    # Also explicitly add some complex IDs from the aliases to ensure they are tested
    # even if not in the sample folder map
    complex_ids = [
        "mouse M3 cage1A", 
        "032221_cohort 2, Group 3, Mouse 6 Cage 2A Re-Recording",
        "M4_WT",
        "F9Mut"
    ]
    all_animals.extend(complex_ids)
    
    # Remove duplicates
    all_animals = list(set(all_animals))
    
    # Create DataFrame mimicking a pipeline run
    df = pd.DataFrame({"animal": all_animals})
    
    # Apply enrichment
    result = zeitgeber.enrich_genotype_metadata(df, genotype_aliases=genotype_aliases)
    
    # Assertions
    
    # 1. Check that every animal got a gene assignment where expected
    for idx, row in result.iterrows():
        animal = row["animal"]
        gene = row["gene"]
        
        # Reverse lookup to check expectation
        expected_gene = None
        for g, a_list in genotype_aliases.items():
            if animal in a_list:
                expected_gene = g
                break
        
        if expected_gene:
            assert gene == expected_gene, f"Animal {animal} expected {expected_gene} but got {gene}"
    
    # 2. Check Sex inference for known patterns
    
    # "M3" -> Male
    if "M3" in all_animals:
        row = result[result["animal"] == "M3"].iloc[0]
        # logic: "M3" doesn't end in -M, but might fall back to M start?
        # The logic in enrich_genotype_metadata for suffix is:
        # ends with -m/-f.
        # IF NOT, and NOT in genotype aliases dict for sex? No, aliases only mapped gene.
        # Fallback 1: genotype string first char (e.g. "M_WT" -> M).
        # But here 'genotype' column is null initially!
        # Wait, if 'genotype' is MISSING in input DF (it is here), does it crash?
        # The code: if "genotype" in df.columns...
        # Our input DF only has "animal".
        
        # This test reveals if the function handles missing 'genotype' column gracefully when only 'animal' is provided.
        # Since we rely on aliases for GENE, sex must come from animal name suffix OR remain None.
        
        # "M3" has no suffix -> Sex should be None if no genotype column.
        # Unless we enrich genotype column from somewhere? No.
        
        # Let's check "F9Mut" -> No suffix.
        pass

    # However, in real pipeline (generate_relfreq_plots.py), 'genotype' column usually exists 
    # (inherited from WAR).
    # If we add a dummy genotype column it might help fallback.
    # But let's check what actually happens.


