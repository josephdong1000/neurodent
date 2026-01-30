"""
Animal Metadata Module
======================
Functions for loading and enriching animal metadata (sex, gene) from config.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_animal_metadata(samples_config: dict) -> dict:
    """
    Load ANIMAL_METADATA from samples config into a lookup dict.
    
    Args:
        samples_config: Dict containing "ANIMAL_METADATA" key with list of
                        {"id": str, "sex": str, "gene": str} objects.
    
    Returns:
        Dict mapping animal_id -> {"sex": str, "gene": str}
    
    Raises:
        KeyError: If ANIMAL_METADATA is missing.
        ValueError: If entries are malformed.
    """
    if "ANIMAL_METADATA" not in samples_config:
        raise KeyError("ANIMAL_METADATA not found in samples config")
    
    metadata_list = samples_config["ANIMAL_METADATA"]
    metadata_dict = {}
    
    for entry in metadata_list:
        if "id" not in entry:
            raise ValueError(f"Missing 'id' in metadata entry: {entry}")
        
        animal_id = entry["id"]
        # Copy all fields from entry
        metadata_dict[animal_id] = entry.copy()
        # Ensure sex and gene keys exist (default to None if missing)
        # This preserves backward compatibility with code expecting these keys
        if "sex" not in metadata_dict[animal_id]:
            metadata_dict[animal_id]["sex"] = None
        if "gene" not in metadata_dict[animal_id]:
            metadata_dict[animal_id]["gene"] = None
    
    logger.info(f"Loaded metadata for {len(metadata_dict)} animals")
    return metadata_dict


def resolve_metadata(animal_id: str, animal_metadata: dict) -> dict:
    """
    Lookup metadata for a single animal.
    
    Args:
        animal_id: The animal identifier.
        animal_metadata: Dict from load_animal_metadata().
    
    Returns:
        Dict with {"sex": str, "gene": str}.
    
    Raises:
        KeyError: If animal_id not found in metadata.
    """
    if animal_id not in animal_metadata:
        raise KeyError(f"Animal '{animal_id}' not found in ANIMAL_METADATA")
    
    return animal_metadata[animal_id]


def enrich_metadata(df: pd.DataFrame, animal_metadata: dict) -> pd.DataFrame:
    """
    Add 'sex' and 'gene' columns to DataFrame from animal metadata.
    
    Args:
        df: DataFrame with 'animal' column.
        animal_metadata: Dict from load_animal_metadata().
    
    Returns:
        DataFrame with 'sex' and 'gene' columns added.
    
    Raises:
        KeyError: If any animal in df is not found in metadata.
    """
    if "animal" not in df.columns:
        logger.warning("DataFrame has no 'animal' column, skipping metadata enrichment")
        return df
    
    df = df.copy()
    
    # Lookup sex and gene for each animal
    sexes = []
    genes = []
    missing_animals = set()
    
    for animal_id in df["animal"]:
        if animal_id in animal_metadata:
            meta = animal_metadata[animal_id]
            sexes.append(meta.get("sex"))
            genes.append(meta.get("gene"))
        else:
            missing_animals.add(animal_id)
            sexes.append(None)
            genes.append(None)
    
    if missing_animals:
        raise KeyError(f"Animals not found in ANIMAL_METADATA: {missing_animals}")
    
    df["sex"] = sexes
    df["gene"] = genes
    
    logger.info(f"Enriched {len(df)} rows with sex/gene metadata")
    return df
