"""
Animal Metadata Module
======================
Functions for loading and enriching animal metadata (sex, gene) from config.
"""

import logging
import pandas as pd

from .. import constants
from .utils import normalize_value_from_aliases

logger = logging.getLogger(__name__)


def load_animal_metadata(samples_config: dict) -> dict:
    """
    Load ANIMAL_METADATA from samples config into a lookup dict.

    The ``sex`` and ``gene`` fields are normalized identically: each raw value is
    mapped to its canonical label via the field's alias dict (``constants.SEX_ALIASES``
    and ``constants.GENE_ALIASES``). A value matching no alias is kept as-is with a
    warning; an empty alias dict (``GENE_ALIASES`` by default) is a silent passthrough,
    so datasets without a ``GENE_ALIASES`` block keep their raw ``gene`` strings. Any
    extra fields on an entry (e.g. ``cohort``) are preserved unchanged.

    Args:
        samples_config: Dict containing "ANIMAL_METADATA" key with list of
                        {"id": str, "sex": str, "gene": str} objects.

    Returns:
        Dict mapping animal_id -> {"sex": str, "gene": str, ...}, with ``sex``/``gene``
        normalized to their canonical labels.

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
        # Copy all fields from entry (extra fields like cohort/notes pass through)
        metadata_dict[animal_id] = entry.copy()
        # Normalize the `sex` and `gene` fields the same way: ensure the key exists
        # (default None), then map the raw value to its canonical label via the
        # field's alias dict (read from constants at call time so inject_config_aliases
        # overrides apply). An empty alias dict is a passthrough that keeps the raw
        # value with no warning -- this is GENE_ALIASES' default, so datasets without
        # a GENE_ALIASES block keep their `gene` string verbatim (backward compatible).
        for field, alias_dict in (("sex", constants.SEX_ALIASES), ("gene", constants.GENE_ALIASES)):
            raw_value = metadata_dict[animal_id].get(field)
            if raw_value is None:
                metadata_dict[animal_id][field] = None
                continue
            if not alias_dict:
                continue  # no aliases configured -> keep raw value, no warning
            normalized = normalize_value_from_aliases(raw_value, alias_dict)
            if normalized is None:
                logger.warning(
                    f"Unrecognized {field} value '{raw_value}' for animal '{animal_id}'; "
                    f"expected one of "
                    f"{[a for aliases in alias_dict.values() for a in aliases]}"
                )
            else:
                metadata_dict[animal_id][field] = normalized
    
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
    Add 'sex' and 'genotype' columns to DataFrame from animal metadata.

    The metadata dict key remains 'gene' (the dataset-config field name); the
    canonical DataFrame column is 'genotype'.

    Args:
        df: DataFrame with 'animal' column.
        animal_metadata: Dict from load_animal_metadata().

    Returns:
        DataFrame with 'sex' and 'genotype' columns added.
    
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
    df["genotype"] = genes

    logger.info(f"Enriched {len(df)} rows with sex/genotype metadata")
    return df
