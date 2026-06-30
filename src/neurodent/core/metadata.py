"""
Animal Metadata Module
======================
Functions for loading and enriching animal metadata (sex, genotype) from config.
"""

import logging
import pandas as pd

from .. import constants
from .utils import normalize_value_from_aliases

logger = logging.getLogger(__name__)


def load_animal_metadata(samples_config: dict) -> dict:
    """
    Load ANIMAL_METADATA from samples config into a lookup dict.

    The ``sex`` and ``genotype`` fields are normalized identically: each raw value is
    mapped to its canonical label via the field's map (``constants.SEX_MAP`` and
    ``constants.GENOTYPE_MAP``, exact match). When a map is populated it is
    **authoritative**: a value it does not cover raises ``ValueError`` (config typos
    surface immediately). An empty map (``GENOTYPE_MAP`` by default) is a passthrough,
    so datasets without a ``GENOTYPE_MAP`` block keep their raw ``genotype`` strings.
    Any extra fields on an entry (e.g. ``cohort``) are preserved unchanged.

    The genotype field is canonically ``genotype``; ``gene`` is still accepted as a
    legacy input alias and normalized to the internal ``genotype`` key.

    Args:
        samples_config: Dict containing "ANIMAL_METADATA" key with list of
                        {"id": str, "sex": str, "genotype" (or legacy "gene"): str}
                        objects.

    Returns:
        Dict mapping animal_id -> {"sex": str, "genotype": str, ...}, with
        ``sex``/``genotype`` normalized to their canonical labels.

    Raises:
        KeyError: If ANIMAL_METADATA is missing.
        ValueError: If entries are malformed, or a value is not covered by a populated
            ``SEX_MAP`` / ``GENOTYPE_MAP``.
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
        # 'genotype' is the canonical field; accept the legacy 'gene' spelling as an
        # input alias and normalize it to the internal 'genotype' key.
        _entry = metadata_dict[animal_id]
        if "gene" in _entry:
            _entry.setdefault("genotype", _entry["gene"])
            del _entry["gene"]
        # Normalize the `sex` and `genotype` fields the same way: ensure the key exists
        # (default None), then map the raw value to its canonical label via the field's
        # map (read from constants at call time so apply_samples_config overrides apply).
        # An empty map is a passthrough that keeps the raw value -- this is GENOTYPE_MAP's
        # default, so datasets without a GENOTYPE_MAP block keep their `genotype` string
        # verbatim. A populated map is authoritative: an uncovered value raises.
        for field, value_map in (("sex", constants.SEX_MAP), ("genotype", constants.GENOTYPE_MAP)):
            raw_value = metadata_dict[animal_id].get(field)
            if raw_value is None:
                metadata_dict[animal_id][field] = None
                continue
            if not value_map:
                continue  # no map configured -> keep raw value
            normalized = normalize_value_from_aliases(raw_value, value_map)
            if normalized is None:
                field_const = "SEX_MAP" if field == "sex" else "GENOTYPE_MAP"
                raise ValueError(
                    f"Unrecognized {field} value '{raw_value}' for animal '{animal_id}'; "
                    f"expected one of "
                    f"{[a for aliases in value_map.values() for a in aliases]}. "
                    f"Add it to {field_const} or fix the value."
                )
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
        Dict with {"sex": str, "genotype": str}.

    Raises:
        KeyError: If animal_id not found in metadata.
    """
    if animal_id not in animal_metadata:
        raise KeyError(f"Animal '{animal_id}' not found in ANIMAL_METADATA")
    
    return animal_metadata[animal_id]


def enrich_metadata(df: pd.DataFrame, animal_metadata: dict) -> pd.DataFrame:
    """
    Add 'sex' and 'genotype' columns to DataFrame from animal metadata.

    The metadata dict key and the DataFrame column are both 'genotype' (the canonical
    name); 'gene' is accepted only as a legacy input spelling in the config.

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
    
    # Lookup sex and genotype for each animal
    sexes = []
    genotypes = []
    missing_animals = set()

    for animal_id in df["animal"]:
        if animal_id in animal_metadata:
            meta = animal_metadata[animal_id]
            sexes.append(meta.get("sex"))
            genotypes.append(meta.get("genotype"))
        else:
            missing_animals.add(animal_id)
            sexes.append(None)
            genotypes.append(None)

    if missing_animals:
        raise KeyError(f"Animals not found in ANIMAL_METADATA: {missing_animals}")

    df["sex"] = sexes
    df["genotype"] = genotypes

    logger.info(f"Enriched {len(df)} rows with sex/genotype metadata")
    return df
