import pytest
import pandas as pd
import numpy as np
from neurodent.analysis import zeitgeber


def test_enrich_metadata_legacy_fallback():
    """CASE 1: Standard M_WT format (Default Strategy)"""
    df_legacy = pd.DataFrame({"genotype": ["M_WT", "F_Het", "M_Mut"]})
    df_res = zeitgeber.enrich_genotype_metadata(df_legacy)
    assert df_res.iloc[0]["sex"] == "Male"
    assert df_res.iloc[0]["gene"] == "WT"
    assert df_res.iloc[1]["sex"] == "Female"
    assert df_res.iloc[2]["gene"] == "Mut"


def test_enrich_metadata_custom_regex():
    """CASE 2, 3: Custom Regex patterns"""
    # CASE 2: Custom Regex (e.g. "Male_WildType")
    df_custom = pd.DataFrame({"genotype": ["Male_WildType", "Female_Heterozygous"]})
    pattern = r"(?P<sex>Male|Female)_(?P<gene>\w+)"
    df_res_custom = zeitgeber.enrich_genotype_metadata(
        df_custom, genotype_pattern=pattern
    )

    assert "sex" in df_res_custom.columns
    assert "gene" in df_res_custom.columns
    assert df_res_custom.iloc[0]["sex"] == "Male"
    assert df_res_custom.iloc[0]["gene"] == "WildType"
    assert df_res_custom.iloc[1]["gene"] == "Heterozygous"

    # CASE 3: Different Delimiter & Order (e.g. "WT-M")
    df_swap = pd.DataFrame({"genotype": ["WT-M", "KO-F"]})
    pattern_swap = r"(?P<gene>\w+)-(?P<sex>[MF])"
    df_res_swap = zeitgeber.enrich_genotype_metadata(
        df_swap, genotype_pattern=pattern_swap
    )

    assert df_res_swap.iloc[0]["gene"] == "WT"
    assert df_res_swap.iloc[0]["sex"] == "Male"  # Mapped from "M"
    assert df_res_swap.iloc[1]["gene"] == "KO"
    assert df_res_swap.iloc[1]["sex"] == "Female"  # Mapped from "F"


def test_enrich_metadata_no_match():
    """CASE 4: Invalid/No Match"""
    df_bad = pd.DataFrame({"genotype": ["InvalidString", "NoMatch"]})
    pattern_strict = r"(?P<sex>[MF])_(?P<gene>.+)"
    df_res_bad = zeitgeber.enrich_genotype_metadata(
        df_bad, genotype_pattern=pattern_strict
    )

    # Columns created with NaNs
    assert "sex" in df_res_bad.columns
    assert pd.isna(df_res_bad.iloc[0]["sex"])
    assert "gene" in df_res_bad.columns
    assert pd.isna(df_res_bad.iloc[0]["gene"])


def test_enrich_metadata_pipeline_integration():
    """CASE 5: Passed through Pipeline"""
    df_pipe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01 12:00", "2023-01-01 12:00"]),
            "genotype": ["WT-M", "KO-F"],
            "feature": [1, 2],
            "total_minutes": [720, 720],  # 12:00 = 720m
        }
    )
    processed = zeitgeber.run_zeitgeber_pipeline(
        df_pipe, genotype_pattern=r"(?P<gene>\w+)-(?P<sex>[MF])"
    )
    assert processed.iloc[0]["sex"] == "Male"
    assert processed.iloc[0]["gene"] == "WT"


def test_enrich_metadata_sex_mapper():
    """CASE 6 & 9: Custom Sex Mapper & Unmapped values"""
    # CASE 6: Custom Sex Mapper
    df_map = pd.DataFrame({"genotype": ["WT-h", "KO-z"]})
    processed_map = zeitgeber.enrich_genotype_metadata(
        df_map,
        genotype_pattern=r"(?P<gene>\w+)-(?P<sex>\w)",
        sex_mapper={"h": "Hermaphrodite", "z": "Zebra"},
    )
    assert processed_map.iloc[0]["sex"] == "Hermaphrodite"
    assert processed_map.iloc[1]["sex"] == "Zebra"

    # CASE 9: Sex Mapper with Unmapped Values
    df_unmapped = pd.DataFrame({"genotype": ["M_WT", "X_WT"]})
    # M maps to Male, X is unknown/unmapped
    df_res_unmapped = zeitgeber.enrich_genotype_metadata(
        df_unmapped,
        genotype_pattern=r"(?P<sex>[MX])_(?P<gene>.+)",
        sex_mapper={"M": "Male"},
    )
    assert df_res_unmapped.iloc[0]["sex"] == "Male"
    # Unmapped value "X" should remain "X"
    assert df_res_unmapped.iloc[1]["sex"] == "X"


def test_enrich_metadata_partial_match():
    """CASE 7 & 8: Partial Matches & Missing Groups"""
    # CASE 7: Partial Match (Mixed Schema)
    df_mixed = pd.DataFrame({"genotype": ["M_WT", "JustString"]})
    pattern_mixed = r"(?P<sex>[MF])_(?P<gene>.+)"
    df_res_mixed = zeitgeber.enrich_genotype_metadata(
        df_mixed, genotype_pattern=pattern_mixed
    )

    assert df_res_mixed.iloc[0]["sex"] == "Male"
    assert df_res_mixed.iloc[0]["gene"] == "WT"
    assert pd.isna(df_res_mixed.iloc[1]["sex"])

    # CASE 8: Regex Missing Groups
    df_groups = pd.DataFrame({"genotype": ["M_WT"]})
    # Pattern has no named groups for sex/gene (wrong names)
    pattern_wrong = r"(?P<other>M)_(?P<stuff>.+)"
    df_res_wrong = zeitgeber.enrich_genotype_metadata(
        df_groups, genotype_pattern=pattern_wrong
    )

    assert "sex" not in df_res_wrong.columns
    assert "gene" not in df_res_wrong.columns
    assert "other" in df_res_wrong.columns


def test_enrich_metadata_empty_df():
    """CASE 10: Empty DataFrame"""
    df_empty = pd.DataFrame({"genotype": []})
    df_res_empty = zeitgeber.enrich_genotype_metadata(df_empty)
    assert "sex" in df_res_empty.columns
    assert "gene" in df_res_empty.columns
    assert len(df_res_empty) == 0
