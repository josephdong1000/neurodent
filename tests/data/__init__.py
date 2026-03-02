"""Example data helpers for pipeline integration testing."""

from .generate import create_synthetic_dataset
from .readers import read_bin_csv_pair

__all__ = ["create_synthetic_dataset", "read_bin_csv_pair"]
