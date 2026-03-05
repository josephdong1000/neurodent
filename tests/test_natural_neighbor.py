"""
Tests for the Natural_Neighbor algorithm (neurodent.core.utils.Natural_Neighbor).

Covers: read/load, initialization, KDTree-based neighbor finding,
algorithm convergence on clusters/identical/two-point data, and CSV I/O edge cases.
"""

import csv

import numpy as np
import pytest
from scipy.spatial import KDTree

from neurodent.core.utils import Natural_Neighbor


class TestNaturalNeighbor:
    """Tests for the Natural_Neighbor algorithm."""

    def test_read_and_asserts(self):
        nn = Natural_Neighbor()
        data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        nn.read(data)
        assert np.array_equal(nn.data, data)
        nn.asserts()
        assert len(nn.knn) == 4
        assert all(nn.nan_num[i] == 0 for i in range(4))

    def test_count_all_zero(self):
        nn = Natural_Neighbor()
        nn.data = np.array([[0], [1], [2]])
        nn.asserts()
        assert nn.count() == 3  # all have zero natural neighbors initially

    def test_algorithm_small_cluster(self):
        np.random.seed(42)
        data = np.vstack(
            [np.random.randn(10, 2) + [0, 0], np.random.randn(10, 2) + [5, 5]]
        )
        nn = Natural_Neighbor()
        nn.read(data)
        r = nn.algorithm()
        assert isinstance(r, int)
        assert r >= 1

    def test_algorithm_identical_points(self):
        """All points identical – edge case for KDTree."""
        data = np.ones((5, 2))
        nn = Natural_Neighbor()
        nn.read(data)
        r = nn.algorithm()
        assert r >= 1

    def test_algorithm_two_points(self):
        """Minimal dataset of two points."""
        data = np.array([[0.0, 0.0], [1.0, 1.0]])
        nn = Natural_Neighbor()
        nn.read(data)
        r = nn.algorithm()
        assert r >= 1

    def test_findKNN(self):
        nn = Natural_Neighbor()
        data = np.array([[0, 0], [1, 0], [2, 0], [10, 0]])
        nn.read(data)

        tree = KDTree(data)
        neighbours = nn.findKNN(data[0], 2, tree)
        assert len(neighbours) == 2
        assert 1 in neighbours

    def test_load_csv(self, tmp_path):
        """Test loading from CSV file."""
        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([1.0, 2.0, "classA"])
            writer.writerow([3.0, 4.0, "classB"])
            writer.writerow([5.0, 6.0, "classA"])
        nn = Natural_Neighbor()
        nn.load(str(csv_path))
        assert nn.data.shape == (3, 2)
        assert nn.target == ["classA", "classB", "classA"]

    def test_load_csv_empty_file(self, tmp_path):
        """Empty CSV should yield empty data."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        nn = Natural_Neighbor()
        nn.load(str(csv_path))
        assert len(nn.data) == 0

    def test_load_csv_invalid_numeric(self, tmp_path):
        """Non-numeric attribute values should raise ValueError."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("abc,def,classA\n")
        nn = Natural_Neighbor()
        with pytest.raises(ValueError):
            nn.load(str(csv_path))
