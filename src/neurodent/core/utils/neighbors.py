"""Natural-neighbor interpolation and channel distance matrices."""

import csv
import math

from typing import Callable

import numpy as np
from sklearn.neighbors import KDTree


class Natural_Neighbor(object):
    """
    Natural Neighbor algorithm implementation for finding natural neighbors in a dataset.

    This class implements the Natural Neighbor algorithm which finds mutual neighbors
    in a dataset by iteratively expanding the neighborhood radius until convergence.
    """

    def __init__(self):
        """
        Initialize the Natural Neighbor algorithm.

        Attributes:
            nan_edges (dict): Graph of mutual neighbors
            nan_num (dict): Number of natural neighbors for each instance
            repeat (dict): Data structure that counts repetitions of the count method
            target (list): Set of classes
            data (list): Set of instances
            knn (dict): Structure that stores neighbors of each instance
        """
        self.nan_edges = {}  # Graph of mutual neighbors
        self.nan_num = {}  # Number of natural neighbors for each instance
        self.repeat = {}  # Data structure that counts repetitions of the count method
        self.target = []  # Set of classes
        self.data = []  # Set of instances
        self.knn = {}  # Structure that stores neighbors of each instance

    def load(self, filename):
        """
        Load dataset from a CSV file, separating attributes and classes.

        Args:
            filename (str): Path to the CSV file containing the dataset
        """
        aux = []
        with open(filename, "r") as dataset:
            data = list(csv.reader(dataset))
            for inst in data:
                inst_class = inst.pop(-1)
                self.target.append(inst_class)
                row = [float(x) for x in inst]
                aux.append(row)
        self.data = np.array(aux)

    def read(self, data: np.ndarray):
        """
        Load data directly from a numpy array.

        Args:
            data (np.ndarray): Input data array
        """
        self.data = data
        self._distance_matrix = None

    def read_distance_matrix(self, distance_matrix: np.ndarray):
        """
        Load a precomputed distance matrix for neighbor search.

        When a distance matrix is provided, :meth:`algorithm` uses
        argsort-based neighbor lookup instead of a KDTree, avoiding the
        need to hold the raw high-dimensional data in memory.

        Args:
            distance_matrix (np.ndarray): Symmetric (n, n) distance matrix.
        """
        self._distance_matrix = distance_matrix
        # Set data length so existing helpers (asserts, count, etc.) work
        self.data = np.empty((distance_matrix.shape[0], 0))

    def asserts(self):
        """
        Initialize data structures for the algorithm.

        Sets up the necessary data structures including:
        - nan_edges as an empty set
        - knn, nan_num, and repeat dictionaries for each instance
        """
        self.nan_edges = set()
        for j in range(len(self.data)):
            self.knn[j] = set()
            self.nan_num[j] = 0
            self.repeat[j] = 0

    def count(self):
        """
        Count the number of instances that have no natural neighbors.

        Returns:
            int: Number of instances with zero natural neighbors
        """
        nan_zeros = 0
        for x in self.nan_num:
            if self.nan_num[x] == 0:
                nan_zeros += 1
        return nan_zeros

    def findKNN(self, inst, r, tree):
        """
        Find the indices of the k nearest neighbors.

        Args:
            inst: Instance to find neighbors for
            r (int): Radius/parameter for neighbor search
            tree: KDTree object for efficient neighbor search

        Returns:
            np.ndarray: Array of neighbor indices (excluding the instance itself)
        """
        _, ind = tree.query([inst], r + 1)
        return np.delete(ind[0], 0)

    def _findKNN_precomputed(self, i, r):
        """
        Find the r nearest neighbors of point *i* using the precomputed
        distance matrix (argsort-based, exact).

        Args:
            i (int): Index of the query point.
            r (int): Number of neighbors to return.

        Returns:
            np.ndarray: Array of neighbor indices (excluding point *i*).
        """
        dists = self._distance_matrix[i]
        # argsort gives indices sorted by distance; index 0 is self (distance 0)
        sorted_idx = np.argsort(dists)
        # Skip self at position 0 (distance to self is always 0)
        return sorted_idx[1 : r + 1]

    def algorithm(self):
        """
        Execute the Natural Neighbor algorithm.

        The algorithm iteratively expands the neighborhood radius until convergence,
        finding mutual neighbors between instances.

        When a precomputed distance matrix is available (see
        :meth:`read_distance_matrix`), neighbor lookup is performed via
        argsort instead of a KDTree, which avoids holding the raw
        high-dimensional data in memory.

        Returns:
            int: The final radius value when convergence is reached
        """
        use_precomputed = (
            hasattr(self, "_distance_matrix")
            and self._distance_matrix is not None
        )

        if not use_precomputed:
            # Initialize KDTree for efficient neighbor search
            tree = KDTree(self.data)
        else:
            tree = None  # not used in precomputed path

        self.asserts()
        flag = 0
        r = 1

        n_points = len(self.data)
        max_r = n_points - 1  # r + 1 must not exceed n_points
        while flag == 0:
            for i in range(n_points):
                if use_precomputed:
                    knn = self._findKNN_precomputed(i, r)
                else:
                    knn = self.findKNN(self.data[i], r, tree)
                n = knn[-1]
                self.knn[i].add(n)
                if i in self.knn[n] and (i, n) not in self.nan_edges:
                    self.nan_edges.add((i, n))
                    self.nan_edges.add((n, i))
                    self.nan_num[i] += 1
                    self.nan_num[n] += 1

            cnt = self.count()
            rep = self.repeat[cnt]
            self.repeat[cnt] += 1
            if cnt == 0 or rep >= math.sqrt(r - rep) or r >= max_r:
                flag = 1
            else:
                r += 1
        return r


def chunked_channel_distance_matrix(
    get_traces_fn: Callable[[int, int], np.ndarray],
    n_channels: int,
    n_samples: int,
    chunk_samples: int,
) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between channels in chunks.

    Instead of loading the full ``(n_samples, n_channels)`` trace matrix at
    once, this function reads ``chunk_samples`` frames at a time and
    accumulates squared distances using the identity

        ||c_i - c_j||^2 = ||c_i||^2 + ||c_j||^2 - 2 * c_i · c_j

    so that peak RAM is proportional to ``chunk_samples * n_channels``
    rather than ``n_samples * n_channels``.

    Args:
        get_traces_fn: ``fn(start_frame, end_frame) -> np.ndarray`` with
            shape ``(frames, n_channels)``.  Typically
            ``recording.get_traces(start_frame=..., end_frame=...,
            return_scaled=True)``.
        n_channels: Number of channels.
        n_samples: Total number of samples in the recording.
        chunk_samples: Number of samples to read per chunk.

    Returns:
        np.ndarray: Symmetric ``(n_channels, n_channels)`` Euclidean
        distance matrix.
    """
    if chunk_samples < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples}")

    sq_dist_accum = np.zeros((n_channels, n_channels), dtype=np.float64)

    for start in range(0, n_samples, chunk_samples):
        end = min(start + chunk_samples, n_samples)
        chunk = get_traces_fn(start, end)  # (chunk_len, n_channels)
        chunk_t = chunk.T  # (n_channels, chunk_len)

        # ||c_i - c_j||^2 = ||c_i||^2 + ||c_j||^2 - 2 * c_i · c_j
        norms_sq = np.sum(chunk_t ** 2, axis=1)  # (n_channels,)
        gram = chunk_t @ chunk_t.T  # (n_channels, n_channels)
        sq_dist_accum += norms_sq[:, None] + norms_sq[None, :] - 2 * gram
        del chunk, chunk_t, norms_sq, gram

    # Clamp tiny negatives from floating-point arithmetic
    np.maximum(sq_dist_accum, 0, out=sq_dist_accum)
    distance_matrix = np.sqrt(sq_dist_accum)
    del sq_dist_accum
    return distance_matrix
