"""LOF-based bad-channel detection.

Mixin for :class:`~neurodent.loading.long_recording_organizer.LongRecordingOrganizer`.
"""

import logging

import numpy as np

from sklearn.neighbors import LocalOutlierFactor

from neurodent.core.utils import (
    Natural_Neighbor,
    chunked_channel_distance_matrix,
)


class LroQualityMixin:
    """Mixin: see module docstring."""

    def compute_bad_channels(
        self,
        lof_threshold: float = None,
        force_recompute: bool = False,
        lof_chunk_duration_s: float = 60,
    ):
        """Compute bad channels using LOF analysis with unified score storage.

        Args:
            lof_threshold (float, optional): Threshold for determining bad channels from LOF scores.
                                           If None, only computes/loads scores without setting bad_channel_names.
            force_recompute (bool): Whether to recompute LOF scores even if they exist.
            lof_chunk_duration_s (float): Duration in seconds of each chunk used
                for the pairwise-distance computation in LOF.  Defaults to 60.
        """
        # Check if LOF scores already exist and are current
        if (
            not force_recompute
            and hasattr(self, "lof_scores")
            and self.lof_scores is not None
        ):
            logging.info("Using existing LOF scores")
        else:
            # Compute new LOF scores. _compute_lof_scores contextualises its own failures, so no
            # wrapping handler here.
            scores = self._compute_lof_scores(
                lof_chunk_duration_s=lof_chunk_duration_s,
            )
            self.lof_scores = scores
            logging.info(f"Computed LOF scores for {len(scores)} channels")

        # Apply threshold if provided
        if lof_threshold is not None:
            self.apply_lof_threshold(lof_threshold)

    def _compute_lof_scores(self, lof_chunk_duration_s: float = 60) -> np.ndarray:
        """Compute raw LOF scores for all channels.

        Pairwise Euclidean distances between channels are computed in
        chunks so that the full recording never needs to be held in
        memory at once.  Both the Natural-Neighbor *k*-selection and the
        LOF fit operate on the precomputed distance matrix.

        Args:
            lof_chunk_duration_s: Duration in seconds of each chunk used
                for the pairwise-distance computation.  Defaults to 60.

        Returns:
            np.ndarray: LOF scores for each channel.
        """
        # Input validation before the compute-guard, so it propagates as ValueError rather than
        # being wrapped as a computation failure.
        if lof_chunk_duration_s <= 0:
            raise ValueError(
                f"lof_chunk_duration_s must be positive, got {lof_chunk_duration_s}."
            )

        try:
            rec = self.LongRecording
            n_channels = rec.get_num_channels()
            n_samples = rec.get_total_samples()
            fs = rec.get_sampling_frequency()

            logging.debug(f"Computing LOF scores for {rec.__str__()}")
            logging.debug(
                f"Recording: {n_channels} channels, {n_samples} samples, {fs} Hz"
            )

            # --- Chunked pairwise-distance computation ---
            chunk_samples_raw = lof_chunk_duration_s * fs
            chunk_samples = max(1, int(round(chunk_samples_raw)))
            distance_matrix = chunked_channel_distance_matrix(
                get_traces_fn=lambda s, e: rec.get_traces(
                    start_frame=s, end_frame=e, return_scaled=True
                ),
                n_channels=n_channels,
                n_samples=n_samples,
                chunk_samples=chunk_samples,
            )
            logging.debug(f"Distance matrix shape: {distance_matrix.shape}")

            # --- Optimal neighbour count via Natural Neighbor ---
            nn = Natural_Neighbor()
            nn.read_distance_matrix(distance_matrix)
            n_neighbors = nn.algorithm()
            logging.info(f"Computed n_neighbors for LOF computation: {n_neighbors}")
            del nn

            # --- LOF on precomputed distances ---
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric="precomputed")
            logging.debug("Computing outlier scores")
            lof.fit(distance_matrix)
            scores = lof.negative_outlier_factor_ * -1
            logging.info(f"LOF computation successful: {len(scores)} channels")
            logging.debug(f"LOF scores: {scores}")

            return scores

        except Exception as e:
            raise RuntimeError(
                f"Failed to compute LOF scores "
                f"(channels={getattr(self, 'channel_names', 'unknown')})"
            ) from e

    def apply_lof_threshold(self, lof_threshold: float):
        """Apply threshold to existing LOF scores to determine bad channels.

        Args:
            lof_threshold (float): Threshold for determining bad channels.
        """
        if not hasattr(self, "lof_scores") or self.lof_scores is None:
            raise ValueError(
                "LOF scores not available. Run compute_bad_channels() first."
            )

        is_inlier = self.lof_scores < lof_threshold
        self.bad_channel_names = [
            self.channel_names[i] for i in np.where(~is_inlier)[0]
        ]
        logging.info(
            f"Applied threshold {lof_threshold}: bad_channel_names = {self.bad_channel_names}"
        )

    def get_lof_scores(self) -> dict:
        """Get LOF scores with channel names.

        Returns:
            dict: Dictionary mapping channel names to LOF scores.
        """
        if not hasattr(self, "lof_scores") or self.lof_scores is None:
            raise ValueError(
                "LOF scores not available. Run compute_bad_channels() first."
            )

        return dict(zip(self.channel_names, self.lof_scores))
