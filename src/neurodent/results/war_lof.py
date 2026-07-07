"""LOF outlier utilities for :class:`WindowAnalysisResult` (issue #134)."""

from __future__ import annotations

import logging

import numpy as np

from neurodent.core.utils import resolve_channel

class WARLofMixin:
    """Mixin: see module docstring."""

    def get_bad_channels_by_lof_threshold(self, lof_threshold: float) -> dict:
        """Apply LOF threshold directly to stored scores to get bad channels.

        Args:
            lof_threshold (float): Threshold for determining bad channels.

        Returns:
            dict: Dictionary mapping animal days to lists of bad channel names.
        """
        if not hasattr(self, "lof_scores_dict") or not self.lof_scores_dict:
            raise ValueError(
                "LOF scores not available in this WAR. Compute LOF scores first."
            )

        bad_channels_dict = {}
        for animalday, lof_data in self.lof_scores_dict.items():
            if "lof_scores" in lof_data and "channel_names" in lof_data:
                scores = np.array(lof_data["lof_scores"])
                channel_names = lof_data["channel_names"]

                is_inlier = scores < lof_threshold
                bad_channels = [channel_names[i] for i in np.where(~is_inlier)[0]]
                bad_channels_dict[animalday] = bad_channels
            else:
                raise ValueError(f"LOF scores not available for {animalday}")

        return bad_channels_dict

    def get_lof_scores(self) -> dict:
        """Get LOF scores from this WAR.

        Returns:
            dict: Dictionary mapping animal days to LOF score dictionaries.
        """
        if not hasattr(self, "lof_scores_dict") or not self.lof_scores_dict:
            raise ValueError(
                "LOF scores not available in this WAR. Compute LOF scores first."
            )

        result = {}
        for animalday, lof_data in self.lof_scores_dict.items():
            if "lof_scores" in lof_data and "channel_names" in lof_data:
                scores = lof_data["lof_scores"]
                channel_names = lof_data["channel_names"]
                result[animalday] = dict(zip(channel_names, scores))
            else:
                raise ValueError(f"LOF scores not available for {animalday}")

        return result

    def evaluate_lof_threshold_binary(
        self,
        ground_truth_bad_channels: dict = None,
        threshold: float = None,
        evaluation_channels: list[str] = None,
    ) -> tuple:
        """Evaluate single threshold against ground truth for binary classification.

        Args:
            ground_truth_bad_channels: Dict mapping animal-day to bad channel sets.
                                     If None, uses self.bad_channels_dict as ground truth.
            threshold: LOF threshold to test
            evaluation_channels: Subset of channels to include in evaluation. If none, uses all channels.

        Returns:
            tuple: (y_true_list, y_pred_list) for sklearn.metrics.f1_score
                   Each element represents one channel from one animal-day
        """
        if not hasattr(self, "lof_scores_dict") or not self.lof_scores_dict:
            raise ValueError(
                "LOF scores not available in this WAR. Run compute_bad_channels() first."
            )

        if threshold is None:
            raise ValueError("threshold parameter is required")

        # Use self.bad_channels_dict as default ground truth
        if ground_truth_bad_channels is None:
            if hasattr(self, "bad_channels_dict") and self.bad_channels_dict:
                ground_truth_bad_channels = {}

                # Filter bad_channels_dict to only include keys that exist in lof_scores_dict
                lof_keys = set(self.lof_scores_dict.keys())
                bad_channels_keys = set(self.bad_channels_dict.keys())

                missing_keys = bad_channels_keys - lof_keys
                if missing_keys:
                    raise ValueError(
                        f"bad_channels_dict contains keys not found in lof_scores_dict: {missing_keys}. "
                        f"Available LOF keys: {sorted(lof_keys)}"
                    )

                # Only use bad channel keys that have corresponding LOF data
                ground_truth_bad_channels = {
                    key: value
                    for key, value in self.bad_channels_dict.items()
                    if key in lof_keys
                }

                logging.info(
                    f"Using filtered bad_channels_dict as ground truth with {len(ground_truth_bad_channels)} animal-day sessions"
                )
            else:
                raise ValueError(
                    "No ground truth provided and self.bad_channels_dict is empty."
                )

        # Get all channels if no subset specified
        if evaluation_channels is None:
            evaluation_channels = self.channel_names

        y_true_list = []
        y_pred_list = []

        # Debug: Log what we're working with
        logging.debug(
            f"evaluate_lof_threshold_binary: evaluation_channels = {evaluation_channels}"
        )
        logging.debug(
            f"evaluate_lof_threshold_binary: ground_truth_bad_channels keys = {list(ground_truth_bad_channels.keys())}"
        )
        logging.debug(
            f"evaluate_lof_threshold_binary: lof_scores_dict keys = {list(self.lof_scores_dict.keys())}"
        )

        # Iterate through each animal-day and evaluate channels
        for animalday, lof_data in self.lof_scores_dict.items():
            if "lof_scores" not in lof_data or "channel_names" not in lof_data:
                raise ValueError(
                    f"Invalid LOF data for {animalday}: missing required fields 'lof_scores' or 'channel_names'"
                )

            scores = np.array(lof_data["lof_scores"])
            channel_names = lof_data["channel_names"]

            # Validate data integrity before processing
            # NOTE address this issue since this should not be happening in the first place
            # if len(scores) == 0:
            #     logging.warning(
            #         f"Skipping {animalday}: No LOF scores available. "
            #         f"This session will be excluded from LOF accuracy evaluation."
            #     )
            #     continue

            # if len(scores) != len(channel_names):
            #     logging.error(
            #         f"Skipping {animalday}: LOF scores ({len(scores)}) and "
            #         f"channels ({len(channel_names)}) length mismatch. "
            #         f"This indicates a data integrity issue - the animalday may have been "
            #         f"improperly mapped during LOF score collection."
            #     )
            #     continue

            # Get ground truth bad channels for this animal-day
            animalday_bad_channels = ground_truth_bad_channels.get(animalday, set())

            # Debug: Log details for this animal-day
            logging.debug(f"Processing {animalday}: channel_names = {channel_names}")
            logging.debug(
                f"Processing {animalday}: animalday_bad_channels = {animalday_bad_channels}"
            )
            logging.debug(f"Processing {animalday}: scores shape = {scores.shape}")

            # Evaluate each channel in the evaluation subset
            channels_processed = 0
            for i, channel in enumerate(channel_names):
                if (
                    channel in evaluation_channels
                    or resolve_channel(channel)
                    in evaluation_channels
                ):
                    channels_processed += 1

                    # Ground truth: 1 if channel is marked as bad, 0 otherwise
                    is_bad_channel = (
                        channel in animalday_bad_channels
                        or resolve_channel(channel)
                        in animalday_bad_channels
                    )

                    y_true = 1 if is_bad_channel else 0
                    # Prediction: 1 if LOF score > threshold, 0 otherwise
                    y_pred = 1 if scores[i] > threshold else 0

                    y_true_list.append(y_true)
                    y_pred_list.append(y_pred)

                    logging.debug(
                        f"Channel {channel}: y_true={y_true}, y_pred={y_pred} (score={scores[i]:.3f}, threshold={threshold})"
                    )

                    # Extra debugging for the alignment issue
                    if y_true == 1:
                        logging.info(
                            f"TRUE POSITIVE CANDIDATE: {channel} mapped to bad channel in: {animalday_bad_channels}"
                        )
                    if y_pred == 1:
                        logging.info(
                            f"LOF PREDICTION: {channel} has score {scores[i]:.3f} > threshold {threshold}"
                        )

            logging.debug(f"Processed {channels_processed} channels for {animalday}")

        return y_true_list, y_pred_list
