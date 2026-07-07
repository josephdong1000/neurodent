"""In-memory split/merge of recordings and merge-compatibility checks.

Mixin for :class:`~neurodent.loading.long_recording_organizer.LongRecordingOrganizer`.
"""

import copy
import logging
import warnings

try:
    import spikeinterface.core as si
except Exception:  # pragma: no cover - optional at import time
    si = None

from neurodent.core.utils import resolve_channels


class LroMergeMixin:
    """Mixin: see module docstring."""

    def split(
        self,
        groups: dict[str, list[str]],
    ) -> dict[str, "LongRecordingOrganizer"]:
        """
        Split the current recording into multiple in-memory LongRecordingOrganizer objects.

        This creates lightweight LRO wrappers around channel-sliced views of the
        recording. No disk I/O is performed. Use `save_recording()` on individual
        results to save them to disk if needed.

        Args:
            groups (dict[str, list[str]]): Dictionary mapping group names (e.g., 'AnimalA')
                                           to lists of channel names.

        Returns:
            dict[str, LongRecordingOrganizer]: Dictionary mapping group names to new LRO instances.

        Raises:
            ValueError: If requested channels are not found in the recording.
            ImportError: If SpikeInterface is not available.

        Example:
            >>> lro = LongRecordingOrganizer("/path/to/data", mode="bin")
            >>> splits = lro.split({"AnimalA": ["Ch1", "Ch2"], "AnimalB": ["Ch3", "Ch4"]})
            >>> splits["AnimalA"].save_recording("/output/AnimalA", format="zarr")
        """
        if si is None:
            raise ImportError("SpikeInterface is required for split()")

        if not hasattr(self, "LongRecording") or self.LongRecording is None:
            raise ValueError("No recording loaded to split")

        # Build channel name to ID mapping
        if not self.channel_names:
            lro_names = [str(x) for x in self.LongRecording.get_channel_ids()]
        else:
            lro_names = self.channel_names

        rec_channel_ids = self.LongRecording.get_channel_ids()
        name_to_id = {}
        if len(lro_names) == len(rec_channel_ids):
            for name, ch_id in zip(lro_names, rec_channel_ids):
                name_to_id[name] = ch_id
        else:
            logging.warning(
                "LRO channel_names length mismatch. Falling back to str(id)."
            )
            for ch_id in rec_channel_ids:
                name_to_id[str(ch_id)] = ch_id

        # Track which channels are used for validation warning
        all_requested_channels = set()
        for channel_list in groups.values():
            all_requested_channels.update(channel_list)

        unused_channels = [ch for ch in lro_names if ch not in all_requested_channels]
        if unused_channels:
            logging.warning(
                f"{len(unused_channels)} channels not included in any group: "
                f"{unused_channels[:5]}{'...' if len(unused_channels) > 5 else ''}"
            )

        lros = {}
        for group_name, channel_subset in groups.items():
            logging.info(
                f"Splitting group '{group_name}' with {len(channel_subset)} channels"
            )

            # Map names to IDs
            target_ids = []
            valid_names = []
            missing = []

            for name in channel_subset:
                if name in name_to_id:
                    target_ids.append(name_to_id[name])
                    valid_names.append(name)
                else:
                    missing.append(name)

            if missing:
                raise ValueError(
                    f"Channels not found in recording for group '{group_name}': {missing}"
                )

            # Slice (in-memory view, no copy)
            sub_rec = self.LongRecording.select_channels(channel_ids=target_ids)

            # Rename channels to string names for consistency
            if hasattr(sub_rec, "rename_channels"):
                sub_rec = sub_rec.rename_channels(new_channel_ids=valid_names)

            # Create in-memory LRO wrapper with barebones instantiation
            child_lro = type(self)(
                item=None,
                recording=sub_rec,
                manual_datetimes=self.manual_datetimes,
                datetimes_are_start=self.datetimes_are_start,
                n_jobs=self.n_jobs,
                truncate=self.n_truncate if self.truncate else False,
            )

            # Inherit file-level timestamps and durations (post-instantiation assignment)
            if hasattr(self, "file_end_datetimes"):
                child_lro.file_end_datetimes = self.file_end_datetimes

            # Inherit parent durations to ensure consistency with timestamps
            if hasattr(self, "file_durations") and self.file_durations:
                child_lro.file_durations = self.file_durations
                child_lro.cumulative_file_durations = self.cumulative_file_durations

            # Inherit bad channels that are present in this split
            if self.bad_channel_names:
                child_lro.bad_channel_names = [
                    ch for ch in self.bad_channel_names if ch in valid_names
                ]

            # Inherit complete metadata (preserving units, scaling, etc.)
            if self.meta:
                child_lro.meta = copy.deepcopy(self.meta)
                child_lro.meta.n_channels = len(valid_names)
                child_lro.meta.channel_names = valid_names

            # Inherit labels (as a copy)
            if hasattr(self, "labels") and self.labels:
                child_lro.labels = dict(self.labels)

            lros[group_name] = child_lro

        return lros

    def merge(self, other_lro):
        """Merge another LRO into this one using si.concatenate_recordings.

        This creates a new concatenated recording from this LRO and the other LRO.
        The other LRO should represent a later time period to maintain temporal order.

        Args:
            other_lro (LongRecordingOrganizer): The LRO to merge into this one

        Raises:
            ValueError: If LROs are incompatible (different channels, sampling rates, etc.)
            ImportError: If SpikeInterface is not available
        """
        if si is None:
            raise ImportError("SpikeInterface is required for LRO merging")

        # Validate merge compatibility
        self._validate_merge_compatibility(other_lro)

        # Skip recording concatenation if other_lro has 0 samples (e.g. empty
        # tail file), but still update metadata so dt_end etc. stay correct.
        # _update_metadata_after_merge filters out 0-duration entries from
        # file_end_datetimes/file_durations to avoid corrupting TimestampMapper.
        if other_lro.LongRecording.get_total_samples() == 0:
            logging.warning(
                f"Skipping recording concatenation of {getattr(other_lro, 'item', 'unknown')}: "
                "0 samples. Updating metadata only."
            )
            self._update_metadata_after_merge(other_lro)
            return

        # Concatenate recordings using SpikeInterface
        logging.info(
            f"Merging LRO {getattr(other_lro, 'item', 'unknown')} into {getattr(self, 'item', 'unknown')}"
        )

        # If channel names differ but abbreviations matched (validated above),
        # rename other recording's channels to match self's for SI concatenation.
        other_rec = other_lro.LongRecording
        if self.channel_names != other_lro.channel_names:
            logging.info(
                f"Renaming channels {other_lro.channel_names} -> {self.channel_names} "
                "for merge compatibility"
            )
            other_rec = other_rec.rename_channels(
                new_channel_ids=self.channel_names
            )
            other_lro.channel_names = list(self.channel_names)

        self.LongRecording = si.concatenate_recordings(
            [self.LongRecording, other_rec]
        )

        # Update metadata after merge
        self._update_metadata_after_merge(other_lro)

        logging.info("Successfully merged LRO recordings")

    def _validate_merge_compatibility(self, other_lro):
        """Validate that two LROs can be safely merged.

        Args:
            other_lro (LongRecordingOrganizer): The LRO to validate against this one

        Raises:
            ValueError: If LROs are incompatible
        """
        # Check channel names — compare by abbreviation to tolerate naming
        # variants (e.g. "L Barrel" vs "L Barrel Ctx" both → "LBar").
        # Unparseable names pass through as-is for exact comparison.
        self_abbrevs = resolve_channels(self.channel_names)
        other_abbrevs = resolve_channels(other_lro.channel_names)
        if self_abbrevs != other_abbrevs:
            raise ValueError(
                f"Channel names mismatch: this LRO has {self.channel_names} "
                f"(abbrevs: {self_abbrevs}), other LRO has {other_lro.channel_names} "
                f"(abbrevs: {other_abbrevs})"
            )

        # Check sampling rates
        if hasattr(self.meta, "f_s") and hasattr(other_lro.meta, "f_s"):
            if self.meta.f_s != other_lro.meta.f_s:
                raise ValueError(
                    f"Sampling rate mismatch: this LRO has {self.meta.f_s} Hz, other LRO has {other_lro.meta.f_s} Hz"
                )

        # Check channel counts
        if hasattr(self.meta, "n_channels") and hasattr(other_lro.meta, "n_channels"):
            if self.meta.n_channels != other_lro.meta.n_channels:
                raise ValueError(
                    f"Channel count mismatch: "
                    f"this LRO has {self.meta.n_channels} channels, "
                    f"other LRO has {other_lro.meta.n_channels} channels"
                )

        # Check that both have valid recordings
        if not hasattr(self, "LongRecording") or self.LongRecording is None:
            raise ValueError("This LRO does not have a valid LongRecording")
        if not hasattr(other_lro, "LongRecording") or other_lro.LongRecording is None:
            raise ValueError("Other LRO does not have a valid LongRecording")

    def _update_metadata_after_merge(self, other_lro):
        """Update this LRO's metadata after merging with another LRO.

        Args:
            other_lro (LongRecordingOrganizer): The LRO that was merged into this one
        """
        if hasattr(other_lro.meta, "dt_end") and hasattr(self.meta, "dt_end"):
            self.meta.dt_end = other_lro.meta.dt_end

        # Merge file timestamps and durations
        has_dates = (
            hasattr(self, "file_end_datetimes")
            and self.file_end_datetimes
            and hasattr(other_lro, "file_end_datetimes")
            and other_lro.file_end_datetimes
        )
        has_durs = (
            hasattr(self, "file_durations")
            and self.file_durations
            and hasattr(other_lro, "file_durations")
            and other_lro.file_durations
        )

        if has_durs:
            if has_dates:
                # Filter out 0-duration entries (from 0-sample recordings) to
                # avoid corrupting TimestampMapper with degenerate mappings.
                for dt, dur in zip(
                    other_lro.file_end_datetimes, other_lro.file_durations
                ):
                    if dur > 0:
                        self.file_end_datetimes.append(dt)
                        self.file_durations.append(dur)
            else:
                # If we are merging durations, we must be able to merge timestamps
                # OR we must drop timestamps entirely to avoid mismatch (destructive).
                # better to raise error and let user fix input data.
                raise ValueError(
                    f"Merge failed: 'other_lro' ({other_lro.display_name}) "
                    "has durations but missing 'file_end_datetimes'. Cannot merge safely without corrupting metadata."
                )

        # Note: Channel names, sampling rate, etc. should already be validated as identical

        # Merge labels
        if hasattr(other_lro, "labels") and other_lro.labels:
            for key, value in other_lro.labels.items():
                if key in self.labels and self.labels[key] != value:
                    warnings.warn(
                        f"Label conflict during merge for key '{key}': "
                        f"'{self.labels[key]}' vs '{value}'. Using value from other LRO.",
                        UserWarning,
                    )
                self.labels[key] = value
