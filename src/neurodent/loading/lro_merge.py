"""In-memory split/merge of recordings and merge-compatibility checks.

Mixin for :class:`~neurodent.loading.long_recording_organizer.LongRecordingOrganizer`.
"""

import copy
import logging
import warnings
from datetime import timedelta

try:
    import spikeinterface.core as si
except Exception:  # pragma: no cover - optional at import time
    si = None

from neurodent.constants import RECORDING_TIME_TOLERANCE_S
from neurodent.core.utils import resolve_channels


def file_intervals(lro) -> list:
    """Each live file's ``(start, end)`` in a recording, or an empty list when timing is unknown.

    A file is live when it has an end datetime and a positive duration. 0-sample files carry
    a 0 duration and are left out, as ``_update_metadata_after_merge`` leaves them out.

    Args:
        lro: A LongRecordingOrganizer, or anything with ``file_end_datetimes`` and
            ``file_durations`` lists. Objects without them, or with something other than a
            list or tuple there (such as a test double), give an empty list.

    Returns:
        list[tuple[datetime, datetime]]: One interval per live file, in the recording's order.
    """
    ends = getattr(lro, "file_end_datetimes", None)
    durations = getattr(lro, "file_durations", None)
    if not isinstance(ends, (list, tuple)) or not isinstance(durations, (list, tuple)):
        return []
    return [
        (end - timedelta(seconds=float(duration)), end)
        for end, duration in zip(ends, durations)
        if end is not None and duration is not None and duration > 0
    ]


def _is_si_recording(rec) -> bool:
    base = getattr(si, "BaseRecording", None)
    return isinstance(base, type) and isinstance(rec, base)


def _repeated(values) -> list:
    return sorted({v for v in values if values.count(v) > 1})


def match_channels(reference_names, names, reference_label="the reference recording", label="the other recording"):
    """For each channel of a reference recording, the position of the same channel in another.

    Channels are matched by exact name when both recordings use the same names. That keeps
    a recording of several animals unambiguous, where one abbreviation stands for one
    electrode per animal. Otherwise they are matched by abbreviation, so 'L Barrel' and
    'L Barrel Ctx' are one channel, which needs each abbreviation to appear once on each
    side. Position plays no part, so two recordings that list the same electrodes in a
    different order still match.

    Args:
        reference_names: The reference recording's channel names.
        names: The other recording's channel names.
        reference_label: Names the reference recording in error messages.
        label: Names the other recording in error messages.

    Returns:
        list[int]: ``order`` such that ``names[order[k]]`` is the channel ``reference_names[k]``.

    Raises:
        ValueError: If either list is empty or repeats a name, if the two do not hold the same
            channels, or if they are spelled differently and an abbreviation repeats.
    """
    reference_names, names = list(reference_names or []), list(names or [])
    for listed, who in ((reference_names, reference_label), (names, label)):
        if not listed:
            raise ValueError(f"{who} has no channel names, so its channels cannot be matched.")
        if _repeated(listed):
            raise ValueError(
                f"{who} lists channel names {_repeated(listed)} more than once, so they cannot "
                "identify channels."
            )
    if set(names) == set(reference_names):
        return [names.index(n) for n in reference_names]

    reference_abbrevs, abbrevs = resolve_channels(reference_names), resolve_channels(names)
    if set(abbrevs) != set(reference_abbrevs):
        raise ValueError(
            f"Channel names mismatch: {reference_label} has {reference_names} (abbrevs: "
            f"{reference_abbrevs}), {label} has {names} (abbrevs: {abbrevs})"
        )
    for listed, who in ((reference_abbrevs, reference_label), (abbrevs, label)):
        if _repeated(listed):
            raise ValueError(
                f"Channel names mismatch: the two recordings spell their channels differently "
                f"({reference_names} and {names}), and in {who} the abbreviations "
                f"{_repeated(listed)} stand for more than one channel, so they cannot be matched."
            )
    return [abbrevs.index(a) for a in reference_abbrevs]


def conform_channels(lro, reference_names, reference_label="the reference recording"):
    """Reorder and respell a recording's channels to match a reference, data included.

    Channels are matched by :func:`match_channels`. The recording is put in the reference's
    order, and its channel ids, its ``channel_names``, its metadata, its bad channels and any
    ``channel_name`` property all take the reference's spelling, so whatever is read from the
    recording later agrees with what the analysis labels it. Before, a respelling changed
    ``channel_names`` but not the recording ids, which spike results are named from, and a
    different order was only logged, so that recording's columns carried other channels'
    names.

    Args:
        lro: The LongRecordingOrganizer to change in place.
        reference_names: The channel names, in order, it should end up with.
        reference_label: Names the reference in messages.

    Raises:
        ValueError: As :func:`match_channels`.
    """
    reference_names = list(reference_names or [])
    current = list(lro.channel_names or [])
    order = match_channels(reference_names, current, reference_label, lro.display_name)
    rec = getattr(lro, "LongRecording", None)

    if order != list(range(len(order))):
        logging.warning(
            f"{lro.display_name} lists its channels in a different order ({current}); "
            f"reordered to match {reference_label}: {reference_names}"
        )
        if rec is not None:
            ids = list(rec.get_channel_ids())
            rec = rec.select_channels(channel_ids=[ids[j] for j in order])

    old_names = [current[j] for j in order]
    ids_differ = _is_si_recording(rec) and [str(i) for i in rec.get_channel_ids()] != reference_names
    if rec is not None and (old_names != reference_names or ids_differ):
        if old_names != reference_names:
            logging.info(f"Renaming channels {old_names} -> {reference_names} to match {reference_label}")
        rec = rec.rename_channels(new_channel_ids=reference_names)
        if _is_si_recording(rec) and "channel_name" in rec.get_property_keys():
            rec.set_property("channel_name", reference_names)

    if rec is not None:
        lro.LongRecording = rec
    lro.channel_names = reference_names
    meta = getattr(lro, "meta", None)
    if meta is not None and hasattr(meta, "channel_names"):
        meta.channel_names = list(reference_names)
    respelled = dict(zip(old_names, reference_names))
    bad = getattr(lro, "bad_channel_names", None)
    if isinstance(bad, list) and bad:
        lro.bad_channel_names = [respelled.get(name, name) for name in bad]


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
                truncate=self.n_truncate if self.truncate else False,
            )

            # Inherit file-level timestamps and durations as copies. Merging appends to these
            # lists in place, so children sharing the parent's list objects would each see
            # the files merged into the others.
            if hasattr(self, "file_end_datetimes"):
                child_lro.file_end_datetimes = list(self.file_end_datetimes)

            # Inherit parent durations to ensure consistency with timestamps
            if hasattr(self, "file_durations") and self.file_durations:
                child_lro.file_durations = list(self.file_durations)
                child_lro.cumulative_file_durations = list(self.cumulative_file_durations)

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
            ValueError: If LROs are incompatible (different channels, sampling rates, etc.),
                or if ``other_lro`` starts before this one ends by more than
                ``RECORDING_TIME_TOLERANCE_S``. A larger gap is logged, not raised.
            ImportError: If SpikeInterface is not available
        """
        if si is None:
            raise ImportError("SpikeInterface is required for LRO merging")

        # Validate merge compatibility
        self._validate_merge_compatibility(other_lro)
        self._validate_merge_timeline(other_lro)

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

        # Concatenation needs identical channel ids in identical order, so the other
        # recording is reordered and respelled to this one's channels, matched by identity.
        conform_channels(self, self.channel_names, self.display_name)
        conform_channels(other_lro, self.channel_names, self.display_name)

        self.LongRecording = si.concatenate_recordings(
            [self.LongRecording, other_lro.LongRecording]
        )

        # Update metadata after merge
        self._update_metadata_after_merge(other_lro)

        logging.info("Successfully merged LRO recordings")

    def merge_many(self, other_lros):
        """Merge several LROs into this one with a single flat concatenation.

        Equivalent to calling :meth:`merge` for each entry, but the recordings are
        concatenated once at the end instead of once per step, so the result is one
        level deep rather than N-1.

        Why that matters: ``si.concatenate_recordings`` does not flatten, and ``merge``
        passes its own output back in as the first operand, so folding N files nests N-1
        deep. Sample lookup then walks the nesting, making a full pass O(N^2) (measured
        2.75x slower at N=500), and past roughly 1000 files ``get_traces`` raises
        RecursionError against Python's default limit. Pickling fails far earlier, at
        about N=125, which breaks any recording sent to a dask worker. The fold also
        leans left, putting the OLDEST file deepest, and analysis reads from frame 0
        first, so the wall is hit immediately rather than partway through.

        Per-step validation, channel renaming, 0-sample handling and metadata
        accumulation are all preserved; only the concatenation is deferred. Each entry's
        start is checked against the end of everything merged before it.

        Args:
            other_lros (list[LongRecordingOrganizer]): LROs to merge, in temporal order.

        Raises:
            ValueError: If any LRO is incompatible with this one, or starts before the
                recordings merged ahead of it end, by more than ``RECORDING_TIME_TOLERANCE_S``.
                A larger gap is logged, not raised.
            ImportError: If SpikeInterface is not available.
        """
        if si is None:
            raise ImportError("SpikeInterface is required for LRO merging")

        if other_lros:
            conform_channels(self, self.channel_names, self.display_name)
        recordings = [self.LongRecording]
        for other_lro in other_lros:
            self._validate_merge_compatibility(other_lro)
            self._validate_merge_timeline(other_lro)

            # A 0-sample tail contributes no samples but its metadata still matters, so
            # keep it out of the concatenation and fold its metadata in, as merge() does.
            if other_lro.LongRecording.get_total_samples() == 0:
                logging.warning(
                    f"Skipping recording concatenation of {getattr(other_lro, 'item', 'unknown')}: "
                    "0 samples. Updating metadata only."
                )
                self._update_metadata_after_merge(other_lro)
                continue

            conform_channels(other_lro, self.channel_names, self.display_name)
            recordings.append(other_lro.LongRecording)
            self._update_metadata_after_merge(other_lro)

        if len(recordings) > 1:
            self.LongRecording = si.concatenate_recordings(recordings)
            logging.info(
                f"Concatenated {len(recordings)} recordings in one flat call (depth 1)"
            )

    def _validate_merge_compatibility(self, other_lro):
        """Validate that two LROs can be safely merged.

        Args:
            other_lro (LongRecordingOrganizer): The LRO to validate against this one

        Raises:
            ValueError: If LROs are incompatible
        """
        # Same channels, matched by identity rather than position: exact names when both
        # use the same ones, abbreviations when the spelling differs. A different order is
        # fine, since the merge reorders the other recording to match.
        match_channels(self.channel_names, other_lro.channel_names, self.display_name, other_lro.display_name)

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

    def _validate_merge_timeline(self, other_lro):
        """Raise if ``other_lro`` starts before this recording ends, beyond clock jitter.

        Merging concatenates samples, so what is appended has to come after what is already
        there. An overlap means a misplaced timestamp or the same data loaded twice, and both
        used to be concatenated without a word, after which every feature was computed on the
        doubled or misordered data. A gap is legitimate, since recordings get stopped and
        restarted, so it is logged instead. Recordings without file timing are not checked.

        Args:
            other_lro (LongRecordingOrganizer): The recording about to be appended.

        Raises:
            ValueError: If ``other_lro`` starts more than ``RECORDING_TIME_TOLERANCE_S``
                before this recording's last file ends.
        """
        mine, theirs = file_intervals(self), file_intervals(other_lro)
        if not mine or not theirs:
            return
        my_end = max(end for _, end in mine)
        their_start = min(start for start, _ in theirs)
        residual = (their_start - my_end).total_seconds()
        if residual < -RECORDING_TIME_TOLERANCE_S:
            raise ValueError(
                f"{other_lro.display_name} starts at {their_start}, {-residual:.1f} s before "
                f"the recording it would be appended to ({self.display_name}) ends at "
                f"{my_end}. Either a timestamp is wrong or the same data is being loaded "
                f"twice, for example by two sources or sessions that reach the same files. "
                f"Overlaps up to {RECORDING_TIME_TOLERANCE_S:g} s are treated as clock jitter."
            )
        if residual > RECORDING_TIME_TOLERANCE_S:
            logging.warning(
                f"{residual:.1f} s gap inside one merged recording: {self.display_name} ends at "
                f"{my_end} and {other_lro.display_name} starts at {their_start}."
            )

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
