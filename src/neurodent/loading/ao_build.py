"""Build LongRecordingOrganizers per animalday, the from_lros factory, and split.

Mixin for :class:`~neurodent.loading.animal_organizer.AnimalOrganizer`.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal, Union

from tqdm import tqdm

from . import long_recording_organizer as _lro
from neurodent.core.utils import resolve_channels


class AoBuildMixin:
    """Mixin: see module docstring."""

    def _create_long_recordings(self, lro_kwargs: dict):
        """Create LongRecordingOrganizer instances for each unique animalday."""
        self.long_recordings: list[_lro.LongRecordingOrganizer] = []
        skipped_animaldays: list[str] = []
        for animalday, items in self._animalday_folder_groups.items():
            kwargs = lro_kwargs.copy()
            if getattr(self, "_processed_timestamps", None) is not None:
                # _processed_timestamps is keyed by full item path, not animalday
                if len(items) == 1:
                    item_key = self._get_item_key(items[0])
                    if item_key in self._processed_timestamps:
                        kwargs["manual_datetimes"] = self._processed_timestamps[item_key]
                        kwargs["datetimes_are_start"] = True  # _compute_global_timeline always returns start times
                        logging.debug(
                            f"Using processed timestamp for {item_key}: {kwargs['manual_datetimes']}"
                        )
                else:
                    # For multi-item animaldays, collect per-item timestamps as a list
                    item_timestamps = []
                    for item in items:
                        item_key = self._get_item_key(item)
                        if item_key in self._processed_timestamps:
                            item_timestamps.append(self._processed_timestamps[item_key])
                    if item_timestamps:
                        kwargs["manual_datetimes"] = item_timestamps
                        kwargs["datetimes_are_start"] = True  # _compute_global_timeline always returns start times
                        logging.debug(
                            f"Using processed timestamps for {animalday}: {item_timestamps}"
                        )

            if len(items) == 1:
                item_to_pass = items[0]
                kw = kwargs.copy()
                if self._is_item_file(item_to_pass) and isinstance(
                    item_to_pass, (list, tuple)
                ):
                    # LRO handles lists of files directly, but we pass input_type='files'? Wait, LRO handles it natively now
                    pass
                lro = _lro.LongRecordingOrganizer(item_to_pass, **kw)
            else:
                logging.info(
                    f"Creating individual LROs for {len(items)} items for {animalday}"
                )
                item_lro_pairs = []
                for item in items:
                    individual_kwargs = kwargs.copy()
                    # Remove session-level timestamp list; replaced per-item
                    # below.  Items missing from _processed_timestamps (e.g.,
                    # zero-byte files skipped from timeline) must not inherit
                    # the session list, which would cause a length mismatch.
                    individual_kwargs.pop("manual_datetimes", None)
                    individual_kwargs.pop("datetimes_are_start", None)
                    # Distribute per-item timestamp so each LRO gets its own
                    if getattr(self, "_processed_timestamps", None) is not None:
                        item_key = self._get_item_key(item)
                        if item_key in self._processed_timestamps:
                            individual_kwargs["manual_datetimes"] = (
                                self._processed_timestamps[item_key]
                            )
                            individual_kwargs["datetimes_are_start"] = True  # _compute_global_timeline always returns start times
                    individual_lro = _lro.LongRecordingOrganizer(
                        item, **individual_kwargs
                    )
                    item_lro_pairs.append((item, individual_lro))

                # Filter out 0-sample LROs (from failed/empty file pairs) before
                # merging. A 0-sample base LRO causes merge metadata failures, and
                # downstream _iter_valid_recordings() cannot recover from that.
                valid_pairs, skipped_names = self._filter_zero_sample_lros(
                    item_lro_pairs, self._get_item_name
                )
                if skipped_names:
                    logging.warning(
                        f"Skipping {len(skipped_names)} 0-sample LRO(s) for "
                        f"'{animalday}' before merge: {skipped_names}"
                    )
                if not valid_pairs:
                    logging.error(
                        f"Skipping animalday '{animalday}' entirely: all {len(item_lro_pairs)} "
                        f"file(s) produced 0-sample LROs. Each file may have been corrupt, "
                        f"empty, or failed during loading (check earlier warnings above for "
                        f"root causes per file). Skipped files: {skipped_names}"
                    )
                    skipped_animaldays.append(animalday)
                    continue
                item_lro_pairs = valid_pairs

                sorted_folder_lro_pairs = self._sort_lros_by_median_time(item_lro_pairs)

                logging.info("LRO merge order for overlapping animalday:")
                for i, (item, lro) in enumerate(sorted_folder_lro_pairs):
                    item_name = self._get_item_name(item)
                    try:
                        duration = (
                            lro.LongRecording.get_duration()
                            if hasattr(lro, "LongRecording") and lro.LongRecording
                            else 0
                        )
                        duration_str = f"{float(duration):.1f}s"
                    except (TypeError, ValueError):
                        duration_str = "mock"
                    logging.info(f"  {i + 1}. {item_name} (duration: {duration_str})")

                merged_lro = sorted_folder_lro_pairs[0][1]
                logging.info(
                    f"Base LRO: {self._get_item_name(sorted_folder_lro_pairs[0][0])}"
                )

                for i, (item, lro) in enumerate(sorted_folder_lro_pairs[1:], 1):
                    item_name = self._get_item_name(item)
                    logging.info(f"Merging LRO {i}: {item_name} into base LRO")
                    merged_lro.merge(lro)

                lro = merged_lro
                logging.info(
                    f"Successfully merged {len(sorted_folder_lro_pairs)} LROs for {animalday}"
                )

            self.long_recordings.append(lro)

        if skipped_animaldays:
            self.unique_animaldays = [
                ad for ad in self.unique_animaldays if ad not in skipped_animaldays
            ]
            self.animaldays = self.unique_animaldays

        if not self.long_recordings:
            raise RuntimeError(
                f"No recordings were loaded for this animal. "
                f"All {len(skipped_animaldays)} animalday(s) were skipped because every "
                f"file produced a 0-sample LRO. This usually indicates a misconfiguration "
                f"(wrong file pattern, wrong data root, or corrupt data). "
                f"Skipped animaldays: {skipped_animaldays}. "
                f"Check the warnings above for per-file root causes."
            )

        # It is possible for long_recordings to contain only 0-sample placeholder LROs.
        # In that case, _iter_valid_recordings() will yield nothing and downstream analysis
        # (e.g. concatenating results) will fail with a less informative error.
        # Guard against this by raising early if there are no valid (nonzero-sample) LROs.
        valid_long_recordings = list(self._iter_valid_recordings())
        if not valid_long_recordings:
            raise RuntimeError(
                "No valid (nonzero-sample) recordings were loaded for this animal. "
                "One or more LongRecordingOrganizer instances were created, but all of "
                "them contain 0 samples. This usually indicates a misconfiguration "
                "(wrong file pattern, wrong data root, or corrupt data). "
                "Check the warnings above for per-file root causes."
            )
        self._log_timeline_summary()

        if len(self.long_recordings) != len(self.unique_animaldays):
            error_msg = (
                f"Mismatch: Created {len(self.long_recordings)} LROs "
                f"but found {len(self.unique_animaldays)} unique animaldays. "
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def convert_colbins_to_rowbins(
        self, overwrite=False, multiprocess_mode: Literal["dask", "serial"] = "serial"
    ):
        for lrec in tqdm(
            self.long_recordings, desc="Converting column bins to row bins"
        ):
            lrec.convert_colbins_to_rowbins(
                overwrite=overwrite, multiprocess_mode=multiprocess_mode
            )

    def convert_rowbins_to_rec(
        self, multiprocess_mode: Literal["dask", "serial"] = "serial"
    ):
        for lrec in tqdm(self.long_recordings, desc="Converting row bins to recs"):
            lrec.convert_rowbins_to_rec(multiprocess_mode=multiprocess_mode)

    def cleanup_rec(self):
        for lrec in self.long_recordings:
            lrec.cleanup_rec()

    @staticmethod
    def _filter_zero_sample_lros(lro_pairs, get_name):
        """Remove 0-sample LROs from *lro_pairs* before a merge loop.

        A 0-sample LRO used as the **base** of a merge causes
        ``si.concatenate_recordings`` to fail or produce corrupt metadata.
        This helper removes such LROs up-front so every caller's merge loop
        starts from a valid base.

        This is intentionally separate from the ``merge()`` check in
        ``LongRecordingOrganizer``, which only guards against a 0-sample
        *other_lro* being merged in.  Together the two checks cover all cases:
        base=0-sample (this helper) and other_lro=0-sample (``merge()``).

        Args:
            lro_pairs: Iterable of ``(key, lro)`` pairs.  *key* is whatever
                the caller uses to name the LRO (item path, string tag, …).
            get_name: Callable ``(key) -> str`` used to produce a human-readable
                name for warning messages.

        Returns:
            ``(valid_pairs, skipped_names)`` where *valid_pairs* is a list of
            ``(key, lro)`` pairs with 0-sample entries removed and
            *skipped_names* is a list of names of the removed LROs.
        """
        valid_pairs = []
        skipped_names = []
        for key, lro in lro_pairs:
            try:
                if (
                    hasattr(lro, "LongRecording")
                    and lro.LongRecording is not None
                    and lro.LongRecording.get_total_samples() == 0
                ):
                    skipped_names.append(get_name(key))
                    continue
            except (TypeError, AttributeError):
                pass  # Non-SI or mock — keep it
            valid_pairs.append((key, lro))
        return valid_pairs, skipped_names

    @classmethod
    def from_lros(
        cls,
        lros: list[_lro.LongRecordingOrganizer],
        animal_id: str,
        genotype: str = "Unknown",
        sex: str = "Unknown",
    ) -> "AnimalOrganizer":
        """
        Create an AnimalOrganizer from an existing list of LongRecordingOrganizer objects.

        This factory method bypasses the normal folder discovery logic and creates
        an AnimalOrganizer directly from pre-existing LROs. If multiple LROs share
        the same date, they will be automatically merged into a single LRO per unique date,
        matching the behavior of the normal __init__ path.

        Args:
            lros (list[LongRecordingOrganizer]): List of LRO instances to wrap.
            animal_id (str): Animal identifier for this organizer.
            genotype (str, optional): Genotype string. Defaults to "Unknown".
            sex (str, optional): Sex string (e.g. "Male", "Female"). Defaults to "Unknown".

        Returns:
            AnimalOrganizer: A new AnimalOrganizer instance wrapping the provided LROs
                (with duplicates merged).

        Raises:
            ValueError: If lros is empty, channel names are inconsistent, or LROs
                with the same date cannot be merged due to incompatible metadata.

        Note:
            Multiple LROs with the same date will be automatically merged in temporal
            order (sorted by median timestamp). This ensures proper handling of
            multi-session recordings consolidated via generate_wars.py.

        Example:
            >>> # After splitting a multi-animal recording across multiple sessions
            >>> all_lros = []
            >>> for session_ao in session_aos:
            ...     splits = session_ao.split({"AnimalA": ["Ch0", "Ch1"]})
            ...     all_lros.append(splits["AnimalA"])
            >>> # from_lros automatically merges LROs with same date
            >>> child_ao = AnimalOrganizer.from_lros(all_lros, animal_id="AnimalA")
        """
        if not lros:
            raise ValueError("Cannot create AnimalOrganizer from empty LRO list")

        # Create instance without calling __init__
        ao = object.__new__(cls)

        # Core attributes
        ao.anim_id = animal_id
        ao.animal_id = animal_id
        ao.genotype = genotype
        ao.sex = sex

        # Step 1: Group LROs by date
        date_to_lros = {}  # dict[str, list[tuple[int, LRO]]]

        for i, lro in enumerate(lros):
            try:
                date_str = lro.get_date_string()
            except ValueError as e:
                raise ValueError(
                    f"Could not determine date for LRO at index {i} (item: {lro.display_name}). "
                    f"Ensure LRO has valid timestamps via metadata or manual_datetimes. Error: {e}"
                )

            if date_str not in date_to_lros:
                date_to_lros[date_str] = []
            date_to_lros[date_str].append((i, lro))

        # Step 2: Merge LROs with duplicate dates
        merged_lros = []
        merged_animaldays = []

        for date_str in sorted(date_to_lros.keys()):  # Sort for deterministic ordering
            lro_group = date_to_lros[date_str]
            animalday = f"{animal_id} {genotype} {date_str}"

            if len(lro_group) == 1:
                # Single LRO for this date - use as-is
                _, lro = lro_group[0]
                merged_lros.append(lro)
                merged_animaldays.append(animalday)
                logging.info(f"Using single LRO for {animalday}")
            else:
                # Multiple LROs for same date - merge them
                logging.info(
                    f"Found {len(lro_group)} LROs for {animalday}. "
                    f"Merging into single LRO (mimicking normal __init__ behavior)."
                )

                # Filter out 0-sample LROs before the merge loop.  A 0-sample
                # base LRO makes si.concatenate_recordings fail; using the same
                # helper as _create_long_recordings keeps the two code paths
                # consistent.
                lro_pairs = [(f"lro_{idx}", lro) for idx, lro in lro_group]
                valid_pairs, skipped_names = cls._filter_zero_sample_lros(
                    lro_pairs, lambda k: k
                )
                if skipped_names:
                    logging.warning(
                        f"Skipping {len(skipped_names)} 0-sample LRO(s) for "
                        f"'{animalday}' before merge: {skipped_names}"
                    )
                if not valid_pairs:
                    logging.warning(
                        f"All {len(lro_group)} LRO(s) for '{animalday}' are "
                        f"0-sample; skipping this date."
                    )
                    continue
                lro_pairs = valid_pairs

                # Sort by median time (same logic as normal __init__)
                sorted_pairs = cls._sort_lros_by_median_time_static(lro_pairs)

                # Merge all LROs into the first one (in temporal order)
                base_lro = sorted_pairs[0][1]
                base_tag = sorted_pairs[0][0]
                logging.info(f"Base LRO: {base_tag}")

                for i, (_, lro) in enumerate(sorted_pairs[1:], 1):
                    try:
                        logging.info(f"Merging LRO {i} into base LRO for {animalday}")
                        base_lro.merge(lro)
                    except ValueError as e:
                        # Provide detailed error for incompatible LROs
                        raise ValueError(
                            f"Cannot merge LROs for {animalday}: {e}\n"
                            f"All LROs with the same date must have compatible metadata "
                            f"(same channels, sampling rate, etc.)."
                        ) from e

                merged_lros.append(base_lro)
                merged_animaldays.append(animalday)
                logging.info(
                    f"Successfully merged {len(lro_group)} LROs for {animalday}"
                )

        # Step 3: Ensure at least one date produced a valid (non-empty) merged LRO.
        # If every date group was filtered out as 0-sample, merged_lros is empty
        # and downstream methods (e.g. compute_windowed_analysis) would fail with
        # less informative errors.  Raise early, consistent with the normal
        # __init__ path which raises when nothing is loadable.
        if not merged_lros:
            raise ValueError(
                f"No non-empty local recording objects (LROs) could be loaded. "
                f"All date groups were 0-sample. Cannot construct an "
                f"AnimalOrganizer with no recordings."
            )

        # Step 4: Set merged LROs and animaldays
        ao.long_recordings = merged_lros
        ao.unique_animaldays = merged_animaldays
        ao.animaldays = (
            merged_animaldays.copy()
        )  # Create separate list for compatibility

        # Step 5: Validate and reconcile channel names across all merged LROs
        ao.channel_names = cls._validate_channel_names(merged_lros)

        # Step 6: CRITICAL VALIDATION - ensure no duplicates after merge
        if len(ao.long_recordings) != len(set(ao.unique_animaldays)):
            duplicate_dates = [
                date
                for date in ao.unique_animaldays
                if ao.unique_animaldays.count(date) > 1
            ]
            raise ValueError(
                f"CRITICAL ERROR: Duplicate animaldays detected after merge! "
                f"This indicates a logic error in the merge process. "
                f"Duplicates: {set(duplicate_dates)}\n"
                f"Expected {len(set(ao.unique_animaldays))} unique dates, "
                f"but got {len(ao.long_recordings)} LROs."
            )

        logging.info(
            f"Validated: {len(ao.long_recordings)} LROs match "
            f"{len(ao.unique_animaldays)} unique animaldays (no duplicates)"
        )

        # Step 7: Initialize default attributes for factory-created instances
        cls._init_factory_defaults(ao, animal_id, merged_lros)

        logging.info(
            f"Created AnimalOrganizer from {len(lros)} input LROs "
            f"(merged into {len(merged_lros)} unique dates) for animal '{animal_id}'"
        )

        return ao

    @staticmethod
    def _validate_channel_names(lros: list[_lro.LongRecordingOrganizer]) -> list[str]:
        """
        Validate that all LROs have consistent channel names.

        Compares abbreviated channel names (via ``resolve_channel``)
        so that cosmetic variants like ``L Barrel`` vs ``L Barrel Ctx`` are
        treated as equivalent. If raw names differ but abbreviations match,
        the mismatched LRO's channel names are renamed to match the reference
        LRO's raw names for downstream consistency.

        If channel names are the same but in different order, the first LRO's
        order is used as reference.

        Args:
            lros: List of LROs to validate.

        Returns:
            list[str]: The canonical channel names (from the first LRO).

        Raises:
            ValueError: If LROs have different abbreviated channel sets.
        """
        if not lros:
            return []

        first_names = lros[0].channel_names if lros[0].channel_names else []
        if not first_names:
            return []

        reference_abbrevs = resolve_channels(first_names)
        reference_set = set(reference_abbrevs)
        # Map abbreviation -> canonical raw name from first LRO
        abbrev_to_raw = dict(zip(reference_abbrevs, first_names))

        for i, lro in enumerate(lros[1:], start=1):
            current_names = lro.channel_names if lro.channel_names else []
            current_abbrevs = resolve_channels(current_names)
            current_set = set(current_abbrevs)

            if current_set != reference_set:
                missing = reference_set - current_set
                extra = current_set - reference_set
                raise ValueError(
                    f"LRO {i} has inconsistent channel names. "
                    f"Abbreviated missing: {missing}, Extra: {extra}"
                )

            # If raw names differ but abbreviations match, rename to reference
            if set(current_names) != set(first_names):
                renamed = [abbrev_to_raw[a] for a in current_abbrevs]
                logging.warning(
                    f"LRO {i} has variant channel names "
                    f"({current_names} vs {first_names}), "
                    f"renaming to reference names: {renamed}"
                )
                lro.channel_names = renamed

            # If same channels but different order, log a warning
            if lro.channel_names != first_names:
                logging.warning(
                    f"LRO {i} has channels in different order, using reference order"
                )

        return first_names

    @staticmethod
    def _init_factory_defaults(
        ao: "AnimalOrganizer", animal_id: str, lros: list[_lro.LongRecordingOrganizer]
    ) -> None:
        """
        Initialize attribute values for factory-created instances.

        Derives values from the provided LROs where possible instead of
        leaving attributes empty.

        Args:
            ao: The AnimalOrganizer instance to initialize.
            animal_id: The animal identifier.
            lros: The LROs to derive metadata from.
        """
        # Standard attributes
        ao.animal_file_match_pattern = [animal_id]
        ao.day_sep = None
        ao.read_mode = "base"

        # Internal cache - not derivable, but private
        ao._animalday_dicts = []
        ao._animalday_folder_groups = {}
        ao._processed_timestamps = None

    def split(
        self,
        groups: dict[str, list[str]],
        output_base: Union[str, Path] = None,
        format: Literal["zarr", "binary"] = "zarr",
        overwrite: bool = False,
        persist_base: Union[str, Path] = None,
    ) -> dict[str, "AnimalOrganizer"]:
        """
        Split this multi-animal AnimalOrganizer into per-animal AnimalOrganizers.

        For each group (animal), this method:
        1. Iterates over all LROs in this AnimalOrganizer
        2. Calls LRO.split() on each to extract the specified channels
        3. Optionally saves each split LRO to disk
        4. Creates a new AnimalOrganizer for each group

        This enables processing of joint-animal recordings where multiple animals
        are recorded on different channels of the same files.

        Args:
            groups (dict[str, list[str]]): Dictionary mapping group names (animal IDs)
                to lists of channel names. Example:
                {"AnimalA": ["Ch0", "Ch1", "Ch2", "Ch3"],
                 "AnimalB": ["Ch4", "Ch5", "Ch6", "Ch7"]}
            output_base (Union[str, Path], optional): Base directory for saving
                split recordings. If None, LROs remain in-memory. Structure:
                output_base/
                    AnimalA/
                        day1.zarr
                        day2.zarr
                    AnimalB/
                        ...
            format (Literal["zarr", "binary"], optional): Format for saved
                recordings. Defaults to "zarr".
            overwrite (bool, optional): Passed to
                :meth:`LongRecordingOrganizer.save_recording`; if True, replace an
                existing (recognized) recording folder. Defaults to False.
            persist_base (Union[str, Path], optional): Deprecated alias for
                ``output_base``. If provided (not None), it is used as ``output_base``
                and emits a :class:`DeprecationWarning`.

        Returns:
            dict[str, AnimalOrganizer]: Dictionary mapping group names to new
                AnimalOrganizer instances.

        Raises:
            ValueError: If requested channels are not found in recordings.

        Example:
            >>> ao = AnimalOrganizer("/path/to/joint_data", "combined")
            >>> splits = ao.split(
            ...     groups={"MouseA": ["Ch0", "Ch1"], "MouseB": ["Ch2", "Ch3"]},
            ...     output_base="/output/split_data",
            ... )
            >>> war_a = AnimalAnalyzer(splits["MouseA"]).compute_windowed_analysis(["all"])
            >>> war_b = AnimalAnalyzer(splits["MouseB"]).compute_windowed_analysis(["all"])
        """
        if not self.long_recordings:
            raise ValueError("No recordings loaded to split")

        if persist_base is not None:
            warnings.warn(
                "The 'persist_base' argument of AnimalOrganizer.split() is deprecated; "
                "use 'output_base'.",
                DeprecationWarning,
                stacklevel=2,
            )
            if output_base is None:
                output_base = persist_base

        if output_base is not None:
            output_base = Path(output_base)
            output_base.mkdir(parents=True, exist_ok=True)

        result = {}

        for group_name, channels in groups.items():
            logging.info(
                f"Splitting group '{group_name}' with {len(channels)} channels "
                f"across {len(self.long_recordings)} days"
            )

            child_lros = []
            for i, lro in enumerate(self.long_recordings):
                # Split this day's LRO
                day_splits = lro.split({group_name: channels})
                child_lro = day_splits[group_name]

                # Save to disk if requested
                if output_base is not None:
                    # Determine day folder name
                    day_name = lro.display_name or f"day{i}"

                    output_dir = output_base / group_name / day_name
                    child_lro.save_recording(output_dir, format=format, overwrite=overwrite)
                    logging.debug(f"Saved {group_name}/{day_name} to {output_dir}")

                child_lros.append(child_lro)

            # Create AnimalOrganizer from the split LROs
            child_ao = type(self).from_lros(
                lros=child_lros,
                animal_id=group_name,
                genotype=self.genotype,
                sex=self.sex,
            )

            result[group_name] = child_ao
            logging.info(
                f"Created AnimalOrganizer for '{group_name}' with "
                f"{len(child_lros)} days, {len(channels)} channels"
            )

        return result
