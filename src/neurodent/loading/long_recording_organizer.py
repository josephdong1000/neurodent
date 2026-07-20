"""Long recording loader for the loading stage.

``LongRecordingOrganizer`` builds a single long recording from disk (or an
in-memory recording) and exposes fragment access, timestamps, quality, persistence,
and split/merge. Its behavior is composed from the ``lro_*`` mixins in this package;
this module keeps the base methods (including the name-mangled index helpers and
their fragment-accessor callers), the class declaration, and ``split_recording``.
"""

import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Union

import mne

from neurodent.core.utils import parse_truncate

from .recording_metadata import RecordingMetadata

from .lro_loading import LroLoadingMixin
from .lro_timestamps import LroTimestampsMixin
from .lro_fragments import LroFragmentsMixin
from .lro_quality import LroQualityMixin
from .lro_persistence import LroPersistenceMixin
from .lro_merge import LroMergeMixin


def split_recording(
    input_path: Union[str, Path],
    groups: dict[str, list[str]],
    output_base: Union[str, Path] = None,
    mode: Literal["si", "mne"] = "si",
    format: Literal["zarr", "binary"] = "zarr",
    save: bool = True,
    overwrite: bool = False,
    persist: Union[bool, None] = None,
    **lro_kwargs,
) -> dict[str, "LongRecordingOrganizer"]:
    """
    Split a multi-animal recording file into separate per-animal recordings.

    This is a standalone convenience function that creates an LRO, splits it,
    and optionally saves the results to disk.

    Args:
        input_path (Union[str, Path]): Path to the input recording file/folder.
        groups (dict[str, list[str]]): Dictionary mapping group names to channel lists.
            Example: {"AnimalA": ["Ch1", "Ch2"], "AnimalB": ["Ch3", "Ch4"]}
        output_base (Union[str, Path], optional): Base directory for output. Required if save=True.
        mode (Literal["si", "mne"], optional): Mode for loading input. Defaults to "si".
        format (Literal["zarr", "binary"], optional): Output format. Defaults to "zarr".
        save (bool, optional): If True, save splits to disk via
            :meth:`LongRecordingOrganizer.save_recording`. Defaults to True.
        overwrite (bool, optional): Passed to ``save_recording``; if True, replace an
            existing (recognized) recording folder. Defaults to False.
        persist (bool, optional): Deprecated alias for ``save``. If provided (not None),
            it overrides ``save`` and emits a :class:`DeprecationWarning`.
        **lro_kwargs: Additional arguments passed to LongRecordingOrganizer.

    Returns:
        dict[str, LongRecordingOrganizer]: Dictionary mapping group names to LRO instances.

    Example:
        >>> from neurodent.loading import split_recording
        >>> splits = split_recording(
        ...     "/path/to/session.bin",
        ...     groups={"AnimalA": ["Ch1", "Ch2"], "AnimalB": ["Ch3", "Ch4"]},
        ...     output_base="/path/to/output",
        ... )
    """
    if persist is not None:
        warnings.warn(
            "The 'persist' argument of split_recording() is deprecated; use 'save'.",
            DeprecationWarning,
            stacklevel=2,
        )
        save = persist

    # Load the input recording
    lro = LongRecordingOrganizer(input_path, mode=mode, **lro_kwargs)

    # Split into in-memory LROs
    splits = lro.split(groups)

    # Save if requested
    if save:
        if output_base is None:
            raise ValueError("output_base is required when save=True")
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)

        for group_name, child_lro in splits.items():
            output_dir = output_base / group_name
            child_lro.save_recording(output_dir, format=format, overwrite=overwrite)

    return splits


class LongRecordingOrganizer(
    LroLoadingMixin,
    LroTimestampsMixin,
    LroFragmentsMixin,
    LroQualityMixin,
    LroPersistenceMixin,
    LroMergeMixin,
):
    """
    Construct a long recording from various file formats or an existing recording object.

    Args:
        item (str | Path | list[str] | DiscoveredFile | None): Input data specification.
            - str/Path: Single file or directory path
            - list[str]: Multiple files to concatenate
            - DiscoveredFile: File(s) discovered by FileDiscoverer (single or multi-file)
            - None: Used when initializing from an existing recording object
        mode (Literal['si', 'mne', None], optional): Data loading mode. Defaults to 'si'.
            - 'si': Use SpikeInterface extractors
            - 'mne': Use MNE-Python extractors (creates intermediate file)
            - None: No data loading (item must be None, recording must be provided)
        truncate (bool | int, optional): If True, truncate to first 10 files.
            If an integer, truncate to first n files. Defaults to False.
        cache_policy (Literal['auto', 'always', 'force_regenerate'], optional):
            Cache policy for intermediate files. Defaults to 'auto'.
        multiprocess_mode (Literal['dask', 'serial'], optional): Processing mode for
            parallel operations when loading multiple files. Defaults to 'serial'.
        extract_func (Callable | str, optional): Function to extract data.
            - If str: name of SpikeInterface or MNE extractor (e.g., 'read_intan', 'read_raw_edf')
            - If Callable: custom extraction function
            - If None: defaults to si.load for SI mode
        manual_datetimes (datetime | list[datetime], optional): Manually provided timestamps.
        datetimes_are_start (bool, optional): If True (default), manual_datetimes are start times.
        n_jobs (int, optional): Number of parallel jobs for MNE resampling. Defaults to 1.
        recording (si.BaseRecording, optional): Existing SpikeInterface recording object
            for in-memory initialization. Use this when creating LRO wrappers around split recordings.
        **kwargs: Additional arguments passed to the data loading functions.

    Attributes:
        LongRecording (si.BaseRecording): The SpikeInterface recording object.
        meta (RecordingMetadata): Technical metadata (sampling rate, channels, etc.).
        channel_names (list[str]): List of channel names.
        file_durations (list[float]): Duration of each individual file in seconds.
        cumulative_file_durations (list[float]): Cumulative duration timestamps for file boundaries.
        temppaths (list[str]): Paths to temporary files created during processing.
        bad_channel_names (list[str]): List of channels identified as bad/noisy.
        _is_in_memory (bool): True if this LRO was created from an in-memory recording (via split()).

    Raises:
        ValueError: If no data files are found, if the folder contains mixed file types,
            or if manual time parameters are invalid.
    """

    def __init__(
        self,
        item: Union[str, Path, list[str], tuple[str], "DiscoveredFile"],
        mode: Literal["si", "mne", None] = "si",
        truncate: Union[bool, int] = False,
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        extract_func: Union[
            Callable[..., "si.BaseRecording"], Callable[..., mne.io.Raw], str
        ] = None,
        manual_datetimes: datetime | list[datetime] = None,
        datetimes_are_start: bool = True,
        n_jobs: int = 1,
        recording: "si.BaseRecording" = None,
        **kwargs,
    ):
        # Import DiscoveredFile here to avoid circular imports
        from .discovery import DiscoveredFile

        if isinstance(item, DiscoveredFile):
            # DiscoveredFile: handle both single and multi-file cases
            self.data_files = None
            self.item = item
        elif isinstance(item, (list, tuple)):
            # List of files: will be concatenated individually
            self.data_files = [str(x) for x in item]
            self.item = self.data_files
        else:
            # Single file/path or None
            self.data_files = None
            self.item = item

        self.n_truncate = parse_truncate(truncate)
        self.truncate = True if self.n_truncate > 0 else False
        if self.truncate:
            warnings.warn(
                f"LongRecording will be truncated to the first {self.n_truncate} files"
            )

        self.manual_datetimes = manual_datetimes
        self.datetimes_are_start = datetimes_are_start
        self.n_jobs = n_jobs
        self.labels = {}

        self.meta = None
        self.channel_names = None   # display labels
        self.channel_ids = None     # stable identity (what configs key on); see _extract_channel_identities
        self.LongRecording = None
        self.temppaths = []
        self.file_durations = []
        self.cumulative_file_durations = []
        self.bad_channel_names = []
        self._is_in_memory = False

        if recording is not None:
            self._init_from_recording(recording)
            return

        if self.item is not None:
            self._validate_manual_time_params()

        if mode is not None and self.item is not None:
            self.detect_and_load_data(
                mode=mode,
                cache_policy=cache_policy,
                multiprocess_mode=multiprocess_mode,
                extract_func=extract_func,
                **kwargs,
            )

    @property
    def display_name(self) -> str:
        """Short display name for logging, derived from the item."""
        from .discovery import DiscoveredFile

        if isinstance(self.item, DiscoveredFile):
            paths = self.item.get_path_list()
            if paths:
                name = Path(paths[0]).name
                return f"{name}..." if len(paths) > 1 else name
        if isinstance(self.item, (list, tuple)) and self.item:
            return Path(self.item[0]).name
        if self.item is not None:
            return str(Path(str(self.item)).name)
        return "unknown"

    def get_num_fragments(self, fragment_len_s):
        frag_len_idx = self.__time_to_idx(fragment_len_s)
        duration_idx = self.LongRecording.get_num_frames()
        return math.ceil(duration_idx / frag_len_idx)

    def __time_to_idx(self, time_s):
        return self.LongRecording.time_to_sample_index(time_s)

    def __idx_to_time(self, idx):
        return self.LongRecording.sample_index_to_time(idx)

    def get_fragment(self, fragment_len_s, fragment_idx):
        startidx, endidx = self.__fragidx_to_startendind(fragment_len_s, fragment_idx)
        return self.LongRecording.frame_slice(startidx, endidx)

    def get_dur_fragment(self, fragment_len_s, fragment_idx):
        startidx, endidx = self.__fragidx_to_startendind(fragment_len_s, fragment_idx)
        return self.__idx_to_time(endidx) - self.__idx_to_time(startidx)

    def __fragidx_to_startendind(self, fragment_len_s, fragment_idx):
        """Convert fragment index to start and end sample indices.

        Args:
            fragment_len_s (float): Length of each fragment in seconds
            fragment_idx (int): Index of the fragment to get indices for

        Returns:
            tuple[int, int]: Start and end sample indices for the fragment. The end index is capped at the recording length.
        """
        frag_len_idx = self.__time_to_idx(fragment_len_s)
        startidx = frag_len_idx * fragment_idx
        endidx = min(
            frag_len_idx * (fragment_idx + 1), self.LongRecording.get_num_frames()
        )
        return startidx, endidx

    def __str__(self):
        """Return a string representation of critical long recording features."""
        if not hasattr(self, "LongRecording") or self.LongRecording is None:
            return "LongRecordingOrganizer: No recording loaded yet"

        n_channels = self.LongRecording.get_num_channels()
        sampling_freq = self.LongRecording.get_sampling_frequency()
        total_duration = self.LongRecording.get_duration()

        n_files = (
            len(self.file_durations)
            if hasattr(self, "file_durations") and self.file_durations
            else 1
        )

        timestamp_info = "No timestamps"
        if hasattr(self, "file_end_datetimes") and self.file_end_datetimes:
            timestamp_coverage = len(
                [x for x in self.file_end_datetimes if x is not None]
            )
            timestamp_info = f"{timestamp_coverage}/{len(self.file_end_datetimes)} files have timestamps"

        channel_info = "No channels"
        if hasattr(self, "channel_names") and self.channel_names:
            # if len(self.channel_names) <= 5:
            channel_info = f"[{', '.join(self.channel_names)}]"
            # else:
            #     channel_info = f"[{', '.join(self.channel_names[:3])}, ..., {self.channel_names[-1]}] ({len(self.channel_names)} total)"

        metadata_info = ""
        if hasattr(self, "meta") and self.meta:
            if hasattr(self.meta, "precision") and self.meta.precision:
                metadata_info = f", {self.meta.precision} precision"
            if hasattr(self.meta, "V_units") and self.meta.V_units:
                metadata_info += f", {self.meta.V_units} units"

        return (
            f"LongRecording: {n_files} files, {n_channels} channels, "
            f"{sampling_freq} Hz, {total_duration:.1f}s duration, "
            f"channels: {channel_info}{metadata_info}, timestamps: {timestamp_info}"
        )

    def __repr__(self):
        """Return a detailed string representation for debugging."""
        return self.__str__()
