import glob
import gzip
import copy
import json
import logging
import math
import os
import statistics
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Literal, Union

try:
    import dask
except (
    Exception
):  # pragma: no cover - optional at import time for tests that don't use dask
    dask = None
import mne
import numpy as np
import pandas as pd

try:
    import spikeinterface.core as si
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre
    import spikeinterface.widgets as sw
except (
    Exception
):  # pragma: no cover - optional at import time for tests not using spikeinterface
    si = None
    se = None
    spre = None
    sw = None
from scipy.signal import decimate
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import pdist, squareform

from .. import constants
from .utils import (
    Natural_Neighbor,
    TimestampMapper,
    convert_colpath_to_rowpath,
    convert_units_to_multiplier,
    extract_mne_unit_info,
    filepath_to_index,
    get_temp_directory,
    parse_truncate,
    get_file_stem,
    should_use_cache_unified,
    get_cache_status_message,
    convert_intan_chname_mne,
)


class RecordingMetadata:
    """Stores metadata information for neural recordings.

    This class handles recording metadata including channel information, sampling rates,
    timestamps, and voltage units. It can be initialized either from a CSV metadata file
    (for backward compatibility with DDF binary format) or directly from parameters.

    Attributes:
        metadata_path (str | Path | None): Path to metadata CSV file if loaded from file
        metadata_df (pd.DataFrame | None): DataFrame containing metadata if loaded from file
        n_channels (int): Number of channels in the recording
        f_s (float): Sampling frequency in Hz
        V_units (str | None): Voltage units (e.g., 'µV', 'mV', 'V')
        mult_to_uV (float | None): Multiplication factor to convert to microvolts
        precision (str | None): Data precision/dtype (e.g., 'float32', 'int16')
        dt_end (datetime | None): End datetime of recording
        channel_names (list[str]): List of channel names

    Examples:
        From parameters:
        >>> meta = RecordingMetadata(
        ...     None,
        ...     n_channels=4,
        ...     f_s=1000.0,
        ...     dt_end=datetime(2023, 1, 1),
        ...     channel_names=['ch1', 'ch2', 'ch3', 'ch4']
        ... )

        From CSV file:
        >>> meta = RecordingMetadata('/path/to/metadata.csv')
    """
    def __init__(
        self,
        metadata_path: str | Path | None,
        *,
        n_channels: int | None = None,
        f_s: float | None = None,
        dt_end: datetime | None = None,
        channel_names: list[str] | None = None,
        V_units: str | None = None,
        mult_to_uV: float | None = None,
    ) -> None:
        """Initialize RecordingMetadata either from a file path or direct parameters.

        Args:
            metadata_path (str | Path | None): Path to metadata CSV file. If provided,
                other parameters are ignored and metadata is loaded from the file.
            n_channels (int, optional): Number of channels in the recording
            f_s (float, optional): Sampling frequency in Hz
            dt_end (datetime, optional): End datetime of recording
            channel_names (list[str], optional): List of channel names
            V_units (str, optional): Voltage units (e.g., 'µV', 'mV', 'V')
            mult_to_uV (float, optional): Multiplication factor to convert to microvolts

        Raises:
            ValueError: If metadata_path is None and required parameters are missing
        """
        if metadata_path is not None:
            self._init_from_path(metadata_path)
        else:
            self._init_from_params(
                n_channels, f_s, dt_end, channel_names, V_units, mult_to_uV
            )

    def _init_from_path(self, metadata_path):
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path)
        if self.metadata_df.empty:
            raise ValueError(f"Metadata file is empty: {metadata_path}")

        self.n_channels = len(self.metadata_df.index)
        self.f_s = self.__getsinglecolval(
            "SampleRate"
        )  # NOTE this may not be the same as LongRecording (Recording object) f_s, which the name should reflect
        self.V_units = self.__getsinglecolval("Units")
        self.mult_to_uV = convert_units_to_multiplier(self.V_units)
        self.precision = self.__getsinglecolval("Precision")

        if "LastEdit" in self.metadata_df.keys():
            self.dt_end = datetime.fromisoformat(self.__getsinglecolval("LastEdit"))
        else:
            self.dt_end = None
            logging.warning(
                "No LastEdit column provided in metadata. dt_end set to None"
            )

        self.channel_names = self.metadata_df["ProbeInfo"].tolist()

    def _init_from_params(
        self, n_channels, f_s, dt_end, channel_names, V_units=None, mult_to_uV=None
    ):
        if None in (n_channels, f_s, channel_names):
            raise ValueError(
                "All parameters must be provided when not using metadata_path"
            )

        self.metadata_path = None
        self.metadata_df = None
        self.n_channels = n_channels
        self.f_s = f_s  # NOTE see above note about f_s
        self.V_units = V_units
        self.mult_to_uV = mult_to_uV
        self.precision = None
        self.dt_end = dt_end

        if not isinstance(channel_names, list):
            raise ValueError("channel_names must be a list")

        self.channel_names = channel_names

    def __getsinglecolval(self, colname):
        vals = self.metadata_df.loc[:, colname]
        if len(np.unique(vals)) > 1:
            warnings.warn(f"Not all {colname}s are equal!")
        if vals.size == 0:
            return None
        return vals.iloc[0]

    def to_dict(self) -> dict:
        """Convert RecordingMetadata to a dictionary for JSON serialization."""
        return {
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "n_channels": self.n_channels,
            "f_s": self.f_s,
            "V_units": self.V_units,
            "mult_to_uV": self.mult_to_uV,
            "precision": self.precision,
            "dt_end": self.dt_end.isoformat() if self.dt_end else None,
            "channel_names": self.channel_names,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecordingMetadata":
        """Create RecordingMetadata from a dictionary (from JSON deserialization)."""
        dt_end = datetime.fromisoformat(data["dt_end"]) if data["dt_end"] else None

        return cls(
            metadata_path=None,  # We're reconstructing from cached data
            n_channels=data["n_channels"],
            f_s=data["f_s"],
            dt_end=dt_end,
            channel_names=data["channel_names"],
            V_units=data.get("V_units"),
            mult_to_uV=data.get("mult_to_uV"),
        )

    def to_json(self, file_path: Path) -> None:
        """Save RecordingMetadata to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, file_path: Path) -> "RecordingMetadata":
        """Load RecordingMetadata from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        # Reconstruct the object, preserving additional fields that were serialized
        instance = cls.from_dict(data)

        # Set additional fields that might not be in from_dict
        instance.V_units = data.get("V_units")
        instance.mult_to_uV = data.get("mult_to_uV")
        instance.precision = data.get("precision")

        return instance

    def update_sampling_rate(self, new_f_s: float) -> None:
        """Update the sampling rate in this metadata object.

        This should be called when the associated recording is resampled.
        """
        old_f_s = self.f_s
        self.f_s = new_f_s
        logging.info(
            f"Updated RecordingMetadata sampling rate from {old_f_s} Hz to {new_f_s} Hz"
        )


# Deprecated: Keep DDFBinaryMetadata for backward compatibility
class DDFBinaryMetadata(RecordingMetadata):
    """Deprecated: Use RecordingMetadata instead.

    This class is maintained for backward compatibility but will be removed in a future version.
    The name DDFBinaryMetadata is no longer appropriate as the pipeline moves beyond
    DDF binary files with metadata sidecars.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DDFBinaryMetadata is deprecated. Use RecordingMetadata instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


def convert_ddfcolbin_to_ddfrowbin(rowdir_path, colbin_path, metadata, save_gzip=True):
    # TODO consider renaming this function to something more descriptive, like convert_colbin_to_rowbin
    # Also don't use the rowdir_path parameter, since this is outside the scope of the function. See utils.convert_colpath_to_rowpath
    assert isinstance(
        metadata, RecordingMetadata
    ), "Metadata needs to be of type RecordingMetadata"

    tempbin = np.fromfile(colbin_path, dtype=metadata.precision)
    tempbin = np.reshape(tempbin, (-1, metadata.n_channels), order="F")

    rowbin_path = convert_colpath_to_rowpath(rowdir_path, colbin_path, gzip=save_gzip)

    if save_gzip:
        # rowbin_path = str(rowbin_path) + ".npy.gz"
        with gzip.GzipFile(rowbin_path, "w") as fcomp:
            np.save(file=fcomp, arr=tempbin)
    else:
        # rowbin_path = str(rowbin_path) + ".bin"
        tempbin.tofile(rowbin_path)

    return rowbin_path


def convert_ddfrowbin_to_si(bin_rowmajor_path, metadata):
    """Convert a row-major binary file to a SpikeInterface recording object.

    Args:
        bin_rowmajor_path (str): Path to the row-major binary file
        metadata (RecordingMetadata): Metadata object containing information about the recording

    Returns:
        tuple: A tuple containing:
            - se.BaseRecording: The SpikeInterface Recording object.
            - str or None: Path to temporary file if created, None otherwise.
    """
    if se is None:
        raise ImportError("SpikeInterface is required for convert_ddfrowbin_to_si")
    assert isinstance(
        metadata, RecordingMetadata
    ), "Metadata needs to be of type RecordingMetadata"

    bin_rowmajor_path = Path(bin_rowmajor_path)
    params = {
        "sampling_frequency": metadata.f_s,
        "dtype": metadata.precision,
        "num_channels": metadata.n_channels,
        "gain_to_uV": metadata.mult_to_uV,
        "offset_to_uV": 0,
        "time_axis": 0,
        "is_filtered": False,
    }

    # Read either .npy.gz files or .bin files into the recording object
    if ".npy.gz" in str(bin_rowmajor_path):
        temppath = os.path.join(get_temp_directory(), os.urandom(24).hex())
        try:
            with open(temppath, "wb") as tmp:
                try:
                    fcomp = gzip.GzipFile(bin_rowmajor_path, "r")
                    bin_rowmajor_decomp = np.load(fcomp)
                    bin_rowmajor_decomp.tofile(tmp)
                except (EOFError, OSError) as e:
                    logging.error(
                        f"Failed to read .npy.gz file: {bin_rowmajor_path}. Try regenerating row-major files."
                    )
                    raise

            rec = se.read_binary(tmp.name, **params)
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temppath):
                os.remove(temppath)
            raise
    else:
        rec = se.read_binary(bin_rowmajor_path, **params)
        temppath = None

    if rec.sampling_frequency != constants.GLOBAL_SAMPLING_RATE:
        warnings.warn(
            f"Sampling rate {rec.sampling_frequency} Hz != {constants.GLOBAL_SAMPLING_RATE} Hz. Resampling"
        )
        rec = spre.resample(rec, constants.GLOBAL_SAMPLING_RATE)
        # Update metadata to reflect the new sampling rate
        metadata.update_sampling_rate(constants.GLOBAL_SAMPLING_RATE)

    rec = spre.astype(rec, dtype=constants.GLOBAL_DTYPE)

    return rec, temppath


def _convert_ddfrowbin_to_si_no_resample(bin_rowmajor_path, metadata):
    """Convert a row-major binary file to a SpikeInterface recording object WITHOUT resampling.

    This is an internal function used by the unified resampling pipeline to avoid
    resampling individual recordings before concatenation. Resampling is applied
    once after concatenation for better performance.

    Args:
        bin_rowmajor_path (str): Path to the row-major binary file
        metadata (DDFBinaryMetadata): Metadata object containing information about the recording

    Returns:
        tuple: A tuple containing:
            - se.BaseRecording: The SpikeInterface Recording object (NOT resampled).
            - str or None: Path to temporary file if created, None otherwise.
    """
    if se is None:
        raise ImportError(
            "SpikeInterface is required for _convert_ddfrowbin_to_si_no_resample"
        )
    assert isinstance(
        metadata, DDFBinaryMetadata
    ), "Metadata needs to be of type DDFBinaryMetadata"

    bin_rowmajor_path = Path(bin_rowmajor_path)
    params = {
        "sampling_frequency": metadata.f_s,
        "dtype": metadata.precision,
        "num_channels": metadata.n_channels,
        "gain_to_uV": metadata.mult_to_uV,
        "offset_to_uV": 0,
        "time_axis": 0,
        "is_filtered": False,
    }

    # Read either .npy.gz files or .bin files into the recording object
    if ".npy.gz" in str(bin_rowmajor_path):
        temppath = os.path.join(get_temp_directory(), os.urandom(24).hex())
        try:
            with open(temppath, "wb") as tmp:
                try:
                    fcomp = gzip.GzipFile(bin_rowmajor_path, "r")
                    bin_rowmajor_decomp = np.load(fcomp)
                    bin_rowmajor_decomp.tofile(tmp)
                except (EOFError, OSError) as e:
                    logging.error(
                        f"Failed to read .npy.gz file: {bin_rowmajor_path}. Try regenerating row-major files."
                    )
                    raise

            rec = se.read_binary(tmp.name, **params)
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temppath):
                os.remove(temppath)
            raise
    else:
        rec = se.read_binary(bin_rowmajor_path, **params)
        temppath = None

    # NOTE: No resampling applied here - will be handled by unified resampling after concatenation
    rec = spre.astype(rec, dtype=constants.GLOBAL_DTYPE)

    return rec, temppath


def split_recording(
    input_path: Union[str, Path],
    groups: dict[str, list[str]],
    output_base: Union[str, Path] = None,
    mode: Literal["si", "mne"] = "si",
    format: Literal["zarr", "binary"] = "zarr",
    persist: bool = True,
    **lro_kwargs,
) -> dict[str, "LongRecordingOrganizer"]:
    """
    Split a multi-animal recording file into separate per-animal recordings.

    This is a standalone convenience function that creates an LRO, splits it,
    and optionally persists the results to disk.

    Args:
        input_path (Union[str, Path]): Path to the input recording file/folder.
        groups (dict[str, list[str]]): Dictionary mapping group names to channel lists.
            Example: {"AnimalA": ["Ch1", "Ch2"], "AnimalB": ["Ch3", "Ch4"]}
        output_base (Union[str, Path], optional): Base directory for output. Required if persist=True.
        mode (Literal["si", "mne"], optional): Mode for loading input. Defaults to "si".
        format (Literal["zarr", "binary"], optional): Output format. Defaults to "zarr".
        persist (bool, optional): If True, save splits to disk. Defaults to True.
        **lro_kwargs: Additional arguments passed to LongRecordingOrganizer.

    Returns:
        dict[str, LongRecordingOrganizer]: Dictionary mapping group names to LRO instances.

    Example:
        >>> from neurodent.core import split_recording
        >>> splits = split_recording(
        ...     "/path/to/session.bin",
        ...     groups={"AnimalA": ["Ch1", "Ch2"], "AnimalB": ["Ch3", "Ch4"]},
        ...     output_base="/path/to/output",
        ... )
    """
    # Load the input recording
    lro = LongRecordingOrganizer(input_path, mode=mode, **lro_kwargs)

    # Split into in-memory LROs
    splits = lro.split(groups)

    # Persist if requested
    if persist:
        if output_base is None:
            raise ValueError("output_base is required when persist=True")
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)

        for group_name, child_lro in splits.items():
            output_dir = output_base / group_name
            child_lro.persist(output_dir, format=format)

    return splits


class LongRecordingOrganizer:
    """
    Construct a long recording from various file formats or an existing recording object.

    Args:
        item (str | Path | list[str] | MultiFileGroup | None): Input data specification.
            - str/Path: Single file or directory path
            - list[str]: Multiple files to concatenate
            - MultiFileGroup: Multiple files that should be loaded together as one unit
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
            - If None: defaults to si.load_extractor for SI mode
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
        item: Union[str, Path, list[str], tuple[str], "MultiFileGroup"],
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
        self.channel_names = None
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

    def _init_from_recording(self, recording: "si.BaseRecording"):
        """Initialize LRO from an existing SpikeInterface recording object (in-memory)."""
        # Enforce global dtype and resampling
        self.LongRecording = self._apply_resampling(recording)
        recording = self.LongRecording

        self._is_in_memory = True

        # Extract metadata from recording
        channel_ids = recording.get_channel_ids()
        self.channel_names = [str(ch) for ch in channel_ids]

        self.meta = RecordingMetadata(
            None,
            n_channels=recording.get_num_channels(),
            f_s=recording.get_sampling_frequency(),
            dt_end=None,  # In-memory recordings don't have timestamps until persisted
            channel_names=self.channel_names,
        )

        # Compute file duration from recording
        duration_s = recording.get_total_duration()
        self.file_durations = [duration_s]
        self.cumulative_file_durations = [duration_s]

    @staticmethod
    def _resolve_dotted_path(path_str: str) -> Callable:
        """Import and return a callable from a dotted path.

        Parameters
        ----------
        path_str : str
            Fully-qualified dotted path, e.g. ``"mypackage.readers.read_bin_csv"``.

        Returns
        -------
        Callable
            The resolved callable.

        Raises
        ------
        ImportError
            If the module cannot be imported.
        AttributeError
            If the attribute does not exist in the module.
        """
        import importlib

        module_path, _, attr_name = path_str.rpartition(".")
        if not module_path:
            raise ImportError(
                f"Cannot resolve '{path_str}' as a dotted import path "
                "(expected 'module.attr' format)"
            )
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    def detect_and_load_data(
        self,
        mode: Literal["si", "mne", None] = "si",
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        extract_func: Union[
            Callable[..., "si.BaseRecording"], Callable[..., mne.io.Raw], str
        ] = None,
        **kwargs,
    ):
        """Load in recording based on mode."""
        if mode == "si":
            if si is None:
                raise ImportError("SpikeInterface is required for mode='si'")

            if isinstance(extract_func, str):
                func_name = extract_func
                # Try SpikeInterface namespaces first
                extract_func = getattr(se, func_name, getattr(si, func_name, None))
                # Fall back to dotted import path (e.g. "mypackage.readers.read_custom")
                if extract_func is None and "." in func_name:
                    extract_func = self._resolve_dotted_path(func_name)
                if extract_func is None:
                    raise ValueError(
                        f"Could not resolve extractor function: {func_name}. "
                        "Provide a SpikeInterface extractor name or a dotted import path."
                    )
            elif extract_func is None:
                extract_func = si.load_extractor

            self.convert_file_with_si_to_recording(
                extract_func=extract_func,
                cache_policy=cache_policy,
                multiprocess_mode=multiprocess_mode,
                **kwargs,
            )
        elif mode == "mne":
            if isinstance(extract_func, str):
                func_name = extract_func
                extract_func = getattr(mne.io, func_name, None)
                # Fall back to dotted import path
                if extract_func is None and "." in func_name:
                    extract_func = self._resolve_dotted_path(func_name)
                if extract_func is None:
                    raise ValueError(
                        f"Could not resolve extractor function: {func_name}. "
                        "Provide an MNE extractor name or a dotted import path."
                    )

            self.convert_file_with_mne_to_recording(
                extract_func=extract_func,
                cache_policy=cache_policy,
                n_jobs=self.n_jobs,
                **kwargs,
            )
        elif mode is None:
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def convert_file_with_si_to_recording(
        self,
        extract_func: Callable[..., "si.BaseRecording"],
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        **kwargs,
    ):
        from .discovery import MultiFileGroup

        if si is None:
            raise ImportError("SpikeInterface is required")

        # Determine number of files being processed
        if isinstance(self.item, MultiFileGroup):
            n_processed_files = 1  # MultiFileGroup is one recording unit
        elif isinstance(self.item, list):
            n_processed_files = len(self.item)
        else:
            n_processed_files = 1

        self._validate_timestamps_for_mode("si", n_processed_files)

        # Handle different item types
        from .discovery import DiscoveredFile
        if isinstance(self.item, DiscoveredFile):
            # DiscoveredFile: handle both single and multi-file cases
            if self.item.is_multi_file:
                # Multi-file group: pass as-is to extract_func (user's custom reader)
                rec: "si.BaseRecording" = extract_func(self.item, **kwargs)
            else:
                # Single file
                rec: "si.BaseRecording" = extract_func(self.item.path, **kwargs)
        elif isinstance(self.item, list):
            # List of files: concatenate individually using multiprocess_mode
            if multiprocess_mode == "dask":
                if dask is None:
                    raise ImportError("dask is required for multiprocess_mode='dask'")
                logging.info(f"Loading {len(self.item)} files in parallel with dask")
                tasks = [dask.delayed(extract_func)(x, **kwargs) for x in self.item]
                recs = list(dask.compute(*tasks))
            else:
                logging.info(f"Loading {len(self.item)} files serially")
                recs = [extract_func(x, **kwargs) for x in self.item]
            rec = si.concatenate_recordings(recs)
        else:
            # Single file/path
            rec: "si.BaseRecording" = extract_func(self.item, **kwargs)

        self._n_processed_files = n_processed_files
        self.LongRecording = self._apply_resampling(rec)

        dt_end = None
        raw_channel_ids = self.LongRecording.get_channel_ids()
        if len(raw_channel_ids) > 0 and isinstance(
            raw_channel_ids[0], (int, np.integer)
        ):
            logging.warning("Channel IDs are integers. Converting to strings.")
        channel_names = [str(ch) for ch in raw_channel_ids]

        self.meta = RecordingMetadata(
            None,
            n_channels=self.LongRecording.get_num_channels(),
            f_s=self.LongRecording.get_sampling_frequency(),
            dt_end=dt_end,
            channel_names=channel_names,
            V_units="µV",
            mult_to_uV=1.0,
        )
        self.channel_names = self.meta.channel_names

        if not hasattr(self, "file_durations") or not self.file_durations:
            if hasattr(self, "_n_processed_files") and self._n_processed_files > 1:
                avg_duration = (
                    self.LongRecording.get_duration() / self._n_processed_files
                )
                self.file_durations = [avg_duration] * self._n_processed_files
            else:
                self.file_durations = [self.LongRecording.get_duration()]
            self.file_end_datetimes = []

        self.finalize_file_timestamps()
        logging.debug(f"LongRecording created via SI: {self}")

    def _load_and_process_mne_data(
        self,
        extract_func,
        input_type,
        datafolder,
        datafile,
        datafiles,
        n_jobs,
        metadata_to_update=None,
        **kwargs,
    ) -> mne.io.Raw:
        """Helper method to load and process MNE data from various input types."""
        # Load data based on input type
        if input_type == "folder":
            raw: mne.io.Raw = extract_func(datafolder, **kwargs)
        elif input_type == "file":
            raw: mne.io.Raw = extract_func(datafile, **kwargs)
        elif input_type == "files":
            logging.info(f"Running extract_func on {len(datafiles)} files")
            raws: list[mne.io.Raw] = [extract_func(x, **kwargs) for x in datafiles]
            logging.info(f"Concatenating {len(raws)} raws")
            raw: mne.io.Raw = mne.concatenate_raws(raws)
            del raws
        else:
            raise ValueError(f"Invalid input_type: {input_type}")

        logging.info(f"raw.info: {raw.info}")

        # Use user-specified n_jobs for MNE resampling, or default to 1
        effective_n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        logging.info(
            f"Using n_jobs={effective_n_jobs} for MNE resampling (method param: {n_jobs}, instance: {self.n_jobs})"
        )

        # Ensure data is preloaded for parallel processing
        if not raw.preload:
            logging.info("Preloading data")
            raw.load_data()

        # Use optimal resampling method with power-of-2 padding for speed
        original_sfreq = raw.info["sfreq"]
        if original_sfreq != constants.GLOBAL_SAMPLING_RATE:
            logging.info(
                f"Resampling from {original_sfreq} to {constants.GLOBAL_SAMPLING_RATE}"
            )
            raw = raw.resample(
                constants.GLOBAL_SAMPLING_RATE,
                n_jobs=effective_n_jobs,
                npad="auto",
                method="fft",
            )

            # Update metadata to reflect the new sampling rate
            if metadata_to_update is not None:
                metadata_to_update.update_sampling_rate(constants.GLOBAL_SAMPLING_RATE)
        else:
            logging.info(
                f"Sampling frequency already matches {constants.GLOBAL_SAMPLING_RATE} Hz, no resampling needed"
            )

        return raw

    def _load_mne_data_no_resample(self, extract_func, **kwargs) -> "mne.io.Raw":
        """Load MNE data without resampling for unified resampling pipeline.

        This method loads and concatenates MNE data but skips resampling,
        allowing the unified resampling to be applied after intermediate file creation.
        """
        if isinstance(self.item, list):
            logging.info(f"Running extract_func on {len(self.item)} files")
            raws = [extract_func(x, **kwargs) for x in self.item]
            logging.info(f"Concatenating {len(raws)} raws")
            raw = mne.concatenate_raws(raws)
            del raws
        else:
            raw = extract_func(self.item, **kwargs)

        logging.info(f"raw.info: {raw.info}")

        # Ensure data is preloaded
        if not raw.preload:
            logging.info("Preloading data")
            raw.load_data()

        # NOTE: No resampling applied here - will be handled by unified resampling after loading from cache
        logging.info(
            f"Data loaded at original sampling rate ({raw.info['sfreq']} Hz) - resampling will be applied later"
        )

        return raw

    def _get_or_create_intermediate_file(
        self,
        fname,
        source_paths,
        cache_policy,
        intermediate,
        extract_func,
        n_jobs,
        **kwargs,
    ):
        """Get cached intermediate file or create it if needed.

        Returns:
            tuple: (recording, raw_object, metadata) where:
                - recording: SpikeInterface recording object
                - raw_object: MNE Raw object (None if using cache)
                - metadata: DDFBinaryMetadata object
        """
        # Define metadata sidecar file path
        meta_fname = fname.with_suffix(fname.suffix + ".meta.json")

        # Check cache policy and validate cache files
        if cache_policy == "force_regenerate":
            use_cache = False
            logging.info(get_cache_status_message(fname, False))
            logging.info("Cache policy 'force_regenerate': ignoring any existing cache")
        else:
            # Check if both data and metadata cache files exist and are valid
            data_cache_valid = should_use_cache_unified(
                fname, source_paths, cache_policy
            )
            meta_cache_valid = meta_fname.exists() if data_cache_valid else False

            # Handle cache validation based on policy
            if not data_cache_valid or not meta_cache_valid:
                if cache_policy == "always":
                    # 'always' policy: raise error if cache missing/invalid
                    missing_files = []
                    if not data_cache_valid:
                        missing_files.append(f"intermediate file ({fname})")
                    if not meta_cache_valid:
                        missing_files.append(f"metadata sidecar ({meta_fname})")
                    raise FileNotFoundError(
                        f"Cache policy 'always' requires existing cache files, but missing: {', '.join(missing_files)}"
                    )
                elif cache_policy == "auto":
                    # 'auto' policy: log and regenerate if cache missing/invalid
                    if not data_cache_valid:
                        logging.info(
                            f"Intermediate file {fname} missing or outdated, regenerating"
                        )
                    if not meta_cache_valid:
                        logging.info(
                            f"Metadata sidecar {meta_fname} missing, regenerating"
                        )
                    use_cache = False
                else:
                    use_cache = False
            else:
                use_cache = True

            if use_cache:
                logging.info(get_cache_status_message(fname, True))
                logging.info(f"Loading cached metadata from {meta_fname}")

                # Load metadata from sidecar file
                try:
                    metadata = DDFBinaryMetadata.from_json(meta_fname)
                    logging.info(
                        f"Loaded cached metadata: {metadata.n_channels} channels, {metadata.f_s} Hz"
                    )
                except Exception as e:
                    if cache_policy == "always":
                        # 'always' policy: raise error if metadata invalid
                        logging.error(
                            f"Cache policy 'always' requires valid metadata, but failed to load {meta_fname}: {e}"
                        )
                        raise
                    elif cache_policy == "auto":
                        # 'auto' policy: log and regenerate if metadata invalid
                        logging.info(
                            f"Failed to load cached metadata from {meta_fname}: {e}"
                        )
                        logging.info(
                            "Regenerating intermediate files due to invalid metadata"
                        )
                        use_cache = False

        if use_cache:
            # Load cached data file
            if intermediate == "edf":
                logging.info("Reading cached edf file")
                rec = se.read_edf(fname)
                return rec, None, metadata  # No raw object when using cache

            elif intermediate == "bin":
                # Use metadata to reconstruct SpikeInterface parameters
                params = {
                    "sampling_frequency": metadata.f_s,
                    "num_channels": metadata.n_channels,
                    "dtype": "float64",  # We standardize on float64 for cached binary files
                    "gain_to_uV": 1,
                    "offset_to_uV": 0,
                    "time_axis": 0,
                    "is_filtered": False,
                }

                logging.info(f"Reading from cached binary file {fname}")
                rec = se.read_binary(fname, **params)
                return rec, None, metadata  # No raw object when using cache

        else:
            # Generate new intermediate files
            logging.info(get_cache_status_message(fname, False))

            # Create metadata object from raw info BEFORE resampling
            # We need to load one file to get the original metadata
            if isinstance(self.item, list):
                sample_raw = extract_func(self.item[0], **kwargs)
            else:
                sample_raw = extract_func(self.item, **kwargs)

            # Create metadata from the original raw object (before resampling)
            original_info = sample_raw.info

            # Extract unit information from MNE Raw object
            unit_str, mult_to_uv = extract_mne_unit_info(original_info)

            metadata = RecordingMetadata(
                metadata_path=None,
                n_channels=original_info["nchan"],
                f_s=original_info["sfreq"],  # Original sampling rate
                dt_end=None,  # Will be set later by finalize_file_timestamps
                channel_names=original_info["ch_names"],
                V_units=unit_str,
                mult_to_uV=mult_to_uv,
            )
            logging.info(
                f"Created metadata from raw: {metadata.n_channels} channels, {metadata.f_s} Hz"
            )
            if unit_str and mult_to_uv:
                logging.info(
                    f"Extracted unit information: {unit_str} (mult_to_uV = {mult_to_uv})"
                )
            else:
                logging.warning(
                    "No unit information could be extracted from MNE Raw object"
                )

            # Load data without resampling (resampling will be applied after intermediate file loading)
            raw = self._load_mne_data_no_resample(extract_func, **kwargs)

            # Check if channel names in MNE Raw object are in Intan format and convert if necessary
            if any("intan" in ch_name.lower() for ch_name in raw.info["ch_names"]):
                logging.info("Converting Intan channel names to MNE format")
                convert_intan_chname_mne(
                    raw
                )  # REVIEW check that this function is robust

            # Create the intermediate file
            if intermediate == "edf":
                logging.info(f"Exporting raw to {fname}")
                try:
                    mne.export.export_raw(fname, raw=raw, fmt="edf", overwrite=True)
                except ValueError as e:
                    # REVIEW JD to me this appears hardcoded -- will check with EDF files as well
                    if "exceeds maximum field length" in str(e):
                        logging.warning(
                            f"EDF export failed due to signal range: {e}. Retrying with robust physical range."
                        )
                        # Calculate robust range (0.01 - 99.99 percentile) to exclude artifacts
                        data = raw.get_data()
                        # Use data percentiles to define physical range, excluding extreme outliers
                        p_min, p_max = np.percentile(data, [0.01, 99.99])

                        # Helper to ensure float fits in 8 chars (EDF limit)
                        def to_valid_edf_float(val):
                            # Try formatting with decreasing precision
                            for fmt in [".6g", ".5g", ".4g", ".3g", ".2g"]:
                                s = f"{val:{fmt}}"
                                if len(s) <= 8:
                                    return float(s)
                            # Fallback
                            return float(f"{val:.2e}")

                        safe_min = to_valid_edf_float(p_min)
                        safe_max = to_valid_edf_float(p_max)

                        logging.info(
                            f"Using robust physical range: ({safe_min}, {safe_max})"
                        )
                        mne.export.export_raw(
                            fname,
                            raw=raw,
                            fmt="edf",
                            overwrite=True,
                            physical_range=(safe_min, safe_max),
                        )
                    else:
                        raise

                logging.info("Reading edf file")
                rec = se.read_edf(fname)

            elif intermediate == "bin":
                # Get raw info for SpikeInterface parameters
                raw_info = raw.info
                params = {
                    "sampling_frequency": raw_info["sfreq"],
                    "num_channels": raw_info["nchan"],
                    "gain_to_uV": 1,
                    "offset_to_uV": 0,
                    "time_axis": 0,
                    "is_filtered": False,
                }

                logging.info(f"Exporting raw to {fname}")
                data: np.ndarray = raw.get_data()  # (n channels, n samples)
                data = data.T  # (n samples, n channels)
                params["dtype"] = data.dtype
                logging.info(f"Writing to {fname}")
                data.tofile(fname)

                logging.info(f"Reading from {fname}")
                rec = se.read_binary(fname, **params)

            else:
                raise ValueError(f"Invalid intermediate: {intermediate}")

            # Save metadata sidecar file
            logging.info(f"Saving metadata to {meta_fname}")
            metadata.to_json(meta_fname)

            return rec, raw, metadata

    def convert_file_with_mne_to_recording(
        self,
        extract_func: Callable[..., mne.io.Raw],
        intermediate: Literal["edf", "bin"] = "edf",
        intermediate_name=None,
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        n_jobs: int = None,
        **kwargs,
    ):
        if se is None:
            raise ImportError(
                "SpikeInterface is required for convert_file_with_mne_to_recording"
            )

        # Determine number of files and source paths
        if isinstance(self.item, list):
            self._validate_timestamps_for_mode("mne", len(self.item))
            source_paths = self.item
            n_processed_files = len(self.item)
        else:
            self._validate_timestamps_for_mode("mne", 1)
            source_paths = [self.item]
            n_processed_files = 1

        self._n_processed_files = n_processed_files

        # Generate intermediate filename
        base_name = Path(source_paths[0]).stem if source_paths else "mne_recording"
        intermediate_name = (
            f"{base_name}_mne-to-rec"
            if intermediate_name is None
            else intermediate_name
        )

        base_dir = Path(source_paths[0]).parent
        fname = base_dir / f"{intermediate_name}.{intermediate}"

        rec, _, metadata = self._get_or_create_intermediate_file(
            fname=fname,
            source_paths=source_paths,
            cache_policy=cache_policy,
            intermediate=intermediate,
            extract_func=extract_func,
            n_jobs=n_jobs,
            **kwargs,
        )

        self.meta = metadata
        self.channel_names = self.meta.channel_names
        self.LongRecording = self._apply_resampling(rec)

        if not hasattr(self, "file_durations") or not self.file_durations:
            if hasattr(self, "_n_processed_files") and self._n_processed_files > 1:
                avg_duration = (
                    self.LongRecording.get_duration() / self._n_processed_files
                )
                self.file_durations = [avg_duration] * self._n_processed_files
            else:
                self.file_durations = [self.LongRecording.get_duration()]
            self.file_end_datetimes = []

        self.finalize_file_timestamps()

    def cleanup_rec(self):
        try:
            del self.LongRecording
        except AttributeError:
            logging.warning("LongRecording does not exist, probably deleted already")
        for tpath in self.temppaths:
            Path.unlink(tpath)

    def split(
        self,
        groups: dict[str, list[str]],
    ) -> dict[str, "LongRecordingOrganizer"]:
        """
        Split the current recording into multiple in-memory LongRecordingOrganizer objects.

        This creates lightweight LRO wrappers around channel-sliced views of the
        recording. No disk I/O is performed. Use `persist()` on individual results
        to save them to disk if needed.

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
            >>> splits["AnimalA"].persist("/output/AnimalA", format="zarr")
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
            child_lro = LongRecordingOrganizer(
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

    def persist(
        self,
        output_dir: Union[str, Path],
        format: Literal["zarr", "binary"] = "zarr",
        n_jobs: int = 1,
        chunk_duration: str = "1s",
        progress_bar: bool = True,
        **kwargs,
    ) -> Path:
        """
        Save this LRO's recording to disk.

        Delegates to SpikeInterface's save() function.

        Args:
            output_dir (Union[str, Path]): Directory to save the recording.
            format (Literal["zarr", "binary"], optional): Save format. Defaults to "zarr".
            n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
            chunk_duration (str, optional): Chunk duration for processing. Defaults to "1s".
            progress_bar (bool, optional): Show progress bar. Defaults to True.
            **kwargs: Additional arguments passed to SI's save().

        Returns:
            Path: The output directory where the recording was saved.
        """
        if si is None:
            raise ImportError("SpikeInterface is required for persist()")

        if self.LongRecording is None:
            raise ValueError("No recording to persist")

        output_dir = Path(output_dir)

        # Ensure parent directory exists
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # For zarr format, SI appends .zarr suffix to folder name
        actual_output_dir = output_dir
        if format == "zarr" and not str(output_dir).endswith(".zarr"):
            actual_output_dir = output_dir.parent / f"{output_dir.name}.zarr"

        if actual_output_dir.exists():
            logging.warning(f"Overwriting existing folder: {actual_output_dir}")
            import shutil

            shutil.rmtree(actual_output_dir)

        saved_rec = self.LongRecording.save(
            folder=output_dir,
            format=format,
            n_jobs=n_jobs,
            chunk_duration=chunk_duration,
            progress_bar=progress_bar,
            **kwargs,
        )

        # SI returns saved recording; update our path to actual location
        self.base_folder_path = actual_output_dir
        self._is_in_memory = False
        logging.info(f"Persisted recording to {actual_output_dir} (format={format})")

        return actual_output_dir

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

    def get_datetime_fragment(self, fragment_len_s, fragment_idx):
        """
        Get the datetime for a specific fragment using the timestamp mapper.

        Args:
            fragment_len_s (float): Length of each fragment in seconds
            fragment_idx (int): Index of the fragment to get datetime for

        Returns:
            datetime: The datetime corresponding to the start of the fragment

        Raises:
            ValueError: If timestamp mapper is not initialized (only available in 'bin' mode)
        """
        return TimestampMapper(
            self.file_end_datetimes, self.file_durations
        ).get_fragment_timestamp(fragment_idx, fragment_len_s)

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

    def convert_to_mne(self) -> mne.io.RawArray:
        """Convert this LongRecording object to an MNE RawArray.

        Returns:
            mne.io.RawArray: The converted MNE RawArray
        """
        data = self.LongRecording.get_traces(
            return_scaled=True
        )  # This gets data in (n_samples, n_channels) format used by SpikeInterface

        # MNE expects data in Volts (V), but SpikeInterface return_scaled=True returns microvolts (uV)
        # Convert uV to V to prevent huge values that crash MNE export (e.g. to EDF)
        data = data * 1e-6

        data = data.T  # Convert to (n_channels, n_samples) format for MNE

        info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=self.LongRecording.get_sampling_frequency(),
            ch_types="eeg",
        )

        return mne.io.RawArray(data=data, info=info)

    def save_to_edf(self, filename: Union[str, Path], overwrite: bool = False):
        """Save the recording to an EDF file via MNE.

        Args:
            filename (str | Path): Path to save the EDF file to.
            overwrite (bool): Whether to overwrite if file exists.
        """
        raw = self.convert_to_mne()
        mne.export.export_raw(str(filename), raw, fmt="edf", overwrite=overwrite)

    def compute_bad_channels(
        self,
        lof_threshold: float = None,
        limit_memory: bool = True,
        force_recompute: bool = False,
    ):
        """Compute bad channels using LOF analysis with unified score storage.

        Args:
            lof_threshold (float, optional): Threshold for determining bad channels from LOF scores.
                                           If None, only computes/loads scores without setting bad_channel_names.
            limit_memory (bool): Whether to reduce memory usage by decimation and float16.
            force_recompute (bool): Whether to recompute LOF scores even if they exist.
        """
        # Check if LOF scores already exist and are current
        if (
            not force_recompute
            and hasattr(self, "lof_scores")
            and self.lof_scores is not None
        ):
            logging.info("Using existing LOF scores")
        else:
            # Compute new LOF scores
            try:
                scores = self._compute_lof_scores(limit_memory=limit_memory)
                self.lof_scores = scores
                logging.info(f"Computed LOF scores for {len(scores)} channels")
            except Exception as e:
                logging.error(f"Failed to compute LOF scores for recording: {e}")
                raise

        # Apply threshold if provided
        if lof_threshold is not None:
            self.apply_lof_threshold(lof_threshold)

    def _compute_lof_scores(self, limit_memory: bool = True) -> np.ndarray:
        """Compute raw LOF scores for all channels.

        Args:
            limit_memory (bool): Whether to reduce memory usage.

        Returns:
            np.ndarray: LOF scores for each channel.
        """
        try:
            nn = Natural_Neighbor()
            rec = self.LongRecording

            logging.debug(f"Computing LOF scores for {rec.__str__()}")
            rec_np = rec.get_traces(return_scaled=True)  # (n_samples, n_channels)

            if rec_np is None or rec_np.size == 0:
                logging.error(
                    "Failed to get traces from recording - data is None or empty"
                )
                raise ValueError("Recording traces are None or empty")
            logging.debug(f"Got recording shape: {rec_np.shape}")

            if limit_memory:
                rec_np = rec_np.astype(np.float16)
                rec_np = decimate(rec_np, 10, axis=0)
            logging.debug(f"Decimated traces shape: {rec_np.shape}")
            rec_np = rec_np.T  # (n_channels, n_samples)
            logging.debug(f"Transposed traces shape: {rec_np.shape}")

            # Compute the optimal number of neighbors
            nn.read(rec_np)
            n_neighbors = nn.algorithm()
            logging.info(f"Computed n_neighbors for LOF computation: {n_neighbors}")

            # Initialize LocalOutlierFactor
            # lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric="minkowski", p=2)
            # distance_vector = pdist(rec_np, metric="seuclidean")
            distance_vector = pdist(rec_np, metric="euclidean")
            distance_matrix = squareform(distance_vector)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric="precomputed")
            # lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric=pdist, )

            # Compute the outlier scores
            logging.debug("Computing outlier scores")
            del nn
            # lof.fit(rec_np)
            lof.fit(distance_matrix)
            del rec_np
            scores = lof.negative_outlier_factor_ * -1
            logging.info(f"LOF computation successful: {len(scores)} channels")
            logging.debug(f"LOF scores: {scores}")

            return scores

        except Exception as e:
            logging.error(f"Failed to compute LOF scores: {e}")
            logging.error(
                f"Recording info: channels={getattr(self, 'channel_names', 'unknown')}, "
                f"duration={getattr(rec, 'duration', 'unknown') if 'rec' in locals() else 'unknown'}"
            )
            raise

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

    def _validate_manual_time_params(self):
        """Validate that manual time parameters are correctly specified."""
        if self.manual_datetimes is not None:
            if not isinstance(self.manual_datetimes, (datetime, list, tuple)):
                raise ValueError(
                    "manual_datetimes must be a datetime object or list of datetime objects"
                )

    def _validate_timestamps_for_mode(self, mode: str, expected_n_files: int = None):
        """Validate that manual timestamps are provided when required for specific modes.

        Args:
            mode (str): The processing mode ('si', 'mne', or 'bin')
            expected_n_files (int, optional): Expected number of files for validation

        Raises:
            ValueError: If timestamps are required but not provided or if count mismatch
        """
        if mode in ["si", "mne"]:
            if self.manual_datetimes is None:
                import logging

                logging.warning(
                    f"manual_datetimes must be provided for {mode} mode when no CSV metadata is available, falling back to file creation times if possible"
                )

            # If list provided and expected files known, validate length
            if expected_n_files is not None and isinstance(self.manual_datetimes, list):
                if len(self.manual_datetimes) != expected_n_files:
                    raise ValueError(
                        f"manual_datetimes length ({len(self.manual_datetimes)}) must match "
                        f"number of input files ({expected_n_files}) for {mode} mode"
                    )

    def _compute_manual_file_datetimes(
        self, n_files: int, durations: list[float]
    ) -> list[datetime]:
        """Compute file end datetimes based on manual time specifications.

        Args:
            n_files (int): Number of files
            durations (list[float]): Duration of each file in seconds

        Returns:
            list[datetime]: End datetime for each file

        Raises:
            ValueError: If manual_datetimes length doesn't match number of files
        """
        if self.manual_datetimes is None:
            return None

        if isinstance(self.manual_datetimes, list):
            # List of times provided - one per file
            if len(self.manual_datetimes) != n_files:
                raise ValueError(
                    f"manual_datetimes length ({len(self.manual_datetimes)}) must match number of files ({n_files})"
                )

            # Convert start times to end times or vice versa
            if self.datetimes_are_start:
                # Convert start times to end times
                file_end_datetimes = [
                    start_time + timedelta(seconds=duration)
                    for start_time, duration in zip(self.manual_datetimes, durations)
                ]
            else:
                # Use as end times directly
                file_end_datetimes = list(self.manual_datetimes)

            # Check contiguity (warn instead of error)
            self._validate_file_contiguity(file_end_datetimes, durations)

            return file_end_datetimes

        else:
            # Single datetime provided - global start or end time
            if self.datetimes_are_start:
                # Global start time - compute cumulative end times
                current_time = self.manual_datetimes
                file_end_datetimes = []
                for duration in durations:
                    current_time += timedelta(seconds=duration)
                    file_end_datetimes.append(current_time)
                return file_end_datetimes
            else:
                # Global end time - work backwards
                total_duration = sum(durations)
                start_time = self.manual_datetimes - timedelta(seconds=total_duration)
                current_time = start_time
                file_end_datetimes = []
                for duration in durations:
                    current_time += timedelta(seconds=duration)
                    file_end_datetimes.append(current_time)
                return file_end_datetimes

    def _validate_file_contiguity(
        self, file_end_datetimes: list[datetime], durations: list[float]
    ):
        """Check that files are contiguous in time and warn if they're not.

        Args:
            file_end_datetimes (list[datetime]): End datetime for each file
            durations (list[float]): Duration of each file in seconds
        """
        if len(file_end_datetimes) <= 1:
            return  # Single file or no files - nothing to check

        tolerance_seconds = 1.0  # Allow 1 second tolerance for rounding errors

        for i in range(len(file_end_datetimes) - 1):
            # Start time of next file should equal end time of current file
            current_end = file_end_datetimes[i]
            next_start = file_end_datetimes[i + 1] - timedelta(seconds=durations[i + 1])

            gap_seconds = (next_start - current_end).total_seconds()
            if gap_seconds > tolerance_seconds:
                warnings.warn(
                    f"Files may not be contiguous: gap of {gap_seconds:.2f}s between "
                    f"file {i} (ends {current_end}) and file {i + 1} (starts {next_start}). "
                    f"Tolerance is {tolerance_seconds}s."
                )
            elif gap_seconds < -tolerance_seconds:
                warnings.warn(
                    f"Files may overlap: negative gap of {gap_seconds:.2f}s between "
                    f"file {i} (ends {current_end}) and file {i + 1} (starts {next_start}). "
                    f"Tolerance is {tolerance_seconds}s."
                )

    def finalize_file_timestamps(self):
        """Finalize file timestamps using manual times if provided, otherwise validate CSV times."""
        logging.info("Finalizing file timestamps")
        if not hasattr(self, "file_durations") or not self.file_durations:
            return  # No file durations available yet

        manual_file_datetimes = self._compute_manual_file_datetimes(
            len(self.file_durations), self.file_durations
        )

        if manual_file_datetimes is not None:
            self.file_end_datetimes = manual_file_datetimes
            logging.info(
                f"Using manual timestamps: {len(manual_file_datetimes)} file end times specified"
            )
        else:
            # Check if CSV times are sufficient (only for bin mode)
            if hasattr(self, "file_end_datetimes") and self.file_end_datetimes:
                if all(x is None for x in self.file_end_datetimes):
                    raise ValueError(
                        "No dates found in any metadata object and no manual times specified!"
                    )
                logging.info("Using CSV metadata timestamps")
            else:
                # For si/mne modes, manual timestamps are ideally required
                logging.warning(
                    "manual_datetimes must be provided when no CSV metadata is available! Falling back to file creation times if possible."
                )

    def get_date_string(self) -> str:
        """
        Get the string representation of the recording date (Start Time).

        Returns:
            str: Date string in format "%b-%d-%Y" (e.g. "Jan-21-2022").

        Raises:
            ValueError: If no timestamps are available in the recording.
        """
        if not hasattr(self, "file_end_datetimes") or not self.file_end_datetimes:
            raise ValueError("Cannot determine date: No file timestamps available.")

        # Find first valid timestamp
        first_valid_idx = next(
            (i for i, x in enumerate(self.file_end_datetimes) if x is not None), None
        )

        if first_valid_idx is None:
            raise ValueError("Cannot determine date: All file timestamps are None.")

        end_time = self.file_end_datetimes[first_valid_idx]
        duration = self.file_durations[first_valid_idx]

        start_time = end_time - timedelta(seconds=duration)
        return start_time.strftime("%b-%d-%Y")

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

    def _apply_resampling(self, recording: "si.BaseRecording") -> "si.BaseRecording":
        """Apply unified resampling and voltage scaling using SpikeInterface preprocessing.

        This method centralizes all resampling logic across the different data loading pipelines
        (binary, MNE, SI) to use the fast SpikeInterface resampling implementation consistently.

        It also enforces the global data type (constants.GLOBAL_DTYPE) for consistency.

        If the recording has scaleable traces (gain_to_uV and offset_to_uV properties),
        voltage scaling is applied first via ``spre.scale_to_uV``. This bakes the correct
        ADC-to-µV conversion into the data and resets gain/offset to 1.0/0.0, which prevents
        offset bugs from subsequent unsigned-to-signed conversion.

        Args:
            recording (si.BaseRecording): The recording to resample

        Returns:
            si.BaseRecording: The resampled recording with data in µV (if scaleable)

        Raises:
            ImportError: If SpikeInterface preprocessing is not available
        """
        # Guard clause: return early if recording is None or invalid
        if recording is None:
            return recording

        if spre is None:
            raise ImportError("SpikeInterface preprocessing is required for resampling")

        # 0. Apply voltage scaling FIRST for integer-typed recordings with gain/offset.
        # SpikeInterface extractors (e.g., read_intan) store gain_to_uV and offset_to_uV
        # which encode how to convert raw ADC values to microvolts. Applying scale_to_uV
        # bakes this conversion into the data and resets gain/offset to 1.0/0.0.
        # This MUST happen before unsigned_to_signed, because unsigned_to_signed shifts
        # the raw data by 2^(bits-1) without updating offset_to_uV, which would cause
        # the offset to be applied twice when get_traces(return_scaled=True) is called.
        # Only apply for integer dtypes — float recordings are assumed to already be in
        # physical units, matching SpikeInterface's own convention (baserecording.py:356).
        dtype = recording.get_dtype() if hasattr(recording, "get_dtype") else None
        is_integer = False
        if dtype is not None and isinstance(dtype, (str, type, np.dtype)):
            try:
                is_integer = np.dtype(dtype).kind in ("i", "u")
            except TypeError:
                pass

        if (
            is_integer
            and hasattr(recording, "has_scaleable_traces")
            and recording.has_scaleable_traces()
        ):
            logging.info("Applying scale_to_uV to convert raw ADC data to microvolts")
            recording = spre.scale_to_uV(recording)

        # 1. Enforce signed integer if unsigned (existing logic preserved)
        dtype = recording.get_dtype() if hasattr(recording, "get_dtype") else None
        # Handle numpy types, strings. Avoid Mock objects
        is_unsigned = False
        if dtype is not None and isinstance(dtype, (str, type, np.dtype)):
            try:
                if np.dtype(dtype).kind == "u":
                    is_unsigned = True
            except TypeError:
                pass

        if is_unsigned:
            logging.info(
                f"Data type is unsigned ({dtype}) and SpikeInterface can't process. Converting it to signed"
            )
            recording = spre.unsigned_to_signed(recording)

        # 2. Enforce GLOBAL_DTYPE (New logic)
        current_dtype = (
            recording.get_dtype() if hasattr(recording, "get_dtype") else None
        )
        if current_dtype is not None and current_dtype != constants.GLOBAL_DTYPE:
            logging.info(
                f"Converting recording dtype from {current_dtype} to {constants.GLOBAL_DTYPE}"
            )
            recording = spre.astype(recording, dtype=constants.GLOBAL_DTYPE)

        # 3. Apply Resampling if needed
        current_rate = (
            recording.get_sampling_frequency()
            if hasattr(recording, "get_sampling_frequency")
            else None
        )
        if current_rate is None and hasattr(recording, "info"):
            current_rate = recording.info.get("sfreq", None)

        target_rate = constants.GLOBAL_SAMPLING_RATE

        if current_rate == target_rate or current_rate is None:
            logging.info(
                f"Recording already at target sampling rate ({target_rate} Hz) or unable to determine, no resampling needed"
            )
            return recording

        logging.info(
            f"Resampling recording from {current_rate} Hz to {target_rate} Hz using SpikeInterface"
        )

        # Use SpikeInterface resampling with margin to reduce edge effects
        resampled_recording = spre.resample(
            recording=recording,
            resample_rate=target_rate,
        )

        # Update metadata to reflect new sampling rate
        if hasattr(self, "meta") and self.meta is not None:
            self.meta.update_sampling_rate(target_rate)

        logging.info(f"Successfully resampled recording to {target_rate} Hz")
        return resampled_recording

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

        # Concatenate recordings using SpikeInterface
        logging.info(
            f"Merging LRO {getattr(other_lro, 'item', 'unknown')} into {getattr(self, 'item', 'unknown')}"
        )
        self.LongRecording = si.concatenate_recordings(
            [self.LongRecording, other_lro.LongRecording]
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
        # Check channel names
        if self.channel_names != other_lro.channel_names:
            raise ValueError(
                f"Channel names mismatch: this LRO has {self.channel_names}, other LRO has {other_lro.channel_names}"
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
                self.file_end_datetimes.extend(other_lro.file_end_datetimes)
                self.file_durations.extend(other_lro.file_durations)
            else:
                # If we are merging durations, we must be able to merge timestamps
                # OR we must drop timestamps entirely to avoid mismatch (destructive).
                # better to raise error and let user fix input data.
                raise ValueError(
                    f"Merge failed: 'other_lro' ({getattr(other_lro, 'base_folder_path', 'unknown')}) "
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

    def __repr__(self):
        """Return a detailed string representation for debugging."""
        return self.__str__()
