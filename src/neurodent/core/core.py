import glob
import gzip
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
except Exception:  # pragma: no cover - optional at import time for tests that don't use dask
    dask = None
import mne
import numpy as np
import pandas as pd

try:
    import spikeinterface.core as si
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre
    import spikeinterface.widgets as sw
except Exception:  # pragma: no cover - optional at import time for tests not using spikeinterface
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
)


class DDFBinaryMetadata:
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
        """Initialize DDFBinaryMetadata either from a file path or direct parameters.

        Args:
            metadata_path (str, optional): Path to metadata CSV file. If provided, other parameters are ignored.
            n_channels (int, optional): Number of channels
            f_s (float, optional): Sampling frequency in Hz
            dt_end (datetime, optional): End datetime of recording
            channel_names (list, optional): List of channel names
            V_units (str, optional): Voltage units (e.g., 'µV', 'mV', 'V')
            mult_to_uV (float, optional): Multiplication factor to convert to microvolts
        """
        if metadata_path is not None:
            self._init_from_path(metadata_path)
        else:
            self._init_from_params(n_channels, f_s, dt_end, channel_names, V_units, mult_to_uV)

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
            logging.warning("No LastEdit column provided in metadata. dt_end set to None")

        self.channel_names = self.metadata_df["ProbeInfo"].tolist()

    def _init_from_params(self, n_channels, f_s, dt_end, channel_names, V_units=None, mult_to_uV=None):
        if None in (n_channels, f_s, channel_names):
            raise ValueError("All parameters must be provided when not using metadata_path")

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
        """Convert DDFBinaryMetadata to a dictionary for JSON serialization."""
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
    def from_dict(cls, data: dict) -> "DDFBinaryMetadata":
        """Create DDFBinaryMetadata from a dictionary (from JSON deserialization)."""
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
        """Save DDFBinaryMetadata to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, file_path: Path) -> "DDFBinaryMetadata":
        """Load DDFBinaryMetadata from a JSON file."""
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
        logging.info(f"Updated DDFBinaryMetadata sampling rate from {old_f_s} Hz to {new_f_s} Hz")


def convert_ddfcolbin_to_ddfrowbin(rowdir_path, colbin_path, metadata, save_gzip=True):
    # TODO consider renaming this function to something more descriptive, like convert_colbin_to_rowbin
    # Also don't use the rowdir_path parameter, since this is outside the scope of the function. See utils.convert_colpath_to_rowpath
    assert isinstance(metadata, DDFBinaryMetadata), "Metadata needs to be of type DDFBinaryMetadata"

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
        metadata (DDFBinaryMetadata): Metadata object containing information about the recording

    Returns:
        tuple: A tuple containing:
            - se.BaseRecording: The SpikeInterface Recording object.
            - str or None: Path to temporary file if created, None otherwise.
    """
    if se is None:
        raise ImportError("SpikeInterface is required for convert_ddfrowbin_to_si")
    assert isinstance(metadata, DDFBinaryMetadata), "Metadata needs to be of type DDFBinaryMetadata"

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
        warnings.warn(f"Sampling rate {rec.sampling_frequency} Hz != {constants.GLOBAL_SAMPLING_RATE} Hz. Resampling")
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
        raise ImportError("SpikeInterface is required for _convert_ddfrowbin_to_si_no_resample")
    assert isinstance(metadata, DDFBinaryMetadata), "Metadata needs to be of type DDFBinaryMetadata"

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


class LongRecordingOrganizer:
    def __init__(
        self,
        base_folder_path,
        mode: Literal["bin", "si", "mne", None] = "bin",
        truncate: Union[bool, int] = False,
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        extract_func: Union[Callable[..., "si.BaseRecording"], Callable[..., mne.io.Raw]] = None,
        input_type: Literal["folder", "file", "files"] = "folder",
        file_pattern: str = None,
        manual_datetimes: datetime | list[datetime] = None,
        datetimes_are_start: bool = True,
        n_jobs: int = 1,
        **kwargs,
    ):
        """Construct a long recording from binary files or EDF files.

        Args:
            base_folder_path (str): Path to the base folder containing the data files.
            mode (Literal['bin', 'si', 'mne', None]): Mode to load data in. Defaults to 'bin'.
            truncate (Union[bool, int], optional): If True, truncate data to first 10 files.
                If an integer, truncate data to the first n files. Defaults to False.
            overwrite_rowbins (bool, optional): If True, overwrite existing row-major binary files. Defaults to False.
            multiprocess_mode (Literal['dask', 'serial'], optional): Processing mode for parallel operations. Defaults to 'serial'.
            extract_func (Callable, optional): Function to extract data when using 'si' or 'mne' mode. Required for those modes.
            input_type (Literal['folder', 'file', 'files'], optional): Type of input to load. Defaults to 'folder'.
            file_pattern (str, optional): Pattern to match files when using 'file' or 'files' input type. Defaults to '*'.
            manual_datetimes (datetime | list[datetime], optional): Manual timestamps for the recording.
                For 'bin' mode: if datetime, used as global start/end time; if list, one timestamp per file.
                For 'si'/'mne' modes: if datetime, used as start/end of entire recording; if list, one per input file.
            datetimes_are_start (bool, optional): If True, manual_datetimes are treated as start times.
                If False, treated as end times. Defaults to True.
            n_jobs (int, optional): Number of jobs for MNE resampling operations. Defaults to 1 for safety.
                Set to -1 for automatic parallel detection, or >1 for specific job count.
            **kwargs: Additional arguments passed to the data loading functions.

        Raises:
            ValueError: If no data files are found, if the folder contains mixed file types,
                or if manual time parameters are invalid.
        """

        self.base_folder_path = Path(base_folder_path)

        self.n_truncate = parse_truncate(truncate)
        self.truncate = True if self.n_truncate > 0 else False
        if self.truncate:
            warnings.warn(f"LongRecording will be truncated to the first {self.n_truncate} files")

        # Store manual time parameters for validation
        self.manual_datetimes = manual_datetimes
        self.datetimes_are_start = datetimes_are_start

        # Store n_jobs parameter for MNE operations
        self.n_jobs = n_jobs

        # Validate manual time parameters
        self._validate_manual_time_params()

        # Initialize core attributes
        self.meta = None
        self.channel_names = None
        self.LongRecording = None
        self.temppaths = []
        self.file_durations = []
        self.cumulative_file_durations = []
        self.bad_channel_names = []

        # Load data if mode is specified
        if mode is not None:
            self.detect_and_load_data(
                mode=mode,
                cache_policy=cache_policy,
                multiprocess_mode=multiprocess_mode,
                extract_func=extract_func,
                input_type=input_type,
                file_pattern=file_pattern,
                **kwargs,
            )

    def detect_and_load_data(
        self,
        mode: Literal["bin", "si", "mne", None] = "bin",
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        extract_func: Union[Callable[..., "si.BaseRecording"], Callable[..., mne.io.Raw]] = None,
        input_type: Literal["folder", "file", "files"] = "folder",
        file_pattern: str = None,
        **kwargs,
    ):
        """Load in recording based on mode."""

        if mode == "bin":
            # Binary file pipeline
            self.convert_colbins_rowbins_to_rec(
                cache_policy=cache_policy,
                multiprocess_mode=multiprocess_mode,
            )
        elif mode == "si":
            # EDF file pipeline
            self.convert_file_with_si_to_recording(
                extract_func=extract_func,
                input_type=input_type,
                file_pattern=file_pattern,
                cache_policy=cache_policy,
                **kwargs,
            )
        elif mode == "mne":
            # MNE file pipeline
            self.convert_file_with_mne_to_recording(
                extract_func=extract_func,
                input_type=input_type,
                file_pattern=file_pattern,
                cache_policy=cache_policy,
                n_jobs=self.n_jobs,
                **kwargs,
            )
        elif mode is None:
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def prepare_colbins_rowbins_metas(self):
        self.colbin_folder_path = self.base_folder_path
        self.rowbin_folder_path = self.base_folder_path

        self.__update_colbins_rowbins_metas()
        self.__check_colbins_rowbins_metas_folders_exist()
        self.__check_colbins_rowbins_metas_not_empty()

        self.meta = DDFBinaryMetadata(self.metas[0])
        self.__metadata_objects = [DDFBinaryMetadata(x) for x in self.metas]
        self._validate_metadata_consistency(self.__metadata_objects)

        self.channel_names = self.meta.channel_names

        # Initialize file_end_datetimes from CSV metadata (will be overridden later if manual times provided)
        file_end_datetimes = [x.dt_end for x in self.__metadata_objects]
        if all(x is None for x in file_end_datetimes):
            # If no CSV times available, manual times will be required later
            self.file_end_datetimes = file_end_datetimes
        else:
            self.file_end_datetimes = file_end_datetimes
            logging.info(
                f"CSV metadata timestamps: {len([x for x in file_end_datetimes if x is not None])} of {len(file_end_datetimes)} files have timestamps"
            )

    def _truncate_file_list(
        self, files: list[Union[str, Path]], ref_list: list[Union[str, Path]] = None
    ) -> list[Union[str, Path]]:
        """Unified method to truncate any list of files.

        Args:
            files: List of files to truncate
            ref_list: Optional list of files to maintain relationships between. Only stems will be compared.
        """

        if not ref_list:
            if not self.truncate or len(files) <= self.n_truncate:
                return files

            # Sort and truncate primary files
            truncated = sorted(files)[: self.n_truncate]
            return truncated
        else:
            # Get a subset of files that match with ref_list
            ref_list_stems = [get_file_stem(f) for f in ref_list]
            files = [f for f in files if get_file_stem(f) in ref_list_stems]
            return files

    def __update_colbins_rowbins_metas(self):
        self.colbins = glob.glob(str(self.colbin_folder_path / "*_ColMajor.bin"))
        self.rowbins = glob.glob(str(self.rowbin_folder_path / "*_RowMajor.npy.gz"))
        self.metas = glob.glob(str(self.colbin_folder_path / "*_Meta.csv"))

        self.colbins.sort(key=filepath_to_index)
        self.rowbins.sort(key=filepath_to_index)
        self.metas.sort(key=filepath_to_index)

        logging.debug(
            f"Before prune: {len(self.colbins)} colbins, {len(self.rowbins)} rowbins, {len(self.metas)} metas"
        )
        self.__prune_empty_files()
        logging.debug(f"After prune: {len(self.colbins)} colbins, {len(self.rowbins)} rowbins, {len(self.metas)} metas")
        if len(self.colbins) != len(self.metas):
            logging.warning("Number of column-major and metadata files do not match")

        metadatas = [DDFBinaryMetadata(x) for x in self.metas]
        for meta in metadatas:
            # if metadata file is empty, remove it and the corresponding column-major and row-major files
            if meta.metadata_df.empty:
                searchstr = Path(meta.metadata_path).name.replace("_Meta", "")
                self.colbins = [x for x in self.colbins if searchstr + "_ColMajor.bin" not in x]
                self.rowbins = [x for x in self.rowbins if searchstr + "_RowMajor.npy.gz" not in x]
                self.metas = [x for x in self.metas if searchstr + "_Meta.csv" not in x]

        # if truncate is True, truncate the lists
        if self.truncate:
            self.colbins = self._truncate_file_list(self.colbins)
            self.rowbins = self._truncate_file_list(
                self.rowbins, ref_list=[x.replace("_ColMajor.bin", "_RowMajor.npy.gz") for x in self.colbins]
            )
            self.metas = self._truncate_file_list(
                self.metas, ref_list=[x.replace("_ColMajor.bin", "_Meta.csv") for x in self.colbins]
            )

    def __prune_empty_files(self):
        # if the column-major file is empty, remove the corresponding row-major and metadata files
        colbins = self.colbins.copy()
        for i, e in enumerate(colbins):
            if Path(e).stat().st_size == 0:
                name = Path(e).name.replace("_ColMajor.bin", "")
                logging.debug(f"Removing {name}")
                self.colbins.remove(e)
                self.rowbins = [x for x in self.rowbins if name + "_RowMajor.npy.gz" not in x]
                self.metas = [x for x in self.metas if name + "_Meta.csv" not in x]
        # remove None values
        self.colbins = [x for x in self.colbins if x is not None]
        self.rowbins = [x for x in self.rowbins if x is not None]
        self.metas = [x for x in self.metas if x is not None]

    def __check_colbins_rowbins_metas_folders_exist(self):
        if not self.colbin_folder_path.exists():
            raise FileNotFoundError(f"Column-major binary files folder not found: {self.colbin_folder_path}")
        if not self.rowbin_folder_path.exists():
            logging.warning(f"Row-major binary files folder not found: {self.rowbin_folder_path}")
        if not self.metas:
            raise FileNotFoundError(f"Metadata files folder not found: {self.metas}")

    def __check_colbins_rowbins_metas_not_empty(self):
        if not self.colbins:
            raise ValueError("No column-major binary files found")
        if not self.rowbins:
            warnings.warn("No row-major binary files found. Convert with convert_colbins_to_rowbins()")
        if not self.metas:
            raise ValueError("No metadata files found")

    def _validate_metadata_consistency(self, metadatas: list[DDFBinaryMetadata]):
        meta0 = metadatas[0]
        # attributes = ['f_s', 'n_channels', 'precision', 'V_units', 'channel_names']
        attributes = ["n_channels", "precision", "V_units", "channel_names"]
        for attr in attributes:
            if not all([getattr(meta0, attr) == getattr(x, attr) for x in metadatas]):
                unequal_values = [getattr(x, attr) for x in metadatas if getattr(x, attr) != getattr(meta0, attr)]
                logging.error(
                    f"Inconsistent {attr} values across metadata files: {getattr(meta0, attr)} != {unequal_values}"
                )
                raise ValueError(f"Metadata files inconsistent at attribute {attr}")
        return

    def convert_colbins_rowbins_to_rec(
        self,
        overwrite_rowbins: bool = False,
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
    ):
        self.prepare_colbins_rowbins_metas()
        self.convert_colbins_to_rowbins(overwrite=overwrite_rowbins, multiprocess_mode=multiprocess_mode)
        self.convert_rowbins_to_rec(multiprocess_mode=multiprocess_mode, cache_policy=cache_policy)
        # Now that file_durations are available, finalize timestamps
        self.finalize_file_timestamps()

    def convert_colbins_to_rowbins(self, overwrite=False, multiprocess_mode: Literal["dask", "serial"] = "serial"):
        """
        Convert column-major binary files to row-major binary files, and save them in the rowbin_folder_path.

        Args:
            overwrite (bool, optional): If True, overwrite existing row-major binary files. Defaults to True.
            multiprocess_mode (Literal['dask', 'serial'], optional): If 'dask', use dask to convert the files in parallel.
                If 'serial', convert the files in serial. Defaults to 'serial'.
        """

        # if overwrite, regenerate regardless of existence
        # else, read them (they exist) or make them (they don't exist)
        # there is no error condition, and rowbins will be recreated regardless of choice

        logging.info(f"Converting {len(self.colbins)} column-major binary files to row-major format")
        if overwrite:
            logging.info("Overwrite flag set - regenerating all row-major files")
        else:
            logging.info("Overwrite flag not set - only generating missing row-major files")

        delayed = []
        for i, e in enumerate(self.colbins):
            if convert_colpath_to_rowpath(self.rowbin_folder_path, e, aspath=False) not in self.rowbins or overwrite:
                logging.info(f"Converting {e}")
                match multiprocess_mode:
                    case "dask":
                        delayed.append(
                            dask.delayed(convert_ddfcolbin_to_ddfrowbin)(
                                self.rowbin_folder_path, e, self.meta, save_gzip=True
                            )
                        )
                    case "serial":
                        convert_ddfcolbin_to_ddfrowbin(self.rowbin_folder_path, e, self.meta, save_gzip=True)
                    case _:
                        raise ValueError(f"Invalid multiprocess_mode: {multiprocess_mode}")

        if multiprocess_mode == "dask":
            # Run all conversions in parallel
            dask.compute(*delayed)

        self.__update_colbins_rowbins_metas()

    def convert_rowbins_to_rec(
        self,
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
    ):
        """
        Convert row-major binary files to SpikeInterface Recording structure.

        Args:
            multiprocess_mode (Literal['dask', 'serial'], optional): If 'dask', use dask to convert the files in parallel.
                If 'serial', convert the files in serial. Defaults to 'serial'.
            cache_policy (Literal): Caching policy for intermediate files (default: "auto")
                - "auto": Use cached files if exist and newer than sources, regenerate with logging if missing/invalid
                - "always": Use cached files if exist, raise error if missing/invalid
                - "force_regenerate": Always regenerate files, overwrite existing cache
        """
        if si is None:
            raise ImportError("SpikeInterface is required for convert_rowbins_to_rec")
        if len(self.rowbins) < len(self.colbins):
            warnings.warn(
                f"{len(self.colbins)} column-major files found, but only {len(self.rowbins)} row-major files found. Some column-major files may be missing."
            )
        elif len(self.rowbins) > len(self.colbins):
            warnings.warn(
                f"{len(self.rowbins)} row-major files found, but only {len(self.colbins)} column-major files found. Some row-major files will be ignored."
            )

        recs = []
        t_cumulative = 0
        self.temppaths = []

        match multiprocess_mode:
            case "dask":
                # Compute all conversions in parallel
                delayed_results = []
                for i, e in enumerate(self.rowbins):
                    delayed_results.append((i, dask.delayed(_convert_ddfrowbin_to_si_no_resample)(e, self.meta)))
                computed_results = dask.compute(*delayed_results)

                # Reconstruct results in the correct order
                results = [None] * len(self.rowbins)
                for i, result in computed_results:
                    results[i] = result
                logging.info(f"self.rowbins: {[Path(x).name for x in self.rowbins]}")

            case "serial":
                results = [_convert_ddfrowbin_to_si_no_resample(e, self.meta) for e in self.rowbins]
            case _:
                raise ValueError(f"Invalid multiprocess_mode: {multiprocess_mode}")

        # Process results
        for i, (rec, temppath) in enumerate(results):
            recs.append(rec)
            self.temppaths.append(temppath)

            duration = rec.get_duration()
            self.file_durations.append(duration)

            t_cumulative += duration  # NOTE  use numpy cumsum later
            self.cumulative_file_durations.append(t_cumulative)

        if not recs:
            raise ValueError("No recordings generated. Check that all row-major files are present and readable.")
        elif len(recs) < len(self.rowbins):
            logging.warning(f"Only {len(recs)} recordings generated. Some row-major files may be missing.")

        # Concatenate recordings first
        concatenated_recording = si.concatenate_recordings(recs).rename_channels(self.channel_names)

        # Apply unified resampling to the concatenated recording
        self.LongRecording: "si.BaseRecording" = self._apply_resampling(concatenated_recording)

        # Debug logging for critical recording features
        logging.info(f"LongRecording created: {self}")

    def convert_file_with_si_to_recording(
        self,
        extract_func: Callable[..., "si.BaseRecording"],
        input_type: Literal["folder", "file", "files"] = "folder",
        file_pattern: str = "*",
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        **kwargs,
    ):
        """Create a SpikeInterface Recording from a folder, a single file, or multiple files.

        This is a thin wrapper around ``extract_func`` that discovers inputs under
        ``self.base_folder_path`` and builds a ``si.BaseRecording`` accordingly.

        Modes:

        - ``folder``: Passes ``self.base_folder_path`` directly to ``extract_func``.
        - ``file``: Uses ``glob`` with ``file_pattern`` relative to ``self.base_folder_path``.
          If multiple matches are found, the first match is used and a warning is issued.
        - ``files``: Uses ``Path.glob`` with ``file_pattern`` under ``self.base_folder_path``,
          optionally truncates via ``self._truncate_file_list(...)``, sorts the files, applies
          ``extract_func`` to each file, and concatenates the resulting recordings via
          ``si.concatenate_recordings``.

        Args:
            extract_func (Callable[..., "si.BaseRecording"]): Function that consumes a path
                (folder or file path) and returns a ``si.BaseRecording``.
            input_type (Literal['folder', 'file', 'files'], optional): How to discover inputs.
                Defaults to ``'folder'``.
            file_pattern (str, optional): Glob pattern used when ``input_type`` is ``'file'`` or
                ``'files'``. Defaults to ``'*'``.
            **kwargs: Additional keyword arguments forwarded to ``extract_func``.

        Side Effects:
            Sets ``self.LongRecording`` to the resulting recording and initializes ``self.meta``
            based on that recording's properties.

        Raises:
            ValueError: If no files are found for the given ``file_pattern`` or ``input_type`` is invalid.
        """
        if si is None:
            raise ImportError("SpikeInterface is required for convert_file_with_si_to_recording")
        # Early validation and file discovery
        if input_type == "folder":
            # For single folder, validate that timestamps are provided
            self._validate_timestamps_for_mode("si", 1)
            datafolder = self.base_folder_path
            rec: "si.BaseRecording" = extract_func(datafolder, **kwargs)
            n_processed_files = 1
        elif input_type == "file":
            # For single file, validate that timestamps are provided
            self._validate_timestamps_for_mode("si", 1)
            datafiles = glob.glob(str(self.base_folder_path / file_pattern))
            if len(datafiles) == 0:
                raise ValueError(f"No files found matching pattern: {file_pattern}")
            elif len(datafiles) > 1:
                warnings.warn(f"Multiple files found matching pattern: {file_pattern}. Using first file.")
            datafile = datafiles[0]
            rec: "si.BaseRecording" = extract_func(datafile, **kwargs)
            n_processed_files = 1
        elif input_type == "files":
            datafiles = [str(x) for x in self.base_folder_path.glob(file_pattern)]
            if len(datafiles) == 0:
                raise ValueError(f"No files found matching pattern: {file_pattern}")
            datafiles = self._truncate_file_list(datafiles)
            # Validate timestamps early before slow processing
            self._validate_timestamps_for_mode("si", len(datafiles))
            datafiles.sort()  # FIXME sort by index, or some other logic. Files may be out of order otherwise, messing up isday calculation
            recs: list["si.BaseRecording"] = [extract_func(x, **kwargs) for x in datafiles]
            rec = si.concatenate_recordings(recs)
            n_processed_files = len(datafiles)
        else:
            raise ValueError(f"Invalid input_type: {input_type}")

        # Store number of processed files for timestamp handling
        self._n_processed_files = n_processed_files

        # Apply unified resampling to the recording
        self.LongRecording = self._apply_resampling(rec)

        # For SI mode, don't use confusing DEFAULT_DAY if we have manual timestamps
        dt_end = None if self.manual_datetimes is not None else None  # Will be set by finalize_file_timestamps

        self.meta = DDFBinaryMetadata(
            None,
            n_channels=self.LongRecording.get_num_channels(),
            f_s=self.LongRecording.get_sampling_frequency(),
            dt_end=dt_end,  # Will be properly set by finalize_file_timestamps
            channel_names=self.LongRecording.get_channel_ids().tolist(),  # NOTE may potentially be a list of integers, which is undesirable. The ability to set names is available in the extractor function itself
            # In the case this is integers, raise a warning and/or error, convert to string, and make a note that you may need to adjust parameters in si extractor
        )
        self.channel_names = self.meta.channel_names

        # For si mode, handle multiple files or single file
        if not hasattr(self, "file_durations") or not self.file_durations:
            if hasattr(self, "_n_processed_files") and self._n_processed_files > 1:
                # Multiple files concatenated - estimate equal durations
                total_duration = self.LongRecording.get_duration()
                avg_duration = total_duration / self._n_processed_files
                self.file_durations = [avg_duration] * self._n_processed_files
            else:
                # Single file or folder
                self.file_durations = [self.LongRecording.get_duration()]
            self.file_end_datetimes = []

        # Apply manual timestamps if provided
        self.finalize_file_timestamps()

        # Debug logging for critical recording features
        logging.debug(f"LongRecording created via SI: {self}")

    def _load_and_process_mne_data(
        self, extract_func, input_type, datafolder, datafile, datafiles, n_jobs, metadata_to_update=None, **kwargs
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
            logging.info(f"Resampling from {original_sfreq} to {constants.GLOBAL_SAMPLING_RATE}")
            raw = raw.resample(constants.GLOBAL_SAMPLING_RATE, n_jobs=effective_n_jobs, npad="auto", method="fft")

            # Update metadata to reflect the new sampling rate
            if metadata_to_update is not None:
                metadata_to_update.update_sampling_rate(constants.GLOBAL_SAMPLING_RATE)
        else:
            logging.info(
                f"Sampling frequency already matches {constants.GLOBAL_SAMPLING_RATE} Hz, no resampling needed"
            )

        return raw

    def _load_mne_data_no_resample(
        self, extract_func, input_type, datafolder, datafile, datafiles, **kwargs
    ) -> mne.io.Raw:
        """Load MNE data without resampling for unified resampling pipeline.

        This method loads and concatenates MNE data but skips resampling,
        allowing the unified resampling to be applied after intermediate file creation.
        """
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
        input_type,
        datafolder,
        datafile,
        datafiles,
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
            data_cache_valid = should_use_cache_unified(fname, source_paths, cache_policy)
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
                        logging.info(f"Intermediate file {fname} missing or outdated, regenerating")
                    if not meta_cache_valid:
                        logging.info(f"Metadata sidecar {meta_fname} missing, regenerating")
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
                    logging.info(f"Loaded cached metadata: {metadata.n_channels} channels, {metadata.f_s} Hz")
                except Exception as e:
                    if cache_policy == "always":
                        # 'always' policy: raise error if metadata invalid
                        logging.error(
                            f"Cache policy 'always' requires valid metadata, but failed to load {meta_fname}: {e}"
                        )
                        raise
                    elif cache_policy == "auto":
                        # 'auto' policy: log and regenerate if metadata invalid
                        logging.info(f"Failed to load cached metadata from {meta_fname}: {e}")
                        logging.info("Regenerating intermediate files due to invalid metadata")
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
            if input_type == "folder":
                sample_raw = extract_func(datafolder, **kwargs)
            elif input_type == "file":
                sample_raw = extract_func(datafile, **kwargs)
            elif input_type == "files":
                sample_raw = extract_func(datafiles[0], **kwargs)
            else:
                raise ValueError(f"Invalid input_type: {input_type}")

            # Create metadata from the original raw object (before resampling)
            original_info = sample_raw.info

            # Extract unit information from MNE Raw object
            unit_str, mult_to_uv = extract_mne_unit_info(original_info)

            metadata = DDFBinaryMetadata(
                metadata_path=None,
                n_channels=original_info["nchan"],
                f_s=original_info["sfreq"],  # Original sampling rate
                dt_end=None,  # Will be set later by finalize_file_timestamps
                channel_names=original_info["ch_names"],
                V_units=unit_str,
                mult_to_uV=mult_to_uv,
            )
            logging.info(f"Created metadata from raw: {metadata.n_channels} channels, {metadata.f_s} Hz")
            if unit_str and mult_to_uv:
                logging.info(f"Extracted unit information: {unit_str} (mult_to_uV = {mult_to_uv})")
            else:
                logging.warning("No unit information could be extracted from MNE Raw object")

            # Load data without resampling (resampling will be applied after intermediate file loading)
            raw = self._load_mne_data_no_resample(extract_func, input_type, datafolder, datafile, datafiles, **kwargs)

            # Create the intermediate file
            if intermediate == "edf":
                logging.info(f"Exporting raw to {fname}")
                mne.export.export_raw(fname, raw=raw, fmt="edf", overwrite=True)

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
        input_type: Literal["folder", "file", "files"] = "folder",
        file_pattern: str = "*",
        intermediate: Literal["edf", "bin"] = "edf",
        intermediate_name=None,
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        n_jobs: int = None,
        **kwargs,
    ):
        """
        Convert MNE-compatible files to SpikeInterface recording format with metadata caching.

        Args:
            extract_func (Callable): Function that takes a file path and returns mne.io.Raw object
            input_type (Literal): Type of input - "folder", "file", or "files"
            file_pattern (str): Glob pattern for file matching (default: "*")
            intermediate (Literal): Intermediate format - "edf" or "bin" (default: "edf")
            intermediate_name (str, optional): Custom name for intermediate file
            cache_policy (Literal): Caching policy for intermediate and metadata files (default: "auto")
                - "auto": Use cached files if both data and metadata exist and cache is newer than sources, regenerate with logging if missing/invalid
                - "always": Use cached files if both data and metadata exist, raise error if missing/invalid
                - "force_regenerate": Always regenerate files, overwrite existing cache
            multiprocess_mode (Literal): Processing mode - "dask" or "serial" (default: "serial")
            n_jobs (int, optional): Number of jobs for MNE resampling. If None (default),
                                uses the instance n_jobs value. Set to -1 for automatic parallel
                                detection, or >1 for specific job count.
            **kwargs: Additional arguments passed to extract_func

        Note:
            Creates two cache files: data file (e.g., file.edf) and metadata sidecar (e.g., file.edf.meta.json).
            Both files must exist for cache to be used. Metadata preserves channel names, original
            sampling rates, and other DDFBinaryMetadata fields across cache hits.
        """
        if se is None:
            raise ImportError("SpikeInterface is required for convert_file_with_mne_to_recording")
        # Early validation and file discovery
        if input_type == "folder":
            self._validate_timestamps_for_mode("mne", 1)
            datafolder = self.base_folder_path
            datafile = None
            datafiles = None
            source_paths = [self.base_folder_path]
            n_processed_files = 1

        elif input_type == "file":
            self._validate_timestamps_for_mode("mne", 1)
            datafiles = list(self.base_folder_path.glob(file_pattern))
            if len(datafiles) == 0:
                raise ValueError(f"No files found matching pattern: {file_pattern}")
            elif len(datafiles) > 1:
                warnings.warn(f"Multiple files found matching pattern: {file_pattern}. Using first file.")
            datafile = datafiles[0]
            datafolder = None
            source_paths = [datafile]
            n_processed_files = 1

        elif input_type == "files":
            datafiles = list(self.base_folder_path.glob(file_pattern))
            if len(datafiles) == 0:
                raise ValueError(f"No files found matching pattern: {file_pattern}")
            datafiles = self._truncate_file_list(datafiles)
            self._validate_timestamps_for_mode("mne", len(datafiles))
            datafiles.sort()
            datafolder = None
            datafile = None
            source_paths = datafiles
            n_processed_files = len(datafiles)

        else:
            raise ValueError(f"Invalid input_type: {input_type}")

        # Store number of processed files for timestamp handling
        self._n_processed_files = n_processed_files

        # Determine intermediate file path
        intermediate_name = (
            f"{self.base_folder_path.name}_mne-to-rec" if intermediate_name is None else intermediate_name
        )
        fname = self.base_folder_path / f"{intermediate_name}.{intermediate}"

        # Get or create the intermediate file (this handles caching logic)
        rec, _, metadata = self._get_or_create_intermediate_file(
            fname=fname,
            source_paths=source_paths,
            cache_policy=cache_policy,
            intermediate=intermediate,
            extract_func=extract_func,
            input_type=input_type,
            datafolder=datafolder,
            datafile=datafile,
            datafiles=datafiles,
            n_jobs=n_jobs,
            **kwargs,
        )

        # Set metadata first so resampling can update it
        self.meta = metadata

        # Apply unified resampling to the loaded recording (this will update metadata sampling rate)
        self.LongRecording = self._apply_resampling(rec)

        # Update dt_end for manual timestamps (will be properly set by finalize_file_timestamps)
        if self.manual_datetimes is not None:
            self.meta.dt_end = None  # Will be set by finalize_file_timestamps
        self.channel_names = self.meta.channel_names

        # For mne mode, handle multiple files or single file
        if not hasattr(self, "file_durations") or not self.file_durations:
            if hasattr(self, "_n_processed_files") and self._n_processed_files > 1:
                # Multiple files concatenated - estimate equal durations
                total_duration = self.LongRecording.get_duration()
                avg_duration = total_duration / self._n_processed_files
                self.file_durations = [avg_duration] * self._n_processed_files
            else:
                # Single file or folder
                self.file_durations = [self.LongRecording.get_duration()]
            self.file_end_datetimes = []

        # Apply manual timestamps if provided
        self.finalize_file_timestamps()

        # Debug logging for critical recording features
        logging.debug(f"LongRecording created via MNE: {self}")

    def cleanup_rec(self):
        try:
            del self.LongRecording
        except AttributeError:
            logging.warning("LongRecording does not exist, probably deleted already")
        for tpath in self.temppaths:
            Path.unlink(tpath)

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
        return TimestampMapper(self.file_end_datetimes, self.file_durations).get_fragment_timestamp(
            fragment_idx, fragment_len_s
        )

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
        endidx = min(frag_len_idx * (fragment_idx + 1), self.LongRecording.get_num_frames())
        return startidx, endidx

    def convert_to_mne(self) -> mne.io.RawArray:
        """Convert this LongRecording object to an MNE RawArray.

        Returns:
            mne.io.RawArray: The converted MNE RawArray
        """
        data = self.LongRecording.get_traces(return_scaled=True)  # This gets data in (n_samples, n_channels) format
        data = data.T  # Convert to (n_channels, n_samples) format for MNE

        info = mne.create_info(
            ch_names=self.channel_names, sfreq=self.LongRecording.get_sampling_frequency(), ch_types="eeg"
        )

        return mne.io.RawArray(data=data, info=info)

    def compute_bad_channels(
        self, lof_threshold: float = None, limit_memory: bool = True, force_recompute: bool = False
    ):
        """Compute bad channels using LOF analysis with unified score storage.

        Args:
            lof_threshold (float, optional): Threshold for determining bad channels from LOF scores.
                                           If None, only computes/loads scores without setting bad_channel_names.
            limit_memory (bool): Whether to reduce memory usage by decimation and float16.
            force_recompute (bool): Whether to recompute LOF scores even if they exist.
        """
        # Check if LOF scores already exist and are current
        if not force_recompute and hasattr(self, "lof_scores") and self.lof_scores is not None:
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
                logging.error("Failed to get traces from recording - data is None or empty")
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
            raise ValueError("LOF scores not available. Run compute_bad_channels() first.")

        is_inlier = self.lof_scores < lof_threshold
        self.bad_channel_names = [self.channel_names[i] for i in np.where(~is_inlier)[0]]
        logging.info(f"Applied threshold {lof_threshold}: bad_channel_names = {self.bad_channel_names}")

    def get_lof_scores(self) -> dict:
        """Get LOF scores with channel names.

        Returns:
            dict: Dictionary mapping channel names to LOF scores.
        """
        if not hasattr(self, "lof_scores") or self.lof_scores is None:
            raise ValueError("LOF scores not available. Run compute_bad_channels() first.")

        return dict(zip(self.channel_names, self.lof_scores))

    def _validate_manual_time_params(self):
        """Validate that manual time parameters are correctly specified."""
        if self.manual_datetimes is not None:
            if not isinstance(self.manual_datetimes, (datetime, list, tuple)):
                raise ValueError("manual_datetimes must be a datetime object or list of datetime objects")

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
                raise ValueError(f"manual_datetimes must be provided for {mode} mode when no CSV metadata is available")

            # If list provided and expected files known, validate length
            if expected_n_files is not None and isinstance(self.manual_datetimes, list):
                if len(self.manual_datetimes) != expected_n_files:
                    raise ValueError(
                        f"manual_datetimes length ({len(self.manual_datetimes)}) must match "
                        f"number of input files ({expected_n_files}) for {mode} mode"
                    )

    def _compute_manual_file_datetimes(self, n_files: int, durations: list[float]) -> list[datetime]:
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

    def _validate_file_contiguity(self, file_end_datetimes: list[datetime], durations: list[float]):
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

        manual_file_datetimes = self._compute_manual_file_datetimes(len(self.file_durations), self.file_durations)

        if manual_file_datetimes is not None:
            self.file_end_datetimes = manual_file_datetimes
            logging.info(f"Using manual timestamps: {len(manual_file_datetimes)} file end times specified")
        else:
            # Check if CSV times are sufficient (only for bin mode)
            if hasattr(self, "file_end_datetimes") and self.file_end_datetimes:
                if all(x is None for x in self.file_end_datetimes):
                    raise ValueError("No dates found in any metadata object and no manual times specified!")
                logging.info("Using CSV metadata timestamps")
            else:
                # For si/mne modes, manual timestamps are required
                raise ValueError("manual_datetimes must be provided when no CSV metadata is available!")

    def __str__(self):
        """Return a string representation of critical long recording features."""
        if not hasattr(self, "LongRecording") or self.LongRecording is None:
            return "LongRecordingOrganizer: No recording loaded yet"

        n_channels = self.LongRecording.get_num_channels()
        sampling_freq = self.LongRecording.get_sampling_frequency()
        total_duration = self.LongRecording.get_duration()

        n_files = len(self.file_durations) if hasattr(self, "file_durations") and self.file_durations else 1

        timestamp_info = "No timestamps"
        if hasattr(self, "file_end_datetimes") and self.file_end_datetimes:
            timestamp_coverage = len([x for x in self.file_end_datetimes if x is not None])
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
        """Apply unified resampling using SpikeInterface preprocessing.

        This method centralizes all resampling logic across the different data loading pipelines
        (binary, MNE, SI) to use the fast SpikeInterface resampling implementation consistently.

        Args:
            recording (si.BaseRecording): The recording to resample

        Returns:
            si.BaseRecording: The resampled recording

        Raises:
            ImportError: If SpikeInterface preprocessing is not available
        """
        if spre is None:
            raise ImportError("SpikeInterface preprocessing is required for resampling")

        current_rate = recording.get_sampling_frequency()
        target_rate = constants.GLOBAL_SAMPLING_RATE

        if current_rate == target_rate:
            logging.info(f"Recording already at target sampling rate ({target_rate} Hz), no resampling needed")
            return recording

        logging.info(f"Resampling recording from {current_rate} Hz to {target_rate} Hz using SpikeInterface")

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
        logging.info(f"Merging LRO {other_lro.base_folder_path} into {self.base_folder_path}")
        self.LongRecording = si.concatenate_recordings([self.LongRecording, other_lro.LongRecording])

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
        # Update end time to reflect the merged recording duration
        if hasattr(other_lro.meta, "dt_end") and hasattr(self.meta, "dt_end"):
            self.meta.dt_end = other_lro.meta.dt_end

        # Note: Channel names, sampling rate, etc. should already be validated as identical

    def __repr__(self):
        """Return a detailed string representation for debugging."""
        return self.__str__()
