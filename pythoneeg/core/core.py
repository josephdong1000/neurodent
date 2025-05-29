# Standard library imports
import glob
import gzip
import math
import os
import statistics
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Literal, Callable
import logging

# Third party imports
import numpy as np
import pandas as pd
import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw
import dask
import neo
import mne

# Local imports
from .utils import convert_colpath_to_rowpath, convert_units_to_multiplier, filepath_to_index, get_temp_directory
from .. import constants

#%%

class DDFBinaryMetadata:

    def __init__(self, 
                 metadata_path: str | Path | None, 
                 *, 
                 n_channels: int | None = None, 
                 f_s: float | None = None, 
                 dt_end: datetime | None = None, 
                 channel_names: list[str] | None = None) -> None:
        """Initialize DDFBinaryMetadata either from a file path or direct parameters.
        
        Args:
            metadata_path (str, optional): Path to metadata CSV file. If provided, other parameters are ignored.
            n_channels (int, optional): Number of channels
            f_s (float, optional): Sampling frequency in Hz
            dt_end (datetime, optional): End datetime of recording
            channel_names (list, optional): List of channel names
        """
        if metadata_path is not None:
            self._init_from_path(metadata_path)
        else:
            self._init_from_params(n_channels, f_s, dt_end, channel_names)

    def _init_from_path(self, metadata_path):
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path)
        if self.metadata_df.empty:
            raise ValueError(f"Metadata file is empty: {metadata_path}")

        self.n_channels = len(self.metadata_df.index)
        self.f_s = self.__getsinglecolval("SampleRate")
        self.V_units = self.__getsinglecolval("Units")
        self.mult_to_uV = convert_units_to_multiplier(self.V_units)
        self.precision = self.__getsinglecolval("Precision")
        
        if "LastEdit" in self.metadata_df.keys():
            self.dt_end = datetime.fromisoformat(self.__getsinglecolval("LastEdit"))
        else:
            self.dt_end = None
            logging.warning("No LastEdit column provided in metadata. dt_end set to None")

        self.channel_names = self.metadata_df["ProbeInfo"].tolist()

    def _init_from_params(self, n_channels, f_s, dt_end, channel_names):
        if None in (n_channels, f_s, channel_names):
            raise ValueError("All parameters must be provided when not using metadata_path")
            
        self.metadata_path = None
        self.metadata_df = None
        self.n_channels = n_channels
        self.f_s = f_s
        # self.V_units = V_units
        self.V_units = None
        # self.mult_to_uV = convert_units_to_multiplier(V_units)
        self.mult_to_uV = None
        # self.precision = precision
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


def convert_ddfcolbin_to_ddfrowbin(rowdir_path, colbin_path, metadata, save_gzip=True):
    assert isinstance(metadata, DDFBinaryMetadata), "Metadata needs to be of type DDFBinaryMetadata"

    tempbin = np.fromfile(colbin_path, dtype=metadata.precision)
    tempbin = np.reshape(tempbin, (-1, metadata.n_channels), order='F')

    # rowbin_path = Path(colbin_path).parent / f'{Path(colbin_path).stem.replace("ColMajor", "RowMajor")}'
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
    # 1-file .MAT containing entire recording trace
    # Returns as SpikeInterface Recording structure
    assert isinstance(metadata, DDFBinaryMetadata), "Metadata needs to be of type DDFBinaryMetadata"

    bin_rowmajor_path = Path(bin_rowmajor_path)
    params = {"sampling_frequency" : metadata.f_s,
              "dtype" : metadata.precision,
              "num_channels" : metadata.n_channels,
              "gain_to_uV" : metadata.mult_to_uV,
              "offset_to_uV" : 0,
              "time_axis" : 0,
              "is_filtered" : False}

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
                    logging.error(f"Failed to read .npy.gz file: {bin_rowmajor_path}. Try regenerating row-major files.")
                    raise

            rec = se.read_binary(tmp.name, **params)
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temppath):
                os.remove(temppath)
            raise e
    else:
        rec = se.read_binary(bin_rowmajor_path, **params)
        temppath = None

    if rec.sampling_frequency != constants.GLOBAL_SAMPLING_RATE:
        warnings.warn(f"Sampling rate {rec.sampling_frequency} Hz != {constants.GLOBAL_SAMPLING_RATE} Hz. Resampling")
        rec = spre.resample(rec, constants.GLOBAL_SAMPLING_RATE)

    rec = spre.astype(rec, dtype=constants.GLOBAL_DTYPE)

    return rec, temppath


class LongRecordingOrganizer:
    def __init__(self, base_folder_path, 
                 mode: Literal['bin', 'si', 'mne', None]='bin',
                 truncate: Union[bool, int]=False,
                 overwrite_rowbins: bool=False,
                 multiprocess_mode: Literal['dask', 'serial']='serial',
                 extract_func: Union[Callable[..., si.BaseRecording], Callable[..., mne.io.Raw]]=None,
                 input_type: Literal['folder', 'file', 'files']='folder',
                 file_pattern: str=None,
                 **kwargs):
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
            **kwargs: Additional arguments passed to the data loading functions.

        Raises:
            ValueError: If no data files are found or if the folder contains mixed file types.
        """
        
        if type(truncate) is int:
            self.truncate = True
            self.n_truncate = truncate
        elif type(truncate) is bool:
            self.truncate = truncate
            self.n_truncate = 10
        else:
            self.truncate = False
            warnings.warn("Invalid truncate parameter, setting truncate = False")
        if self.truncate:
            warnings.warn(f"truncate = True. Only the first {self.n_truncate} files of each animal will be used")

        self.base_folder_path = Path(base_folder_path)
        
        # Initialize core attributes
        self.meta = None
        self.channel_names = None
        self.LongRecording = None
        self.temppaths = []
        self.start_datetime = None
        self.end_relative = []
        
        # Load data if mode is specified
        if mode is not None:
            self.detect_and_load_data(
                mode=mode,
                overwrite_rowbins=overwrite_rowbins,
                multiprocess_mode=multiprocess_mode,
                extract_func=extract_func,
                input_type=input_type,
                file_pattern=file_pattern,
                **kwargs
            )

    def detect_and_load_data(self, 
            mode: Literal['bin', 'si', 'mne', None]='bin', 
            overwrite_rowbins: bool=False,
            multiprocess_mode: Literal['dask', 'serial']='serial',
            extract_func: Union[Callable[..., si.BaseRecording], Callable[..., mne.io.Raw]]=None,
            input_type: Literal['folder', 'file', 'files']='folder',
            file_pattern: str=None,
            **kwargs):
        """Load in recording based on mode."""

        if mode == 'bin':
            # Binary file pipeline
            self.convert_colbins_rowbins_to_rec(overwrite_rowbins=overwrite_rowbins, multiprocess_mode=multiprocess_mode)
        elif mode == 'si':
            # EDF file pipeline
            self.convert_file_with_si_to_recording(
                extract_func=extract_func,
                input_type=input_type,
                file_pattern=file_pattern,
                **kwargs
            )
        elif mode == 'mne':
            # MNE file pipeline
            self.convert_file_with_mne_to_recording(
                extract_func=extract_func,
                input_type=input_type,
                file_pattern=file_pattern,
                **kwargs
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

        dt_ends = [x.dt_end for x in self.__metadata_objects]
        if all(x is None for x in dt_ends):
            raise ValueError("No dates found in any metadata object!")
        
        self._median_datetime = statistics.median_low(pd.Series(dt_ends).dropna())
        self._idx_median_datetime = dt_ends.index(self._median_datetime)


    # def __truncate_lists(self, colbins, rowbins, metas):
    #     if len(colbins) > self.n_truncate:
    #         out_colbins = colbins[:self.n_truncate]
    #     else:
    #         out_colbins = colbins

    #     out_rowbins = []
    #     out_metas = []
    #     for i, e in enumerate(rowbins):
    #         tempcolname = Path(e).name.replace("RowMajor.npy.gz", "ColMajor.bin")
    #         if str(self.colbin_folder_path / tempcolname) in out_colbins:
    #             out_rowbins.append(e)
    #     for i, e in enumerate(metas):
    #         tempcolname = Path(e).name.replace("Meta.csv", "ColMajor.bin")
    #         if str(self.colbin_folder_path / tempcolname) in out_colbins:
    #             out_metas.append(e)

    #     return out_colbins, out_rowbins, out_metas

    def _truncate_file_list(self, files: list[Union[str, Path]], 
                            ref_list: list[Union[str, Path]] = None) -> list[Union[str, Path]]:
        """Unified method to truncate any list of files.
        
        Args:
            files: List of files to truncate
            ref_list: Optional list of files to maintain relationships between. Only stems will be compared.
        """

        if not ref_list:
            if not self.truncate or len(files) <= self.n_truncate:
                return files
                
            # Sort and truncate primary files
            truncated = sorted(files)[:self.n_truncate]
            return truncated
        else:
            # Get a subset of files that match with ref_list
            ref_list = [Path(f).stem for f in ref_list]
            files = [f for f in files if Path(f).stem in ref_list]
            return files

    def __update_colbins_rowbins_metas(self):
        self.colbins = glob.glob(str(self.colbin_folder_path / "*_ColMajor.bin"))
        self.rowbins = glob.glob(str(self.rowbin_folder_path / "*_RowMajor.npy.gz"))
        self.metas = glob.glob(str(self.colbin_folder_path / "*_Meta.csv"))

        self.colbins.sort(key=filepath_to_index)
        self.rowbins.sort(key=filepath_to_index)
        self.metas.sort(key=filepath_to_index)

        logging.debug(f"Before prune: {len(self.colbins)} colbins, {len(self.rowbins)} rowbins, {len(self.metas)} metas")
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
            self.rowbins = self._truncate_file_list(self.rowbins, 
                                                    ref_list=[x.replace("_ColMajor.bin", "_RowMajor.npy.gz") for x in self.colbins])
            self.metas = self._truncate_file_list(self.metas, 
                                                  ref_list=[x.replace("_ColMajor.bin", "_Meta.csv") for x in self.colbins])

            # self.colbins, self.rowbins, self.metas = self.__truncate_lists(self.colbins, self.rowbins, self.metas)

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

    def _validate_metadata_consistency(self, metadatas:list[DDFBinaryMetadata]):
        meta0 = metadatas[0]
        # attributes = ['f_s', 'n_channels', 'precision', 'V_units', 'channel_names']
        attributes = ['n_channels', 'precision', 'V_units', 'channel_names']
        for attr in attributes:
            if not all([getattr(meta0, attr) == getattr(x, attr) for x in metadatas]):
                unequal_values = [getattr(x, attr) for x in metadatas if getattr(x, attr) != getattr(meta0, attr)]
                logging.error(f"Inconsistent {attr} values across metadata files: {getattr(meta0, attr)} != {unequal_values}")
                raise ValueError(f"Metadata files inconsistent at attribute {attr}")
        return

    def convert_colbins_rowbins_to_rec(self, overwrite_rowbins: bool=False, multiprocess_mode: Literal['dask', 'serial']='serial'):
        self.prepare_colbins_rowbins_metas()
        self.convert_colbins_to_rowbins(overwrite=overwrite_rowbins, multiprocess_mode=multiprocess_mode)
        self.convert_rowbins_to_rec(multiprocess_mode=multiprocess_mode)

    def convert_colbins_to_rowbins(self, overwrite=False, multiprocess_mode: Literal['dask', 'serial']='serial'):
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
                    case 'dask':
                        delayed.append(dask.delayed(convert_ddfcolbin_to_ddfrowbin)(self.rowbin_folder_path, e, self.meta, save_gzip=True))
                    case 'serial':
                        convert_ddfcolbin_to_ddfrowbin(self.rowbin_folder_path, e, self.meta, save_gzip=True)
                    case _:
                        raise ValueError(f"Invalid multiprocess_mode: {multiprocess_mode}")

        if multiprocess_mode == 'dask':
            # Run all conversions in parallel
            dask.compute(*delayed)

        self.__update_colbins_rowbins_metas()

    def convert_rowbins_to_rec(self, multiprocess_mode: Literal['dask', 'serial']='serial'):
        """
        Convert row-major binary files to SpikeInterface Recording structure.

        Args:
            multiprocess_mode (Literal['dask', 'serial'], optional): If 'dask', use dask to convert the files in parallel.
                If 'serial', convert the files in serial. Defaults to 'serial'.
        """
        if len(self.rowbins) < len(self.colbins):
            warnings.warn(f"{len(self.colbins)} column-major files found, but only {len(self.rowbins)} row-major files found. Some column-major files may be missing.")
        elif len(self.rowbins) > len(self.colbins):
            warnings.warn(f"{len(self.rowbins)} row-major files found, but only {len(self.colbins)} column-major files found. Some row-major files will be ignored.")

        recs = []
        self.end_relative = []
        t_to_median = 0
        t_cumulative = 0
        self.temppaths = []

        match multiprocess_mode:
            case 'dask':
                # Create delayed objects for parallel processing
                delayed_results = []
                for i, e in enumerate(self.rowbins):
                    delayed_results.append(dask.delayed(convert_ddfrowbin_to_si)(e, self.meta))

                # Compute all conversions in parallel
                results = dask.compute(*delayed_results)

            case 'serial':
                results = [convert_ddfrowbin_to_si(e, self.meta) for e in self.rowbins]
            case _:
                raise ValueError(f"Invalid multiprocess_mode: {multiprocess_mode}")

        # Process results
        for i, (rec, temppath) in enumerate(results):
            recs.append(rec)
            self.temppaths.append(temppath)

            if i <= self._idx_median_datetime:
                t_to_median += rec.get_duration()
            t_cumulative += rec.get_duration()
            self.end_relative.append(t_cumulative)

        if not recs:
            raise ValueError("No recordings generated. Check that all row-major files are present and readable.")
        elif len(recs) < len(self.rowbins):
            logging.warning(f"Only {len(recs)} recordings generated. Some row-major files may be missing.")

        self.LongRecording: si.BaseRecording = si.concatenate_recordings(recs).rename_channels(self.channel_names)
        self.start_datetime = self._median_datetime - timedelta(seconds=t_to_median)

    def convert_file_with_si_to_recording(self, 
                                          extract_func: Callable[..., si.BaseRecording], 
                                          input_type: Literal['folder', 'file', 'files']='folder', 
                                          file_pattern: str='*', 
                                          **kwargs):
        if input_type == 'folder':
            datafolder = self.base_folder_path
            rec: si.BaseRecording = extract_func(datafolder, **kwargs)
        elif input_type == 'file':
            datafiles = glob.glob(str(self.base_folder_path / file_pattern))
            if len(datafiles) == 0:
                raise ValueError(f"No files found matching pattern: {file_pattern}")
            elif len(datafiles) > 1:
                warnings.warn(f"Multiple files found matching pattern: {file_pattern}. Using first file.")
            datafile = datafiles[0]
            rec: si.BaseRecording = extract_func(datafile, **kwargs)
        elif input_type == 'files':
            datafiles = [str(x) for x in self.base_folder_path.glob(file_pattern)]
            if len(datafiles) == 0:
                raise ValueError(f"No files found matching pattern: {file_pattern}")
            datafiles = self._truncate_file_list(datafiles)
            datafiles.sort() # FIXME sort by index, or some other logic. Files may be out of order otherwise, messing up isday calculation
            recs: list[si.BaseRecording] = [extract_func(x, **kwargs) for x in datafiles]
            rec = si.concatenate_recordings(recs)
        else:
            raise ValueError(f"Invalid mode: {input_type}")

        self.LongRecording = rec
        self.meta = DDFBinaryMetadata(n_channels=self.LongRecording.get_num_channels(),
                                      f_s=self.LongRecording.get_sampling_frequency(),
                                      dt_end=constants.DEFAULT_DAY, # NOTE parse timestamp from SI file date/metadata?
                                      channel_names=self.LongRecording.get_channel_ids().tolist())
        
    def convert_file_with_mne_to_recording(self, 
                                           extract_func: Callable[..., mne.io.Raw], 
                                           input_type: Literal['folder', 'file', 'files']='folder', 
                                           file_pattern: str='*', 
                                           intermediate: Literal['edf', 'bin'] = 'edf',
                                           intermediate_name = None,
                                           overwrite = True,
                                           multiprocess_mode: Literal['dask', 'serial']='serial',
                                           **kwargs):
        
        if input_type == 'folder':
            datafolder = self.base_folder_path
            raw: mne.io.Raw = extract_func(datafolder, **kwargs)
        elif input_type == 'file':
            datafiles = list(self.base_folder_path.glob(file_pattern))
            if len(datafiles) == 0:
                raise ValueError(f"No files found matching pattern: {file_pattern}")
            elif len(datafiles) > 1:
                warnings.warn(f"Multiple files found matching pattern: {file_pattern}. Using first file.")
            datafile = datafiles[0]
            raw: mne.io.Raw = extract_func(datafile, **kwargs)
        elif input_type == 'files':
            datafiles = list(self.base_folder_path.glob(file_pattern))
            if len(datafiles) == 0:
                raise ValueError(f"No files found matching pattern: {file_pattern}")
            datafiles = self._truncate_file_list(datafiles)
            datafiles.sort() # FIXME sort by index, or some other logic. Files may be out of order otherwise, messing up isday calculation
            logging.debug(f"Running extract_func on {len(datafiles)} files")
            raws: list[mne.io.Raw] = [extract_func(x, **kwargs) for x in datafiles]
            logging.debug(f"Concatenating {len(raws)} raws")
            raw: mne.io.Raw = mne.concatenate_raws(raws)
            del raws
        else:
            raise ValueError(f"Invalid mode: {input_type}")
        
        logging.info(raw.info)
        logging.debug(f"Old sampling frequency: {raw.info['sfreq']}")
        raw = raw.resample(constants.GLOBAL_SAMPLING_RATE)
        logging.debug(f"New sampling frequency: {raw.info['sfreq']}")
        
        # Cache raw to binary
        intermediate_name = f"{self.base_folder_path.name}_mne-to-rec" if intermediate_name is None else intermediate_name
        if intermediate == 'edf':
            fname = self.base_folder_path / f"{intermediate_name}.edf"
            logging.debug(f"Exporting raw to {fname}")
            mne.export.export_raw(fname, raw=raw, fmt='edf', overwrite=overwrite) # NOTE this causes a lot of OOM issues
            logging.debug("Reading edf file")
            rec = se.read_edf(fname)
        elif intermediate == 'bin':
            fname = self.base_folder_path / f"{intermediate_name}.bin"
            logging.debug(f"Exporting raw to {fname}")
            data: np.ndarray = raw.get_data() # (n channels, n samples)
            data = data.T # (n samples, n channels)
            logging.debug(f"Writing to {fname}")
            data.tofile(fname)

            params = {
                "sampling_frequency" : raw.info.sfreq,
                "dtype" : data.dtype,
                "num_channels" : raw.info.nchan,
                "gain_to_uV" : 1,
                "offset_to_uV" : 0,
                "time_axis" : 0,
                "is_filtered" : False
            }
            logging.debug(f"Reading from {fname}")
            rec = se.read_binary(fname, **params)

        else:
            raise ValueError(f"Invalid intermediate: {intermediate}")

        self.LongRecording = rec
        self.meta = DDFBinaryMetadata(n_channels=self.LongRecording.get_num_channels(),
                                      f_s=self.LongRecording.get_sampling_frequency(),
                                      dt_end=constants.DEFAULT_DAY, # NOTE parse timestamp from MNE file date/metadata?
                                      channel_names=self.LongRecording.get_channel_ids().tolist())


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
        idx, _ = self.__fragidx_to_startendind(fragment_len_s, fragment_idx)
        return self.start_datetime + timedelta(seconds=self.__idx_to_time(idx))
        
    def __fragidx_to_startendind(self, fragment_len_s, fragment_idx):
        frag_len_idx = self.__time_to_idx(fragment_len_s)
        startidx = frag_len_idx * fragment_idx
        endidx = min(frag_len_idx * (fragment_idx + 1), self.LongRecording.get_num_frames())
        return startidx, endidx
