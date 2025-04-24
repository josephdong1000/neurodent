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
from typing import Union, Literal
import logging

# Third party imports
import numpy as np
import pandas as pd
import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw
import dask

# Local imports
from .utils import convert_colpath_to_rowpath, convert_units_to_multiplier, filepath_to_index, get_temp_directory
from .. import constants

#%%

class DDFBinaryMetadata:

    def __init__(self, metadata_path, verbose=False) -> None:
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path)
        if self.metadata_df.empty: # Handle empty metadata more elegantly?
            raise ValueError(f"Metadata file is empty: {metadata_path}")

        self.verbose = verbose

        self.n_channels = len(self.metadata_df.index)
        self.f_s = self.__getsinglecolval("SampleRate")
        self.V_units = self.__getsinglecolval("Units")
        self.mult_to_uV = convert_units_to_multiplier(self.V_units)
        self.precision = self.__getsinglecolval("Precision")
        self.dt_end: datetime
        self.dt_start: datetime
        if "LastEdit" in self.metadata_df.keys():
            self.dt_end = datetime.fromisoformat(self.__getsinglecolval("LastEdit"))
        else:
            self.dt_end = None
            logging.warning("No LastEdit column provided in metadata. dt_end set to None")

        self.channel_to_info = self.metadata_df.loc[:, ["BinColumn", "ProbeInfo"]].set_index('BinColumn')
        self.channel_to_info = self.channel_to_info.T.to_dict('list')
        self.channel_to_info = {k:v[0] for k,v in self.channel_to_info.items()}
        # self.id_to_info = {k-1:v for k,v in self.channel_to_info.items()}
        # self.entity_to_info = self.metadata_df.loc[:, ["Entity", "ProbeInfo"]].set_index('Entity').T.to_dict('list')
        # self.entity_to_info = {k:v[0] for k,v in self.entity_to_info.items()}
        self.channel_names = list(self.channel_to_info.values())

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


def convert_ddfrowbin_to_si(bin_rowmajor_path, metadata, verbose=False):
    # 1-file .MAT containing entire recording trace
    # Returns as SpikeInterface Recording structure
    assert isinstance(metadata, DDFBinaryMetadata), "Metadata needs to be of type DDFBinaryMetadata"

    bin_rowmajor_path = Path(bin_rowmajor_path)
    params = {"sampling_frequency" : metadata.f_s,
              "dtype" : metadata.precision,
              "num_channels" : metadata.n_channels,
              "gain_to_uV" : metadata.mult_to_uV,
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

    return rec, temppath


# In[15]:


class LongRecordingOrganizer:
    def __init__(self, base_folder_path, 
                 colbin_folder_path=None,
                 rowbin_folder_path=None,
                 metadata_path=None,
                 make_folders=False,
                 truncate: Union[bool, int]=False,
                 verbose: bool=False) -> None:
        """Construct a long recording from binary files.

        Args:
            base_folder_path (str): Path to the base folder containing the binary files.
            colbin_folder_path (str, optional): If not None, overrides base_folder_path for column-major binary files. Defaults to None.
            rowbin_folder_path (str, optional): If not None, overrides base_folder_path for row-major binary files. Defaults to None.
            metadata_path (str, optional): If not None, overrides metadata files at colbin_folder_path. Defaults to None.
            truncate (Union[bool, int], optional): If True, truncate data to first 10 files. 
                If an integer, truncate data to the first n files. Defaults to False.

        Raises:
            ValueError: If no data files are found.
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
        self.verbose = verbose

        self.colbin_folder_path = self.base_folder_path if colbin_folder_path is None else Path(colbin_folder_path)
        if make_folders:
            os.makedirs(self.colbin_folder_path, exist_ok=True)
        self.rowbin_folder_path = self.colbin_folder_path if rowbin_folder_path is None else Path(rowbin_folder_path)
        if make_folders:
            os.makedirs(self.rowbin_folder_path, exist_ok=True)

        self.__update_colbins_rowbins_metas()
        self.__check_colbins_rowbins_metas_folders_exist()
        self.__check_colbins_rowbins_metas_not_empty()

        if metadata_path is not None:
            self.meta = DDFBinaryMetadata(metadata_path)
            self.metadata_objects = [self.meta]
        else:
            self.meta = DDFBinaryMetadata(self.metas[0])
            self.metadata_objects = [DDFBinaryMetadata(x) for x in self.metas]
            self._validate_metadata_consistency(self.metadata_objects)
        self.channel_names = self.meta.channel_names

        dt_ends = [x.dt_end for x in self.metadata_objects]
        if all(x is None for x in dt_ends):
            raise ValueError("No dates found in any metadata object!")
        
        self._median_datetime = statistics.median_low(pd.Series(dt_ends).dropna())
        self._idx_median_datetime = dt_ends.index(self._median_datetime)

    def __truncate_lists(self, colbins, rowbins, metas):
        if len(colbins) > self.n_truncate:
            out_colbins = colbins[:self.n_truncate]
        else:
            out_colbins = colbins

        out_rowbins = []
        out_metas = []
        for i, e in enumerate(rowbins):
            tempcolname = Path(e).name.replace("RowMajor.npy.gz", "ColMajor.bin")
            if str(self.colbin_folder_path / tempcolname) in out_colbins:
                out_rowbins.append(e)
        for i, e in enumerate(metas):
            tempcolname = Path(e).name.replace("Meta.csv", "ColMajor.bin")
            if str(self.colbin_folder_path / tempcolname) in out_colbins:
                out_metas.append(e)

        return out_colbins, out_rowbins, out_metas

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
            self.colbins, self.rowbins, self.metas = self.__truncate_lists(self.colbins, self.rowbins, self.metas)

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
                    delayed_results.append(dask.delayed(convert_ddfrowbin_to_si)(e, self.meta, verbose=self.verbose))

                # Compute all conversions in parallel
                results = dask.compute(*delayed_results)

            case 'serial':
                results = [convert_ddfrowbin_to_si(e, self.meta, verbose=self.verbose) for e in self.rowbins]

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


# %%
