# Standard library imports
import glob
import gzip
import math
import os
import statistics
import sys
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd
import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.widgets as sw

# Local imports
from .utils import convert_colpath_to_rowpath, convert_units_to_multiplier, filepath_to_index

# Commented out imports
# from probeinterface.plotting import plot_probe_group, plot_probe

#%%

class DDFBinaryMetadata:

    def __init__(self, metadata_path, verbose=False) -> None:
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path)
        self.verbose = verbose
        if verbose > 0:
            print(self.metadata_df)

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
            warnings.warn("No LastEdit column provided in metadata. dt_end set to None")

        self.channel_to_info = self.metadata_df.loc[:, ["BinColumn", "ProbeInfo"]].set_index('BinColumn').T.to_dict('list')
        self.channel_to_info = {k:v[0] for k,v in self.channel_to_info.items()}
        self.id_to_info = {k-1:v for k,v in self.channel_to_info.items()}
        self.entity_to_info = self.metadata_df.loc[:, ["Entity", "ProbeInfo"]].set_index('Entity').T.to_dict('list')
        self.entity_to_info = {k:v[0] for k,v in self.entity_to_info.items()}
        self.channel_names = list(self.channel_to_info.values())

        # TODO read probe geometry information, may be user-defined

    def __getsinglecolval(self, colname):
        vals = self.metadata_df.loc[:, colname]
        if len(np.unique(vals)) > 1:
            warnings.warn(f"Not all {colname}s are equal!")
        if vals.size == 0:
            return None
        return vals.iloc[0]


class _HiddenPrints:
    def __init__(self, silence=True) -> None:
        self.silence = silence

    def __enter__(self):
        if self.silence:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.silence:
            sys.stdout.close()
            sys.stdout = self._original_stdout


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
              "time_axis" : 0,
              "is_filtered" : False}

    # Read either .npy.gz files or .bin files into the recording object
    if ".npy.gz" in str(bin_rowmajor_path):
        temppath = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        print(f"Opening tempfile {temppath}")
        with open(temppath, "wb") as tmp:
            fcomp = gzip.GzipFile(bin_rowmajor_path, "r")
            bin_rowmajor_decomp = np.load(fcomp)
            bin_rowmajor_decomp.tofile(tmp)

            rec = se.read_binary(tmp.name, **params)
    else:
        rec = se.read_binary(bin_rowmajor_path, **params)
        temppath = None

    return rec, temppath


# In[15]:


class LongRecordingOrganizer:
    def __init__(self, base_folder_path, 
                 colbin_folder_path=None,
                 rowbin_folder_path=None,
                 metadata_path=None,
                 truncate=False) -> None:
        
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

        self.colbin_folder_path = self.base_folder_path if colbin_folder_path is None else Path(colbin_folder_path)
        os.makedirs(self.colbin_folder_path, exist_ok=True)
        self.rowbin_folder_path = self.colbin_folder_path if rowbin_folder_path is None else Path(rowbin_folder_path)
        os.makedirs(self.rowbin_folder_path, exist_ok=True)

        self.__update_colbins_rowbins_metas()

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

        metadatas = [DDFBinaryMetadata(x) for x in self.metas]
        for meta in metadatas:
            if meta.metadata_df.empty:
                searchstr = Path(meta.metadata_path).name.replace("_Meta", "")
                self.colbins = [x for x in self.colbins if searchstr not in x]
                self.rowbins = [x for x in self.rowbins if searchstr not in x]
                self.metas = [x for x in self.metas if searchstr not in x]

        if self.truncate:
            self.colbins, self.rowbins, self.metas = self.__truncate_lists(self.colbins, self.rowbins, self.metas)

    def _validate_metadata_consistency(self, metadatas:list[DDFBinaryMetadata]):
        meta0 = metadatas[0]
        attributes = ['f_s', 'n_channels', 'precision', 'V_units', 'channel_names']
        for attr in attributes:
            if not all([getattr(meta0, attr) == getattr(x, attr) for x in metadatas]):
                raise ValueError(f"Metadata files inconsistent at attribute {attr}")
        return

    def convert_colbins_to_rowbins(self, overwrite=True):
        if not overwrite and self.rowbins:
            warnings.warn("Row-major binary files already exist! Skipping existing files")
            # else:
            #     raise FileExistsError("Row-major binary files already exist! overwrite=False")
        for i, e in enumerate(self.colbins):
            if convert_colpath_to_rowpath(self.rowbin_folder_path, e, aspath=False) not in self.rowbins or overwrite:
                print(f"Converting {e}")
                convert_ddfcolbin_to_ddfrowbin(self.rowbin_folder_path, e, self.meta)
        self.__update_colbins_rowbins_metas()

    def convert_rowbins_to_rec(self):
        recs = []
        self.end_relative = []
        t_to_median = 0
        t_cumulative = 0
        self.temppaths = []
        for i, e in enumerate(self.rowbins):
            print(f"Reading {e}")
            rec, temppath = convert_ddfrowbin_to_si(e, self.meta)
            recs.append(rec)
            self.temppaths.append(temppath)

            if i <= self._idx_median_datetime:
                t_to_median += rec.get_duration()
            t_cumulative += rec.get_duration()
            self.end_relative.append(t_cumulative)

        self.LongRecording = si.concatenate_recordings(recs)
        self.start_datetime = self._median_datetime - timedelta(seconds=t_to_median)

    def cleanup_rec(self):
        try:
            del self.LongRecording
        except AttributeError:
            warnings.warn("LongRecording does not exist, probably deleted already")
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

