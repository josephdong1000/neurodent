# Standard library imports
import os
import copy
import glob
import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal
import tempfile
import logging

# Third party imports
# import cmasher as cmr
import numpy as np
import pandas as pd
from scipy.stats import zscore
import dask
import dask.dataframe as dd
import dask.array as da
from dask import delayed
from dask.distributed import Client
from tqdm.dask import TqdmCallback
from tqdm import tqdm
import h5py
import spikeinterface as si
import mne
# Local imports
from .. import constants
from .. import core
from ..core import FragmentAnalyzer, get_temp_directory

class AnimalFeatureParser:
    # REVIEW make this a utility function and refactor across codebase
    def _average_feature(self, df:pd.DataFrame, colname:str, weightsname:str|None='duration'):
        column = df[colname]
        if weightsname is None:
            weights = np.ones(column.size)
        else:
            weights = df[weightsname]
        colitem = column.iloc[0]

        match colname:
            case 'rms' | 'ampvar' | 'psdtotal' | 'pcorr' | 'nspike':
                col_agg = np.stack(column, axis=-1)
            case 'psdslope':
                col_agg = np.array([*column.tolist()])
                col_agg = col_agg.transpose(1, 2, 0)
            case 'cohere' | 'psdband' | 'psdfrac':
                col_agg = {k : np.stack([d[k] for d in column], axis=-1) for k in colitem.keys()}
            case 'psd':
                col_agg = np.stack([x[1] for x in column], axis=-1)
                col_agg = (colitem[0], col_agg)
            case _:
                raise TypeError(f"Unrecognized type in column {colname}: {colitem}")

        if type(col_agg) is dict:
            avg = {k:core.nanaverage(v, axis=-1, weights=weights) for k,v in col_agg.items()}
        elif type(col_agg) is tuple:
            if colname == 'psd':
                avg = (col_agg[0], core.nanaverage(col_agg[1], axis=-1, weights=weights))
            else:
                avg = None
        elif np.isnan(col_agg).all():
            avg = np.nan
        else:
            avg = core.nanaverage(col_agg, axis=-1, weights=weights)
        return avg
    

class AnimalOrganizer(AnimalFeatureParser):

    def __init__(self, 
                 base_folder_path, 
                 anim_id: str, 
                 date_format="NN*NN*NNNN", 
                 mode: Literal["nest", "concat", "base", "noday"] = "concat", 
                 assume_from_number=False,
                 skip_days: list[str] = [], 
                 truncate:bool|int=False) -> None:
        
        self.base_folder_path = Path(base_folder_path)
        self.anim_id = anim_id
        self.date_format = date_format.replace("N", "[0-9]")
        self.__readmode = mode
        self.assume_from_number = assume_from_number

        match mode:
            case "nest":
                self.bin_folder_pat = self.base_folder_path / f"*{self.anim_id}*" / f"*{self.date_format}*"
            case "concat":
                self.bin_folder_pat = self.base_folder_path / f"*{self.anim_id}*{self.date_format}*"
            case "base":
                self.bin_folder_pat = self.base_folder_path
            case "noday":
                self.bin_folder_pat = self.base_folder_path / f"*{self.anim_id}*"
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        self.__bin_folders = glob.glob(str(self.bin_folder_pat))
        self.__bin_folders = [x for x in self.__bin_folders if not any(y in x for y in skip_days)]

        if mode == "noday" and len(self.__bin_folders) > 1:
            raise ValueError(f"Animal ID '{self.anim_id}' is not unique, found: {', '.join(self.__bin_folders)}")
        elif len(self.__bin_folders) == 0:
            raise ValueError(f"No files found for animal ID {self.anim_id} and date format {date_format}")
        
        self.long_recordings: list[core.LongRecordingOrganizer] = []
        self.long_analyzers: list[core.LongRecordingAnalyzer] = []
        self.metadatas: list[core.DDFBinaryMetadata] = []
        self.animaldays: list[str] = []
        for e in self.__bin_folders:
            self.long_recordings.append(core.LongRecordingOrganizer(e, truncate=truncate))
            self.metadatas.append(self.long_recordings[-1].meta)
            self.animaldays.append(core.parse_path_to_animalday(e))

        self.__genotypes = [core.parse_path_to_genotype(x) for x in self.animaldays]
        if len(set(self.__genotypes)) > 1:
            raise ValueError(f"Inconsistent genotypes in {self.animaldays}")
        self.genotype = self.__genotypes[0]
        self.__channel_names = [x.channel_names for x in self.long_recordings]
        if len(set([" ".join(x) for x in self.__channel_names])) > 1:
            raise ValueError(f"Inconsistent channel names in {self.__channel_names}")
        self.channel_names = self.__channel_names[0]
        self.__animal_ids = [core.parse_path_to_animal(x) for x in self.animaldays]
        if len(set(self.__animal_ids)) > 1:
            raise ValueError(f"Inconsistent animal IDs in {self.animaldays}")
        self.animal_id = self.__animal_ids[0]

        self.features_df: pd.DataFrame = pd.DataFrame()
        self.features_avg_df: pd.DataFrame = pd.DataFrame()

    def convert_colbins_to_rowbins(self, overwrite=False):
        for lrec in tqdm(self.long_recordings, desc="Converting column bins to row bins"):
            lrec.convert_colbins_to_rowbins(overwrite=overwrite)

    def convert_rowbins_to_rec(self, multiprocess_mode: Literal['dask', 'serial']='serial'):
        for lrec in tqdm(self.long_recordings, desc="Converting row bins to recs"):
            lrec.convert_rowbins_to_rec(multiprocess_mode=multiprocess_mode)

    def cleanup_rec(self):
        for lrec in self.long_recordings:
            lrec.cleanup_rec()

    def compute_windowed_analysis(self, 
                                  features: list[str], 
                                  exclude: list[str]=['nspike'], 
                                  window_s=4, 
                                  multiprocess_mode: Literal['dask', 'serial']='serial', 
                                  **kwargs):
        """Computes windowed analysis of animal recordings. The data is divided into windows (time bins), then features are extracted from each window. The result is
        formatted to a Dataframe and wrapped into a WindowAnalysisResult object.

        Args:
            features (list[str]): List of features to compute. See individual compute_...() functions for output format
            exclude (list[str], optional): List of features to ignore. Will override the features parameter. Defaults to [].
            window_s (int, optional): Length of each window in seconds. Note that some features break with very short window times. Defaults to 4.

        Raises:
            AttributeError: If a feature's compute_...() function was not implemented, this error will be raised.

        Returns:
            window_analysis_result: a WindowAnalysisResult object
        """
        features = _sanitize_feature_request(features, exclude)

        dataframes = []
        for lrec in self.long_recordings: # Iterate over all long recordings
            logging.debug(f"Initializing LongRecordingAnalyzer for {lrec.base_folder_path}")
            lan = core.LongRecordingAnalyzer(lrec, fragment_len_s=window_s)
            if lan.n_fragments == 0:
                logging.warning(f"No fragments found for {lrec.base_folder_path}. Skipping.")
                continue

            logging.debug(f"Processing {lan.n_fragments} fragments")
            miniters = int(lan.n_fragments / 100)
            match multiprocess_mode:
                case 'dask':
                    # Save fragments to tempfile
                    tmppath = os.path.join(get_temp_directory(), f"temp_{os.urandom(24).hex()}.h5")
                    # NOTE the last fragment is not included because it makes the dask array ragged. Maybe have a small statement that adds it back in
                    logging.debug("Converting LongRecording to numpy array")
                    # Pre-allocate array for faster stacking
                    first_fragment = lan.get_fragment_np(0)
                    np_fragments = np.empty((lan.n_fragments - 1,) + first_fragment.shape, dtype=first_fragment.dtype)
                    np_fragments[0] = first_fragment
                    for idx in range(1, lan.n_fragments - 1):
                        np_fragments[idx] = lan.get_fragment_np(idx)

                    logging.debug(f"Caching numpy array with h5py in {tmppath}")
                    with h5py.File(tmppath, 'w', libver='latest') as f:
                        f.create_dataset('fragments', 
                                       data=np_fragments,
                                       chunks=True,  # Let h5py auto-chunk, or specify tuple like (1, -1, -1)
                                       maxshape=None,  # Allow dataset to be resizable
                                    #    compression="gzip",  # Use GZIP compression
                                    #    compression_opts=4,  # Compression level 0-9 (higher = more compression but slower)
                                    #    rdcc_nbytes=10 * 1024 * 1024  # 10MB chunk cache
                                       )  
                    del np_fragments # cleanup memory

                    # This is not parallelized
                    logging.debug("Processing metadata serially")
                    metadatas = [self._process_fragment_metadata(idx, lan, window_s) for idx in range(lan.n_fragments - 1)]

                    # Process fragments in parallel using Dask
                    logging.debug("Processing features in parallel")
                    with h5py.File(tmppath, 'r', libver='latest') as f:
                        np_fragments_reconstruct = da.from_array(f['fragments'], chunks='1 GB')
                        feature_values = [delayed(FragmentAnalyzer._process_fragment_features_dask)(
                            np_fragments_reconstruct[idx], 
                            lan.f_s, 
                            features, 
                            kwargs
                        ) for idx in range(lan.n_fragments - 1)]
                        del np_fragments_reconstruct # cleanup memory
                        feature_values = dask.compute(*feature_values)
                        f.close() # ensure file is closed

                    # Clean up temp file after processing
                    logging.debug("Cleaning up temp file")
                    try:
                        os.remove(tmppath)
                    except (OSError, FileNotFoundError) as e:
                        logging.warning(f"Failed to remove temporary file {tmppath}: {e}")

                    # Combine metadata and feature values
                    logging.debug("Combining metadata and feature values")
                    meta_df = pd.DataFrame(metadatas)
                    feat_df = pd.DataFrame(feature_values)
                    lan_df = pd.concat([meta_df, feat_df], axis=1)

                case _:
                    logging.debug("Processing serially")
                    lan_df = []
                    for idx in tqdm(range(lan.n_fragments), desc='Processing rows', miniters=miniters):
                        lan_df.append(self._process_fragment_serial(idx, features, lan, window_s, kwargs))

            lan_df = pd.DataFrame(lan_df)
            lan_df.sort_values('timestamp', inplace=True)

            self.long_analyzers.append(lan)
            dataframes.append(lan_df)
        
        self.features_df = pd.concat(dataframes)
        self.features_df.reset_index(inplace=True)

        self.window_analysis_result = WindowAnalysisResult(self.features_df, 
                                                           self.animal_id, 
                                                           self.genotype, 
                                                           self.channel_names, 
                                                           self.assume_from_number)

        return self.window_analysis_result
    
    def compute_spike_analysis(self, multiprocess_mode: Literal['dask', 'serial']='serial'):
        """Compute spike sorting on all long recordings and return a list of SpikeAnalysisResult objects

        Args:
            multiprocess_mode (Literal['dask', 'serial']): Whether to use Dask for parallel processing. Defaults to 'serial'.

        Returns:
            spike_analysis_results: list[SpikeAnalysisResult]. Each SpikeAnalysisResult object corresponds to a LongRecording object,
            typically a different day or recording session.
        """
        sars = []
        lrec_sorts = []
        lrec_recs = []
        recs = [lrec.LongRecording for lrec in self.long_recordings]
        for rec in recs:
            if rec.get_total_samples() == 0:
                logging.warning(f"Skipping {rec.__str__()} because it has no samples")
                sortings, recordings = [], []
            else:
                sortings, recordings = core.MountainSortAnalyzer.sort_recording(rec, multiprocess_mode=multiprocess_mode)
            lrec_sorts.append(sortings)
            lrec_recs.append(recordings)

        if multiprocess_mode == 'dask':
            lrec_sorts = dask.compute(*lrec_sorts)

        lrec_sas = [[si.create_sorting_analyzer(sorting, recording, sparse=False) 
                    for sorting, recording in zip(sortings, recordings)] 
                    for sortings, recordings in zip(lrec_sorts, lrec_recs)]
        sars = [SpikeAnalysisResult(result_sas=sas,
                                    result_mne=None,
                                    animal_id=self.animal_id,
                                    genotype=self.genotype,
                                    animal_day=self.animaldays[i],
                                    metadata=self.metadatas[i],
                                    channel_names=self.channel_names,
                                    assume_from_number=self.assume_from_number)
                for i, sas in enumerate(lrec_sas)]
        
        self.spike_analysis_results = sars
        return self.spike_analysis_results


    def _process_fragment_serial(self, idx, features, lan: core.LongRecordingAnalyzer, window_s, kwargs: dict):
        row = self._process_fragment_metadata(idx, lan, window_s)
        row.update(self._process_fragment_features(idx, features, lan, kwargs))
        return row
    
    def _process_fragment_metadata(self, idx, lan: core.LongRecordingAnalyzer, window_s):
        row = {}

        lan_folder = lan.LongRecording.base_folder_path
        row['animalday'] = core.parse_path_to_animalday(lan_folder)
        row['animal'] = core.parse_path_to_animal(lan_folder)
        row['day'] = core.parse_path_to_day(lan_folder)
        row['genotype'] = core.parse_path_to_genotype(lan_folder)
        row['duration'] = lan.LongRecording.get_dur_fragment(window_s, idx)
        row['endfile'] = lan.get_file_end(idx)

        frag_dt = lan.LongRecording.get_datetime_fragment(window_s, idx)
        row['timestamp'] = frag_dt
        row['isday'] = core.is_day(frag_dt)

        return row
    
    def _process_fragment_features(self, idx, features, lan: core.LongRecordingAnalyzer, kwargs: dict):
        row = {}
        for feat in features:
            func = getattr(lan, f"compute_{feat}")
            if callable(func):
                row[feat] = func(idx, **kwargs)
            else:
                raise AttributeError(f"Invalid function {func}")
        return row

def _sanitize_feature_request(features: list[str], exclude: list[str]=[]):
    """
    Sanitizes a list of requested features for WindowAnalysisResult

    Args:
        features (list[str]): List of features to include. If "all", include all features in constants.FEATURES except for exclude.
        exclude (list[str], optional): List of features to exclude. Defaults to [].

    Returns:
        list[str]: Sanitized list of features.
    """
    if features == ["all"]:
        feat = copy.deepcopy(constants.FEATURES)
    elif not features:
        raise ValueError("Features cannot be empty")
    else:
        if not all(f in constants.FEATURES for f in features):
            raise ValueError(f"Available features are: {constants.FEATURES}")
        feat = copy.deepcopy(features)
    if exclude is not None:
        for e in exclude:
            try:
                feat.remove(e)
            except ValueError:
                pass
    return feat


class WindowAnalysisResult(AnimalFeatureParser):
    """
    Wrapper for output of windowed analysis. Has useful features like group-wise and global averaging, filtering, and saving
    """
    def __init__(self, result: pd.DataFrame, animal_id:str=None, genotype:str=None, channel_names:list[str]=None, assume_from_number=False) -> None:
        """
        Args:
            result (pd.DataFrame): Result comes from AnimalOrganizer.compute_windowed_analysis()
            animal_id (str, optional): Identifier for the animal where result was computed from. Defaults to None.
            genotype (str, optional): Genotype of animal. Defaults to None.
            channel_names (list[str], optional): List of channel names. Defaults to None.
            assume_channels (bool, optional): If true, assumes channel names according to AnimalFeatureParser.DEFAULT_CHNUM_TO_NAME. Defaults to False.
        """
        self.result = result
        columns = result.columns
        self.feature_names = [x for x in columns if x in constants.FEATURES]
        self._nonfeat_cols = [x for x in columns if x not in constants.FEATURES]
        animaldays = result.loc[:, "animalday"].unique()
        
        self.animaldays = animaldays
        self.avg_result: pd.DataFrame
        self.animal_id = animal_id
        self.genotype = genotype
        self.channel_names = channel_names
        self.assume_from_number = assume_from_number
        self.channel_abbrevs = [core.parse_chname_to_abbrev(x, assume_from_number=assume_from_number) for x in self.channel_names]

        print(f"Channel names: \t{self.channel_names}")
        print(f"Channel abbreviations: \t{self.channel_abbrevs}")

    def __str__(self) -> str:
        return f"{self.animal_id} {self.genotype} {self.animaldays}"

    def read_sars_spikes(self, sars: list['SpikeAnalysisResult'], read_mode: Literal['sa', 'mne']='sa', inplace=True):
        match read_mode:
            case 'sa':
                spikes_all = []
                for sar in sars: # for each continuous recording session
                    spikes_channel = []
                    for i, sa in enumerate(sar.result_sas): # for each channel
                        spike_times = []
                        for unit in sa.sorting.get_unit_ids(): # Flatten units
                            spike_times.extend(sa.sorting.get_unit_spike_train(unit_id=unit).tolist())
                        spike_times = np.array(spike_times) / sa.sorting.get_sampling_frequency()
                        spikes_channel.append(spike_times)
                    spikes_all.append(spikes_channel)
                return self._read_from_spikes_all(spikes_all, inplace=inplace)
            case 'mne':
                raws = [sar.result_mne for sar in sars]
                return self.read_mnes_spikes(raws, inplace=inplace)
            case _:
                raise ValueError(f"Invalid read_mode: {read_mode}")
        
    def read_mnes_spikes(self, raws: list[mne.io.RawArray], inplace=True):
        spikes_all = []
        for raw in raws:
            # each mne is a contiguous recording session
            events, event_id = mne.events_from_annotations(raw)
            event_id = {k.item(): v for k, v in event_id.items()}

            spikes_channel = []
            for channel in raw.ch_names:
                if channel not in event_id.keys():
                    logging.warning(f"Channel {channel} not found in event_id")
                    spikes_channel.append([])
                    continue
                event_id_channel = event_id[channel]
                spike_times = events[events[:, 2] == event_id_channel, 0]
                spike_times = spike_times / raw.info['sfreq']
                spikes_channel.append(spike_times)
            spikes_all.append(spikes_channel)
        return self._read_from_spikes_all(spikes_all, inplace=inplace)
    
    def _read_from_spikes_all(self, spikes_all: list[list[list[float]]], inplace=True):
        # Each groupby animalday is a recording session
        grouped = self.result.groupby('animalday')
        animaldays = grouped.groups.keys()
        logging.debug(f"Animal days: {animaldays}")
        spike_counts = dict(zip(animaldays, spikes_all))
        spike_counts = grouped.apply(lambda x: _bin_spike_df(x, spikes_channel=spike_counts[x.name]))
        spike_counts: pd.Series = spike_counts.explode()

        if spike_counts.size != self.result.shape[0]:
            logging.warning(f"Spike counts size {spike_counts.size} does not match result size {self.result.shape[0]}")

        result = self.result.copy()
        result['nspike'] = spike_counts.tolist()
        if inplace:
            self.result = result
        return result

    def get_info(self):
        """Returns a formatted string with basic information about the WindowAnalysisResult object"""
        info = []
        info.append(f"feature names: {', '.join(self.feature_names)}")
        info.append(f"animaldays: {', '.join(self.result['animalday'].unique())}")
        info.append(f"animal_id: {self.result['animal'].unique()[0] if 'animal' in self.result.columns else self.animal_id}")
        info.append(f"genotype: {self.result['genotype'].unique()[0] if 'genotype' in self.result.columns else self.genotype}")
        info.append(f"channel_names: {', '.join(self.channel_names) if self.channel_names else 'None'}")
        
        return "\n".join(info)

    def get_result(self, features: list[str], exclude: list[str]=[], allow_missing=False):
        """Get windowed analysis result dataframe, with helpful filters

        Args:
            features (list[str]): List of features to get from result
            exclude (list[str], optional): List of features to exclude from result; will override the features parameter. Defaults to [].
            allow_missing (bool, optional): If True, will return all requested features as columns regardless if they exist in result. Defaults to False.

        Returns:
            result: pd.DataFrame object with features in columns and windows in rows
        """
        features = _sanitize_feature_request(features, exclude)
        if not allow_missing:
            return self.result.loc[:, self._nonfeat_cols + features]
        else:
            return self.result.reindex(columns=self._nonfeat_cols + features)

    def get_groupavg_result(self, features: list[str], exclude: list[str]=[], df: pd.DataFrame = None, groupby="animalday"):
        """Group result and average within groups. Preserves data structure and shape for each feature.

        Args:
            features (list[str]): List of features to get from result
            exclude (list[str], optional): List of features to exclude from result. Will override the features parameter. Defaults to [].
            df (pd.DataFrame, optional): If not None, this function will use this dataframe instead of self.result. Defaults to None.
            groupby (str, optional): Feature or list of features to group by before averaging. Passed to the `by` parameter in pd.DataFrame.groupby(). Defaults to "animalday".

        Returns:
            grouped_result: result grouped by `groupby` and averaged for each group.
        """
        result_grouped, result_validcols = self.__get_groups(features=features, exclude=exclude, df=df, groupby=groupby)
        features = _sanitize_feature_request(features, exclude)

        avg_results = []
        for f in features:
            if f in result_validcols:
                avg_result_col = result_grouped.apply(self._average_feature, f, "duration", include_groups=False)
                avg_result_col.name = f
                avg_results.append(avg_result_col)
            else:
                logging.warning(f"{f} not calculated, skipping")

        return pd.concat(avg_results, axis=1)

    def __get_groups(self, features: list[str], exclude: list[str]=[], df: pd.DataFrame = None, groupby="animalday"):
        features = _sanitize_feature_request(features, exclude)
        result_win = self.result if df is None else df
        return result_win.groupby(groupby), result_win.columns

    def get_grouprows_result(self, features: list[str], exclude: list[str]=[], df: pd.DataFrame = None,
                            multiindex=["animalday", "animal", "genotype"], include=["duration", "endfile"]):
        features = _sanitize_feature_request(features, exclude)
        result_win = self.result if df is None else df
        result_win = result_win.filter(features + multiindex + include)
        return result_win.set_index(multiindex)

    # NOTE add this info to documentation: False = remove, True = keep. Will need to AND the arrays together to get the final list
    def get_filter_rms_range(self, df:pd.DataFrame=None, z_range=3, **kwargs):
        result = df.copy() if df is not None else self.result.copy()
        z_range = abs(z_range)
        np_rms = np.array(result['rms'].tolist())
        np_rms = np.log(np_rms)
        np_rmsz = zscore(np_rms, axis=0, nan_policy='omit')
        np_rms[(np_rmsz > z_range) | (np_rmsz < -z_range)] = np.nan
        result['rms'] = np_rms.tolist()

        out = np.full(np_rms.shape, True)
        out[(np_rmsz > z_range) | (np_rmsz < -z_range)] = False
        return out

    def get_filter_high_rms(self, df:pd.DataFrame=None, max_rms=3000, **kwargs):
        result = df.copy() if df is not None else self.result.copy()
        np_rms = np.array(result['rms'].tolist())
        np_rmsnan = np_rms.copy()
        np_rmsnan[np_rms > max_rms] = np.nan
        result['rms'] = np_rmsnan.tolist()

        out = np.full(np_rms.shape, True)
        out[np_rms > max_rms] = False
        return out

    def get_filter_high_beta(self, df:pd.DataFrame=None, max_beta=0.25, throw_all=True, **kwargs):
        result = df.copy() if df is not None else self.result.copy()
        df_psdband = pd.DataFrame(result['psdband'].tolist()) # REVIEW implement this with psdfrac instead?
        np_beta = np.array(df_psdband['beta'].tolist())
        np_allbands = np.array(df_psdband.values.tolist())
        np_allbands = np_allbands.sum(axis=1)
        np_prop = np_beta / np_allbands

        out = np.full(np_prop.shape, True)
        out[np_prop > max_beta] = False
        out = np.broadcast_to(np.all(out, axis=-1)[:, np.newaxis], out.shape)
        return out

    def get_filter_reject_channels(self, channels: list[str]):
        n_samples = len(self.result)
        n_channels = len(self.channel_abbrevs)
        mask = np.ones((n_samples, n_channels), dtype=bool)
        # Set False for channels to reject
        for ch in channels:
            if ch in self.channel_abbrevs:
                mask[:, self.channel_abbrevs.index(ch)] = False
            else:
                warnings.warn(f"Channel {ch} not found in {self.channel_abbrevs}")
        return mask

    def filter_all(self, df:pd.DataFrame=None,
                   inplace=True, 
                   reject_channels: list[str]=None, 
                   **kwargs):
        filters = [self.get_filter_rms_range, self.get_filter_high_rms, self.get_filter_high_beta]
        filt_bools = []
        # Automatically filter bad windows
        for filt in filters:
            filt_bool = filt(df, **kwargs)
            filt_bools.append(filt_bool)
            logging.info(f"{filt.__name__}:\tfiltered {filt_bool.size - np.count_nonzero(filt_bool)}/{filt_bool.size}")
        # Filter channels manually
        # REVIEW add a function that just filters out bad channels separately?
        if reject_channels is not None:
            filt_bools.append(self.get_filter_reject_channels(reject_channels))
            logging.debug(f"Reject channels: {filt_bools[-1]}")
        # Apply all filters
        filt_bool_all = np.prod(np.stack(filt_bools, axis=-1), axis=-1).astype(bool)
        filtered_result = self._apply_filter(filt_bool_all)
        if inplace:
            del self.result
            self.result = filtered_result
        return WindowAnalysisResult(filtered_result,
                                  self.animal_id,
                                  self.genotype, 
                                  self.channel_names,
                                  self.assume_from_number)

    def _apply_filter(self, filter_tfs:np.ndarray, verbose=True):
        result = self.result.copy()
        filter_tfs = np.array(filter_tfs, dtype=bool) # (M fragments, N channels)
        for feat in constants.FEATURES:
            if feat not in result.columns:
                logging.info(f"Skipping {feat} because it is not in result")
                continue
            if verbose:
                logging.info(f"Filtering {feat}")
            match feat:
                case 'rms' | 'ampvar' | 'psdtotal' | 'nspike':
                    vals = np.array(result[feat].tolist())
                    vals[~filter_tfs] = np.nan
                    result[feat] = vals.tolist()
                case 'psd':
                    coords = np.array([x[0] for x in result[feat].tolist()])
                    vals = np.array([x[1] for x in result[feat].tolist()])
                    mask = np.broadcast_to(filter_tfs[:, np.newaxis, :], vals.shape)
                    vals[~mask] = np.nan
                    outs = [(c, vals[i, :, :]) for i,c in enumerate(coords)]
                    result[feat] = outs
                case 'psdband' | 'psdfrac':
                    vals = pd.DataFrame(result[feat].tolist())
                    for colname in vals.columns:
                        v = np.array(vals[colname].tolist())
                        v[~filter_tfs] = np.nan
                        vals[colname] = v.tolist()
                    result[feat] = vals.to_dict('records')
                case 'psdslope':
                    vals = np.array(result[feat].tolist())
                    mask = np.broadcast_to(filter_tfs[:, :, np.newaxis], vals.shape)
                    vals[~mask] = np.nan
                    # vals = [list(map(tuple, x)) for x in vals.tolist()]
                    result[feat] = vals.tolist()
                case 'cohere':
                    vals = pd.DataFrame(result[feat].tolist())
                    shape = np.array(vals.iloc[:, 0].tolist()).shape
                    mask = np.broadcast_to(filter_tfs[:, :, np.newaxis], shape)
                    for colname in vals.columns:
                        v = np.array(vals[colname].tolist())
                        v[~mask] = np.nan
                        v[~mask.transpose(0, 2, 1)] = np.nan
                        vals[colname] = v.tolist()
                    result[feat] = vals.to_dict('records')
                case 'pcorr':
                    vals = np.array(result[feat].tolist())
                    mask = np.broadcast_to(filter_tfs[:, :, np.newaxis], vals.shape)
                    vals[~mask] = np.nan
                    vals[~mask.transpose(0, 2, 1)] = np.nan
                    result[feat] = vals.tolist()
                case _:
                    raise ValueError(f'Unknown feature to filter {feat}')
        return result
    
    def save_pickle_and_json(self, folder: str | Path, make_folder=True, save_abbrevs_as_chnames=False):
        """Archive window analysis result into the folder specified, as a pickle and json file.

        Args:
            folder (str | Path): Destination folder to save results to
            make_folder (bool, optional): If True, create the folder if it doesn't exist. Defaults to True.
            save_abbrevs_as_chnames (bool, optional): If True, save the channel abbreviations as the channel names in the json file. Defaults to False.
        """
        folder = Path(folder)
        if make_folder:
            folder.mkdir(parents=True, exist_ok=True)

        filebase = str(folder / f"{self.animal_id}-{self.genotype}")

        self.result.to_pickle(filebase + '.pkl')
        json_dict = {
            'animal_id': self.animal_id,
            'genotype': self.genotype,
            'channel_names': self.channel_abbrevs if save_abbrevs_as_chnames else self.channel_names,
            'assume_from_number': False if save_abbrevs_as_chnames else self.assume_from_number
        }

        with open(filebase + ".json", "w") as f:
            json.dump(json_dict, f, indent=2)


    @classmethod
    def load_pickle_and_json(cls, folder_path=None):
        """Load WindowAnalysisResult from folder

        Args:
            folder_path (str, optional): Path of folder containing one .pkl and .json file each. Defaults to None.
            df_pickle_path (str, optional): Path of .pkl file. If this and folder_path are not None, raises an error. Defaults to None.
            json_path (str, optional): Path of .json file. If this and folder_path are not None, raises an error. Defaults to None.

        Raises:
            ValueError: Both df_pickle_path and json_path must be None if folder_path is provided
            ValueError: Expected exactly one pickle and one json file in folder_path

        Returns:
            result: WindowAnalysisResult object
        """
        if folder_path is not None:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                raise ValueError(f"Folder path {folder_path} does not exist")
            
            pkl_files = list(folder_path.glob('*.pkl'))
            json_files = list(folder_path.glob('*.json'))
            
            if len(pkl_files) != 1 or len(json_files) != 1:
                raise ValueError(f"Expected exactly one pickle and one json file in {folder_path}")
                
            df_pickle_path = pkl_files[0]
            json_path = json_files[0]

        with open(df_pickle_path, 'rb') as f:
            data = pd.read_pickle(f)
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        return cls(data, **metadata)
    
def bin_spike_times(spike_times: list[float], fragment_durations: list[float]) -> list[int]:
    """Bin spike times into counts based on fragment durations.
    
    Args:
        spike_times (list[float]): List of spike timestamps in seconds
        fragment_durations (list[float]): List of fragment durations in seconds
    
    Returns:
        list[int]: List of spike counts per fragment
    """
    # Convert fragment durations to bin edges
    bin_edges = np.cumsum([0] + fragment_durations)
    
    # Use numpy's histogram function to count spikes in each bin
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    
    return counts.tolist()

def _bin_spike_df(df: pd.DataFrame, spikes_channel: list[list[float]]) -> np.ndarray:
        """
        Bins spike times into a matrix of shape (n_windows, n_channels), based on duration of each window in df
        """
        durations = df['duration'].tolist()
        out = np.empty((len(durations), len(spikes_channel)))
        for i, spike_times in enumerate(spikes_channel):
            out[:, i] = bin_spike_times(spike_times, durations)
        return out


class SpikeAnalysisResult(AnimalFeatureParser):
    def __init__(self, result_sas: list[si.SortingAnalyzer], 
                 result_mne: mne.io.RawArray=None,
                 animal_id:str=None, 
                 genotype:str=None, 
                 animal_day:str=None,
                 metadata:core.DDFBinaryMetadata=None,
                 channel_names:list[str]=None, 
                 assume_from_number=False) -> None:
        """
        Args:
            result (list[si.SortingAnalyzer]): Result comes from AnimalOrganizer.compute_spike_analysis(). Each SortingAnalyzer is a single channel.
            animal_id (str, optional): Identifier for the animal where result was computed from. Defaults to None.
            genotype (str, optional): Genotype of animal. Defaults to None.
            channel_names (list[str], optional): List of channel names. Defaults to None.
            assume_channels (bool, optional): If true, assumes channel names according to AnimalFeatureParser.DEFAULT_CHNUM_TO_NAME. Defaults to False.
        """
        self.result_sas = result_sas
        self.result_mne = result_mne
        if (result_mne is None) == (result_sas is None):
            raise ValueError("Exactly one of result_sas or result_mne must be provided")
        self.animal_id = animal_id
        self.genotype = genotype
        self.animal_day = animal_day
        self.metadata = metadata
        self.channel_names = channel_names
        self.assume_from_number = assume_from_number
        self.channel_abbrevs = [core.parse_chname_to_abbrev(x, assume_from_number=assume_from_number) for x in self.channel_names]

        logging.info(f"Channel names: \t{self.channel_names}")
        logging.info(f"Channel abbreviations: \t{self.channel_abbrevs}")

    def convert_to_mne(self, chunk_len:float=60, save_raw=True) -> mne.io.RawArray:
        if self.result_mne is None:
            result_mne = SpikeAnalysisResult.convert_sas_to_mne(self.result_sas, chunk_len)
            if save_raw:
                self.result_mne = result_mne
            else:
                return result_mne
        return self.result_mne
            

    def save_fif_and_json(self, folder: str | Path,
                          convert_to_mne=True,
                          make_folder=True, 
                          save_abbrevs_as_chnames=False,
                          overwrite=False):
        """Archive spike analysis result into the folder specified, as a fif and json file.

        Args:
            folder (str | Path): Destination folder to save results to
            convert_to_mne (bool, optional): If True, convert the SortingAnalyzers to a MNE RawArray if self.result_mne is None. Defaults to True.
            make_folder (bool, optional): If True, create the folder if it doesn't exist. Defaults to True.
            save_abbrevs_as_chnames (bool, optional): If True, save the channel abbreviations as the channel names in the json file. Defaults to False.
            overwrite (bool, optional): If True, overwrite the existing files. Defaults to False.
        """
        if self.result_mne is None:
            if convert_to_mne:
                result_mne = self.convert_to_mne(save_raw=True)
                if result_mne is None:
                    warnings.warn("No SortingAnalyzers found, skipping saving")
                    return
            else:
                raise ValueError("No MNE RawArray found, and convert_to_mne is False. Run convert_to_mne() first.")
        else:
            result_mne = self.result_mne
        
        folder = Path(folder)
        if make_folder:
            folder.mkdir(parents=True, exist_ok=True)

        filebase = str(folder / f"{self.animal_id}-{self.genotype}-{self.animal_day}")
        if not overwrite:
            if filebase + '.json' in folder.glob('*.json'):
                raise FileExistsError(f"File {filebase}.json already exists")
            if filebase + '.fif' in folder.glob('*.fif'):
                raise FileExistsError(f"File {filebase}.fif already exists")
        else:
            for f in folder.glob('*'):
                f.unlink()
        result_mne.save(filebase + '-raw.fif', overwrite=overwrite)
        del result_mne

        json_dict = {
            'animal_id': self.animal_id,
            'genotype': self.genotype,
            'animal_day': self.animal_day,
            'metadata': self.metadata.metadata_path,
            'channel_names': self.channel_abbrevs if save_abbrevs_as_chnames else self.channel_names,
            'assume_from_number': False if save_abbrevs_as_chnames else self.assume_from_number
        }
        with open(filebase + ".json", "w") as f:
            json.dump(json_dict, f, indent=2)


    @classmethod
    def load_fif_and_json(cls, folder: str | Path):
        folder = Path(folder)
        if not folder.exists():
            raise ValueError(f"Folder {folder} does not exist")
        
        fif_files = list(folder.glob('*.fif')) # there may be more than 1 fif file
        json_files = list(folder.glob('*.json'))

        if len(json_files) != 1: 
            raise ValueError(f"Expected exactly one json file in {folder}")
        
        fif_path = fif_files[0]
        json_path = json_files[0]

        with open(json_path, 'r') as f:
            data = json.load(f)
        #data['metadata'] = core.DDFBinaryMetadata(data['metadata'])
        data['result_mne'] = mne.io.read_raw_fif(fif_path)
        data['result_sas'] = None
        return cls(**data)


    @staticmethod
    def convert_sas_to_mne(sas: list[si.SortingAnalyzer], chunk_len:float=60) -> mne.io.RawArray:
        """Convert a list of SortingAnalyzers to a MNE RawArray.
        
        Args:
            sas (list[si.SortingAnalyzer]): The list of SortingAnalyzers to convert
            chunk_len (float, optional): The length of the chunks to use for the conversion. Defaults to 60.

        Returns:
            mne.io.RawArray: The converted RawArray, with spikes labeled as annotations
        """
        if len(sas) == 0:
            return None

        # Check that all SortingAnalyzers have the same sampling frequency
        sfreqs = [sa.recording.get_sampling_frequency() for sa in sas]
        if not all(sf == sfreqs[0] for sf in sfreqs):
            raise ValueError(f"All SortingAnalyzers must have the same sampling frequency. Got frequencies: {sfreqs}")
        
        # Preallocate data array
        total_frames = int(sas[0].recording.get_duration() * sfreqs[0])
        n_channels = len(sas)
        data = np.empty((n_channels, total_frames))
        
        # Fill data array one channel at a time
        for i, sa in enumerate(sas):
            logging.debug(f"Converting channel {i} of {n_channels}")
            data[i, :] = SpikeAnalysisResult.convert_sa_to_np(sa, chunk_len)
            
        channel_names = [str(sa.recording.get_channel_ids().item()) for sa in sas]
        logging.debug(f"Data shape: {data.shape}")
        logging.debug(f"Channel names: {channel_names}")
        sfreq = sfreqs[0]

        # Extract spike times for each unit and create annotations
        onset = []
        description = []
        for sa in sas:
            for unit_id in sa.sorting.get_unit_ids():
                spike_train = sa.sorting.get_unit_spike_train(unit_id)
                # Convert to seconds and filter to recording duration
                spike_times = spike_train / sa.sorting.get_sampling_frequency()
                mask = spike_times < sa.recording.get_duration()
                spike_times = spike_times[mask]
                
                # Create annotation for each spike
                onset.extend(spike_times)
                description.extend([sa.recording.get_channel_ids().item()] * len(spike_times)) # collapse all units into 1 spike train
        annotations = mne.Annotations(onset, duration=0, description=description)

        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data=data, info=info)
        raw = raw.set_annotations(annotations)
        return raw


    @staticmethod
    def convert_sa_to_np(sa: si.SortingAnalyzer, chunk_len:float=60) -> np.ndarray:
        """Convert a SortingAnalyzer to an MNE RawArray.
        
        Args:
            sa (si.SortingAnalyzer): The SortingAnalyzer to convert. Must have only 1 channel.
            chunk_len (float, optional): The length of the chunks to use for the conversion. Defaults to 60.
        Returns:
            np.ndarray: The converted traces
        """
        # Check that SortingAnalyzer only has 1 channel
        if len(sa.recording.get_channel_ids()) != 1:
            raise ValueError(f"Expected SortingAnalyzer to have 1 channel, but got {len(sa.recording.get_channel_ids())} channels")
        
        rec = sa.recording
        logging.debug(f"Recording info: {rec}")
        n_chunks = int(rec.get_duration()) // chunk_len

        # Calculate total number of frames
        total_frames = int(rec.get_duration() * rec.get_sampling_frequency())
        traces = np.empty(total_frames)

        for j in range(n_chunks + 1):  # Get all chunks including the last partial one
            start_frame = round(j * chunk_len * rec.get_sampling_frequency())
            if j == n_chunks:  # Last chunk
                end_frame = total_frames
            else:
                end_frame = round((j + 1) * chunk_len * rec.get_sampling_frequency())
            traces[start_frame:end_frame] = rec.get_traces(start_frame=start_frame,
                                                            end_frame=end_frame,
                                                            return_scaled=True).flatten()
        traces *= 1e-6 # convert from uV to V
        return traces
