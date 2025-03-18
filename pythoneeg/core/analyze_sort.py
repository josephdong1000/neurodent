# Standard library imports
import os
import warnings
import tempfile
from pathlib import Path
from typing import Literal
import logging

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import probeinterface as pi
from mountainsort5 import Scheme2SortingParameters, sorting_scheme2
from mountainsort5.util import create_cached_recording
import dask
import dask.distributed

# Local imports
from .utils import _HiddenPrints, get_temp_directory
from .. import constants


class MountainSortAnalyzer:

    @staticmethod
    def sort_recording(recording: si.BaseRecording, 
                       plot_probe=False,
                       multiprocess_mode: Literal['dask', 'serial']='serial'
                       ) -> tuple[list[si.BaseSorting], list[si.BaseRecording]]:
        """Sort a recording using MountainSort.

        Args:
            recording (si.BaseRecording): The recording to sort.
            plot_probe (bool, optional): Whether to plot the probe. Defaults to False.

        Returns:
            list[si.SortingAnalyzer]: A list of independent sorting analyzers, one for each channel.
        """
        logging.debug(f"Sorting recording info: {recording}")
        logging.debug(f"Sorting recording channel names: {recording.get_channel_ids()}")
        
        rec = recording.clone()
        probe = MountainSortAnalyzer._get_dummy_probe(rec) # TODO at some point, we should use a map of probe geometry instead
        rec = rec.set_probe(probe)

        if plot_probe:
            _, ax2 = plt.subplots(1, 1)
            plot_probe(probe, ax=ax2, with_device_index=True, with_contact_id=True)
            plt.show()
        
        # Get recordings for sorting and waveforms
        sort_rec = MountainSortAnalyzer._get_recording_for_sorting(rec)
        wave_rec = MountainSortAnalyzer._get_recording_for_waveforms(rec)
        
        # Split recording into separate channels
        sort_recs = MountainSortAnalyzer._split_recording(sort_rec)
        wave_recs = MountainSortAnalyzer._split_recording(wave_rec)

        # Run sorting
        match multiprocess_mode:
            case 'dask':
                cached_recs = [dask.delayed(MountainSortAnalyzer._cache_recording)(sort_rec) for sort_rec in sort_recs]
                sortings = [dask.delayed(MountainSortAnalyzer._run_sorting)(cached_rec) for cached_rec in cached_recs]
            case 'serial':
                cached_recs = [MountainSortAnalyzer._cache_recording(sort_rec) for sort_rec in sort_recs]
                sortings = [MountainSortAnalyzer._run_sorting(cached_rec) for cached_rec in cached_recs]

        return sortings, wave_recs

    @staticmethod
    def _get_dummy_probe(recording: si.BaseRecording) -> pi.Probe:
        linprobe = pi.generate_linear_probe(recording.get_num_channels(), ypitch=40)
        linprobe.set_device_channel_indices(np.arange(recording.get_num_channels()))
        linprobe.set_contact_ids(recording.get_channel_ids())
        return linprobe
    
    @staticmethod
    def _get_recording_for_sorting(recording: si.BaseRecording) -> si.BaseRecording:
        return MountainSortAnalyzer._apply_preprocessing(recording, constants.SORTING_PARAMS)
    
    @staticmethod
    def _get_recording_for_waveforms(recording: si.BaseRecording) -> si.BaseRecording:
        return MountainSortAnalyzer._apply_preprocessing(recording, constants.WAVEFORM_PARAMS)

    @staticmethod
    def _apply_preprocessing(recording: si.BaseRecording, params: dict) -> si.BaseRecording:
        rec = recording.clone()

        if params['notch_freq']:
            rec = spre.notch_filter(rec, freq=params['notch_freq'], q=100)
        if params['common_ref']:
            rec = spre.common_reference(rec)
        if params['scale']:
            rec = spre.scale(rec, gain=params['scale']) # Scaling for whitening to work properly
        if params['whiten']:
            rec = spre.whiten(rec)
            
        if params['freq_min']:
            rec = spre.highpass_filter(rec, freq_min=params['freq_min'], ftype='bessel')
        if params['freq_max']:
            rec = spre.bandpass_filter(rec, freq_min=0, freq_max=params['freq_max'], ftype='bessel')

        return rec

    @staticmethod
    def _split_recording(recording: si.BaseRecording) -> list[si.BaseRecording]:
        rec_preps = []
        for channel_id in recording.get_channel_ids():
            rec_preps.append(recording.clone().select_channels([channel_id]))
        return rec_preps

    @staticmethod
    def _cache_recording(recording: si.BaseRecording) -> si.BaseRecording:
        temp_dir = get_temp_directory() / os.urandom(24).hex()
        # dask.distributed.print(f"Caching recording to {temp_dir}")
        os.makedirs(temp_dir)
        cached_rec = create_cached_recording(recording.clone(), folder=temp_dir, chunk_duration='60s')
        return cached_rec

    @staticmethod
    def _run_sorting(recording: si.BaseRecording) -> si.BaseSorting:
        # Confusingly, the snippet_T1 and snippet_T2 parameters in MS are in samples, not seconds
        snippet_T1 = constants.SCHEME2_SORTING_PARAMS['snippet_T1']
        snippet_T2 = constants.SCHEME2_SORTING_PARAMS['snippet_T2']
        snippet_T1_samples = round(recording.get_sampling_frequency() * snippet_T1)
        snippet_T2_samples = round(recording.get_sampling_frequency() * snippet_T2)

        sort_params = Scheme2SortingParameters(
            phase1_detect_channel_radius=constants.SCHEME2_SORTING_PARAMS['phase1_detect_channel_radius'],
            detect_channel_radius=constants.SCHEME2_SORTING_PARAMS['detect_channel_radius'],
            snippet_T1=snippet_T1_samples,
            snippet_T2=snippet_T2_samples,
        )

        with _HiddenPrints(): # REVIEW could also dask delay this. Same problem
            sorting = sorting_scheme2(
                recording=recording,
                sorting_parameters=sort_params
            )

        return sorting
