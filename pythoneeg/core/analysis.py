# Standard library imports
import os
import tempfile
import warnings
from pathlib import Path
from typing import Literal

# Third party imports
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from mne import set_config
from mne.time_frequency import (
    csd_array_fourier
)
from mne_connectivity import (
    envelope_correlation
)
import spikeinterface.core as si

# Local imports
from .. import core
from .. import constants
from .analyze_frag import FragmentAnalyzer

#%%

class LongRecordingAnalyzer:

    # FEATURES = ['rms', 'ampvar', 'psd', 'psdtotal', 'psdband', 'psdslope', 'cohere', 'pcorr']
    # GLOBAL_FEATURES = ['templates']
    # FREQ_BANDS = {'delta' : (0.1, 4),
    #             'theta' : (4, 8),
    #             'alpha' : (8, 13),
    #             'beta'  : (13, 25),
    #             'gamma' : (25, 50)}
    # FREQ_BAND_TOTAL = (0.1, 50)
    # FREQ_MINS = [v[0] for k,v in FREQ_BANDS.items()]
    # FREQ_MAXS = [v[1] for k,v in FREQ_BANDS.items()]
    # FREQ_BAND_NAMES = list(FREQ_BANDS.keys())

    def __init__(self, longrecording, fragment_len_s=10, notch_freq=60) -> None:

        assert isinstance(longrecording, core.LongRecordingOrganizer)

        self.LongRecording = longrecording
        self.fragment_len_s = fragment_len_s
        self.n_fragments = longrecording.get_num_fragments(fragment_len_s)
        self.channel_to_info = longrecording.meta.channel_to_info
        self.channel_names = longrecording.channel_names
        self.n_channels = longrecording.meta.n_channels
        self.V_units = longrecording.meta.V_units
        self.mult_to_uV = longrecording.meta.mult_to_uV
        self.f_s = int(longrecording.meta.f_s)
        self.notch_freq = notch_freq


    def get_fragment_rec(self, index) -> si.BaseRecording:
        """Get window at index as a spikeinterface recording object

        Args:
            index (int): Index of time window

        Returns:
            si.BaseRecording: spikeinterface recording object
        """
        return self.LongRecording.get_fragment(self.fragment_len_s, index)

    def get_fragment_np(self, index, recobj=None) -> np.ndarray:
        """Get window at index as a numpy array object

        Args:
            index (int): Index of time window
            recobj (si.BaseRecording, optional): If not None, uses this recording object to get the numpy array. Defaults to None.

        Returns:
            np.ndarray: Numpy array with dimensions (N, M), N = number of samples, M = number of channels. Values in uV
        """
        assert isinstance(recobj, si.BaseRecording) or recobj is None
        if recobj is None:
            return self.get_fragment_rec(index).get_traces(return_scaled=True) # (num_samples, num_channels), in units uV
        else:
            return recobj.get_traces(return_scaled=True)

    def get_fragment_mne(self, index, recobj=None) -> np.ndarray:
        """Get window at index as a numpy array object, formatted for ease of use with MNE functions

        Args:
            index (int): Index of time window
            recobj (si.BaseRecording, optional): If not None, uses this recording object to get the numpy array. Defaults to None.

        Returns:
            np.ndarray: Numpy array with dimensions (1, M, N), M = number of channels, N = number of samples. 1st dimension corresponds
             to number of epochs, which there is only 1 in a window. Values in uV
        """
        rec = self.get_fragment_np(index, recobj=recobj)[..., np.newaxis]
        return np.transpose(rec, (2, 1, 0)) # (1 epoch, num_channels, num_samples)

    def compute_rms(self, index, **kwargs):
        """Compute average root mean square amplitude

        Args:
            index (int): Index of time window

        Returns:
            result: np.ndarray with shape (1, M), M = number of channels
        """
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_rms(rec=rec, **kwargs)
    
    def compute_ampvar(self, index, **kwargs):
        """Compute average amplitude variance

        Args:
            index (int): Index of time window

        Returns:
            result: np.ndarray with shape (1, M), M = number of channels
        """
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_ampvar(rec=rec, **kwargs)

    def compute_psd(self, index, welch_bin_t=1, notch_filter=True, multitaper=False, **kwargs):
        """Compute PSD (power spectral density)

        Args:
            index (int): Index of time window
            welch_bin_t (float, optional): Length of time bins to use in Welch's method, in seconds. Defaults to 1.
            notch_filter (bool, optional): If True, applies notch filter at line frequency. Defaults to True.
            multitaper (bool, optional): If True, uses multitaper method instead of Welch's method. Defaults to False.

        Returns:
            f (np.ndarray): Array of sample frequencies
            psd (np.ndarray): Array of PSD values at sample frequencies. (X, M), X = number of sample frequencies, M = number of channels.
            If sample window length is too short, PSD is interpolated
        """
        rec = self.get_fragment_np(index)

        f, psd = FragmentAnalyzer.compute_psd(rec=rec, 
                                              f_s=self.f_s, 
                                              welch_bin_t=welch_bin_t, 
                                              notch_filter=notch_filter, 
                                              multitaper=multitaper,
                                              **kwargs)

        if index == self.n_fragments - 1 and self.n_fragments > 1:
            f_prev, _ = self.compute_psd(index - 1, welch_bin_t, notch_filter, multitaper)
            psd = Akima1DInterpolator(f, psd, axis=0, extrapolate=True)(f_prev)
            f = f_prev

        return f, psd
    
    def compute_psdband(self, index, welch_bin_t=1, notch_filter=True, bands: list[tuple[float, float]]=constants.FREQ_BANDS, multitaper=False, **kwargs):
        """Compute power spectral density of the signal for each frequency band.

        Args:
            index (int): Index of time window
            welch_bin_t (float, optional): Length of time bins to use in Welch's method, in seconds. Defaults to 1.
            notch_filter (bool, optional): If True, applies notch filter at line frequency. Defaults to True.
            bands (list[tuple[float, float]], optional): List of frequency bands to compute PSD for. Defaults to constants.FREQ_BANDS.
            multitaper (bool, optional): If True, uses multitaper method instead of Welch's method. Defaults to False.

        Returns:
            dict: Dictionary mapping band names to PSD values for each channel
        """
        
        rec = self.get_fragment_np(index)

        return FragmentAnalyzer.compute_psdband(rec=rec,
                                                f_s=self.f_s,
                                                welch_bin_t=welch_bin_t,
                                                notch_filter=notch_filter,
                                                bands=bands,
                                                multitaper=multitaper,
                                                **kwargs)

    def compute_psdtotal(self, index, welch_bin_t=1, notch_filter=True, band: tuple[float, float]=constants.FREQ_BAND_TOTAL, multitaper=False, **kwargs):
        """Compute total power over PSD (power spectral density) plot within a specified frequency band

        Args:
            index (int): Index of time window
            welch_bin_t (float, optional): Length of time bins to use in Welch's method, in seconds. Defaults to 1.
            notch_filter (bool, optional): If True, applies notch filter at line frequency. Defaults to True.
            band (tuple[float, float], optional): Frequency band to calculate over. Defaults to constants.FREQ_BAND_TOTAL.
            multitaper (bool, optional): If True, uses multitaper method instead of Welch's method. Defaults to False.

        Returns:
            psdtotal (np.ndarray): (M,) long array, M = number of channels. Each value corresponds to sum total of PSD in that band at that channel
        """
        rec = self.get_fragment_np(index)

        return FragmentAnalyzer.compute_psdtotal(rec=rec,
                                                f_s=self.f_s,
                                                welch_bin_t=welch_bin_t,
                                                notch_filter=notch_filter,
                                                band=band,
                                                multitaper=multitaper,
                                                **kwargs)

    def compute_psdslope(self, index, welch_bin_t=1, notch_filter=True, band: tuple[float, float]=constants.FREQ_BAND_TOTAL, multitaper=False, **kwargs):
        """Compute the slope of the power spectral density of the signal.

        Args:
            index (int): Index of time window
            welch_bin_t (float, optional): Length of time bins to use in Welch's method, in seconds. Defaults to 1.
            notch_filter (bool, optional): If True, applies notch filter at line frequency. Defaults to True.
            band (tuple[float, float], optional): Frequency band to calculate over. Defaults to constants.FREQ_BAND_TOTAL.
            multitaper (bool, optional): If True, uses multitaper method instead of Welch's method. Defaults to False.

        Returns:
            np.ndarray: Array of shape (M,2) where M is number of channels. Each row contains [slope, intercept] of log-log fit.
        """
        rec = self.get_fragment_np(index)

        return FragmentAnalyzer.compute_psdslope(rec=rec,
                                                f_s=self.f_s,
                                                welch_bin_t=welch_bin_t,
                                                notch_filter=notch_filter,
                                                band=band,
                                                multitaper=multitaper,
                                                **kwargs)
    
    # Needs work; will need to accept a geometry file to effectively find multielectrode events
    # ... that actually sounds like a lot of work
    # def compute_spikes(self, verbose=False, n_jobs_si: None | int = None, **kwargs):
    #     mso = core.MountainSortAnalyzer(self.LongRecording.LongRecording, verbose=verbose, n_jobs=n_jobs_si)
    #     mso.preprocess_recording()
    #     mso.extract_spikes()
    #     mso.preprocess_final_recording()
    #     self.sorting_analyzer, self.sorting_analyzers = mso.get_final_analyzer()
    #     return self.sorting_analyzer, self.sorting_analyzers


    def convert_idx_to_timebound(self, index: int) -> tuple[float, float]:
        """Convert fragment index to timebound (start time, end time)

        Args:
            index (int): Fragment index

        Returns:
            tuple[float, float]: Timebound in seconds
        """
        frag_len_idx = round(self.fragment_len_s * self.f_s)
        startidx = frag_len_idx * index
        endidx = min(frag_len_idx * (index + 1), self.LongRecording.LongRecording.get_num_frames())
        return (startidx / self.f_s, endidx / self.f_s)

    def compute_wavetemp(self, index, sa_sas=None, ms_before=200, ms_after=200, **kwargs):
        if sa_sas is None:
            if not hasattr(self, "sorting_analyzer") or not hasattr(self, "sorting_analyzers"):
                self.compute_spikes()
            sa_sas = (self.sorting_analyzer, self.sorting_analyzers)
        sa, sas = sa_sas
        assert isinstance(sa, si.SortingAnalyzer)
        for e in sas:
            assert isinstance(e, si.SortingAnalyzer)

        if hasattr(self, 'computed_sorting_analyzer') and hasattr(self, 'computed_sorting_analyzers'):
            return self.computed_sorting_analyzer, self.computed_sorting_analyzers
        else:
            if sa.get_num_units() > 0:
                sa.compute("random_spikes", max_spikes_per_unit=1000)
                sa.compute("waveforms", ms_before=ms_before, ms_after=ms_after)
                sa.compute("templates", operators=["average", "median", "std"])
            else:
                print("No units across all channels, skipping..")
            for i,e in enumerate(sas):
                if e.get_num_units() == 0:
                    # print(f"No units in channel {i}, skipping..")
                    continue
                e.compute("random_spikes", max_spikes_per_unit=1000)
                e.compute("waveforms", ms_before=ms_before, ms_after=ms_after)
                e.compute("templates", operators=["average", "median", "std"])

            self.computed_sorting_analyzer = sa
            self.computed_sorting_analyzers = sas
            return sa, sas

    # def __get_freqs_cycles(self, index, freq_res, n_cycles_max, geomspace, mode:str, epsilon):
    #     if geomspace:
    #         freqs = np.geomspace(constants.FREQ_BAND_TOTAL[0], constants.FREQ_BAND_TOTAL[1], round((np.diff(constants.FREQ_BAND_TOTAL) / freq_res).item()))
    #     else:
    #         freqs = np.arange(constants.FREQ_BAND_TOTAL[0], constants.FREQ_BAND_TOTAL[1], freq_res)

    #     frag_len_s = self.LongRecording.get_dur_fragment(self.fragment_len_s, index)
    #     match mode:
    #         case 'cwt_morlet':
    #             maximum_cyc = (frag_len_s * self.f_s + 1) * np.pi / 5 * freqs / self.f_s
    #             # print(fwhm(freqs, n_cycles_max))
    #         case 'multitaper':
    #             maximum_cyc = frag_len_s * freqs
    #         case _:
    #             raise ValueError(f"Invalid mode {mode}, pick 'cwt_morlet' or 'multitaper'")
    #     maximum_cyc = maximum_cyc - epsilon # Shave off a bit to avoid indexing errors
    #     n_cycles = np.minimum(np.full(maximum_cyc.shape, n_cycles_max), maximum_cyc)
    #     return freqs, n_cycles

    def compute_cohere(self, index, freq_res: float=1, n_cycles_max: float=7, 
                       geomspace: bool=True, 
                       mode: Literal['cwt_morlet', 'multitaper']='cwt_morlet', 
                       downsamp_q: int=4, epsilon: float=1e-2, **kwargs) -> np.ndarray:
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_cohere(rec=rec,
                                                f_s=self.f_s,
                                                freq_res=freq_res,
                                                n_cycles_max=n_cycles_max,
                                                geomspace=geomspace,
                                                mode=mode,
                                                downsamp_q=downsamp_q,
                                                epsilon=epsilon,
                                                **kwargs)
    
    # def compute_cacoh(self, index, freq_res=1, n_cycles_max=7.0, geomspace=True, mode:str='cwt_morlet', downsamp_q=4, epsilon=1e-2, mag_phase=True, indices=None, **kwargs):
    #     rec = self.get_fragment_mne(index)
    #     rec = decimate(rec, q=downsamp_q, axis=-1)
    #     freqs, n_cycles = self.__get_freqs_cycles(index=index, freq_res=freq_res, n_cycles_max=n_cycles_max, geomspace=geomspace, mode=mode, epsilon=epsilon)
    #     try:
    #         con = spectral_connectivity_time(rec,
    #                                         freqs=freqs,
    #                                         method='cacoh',
    #                                         average=True,
    #                                         mode=mode,
    #                                         fmin=constants.FREQ_BAND_TOTAL[0],
    #                                         fmax=constants.FREQ_BAND_TOTAL[1],
    #                                         sfreq=self.f_s / downsamp_q,
    #                                         n_cycles=n_cycles,
    #                                         indices=indices, # TODO implement L/R hemisphere coherence metrics
    #                                         verbose=False)
    #     except MemoryError as e:
    #         raise MemoryError("Out of memory, use a larger freq_res parameter") from e

    #     data:np.ndarray = con.get_data().squeeze()
    #     if mag_phase:
    #         return np.abs(data), np.angle(data, deg=True), con.freqs
    #     else:
    #         return data, con.freqs

    def compute_pcorr(self, index, lower_triag=True, **kwargs) -> np.ndarray:
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_pcorr(rec=rec,
                                              f_s=self.f_s,
                                              lower_triag=lower_triag,
                                              **kwargs)


    # def compute_csd(self, index, magnitude=True, n_jobs=None, **kwargs) -> np.ndarray:
    #     rec = self.get_fragment_mne(index)
    #     csd = csd_array_fourier(rec, self.f_s,
    #                             fmin=constants.FREQ_BAND_TOTAL[0],
    #                             fmax=constants.FREQ_BAND_TOTAL[1],
    #                             ch_names=self.channel_names,
    #                             n_jobs=n_jobs,
    #                             verbose=False)
    #     out = {}
    #     for k,v in constants.FREQ_BANDS.items():
    #         try:
    #             csd_band = csd.mean(fmin=v[0], fmax=v[1]) # Breaks if slice is too short
    #         except (IndexError, UnboundLocalError):
    #             timebound = self.convert_idx_to_timebound(index)
    #             warnings.warn(f"compute_csd failed for window {index}, {round(timebound[1]-timebound[0], 5)} s. Likely too short")
    #             data = self.compute_csd(index - 1, magnitude)[k]
    #         else:
    #             data = csd_band.get_data()
    #         finally:
    #             if magnitude:
    #                 out[k] = np.abs(data)
    #             else:
    #                 out[k] = data
    #     return out

    # def compute_envcorr(self, index, **kwargs) -> np.ndarray:
    #     rec = spre.bandpass_filter(self.get_fragment_rec(index),
    #                                 freq_min=constants.FREQ_BAND_TOTAL[0],
    #                                 freq_max=constants.FREQ_BAND_TOTAL[1])
    #     rec = self.get_fragment_mne(index, rec)
    #     envcor = envelope_correlation(rec, self.channel_names)
    #     return envcor.get_data().reshape((self.n_channels, self.n_channels))

    # def compute_pac(self, index):
    #     ... # TODO implement CFC measures


    def get_file_end(self, index, **kwargs):
        tstart, tend = self.convert_idx_to_timebound(index)
        for tfile in self.LongRecording.end_relative:
           if tstart <= tfile < tend:
               return tfile - tstart
        return None
