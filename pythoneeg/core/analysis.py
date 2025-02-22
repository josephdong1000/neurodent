# Standard library imports
import os
import tempfile
import warnings
from pathlib import Path

# Third party imports
import numpy as np
from scipy.integrate import simpson
from scipy.signal import welch, decimate
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import linregress, pearsonr
from mne import set_config
from mne.time_frequency import (
    psd_array_multitaper,
    csd_array_fourier
)
from mne_connectivity import (
    envelope_correlation,
    spectral_connectivity_time
)
import spikeinterface.core as si
import spikeinterface.preprocessing as spre

# Local imports
from .core import LongRecordingOrganizer
from .sorting import MountainSortOrganizer
from .. import constants
#%%

class LongRecordingAnalyzer:

    # FEATURES = ['rms', 'ampvar', 'psd', 'psdtotal', 'psdband', 'psdslope', 'cohere', 'pcorr', 'nspike', 'wavetemp']
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

        assert isinstance(longrecording, LongRecordingOrganizer)

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
        return np.sqrt((rec ** 2).sum(axis=0) / rec.shape[0])

    def compute_ampvar(self, index, **kwargs):
        """Compute average amplitude variance

        Args:
            index (int): Index of time window

        Returns:
            result: np.ndarray with shape (1, M), M = number of channels
        """
        rec = self.get_fragment_np(index)
        return np.std(rec, axis=0) ** 2

    def compute_psd(self, index, welch_bin_t=1, notch_filter=True, multitaper=False, n_jobs=None, **kwargs):
        """Compute PSD (power spectral density)

        Args:
            index (int): Index of time window
            welch_bin_t (int, optional): Length of time bins to use in Welch's method, in seconds. Defaults to 1.
            notch_filter (bool, optional): If True, inserts a notch filter at the line frequency specified in self.notch_freq. Defaults to True.
            multitaper (bool, optional): If True, uses the multitaper method in MNE instead of Welch's method to compute the PSD. Defaults to False.
            n_jobs (int, optional): Number of jobs to use in multitaper computation. Defaults to None.

        Returns:
            f (np.ndarray): Array of sample frequencies
            psd (np.ndarray): Array of PSD values at sample frequencies. (X, M), X = number of sample frequencies, M = number of channels.
            If sample window length is too short, PSD is interpolated
        """
        rec = self.get_fragment_rec(index)
        if notch_filter:
            rec = spre.notch_filter(rec, freq=self.notch_freq, q=100)
        rec_np = rec.get_traces(return_scaled=True)

        if not multitaper:
            f, psd = welch(rec_np, fs=self.f_s, nperseg=round(welch_bin_t * self.f_s), axis=0)

            if index == self.n_fragments - 1 and self.n_fragments > 1:
                f_prev, _ = self.compute_psd(index - 1, welch_bin_t, notch_filter, multitaper)
                psd = Akima1DInterpolator(f, psd, axis=0, extrapolate=True)(f_prev)
                f = f_prev
        else:
            psd, f = psd_array_multitaper(rec_np.transpose(), self.f_s, fmax=constants.FREQ_BAND_TOTAL[1],
                                            adaptive=True, n_jobs=n_jobs, normalization='full', low_bias=False, verbose=0)
            psd = psd.transpose()
        return f, psd

    def compute_psdband(self, index, welch_bin_t=1, notch_filter=True, bands=None, multitaper=False, f_psd=None, **kwargs):
        fbands = constants.FREQ_BANDS if bands is None else bands
        if f_psd is not None:
            f, psd = f_psd
        else:
            f, psd = self.compute_psd(index, welch_bin_t, notch_filter, multitaper)
        deltaf = np.diff(f).mean()

        out = {}
        for k,v in fbands.items():
            out_v = simpson(psd[np.logical_and(f >= v[0], f <= v[1]), :], dx=deltaf, axis=0)
            out[k] = out_v
        return out

    def compute_psdtotal(self, index, welch_bin_t=1, notch_filter=True, band: list[int]=None, multitaper=False, f_psd=None, **kwargs):
        """Compute total power over PSD (power spectral density) plot within a specified frequency band

        Args:
            index (int): Index of time window
            welch_bin_t (int, optional): Length of time bins to use in Welch's method, in seconds. Defaults to 1.
            notch_filter (bool, optional): If True, inserts a notch filter at the line frequency specified in self.notch_freq. Defaults to True.
            band (list[int], optional): Frequency band to calculate over. band[0] is the lowest frequency, band[1] is the highest. If None, uses self.FREQ_BAND_TOTAL. Defaults to None.
            multitaper (bool, optional): If True, uses the multitaper method in MNE instead of Welch's method to compute the PSD. Defaults to False.
            f_psd (optional): Output of compute_psd. If not None, will be used to calculate statistic, otherwise interally calls compute_psd. Useful to avoid re-computing the PSD. Defaults to None.
            n_jobs (int, optional): Number of jobs to use in multitaper computation. Defaults to None.

        Returns:
            psdtotal (np.ndarray): (M,) long array, M = number of channels. Each value corresponds to sum total of PSD in that band at that channel
        """

        fband = constants.FREQ_BAND_TOTAL if band is None else band
        if f_psd is not None:
            f, psd = f_psd
        else:
            f, psd = self.compute_psd(index, welch_bin_t, notch_filter, multitaper)
        deltaf = np.diff(f).mean()

        return simpson(psd[np.logical_and(f >= fband[0], f <= fband[1]), :], dx=deltaf, axis=0)

    def compute_psdslope(self, index, welch_bin_t=1, notch_filter=True, band=None, multitaper=False, f_psd=None, **kwargs):
        fband = constants.FREQ_BAND_TOTAL if band is None else band
        if f_psd is not None:
            f, psd = f_psd
        else:
            f, psd = self.compute_psd(index, welch_bin_t, notch_filter, multitaper)

        frange = np.logical_and(f >= fband[0], f <= fband[1])
        logf = np.log10(f[frange])
        logpsd = np.log10(psd[frange, :])

        out = []
        for i in range(logpsd.shape[1]):
            result = linregress(logf, logpsd[:, i], 'less')
            out.append((result.slope, result.intercept))
        return out

    # Needs work; will need to accept a geometry file to effectively find multielectrode events
    def compute_spikes(self, verbose=False, n_jobs_si: None | int = None, **kwargs):
        mso = MountainSortOrganizer(self.LongRecording.LongRecording, verbose=verbose, n_jobs=n_jobs_si)
        mso.preprocess_recording()
        mso.extract_spikes()
        mso.preprocess_final_recording()
        self.sorting_analyzer, self.sorting_analyzers = mso.get_final_analyzer()
        return self.sorting_analyzer, self.sorting_analyzers

    def compute_nspike(self, index, sa_sas=None, **kwargs):
        if sa_sas is None:
            if not hasattr(self, "sorting_analyzer") or not hasattr(self, "sorting_analyzers"):
                self.compute_spikes(**kwargs)
            sa_sas = (self.sorting_analyzer, self.sorting_analyzers)
        sa, sas = sa_sas
        assert isinstance(sa, si.SortingAnalyzer)
        for e in sas:
            assert isinstance(e, si.SortingAnalyzer)

        tbound = self.__frag_idx_to_timebound(index)
        nspike_unit = []
        if sa.get_num_units() > 0:
            for unit_id in sa.sorting.unit_ids:
                t_spike = sa.sorting.get_unit_spike_train(unit_id=unit_id) / self.f_s
                nspike_unit.append(((tbound[0] <= t_spike) & (t_spike < tbound[1])).sum())
        else:
            print("No units across all channels, skipping..")

        nspikes_unit = []
        for i,e in enumerate(sas):
            nspikes_unit.append([])
            if e.get_num_units() == 0:
                # print(f"No units in channel {i}, skipping..")
                continue
            for unit_id in e.sorting.unit_ids:
                t_spike = e.sorting.get_unit_spike_train(unit_id=unit_id) / self.f_s
                nspikes_unit[-1].append(((tbound[0] <= t_spike) & (t_spike < tbound[1])).sum())

        return nspikes_unit, nspikes_unit

    def __frag_idx_to_timebound(self, index):
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

    def __get_freqs_cycles(self, index, freq_res, n_cycles_max, geomspace, mode:str, epsilon):
        if geomspace:
            freqs = np.geomspace(constants.FREQ_BAND_TOTAL[0], constants.FREQ_BAND_TOTAL[1], round((np.diff(constants.FREQ_BAND_TOTAL) / freq_res).item()))
        else:
            freqs = np.arange(constants.FREQ_BAND_TOTAL[0], constants.FREQ_BAND_TOTAL[1], freq_res)

        frag_len_s = self.LongRecording.get_dur_fragment(self.fragment_len_s, index)
        match mode:
            case 'cwt_morlet':
                maximum_cyc = (frag_len_s * self.f_s + 1) * np.pi / 5 * freqs / self.f_s
                # print(fwhm(freqs, n_cycles_max))
            case 'multitaper':
                maximum_cyc = frag_len_s * freqs
            case _:
                raise ValueError(f"Invalid mode {mode}, pick 'cwt_morlet' or 'multitaper'")
        maximum_cyc = maximum_cyc - epsilon # Shave off a bit to avoid indexing errors
        n_cycles = np.minimum(np.full(maximum_cyc.shape, n_cycles_max), maximum_cyc)
        return freqs, n_cycles

    def compute_cohere(self, index, freq_res=1, n_cycles_max=7.0, geomspace=True, mode:str='cwt_morlet', downsamp_q=4, epsilon=1e-2, n_jobs_coh=None, **kwargs):
        rec = self.get_fragment_mne(index)
        rec = decimate(rec, q=downsamp_q, axis=-1)
        freqs, n_cycles = self.__get_freqs_cycles(index=index, freq_res=freq_res, n_cycles_max=n_cycles_max, geomspace=geomspace, mode=mode, epsilon=epsilon)
        try:
            con = spectral_connectivity_time(rec,
                                            freqs=freqs,
                                            method='coh',
                                            average=True,
                                            faverage=True,
                                            mode=mode,
                                            fmin=constants.FREQ_MINS,
                                            fmax=constants.FREQ_MAXS,
                                            sfreq=self.f_s / downsamp_q,
                                            n_cycles=n_cycles,
                                            n_jobs=n_jobs_coh,
                                            verbose=False)
        except MemoryError as e:
            raise MemoryError("Out of memory, use a larger freq_res parameter") from e
        data = con.get_data()
        out = {}
        for i in range(data.shape[1]):
            out[constants.FREQ_BAND_NAMES[i]] = data[:, i].reshape((self.n_channels, self.n_channels))
        return out

    def compute_cacoh(self, index, freq_res=1, n_cycles_max=7.0, geomspace=True, mode:str='cwt_morlet', downsamp_q=4, epsilon=1e-2, mag_phase=True, indices=None, **kwargs):
        rec = self.get_fragment_mne(index)
        rec = decimate(rec, q=downsamp_q, axis=-1)
        freqs, n_cycles = self.__get_freqs_cycles(index=index, freq_res=freq_res, n_cycles_max=n_cycles_max, geomspace=geomspace, mode=mode, epsilon=epsilon)
        try:
            con = spectral_connectivity_time(rec,
                                            freqs=freqs,
                                            method='cacoh',
                                            average=True,
                                            mode=mode,
                                            fmin=constants.FREQ_BAND_TOTAL[0],
                                            fmax=constants.FREQ_BAND_TOTAL[1],
                                            sfreq=self.f_s / downsamp_q,
                                            n_cycles=n_cycles,
                                            indices=indices, # TODO implement L/R hemisphere coherence metrics
                                            verbose=False)
        except MemoryError as e:
            raise MemoryError("Out of memory, use a larger freq_res parameter") from e

        data:np.ndarray = con.get_data().squeeze()
        if mag_phase:
            return np.abs(data), np.angle(data, deg=True), con.freqs
        else:
            return data, con.freqs

    def compute_pcorr(self, index, lower_triag=True, **kwargs) -> np.ndarray:
        rec = spre.bandpass_filter(self.get_fragment_rec(index),
                                    freq_min=constants.FREQ_BAND_TOTAL[0],
                                    freq_max=constants.FREQ_BAND_TOTAL[1])
        rec = self.get_fragment_np(index, rec).transpose()
        result = pearsonr(rec[:, np.newaxis, :], rec, axis=-1)
        if lower_triag:
            return np.tril(result.correlation, k=-1)
        else:
            return result.correlation

    def compute_csd(self, index, magnitude=True, n_jobs=None, **kwargs) -> np.ndarray:
        rec = self.get_fragment_mne(index)
        csd = csd_array_fourier(rec, self.f_s,
                                fmin=constants.FREQ_BAND_TOTAL[0],
                                fmax=constants.FREQ_BAND_TOTAL[1],
                                ch_names=self.channel_names,
                                n_jobs=n_jobs,
                                verbose=False)
        out = {}
        for k,v in constants.FREQ_BANDS.items():
            try:
                csd_band = csd.mean(fmin=v[0], fmax=v[1]) # Breaks if slice is too short
            except (IndexError, UnboundLocalError):
                timebound = self.__frag_idx_to_timebound(index)
                warnings.warn(f"compute_csd failed for window {index}, {round(timebound[1]-timebound[0], 5)} s. Likely too short")
                data = self.compute_csd(index - 1, magnitude)[k]
            else:
                data = csd_band.get_data()
            finally:
                if magnitude:
                    out[k] = np.abs(data)
                else:
                    out[k] = data
        return out

    def compute_envcorr(self, index, **kwargs) -> np.ndarray:
        rec = spre.bandpass_filter(self.get_fragment_rec(index),
                                    freq_min=constants.FREQ_BAND_TOTAL[0],
                                    freq_max=constants.FREQ_BAND_TOTAL[1])
        rec = self.get_fragment_mne(index, rec)
        envcor = envelope_correlation(rec, self.channel_names)
        return envcor.get_data().reshape((self.n_channels, self.n_channels))

    def compute_pac(self, index):
        ... # TODO implement CFC measures


    def get_file_end(self, index, **kwargs):
        tstart, tend = self.__frag_idx_to_timebound(index)
        for tfile in self.LongRecording.end_relative:
           if tstart <= tfile < tend:
               return tfile - tstart
        return None

    def setup_njobs(self):
        set_config('MNE_MEMMAP_MIN_SIZE', '30M')
        set_config('MNE_CACHE_DIR', Path(tempfile.gettempdir()) / os.urandom(24).hex())