from typing import Literal

import numpy as np
try:
    import spikeinterface.core as si
    import spikeinterface.preprocessing as spre
except Exception:  # pragma: no cover - optional at import time for tests not using spikeinterface
    si = None
    spre = None
try:
    from mne import set_config
    from mne.time_frequency import csd_array_fourier
except Exception:  # pragma: no cover
    set_config = None
    csd_array_fourier = None
try:
    from mne_connectivity import envelope_correlation
except Exception:  # pragma: no cover
    envelope_correlation = None
from scipy.interpolate import Akima1DInterpolator

from .. import constants, core
from .analyze_frag import FragmentAnalyzer


class LongRecordingAnalyzer:
    def __init__(self, longrecording: core.LongRecordingOrganizer, fragment_len_s=10, apply_notch_filter=True) -> None:
        assert isinstance(longrecording, core.LongRecordingOrganizer)

        self.LongRecording = longrecording
        self.fragment_len_s = fragment_len_s
        self.n_fragments = longrecording.get_num_fragments(fragment_len_s)
        self.channel_names = longrecording.channel_names
        self.n_channels = longrecording.meta.n_channels
        self.mult_to_uV = longrecording.meta.mult_to_uV
        self.f_s = int(longrecording.LongRecording.get_sampling_frequency())
        self.apply_notch_filter = apply_notch_filter

    def get_fragment_rec(self, index) -> "si.BaseRecording":
        """Get window at index as a spikeinterface recording object

        Args:
            index (int): Index of time window

        Returns:
            si.BaseRecording: spikeinterface recording object with optional notch filtering applied
        """
        if si is None:
            raise ImportError("spikeinterface is required for get_fragment_rec")
        
        rec = self.LongRecording.get_fragment(self.fragment_len_s, index)
        
        if self.apply_notch_filter and spre is not None:
            rec = spre.notch_filter(rec, freq=constants.LINE_FREQ)
        
        return rec

    def get_fragment_np(self, index, recobj=None) -> np.ndarray:
        """Get window at index as a numpy array object

        Args:
            index (int): Index of time window
            recobj (si.BaseRecording, optional): If not None, uses this recording object to get the numpy array. Defaults to None.

        Returns:
            np.ndarray: Numpy array with dimensions (N, M), N = number of samples, M = number of channels. Values in uV
        """
        if si is not None:
            assert isinstance(recobj, si.BaseRecording) or recobj is None
        if recobj is None:
            return self.get_fragment_rec(index).get_traces(
                return_scaled=True
            )  # (num_samples, num_channels), in units uV
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
        return np.transpose(rec, (2, 1, 0))  # (1 epoch, num_channels, num_samples)

    def get_file_end(self, index, **kwargs):
        tstart, tend = self.convert_idx_to_timebound(index)
        for tfile in self.LongRecording.cumulative_file_durations:
            if tstart <= tfile < tend:
                return tfile - tstart
        return None

    def compute_rms(self, index, **kwargs):
        """Compute average root mean square amplitude

        Args:
            index (int): Index of time window

        Returns:
            result: np.ndarray with shape (1, M), M = number of channels
        """
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_rms(rec=rec, **kwargs)

    def compute_logrms(self, index, **kwargs):
        """Compute the log of the root mean square amplitude"""
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_logrms(rec=rec, **kwargs)

    def compute_ampvar(self, index, **kwargs):
        """Compute average amplitude variance

        Args:
            index (int): Index of time window

        Returns:
            result: np.ndarray with shape (1, M), M = number of channels
        """
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_ampvar(rec=rec, **kwargs)

    def compute_logampvar(self, index, **kwargs):
        """Compute the log of the amplitude variance"""
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_logampvar(rec=rec, **kwargs)

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

        f, psd = FragmentAnalyzer.compute_psd(
            rec=rec, f_s=self.f_s, welch_bin_t=welch_bin_t, notch_filter=notch_filter, multitaper=multitaper, **kwargs
        )

        if index == self.n_fragments - 1 and self.n_fragments > 1:
            f_prev, _ = self.compute_psd(index - 1, welch_bin_t, notch_filter, multitaper)
            psd = Akima1DInterpolator(f, psd, axis=0, extrapolate=True)(f_prev)
            f = f_prev

        return f, psd

    def compute_psdband(
        self,
        index,
        welch_bin_t=1,
        notch_filter=True,
        bands: list[tuple[float, float]] = constants.FREQ_BANDS,
        multitaper=False,
        **kwargs,
    ):
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

        return FragmentAnalyzer.compute_psdband(
            rec=rec,
            f_s=self.f_s,
            welch_bin_t=welch_bin_t,
            notch_filter=notch_filter,
            bands=bands,
            multitaper=multitaper,
            **kwargs,
        )

    def compute_logpsdband(
        self,
        index,
        welch_bin_t=1,
        notch_filter=True,
        bands: list[tuple[float, float]] = constants.FREQ_BANDS,
        multitaper=False,
        **kwargs,
    ):
        """Compute the log of the power spectral density of the signal for each frequency band."""
        rec = self.get_fragment_np(index)

        return FragmentAnalyzer.compute_logpsdband(
            rec=rec,
            f_s=self.f_s,
            welch_bin_t=welch_bin_t,
            notch_filter=notch_filter,
            bands=bands,
            multitaper=multitaper,
            **kwargs,
        )

    def compute_psdtotal(
        self,
        index,
        welch_bin_t=1,
        notch_filter=True,
        band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper=False,
        **kwargs,
    ):
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

        return FragmentAnalyzer.compute_psdtotal(
            rec=rec,
            f_s=self.f_s,
            welch_bin_t=welch_bin_t,
            notch_filter=notch_filter,
            band=band,
            multitaper=multitaper,
            **kwargs,
        )

    def compute_logpsdtotal(
        self,
        index,
        welch_bin_t=1,
        notch_filter=True,
        band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper=False,
        **kwargs,
    ):
        """Compute the log of the total power over PSD (power spectral density) plot within a specified frequency band"""
        rec = self.get_fragment_np(index)

        return FragmentAnalyzer.compute_logpsdtotal(
            rec=rec,
            f_s=self.f_s,
            welch_bin_t=welch_bin_t,
            notch_filter=notch_filter,
            band=band,
            multitaper=multitaper,
            **kwargs,
        )

    def compute_psdfrac(
        self,
        index,
        welch_bin_t=1,
        notch_filter=True,
        bands: list[tuple[float, float]] = constants.FREQ_BANDS,
        total_band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper=False,
        **kwargs,
    ):
        """Compute the power spectral density in each band as a fraction of the total power."""
        rec = self.get_fragment_np(index)

        return FragmentAnalyzer.compute_psdfrac(
            rec=rec,
            f_s=self.f_s,
            welch_bin_t=welch_bin_t,
            notch_filter=notch_filter,
            bands=bands,
            total_band=total_band,
            multitaper=multitaper,
            **kwargs,
        )

    def compute_logpsdfrac(
        self,
        index,
        welch_bin_t=1,
        notch_filter=True,
        bands: list[tuple[float, float]] = constants.FREQ_BANDS,
        total_band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper=False,
        **kwargs,
    ):
        """Compute the log of the power spectral density in each band as a fraction of the total power."""
        rec = self.get_fragment_np(index)

        return FragmentAnalyzer.compute_logpsdfrac(
            rec=rec,
            f_s=self.f_s,
            welch_bin_t=welch_bin_t,
            notch_filter=notch_filter,
            bands=bands,
            total_band=total_band,
            multitaper=multitaper,
            **kwargs,
        )

    def compute_psdslope(
        self,
        index,
        welch_bin_t=1,
        notch_filter=True,
        band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper=False,
        **kwargs,
    ):
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

        return FragmentAnalyzer.compute_psdslope(
            rec=rec,
            f_s=self.f_s,
            welch_bin_t=welch_bin_t,
            notch_filter=notch_filter,
            band=band,
            multitaper=multitaper,
            **kwargs,
        )

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

    def compute_cohere(
        self,
        index,
        freq_res: float = 1,
        mode: Literal["cwt_morlet", "multitaper"] = "multitaper",
        geomspace: bool = False,
        cwt_n_cycles_max: float = 7.0,
        mt_bandwidth: float = 4.0,
        downsamp_q: int = 4,
        epsilon: float = 1e-2,
        **kwargs,
    ) -> np.ndarray:
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_cohere(
            rec=rec,
            f_s=self.f_s,
            freq_res=freq_res,
            mode=mode,
            geomspace=geomspace,
            cwt_n_cycles_max=cwt_n_cycles_max,
            mt_bandwidth=mt_bandwidth,
            downsamp_q=downsamp_q,
            epsilon=epsilon,
            **kwargs,
        )

    def compute_zcohere(self, index, z_epsilon: float = 1e-6, **kwargs) -> np.ndarray:
        """Compute the Fisher z-transformed coherence of the signal.
        
        Args:
            index (int): Index of time window
            z_epsilon (float): Small value to prevent arctanh(1) = inf. Values are clipped to [-1+z_epsilon, 1-z_epsilon]
            **kwargs: Additional arguments passed to compute_zcohere
        """
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_zcohere(rec=rec, f_s=self.f_s, z_epsilon=z_epsilon, **kwargs)

    def compute_imcoh(self, index, **kwargs) -> np.ndarray:
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_imcoh(rec=rec, f_s=self.f_s, **kwargs)
    
    def compute_zimcoh(self, index, z_epsilon: float = 1e-6, **kwargs) -> np.ndarray:
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_zimcoh(rec=rec, f_s=self.f_s, z_epsilon=z_epsilon, **kwargs)

    def compute_pcorr(self, index, lower_triag=False, **kwargs) -> np.ndarray:
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_pcorr(rec=rec, f_s=self.f_s, lower_triag=lower_triag, **kwargs)

    def compute_zpcorr(self, index, z_epsilon: float = 1e-6, **kwargs) -> np.ndarray:
        """Compute the Fisher z-transformed Pearson correlation coefficient of the signal.
        
        Args:
            index (int): Index of time window
            z_epsilon (float): Small value to prevent arctanh(1) = inf. Values are clipped to [-1+z_epsilon, 1-z_epsilon]
            **kwargs: Additional arguments passed to compute_zpcorr
        """
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_zpcorr(rec=rec, f_s=self.f_s, z_epsilon=z_epsilon, **kwargs)

    def compute_nspike(self, index, **kwargs):
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_nspike(rec=rec, f_s=self.f_s, **kwargs)

    def compute_lognspike(self, index, **kwargs):
        rec = self.get_fragment_np(index)
        return FragmentAnalyzer.compute_lognspike(rec=rec, f_s=self.f_s, **kwargs)