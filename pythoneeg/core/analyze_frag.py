import logging
from typing import Literal

import numpy as np
from mne.time_frequency import psd_array_multitaper
from mne_connectivity import spectral_connectivity_time
from scipy.integrate import simpson
from scipy.signal import butter, decimate, filtfilt, iirnotch, sosfiltfilt, welch
from scipy.stats import linregress, pearsonr

from .. import constants
from ..core import log_transform


class FragmentAnalyzer:
    """Static class for analyzing fragments of EEG data.
    All functions receive a (N x M) numpy array, where N is the number of samples, and M is the number of channels.
    """

    @staticmethod
    def _process_fragment_features_dask(rec: np.ndarray, f_s: int, features: list[str], kwargs: dict):
        row = {}
        for feat in features:
            func = getattr(FragmentAnalyzer, f"compute_{feat}")
            if callable(func):
                row[feat] = func(rec=rec, f_s=f_s, **kwargs)
            else:
                raise AttributeError(f"Invalid function {func}")
        return row

    @staticmethod
    def _check_rec_np(rec: np.ndarray, **kwargs):
        """Check if the recording is a numpy array and has the correct shape."""
        if not isinstance(rec, np.ndarray):
            raise ValueError("rec must be a numpy array")
        if rec.ndim != 2:
            raise ValueError("rec must be a 2D numpy array")

    @staticmethod
    def _check_rec_mne(rec: np.ndarray, **kwargs):
        """Check if the recording is a MNE-ready numpy array."""
        if not isinstance(rec, np.ndarray):
            raise ValueError("rec must be a numpy array")
        if rec.ndim != 3:
            raise ValueError("rec must be a 3D numpy array")
        if rec.shape[0] != 1:
            raise ValueError("rec must be a 1 x M x N array")

    @staticmethod
    def _reshape_np_for_mne(rec: np.ndarray, **kwargs) -> np.ndarray:
        """Reshape numpy array of (N x M) to (1 x M x N) array for MNE. N = number of samples, M = number of channels."""
        FragmentAnalyzer._check_rec_np(rec)
        rec = rec[..., np.newaxis]
        return np.transpose(rec, (2, 1, 0))

    @staticmethod
    def compute_rms(rec: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the root mean square of the signal."""
        FragmentAnalyzer._check_rec_np(rec)
        out = np.sqrt((rec**2).sum(axis=0) / rec.shape[0])
        # del rec
        return out

    @staticmethod
    def compute_logrms(rec: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the log of the root mean square of the signal."""
        FragmentAnalyzer._check_rec_np(rec)
        return log_transform(FragmentAnalyzer.compute_rms(rec, **kwargs))

    @staticmethod
    def compute_ampvar(rec: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the amplitude variance of the signal."""
        FragmentAnalyzer._check_rec_np(rec)
        return np.std(rec, axis=0) ** 2

    @staticmethod
    def compute_logampvar(rec: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the log of the amplitude variance of the signal."""
        FragmentAnalyzer._check_rec_np(rec)
        return log_transform(FragmentAnalyzer.compute_ampvar(rec, **kwargs))

    @staticmethod
    def compute_psd(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        multitaper: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Compute the power spectral density of the signal."""
        FragmentAnalyzer._check_rec_np(rec)

        if notch_filter:
            b, a = iirnotch(constants.LINE_FREQ, 30, fs=f_s)
            rec = filtfilt(b, a, rec, axis=0)

        if not multitaper:
            f, psd = welch(rec, fs=f_s, nperseg=round(welch_bin_t * f_s), axis=0)
        else:
            # REVIEW psd calulation will give different bins if using multitaper
            psd, f = psd_array_multitaper(
                rec.transpose(),
                f_s,
                fmax=constants.FREQ_BAND_TOTAL[1],
                adaptive=True,
                normalization="full",
                low_bias=False,
                verbose=0,
            )
            psd = psd.transpose()
        return f, psd

    @staticmethod
    def compute_psdband(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        bands: list[tuple[float, float]] = constants.FREQ_BANDS,
        multitaper: bool = False,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Compute the power spectral density of the signal for each frequency band."""
        FragmentAnalyzer._check_rec_np(rec)

        f, psd = FragmentAnalyzer.compute_psd(rec, f_s, welch_bin_t, notch_filter, multitaper, **kwargs)
        deltaf = np.median(np.diff(f))
        return {k: simpson(psd[np.logical_and(f >= v[0], f <= v[1]), :], dx=deltaf, axis=0) for k, v in bands.items()}

    @staticmethod
    def compute_logpsdband(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        bands: list[tuple[float, float]] = constants.FREQ_BANDS,
        multitaper: bool = False,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Compute the log of the power spectral density of the signal for each frequency band."""
        FragmentAnalyzer._check_rec_np(rec)

        psd = FragmentAnalyzer.compute_psdband(rec, f_s, welch_bin_t, notch_filter, bands, multitaper, **kwargs)
        return {k: log_transform(v) for k, v in psd.items()}

    @staticmethod
    def compute_psdtotal(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Compute the total power spectral density of the signal."""
        FragmentAnalyzer._check_rec_np(rec)

        f, psd = FragmentAnalyzer.compute_psd(rec, f_s, welch_bin_t, notch_filter, multitaper, **kwargs)
        deltaf = np.median(np.diff(f))

        return simpson(psd[np.logical_and(f >= band[0], f <= band[1]), :], dx=deltaf, axis=0)

    @staticmethod
    def compute_logpsdtotal(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Compute the log of the total power spectral density of the signal."""
        FragmentAnalyzer._check_rec_np(rec)

        return log_transform(
            FragmentAnalyzer.compute_psdtotal(rec, f_s, welch_bin_t, notch_filter, band, multitaper, **kwargs)
        )

    @staticmethod
    def compute_psdfrac(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        bands: list[tuple[float, float]] = constants.FREQ_BANDS,
        total_band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Compute the power spectral density of bands as a fraction of the total power."""
        FragmentAnalyzer._check_rec_np(rec)

        psd = FragmentAnalyzer.compute_psdband(rec, f_s, welch_bin_t, notch_filter, bands, multitaper, **kwargs)
        psdtotal = FragmentAnalyzer.compute_psdtotal(
            rec, f_s, welch_bin_t, notch_filter, total_band, multitaper, **kwargs
        )
        return {k: v / psdtotal for k, v in psd.items()}

    @staticmethod
    def compute_logpsdfrac(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        bands: list[tuple[float, float]] = constants.FREQ_BANDS,
        total_band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper: bool = False,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Compute the log of the power spectral density of bands as a fraction of the log total power."""
        FragmentAnalyzer._check_rec_np(rec)

        # logpsd = FragmentAnalyzer.compute_logpsdband(rec, f_s, welch_bin_t, notch_filter, bands, multitaper, **kwargs)
        # logpsdtotal = FragmentAnalyzer.compute_logpsdtotal(rec, f_s, welch_bin_t, notch_filter, total_band, multitaper, **kwargs)
        # return {k: v / logpsdtotal for k, v in logpsd.items()}

        psd_band = FragmentAnalyzer.compute_psdband(rec, f_s, welch_bin_t, notch_filter, bands, multitaper, **kwargs)
        psd_total = FragmentAnalyzer.compute_psdtotal(
            rec, f_s, welch_bin_t, notch_filter, total_band, multitaper, **kwargs
        )
        return {k: log_transform(v / psd_total) for k, v in psd_band.items()}

    @staticmethod
    def compute_psdslope(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Compute the slope of the power spectral density of the signal on a log-log scale."""
        FragmentAnalyzer._check_rec_np(rec)

        f, psd = FragmentAnalyzer.compute_psd(rec, f_s, welch_bin_t, notch_filter, multitaper, **kwargs)

        freqs = f[np.logical_and(f >= band[0], f <= band[1])]
        psd_band = psd[np.logical_and(f >= band[0], f <= band[1]), :]
        logpsd = np.log10(psd_band)
        logf = np.log10(freqs)

        # Fit a line to the log-transformed data
        out = []
        for i in range(psd_band.shape[1]):
            result = linregress(logf, logpsd[:, i])
            out.append([result.slope, result.intercept])
        return np.array(out)

    @staticmethod
    def _get_freqs_cycles(
        rec: np.ndarray,
        f_s: float,
        freq_res: float,
        n_cycles_max: float,
        geomspace: bool,
        mode: Literal["cwt_morlet", "multitaper"],
        epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the frequencies and number of cycles for the signal.
        rec is a (1 x M x N) numpy array for MNE. N = number of samples, M = number of channels.
        """
        FragmentAnalyzer._check_rec_mne(rec)

        if geomspace:
            freqs = np.geomspace(
                constants.FREQ_BAND_TOTAL[0],
                constants.FREQ_BAND_TOTAL[1],
                round((np.diff(constants.FREQ_BAND_TOTAL) / freq_res).item()),
            )
        else:
            freqs = np.arange(constants.FREQ_BAND_TOTAL[0], constants.FREQ_BAND_TOTAL[1], freq_res)

        frag_len_s = rec.shape[2] / f_s
        match mode:
            case "cwt_morlet":
                maximum_cyc = (frag_len_s * f_s + 1) * np.pi / 5 * freqs / f_s
            case "multitaper":
                maximum_cyc = frag_len_s * freqs

        maximum_cyc = maximum_cyc - epsilon  # Shave off a bit to avoid indexing errors
        n_cycles = np.minimum(np.full(maximum_cyc.shape, n_cycles_max), maximum_cyc)
        return freqs, n_cycles

    @staticmethod
    def compute_cohere(
        rec: np.ndarray,
        f_s: float,
        freq_res: float = 1,
        n_cycles_max: float = 7,
        geomspace: bool = True,
        mode: Literal["cwt_morlet", "multitaper"] = "cwt_morlet",
        downsamp_q: int = 4,
        epsilon: float = 1e-2,
        **kwargs,
    ) -> np.ndarray:
        """Compute the coherence of the signal."""
        FragmentAnalyzer._check_rec_np(rec)

        rec_mne = FragmentAnalyzer._reshape_np_for_mne(rec)
        rec_mne = decimate(rec_mne, q=downsamp_q, axis=2)  # Along the time axis
        f_s = int(f_s / downsamp_q)

        f, n_cycles = FragmentAnalyzer._get_freqs_cycles(rec_mne, f_s, freq_res, n_cycles_max, geomspace, mode, epsilon)

        try:
            con = spectral_connectivity_time(
                rec_mne,
                freqs=f,
                method="coh",
                average=True,
                faverage=True,
                mode=mode,
                fmin=constants.FREQ_MINS,
                fmax=constants.FREQ_MAXS,
                sfreq=f_s,
                n_cycles=n_cycles,
                verbose=False,
            )
        except MemoryError as e:
            raise MemoryError(
                "Out of memory. Use a larger freq_res parameter, a smaller n_cycles_max parameter, or a larger downsamp_q parameter"
            ) from e

        data = con.get_data()
        out = {}
        for i in range(data.shape[1]):
            out[constants.BAND_NAMES[i]] = data[:, i].reshape((rec.shape[1], rec.shape[1]))
        return out

    @staticmethod
    def compute_pcorr(rec: np.ndarray, f_s: float, lower_triag: bool = True, **kwargs) -> np.ndarray:
        """Compute the Pearson correlation coefficient of the signal."""
        FragmentAnalyzer._check_rec_np(rec)

        sos = butter(2, constants.FREQ_BAND_TOTAL, btype="bandpass", output="sos", fs=f_s)
        rec = sosfiltfilt(sos, rec, axis=0)

        rec = rec.transpose()
        result = pearsonr(rec[:, np.newaxis, :], rec, axis=-1)
        if lower_triag:
            return np.tril(result.correlation, k=-1)
        else:
            return result.correlation

    @staticmethod
    def compute_zpcorr(rec: np.ndarray, f_s: float, **kwargs) -> np.ndarray:
        """Compute the Fisher z-transformed Pearson correlation coefficient of the signal."""
        FragmentAnalyzer._check_rec_np(rec)

        pcorr = FragmentAnalyzer.compute_pcorr(rec, f_s, **kwargs)
        return np.arctanh(pcorr)

    @staticmethod
    def compute_zcohere(rec: np.ndarray, f_s: float, **kwargs) -> dict[str, np.ndarray]:
        """Compute the Fisher z-transformed coherence of the signal."""
        FragmentAnalyzer._check_rec_np(rec)

        cohere = FragmentAnalyzer.compute_cohere(rec, f_s, **kwargs)
        return {k: np.arctanh(v) for k, v in cohere.items()}

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
    #     ... # NOTE implement CFC measures

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
    #                                         indices=indices, # NOTE implement L/R hemisphere coherence metrics
    #                                         verbose=False)
    #     except MemoryError as e:
    #         raise MemoryError("Out of memory, use a larger freq_res parameter") from e

    #     data:np.ndarray = con.get_data().squeeze()
    #     if mag_phase:
    #         return np.abs(data), np.angle(data, deg=True), con.freqs
    #     else:
    #         return data, con.freqs