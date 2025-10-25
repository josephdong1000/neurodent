from typing import Literal, Dict, List, Any
import warnings

import numpy as np

try:
    from mne.time_frequency import psd_array_multitaper
except Exception:  # pragma: no cover - optional at import time
    psd_array_multitaper = None
try:  # keep optional to allow running light tests that do not exercise connectivity
    from mne_connectivity import spectral_connectivity_time, spectral_connectivity_epochs
except Exception:  # pragma: no cover - optional at import time
    spectral_connectivity_time = None
    spectral_connectivity_epochs = None
from scipy.integrate import trapezoid
from scipy.signal import butter, decimate, filtfilt, iirnotch, sosfiltfilt, welch
from scipy.stats import linregress, pearsonr

from .. import constants
from .utils import log_transform


class FragmentAnalyzer:
    """Static class for analyzing fragments of EEG data.
    All functions receive a (N x M) numpy array, where N is the number of samples, and M is the number of channels.
    """

    # Unified feature dependency mapping - all dependencies as lists for consistency
    FEATURE_DEPENDENCIES = {
        # Log transforms
        "logrms": ["rms"],
        "logampvar": ["ampvar"],
        "lognspike": ["nspike"],
        "logpsdband": ["psdband"],
        "logpsdtotal": ["psdtotal"],
        "logpsdfrac": ["psdfrac"],
        # Z-transforms
        "zpcorr": ["pcorr"],
        "zcohere": ["cohere"],
        "zimcoh": ["imcoh"],
        # PSD-dependent features
        "psdband": ["psd"],
        "psdtotal": ["psd"],
        "psdslope": ["psd"],
        "psdfrac": ["psdband"],
        # Coherency-dependent features
        "cohere": ["coherency"],
        "imcoh": ["coherency"],
    }

    @staticmethod
    def _process_fragment_features_dask(rec: np.ndarray, f_s: int, features: list[str], kwargs: dict):
        """
        Legacy fragment processing method without dependency optimization.

        Note: Consider using process_fragment_with_dependencies() instead for better performance
        when computing interdependent features like PSD-based features.
        """
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
        """Reshape numpy array of (N x M) to (1 x M x N) array for MNE. 1 epoch, M = number of channels, N = number of samples."""
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
    def compute_logrms(rec: np.ndarray, precomputed_rms: np.ndarray = None, **kwargs) -> np.ndarray:
        """Compute the log of the root mean square of the signal."""
        FragmentAnalyzer._check_rec_np(rec)
        # Local import to avoid importing heavy dependencies from utils at module import time

        if precomputed_rms is not None:
            return log_transform(precomputed_rms)
        else:
            return log_transform(FragmentAnalyzer.compute_rms(rec, **kwargs))

    @staticmethod
    def compute_ampvar(rec: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the amplitude variance of the signal."""
        FragmentAnalyzer._check_rec_np(rec)
        return np.std(rec, axis=0) ** 2

    @staticmethod
    def compute_logampvar(rec: np.ndarray, precomputed_ampvar: np.ndarray = None, **kwargs) -> np.ndarray:
        """Compute the log of the amplitude variance of the signal."""
        FragmentAnalyzer._check_rec_np(rec)
        # Local import to avoid importing heavy dependencies from utils at module import time

        if precomputed_ampvar is not None:
            return log_transform(precomputed_ampvar)
        else:
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
            if psd_array_multitaper is None:
                raise ImportError("mne is required for multitaper PSD; install mne or set multitaper=False")
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
        precomputed_psd: tuple = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Compute the power spectral density of the signal for each frequency band."""
        FragmentAnalyzer._check_rec_np(rec)

        if precomputed_psd is not None:
            f, psd = precomputed_psd
        else:
            f, psd = FragmentAnalyzer.compute_psd(rec, f_s, welch_bin_t, notch_filter, multitaper, **kwargs)
        deltaf = np.median(np.diff(f))

        # Integrate each band separately using trapezoidal integration
        result = {}

        for band_name, (f_low, f_high) in bands.items():
            # Always use inclusive boundaries for both ends
            freq_mask = np.logical_and(f >= f_low, f <= f_high)
            result[band_name] = trapezoid(psd[freq_mask, :], dx=deltaf, axis=0)

        return result

    @staticmethod
    def compute_logpsdband(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        bands: list[tuple[float, float]] = constants.FREQ_BANDS,
        multitaper: bool = False,
        precomputed_psd: tuple = None,
        precomputed_psdband: dict = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Compute the log of the power spectral density of the signal for each frequency band."""
        FragmentAnalyzer._check_rec_np(rec)

        # Local import to avoid importing heavy dependencies from utils at module import time

        if precomputed_psdband is not None:
            psd = precomputed_psdband
        else:
            psd = FragmentAnalyzer.compute_psdband(
                rec, f_s, welch_bin_t, notch_filter, bands, multitaper, precomputed_psd=precomputed_psd, **kwargs
            )
        return {k: log_transform(v) for k, v in psd.items()}

    @staticmethod
    def compute_psdtotal(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper: bool = False,
        precomputed_psd: tuple = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute the total power spectral density of the signal."""
        FragmentAnalyzer._check_rec_np(rec)

        if precomputed_psd is not None:
            f, psd = precomputed_psd
        else:
            f, psd = FragmentAnalyzer.compute_psd(rec, f_s, welch_bin_t, notch_filter, multitaper, **kwargs)
        deltaf = np.median(np.diff(f))

        # Use inclusive bounds for total power calculation
        freq_mask = np.logical_and(f >= band[0], f <= band[1])
        return trapezoid(psd[freq_mask, :], dx=deltaf, axis=0)

    @staticmethod
    def compute_logpsdtotal(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper: bool = False,
        precomputed_psd: tuple = None,
        precomputed_psdtotal: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute the log of the total power spectral density of the signal."""
        FragmentAnalyzer._check_rec_np(rec)

        # Local import to avoid importing heavy dependencies from utils at module import time

        if precomputed_psdtotal is not None:
            return log_transform(precomputed_psdtotal)
        else:
            return log_transform(
                FragmentAnalyzer.compute_psdtotal(
                    rec, f_s, welch_bin_t, notch_filter, band, multitaper, precomputed_psd=precomputed_psd, **kwargs
                )
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
        precomputed_psdband: dict = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Compute the power spectral density of bands as a fraction of the total power."""
        FragmentAnalyzer._check_rec_np(rec)

        if precomputed_psdband is not None:
            psdband = precomputed_psdband
        else:
            psdband = FragmentAnalyzer.compute_psdband(rec, f_s, welch_bin_t, notch_filter, bands, multitaper, **kwargs)
        psdtotal = sum(psdband.values())

        return {k: v / psdtotal for k, v in psdband.items()}

    @staticmethod
    def compute_logpsdfrac(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        bands: list[tuple[float, float]] = constants.FREQ_BANDS,
        total_band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper: bool = False,
        precomputed_psdfrac: dict = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Compute the log of the power spectral density of bands as a fraction of the log total power."""
        FragmentAnalyzer._check_rec_np(rec)

        if precomputed_psdfrac is not None:
            psdfrac = precomputed_psdfrac
        else:
            psdfrac = FragmentAnalyzer.compute_psdfrac(
                rec, f_s, welch_bin_t, notch_filter, bands, total_band, multitaper, **kwargs
            )

        return {k: log_transform(v) for k, v in psdfrac.items()}

    @staticmethod
    def compute_psdslope(
        rec: np.ndarray,
        f_s: float,
        welch_bin_t: float = 1,
        notch_filter: bool = True,
        band: tuple[float, float] = constants.FREQ_BAND_TOTAL,
        multitaper: bool = False,
        precomputed_psd: tuple = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute the slope of the power spectral density of the signal on a log-log scale."""
        FragmentAnalyzer._check_rec_np(rec)

        if precomputed_psd is not None:
            f, psd = precomputed_psd
        else:
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
        geomspace: bool,
        mode: Literal["cwt_morlet", "multitaper"],
        cwt_n_cycles_max: float,
        epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the frequencies and number of cycles for the signal.
        rec is a (1 x M x N) numpy array for MNE. N = number of samples, M = number of channels.
        """
        FragmentAnalyzer._check_rec_mne(rec)

        if geomspace:  # REVIEW by default geomspace is True, but a linear scale is simpler for DOF calculation -> zcohere correction
            freqs = np.geomspace(
                constants.FREQ_BAND_TOTAL[0],
                constants.FREQ_BAND_TOTAL[1],
                round((np.diff(constants.FREQ_BAND_TOTAL) / freq_res).item()),
            )
        else:
            freqs = np.arange(constants.FREQ_BAND_TOTAL[0], constants.FREQ_BAND_TOTAL[1], freq_res)

        frag_len_s = rec.shape[2] / f_s
        if mode == "cwt_morlet":
            maximum_cyc = (frag_len_s * f_s + 1) * np.pi / 5 * freqs / f_s
            maximum_cyc = maximum_cyc - epsilon  # Shave off a bit to avoid indexing errors
            n_cycles = np.minimum(np.full(maximum_cyc.shape, cwt_n_cycles_max), maximum_cyc)
        elif mode == "multitaper":
            maximum_cyc = frag_len_s * freqs  # Maximize number of cycles for maximum frequency resolution
            maximum_cyc = maximum_cyc - epsilon
            n_cycles = maximum_cyc

        return freqs, n_cycles

    @staticmethod
    def compute_coherency(
        rec: np.ndarray,
        f_s: float,
        freq_res: float = 1,
        mode: Literal["cwt_morlet", "multitaper"] = "multitaper",
        geomspace: bool = False,
        cwt_n_cycles_max: float = 7.0,
        mt_bandwidth: float = 4.0,
        downsamp_q: int = 4,
        epsilon: float = 1e-2,
        **kwargs,
    ) -> np.ndarray:
        """Compute the complex coherency of the signal."""
        FragmentAnalyzer._check_rec_np(rec)

        rec_mne = FragmentAnalyzer._reshape_np_for_mne(rec)
        rec_mne = decimate(rec_mne, q=downsamp_q, axis=2)  # Along the time axis
        f_s = int(f_s / downsamp_q)

        f, n_cycles = FragmentAnalyzer._get_freqs_cycles(
            rec=rec_mne,
            f_s=f_s,
            freq_res=freq_res,
            geomspace=geomspace,
            mode=mode,
            cwt_n_cycles_max=cwt_n_cycles_max,
            epsilon=epsilon,
        )

        if spectral_connectivity_epochs is None:
            raise ImportError("mne_connectivity is required for connectivity computations")
        try:
            con = spectral_connectivity_epochs(
                rec_mne,
                # freqs=f,
                method="cohy",
                # average=True,
                faverage=True,
                mode=mode,
                fmin=constants.FREQ_MINS,
                fmax=constants.FREQ_MAXS,
                sfreq=f_s,
                cwt_freqs=f,
                cwt_n_cycles=n_cycles,
                mt_bandwidth=mt_bandwidth,
                verbose=False,
            )
        except MemoryError as e:
            raise MemoryError(
                "Out of memory. Use a larger freq_res parameter, a smaller n_cycles_max parameter, or a larger downsamp_q parameter"
            ) from e

        data = con.get_data()
        n_channels = rec.shape[1]

        out = {}
        # Make data symmetric
        for i, band_name in enumerate(constants.BAND_NAMES):
            if i >= data.shape[1]:  # Skip if we don't have data for this band
                warnings.warn(f"No coherence data for band {band_name}")
                continue

            band_data = data[:, i]

            full_matrix = band_data.reshape((n_channels, n_channels))

            symmetric_matrix = full_matrix.copy()
            symmetric_matrix = np.triu(symmetric_matrix.T, k=1) + np.tril(symmetric_matrix, k=-1)

            np.fill_diagonal(symmetric_matrix, 1.0)

            out[band_name] = symmetric_matrix
        return out

    @staticmethod
    def compute_cohere(
        rec: np.ndarray,
        f_s: float,
        precomputed_coherency: dict = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute the coherence of the signal."""
        FragmentAnalyzer._check_rec_np(rec)
        if precomputed_coherency is not None:
            cohere = precomputed_coherency
        else:
            cohere = FragmentAnalyzer.compute_coherency(rec, f_s, **kwargs)
        return {k: np.abs(v) for k, v in cohere.items()}

    @staticmethod
    def compute_zcohere(
        rec: np.ndarray, f_s: float, z_epsilon: float = 1e-6, precomputed_cohere=None, **kwargs
    ) -> dict[str, np.ndarray]:
        """Compute the Fisher z-transformed coherence of the signal.

        Args:
            rec: Input signal array
            f_s: Sampling frequency
            z_epsilon: Small value to prevent arctanh(1) = inf. Values are clipped to [-1+z_epsilon, 1-z_epsilon]
            **kwargs: Additional arguments passed to compute_cohere
        """
        FragmentAnalyzer._check_rec_np(rec)

        if precomputed_cohere is not None:
            cohere = precomputed_cohere.copy()
        else:
            cohere = FragmentAnalyzer.compute_cohere(rec, f_s, **kwargs)
        clip_max = 1.0 - z_epsilon
        clip_min = -1.0 + z_epsilon
        return {k: np.arctanh(np.clip(v, clip_min, clip_max)) for k, v in cohere.items()}

    @staticmethod
    def compute_imcoh(
        rec: np.ndarray, f_s: float, precomputed_coherency: dict = None, **kwargs
    ) -> dict[str, np.ndarray]:
        """Compute the imaginary coherence of the signal."""
        FragmentAnalyzer._check_rec_np(rec)
        if precomputed_coherency is not None:
            cohere = precomputed_coherency
        else:
            cohere = FragmentAnalyzer.compute_coherency(rec, f_s, **kwargs)
        return {k: np.imag(v) for k, v in cohere.items()}

    @staticmethod
    def compute_zimcoh(
        rec: np.ndarray, f_s: float, z_epsilon: float = 1e-6, precomputed_imcoh: dict = None, **kwargs
    ) -> dict[str, np.ndarray]:
        """Compute the Fisher z-transformed imaginary coherence of the signal."""
        FragmentAnalyzer._check_rec_np(rec)
        if precomputed_imcoh is not None:
            imcoh = precomputed_imcoh
        else:
            imcoh = FragmentAnalyzer.compute_imcoh(rec, f_s, **kwargs)
        clip_max = 1.0 - z_epsilon
        clip_min = -1.0 + z_epsilon
        return {k: np.arctanh(np.clip(v, clip_min, clip_max)) for k, v in imcoh.items()}

    @staticmethod
    def compute_pcorr(rec: np.ndarray, f_s: float, lower_triag: bool = False, **kwargs) -> np.ndarray:
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
    def compute_zpcorr(
        rec: np.ndarray, f_s: float, z_epsilon: float = 1e-6, precomputed_pcorr: np.ndarray = None, **kwargs
    ) -> np.ndarray:
        """Compute the Fisher z-transformed Pearson correlation coefficient of the signal.

        Args:
            rec: Input signal array
            f_s: Sampling frequency
            z_epsilon: Small value to prevent arctanh(1) = inf. Values are clipped to [-1+z_epsilon, 1-z_epsilon]
            **kwargs: Additional arguments passed to compute_pcorr
        """
        FragmentAnalyzer._check_rec_np(rec)

        if precomputed_pcorr is not None:
            pcorr = precomputed_pcorr.copy()
        else:
            # Get full correlation matrix for z-transform
            pcorr = FragmentAnalyzer.compute_pcorr(rec, f_s, lower_triag=False, **kwargs)
        clip_max = 1.0 - z_epsilon
        clip_min = -1.0 + z_epsilon
        return np.arctanh(np.clip(pcorr, clip_min, clip_max))

    @staticmethod
    def compute_nspike(rec: np.ndarray, **kwargs):
        """Returns NaN array as placeholder. Compute and load in spikes with SpikeAnalysisResult"""
        return np.full(rec.shape[1], np.nan)

    @staticmethod
    def compute_lognspike(rec: np.ndarray, precomputed_nspike: np.ndarray = None, **kwargs):
        """Returns log-transformed NaN array as placeholder. Compute and load in spikes with SpikeAnalysisResult"""
        # Local import to avoid importing heavy dependencies from utils at module import time

        if precomputed_nspike is not None:
            return log_transform(precomputed_nspike)
        else:
            n_spike = FragmentAnalyzer.compute_nspike(rec, **kwargs)
            return log_transform(n_spike)

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

    @staticmethod
    def process_fragment_with_dependencies(
        fragment_data: np.ndarray, f_s: int, features: List[str], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single fragment with efficient dependency resolution.

        This is the enhanced replacement for _process_fragment_features_dask that
        automatically resolves feature dependencies and reuses intermediate calculations
        to avoid redundant computations (e.g., computing PSD once for multiple dependent features).

        Args:
            fragment_data: Single fragment data with shape (n_samples, n_channels)
            f_s: Sampling frequency
            features: List of features to compute
            kwargs: Additional parameters for feature computation

        Returns:
            Dictionary of computed features for this fragment
        """
        computed_cache = {}
        results = {}

        # Compute all requested features using dependency resolution
        for feature in features:
            if feature not in results:
                results[feature] = FragmentAnalyzer._resolve_feature_dependencies(
                    feature, fragment_data, f_s, kwargs, computed_cache
                )

        return results

    @staticmethod
    def _resolve_feature_dependencies(
        feature: str, fragment_data: np.ndarray, f_s: int, kwargs: Dict[str, Any], computed_cache: Dict[str, Any]
    ) -> Any:
        """
        Resolve feature dependencies recursively, caching intermediate results.

        This handles the dependency tree resolution automatically. For example:
        - logpsdfrac -> psdfrac -> [psdband, psdtotal] -> psd
        - All intermediate results are cached and reused within the fragment

        Args:
            feature: Name of feature to compute
            fragment_data: EEG fragment data (n_samples, n_channels)
            f_s: Sampling frequency
            kwargs: Computation parameters
            computed_cache: Cache to store intermediate results for this fragment

        Returns:
            Computed feature value
        """
        # Return cached result if already computed
        if feature in computed_cache:
            return computed_cache[feature]

        # Check if feature has dependencies
        if feature in FragmentAnalyzer.FEATURE_DEPENDENCIES:
            dependencies = FragmentAnalyzer.FEATURE_DEPENDENCIES[feature]

            # Recursively compute all dependencies first
            precomputed_kwargs = kwargs.copy()
            for dependency in dependencies:
                if dependency not in computed_cache:
                    computed_cache[dependency] = FragmentAnalyzer._resolve_feature_dependencies(
                        dependency, fragment_data, f_s, kwargs, computed_cache
                    )
                precomputed_key = f"precomputed_{dependency}"
                precomputed_kwargs[precomputed_key] = computed_cache[dependency]

            # Compute the feature using precomputed dependencies
            func = getattr(FragmentAnalyzer, f"compute_{feature}")
            result = func(rec=fragment_data, f_s=f_s, **precomputed_kwargs)
        else:
            # Base feature with no dependencies - compute directly
            func = getattr(FragmentAnalyzer, f"compute_{feature}")
            result = func(rec=fragment_data, f_s=f_s, **kwargs)

        # Cache and return result
        computed_cache[feature] = result
        return result
