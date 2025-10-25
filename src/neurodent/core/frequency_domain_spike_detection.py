"""
Frequency Domain Spike Detection Module
======================================

This module implements frequency-domain spike detection using Short-Time Fourier Transform (STFT)
and Smoothed Nonlinear Energy Operator (SNEO) for detecting epileptic spikes in EEG recordings.

The algorithm combines spectral analysis with multi-band consensus voting to provide robust
spike detection that is less susceptible to artifacts compared to traditional amplitude-based methods.
"""

import logging
import warnings
from typing import Literal

try:
    import dask
except Exception:  # pragma: no cover
    dask = None

import numpy as np
import mne
from scipy.signal import iirfilter, iirnotch, sosfiltfilt, filtfilt, stft, windows

try:
    import spikeinterface.core as si
    SPIKEINTERFACE_AVAILABLE = True
except ImportError:  # pragma: no cover
    si = None
    SPIKEINTERFACE_AVAILABLE = False


class FrequencyDomainSpikeDetector:
    """
    Static class for frequency-domain spike detection using STFT and SNEO.

    This detector implements a multi-stage pipeline:
    1. Preprocessing (bandpass + notch filtering)
    2. STFT analysis at specific frequency bands
    3. SNEO detection with multi-band consensus voting
    4. Spike refinement and morphological validation
    5. Temporal clustering and deduplication
    """

    # Default detection parameters
    DEFAULT_PARAMS = {
        'bp': [3.0, 40.0],              # bandpass filter Hz
        'notch': 60.0,                  # notch filter Hz
        'notch_q': 30.0,                # notch filter quality factor
        'freq_slices': [10.0, 20.0],    # STFT energy slices Hz
        'window_s': 0.125,              # STFT window size s
        'sneo_percentile': 99.99,       # SNEO threshold percentile
        'cluster_gap_ms': 80.0,         # clustering gap ms
        'search_ms': 160.0,             # spike refinement search window ms
        'baseline_ms': 500.0,           # baseline analysis window ms
        'k_sigma': 3.0,                 # statistical significance threshold
        'smooth_window': 7,             # smoothing kernel size
        'vote_k': 2,                    # minimum votes required for detection
        'smooth_len': 5,                # SNEO smoothing length
    }

    @staticmethod
    def detect_spikes_recording(
        recording: "si.BaseRecording",
        detection_params: dict = None,
        max_length: int = None,
        multiprocess_mode: Literal["dask", "serial"] = "serial"
    ) -> tuple[list[np.ndarray], "mne.io.RawArray"]:
        """
        Detect spikes in a recording using frequency-domain analysis.

        Args:
            recording (si.BaseRecording): The recording to analyze
            detection_params (dict, optional): Detection parameters. Uses DEFAULT_PARAMS if None
            max_length (int, optional): Maximum length in samples to analyze
            multiprocess_mode (Literal["dask", "serial"]): Processing mode

        Returns:
            tuple: (spike_indices_per_channel, mne_raw_with_annotations)
                - spike_indices_per_channel: List of arrays with spike sample indices per channel
                - mne_raw_with_annotations: MNE RawArray with spike annotations
        """
        if not SPIKEINTERFACE_AVAILABLE:
            raise ImportError("SpikeInterface is required for frequency domain spike detection")

        # Merge with default parameters
        params = FrequencyDomainSpikeDetector.DEFAULT_PARAMS.copy()
        if detection_params:
            params.update(detection_params)

        logging.info(f"Detecting spikes with parameters: {params}")
        logging.debug(f"Recording info: {recording}")
        logging.debug(f"Recording channels: {recording.get_channel_ids()}")

        # Get preprocessed recording
        rec_preprocessed = FrequencyDomainSpikeDetector._apply_preprocessing(recording, params)

        # Extract data for analysis
        raw_data = rec_preprocessed.get_traces(return_in_uV=True) # (samples, channels)
        raw_data = raw_data.T # (channels, samples)

        sampling_freq = rec_preprocessed.get_sampling_frequency()
        channel_names = [str(ch_id) for ch_id in rec_preprocessed.get_channel_ids()]
        n_channels = len(channel_names)

        # Apply max_length if specified
        if max_length and raw_data.shape[1] > max_length:
            raw_data = raw_data[:, :max_length]

        # Create MNE RawArray for consistency
        info = mne.create_info(ch_names=channel_names, sfreq=sampling_freq, ch_types='eeg')
        mne_raw = mne.io.RawArray(data=raw_data, info=info)

        # Run spike detection
        match multiprocess_mode:
            case "dask":
                if dask is None:
                    raise ImportError("dask is required for multiprocess_mode='dask'")
                spike_tasks = [
                    dask.delayed(FrequencyDomainSpikeDetector._detect_spikes_channel)(
                        raw_data[ch, :], sampling_freq, params
                    ) for ch in range(n_channels)
                ]
                spike_indices_per_channel = dask.compute(*spike_tasks)
            case "serial":
                spike_indices_per_channel = [
                    FrequencyDomainSpikeDetector._detect_spikes_channel(
                        raw_data[ch, :], sampling_freq, params
                    ) for ch in range(n_channels)
                ]

        # Add spike annotations to MNE object
        mne_raw_with_annotations = FrequencyDomainSpikeDetector._add_spike_annotations(
            mne_raw, spike_indices_per_channel, sampling_freq
        )

        return spike_indices_per_channel, mne_raw_with_annotations

    @staticmethod
    def _apply_preprocessing(recording: "si.BaseRecording", params: dict) -> "si.BaseRecording":
        """Apply bandpass and notch filtering to the recording."""
        rec = recording.clone()

        # Get raw data for scipy filtering (SpikeInterface preprocessing can be complex)
        raw_data = rec.get_traces(return_in_uV=True) # (samples, channels)
        raw_data = raw_data.T # (channels, samples)

        sampling_freq = rec.get_sampling_frequency()

        # Apply bandpass filter
        bp_lo, bp_hi = params['bp']
        sos_bp = iirfilter(N=4, Wn=[bp_lo, bp_hi], btype='bandpass',
                          ftype='butter', output='sos', fs=sampling_freq)
        raw_filtered = sosfiltfilt(sos_bp, raw_data, axis=-1)

        # Apply notch filter
        notch_freq = params['notch']
        notch_q = params['notch_q']
        b_notch, a_notch = iirnotch(w0=notch_freq, Q=notch_q, fs=sampling_freq)
        raw_notch = filtfilt(b_notch, a_notch, raw_filtered, axis=-1)

        # Create new recording with filtered data
        # info = mne.create_info(
        #     ch_names=[str(ch_id) for ch_id in rec.get_channel_ids()],
        #     sfreq=sampling_freq,
        #     ch_types='eeg'
        # )
        # mne_raw = mne.io.RawArray(data=raw_notch, info=info)

        # Convert back to SpikeInterface format
        # Note: This is a simplified approach. In production, might want to use SpikeInterface preprocessing
        # SpikeInterface expects (n_times, n_channels) while our data is (n_channels, n_times)
        filtered_rec = si.NumpyRecording(raw_notch.T, sampling_frequency=sampling_freq, channel_ids=rec.get_channel_ids())

        return filtered_rec

    @staticmethod
    def _detect_spikes_channel(signal: np.ndarray, fs: float, params: dict) -> np.ndarray:
        """
        Detect spikes in a single channel using the STFT+SNEO algorithm.

        Args:
            signal: 1D signal array
            fs: Sampling frequency
            params: Detection parameters dictionary

        Returns:
            spike_indices: Array of spike sample indices
        """
        logging.debug(f"Starting spike detection for channel - signal length: {len(signal)} samples, fs: {fs} Hz")

        # Step 1: STFT analysis
        logging.debug(
            f"Step 1/4: Computing STFT slices at frequencies {params['freq_slices']} Hz with window size {params['window_s']} s"
        )
        slices_dict = FrequencyDomainSpikeDetector._compute_stft_slices(
            signal, fs, freqs=params['freq_slices'], window_s=params['window_s']
        )
        logging.debug(f"Step 1/4: Completed STFT analysis - extracted {len(slices_dict)} frequency bands")

        # Step 2: SNEO detection with multi-band voting
        logging.debug(
            f"Step 2/4: Applying SNEO detection with threshold percentile {params['sneo_percentile']} and vote_k={params['vote_k']}"
        )
        sneo_spikes, _ = FrequencyDomainSpikeDetector._apply_sneo_on_slices(
            slices_dict, fs,
            threshold_percentile=params['sneo_percentile'],
            smooth_len=params['smooth_len'],
            vote_k=params['vote_k']
        )
        logging.info(f"Step 2/4: SNEO detection found {len(sneo_spikes)} candidate spikes")

        # Step 3: Spike refinement and morphological validation
        logging.debug(
            f"Step 3/4: Refining spikes with morphological validation (search_ms={params['search_ms']}, k_sigma={params['k_sigma']})"
        )
        neg_spikes = FrequencyDomainSpikeDetector._enforce_downward_and_refine_minimal(
            signal, fs, sneo_spikes,
            search_ms=params['search_ms'],
            baseline_ms=params['baseline_ms'],
            k_sigma=params['k_sigma'],
            smooth_window=params['smooth_window']
        )
        logging.info(f"Step 3/4: After refinement and validation: {len(neg_spikes)} spikes retained")

        # Step 4: Temporal clustering and deduplication
        logging.debug(f"Step 4/4: Clustering and deduplicating spikes (min_gap_ms={params['cluster_gap_ms']})")
        final_spikes = FrequencyDomainSpikeDetector._filter_close_spikes_by_min_local(
            signal, fs, neg_spikes,
            min_gap_ms=params['cluster_gap_ms'],
            window_ms=60.0,
            smooth_window=5
        )
        logging.info(f"Step 4/4: Final spike count after clustering: {len(final_spikes)} spikes")

        return final_spikes.astype(int)

    @staticmethod
    def _compute_stft_slices(signal, fs, freqs=(40, 60), window='hann', window_s=0.125, noverlap=None):
        """
        Compute STFT and extract power at specific frequency bands.

        Args:
            signal: Input signal
            fs: Sampling frequency
            freqs: Frequency points to extract
            window: Window function for STFT
            nperseg: Length of each segment (defaults to ~125ms)
            noverlap: Number of points to overlap

        Returns:
            dict: Frequency -> energy time series mapping
        """
        N = len(signal)
        logging.debug(f"Computing STFT slices - signal length: {N} samples, fs: {fs} Hz")

        # Default window length informed by expected spike width (~125 ms)
        nperseg = int(round(fs * window_s))
        logging.debug(
            f"STFT parameters: window={window}, nperseg={nperseg} samples ({window_s} s), noverlap={noverlap}"
        )

        f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        t_samp = (t * fs).astype(float)

        logging.debug(f"STFT output dimensions: frequency bins={len(f)}, time bins={len(t)}, Zxx shape={Zxx.shape}")
        logging.debug(f"Frequency range: {f[0]:.2f} - {f[-1]:.2f} Hz, frequency resolution: {f[1] - f[0]:.2f} Hz")
        logging.debug(
            f"Time range: {t[0]:.4f} - {t[-1]:.4f} s, STFT time samples: {len(t_samp)}, Original signal samples: {N}"
        )

        slices_dict = {}
        pow_spec = np.abs(Zxx) ** 2

        for f0 in freqs:
            idx = int(np.argmin(np.abs(f - f0)))
            actual_freq = f[idx]
            energy = pow_spec[idx]
            logging.debug(
                f"Extracting frequency slice: target={f0} Hz, actual={actual_freq:.2f} Hz (index={idx}), energy length={len(energy)}"
            )

            # REVIEW the output of stft should already be the correct length, so unsure why interpolation is required
            # Interpolate back to original sampling rate
            energy_resampled = np.interp(np.arange(N), t_samp, energy, left=energy[0], right=energy[-1])
            logging.debug(f"Interpolated energy from {len(energy)} to {len(energy_resampled)} samples")
            logging.debug(
                f"Energy stats - original: min={energy.min():.2e}, max={energy.max():.2e}, mean={energy.mean():.2e}"
            )
            logging.debug(
                f"Energy stats - resampled: min={energy_resampled.min():.2e}, max={energy_resampled.max():.2e}, mean={energy_resampled.mean():.2e}"
            )

            slices_dict[float(f0)] = energy_resampled

        logging.debug(f"STFT slices computed for {len(slices_dict)} frequency bands")
        return slices_dict

    @staticmethod
    def _sneo(x):
        """Smoothed Nonlinear Energy Operator (SNEO)."""
        return x[1:-1]**2 - x[2:] * x[:-2]

    @staticmethod
    def _apply_sneo_on_slices(slice_dict, fs, threshold_percentile=99.9, smooth_len=5, vote_k=2):
        """
        Apply SNEO on frequency slices with multi-band consensus voting.

        Args:
            slice_dict: Dictionary of frequency -> energy arrays
            fs: Sampling frequency
            threshold_percentile: Percentile threshold for detection
            smooth_len: Smoothing window length
            vote_k: Minimum number of bands that must agree

        Returns:
            tuple: (final_spikes, sneo_combined)
        """
        sneo_len = len(next(iter(slice_dict.values()))) - 2 # REVIEW fragile hardcoded -2, what if the energy operation doesn't reduce the length by 2?
        sneo_combined = np.zeros(sneo_len)
        detections = []

        # Bartlett window for smoothing
        win = windows.bartlett(smooth_len) if smooth_len > 1 else None # REVIEW I wonder if this can be replaced with a filter function -- becomes independent of sample frequency

        for _, slice_ in slice_dict.items():
            s = FrequencyDomainSpikeDetector._sneo(slice_)

            # Apply smoothing if specified
            if win is not None:
                if len(s) >= len(win):
                    s = np.convolve(s, win, mode='same')
                else:
                    warnings.warn(f"Skipping smoothing, smoothing window length {len(win)} is greater than the length of the SNEO signal {len(s)}")

            sneo_combined += s

            # Threshold detection
            thr = np.percentile(s, threshold_percentile)
            # REVIEW +1 seems hardcoded -- better alternative would be to extrapolate the SNEO signal to the original length
            # or an implementation of the SNEO algorithm that preserves signal length
            cand = np.where(s > thr)[0] + 1  # +1 to account for SNEO shift
            detections.append(cand)

        if not detections:
            return np.array([]), sneo_combined

        # Multi-band consensus voting
        all_cand = np.concatenate(detections)
        unique, counts = np.unique(all_cand, return_counts=True)
        final_spikes = unique[counts >= vote_k]

        return final_spikes, sneo_combined

    @staticmethod
    def _enforce_downward_and_refine_minimal(
        signal, fs, candidates,
        search_ms=160, baseline_ms=500, k_sigma=3.0, smooth_window=7
    ):
        """
        Refine spike candidates by enforcing downward deflection and statistical significance.

        Args:
            signal: Input signal
            fs: Sampling frequency
            candidates: Initial spike candidates
            search_ms: Search window around candidate (ms)
            baseline_ms: Baseline analysis window (ms)
            k_sigma: Statistical significance threshold
            smooth_window: Smoothing window size

        Returns:
            refined_spikes: Array of refined spike indices
        """
        if len(candidates) == 0:
            return np.array([])

        N = len(signal)
        half = int(round(fs * (search_ms / 1e3) / 2)) # REVIEW rename these variables -- spike search window
        base_half = int(round(fs * (baseline_ms / 1e3) / 2)) # REVIEW rename these variables -- window size to use as a baseline comparison
        out = []

        for c in np.asarray(candidates, dtype=int):
            # Extract search window
            L = max(0, c - half)
            R = min(N, c + half + 1)
            w = signal[L:R]

            # Apply smoothing
            if smooth_window and smooth_window > 1 and len(w) >= smooth_window:
                pad = smooth_window // 2
                wpad = np.pad(w, (pad, pad), mode='edge')
                kern = np.ones(smooth_window) / smooth_window # REVIEW I wonder if this can be replaced with a filter function -- becomes independent of sample frequency
                w = np.convolve(wpad, kern, mode='valid')

            # Find minimum (most negative deflection)
            rel_min = np.argmin(w)
            idx = L + rel_min

            # Extract baseline for statistical analysis
            B0 = max(0, idx - base_half)
            B1 = min(N, idx + base_half)
            base = signal[B0:B1]

            if len(base) < 10:
                warnings.warn(f"Skipping baseline analysis for spike at index {idx}, baseline window length {len(base)} is less than 10")
                continue

            # Exclude spike region from baseline
            exc0 = max(0, idx - half) - B0
            exc1 = min(N, idx + half + 1) - B0
            base_mask = np.ones(len(base), dtype=bool)
            base_mask[exc0:exc1] = False
            baseline_without_spike = base[base_mask] if base_mask.any() else base

            # Statistical significance test using robust statistics
            baseline_median = np.median(baseline_without_spike)
            baseline_mad = np.median(np.abs(baseline_without_spike - baseline_median)) + 1e-12  # REVIEW should this epsilon be a parameter?

            if signal[idx] <= baseline_median - k_sigma * 1.4826 * baseline_mad:
                # Morphological validation: check derivative pattern
                # REVIEW this appears hardcoded. Checks for a negative then positive derivative but
                # what if the before/after is noisy but there's still a trend?
                # maybe better to fit a line to the before/after and check slopes
                d = np.diff(signal[max(0, idx-3):min(N, idx+4)])
                if d.size >= 2 and (np.any(d[:max(1,len(d)//2)] < 0) and np.any(d[max(1,len(d)//2):] > 0)):
                    out.append(idx)

        return np.array(sorted(set(out)))

    @staticmethod
    def _filter_close_spikes_by_min_local(
        signal, fs, spike_indices, min_gap_ms=20, window_ms=60, smooth_window=5
    ):
        """
        Filter out closely spaced spikes by selecting the most negative in each cluster.

        Args:
            signal: Input signal
            fs: Sampling frequency
            spike_indices: Array of spike indices
            min_gap_ms: Minimum gap between spikes (ms)
            window_ms: Window for local optimization (ms)
            smooth_window: Smoothing window size

        Returns:
            filtered_spikes: Array of filtered spike indices
        """
        if len(spike_indices) == 0:
            return np.array([])

        spike_indices = np.sort(np.asarray(spike_indices, dtype=int))
        min_gap = int(round(fs * (min_gap_ms / 1e3)))
        half_w = int(round(fs * (window_ms / 1e3) / 2))
        N = signal.shape[-1]

        # Group nearby spikes
        groups, current_cluster = [], [spike_indices[0]]
        for idx in spike_indices[1:]:
            if idx - current_cluster[-1] <= min_gap:
                current_cluster.append(idx)
            else:
                groups.append(current_cluster)
                current_cluster = [idx]
        groups.append(current_cluster)

        # Select best spike from each group
        chosen = []
        for g in groups:
            g = np.asarray(g)
            L = max(0, g.min() - half_w)
            R = min(N, g.max() + half_w + 1)
            w = signal[L:R]

            # Apply smoothing
            if smooth_window and smooth_window > 1 and len(w) >= smooth_window:
                pad = smooth_window // 2
                wpad = np.pad(w, (pad, pad), mode='edge')
                kern = np.ones(smooth_window, dtype=float) / smooth_window # REVIEW I wonder if this can be replaced with a filter function -- becomes independent of sample frequency
                w = np.convolve(wpad, kern, mode='valid')

            rel_min = int(np.argmin(w))
            best_idx = L + rel_min
            chosen.append(best_idx)

        return np.array(sorted(set(chosen)), dtype=int)

    @staticmethod
    def _add_spike_annotations(mne_raw, spike_indices_per_channel, sampling_freq):
        """
        Add spike annotations to MNE RawArray object.

        Args:
            mne_raw: MNE RawArray object
            spike_indices_per_channel: List of spike indices per channel
            sampling_freq: Sampling frequency

        Returns:
            mne_raw_with_annotations: MNE RawArray with spike annotations added
        """
        spike_annotations = []

        for ch_idx, channel_spikes in enumerate(spike_indices_per_channel):
            if len(channel_spikes) > 0:
                # Convert sample indices to seconds
                spike_times_seconds = (channel_spikes + 1) / sampling_freq # REVIEW why is +1 here?
                for spike_time in spike_times_seconds:
                    spike_annotations.append({
                        'onset': float(spike_time),
                        'duration': 0.0,
                        'description': f'Spike_Ch{ch_idx}'
                    })

        if spike_annotations:
            # Sort by onset time
            spike_annotations.sort(key=lambda x: x['onset'])
            annotations = mne.Annotations(
                onset=[a['onset'] for a in spike_annotations],
                duration=[a['duration'] for a in spike_annotations],
                description=[a['description'] for a in spike_annotations]
            )
            mne_raw = mne_raw.copy().set_annotations(annotations)

        return mne_raw