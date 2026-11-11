"""Analysis stage for a single animal.

``AnimalAnalyzer`` runs the LOF / windowed-analysis / spike-detection steps on an
:class:`~neurodent.loading.animal_organizer.AnimalOrganizer`'s already-loaded
recordings. It reads the organizer's recordings and metadata and stores the
results on itself, keeping loading (``AnimalOrganizer``) and analysis separate.

Construct it as ``AnimalAnalyzer(ao)`` and reuse one instance across the
``compute_*`` calls so intermediate state (e.g. bad channels) is shared, the same
way you reuse a :class:`~neurodent.analysis.long_recording_analyzer.LongRecordingAnalyzer`.
"""

import logging
import warnings
from typing import Literal, Optional

import dask
import dask.array as da
import pandas as pd
from dask import delayed
from tqdm import tqdm

from neurodent import constants
from .long_recording_analyzer import LongRecordingAnalyzer
from neurodent.core.utils import validate_timestamps, is_day
from neurodent.core import utils as core_utils
from .fragment_analyzer import FragmentAnalyzer
from .spike_detection import FrequencyDomainSpikeDetector
from neurodent.results import WindowAnalysisResult, _sanitize_feature_request

try:
    import spikeinterface.preprocessing as spre
except ImportError:  # pragma: no cover
    spre = None


class AnimalAnalyzer:
    """Runs analysis on a loaded :class:`AnimalOrganizer`, owning the results.

    Analysis state (``bad_channels_dict``, ``features_df``, ``window_analysis_result``,
    ``frequency_domain_spike_analysis_results``) is stored on the instance, so reuse a
    single ``AnimalAnalyzer(ao)`` across ``compute_bad_channels`` /
    ``compute_windowed_analysis`` / ``compute_frequency_domain_spike_analysis``. LOF scores
    are kept on the organizer's recordings, and ``compute_windowed_analysis`` records them in
    the result's ``lof_scores_dict``. The result's ``bad_channels_dict`` always starts empty:
    the channels a LOF threshold picks stay in this instance's ``bad_channels_dict``, and a
    result gives them with ``get_bad_channels_by_lof_threshold``. The result used to start
    with this instance's ``bad_channels_dict``, so a threshold given to
    ``compute_bad_channels`` put LOF's picks into it.

    Args:
        ao (AnimalOrganizer): A loaded organizer whose recordings are analyzed.
    """

    def __init__(self, ao):
        self.ao = ao
        self.long_analyzers: list[LongRecordingAnalyzer] = []
        self.bad_channels_dict = {}
        self.features_df = pd.DataFrame()
        self.window_analysis_result = None
        self.frequency_domain_spike_analysis_results = None

    def compute_bad_channels(
        self, lof_threshold: float = None, force_recompute: bool = False,
        lof_chunk_duration_s: float = 60,
    ):
        """Compute bad channels using LOF analysis for all recordings.

        The scores are recorded in the result of :meth:`compute_windowed_analysis`; the channels
        a threshold picks are kept in ``self.bad_channels_dict`` only, not in that result.

        Args:
            lof_threshold (float, optional): Threshold for determining bad channels from LOF scores.
                                           If None, only computes/loads scores without setting bad_channel_names.
            force_recompute (bool): Whether to recompute LOF scores even if they exist.
            lof_chunk_duration_s (float): Duration in seconds of each chunk used
                for the pairwise-distance computation in LOF.  Defaults to 60.
        """
        logging.info(
            f"Computing bad channels for {len(self.ao.long_recordings)} recordings with threshold={lof_threshold}"
        )
        for i, lrec in self.ao._iter_valid_recordings():
            logging.debug(
                f"Computing bad channels for recording {i}: {self.ao.animaldays[i]}"
            )
            lrec.compute_bad_channels(
                lof_threshold=lof_threshold, force_recompute=force_recompute,
                lof_chunk_duration_s=lof_chunk_duration_s,
            )
            logging.debug(
                f"Recording {i} LOF scores computed: {hasattr(lrec, 'lof_scores') and lrec.lof_scores is not None}"
            )

        # Update bad channels dict if threshold was applied
        if lof_threshold is not None:
            self.bad_channels_dict = {
                (getattr(lrec, "animalday", None) or animalday): lrec.bad_channel_names
                for animalday, lrec in zip(self.ao.animaldays, self.ao.long_recordings)
            }

    def apply_lof_threshold(self, lof_threshold: float):
        """Apply threshold to existing LOF scores to determine bad channels for all recordings.

        Args:
            lof_threshold (float): Threshold for determining bad channels.
        """
        for lrec in self.ao.long_recordings:
            lrec.apply_lof_threshold(lof_threshold)

        self.bad_channels_dict = {
            (getattr(lrec, "animalday", None) or animalday): lrec.bad_channel_names
            for animalday, lrec in zip(self.ao.animaldays, self.ao.long_recordings)
        }

    def get_all_lof_scores(self) -> dict:
        """Get LOF scores for all recordings.

        Returns:
            dict: Dictionary mapping animal days to LOF score dictionaries.
        """
        return {
            animalday: lrec.get_lof_scores()
            for animalday, lrec in zip(self.ao.animaldays, self.ao.long_recordings)
        }

    def compute_windowed_analysis(
        self,
        features: list[str],
        exclude: list[str] = [],
        window_s=5,
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        suppress_short_interval_error=False,
        apply_notch_filter=True,
        chunk_duration_s: Optional[float] = 3600,
        **kwargs,
    ) -> "WindowAnalysisResult":
        """Computes windowed analysis of animal recordings. The data is divided into windows (time bins), then features are extracted from each window. The result is
        formatted to a Dataframe and wrapped into a WindowAnalysisResult object.

        Args:
            features (list[str]): List of features to compute. See individual ``compute_...()`` functions for output format
            exclude (list[str], optional): List of features to ignore. Will override the features parameter. Defaults to [].
            window_s (int, optional): Length of each window in seconds. Note that some features break with very short window times. Defaults to 5.
            multiprocess_mode (Literal["dask", "serial"], optional): Whether to compute windows in parallel with Dask, one task per window, or serially.
                Process-level parallelism requires an active ``dask.distributed`` Client. Both modes analyse the same windows, the complete
                windows of each recording, floor(samples / window). The samples after the last complete window are logged and not analysed,
                and a recording shorter than one window gives no rows and a warning. Before, the Dask path dropped the last window
                unconditionally, so it gave one window fewer than "serial" when a recording was an exact multiple of the window length, and
                "serial" kept a partial last window whose PSD it interpolated. Defaults to "serial".
            suppress_short_interval_error (bool, optional): If True, suppress ValueError for short intervals between timestamps in resulting WindowAnalysisResult. Useful for aggregated WARs. Defaults to False.
            apply_notch_filter (bool, optional): Whether to apply notch filtering to remove line noise. Uses constants.LINE_FREQ. Defaults to True.
            chunk_duration_s (float, optional): Duration in seconds of data to hold
                in memory at once during the Dask processing path.  Internally
                converted to a number of fragments via
                ``int(chunk_duration_s / window_s)``.  When ``None``,
                all fragments are loaded into a single NumPy array before being
                written to the intermediate zarr store — the original behavior,
                which maximizes throughput but requires enough RAM to hold the
                entire recording at once.  When set to a positive value, only the
                corresponding number of fragments are buffered at a time, streaming
                them to zarr incrementally; use a small value (e.g. 250) on
                memory-constrained machines and a larger value (e.g. 2500+) on
                high-memory nodes for maximum throughput.  Only has an effect when
                ``multiprocess_mode="dask"``.  Defaults to 3600.

        Note:
            The Dask path caches each recording's windows in a temporary zarr store in the
            temp directory, named by ``core.utils.zarr_cache_path``, and removes it on every
            exit: after the computation, on an error, and on the ``SystemExit`` that
            ``setup_snakemake_logging`` raises when SLURM sends SIGTERM at a job's time limit.
            Before, it was removed only when the computation succeeded.

        Raises:
            AttributeError: If a feature's ``compute_...()`` function was not implemented, this error will be raised.
            ValueError: If no recording of the animal is as long as one window.

        Returns:
            WindowAnalysisResult: A WindowAnalysisResult object containing extracted features for all recordings,
            the LOF scores of every recording that has them, and an empty ``bad_channels_dict``. The
            ``bad_channels_dict`` used to be this analyzer's, holding any LOF threshold's picks.
        """
        features = _sanitize_feature_request(features, exclude)

        self.ao._validate_sampling_rates()

        dataframes = []
        for _i, lrec in self.ao._iter_valid_recordings():
            logging.info(f"Computing windowed analysis for {lrec.display_name}")
            lan = LongRecordingAnalyzer(
                lrec, fragment_len_s=window_s, apply_notch_filter=apply_notch_filter
            )
            # Both modes analyse the same complete windows, cut at this sample count.
            n_samples_per_frag = lrec.get_fragment_num_samples(window_s)
            n_leftover = lrec.LongRecording.get_num_frames() - lan.n_fragments * n_samples_per_frag
            if lan.n_fragments == 0:
                logging.warning(
                    f"{lrec.display_name} has {n_leftover} samples, fewer than one {window_s} s "
                    f"window of {n_samples_per_frag}, so it gives no rows."
                )
                continue
            if n_leftover:
                logging.info(
                    f"{lrec.display_name}: {lan.n_fragments} windows of {window_s} s. The last "
                    f"{n_leftover} samples ({n_leftover / lan.f_s:.3f} s) are less than a window "
                    f"and are not analysed."
                )

            logging.debug(f"Processing {lan.n_fragments} fragments")
            miniters = int(lan.n_fragments / 100)
            match multiprocess_mode:
                case "dask":
                    logging.debug("Converting LongRecording to numpy array")

                    n_fragments_war = lan.n_fragments

                    # Apply notch filter once to the entire recording (lazy SI wrapper)
                    rec = lrec.LongRecording
                    if lan.apply_notch_filter:
                        if spre is not None:
                            rec = spre.notch_filter(rec, freq=constants.LINE_FREQ)
                        else:
                            logging.warning(
                                "apply_notch_filter=True but spikeinterface.preprocessing "
                                "is not available; notch filter will be skipped."
                            )

                    if chunk_duration_s is not None:
                        # Convert seconds → number of fragments
                        n_frag_per_chunk = max(1, int(chunk_duration_s / window_s))
                        # Streaming path: stream recording to zarr in batches,
                        # keeping only `n_frag_per_chunk` fragments in RAM at a time.
                        tmppath = core_utils.stream_recording_to_zarr(
                            rec,
                            n_fragments_war,
                            n_samples_per_frag,
                            n_frag_per_chunk,
                        )
                    else:
                        # Default path: read all traces at once then write to zarr.
                        # Maximises throughput on high-memory systems.
                        total_samples = n_fragments_war * n_samples_per_frag
                        all_traces = rec.get_traces(
                            start_frame=0,
                            end_frame=total_samples,
                            return_scaled=True,
                        )
                        np_fragments = all_traces.reshape(
                            n_fragments_war, n_samples_per_frag, rec.get_num_channels()
                        )
                        logging.debug(f"np_fragments.shape: {np_fragments.shape}")
                        # Cache fragments to zarr
                        tmppath, _ = core_utils.cache_fragments_to_zarr(
                            np_fragments, n_fragments_war
                        )
                        del all_traces, np_fragments

                    try:
                        logging.debug("Processing metadata serially")
                        metadatas = [
                            self._process_fragment_metadata(idx, lan, window_s)
                            for idx in range(n_fragments_war)
                        ]
                        meta_df = pd.DataFrame(metadatas)

                        logging.debug("Processing features in parallel")
                        np_fragments_reconstruct = da.from_zarr(
                            tmppath, chunks=("auto", -1, -1)
                        )
                        logging.debug(f"Dask array shape: {np_fragments_reconstruct.shape}")
                        logging.debug(
                            f"Dask array chunks: {np_fragments_reconstruct.chunks}"
                        )

                        # Create delayed tasks for each fragment using efficient dependency resolution
                        feature_values = [
                            delayed(FragmentAnalyzer.process_fragment_with_dependencies)(
                                np_fragments_reconstruct[idx], lan.f_s, features, kwargs
                            )
                            for idx in range(n_fragments_war)
                        ]

                        # Compute features in parallel
                        feature_values = dask.compute(*feature_values)
                    finally:
                        # On every exit, including an error or the SystemExit that
                        # setup_snakemake_logging raises on SIGTERM
                        logging.debug("Removing the zarr cache")
                        core_utils.remove_zarr_cache(tmppath)

                    logging.debug("Combining metadata and feature values")
                    feat_df = pd.DataFrame(feature_values)
                    lan_df = pd.concat([meta_df, feat_df], axis=1)

                case _:
                    logging.debug("Processing serially")
                    lan_df = []
                    for idx in tqdm(
                        range(lan.n_fragments),
                        desc="Processing rows",
                        miniters=miniters,
                    ):
                        lan_df.append(
                            self._process_fragment_serial(
                                idx, features, lan, window_s, kwargs
                            )
                        )

            lan_df = pd.DataFrame(lan_df)

            logging.debug("Validating timestamps")
            validate_timestamps(lan_df["timestamp"].tolist())
            lan_df = lan_df.sort_values("timestamp").reset_index(drop=True)

            self.long_analyzers.append(lan)
            dataframes.append(lan_df)

        if not dataframes:
            raise ValueError(
                f"No recording of {self.ao.animal_id} is as long as one {window_s} s window, so "
                f"there is no window to analyse."
            )
        self.features_df = pd.concat(dataframes)

        # Collect LOF scores from long recordings
        lof_scores_dict = {}
        missing_lof_animaldays = []
        for animalday, lrec in zip(self.ao.animaldays, self.ao.long_recordings):
            animalday = getattr(lrec, "animalday", None) or animalday  # canonical single-source key
            logging.debug(
                f"Checking LOF scores for {animalday}: has_attr={hasattr(lrec, 'lof_scores')}, "
                f"is_not_none={getattr(lrec, 'lof_scores', None) is not None}"
            )
            if hasattr(lrec, "lof_scores") and lrec.lof_scores is not None:
                lof_scores_dict[animalday] = {
                    "lof_scores": lrec.lof_scores.tolist(),
                    "channel_names": lrec.channel_names,
                }
                logging.info(
                    f"Added LOF scores for {animalday}: {len(lrec.lof_scores)} channels"
                )
            else:
                missing_lof_animaldays.append(animalday)
                logging.warning(
                    f"Missing LOF scores for {animalday}! LOF computation may have failed or "
                    f"compute_bad_channels() was not called for this LRO."
                )

        logging.info(f"Total LOF scores collected: {len(lof_scores_dict)} animal days")

        # Warn loudly if any animaldays are missing LOF scores
        if missing_lof_animaldays:
            warning_msg = (
                f"WARNING: {len(missing_lof_animaldays)} animalday(s) are missing LOF scores: {missing_lof_animaldays}. "
                f"Expected {len(self.ao.animaldays)} but got {len(lof_scores_dict)}. "
                f"These sessions will be auto-populated with empty placeholders and excluded from LOF-based analysis."
            )
            logging.warning(warning_msg)
            warnings.warn(warning_msg)

        # The result starts with no bad channels; a channel filter records what it masks
        self.window_analysis_result = WindowAnalysisResult(
            self.features_df,
            self.ao.animal_id,
            self.ao.genotype,
            self.ao.sex,
            self.ao.channel_names,
            {},
            suppress_short_interval_error,
            lof_scores_dict,
        )

        return self.window_analysis_result

    def compute_frequency_domain_spike_analysis(
        self,
        detection_params: dict = None,
        chunk_duration_s: float = 3600,
        multiprocess_mode: Literal["dask", "serial"] = "serial",
    ):
        """
        Compute frequency-domain spike detection on all long recordings.

        Args:
            detection_params (dict, optional): Detection parameters. Uses defaults if None.
            chunk_duration_s (float): Duration in seconds of each
                processing chunk.  Defaults to 3600 (1 hour).  The full
                recording is always analysed; this parameter controls peak RAM
                by processing in overlapping chunks.  ``None`` loads the full
                recording at once (fastest).
            multiprocess_mode (Literal["dask", "serial"]): Whether to detect spikes in
                parallel with Dask or serially. Defaults to "serial".

        Returns:
            list[FrequencyDomainSpikeAnalysisResult]: Results for each recording session

        Raises:
            ImportError: If SpikeInterface is not available
        """
        # Import here to avoid circular imports
        from neurodent.results import FrequencyDomainSpikeAnalysisResult

        fdsar_list = []

        logging.info(
            f"Running frequency-domain spike detection on {len(self.ao.long_recordings)} recordings"
        )
        logging.info(f"Detection parameters: {detection_params}")

        for i, lrec in self.ao._iter_valid_recordings():
            rec = lrec.LongRecording

            try:
                # Run frequency domain spike detection
                spike_indices_per_channel = (
                    FrequencyDomainSpikeDetector.detect_spikes_recording(
                        rec,
                        detection_params=detection_params,
                        chunk_duration_s=chunk_duration_s,
                        multiprocess_mode=multiprocess_mode,
                    )
                )

                # Create FrequencyDomainSpikeAnalysisResult
                fdsar = FrequencyDomainSpikeAnalysisResult.from_detection_results(
                    spike_indices_per_channel=spike_indices_per_channel,
                    recording=rec,
                    detection_params=detection_params or {},
                    animal_id=self.ao.animal_id,
                    genotype=self.ao.genotype,
                    # The name the WAR gives this recording's rows (see
                    # _process_fragment_metadata), so read_sars_spikes can match spikes to rows
                    # by name. ao.animaldays agrees with it on the pipeline's from_lros path
                    # but not on every construction path.
                    animal_day=getattr(lrec, "animalday", None) or self.ao.animaldays[i],
                    bin_folder_name=None,
                    metadata=self.ao.long_recordings[i].meta,
                )

                fdsar_list.append(fdsar)

                # Log results
                total_spikes = sum(len(spikes) for spikes in spike_indices_per_channel)
                logging.info(
                    f"Recording {i + 1}/{len(self.ao.long_recordings)}: Detected {total_spikes} spikes across {len(spike_indices_per_channel)} channels"
                )

            except Exception as e:
                raise RuntimeError(
                    f"Failed processing recording {i + 1}/{len(self.ao.long_recordings)}"
                ) from e

        # Store results for later access
        self.frequency_domain_spike_analysis_results = fdsar_list

        logging.info(
            f"Completed frequency-domain spike detection. Total recordings processed: {len(fdsar_list)}"
        )
        return fdsar_list

    def _process_fragment_serial(
        self, idx, features, lan: LongRecordingAnalyzer, window_s, kwargs: dict
    ):
        row = self._process_fragment_metadata(idx, lan, window_s)
        row.update(self._process_fragment_features(idx, features, lan, kwargs))
        return row

    def _process_fragment_metadata(
        self, idx, lan: LongRecordingAnalyzer, window_s
    ):
        row = {}

        # Build session labels from LRO's DiscoveredFile metadata
        from neurodent.loading import DiscoveredFile

        lro = lan.LongRecording
        item = getattr(lro, "item", None)

        animal = self.ao.animal_id or "unknown"
        genotype = self.ao.genotype or "Unknown"
        sex = self.ao.sex or "Unknown"
        session = None

        if isinstance(item, DiscoveredFile) and item.metadata:
            meta = item.metadata
            animal = meta.get("animal", animal)
            session = meta.get("session")
            genotype = constants.ANIMAL_METADATA.get(animal, {}).get("genotype", genotype)
            sex = constants.ANIMAL_METADATA.get(animal, {}).get("sex", sex)

        if session is None:
            try:
                session = lro.get_date_string()
            except (ValueError, AttributeError):
                session = "unknown"

        # Single source of truth: read the canonical animalday stamped on the LRO (in ao_build's
        # from_lros / _create_long_recordings) so the WAR column matches ``ao.animaldays`` and the
        # LOF/bad-channel dicts exactly. Fall back to the legacy derivation only for unstamped LROs.
        row["animalday"] = getattr(lro, "animalday", None) or f"{animal} {genotype} {session}"
        row["animal"] = animal
        row["day"] = session
        row["genotype"] = genotype
        row["sex"] = sex
        row["duration"] = lan.LongRecording.get_dur_fragment(window_s, idx)
        row["endfile"] = lan.get_file_end(idx)

        frag_dt = lan.LongRecording.get_datetime_fragment(window_s, idx)
        row["timestamp"] = frag_dt
        row["isday"] = is_day(frag_dt)

        return row

    def _process_fragment_features(
        self, idx, features, lan: LongRecordingAnalyzer, kwargs: dict
    ):
        row = {}
        for feat in features:
            func = getattr(lan, f"compute_{feat}")
            if callable(func):
                row[feat] = func(idx, **kwargs)
            else:
                raise AttributeError(f"Invalid function {func}")
        return row
