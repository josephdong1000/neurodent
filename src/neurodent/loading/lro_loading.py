"""Loading/decode: detect format, run SI/MNE extractors, resample, build metadata.

Mixin for :class:`~neurodent.loading.long_recording_organizer.LongRecordingOrganizer`.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Literal, Union

try:
    import dask
except Exception:  # pragma: no cover - optional at import time
    dask = None

import mne

import numpy as np

try:
    import spikeinterface.core as si
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre
except Exception:  # pragma: no cover - optional at import time
    si = None
    se = None
    spre = None

from .. import constants
from neurodent.core.utils import (
    assert_microvolts,
    extract_mne_unit_info,
    get_temp_directory,
    should_use_cache_unified,
    get_cache_status_message,
    rename_mne_channels,
    atomic_output_path,
    safe_unlink,
)
from .recording_metadata import RecordingMetadata

# What an unreadable cache actually raises. The readers do not share an exception hierarchy, so this
# is measured, not assumed:
#   se.read_edf    -> RuntimeError ("not EDF(+) compliant"), OSError (garbage/empty), FileNotFoundError
#   se.read_binary -> ValueError ("truncated binary file"), FileNotFoundError; and it silently returns
#                     a SHORT recording for some truncations -- see _bin_is_intact
#   from_json      -> JSONDecodeError (malformed), FileNotFoundError, KeyError (stale schema)
#
# The point of listing them is not brevity but exclusion: TypeError, AttributeError and NameError --
# the signatures of a bug in OUR code -- are absent, so they propagate instead of being laundered into
# "the cache was corrupt, regenerating", which deletes good files and hides the fault forever (the
# regenerated file hits the same bug on the next run).
CACHE_DATA_ERRORS = (OSError, ValueError, RuntimeError)
CACHE_METADATA_ERRORS = (OSError, json.JSONDecodeError, KeyError)


def _bin_is_intact(fname: Path, metadata: RecordingMetadata, itemsize: int = 8) -> bool:
    """Whether a cached binary looks complete.

    ``se.read_binary`` is memmap-backed and does not raise on a truncated or empty file: it simply
    reports fewer frames. So corruption there cannot be caught by a ``try``, and has to be measured.
    """
    n_bytes = fname.stat().st_size
    frame_bytes = metadata.n_channels * itemsize
    return n_bytes > 0 and frame_bytes > 0 and n_bytes % frame_bytes == 0


def _gain_to_uV(metadata: RecordingMetadata) -> float:
    """SpikeInterface ``gain_to_uV`` for a binary intermediate.

    The bin holds the source's native units (volts, for MNE), so the gain is ``mult_to_uV``. Falls
    back to 1.0 when the source units are unknown.
    """
    mult = getattr(metadata, "mult_to_uV", None)
    if mult is None or not np.isfinite(mult) or mult <= 0:
        logging.warning(
            "No usable mult_to_uV (got %r); assuming the binary intermediate is already in uV.", mult
        )
        return 1.0
    return float(mult)


class LroLoadingMixin:
    """Mixin: see module docstring."""

    def _validate_units_uV(self, recording, n_frames: int = 100_000) -> None:
        """Raise if a newly-constructed recording is not plausibly in µV.

        Filters, features and plots all take ``get_traces(return_scaled=True)`` at its word, so a unit
        slip corrupts them while still looking plausible. Samples the head rather than reading the
        whole recording. Non-recordings are skipped by type check, not by catching their errors.
        """
        base = getattr(si, "BaseRecording", None)
        if not isinstance(base, type) or not isinstance(recording, base):
            return  # SpikeInterface absent, or not a real recording (a test double)

        kwargs = {"segment_index": 0} if recording.get_num_segments() > 1 else {}
        traces = recording.get_traces(
            start_frame=0, end_frame=n_frames, return_scaled=True, **kwargs
        )
        assert_microvolts(np.asarray(traces, dtype=float), context=f"{type(self).__name__} recording")

    @staticmethod
    def _extract_channel_names(recording: "si.BaseRecording") -> list[str]:
        """Extract human-readable channel names from a SpikeInterface recording.

        Prefers the ``channel_name`` property (set by extractors like
        ``read_edf``) over raw channel IDs, which are often opaque
        integer indices.

        Args:
            recording: A SpikeInterface recording.

        Returns:
            List of channel name strings.
        """
        try:
            prop_keys = recording.get_property_keys()
            if "channel_name" in prop_keys:
                names = recording.get_property("channel_name")
                return [str(n) for n in names]
        except (AttributeError, TypeError):
            pass

        raw_ids = recording.get_channel_ids()
        if len(raw_ids) > 0 and isinstance(raw_ids[0], (int, np.integer)):
            logging.warning("Channel IDs are integers. Converting to strings.")
        return [str(ch) for ch in raw_ids]

    def _init_from_recording(self, recording: "si.BaseRecording"):
        """Initialize LRO from an existing SpikeInterface recording object (in-memory)."""
        # Enforce global dtype and resampling
        self.LongRecording = self._apply_resampling(recording)
        self._validate_units_uV(self.LongRecording)
        recording = self.LongRecording

        self._is_in_memory = True

        # Extract metadata from recording
        self.channel_names = self._extract_channel_names(recording)

        self.meta = RecordingMetadata(
            None,
            n_channels=recording.get_num_channels(),
            f_s=recording.get_sampling_frequency(),
            dt_end=None,  # In-memory recordings don't have timestamps until persisted
            channel_names=self.channel_names,
        )

        # Compute file duration from recording
        duration_s = recording.get_total_duration()
        self.file_durations = [duration_s]
        self.cumulative_file_durations = [duration_s]

    @staticmethod
    def _resolve_func_path(path_str: str) -> Callable:
        """Import and return a callable from a ``"file.py:function"`` path.

        Parameters
        ----------
        path_str : str
            ``"path/to/readers.py:read_bin_csv"`` format string.

        Returns
        -------
        Callable
            The resolved callable.

        Raises
        ------
        ImportError
            If the file cannot be loaded or no ``:`` separator is found.
        AttributeError
            If the attribute does not exist in the module.
        """
        if ":" not in path_str:
            raise ImportError(
                f"Cannot resolve '{path_str}': expected "
                "'path/to/file.py:func_name' format"
            )

        import importlib.util

        file_path, _, attr_name = path_str.rpartition(":")
        spec = importlib.util.spec_from_file_location("_user_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from file: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, attr_name)

    def detect_and_load_data(
        self,
        mode: Literal["si", "mne", None] = "si",
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        extract_func: Union[
            Callable[..., "si.BaseRecording"], Callable[..., mne.io.Raw], str
        ] = None,
        **kwargs,
    ):
        """Load in recording based on mode.

        Parameters
        ----------
        mode : {"si", "mne", None}
            Backend to use for loading recordings.
        cache_policy : {"auto", "always", "force_regenerate"}
            Caching strategy for loaded recordings.
        multiprocess_mode : {"dask", "serial"}
            Parallelism strategy.
        extract_func : callable or str, optional
            Function (or reference to one) used to load each discovered file
            into a recording object. When a string, resolved in this order:

            1. **Short name** — looked up in ``spikeinterface.extractors`` /
               ``spikeinterface`` (for ``mode="si"``) or ``mne.io``
               (for ``mode="mne"``).  Example: ``"read_intan"``.
            2. **File path** (contains ``:``) — loads a function directly from
               a Python file.  The ``.py`` extension is required.
               Example: ``"tests/integration/readers.py:read_bin_csv_pair"`` or
               ``"/absolute/path/to/readers.py:my_func"``.
        **kwargs
            Forwarded to the backend loading method.
        """
        if mode == "si":
            if si is None:
                raise ImportError("SpikeInterface is required for mode='si'")

            if isinstance(extract_func, str):
                func_name = extract_func
                # Try SpikeInterface namespaces first
                extract_func = getattr(se, func_name, getattr(si, func_name, None))
                # Resolve file path: "path/to/readers.py:read_custom"
                if extract_func is None and ":" in func_name:
                    extract_func = self._resolve_func_path(func_name)
                if extract_func is None:
                    raise ValueError(
                        f"Could not resolve extractor function: {func_name}. "
                        "Provide a SpikeInterface extractor name "
                        "or a file path (e.g. 'path/to/readers.py:func_name')."
                    )
            elif extract_func is None:
                extract_func = si.load

            self.convert_file_with_si_to_recording(
                extract_func=extract_func,
                cache_policy=cache_policy,
                multiprocess_mode=multiprocess_mode,
                **kwargs,
            )
        elif mode == "mne":
            if isinstance(extract_func, str):
                func_name = extract_func
                extract_func = getattr(mne.io, func_name, None)
                # Resolve file path: "path/to/readers.py:read_custom"
                if extract_func is None and ":" in func_name:
                    extract_func = self._resolve_func_path(func_name)
                if extract_func is None:
                    raise ValueError(
                        f"Could not resolve extractor function: {func_name}. "
                        "Provide an MNE extractor name "
                        "or a file path (e.g. 'path/to/readers.py:func_name')."
                    )

            self.convert_file_with_mne_to_recording(
                extract_func=extract_func,
                cache_policy=cache_policy,
                n_jobs=self.n_jobs,
                **kwargs,
            )
        elif mode is None:
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @staticmethod
    def _create_empty_si_recording() -> "si.BaseRecording":
        """Return a 0-sample SpikeInterface recording as a placeholder.

        Used when a file group fails to load (e.g. corrupt or empty metadata).
        The 0-sample recording is detected by the unified check in
        ``convert_file_with_si_to_recording`` and skipped by
        ``_iter_valid_recordings`` in downstream processing.
        """
        return si.NumpyRecording(
            traces_list=[np.zeros((0, 1), dtype=np.float32)],
            sampling_frequency=constants.GLOBAL_SAMPLING_RATE,
            channel_ids=["placeholder"],
        )

    def convert_file_with_si_to_recording(
        self,
        extract_func: Callable[..., "si.BaseRecording"],
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        **kwargs,
    ):
        from .discovery import DiscoveredFile

        if si is None:
            raise ImportError("SpikeInterface is required")

        # Determine number of files being processed
        if isinstance(self.item, list):
            n_processed_files = len(self.item)
        else:
            n_processed_files = 1

        self._validate_timestamps_for_mode("si", n_processed_files)

        # Handle different item types
        if isinstance(self.item, DiscoveredFile):
            # DiscoveredFile: handle both single and multi-file cases
            if self.item.is_multi_file:
                # Multi-file group: pass as-is to extract_func (user's custom reader).
                # Wrap in try/except so a corrupt/empty file group produces a 0-sample
                # placeholder that downstream code skips, rather than crashing the
                # entire animal's pipeline run.
                try:
                    rec: "si.BaseRecording" = extract_func(self.item, **kwargs)
                except (ValueError, IndexError, OSError, KeyError) as e:
                    logging.warning(
                        f"extract_func failed for {self.display_name} "
                        f"({self.item.paths}): {e}. "
                        f"Creating 0-sample placeholder; this recording will be skipped."
                    )
                    rec = self._create_empty_si_recording()
            else:
                # Single file
                rec: "si.BaseRecording" = extract_func(self.item.path, **kwargs)
        elif isinstance(self.item, list):
            # List of files: concatenate individually using multiprocess_mode
            if multiprocess_mode == "dask":
                if dask is None:
                    raise ImportError("dask is required for multiprocess_mode='dask'")
                logging.info(f"Loading {len(self.item)} files in parallel with dask")
                tasks = [dask.delayed(extract_func)(x, **kwargs) for x in self.item]
                recs = list(dask.compute(*tasks))
            else:
                logging.info(f"Loading {len(self.item)} files serially")
                recs = [extract_func(x, **kwargs) for x in self.item]
            # Filter out empty recordings before concatenation to avoid
            # spikeinterface crashes on empty segments (e.g. 0-byte files)
            valid_recs = []
            for r in recs:
                try:
                    if r.get_total_samples() == 0:
                        logging.warning("Skipping empty recording (0 samples) during concatenation")
                        continue
                except (TypeError, AttributeError):
                    pass  # Non-SI recording or mock — keep it
                valid_recs.append(r)
            if not valid_recs:
                raise ValueError("All recordings in this session have 0 samples")
            rec = si.concatenate_recordings(valid_recs)
        else:
            # Single file/path
            rec: "si.BaseRecording" = extract_func(self.item, **kwargs)

        # Unified 0-sample check for DiscoveredFile and single-file branches.
        # The list branch already filters individual 0-sample files above;
        # this catches the remaining cases so _iter_valid_recordings() can
        # skip this LRO downstream instead of crashing in the resampler.
        try:
            if rec.get_total_samples() == 0:
                logging.warning(
                    f"Loaded 0-sample recording ({self.display_name}). "
                    f"This recording will be skipped by downstream processing."
                )
        except (TypeError, AttributeError):
            pass  # Non-SI recording or mock — keep it

        self._n_processed_files = n_processed_files
        self.LongRecording = self._apply_resampling(rec)
        self._validate_units_uV(self.LongRecording)

        dt_end = None
        channel_names = self._extract_channel_names(self.LongRecording)

        self.meta = RecordingMetadata(
            None,
            n_channels=self.LongRecording.get_num_channels(),
            f_s=self.LongRecording.get_sampling_frequency(),
            dt_end=dt_end,
            channel_names=channel_names,
            V_units="µV",
            mult_to_uV=1.0,
        )
        self.channel_names = self.meta.channel_names

        if not hasattr(self, "file_durations") or not self.file_durations:
            if hasattr(self, "_n_processed_files") and self._n_processed_files > 1:
                avg_duration = (
                    self.LongRecording.get_duration() / self._n_processed_files
                )
                self.file_durations = [avg_duration] * self._n_processed_files
            else:
                self.file_durations = [self.LongRecording.get_duration()]
            self.file_end_datetimes = []

        self.finalize_file_timestamps()
        logging.debug(f"LongRecording created via SI: {self}")

    def _load_and_process_mne_data(
        self,
        extract_func,
        input_type,
        datafolder,
        datafile,
        datafiles,
        n_jobs,
        metadata_to_update=None,
        **kwargs,
    ) -> mne.io.Raw:
        """Helper method to load and process MNE data from various input types."""
        # Load data based on input type
        if input_type == "folder":
            raw: mne.io.Raw = extract_func(datafolder, **kwargs)
        elif input_type == "file":
            raw: mne.io.Raw = extract_func(datafile, **kwargs)
        elif input_type == "files":
            logging.info(f"Running extract_func on {len(datafiles)} files")
            raws: list[mne.io.Raw] = [extract_func(x, **kwargs) for x in datafiles]
            logging.info(f"Concatenating {len(raws)} raws")
            raw: mne.io.Raw = mne.concatenate_raws(raws)
            del raws
        else:
            raise ValueError(f"Invalid input_type: {input_type}")

        logging.info(f"raw.info: {raw.info}")

        # Use user-specified n_jobs for MNE resampling, or default to 1
        effective_n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        logging.info(
            f"Using n_jobs={effective_n_jobs} for MNE resampling (method param: {n_jobs}, instance: {self.n_jobs})"
        )

        # Ensure data is preloaded for parallel processing
        if not raw.preload:
            logging.info("Preloading data")
            raw.load_data()

        # Use optimal resampling method with power-of-2 padding for speed
        original_sfreq = raw.info["sfreq"]
        if original_sfreq != constants.GLOBAL_SAMPLING_RATE:
            logging.info(
                f"Resampling from {original_sfreq} to {constants.GLOBAL_SAMPLING_RATE}"
            )
            raw = raw.resample(
                constants.GLOBAL_SAMPLING_RATE,
                n_jobs=effective_n_jobs,
                npad="auto",
                method="fft",
            )

            # Update metadata to reflect the new sampling rate
            if metadata_to_update is not None:
                metadata_to_update.update_sampling_rate(constants.GLOBAL_SAMPLING_RATE)
        else:
            logging.info(
                f"Sampling frequency already matches {constants.GLOBAL_SAMPLING_RATE} Hz, no resampling needed"
            )

        return raw

    def _load_mne_data_no_resample(self, extract_func, **kwargs) -> "mne.io.Raw":
        """Load MNE data without resampling for unified resampling pipeline.

        This method loads and concatenates MNE data but skips resampling,
        allowing the unified resampling to be applied after intermediate file creation.
        """
        if isinstance(self.item, list):
            logging.info(f"Running extract_func on {len(self.item)} files")
            raws = [extract_func(x, **kwargs) for x in self.item]
            logging.info(f"Concatenating {len(raws)} raws")
            raw = mne.concatenate_raws(raws)
            del raws
        else:
            raw = extract_func(self.item, **kwargs)

        logging.info(f"raw.info: {raw.info}")

        # Ensure data is preloaded
        if not raw.preload:
            logging.info("Preloading data")
            raw.load_data()

        # NOTE: No resampling applied here - will be handled by unified resampling after loading from cache
        logging.info(
            f"Data loaded at original sampling rate ({raw.info['sfreq']} Hz) - resampling will be applied later"
        )

        return raw

    def _get_or_create_intermediate_file(
        self,
        fname,
        source_paths,
        cache_policy,
        intermediate,
        extract_func,
        n_jobs,
        **kwargs,
    ):
        """Get cached intermediate file or create it if needed.

        Returns:
            tuple: (recording, raw_object, metadata) where:
                - recording: SpikeInterface recording object
                - raw_object: MNE Raw object (None if using cache)
                - metadata: RecordingMetadata object
        """
        # Define metadata sidecar file path
        meta_fname = fname.with_suffix(fname.suffix + ".meta.json")

        # Check cache policy and validate cache files
        if cache_policy == "force_regenerate":
            use_cache = False
            logging.info(get_cache_status_message(fname, False))
            logging.info("Cache policy 'force_regenerate': ignoring any existing cache")
        else:
            # Check if both data and metadata cache files exist and are valid
            data_cache_valid = should_use_cache_unified(
                fname, source_paths, cache_policy
            )
            meta_cache_valid = meta_fname.exists() if data_cache_valid else False

            # Handle cache validation based on policy
            if not data_cache_valid or not meta_cache_valid:
                if cache_policy == "always":
                    # 'always' policy: raise error if cache missing/invalid
                    missing_files = []
                    if not data_cache_valid:
                        missing_files.append(f"intermediate file ({fname})")
                    if not meta_cache_valid:
                        missing_files.append(f"metadata sidecar ({meta_fname})")
                    raise FileNotFoundError(
                        f"Cache policy 'always' requires existing cache files, but missing: {', '.join(missing_files)}"
                    )
                elif cache_policy == "auto":
                    # 'auto' policy: log and regenerate if cache missing/invalid
                    if not data_cache_valid:
                        logging.info(
                            f"Intermediate file {fname} missing or outdated, regenerating"
                        )
                    if not meta_cache_valid:
                        logging.info(
                            f"Metadata sidecar {meta_fname} missing, regenerating"
                        )
                    use_cache = False
                else:
                    use_cache = False
            else:
                use_cache = True

            if use_cache:
                logging.info(get_cache_status_message(fname, True))
                logging.info(f"Loading cached metadata from {meta_fname}")

                # Load metadata from sidecar file
                try:
                    metadata = RecordingMetadata.from_json(meta_fname)
                    logging.info(
                        f"Loaded cached metadata: {metadata.n_channels} channels, {metadata.f_s} Hz"
                    )
                except CACHE_METADATA_ERRORS as e:
                    if cache_policy == "always":
                        logging.error(
                            f"Cache policy 'always' requires valid metadata, but failed to load {meta_fname}: {e}"
                        )
                        raise
                    # Unconditional, not `elif policy == "auto"`: with no else branch, any other
                    # policy left use_cache True and metadata unbound, for an UnboundLocalError below.
                    logging.warning(
                        f"Cached metadata {meta_fname} is unreadable ({type(e).__name__}: {e}); "
                        f"regenerating the intermediate files"
                    )
                    use_cache = False

        if use_cache:
            # The cache is validated only by existence + mtime, not integrity, so a truncated file
            # (from a write interrupted by a killed job) can reach here. Self-heal by deleting it and
            # regenerating -- but only for a genuinely unreadable FILE. Anything else raised below is a
            # bug in our code, and laundering it into "the cache was corrupt" would delete good data
            # and hide the fault.
            #
            # Built outside the try for that reason: a bad unit scale or a metadata schema change is
            # not a corrupt cache.
            params = None
            if intermediate == "bin":
                params = {
                    "sampling_frequency": metadata.f_s,
                    "num_channels": metadata.n_channels,
                    "dtype": "float64",  # We standardize on float64 for cached binary files
                    "gain_to_uV": _gain_to_uV(metadata),
                    "offset_to_uV": 0,
                    "time_axis": 0,
                    "is_filtered": False,
                    "channel_ids": metadata.channel_names,
                }
            elif intermediate != "edf":
                raise ValueError(f"Invalid intermediate: {intermediate}")

            try:
                if intermediate == "edf":
                    logging.info("Reading cached edf file")
                    rec = se.read_edf(fname)
                    return rec, None, metadata  # No raw object when using cache

                # read_binary is memmap-backed and returns a SHORTER recording rather than raising on
                # a truncated file, so that corruption has to be measured, not caught.
                if not _bin_is_intact(fname, metadata):
                    raise OSError(
                        f"cached binary {fname} is empty or truncated mid-frame "
                        f"({fname.stat().st_size} bytes, {metadata.n_channels} channels)"
                    )
                logging.info(f"Reading from cached binary file {fname}")
                rec = se.read_binary(fname, **params)
                return rec, None, metadata  # No raw object when using cache
            except CACHE_DATA_ERRORS as e:
                if cache_policy == "always":
                    logging.error(
                        f"Cache policy 'always' requires a readable cached file, "
                        f"but {fname} could not be read: {e}"
                    )
                    raise
                logging.warning(
                    f"Cached intermediate file {fname} could not be read ({e}); "
                    f"deleting and regenerating"
                )
                safe_unlink(fname)
                safe_unlink(meta_fname)
                use_cache = False

        if not use_cache:
            # Generate new intermediate files
            logging.info(get_cache_status_message(fname, False))

            # Create metadata object from raw info BEFORE resampling
            # We need to load one file to get the original metadata
            if isinstance(self.item, list):
                sample_raw = extract_func(self.item[0], **kwargs)
            else:
                sample_raw = extract_func(self.item, **kwargs)

            # Create metadata from the original raw object (before resampling)
            original_info = sample_raw.info

            # Extract unit information from MNE Raw object
            unit_str, mult_to_uv = extract_mne_unit_info(original_info)

            metadata = RecordingMetadata(
                metadata_path=None,
                n_channels=original_info["nchan"],
                f_s=original_info["sfreq"],  # Original sampling rate
                dt_end=None,  # Will be set later by finalize_file_timestamps
                channel_names=original_info["ch_names"],
                V_units=unit_str,
                mult_to_uV=mult_to_uv,
            )
            logging.info(
                f"Created metadata from raw: {metadata.n_channels} channels, {metadata.f_s} Hz"
            )
            if unit_str and mult_to_uv:
                logging.info(
                    f"Extracted unit information: {unit_str} (mult_to_uV = {mult_to_uv})"
                )
            else:
                logging.warning(
                    "No unit information could be extracted from MNE Raw object"
                )

            # Load data without resampling (resampling will be applied after intermediate file loading)
            raw = self._load_mne_data_no_resample(extract_func, **kwargs)

            # Check if channel names in MNE Raw object are in Intan format and convert if necessary
            if any("intan" in ch_name.lower() for ch_name in raw.info["ch_names"]):
                logging.info("Converting Intan channel names to MNE format")
                rename_mne_channels(
                    raw
                )  # REVIEW check that this function is robust

            # Create the intermediate file
            if intermediate == "edf":
                logging.info(f"Exporting raw to {fname}")
                # Export to a temp sibling and atomically rename, so an interrupted
                # write never leaves a partial EDF at the canonical cache path.
                with atomic_output_path(fname) as tmp:
                    try:
                        mne.export.export_raw(tmp, raw=raw, fmt="edf", overwrite=True)
                    except ValueError as e:
                        # REVIEW JD to me this appears hardcoded -- will check with EDF files as well
                        if "exceeds maximum field length" in str(e):
                            logging.warning(
                                f"EDF export failed due to signal range: {e}. Retrying with robust physical range."
                            )
                            # Calculate robust range (0.01 - 99.99 percentile) to exclude artifacts
                            data = raw.get_data()
                            # Use data percentiles to define physical range, excluding extreme outliers
                            p_min, p_max = np.percentile(data, [0.01, 99.99])

                            # Helper to ensure float fits in 8 chars (EDF limit)
                            def to_valid_edf_float(val):
                                # Try formatting with decreasing precision
                                for fmt in [".6g", ".5g", ".4g", ".3g", ".2g"]:
                                    s = f"{val:{fmt}}"
                                    if len(s) <= 8:
                                        return float(s)
                                # Fallback
                                return float(f"{val:.2e}")

                            safe_min = to_valid_edf_float(p_min)
                            safe_max = to_valid_edf_float(p_max)

                            logging.info(
                                f"Using robust physical range: ({safe_min}, {safe_max})"
                            )
                            mne.export.export_raw(
                                tmp,
                                raw=raw,
                                fmt="edf",
                                overwrite=True,
                                physical_range=(safe_min, safe_max),
                            )
                        else:
                            raise

                logging.info("Reading edf file")
                rec = se.read_edf(fname)

            elif intermediate == "bin":
                # Get raw info for SpikeInterface parameters
                raw_info = raw.info
                params = {
                    "sampling_frequency": raw_info["sfreq"],
                    "num_channels": raw_info["nchan"],
                    # raw.get_data() writes MNE's native units (volts) to disk, so the gain must
                    # convert them, not be 1.
                    "gain_to_uV": _gain_to_uV(metadata),
                    "offset_to_uV": 0,
                    "time_axis": 0,
                    "is_filtered": False,
                    "channel_ids": raw_info["ch_names"],
                }

                logging.info(f"Exporting raw to {fname}")
                data: np.ndarray = raw.get_data()  # (n channels, n samples)
                data = data.T  # (n samples, n channels)
                params["dtype"] = data.dtype
                logging.info(f"Writing to {fname}")
                # Write to a temp sibling and atomically rename, so an interrupted
                # write never leaves a partial binary file at the cache path.
                with atomic_output_path(fname) as tmp:
                    data.tofile(tmp)

                logging.info(f"Reading from {fname}")
                rec = se.read_binary(fname, **params)

            else:
                raise ValueError(f"Invalid intermediate: {intermediate}")

            # Save metadata sidecar file
            logging.info(f"Saving metadata to {meta_fname}")
            metadata.to_json(meta_fname)

            return rec, raw, metadata

    def convert_file_with_mne_to_recording(
        self,
        extract_func: Callable[..., mne.io.Raw],
        intermediate: Literal["edf", "bin"] = "edf",
        intermediate_name=None,
        intermediate_dir=None,
        cache_policy: Literal["auto", "always", "force_regenerate"] = "auto",
        multiprocess_mode: Literal["dask", "serial"] = "serial",
        n_jobs: int = None,
        **kwargs,
    ):
        if se is None:
            raise ImportError(
                "SpikeInterface is required for convert_file_with_mne_to_recording"
            )

        # Determine number of files and source paths
        if isinstance(self.item, list):
            self._validate_timestamps_for_mode("mne", len(self.item))
            source_paths = self.item
            n_processed_files = len(self.item)
        else:
            self._validate_timestamps_for_mode("mne", 1)
            source_paths = [self.item]
            n_processed_files = 1

        self._n_processed_files = n_processed_files

        # Generate intermediate filename
        base_name = Path(source_paths[0]).stem if source_paths else "mne_recording"
        intermediate_name = (
            f"{base_name}_mne-to-rec"
            if intermediate_name is None
            else intermediate_name
        )

        # Determine directory for intermediate files
        # Priority: intermediate_dir parameter > temp directory
        use_temp_dir = intermediate_dir is None
        if intermediate_dir is not None:
            # User specified directory - always keep files for reuse
            base_dir = Path(intermediate_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use temp directory for intermediate files to avoid cluttering source directories
            import tempfile
            try:
                base_dir = get_temp_directory()
            except KeyError:
                # Fall back to system temp directory if TMPDIR not set
                base_dir = Path(tempfile.gettempdir()) / "neurodent_mne_cache"
                base_dir.mkdir(parents=True, exist_ok=True)

        fname = base_dir / f"{intermediate_name}.{intermediate}"
        meta_fname = fname.with_suffix(fname.suffix + ".meta.json")

        try:
            rec, _, metadata = self._get_or_create_intermediate_file(
                fname=fname,
                source_paths=source_paths,
                cache_policy=cache_policy,
                intermediate=intermediate,
                extract_func=extract_func,
                n_jobs=n_jobs,
                **kwargs,
            )

            self.meta = metadata
            self.channel_names = self.meta.channel_names
            self.LongRecording = self._apply_resampling(rec)
            self._validate_units_uV(self.LongRecording)
        finally:
            # Clean up intermediate files if using temp directory with force_regenerate policy
            # This integrates cleanup with cache policy: files are only kept when caching is intended
            if use_temp_dir and cache_policy == "force_regenerate":
                # Remove intermediate files since they won't be reused
                try:
                    fname.unlink()
                    logging.debug(f"Cleaned up intermediate file: {fname}")
                except FileNotFoundError:
                    pass
                except (OSError, PermissionError) as e:
                    logging.warning(f"Failed to clean up intermediate file {fname}: {e}")

                try:
                    meta_fname.unlink()
                    logging.debug(f"Cleaned up metadata file: {meta_fname}")
                except FileNotFoundError:
                    pass
                except (OSError, PermissionError) as e:
                    logging.warning(f"Failed to clean up metadata file {meta_fname}: {e}")

        if not hasattr(self, "file_durations") or not self.file_durations:
            if hasattr(self, "_n_processed_files") and self._n_processed_files > 1:
                avg_duration = (
                    self.LongRecording.get_duration() / self._n_processed_files
                )
                self.file_durations = [avg_duration] * self._n_processed_files
            else:
                self.file_durations = [self.LongRecording.get_duration()]
            self.file_end_datetimes = []

        self.finalize_file_timestamps()

    def _apply_resampling(self, recording: "si.BaseRecording") -> "si.BaseRecording":
        """Apply unified resampling and voltage scaling using SpikeInterface preprocessing.

        This method centralizes all resampling logic across the different data loading pipelines
        (binary, MNE, SI) to use the fast SpikeInterface resampling implementation consistently.

        It also enforces the global data type (constants.GLOBAL_DTYPE) for consistency.

        If the recording has scaleable traces (gain_to_uV and offset_to_uV properties),
        voltage scaling is applied first via ``spre.scale_to_uV``. This bakes the correct
        ADC-to-µV conversion into the data and resets gain/offset to 1.0/0.0, which prevents
        offset bugs from subsequent unsigned-to-signed conversion.

        Args:
            recording (si.BaseRecording): The recording to resample

        Returns:
            si.BaseRecording: The resampled recording with data in µV (if scaleable)

        Raises:
            ImportError: If SpikeInterface preprocessing is not available
        """
        # Guard clause: return early if recording is None or invalid
        if recording is None:
            return recording

        if spre is None:
            raise ImportError("SpikeInterface preprocessing is required for resampling")

        # 0. Apply voltage scaling FIRST for integer-typed recordings with gain/offset.
        # SpikeInterface extractors (e.g., read_intan) store gain_to_uV and offset_to_uV
        # which encode how to convert raw ADC values to microvolts. Applying scale_to_uV
        # bakes this conversion into the data and resets gain/offset to 1.0/0.0.
        # This MUST happen before unsigned_to_signed, because unsigned_to_signed shifts
        # the raw data by 2^(bits-1) without updating offset_to_uV, which would cause
        # the offset to be applied twice when get_traces(return_scaled=True) is called.
        # Only apply for integer dtypes — float recordings are assumed to already be in
        # physical units, matching SpikeInterface's own convention (baserecording.py:356).
        dtype = recording.get_dtype() if hasattr(recording, "get_dtype") else None
        is_integer = False
        if dtype is not None and isinstance(dtype, (str, type, np.dtype)):
            try:
                is_integer = np.dtype(dtype).kind in ("i", "u")
            except TypeError:
                pass

        if (
            is_integer
            and hasattr(recording, "has_scaleable_traces")
            and recording.has_scaleable_traces()
        ):
            logging.info("Applying scale_to_uV to convert raw ADC data to microvolts")
            recording = spre.scale_to_uV(recording)

        # 1. Enforce signed integer if unsigned (existing logic preserved)
        dtype = recording.get_dtype() if hasattr(recording, "get_dtype") else None
        # Handle numpy types, strings. Avoid Mock objects
        is_unsigned = False
        if dtype is not None and isinstance(dtype, (str, type, np.dtype)):
            try:
                if np.dtype(dtype).kind == "u":
                    is_unsigned = True
            except TypeError:
                pass

        if is_unsigned:
            logging.info(
                f"Data type is unsigned ({dtype}) and SpikeInterface can't process. Converting it to signed"
            )
            recording = spre.unsigned_to_signed(recording)

        # 2. Enforce GLOBAL_DTYPE (New logic)
        current_dtype = (
            recording.get_dtype() if hasattr(recording, "get_dtype") else None
        )
        if current_dtype is not None and current_dtype != constants.GLOBAL_DTYPE:
            logging.info(
                f"Converting recording dtype from {current_dtype} to {constants.GLOBAL_DTYPE}"
            )
            recording = spre.astype(recording, dtype=constants.GLOBAL_DTYPE)

        # 3. Apply Resampling if needed
        current_rate = (
            recording.get_sampling_frequency()
            if hasattr(recording, "get_sampling_frequency")
            else None
        )
        if current_rate is None and hasattr(recording, "info"):
            current_rate = recording.info.get("sfreq", None)

        target_rate = constants.GLOBAL_SAMPLING_RATE

        if current_rate == target_rate or current_rate is None:
            logging.info(
                f"Recording already at target sampling rate ({target_rate} Hz) or unable to determine, no resampling needed"
            )
            return recording

        logging.info(
            f"Resampling recording from {current_rate} Hz to {target_rate} Hz using SpikeInterface"
        )

        # Use SpikeInterface resampling with margin to reduce edge effects
        resampled_recording = spre.resample(
            recording=recording,
            resample_rate=target_rate,
        )

        # Update metadata to reflect new sampling rate
        if hasattr(self, "meta") and self.meta is not None:
            self.meta.update_sampling_rate(target_rate)

        logging.info(f"Successfully resampled recording to {target_rate} Hz")
        return resampled_recording
