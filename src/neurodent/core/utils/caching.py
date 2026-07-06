"""Zarr fragment caching/streaming and cache-decision policy."""

import logging
import os

from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
import zarr

from .paths import get_temp_directory


def cache_fragments_to_zarr(
    np_fragments: np.ndarray,
    n_fragments: int,
    tmpdir: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> tuple[str, "zarr.Array"]:
    """
    Cache numpy fragments array to zarr format for efficient memory management.

    This function converts a numpy array of recording fragments to a zarr array stored
    in a temporary location. This allows better memory management and garbage collection
    by avoiding keeping large numpy arrays in memory for extended periods.

    Args:
        np_fragments (np.ndarray): Numpy array of shape (n_fragments, n_samples, n_channels)
            containing the recording fragments to cache.
        n_fragments (int): Number of fragments to cache (allows for subset caching).
        tmpdir (str, optional): Directory path for temporary zarr storage. If None,
            uses get_temp_directory(). Defaults to None.
        chunk_size (int, optional): Number of fragments per zarr chunk along the first
            axis. Controls the read/write granularity when accessing the zarr array.
            Smaller values reduce memory overhead per chunk; larger values improve
            sequential throughput. When None, defaults to ``min(100, n_fragments)``.

    Returns:
        tuple[str, zarr.Array]: A tuple containing:
            - str: Path to the temporary zarr file
            - zarr.Array: The zarr array object for accessing cached data

    Raises:
        ImportError: If zarr is not available
        ValueError: If ``chunk_size`` is not None and is less than 1
    """
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(
            f"chunk_size must be >= 1, got {chunk_size!r}. "
            "Pass None to use the default chunk size."
        )

    try:
        import zarr
    except ImportError:
        raise ImportError("zarr package is required for fragment caching")

    if tmpdir is None:
        tmpdir = get_temp_directory()

    # Generate unique temporary path
    tmppath = os.path.join(tmpdir, f"temp_{os.urandom(24).hex()}.zarr")

    logging.debug(f"Caching numpy array with zarr in {tmppath}")

    # Create Zarr array with optimal settings for fragment-wise access
    if chunk_size is None:
        chunk_size = min(100, n_fragments)  # Cap at 100 fragments per chunk
    else:
        if not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer or None")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1 when provided")
        chunk_size = min(chunk_size, n_fragments)
    zarr_array = zarr.open(
        tmppath,
        mode="w",
        shape=np_fragments.shape,
        chunks=(
            chunk_size,
            -1,  # No chunking along timestamp dimension
            -1,  # No chunking along channel dimension
        ),
        dtype=np_fragments.dtype,
        compressor=zarr.Blosc(cname="lz4", clevel=3, shuffle=zarr.Blosc.SHUFFLE),  # Fast compression
    )
    zarr_array[:n_fragments] = np_fragments[:n_fragments]

    # Log debug properties of the zarr array
    total_memory_bytes = zarr_array.nbytes
    total_memory_mb = total_memory_bytes / (1024 * 1024)
    total_memory_gb = total_memory_mb / 1024

    logging.debug(f"  - Total memory footprint: {total_memory_mb:.2f} MB, {total_memory_gb:.3f} GB")
    logging.debug(f"  - Zarr array shape: {zarr_array.shape}")
    logging.debug(f"  - Zarr array chunks: {zarr_array.chunks}")

    return tmppath, zarr_array


def stream_fragments_to_zarr(
    get_fragment_fn: Callable[[int], np.ndarray],
    n_fragments: int,
    fragment_shape: tuple,
    fragment_dtype: np.dtype,
    chunk_size: int,
    tmpdir: Optional[str] = None,
) -> str:
    """Stream recording fragments to a zarr store in memory-bounded batches.

    Unlike :func:`cache_fragments_to_zarr`, this function never holds more than
    ``chunk_size`` fragments in RAM at once.  It calls ``get_fragment_fn`` one
    batch at a time, writes each batch to the zarr store, and immediately frees
    the batch buffer — so peak RAM is proportional to ``chunk_size`` rather than
    ``n_fragments``.

    Args:
        get_fragment_fn (Callable[[int], np.ndarray]): A callable that accepts a
            fragment index (0-based) and returns the corresponding fragment as a
            NumPy array of shape ``fragment_shape``.
        n_fragments (int): Total number of fragments to stream.
        fragment_shape (tuple): Shape of a single fragment (e.g. ``(n_samples,
            n_channels)``).
        fragment_dtype (np.dtype): Data-type of the fragment arrays.
        chunk_size (int): Number of fragments to buffer per batch.  Must be >= 1.
            Larger values improve sequential write throughput; smaller values
            reduce peak RAM.
        tmpdir (str, optional): Directory for the temporary zarr file.  If
            ``None``, uses :func:`get_temp_directory`.

    Returns:
        str: Path to the temporary zarr file on disk.

    Raises:
        ValueError: If ``chunk_size`` < 1.
        ImportError: If zarr is not available.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    if tmpdir is None:
        tmpdir = get_temp_directory()

    tmppath = os.path.join(tmpdir, f"temp_{os.urandom(24).hex()}.zarr")
    batch = min(chunk_size, n_fragments)

    logging.debug(
        f"Streaming {n_fragments} fragments to zarr in batches of {batch} "
        f"(path: {tmppath})"
    )

    zarr_array = zarr.open(
        tmppath,
        mode="w",
        shape=(n_fragments,) + fragment_shape,
        chunks=(batch, -1, -1),
        dtype=fragment_dtype,
        compressor=zarr.Blosc(cname="lz4", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
    )

    for batch_start in range(0, n_fragments, batch):
        batch_end = min(batch_start + batch, n_fragments)
        batch_len = batch_end - batch_start
        np_batch = np.empty((batch_len,) + fragment_shape, dtype=fragment_dtype)
        for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
            np_batch[local_idx] = get_fragment_fn(global_idx)
        zarr_array[batch_start:batch_end] = np_batch
        del np_batch

    logging.debug(f"  - Zarr array shape: {zarr_array.shape}")
    logging.debug(f"  - Zarr array chunks: {zarr_array.chunks}")
    del zarr_array

    return tmppath


def stream_recording_to_zarr(
    recording,
    n_fragments: int,
    n_samples_per_frag: int,
    n_frag_per_chunk: int,
    tmpdir: Optional[str] = None,
) -> str:
    """Stream a SpikeInterface recording to a zarr store in memory-bounded batches.

    Reads chunk-sized slices from ``recording.get_traces()``, reshapes each
    chunk to ``(n_frags_in_chunk, n_samples_per_frag, n_channels)``, and
    writes it to a zarr store.  Peak RAM is proportional to ``n_frag_per_chunk``
    rather than ``n_fragments``.

    Args:
        recording: A SpikeInterface ``BaseRecording`` object (may be a lazy
            wrapper such as a ``NotchFilterRecording``).
        n_fragments (int): Total number of fragments to stream.
        n_samples_per_frag (int): Number of samples in each fragment.
        n_frag_per_chunk (int): Number of fragments to buffer per batch.
            Must be >= 1.  Larger values improve sequential write throughput;
            smaller values reduce peak RAM.
        tmpdir (str, optional): Directory for the temporary zarr file.  If
            ``None``, uses :func:`get_temp_directory`.

    Returns:
        str: Path to the temporary zarr file on disk.

    Raises:
        ValueError: If ``n_frag_per_chunk`` < 1.
        ImportError: If zarr is not available.
    """
    if n_frag_per_chunk < 1:
        raise ValueError(f"n_frag_per_chunk must be >= 1, got {n_frag_per_chunk}")

    try:
        import zarr
    except ImportError:
        raise ImportError("zarr package is required for fragment caching")

    if tmpdir is None:
        tmpdir = get_temp_directory()

    n_channels = recording.get_num_channels()
    fragment_dtype = recording.get_dtype()
    tmppath = os.path.join(tmpdir, f"temp_{os.urandom(24).hex()}.zarr")
    # Cap the batch size so we never request more fragments than exist
    batch_size = min(n_frag_per_chunk, n_fragments)

    logging.debug(
        f"Streaming recording ({n_fragments} fragments × {n_samples_per_frag} samples) "
        f"to zarr in batches of {batch_size} (path: {tmppath})"
    )

    zarr_array = zarr.open(
        tmppath,
        mode="w",
        shape=(n_fragments, n_samples_per_frag, n_channels),
        chunks=(batch_size, -1, -1),
        dtype=fragment_dtype,
        compressor=zarr.Blosc(cname="lz4", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
    )

    for batch_start in range(0, n_fragments, batch_size):
        batch_end = min(batch_start + batch_size, n_fragments)
        start_sample = batch_start * n_samples_per_frag
        end_sample = batch_end * n_samples_per_frag
        chunk_traces = recording.get_traces(
            start_frame=start_sample, end_frame=end_sample, return_scaled=True
        )
        chunk_fragments = chunk_traces.reshape(
            batch_end - batch_start, n_samples_per_frag, n_channels
        )
        zarr_array[batch_start:batch_end] = chunk_fragments
        # Explicitly free each batch to keep peak RAM bounded to batch_size fragments
        del chunk_traces, chunk_fragments

    logging.debug(f"  - Zarr array shape: {zarr_array.shape}")
    logging.debug(f"  - Zarr array chunks: {zarr_array.chunks}")
    del zarr_array

    return tmppath


def should_use_cached_file(
    cache_path: Union[str, Path],
    source_paths: list[Union[str, Path]],
    use_cached: Literal["auto", "always", "never", "error"] = "auto",
) -> bool:
    """
    Determine whether to use a cached intermediate file based on caching policy and file timestamps.

    Args:
        cache_path: Path to the cached intermediate file
        source_paths: List of source file paths that the cache depends on
        use_cached: Caching policy
            - "auto": Use cached if exists and newer than all sources (default)
            - "always": Always use cached if it exists
            - "never": Never use cached (always regenerate)
            - "error": Raise error if cached doesn't exist

    Returns:
        bool: True if cached file should be used, False if it should be regenerated

    Raises:
        FileNotFoundError: When use_cached="error" and cache doesn't exist
        ValueError: For invalid use_cached values
    """
    cache_path = Path(cache_path)
    source_paths = [Path(p) for p in source_paths]

    if use_cached == "never":
        return False
    elif use_cached == "error":
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file required but not found: {cache_path}")
        return True
    elif use_cached == "always":
        return cache_path.exists()
    elif use_cached == "auto":
        if not cache_path.exists():
            return False

        # Check if cache is newer than all source files
        cache_mtime = cache_path.stat().st_mtime

        for source_path in source_paths:
            if not source_path.exists():
                continue  # Skip missing source files
            if source_path.stat().st_mtime > cache_mtime:
                logging.info(f"Cache {cache_path.name} is older than {source_path.name}, regenerating")
                return False

        logging.info(f"Using cached intermediate file: {cache_path.name}")
        return True
    else:
        raise ValueError(f"Invalid use_cached value: {use_cached}")


def get_cache_status_message(cache_path: Union[str, Path], use_cached: bool) -> str:
    """Generate a descriptive message about cache usage for logging."""
    cache_path = Path(cache_path)

    if use_cached:
        return f"Using cached intermediate: {cache_path.name}"
    else:
        return f"Regenerating intermediate: {cache_path.name}"


def should_use_cache_unified(
    cache_path: Union[str, Path],
    source_paths: list[Union[str, Path]],
    cache_policy: Literal["auto", "always", "force_regenerate"],
) -> bool:
    """Unified cache decision logic for all intermediate files.

    Args:
        cache_path: Path to the cache file
        source_paths: List of source file paths to check timestamps against
        cache_policy: Caching policy:
            - "auto": Use cache if exists and newer than sources, regenerate with logging if missing/invalid
            - "always": Use cache if exists, raise error if missing/invalid
            - "force_regenerate": Always regenerate and overwrite existing cache

    Returns:
        bool: True if cache should be used, False if should regenerate

    Raises:
        ValueError: If cache_policy is invalid
    """
    if cache_policy == "force_regenerate":
        return False
    elif cache_policy == "always":
        return Path(cache_path).exists()
    elif cache_policy == "auto":
        return should_use_cached_file(cache_path, source_paths, "auto")
    else:
        raise ValueError(f"Invalid cache_policy: {cache_policy}. Must be one of: auto, always, force_regenerate")
