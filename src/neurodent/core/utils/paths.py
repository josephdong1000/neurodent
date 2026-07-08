"""Filesystem path helpers, temp dirs, atomic and guarded I/O."""

import contextlib
import json
import logging
import os
import re
import shutil
import uuid

from pathlib import Path
from typing import Any, Union

from neurodent import constants


def set_temp_directory(path: str | Path) -> None:
    """
    Set the temporary directory for NeuRodent operations.

    This function configures the temporary directory used by NeuRodent for intermediate
    files and operations. The directory will be created if it doesn't exist.

    Args:
        path (str | Path): Path to the temporary directory. Will be created if it doesn't exist.

    Examples:
        >>> set_temp_directory("/tmp/neurodent_temp")
        >>> set_temp_directory(Path.home() / "neurodent_workspace" / "temp")

    Note:
        This function modifies the TMPDIR environment variable, which affects
        the behavior of other temporary file operations in the process.
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(path)
    logging.info(f"Temporary directory set to {path}")


def get_temp_directory() -> Path:
    """
    Get the current temporary directory used by NeuRodent.

    Returns:
        Path: Path object representing the current temporary directory.

    Examples:
        >>> temp_dir = get_temp_directory()
        >>> print(f"Current temp directory: {temp_dir}")
        Current temp directory: /tmp/neurodent_temp

    Raises:
        KeyError: If TMPDIR environment variable is not set.
    """
    return Path(os.environ["TMPDIR"])


def safe_unlink(path: Union[str, Path]) -> None:
    """Delete a file if it exists, ignoring a missing file.

    Used for self-healing cache deletion: a corrupt cache file is removed so it
    can be regenerated, and a concurrently-removed file is not an error.

    Args:
        path: Path to the file to delete.
    """
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except (OSError, PermissionError) as e:
        logging.warning(f"Failed to delete {path}: {e}")


def is_si_recording_folder(path: Union[str, Path]) -> bool:
    """Return True if ``path`` looks like a SpikeInterface recording output folder.

    Recognizes the two formats written by :meth:`LongRecording.save` as well as
    folders written by NeuRodent's own :meth:`LongRecordingOrganizer.save_recording`.
    This is a safety gate so destructive overwrites only ever target folders we
    actually produced — never an arbitrary user directory.

    A folder qualifies when it is a directory and any of the following hold:

    - **Zarr**: the folder ends in ``.zarr`` and contains zarr group metadata
      (``.zattrs``, ``.zmetadata``, or ``zarr.json``).
    - **Binary**: the folder contains SpikeInterface's recognition marker
      ``si_folder.json`` (or ``binary.json``).
    - **NeuRodent**: the folder contains our own sidecar
      (:data:`~neurodent.constants.NEURODENT_SIDECAR_NAME`).

    Args:
        path: Path to inspect.

    Returns:
        bool: True if ``path`` is a recognized recording output folder.
    """
    p = Path(path)
    if not p.is_dir():
        return False

    # NeuRodent sidecar — recognizes a folder we wrote even across SI versions.
    if (p / constants.NEURODENT_SIDECAR_NAME).exists():
        return True

    # Zarr folder: suffix + zarr group metadata.
    if p.suffix == ".zarr" and (
        (p / ".zattrs").exists()
        or (p / ".zmetadata").exists()
        or (p / "zarr.json").exists()
    ):
        return True

    # Binary folder: SpikeInterface's own recognition markers.
    if (p / "si_folder.json").exists() or (p / "binary.json").exists():
        return True

    return False


def safe_rmtree(path: Union[str, Path], *, require_marker: bool = True) -> None:
    """Recursively delete a directory tree, refusing unrecognized targets.

    A guarded counterpart to :func:`safe_unlink` for directories. By default it
    will only delete a directory that :func:`is_si_recording_folder` recognizes,
    so a mistyped or malicious path can never wipe an arbitrary data directory.

    Args:
        path: Directory to remove.
        require_marker: When True (default), raise :class:`ValueError` unless the
            target is a recognized SpikeInterface/NeuRodent recording folder.

    Raises:
        ValueError: If ``require_marker`` is True and the target is not a
            recognized recording folder.
    """
    p = Path(path)
    if not p.exists():
        return
    if require_marker and not is_si_recording_folder(p):
        raise ValueError(
            f"Refusing to delete {p}: it does not look like a SpikeInterface "
            "recording output folder. Delete it manually if you are sure."
        )
    try:
        shutil.rmtree(p)
    except FileNotFoundError:
        pass
    except (OSError, PermissionError) as e:
        logging.warning(f"Failed to remove {p}: {e}")


@contextlib.contextmanager
def atomic_output_path(final_path: Union[str, Path]):
    """Context manager yielding a temporary sibling path for an atomic write.

    The caller writes to the yielded temporary path. On clean exit the temp file
    is atomically moved into place with :func:`os.replace`; on exception the temp
    file is removed and the original error re-raised. Because the temp file lives
    in the same directory as ``final_path`` (same filesystem), ``os.replace`` is
    atomic, so a crash mid-write can never leave a partial file at ``final_path``.

    Args:
        final_path: The destination path the content should end up at.

    Yields:
        Path: A temporary path in the same directory to write to.

    Examples:
        >>> with atomic_output_path("out.bin") as tmp:  # doctest: +SKIP
        ...     data.tofile(tmp)
    """
    final_path = Path(final_path)
    tmp_path = final_path.with_name(f"{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        yield tmp_path
    except BaseException:
        safe_unlink(tmp_path)
        raise
    else:
        os.replace(tmp_path, final_path)


def atomic_write_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    """Atomically write ``obj`` to ``path`` as JSON.

    Serializes to a temporary sibling file and atomically renames it into place,
    so an interrupted write never leaves a partial/corrupt JSON file at ``path``.

    Args:
        path: Destination JSON file path.
        obj: JSON-serializable object to write.
        indent: Indentation passed to :func:`json.dump`.
    """
    with atomic_output_path(path) as tmp:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=indent)


def filepath_to_index(filepath) -> int:
    """
    Extract the index number from a filepath.

    This function extracts the last number found in a filepath after removing common suffixes
    and file extensions. For example, from "/path/to/data_ColMajor_001.bin" it returns 1.

    Args:
        filepath (str | Path): Path to the file to extract index from.

    Returns:
        int: The extracted index number, or 0 if no number is found in the filename.

    Examples:
        >>> filepath_to_index("/path/to/data_ColMajor_001.bin")
        1
        >>> filepath_to_index("/path/to/data_2023_015_ColMajor.bin")
        15
        >>> filepath_to_index("/path/to/data_Meta_010.json")
        10
    """
    fpath = str(filepath)
    for suffix in ["_RowMajor", "_ColMajor", "_Meta"]:
        fpath = fpath.replace(suffix, "")

    # Remove only the actual file extension, not dots within the filename
    path_obj = Path(fpath)
    if path_obj.suffix:
        fpath = str(path_obj.with_suffix(""))

    fname = Path(fpath).name
    fname = re.split(r"\D+", fname)
    fname = list(filter(None, fname))
    if not fname:
        return 0
    return int(fname[-1])


def get_file_stem(filepath: Union[str, Path]) -> str:
    """Get the true stem for files, handling double extensions like .npy.gz."""
    filepath = Path(filepath)
    name = filepath.name

    return name.split(".")[0]
