"""Open discovered files the way a load would, reading headers only.

The preflight check (``scripts/iqsec2/validate_config.py --readable``) used to open files
with its own hard-coded Intan call and skip every other reader without a word, so DataWave,
EDF and other datasets got no check at all. These helpers open an item with the reader and
settings the loader would use, and say what the loader would do if it failed.
"""

from __future__ import annotations

import inspect

from .discovery import DiscoveredFile
from .long_recording_organizer import LongRecordingOrganizer

# Settings the organizer consumes itself rather than handing to the reader.
_ORGANIZER_PARAMS = frozenset(inspect.signature(LongRecordingOrganizer.__init__).parameters) - {"self", "kwargs"}
_NOT_FOR_THE_READER = _ORGANIZER_PARAMS | {"skip_unreadable_files", "manual_datetimes"}

# What the loader does with a reader error on a group of files (a bin and its sidecar): it
# substitutes an empty recording and drops the file, unless the error is about the storage.
_DROPPED_ERRORS = (ValueError, IndexError, OSError, KeyError)


def reader_kwargs(lro_kwargs: dict) -> dict:
    """The part of an animal's resolved lro_kwargs that reaches the reader."""
    return {
        k: v for k, v in (lro_kwargs or {}).items()
        if k not in _NOT_FOR_THE_READER and not k.startswith("intermediate")
    }


def open_item(item: DiscoveredFile, lro_kwargs: dict) -> dict:
    """Open one discovered item with the loader's reader and describe it, reading no data.

    Args:
        item: The discovered file or file group.
        lro_kwargs: The settings the loader resolved for this item's source.

    Returns:
        dict: ``channel_names``, ``sampling_rate`` and ``n_samples`` as the reader reports
        them, before any resampling or channel selection.

    Raises:
        Exception: Whatever the reader raises.
    """
    mode = (lro_kwargs or {}).get("mode", "si")
    reader = LongRecordingOrganizer.resolve_extract_func(mode, (lro_kwargs or {}).get("extract_func"))
    kwargs = reader_kwargs(lro_kwargs)
    if mode == "mne":
        raw = reader(item, **kwargs)
        return {
            "channel_names": list(raw.ch_names),
            "sampling_rate": float(raw.info["sfreq"]),
            "n_samples": int(raw.n_times),
        }
    rec = reader(item if item.is_multi_file else item.path, **kwargs)
    names = [str(i) for i in rec.get_channel_ids()]
    if "channel_name" in rec.get_property_keys():
        names = [str(n) for n in rec.get_property("channel_name")]
    return {
        "channel_names": names,
        "sampling_rate": float(rec.get_sampling_frequency()),
        "n_samples": int(rec.get_total_samples()),
    }


def failure_outcome(item: DiscoveredFile, lro_kwargs: dict, exc: Exception) -> str:
    """What a real load would do with a reader failure on this item.

    Returns:
        str: ``"drop"`` when the loader would replace the item with an empty recording and
        carry on (a file group with a corrupt or empty payload, or a single file when the
        config sets ``skip_unreadable_files``), ``"abort"`` when the load would stop.
    """
    lro_kwargs = lro_kwargs or {}
    tolerated = item.is_multi_file or lro_kwargs.get("skip_unreadable_files", False)
    if lro_kwargs.get("mode", "si") != "si" or not tolerated or not isinstance(exc, _DROPPED_ERRORS):
        return "abort"
    try:
        LongRecordingOrganizer._reraise_if_environment_error(exc, item.get_path_list())
    except Exception:  # noqa: BLE001 - the loader re-raises exactly these
        return "abort"
    return "drop"
