"""Save/load recordings and NeuRodent sidecar payloads; teardown.

Mixin for :class:`~neurodent.loading.long_recording_organizer.LongRecordingOrganizer`.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

try:
    import spikeinterface.core as si
except Exception:  # pragma: no cover - optional at import time
    si = None

from .. import constants
from neurodent.core.utils import (
    atomic_write_json,
    safe_rmtree,
    is_si_recording_folder,
)
from .recording_metadata import RecordingMetadata


class LroPersistenceMixin:
    """Mixin: see module docstring."""

    def cleanup_rec(self):
        try:
            del self.LongRecording
        except AttributeError:
            logging.warning("LongRecording does not exist, probably deleted already")
        for tpath in self.temppaths:
            Path.unlink(tpath)

    def save_recording(
        self,
        output_dir: Union[str, Path],
        format: Literal["zarr", "binary"] = "zarr",
        overwrite: bool = False,
        n_jobs: int = 1,
        chunk_duration: str = "1s",
        progress_bar: bool = True,
        **kwargs,
    ) -> Path:
        """Save this LRO's recording to disk as a self-contained folder.

        Delegates the recording write to SpikeInterface's ``save()`` and additionally
        writes a NeuRodent metadata sidecar (:data:`~neurodent.constants.NEURODENT_SIDECAR_NAME`)
        inside the output folder so the full LRO state — timestamps, durations, units,
        bad channels, and labels — can be restored faithfully via :meth:`load_recording`.

        The sidecar is written last (and atomically), so an interrupted save leaves a
        folder with no sidecar, which :meth:`load_recording` detects and degrades on.

        Args:
            output_dir (Union[str, Path]): Directory to save the recording to. For
                ``format="zarr"`` a ``.zarr`` suffix is appended if not already present.
            format (Literal["zarr", "binary"], optional): Save format. Defaults to "zarr".
            overwrite (bool, optional): If False (default), raise :class:`FileExistsError`
                when the target already exists. If True, the existing target is removed
                first — but **only** when it is a recognized SpikeInterface/NeuRodent
                recording folder; an unrecognized directory is never deleted (raises
                :class:`ValueError`). Defaults to False.
            n_jobs (int, optional): Workers for SpikeInterface's chunked disk write.
                Controls the write only. Feature computation is parallelized separately,
                by ``multiprocess_mode`` on the analysis functions. Defaults to 1.
            chunk_duration (str, optional): Chunk duration for processing. Defaults to "1s".
            progress_bar (bool, optional): Show progress bar. Defaults to True.
            **kwargs: Additional arguments passed to SI's save().

        Returns:
            Path: The actual output directory where the recording was saved (with the
            ``.zarr`` suffix applied for zarr format).

        Raises:
            ImportError: If SpikeInterface is not available.
            ValueError: If there is no recording to save, or ``overwrite=True`` targets
                a directory that is not a recognized recording folder.
            FileExistsError: If the target exists and ``overwrite=False``, or the target
                exists but is not a directory.
        """
        if si is None:
            raise ImportError("SpikeInterface is required for save_recording()")

        if self.LongRecording is None:
            raise ValueError("No recording to save")

        output_dir = Path(output_dir)

        # Ensure parent directory exists
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # For zarr format, SI appends .zarr suffix to folder name. Validation and
        # deletion below must operate on this resolved path so that binary format
        # (which has no suffix) is guarded just as carefully as zarr.
        actual_output_dir = output_dir
        if format == "zarr" and not str(output_dir).endswith(".zarr"):
            actual_output_dir = output_dir.parent / f"{output_dir.name}.zarr"

        if actual_output_dir.exists():
            if not actual_output_dir.is_dir():
                raise FileExistsError(
                    f"{actual_output_dir} exists and is not a directory; refusing to overwrite."
                )
            if not overwrite:
                raise FileExistsError(
                    f"{actual_output_dir} already exists; pass overwrite=True to replace it."
                )
            # overwrite=True: only delete folders we recognize as recording output.
            if not is_si_recording_folder(actual_output_dir):
                raise ValueError(
                    f"Refusing to overwrite {actual_output_dir}: it does not look like a "
                    "SpikeInterface recording output folder. Delete it manually if you are sure."
                )
            logging.warning(f"Overwriting existing recording folder: {actual_output_dir}")
            safe_rmtree(actual_output_dir)

        self.LongRecording.save(
            folder=output_dir,
            format=format,
            n_jobs=n_jobs,
            chunk_duration=chunk_duration,
            progress_bar=progress_bar,
            **kwargs,
        )

        # Write the metadata sidecar inside the folder SI just created (written last
        # so its presence signals a complete save).
        sidecar_path = actual_output_dir / constants.NEURODENT_SIDECAR_NAME
        atomic_write_json(sidecar_path, self._create_sidecar_payload(format))

        self.base_folder_path = actual_output_dir
        self._is_in_memory = False
        logging.info(f"Saved recording to {actual_output_dir} (format={format})")

        return actual_output_dir

    def _create_sidecar_payload(self, format: str) -> dict:
        """Build the JSON-serializable LRO metadata sidecar payload.

        Captures the LRO-level state that the raw SpikeInterface recording folder does
        not carry, mirroring exactly what :meth:`split` propagates to child LROs.

        Args:
            format (str): The SI save format used ("zarr" or "binary").

        Returns:
            dict: JSON-serializable sidecar contents.
        """
        file_end_datetimes = getattr(self, "file_end_datetimes", []) or []
        return {
            "neurodent_sidecar_version": 1,
            "format": format,
            "channel_names": list(self.channel_names) if self.channel_names else [],
            "bad_channel_names": list(self.bad_channel_names),
            "labels": dict(self.labels),
            "n_truncate": self.n_truncate,
            "truncate": self.truncate,
            "datetimes_are_start": self.datetimes_are_start,
            "file_durations": list(self.file_durations),
            "cumulative_file_durations": list(self.cumulative_file_durations),
            "file_end_datetimes": [
                dt.isoformat() if dt is not None else None for dt in file_end_datetimes
            ],
            "meta": self.meta.to_dict() if self.meta is not None else None,
        }

    def _overlay_sidecar(self, data: dict) -> None:
        """Restore LRO-level metadata from a sidecar payload onto this instance.

        Args:
            data (dict): Sidecar contents produced by :meth:`_create_sidecar_payload`.
        """
        self.channel_names = data["channel_names"]
        self.bad_channel_names = list(data.get("bad_channel_names", []))
        self.labels = dict(data.get("labels", {}))
        self.n_truncate = data.get("n_truncate", 0)
        self.truncate = data.get("truncate", False)
        self.datetimes_are_start = data.get("datetimes_are_start", True)
        self.file_durations = list(data.get("file_durations", []))
        self.cumulative_file_durations = list(data.get("cumulative_file_durations", []))
        self.file_end_datetimes = [
            datetime.fromisoformat(x) if x else None
            for x in data.get("file_end_datetimes", [])
        ]
        if data.get("meta") is not None:
            meta = RecordingMetadata.from_dict(data["meta"])
            # from_dict drops these extra fields; restore them as from_json does.
            meta.V_units = data["meta"].get("V_units")
            meta.mult_to_uV = data["meta"].get("mult_to_uV")
            meta.precision = data["meta"].get("precision")
            self.meta = meta

    @classmethod
    def load_recording(
        cls,
        folder: Union[str, Path],
        *,
        strict: bool = False,
    ) -> "LongRecordingOrganizer":
        """Load a recording previously written by :meth:`save_recording`.

        Reloads the SpikeInterface recording from ``folder`` and overlays the NeuRodent
        metadata sidecar to restore the full LRO state (timestamps, durations, units,
        bad channels, labels). This is the faithful round-trip counterpart to
        :meth:`save_recording`, and gives :attr:`base_folder_path` a concrete reader.

        Args:
            folder (Union[str, Path]): Path to a saved recording folder (zarr or binary).
            strict (bool, optional): If True, raise when the sidecar is missing or
                invalid. If False (default), fall back to a bare reload (traces only,
                no restored timestamps) and emit a warning. Defaults to False.

        Returns:
            LongRecordingOrganizer: The reloaded LRO.

        Raises:
            ImportError: If SpikeInterface is not available.
            FileNotFoundError: If ``folder`` is not a directory, or (with ``strict=True``)
                the sidecar is missing.
            ValueError: With ``strict=True``, if the sidecar is present but invalid.
        """
        if si is None:
            raise ImportError("SpikeInterface is required for load_recording()")

        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Not a directory: {folder}")

        rec = si.load(folder)
        lro = cls(item=None, mode=None, recording=rec)

        sidecar_path = folder / constants.NEURODENT_SIDECAR_NAME
        if not sidecar_path.exists():
            msg = (
                f"No NeuRodent sidecar ({constants.NEURODENT_SIDECAR_NAME}) in {folder}; "
                "timestamps and labels cannot be restored."
            )
            if strict:
                raise FileNotFoundError(msg)
            logging.warning(msg)
            lro.base_folder_path = folder
            lro._is_in_memory = False
            return lro

        try:
            with open(sidecar_path, "r") as f:
                data = json.load(f)
            lro._overlay_sidecar(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            msg = f"Invalid NeuRodent sidecar in {folder}: {e}"
            if strict:
                raise ValueError(msg) from e
            logging.warning(f"{msg}. Falling back to bare reload.")
            lro.base_folder_path = folder
            lro._is_in_memory = False
            return lro

        # Consistency guard: channel count and total duration should agree.
        if lro.channel_names and len(lro.channel_names) != rec.get_num_channels():
            logging.warning(
                f"Sidecar channel count ({len(lro.channel_names)}) != recording channels "
                f"({rec.get_num_channels()}) for {folder}."
            )
        total_duration = rec.get_total_duration()
        sidecar_total = sum(lro.file_durations) if lro.file_durations else None
        if sidecar_total is not None and abs(sidecar_total - total_duration) > (
            1.0 / constants.GLOBAL_SAMPLING_RATE
        ):
            logging.warning(
                f"Sidecar total duration ({sidecar_total:.3f}s) differs from recording "
                f"duration ({total_duration:.3f}s) for {folder}."
            )

        lro.base_folder_path = folder
        lro._is_in_memory = False
        return lro
