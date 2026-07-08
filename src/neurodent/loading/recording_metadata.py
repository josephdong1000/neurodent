"""Recording metadata (channels, sampling rate, units, timestamps) for the loading stage."""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

import pandas as pd

from neurodent.core.utils import (
    convert_units_to_multiplier,
    atomic_write_json,
)


class RecordingMetadata:
    """Stores metadata information for neural recordings.

    This class handles recording metadata including channel information, sampling rates,
    timestamps, and voltage units. It can be initialized either from a CSV metadata file
    (for backward compatibility with DDF binary format) or directly from parameters.

    Attributes:
        metadata_path (str | Path | None): Path to metadata CSV file if loaded from file
        metadata_df (pd.DataFrame | None): DataFrame containing metadata if loaded from file
        n_channels (int): Number of channels in the recording
        f_s (float): Sampling frequency in Hz
        V_units (str | None): Voltage units (e.g., 'µV', 'mV', 'V')
        mult_to_uV (float | None): Multiplication factor to convert to microvolts
        precision (str | None): Data precision/dtype (e.g., 'float32', 'int16')
        dt_end (datetime | None): End datetime of recording
        channel_names (list[str]): List of channel names

    Examples:
        From parameters:
        >>> meta = RecordingMetadata(
        ...     None,
        ...     n_channels=4,
        ...     f_s=1000.0,
        ...     dt_end=datetime(2023, 1, 1),
        ...     channel_names=['ch1', 'ch2', 'ch3', 'ch4']
        ... )

        From CSV file:
        >>> meta = RecordingMetadata('/path/to/metadata.csv')
    """
    def __init__(
        self,
        metadata_path: str | Path | None,
        *,
        n_channels: int | None = None,
        f_s: float | None = None,
        dt_end: datetime | None = None,
        channel_names: list[str] | None = None,
        V_units: str | None = None,
        mult_to_uV: float | None = None,
    ) -> None:
        """Initialize RecordingMetadata either from a file path or direct parameters.

        Args:
            metadata_path (str | Path | None): Path to metadata CSV file. If provided,
                other parameters are ignored and metadata is loaded from the file.
            n_channels (int, optional): Number of channels in the recording
            f_s (float, optional): Sampling frequency in Hz
            dt_end (datetime, optional): End datetime of recording
            channel_names (list[str], optional): List of channel names
            V_units (str, optional): Voltage units (e.g., 'µV', 'mV', 'V')
            mult_to_uV (float, optional): Multiplication factor to convert to microvolts

        Raises:
            ValueError: If metadata_path is None and required parameters are missing
        """
        if metadata_path is not None:
            self._init_from_path(metadata_path)
        else:
            self._init_from_params(
                n_channels, f_s, dt_end, channel_names, V_units, mult_to_uV
            )

    def _init_from_path(self, metadata_path):
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path)
        if self.metadata_df.empty:
            raise ValueError(f"Metadata file is empty: {metadata_path}")

        self.n_channels = len(self.metadata_df.index)
        self.f_s = self.__getsinglecolval(
            "SampleRate"
        )  # NOTE this may not be the same as LongRecording (Recording object) f_s, which the name should reflect
        self.V_units = self.__getsinglecolval("Units")
        self.mult_to_uV = convert_units_to_multiplier(self.V_units)
        self.precision = self.__getsinglecolval("Precision")

        if "LastEdit" in self.metadata_df.keys():
            self.dt_end = datetime.fromisoformat(self.__getsinglecolval("LastEdit"))
        else:
            self.dt_end = None
            logging.warning(
                "No LastEdit column provided in metadata. dt_end set to None"
            )

        self.channel_names = self.metadata_df["ProbeInfo"].tolist()

    def _init_from_params(
        self, n_channels, f_s, dt_end, channel_names, V_units=None, mult_to_uV=None
    ):
        if None in (n_channels, f_s, channel_names):
            raise ValueError(
                "All parameters must be provided when not using metadata_path"
            )

        self.metadata_path = None
        self.metadata_df = None
        self.n_channels = n_channels
        self.f_s = f_s  # NOTE see above note about f_s
        self.V_units = V_units
        self.mult_to_uV = mult_to_uV
        self.precision = None
        self.dt_end = dt_end

        if not isinstance(channel_names, list):
            raise ValueError("channel_names must be a list")

        self.channel_names = channel_names

    def __getsinglecolval(self, colname):
        vals = self.metadata_df.loc[:, colname]
        if len(np.unique(vals)) > 1:
            warnings.warn(f"Not all {colname}s are equal!")
        if vals.size == 0:
            return None
        return vals.iloc[0]

    def to_dict(self) -> dict:
        """Convert RecordingMetadata to a dictionary for JSON serialization."""
        return {
            "metadata_path": Path(self.metadata_path).as_posix() if self.metadata_path else None,
            "n_channels": self.n_channels,
            "f_s": self.f_s,
            "V_units": self.V_units,
            "mult_to_uV": self.mult_to_uV,
            "precision": self.precision,
            "dt_end": self.dt_end.isoformat() if self.dt_end else None,
            "channel_names": self.channel_names,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecordingMetadata":
        """Create RecordingMetadata from a dictionary (from JSON deserialization)."""
        dt_end = datetime.fromisoformat(data["dt_end"]) if data["dt_end"] else None

        return cls(
            metadata_path=None,  # We're reconstructing from cached data
            n_channels=data["n_channels"],
            f_s=data["f_s"],
            dt_end=dt_end,
            channel_names=data["channel_names"],
            V_units=data.get("V_units"),
            mult_to_uV=data.get("mult_to_uV"),
        )

    def to_json(self, file_path: Path) -> None:
        """Save RecordingMetadata to a JSON file.

        The file is written atomically (temp file + rename) so an interrupted
        write never leaves a partial/corrupt metadata sidecar.
        """
        atomic_write_json(file_path, self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, file_path: Path) -> "RecordingMetadata":
        """Load RecordingMetadata from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        # Reconstruct the object, preserving additional fields that were serialized
        instance = cls.from_dict(data)

        # Set additional fields that might not be in from_dict
        instance.V_units = data.get("V_units")
        instance.mult_to_uV = data.get("mult_to_uV")
        instance.precision = data.get("precision")

        return instance

    def update_sampling_rate(self, new_f_s: float) -> None:
        """Update the sampling rate in this metadata object.

        This should be called when the associated recording is resampled.
        """
        old_f_s = self.f_s
        self.f_s = new_f_s
        logging.info(
            f"Updated RecordingMetadata sampling rate from {old_f_s} Hz to {new_f_s} Hz"
        )

