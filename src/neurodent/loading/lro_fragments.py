"""Fragment-to-MNE conversion and EDF export.

Mixin for :class:`~neurodent.loading.long_recording_organizer.LongRecordingOrganizer`.
"""

from pathlib import Path
from typing import Union

import mne


class LroFragmentsMixin:
    """Mixin: see module docstring."""

    def convert_to_mne(self) -> mne.io.RawArray:
        """Convert this LongRecording object to an MNE RawArray.

        Returns:
            mne.io.RawArray: The converted MNE RawArray
        """
        data = self.LongRecording.get_traces(
            return_scaled=True
        )  # This gets data in (n_samples, n_channels) format used by SpikeInterface

        # MNE expects data in Volts (V), but SpikeInterface return_scaled=True returns microvolts (uV)
        # Convert uV to V to prevent huge values that crash MNE export (e.g. to EDF)
        data = data * 1e-6

        data = data.T  # Convert to (n_channels, n_samples) format for MNE

        info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=self.LongRecording.get_sampling_frequency(),
            ch_types="eeg",
        )

        return mne.io.RawArray(data=data, info=info)

    def save_to_edf(self, filename: Union[str, Path], overwrite: bool = False):
        """Save the recording to an EDF file via MNE.

        Args:
            filename (str | Path): Path to save the EDF file to.
            overwrite (bool): Whether to overwrite if file exists.
        """
        raw = self.convert_to_mne()
        mne.export.export_raw(str(filename), raw, fmt="edf", overwrite=overwrite)
