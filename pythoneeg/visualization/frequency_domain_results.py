"""
Frequency Domain Spike Analysis Results
======================================

This module contains result classes for frequency-domain spike detection analysis,
providing compatibility with the existing SpikeAnalysisResult infrastructure.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import mne
from django.utils.text import slugify

try:
    import spikeinterface.core as si
    SPIKEINTERFACE_AVAILABLE = True
except ImportError:  # pragma: no cover
    si = None
    SPIKEINTERFACE_AVAILABLE = False

from .. import core
from .results import AnimalFeatureParser, SpikeAnalysisResult


class FrequencyDomainSpikeAnalysisResult(AnimalFeatureParser):
    """
    Wrapper for frequency-domain spike detection results.

    This class mirrors the SpikeAnalysisResult interface to ensure compatibility
    with existing WindowAnalysisResult.read_sars_spikes() infrastructure.
    """

    def __init__(
        self,
        result_sas: list[si.SortingAnalyzer] = None,
        result_mne: mne.io.RawArray = None,
        spike_indices: list[np.ndarray] = None,
        detection_params: dict = None,
        animal_id: str = None,
        genotype: str = None,
        animal_day: str = None,
        bin_folder_name: str = None,
        metadata: core.DDFBinaryMetadata = None,
        channel_names: list[str] = None,
        assume_from_number=False,
    ) -> None:
        """
        Initialize FrequencyDomainSpikeAnalysisResult.

        Args:
            result_sas (list[si.SortingAnalyzer], optional): SpikeInterface SortingAnalyzers for compatibility
            result_mne (mne.io.RawArray, optional): MNE RawArray with spike annotations
            spike_indices (list[np.ndarray], optional): Raw spike detection results per channel
            detection_params (dict, optional): Parameters used for spike detection
            animal_id (str, optional): Identifier for the animal
            genotype (str, optional): Genotype of animal
            animal_day (str, optional): Recording day identifier
            bin_folder_name (str, optional): Binary folder name
            metadata (core.DDFBinaryMetadata, optional): Recording metadata
            channel_names (list[str], optional): List of channel names
            assume_from_number (bool, optional): Assume channel names from numbers
        """
        # Ensure exactly one of result_sas or result_mne is provided (like SpikeAnalysisResult)
        if (result_mne is None) == (result_sas is None):
            raise ValueError("Exactly one of result_sas or result_mne must be provided")

        self.result_sas = result_sas
        self.result_mne = result_mne
        self.spike_indices = spike_indices or []
        self.detection_params = detection_params or {}
        self.animal_id = animal_id
        self.genotype = genotype
        self.animal_day = animal_day
        self.bin_folder_name = bin_folder_name
        self.metadata = metadata
        self.channel_names = channel_names
        self.assume_from_number = assume_from_number

        if channel_names:
            self.channel_abbrevs = []
            for x in self.channel_names:
                try:
                    abbrev = core.parse_chname_to_abbrev(x, assume_from_number=assume_from_number)
                    self.channel_abbrevs.append(abbrev)
                except (ValueError, AttributeError):
                    # If parsing fails, use the original channel name
                    logging.warning(f"Failed to parse channel name {x}, using original name")
                    self.channel_abbrevs.append(x)
        else:
            self.channel_abbrevs = []

        logging.info(f"Channel names: \t{self.channel_names}")
        logging.info(f"Channel abbreviations: \t{self.channel_abbrevs}")
        logging.info(f"Detection parameters: \t{self.detection_params}")

    @classmethod
    def from_detection_results(
        cls,
        spike_indices_per_channel: list[np.ndarray],
        mne_raw_with_annotations: mne.io.RawArray,
        detection_params: dict,
        animal_id: str = None,
        genotype: str = None,
        animal_day: str = None,
        bin_folder_name: str = None,
        metadata: core.DDFBinaryMetadata = None,
        assume_from_number: bool = False,
    ):
        """
        Create FrequencyDomainSpikeAnalysisResult from raw detection outputs.

        Args:
            spike_indices_per_channel: List of spike sample indices per channel
            mne_raw_with_annotations: MNE RawArray with spike annotations
            detection_params: Parameters used for detection
            animal_id: Identifier for the animal
            genotype: Genotype of animal
            animal_day: Recording day identifier
            bin_folder_name: Binary folder name
            metadata: Recording metadata
            assume_from_number: Assume channel names from numbers

        Returns:
            FrequencyDomainSpikeAnalysisResult: Initialized result object
        """
        if not SPIKEINTERFACE_AVAILABLE:
            raise ImportError("SpikeInterface is required for FrequencyDomainSpikeAnalysisResult")

        # Convert to SpikeInterface format for compatibility
        result_sas = cls._convert_to_spikeinterface(
            spike_indices_per_channel,
            mne_raw_with_annotations
        )

        channel_names = mne_raw_with_annotations.ch_names

        # Create instance with SAS first, then set MNE
        instance = cls(
            result_sas=result_sas,
            result_mne=None,
            spike_indices=spike_indices_per_channel,
            detection_params=detection_params,
            animal_id=animal_id,
            genotype=genotype,
            animal_day=animal_day,
            bin_folder_name=bin_folder_name,
            metadata=metadata,
            channel_names=channel_names,
            assume_from_number=assume_from_number,
        )

        # Now set the MNE object after initialization
        instance.result_mne = mne_raw_with_annotations

        return instance

    @staticmethod
    def _convert_to_spikeinterface(
        spike_indices_per_channel: list[np.ndarray],
        mne_raw: mne.io.RawArray
    ) -> list[si.SortingAnalyzer]:
        """
        Convert spike detection results to SpikeInterface SortingAnalyzers.

        Uses NumpySorting.from_unit_dict to create compatible objects that work
        with existing WindowAnalysisResult.read_sars_spikes() infrastructure.

        Args:
            spike_indices_per_channel: List of spike indices per channel
            mne_raw: MNE RawArray with the original data

        Returns:
            list[si.SortingAnalyzer]: SortingAnalyzers for each channel
        """
        if not SPIKEINTERFACE_AVAILABLE:
            raise ImportError("SpikeInterface is required for conversion")

        # Create SpikeInterface recording from MNE data
        data = mne_raw.get_data()  # Shape: (n_channels, n_times)
        data = data.T  # SpikeInterface expects (n_times, n_channels)
        sampling_freq = mne_raw.info['sfreq']
        channel_ids = mne_raw.ch_names

        # Create base recording with channel IDs
        recording = si.NumpyRecording(data, sampling_frequency=sampling_freq, channel_ids=channel_ids)

        # Set a simple linear probe
        from probeinterface import Probe
        probe = Probe(ndim=2)
        probe.set_contacts(
            positions=[(0, i) for i in range(len(channel_ids))],
            shapes='circle',
            shape_params={'radius': 10}
        )
        probe.set_device_channel_indices(list(range(len(channel_ids))))
        recording = recording.set_probe(probe)

        sorting_analyzers = []

        for ch_idx, spike_indices in enumerate(spike_indices_per_channel):
            # Create single-channel recording using actual recording channel IDs
            actual_channel_ids = recording.get_channel_ids()
            channel_id = actual_channel_ids[ch_idx]
            channel_recording = recording.select_channels([channel_id])

            if len(spike_indices) > 0:
                # Create sorting with all spikes as a single unit (unit corresponding to channel index)
                unit_dict = {str(ch_idx): spike_indices}  # Keep as numpy array
                sorting = si.NumpySorting.from_unit_dict(
                    unit_dict,
                    sampling_frequency=sampling_freq
                )
            else:
                # Create empty sorting if no spikes detected
                sorting = si.NumpySorting.from_unit_dict(
                    {},
                    sampling_frequency=sampling_freq
                )

            # Create SortingAnalyzer for compatibility
            sorting_analyzer = si.create_sorting_analyzer(
                sorting, channel_recording, sparse=False
            )
            sorting_analyzers.append(sorting_analyzer)

        return sorting_analyzers

    def convert_to_mne(self, chunk_len: float = 60, save_raw=True) -> mne.io.RawArray:
        """
        Convert SortingAnalyzers to MNE RawArray (mirrors SpikeAnalysisResult interface).

        Args:
            chunk_len: Chunk length for processing (compatibility parameter)
            save_raw: Whether to save the result internally

        Returns:
            mne.io.RawArray: MNE RawArray with spike annotations
        """
        if self.result_mne is None:
            if self.result_sas:
                # Use existing conversion method from SpikeAnalysisResult
                # REVIEW this could possibly be refactored into utilities
                result_mne = SpikeAnalysisResult.convert_sas_to_mne(self.result_sas, chunk_len)
                if save_raw:
                    self.result_mne = result_mne
                else:
                    return result_mne
            else:
                raise ValueError("No data available for conversion")
        return self.result_mne

    def save_fif_and_json(
        self,
        folder: Union[str, Path],
        convert_to_mne=True,
        make_folder=True,
        slugify_filebase=True,
        save_abbrevs_as_chnames=False,
        overwrite=False,
    ):
        """
        Archive frequency domain spike analysis result as fif and json files.
        Mirrors the SpikeAnalysisResult.save_fif_and_json interface.

        Args:
            folder: Destination folder to save results
            convert_to_mne: If True, convert to MNE if needed
            make_folder: If True, create folder if it doesn't exist
            slugify_filebase: If True, slugify the filename base
            save_abbrevs_as_chnames: If True, save abbreviations as channel names
            overwrite: If True, overwrite existing files
        """
        if self.result_mne is None:
            if convert_to_mne and self.result_sas:
                result_mne = self.convert_to_mne(save_raw=True)
                if result_mne is None:
                    warnings.warn("No data found for saving")
                    return
            else:
                raise ValueError("No MNE RawArray found, and convert_to_mne is False")
        else:
            result_mne = self.result_mne

        folder = Path(folder)
        if make_folder:
            folder.mkdir(parents=True, exist_ok=True)

        if slugify_filebase:
            filebase = folder / slugify(f"{self.animal_id}-{self.genotype}-{self.animal_day}")
        else:
            filebase = folder / f"{self.animal_id}-{self.genotype}-{self.animal_day}"
        filebase = str(filebase)

        if not overwrite:
            if Path(filebase + ".json").exists():
                raise FileExistsError(f"File {filebase}.json already exists")
            if Path(filebase + "-raw.fif").exists():
                raise FileExistsError(f"File {filebase}-raw.fif already exists")
        else:
            # Clean up existing files
            for pattern in ["*.json", "*-raw.fif"]:
                for f in folder.glob(pattern):
                    f.unlink()

        # Save MNE data
        result_mne.save(filebase + "-raw.fif", overwrite=overwrite)

        # Save metadata as JSON
        json_dict = {
            "animal_id": self.animal_id,
            "genotype": self.genotype,
            "animal_day": self.animal_day,
            "bin_folder_name": self.bin_folder_name,
            "metadata": self.metadata.metadata_path if self.metadata else None,
            "channel_names": self.channel_abbrevs if save_abbrevs_as_chnames else self.channel_names,
            "assume_from_number": False if save_abbrevs_as_chnames else self.assume_from_number,
            "detection_params": self.detection_params,
            "spike_counts_per_channel": [len(spikes) for spikes in self.spike_indices] if self.spike_indices else [],
        }

        with open(filebase + ".json", "w") as f:
            json.dump(json_dict, f, indent=2)

        logging.info(f"Saved FrequencyDomainSpikeAnalysisResult to {folder}")

    @classmethod
    def load_fif_and_json(cls, folder: Union[str, Path]):
        """
        Load FrequencyDomainSpikeAnalysisResult from fif and json files.
        Mirrors the SpikeAnalysisResult.load_fif_and_json interface.

        Args:
            folder: Folder containing the saved files

        Returns:
            FrequencyDomainSpikeAnalysisResult: Loaded result object
        """
        folder = Path(folder)
        if not folder.exists():
            raise ValueError(f"Folder {folder} does not exist")

        fif_files = list(folder.glob("*-raw.fif"))
        json_files = list(folder.glob("*.json"))

        if len(json_files) != 1:
            raise ValueError(f"Expected exactly one json file in {folder}")
        if len(fif_files) != 1:
            raise ValueError(f"Expected exactly one fif file in {folder}")

        fif_path = fif_files[0]
        json_path = json_files[0]

        with open(json_path, "r") as f:
            data = json.load(f)

        # Load MNE data
        result_mne = mne.io.read_raw_fif(fif_path, preload=True)

        # Extract spike indices from MNE annotations
        spike_indices = cls._extract_spike_indices_from_mne(result_mne)

        # Fix detection params: convert lists back to tuples for specific parameters
        detection_params = data.get("detection_params", {})
        tuple_params = ['bp', 'notch', 'freq_slices']
        for param in tuple_params:
            if param in detection_params and isinstance(detection_params[param], list):
                detection_params[param] = tuple(detection_params[param])

        return cls(
            result_sas=None,  # Will be generated on demand
            result_mne=result_mne,
            spike_indices=spike_indices,
            detection_params=detection_params,
            animal_id=data["animal_id"],
            genotype=data["genotype"],
            animal_day=data["animal_day"],
            bin_folder_name=data["bin_folder_name"],
            metadata=None,  # Would need to be reconstructed
            channel_names=data["channel_names"],
            assume_from_number=data["assume_from_number"],
        )

    def plot_spike_averaged_traces(
        self,
        tmin=-0.5,
        tmax=0.5,
        baseline=None,
        save_dir=None,
        animal_id=None,
        save_epoch=True
    ):
        """
        Plot spike-triggered averages for each channel.

        Based on plot_spike_evoked_by_channel from the pipeline script.

        Args:
            tmin: Start time for epochs (seconds)
            tmax: End time for epochs (seconds)
            baseline: Baseline correction period
            save_dir: Directory to save plots and epoch data
            animal_id: Animal identifier for filenames
            save_epoch: Whether to save epoch data

        Returns:
            dict: Spike counts per channel, keyed by channel index
        """
        if self.result_mne is None:
            raise ValueError("No MNE RawArray available for plotting")

        raw = self.result_mne
        events, event_id = mne.events_from_annotations(raw)
        spike_event_id = {k: v for k, v in event_id.items() if k.startswith("Spike_Ch")}

        if not spike_event_id:
            logging.warning("No spike events with label 'Spike_Ch*' found.")
            return {ch_idx: 0 for ch_idx in range(len(raw.ch_names))}

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize event counts dictionary for all channels
        n_ch = len(raw.ch_names)
        event_counts = {ch_idx: 0 for ch_idx in range(n_ch)}

        for event_name, code in spike_event_id.items():
            ch_idx = int(event_name.split("_Ch")[-1])

            channel_events = events[events[:, 2] == code]
            event_counts[ch_idx] = len(channel_events)
            
            if len(channel_events) == 0:
                logging.warning(f"No events found for {event_name}")
                continue

            # Create epochs for this channel
            epochs = mne.Epochs(
                raw, channel_events, event_id={event_name: code},
                tmin=tmin, tmax=tmax, baseline=baseline,
                picks=[ch_idx], preload=True, event_repeated='merge'
            )

            # Save epoch data if requested
            if save_epoch and animal_id and save_dir:
                saveFile_MNE = f"{animal_id}_fdsar_epoch_{ch_idx}.fif"
                savePath_MNE = save_dir / saveFile_MNE
                epochs.save(str(savePath_MNE), overwrite=True)

            # Create and save plot
            try:
                fig = epochs.plot_image(
                    title=f"{event_name} (FD Detection)",
                    show=(save_dir is None),
                    combine="mean",
                )
                if save_dir:
                    # Include animal_id in filename to prevent overwriting across days
                    if animal_id:
                        fig_path = save_dir / f"{animal_id}_{event_name}_fd_detection.png"
                    else:
                        fig_path = save_dir / f"{event_name}_fd_detection.png"
                    fig[0].savefig(str(fig_path), dpi=300, bbox_inches='tight')
                    plt.close(fig[0])
                    logging.info(f"Saved spike-averaged plot: {fig_path}")
            except Exception as e:
                logging.error(f"Failed to create plot for {event_name}: {e}")
                raise e

        return event_counts

    @staticmethod
    def _extract_spike_indices_from_mne(mne_raw: mne.io.RawArray) -> list[np.ndarray]:
        """
        Extract spike sample indices from MNE annotations.

        Args:
            mne_raw: MNE RawArray with spike annotations

        Returns:
            list[np.ndarray]: Spike indices per channel
        """
        events, event_id = mne.events_from_annotations(mne_raw)
        spike_event_id = {k: v for k, v in event_id.items() if k.startswith("Spike_Ch")}

        n_channels = len(mne_raw.ch_names)
        spike_indices = [np.array([], dtype=int) for _ in range(n_channels)]

        for event_name, code in spike_event_id.items():
            ch_idx = int(event_name.split("_Ch")[-1])
            if ch_idx < n_channels:
                # Extract sample indices from events array (column 0)
                channel_spike_samples = events[events[:, 2] == code, 0]
                spike_indices[ch_idx] = channel_spike_samples

        return spike_indices

    def get_spike_counts_per_channel(self) -> list[int]:
        """
        Get spike counts per channel.

        Returns:
            list: Number of detected spikes per channel
        """
        if self.spike_indices:
            return [len(spikes) for spikes in self.spike_indices]
        elif self.result_mne:
            # Extract from MNE annotations if spike_indices not available
            spike_indices = self._extract_spike_indices_from_mne(self.result_mne)
            return [len(spikes) for spikes in spike_indices]
        else:
            return []

    def get_total_spike_count(self) -> int:
        """Get total number of detected spikes across all channels."""
        return sum(self.get_spike_counts_per_channel())

    def __str__(self):
        """String representation of the result object."""
        spike_counts = self.get_spike_counts_per_channel()
        total_spikes = sum(spike_counts)

        return (f"FrequencyDomainSpikeAnalysisResult("
                f"animal_id={self.animal_id}, "
                f"genotype={self.genotype}, "
                f"animal_day={self.animal_day}, "
                f"channels={len(spike_counts)}, "
                f"total_spikes={total_spikes})")

    def __repr__(self):
        return self.__str__()