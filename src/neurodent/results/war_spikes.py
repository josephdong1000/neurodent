"""Spike ingestion for :class:`WindowAnalysisResult`."""

from __future__ import annotations

import copy
import logging
from typing import Literal, TYPE_CHECKING

import mne
import numpy as np
import pandas as pd

from neurodent.core.utils import log_transform

if TYPE_CHECKING:
    from .frequency_domain_results import FrequencyDomainSpikeAnalysisResult


def _bin_spike_df(*args, **kwargs):
    from .window_analysis_result import _bin_spike_df as _f
    return _f(*args, **kwargs)

class WARSpikeMixin:
    """Mixin: see module docstring."""

    def read_sars_spikes(
        self,
        sars: list["FrequencyDomainSpikeAnalysisResult"],
        read_mode: Literal["sa", "mne"] = "sa",
        inplace=True,
    ):
        """
        Integrate spike analysis results into WAR by adding nspike/lognspike features.

        This method extracts spike timing information from spike detection results and bins
        them according to the WAR's time windows, adding spike count features to each row.

        Args:
            sars: List of FrequencyDomainSpikeAnalysisResult objects.
                  One result per recording session (animalday).
            read_mode: Mode for extracting spike data:
                - "sa": Read from SortingAnalyzer objects (result_sas attribute)
                - "mne": Read from MNE RawArray objects (result_mne attribute)
            inplace: If True, modifies self.result and returns self.
                    If False, returns a new WindowAnalysisResult.

        Returns:
            WindowAnalysisResult: WAR object with added spike features (nspike, lognspike).
                - If inplace=True: returns self with modified result DataFrame
                - If inplace=False: returns new WAR object with enhanced result DataFrame

        Notes:
            - The number of sars must match the number of unique animaldays in self.result
            - Spikes are binned into time windows matching the existing WAR fragments
            - nspike: array of spike counts per channel for each time window
            - lognspike: log-transformed spike counts using log_transform()

        Example:
            >>> # After computing WAR and spike detection
            >>> enhanced_war = war.read_sars_spikes(fdsar_list, read_mode="sa", inplace=False)
            >>> enhanced_war.result['nspike']  # Spike counts per channel per window
        """
        match read_mode:
            case "sa":
                spikes_all = []
                for sar in sars:  # for each continuous recording session
                    spikes_channel = []
                    for i, sa in enumerate(sar.result_sas):  # for each channel
                        spike_times = []
                        for unit in sa.sorting.get_unit_ids():  # Flatten units
                            spike_times.extend(
                                sa.sorting.get_unit_spike_train(unit_id=unit).tolist()
                            )
                        spike_times = (
                            np.array(spike_times) / sa.sorting.get_sampling_frequency()
                        )
                        spikes_channel.append(spike_times)
                    spikes_all.append(spikes_channel)
                return self._read_from_spikes_all(
                    spikes_all, inplace=inplace, animaldays=self._spike_animaldays(sars)
                )
            case "mne":
                raws = [sar.result_mne for sar in sars]
                return self.read_mnes_spikes(
                    raws, inplace=inplace, animaldays=self._spike_animaldays(sars)
                )
            case _:
                raise ValueError(f"Invalid read_mode: {read_mode}")

    def _spike_animaldays(self, sars):
        """The animalday names to match spike results by, or None to pair them in sorted order.

        Names are used when every one is an animalday of this WAR. Spike results computed by
        AnimalAnalyzer carry exactly the WAR's names. Ones saved by an earlier version may not,
        and those fall back to the old pairing, in sorted animalday order, with a warning.
        """
        names = [getattr(sar, "animal_day", None) for sar in sars]
        in_war = set(self.result["animalday"])
        if all(name in in_war for name in names):
            return names
        logging.warning(
            f"Spike results are named {names}, which are not all animaldays of this WAR "
            f"({sorted(in_war)}). Pairing them with the animaldays in sorted order instead."
        )
        return None

    def read_mnes_spikes(self, raws: list[mne.io.RawArray], inplace=True, animaldays=None):
        """
        Extract spike features from MNE RawArray objects with spike annotations.

        This method extracts spike timing from MNE annotations (where spikes are marked
        with channel-specific event labels) and bins them into WAR time windows.

        Args:
            raws: List of MNE RawArray objects with spike annotations. One per recording
                  session (animalday). Each should have annotations with channel names
                  as event labels (e.g., 'LMot', 'RMot', etc.).
            inplace: If True, modifies self.result and returns self.
                    If False, returns a new WindowAnalysisResult.
            animaldays: The animalday each raw belongs to, in the same order. Strongly
                    recommended; see _read_from_spikes_all for what happens without it.

        Returns:
            WindowAnalysisResult: WAR object with added spike features (nspike, lognspike).

        Notes:
            - Expects MNE annotations with channel names as event descriptions
            - Spike times are extracted from event onsets and binned to WAR windows
            - Channels not found in annotations will have empty spike arrays
            - Delegates to _read_from_spikes_all() for the actual binning logic

        Example:
            >>> # From MNE spike annotations
            >>> enhanced_war = war.read_mnes_spikes([mne_raw1, mne_raw2], inplace=False)
        """
        spikes_all = []
        for raw in raws:
            # each mne is a contiguous recording session
            events, event_id = mne.events_from_annotations(raw)
            event_id = {k.item(): v for k, v in event_id.items()}

            spikes_channel = []
            for channel in raw.ch_names:
                if channel not in event_id.keys():
                    logging.warning(f"Channel {channel} not found in event_id")
                    spikes_channel.append([])
                    continue
                event_id_channel = event_id[channel]
                spike_times = events[events[:, 2] == event_id_channel, 0]
                spike_times = spike_times / raw.info["sfreq"]
                spikes_channel.append(spike_times)
            spikes_all.append(spikes_channel)
        return self._read_from_spikes_all(spikes_all, inplace=inplace, animaldays=animaldays)

    def _read_from_spikes_all(
        self, spikes_all: list[list[list[float]]], inplace=True, animaldays=None
    ):
        """
        Internal method to bin spike times into WAR time windows and add as features.

        This is the common endpoint for both read_sars_spikes() and read_mnes_spikes().
        It bins spike times according to the WAR's time windows and adds nspike/lognspike
        features to the result DataFrame.

        Args:
            spikes_all: Nested list structure of spike times in seconds:
                - Outer list: recording sessions (one per animalday)
                - Middle list: channels (one per EEG channel)
                - Inner list/array: spike times in seconds for that channel
                Example: [[[0.5, 1.2], [0.8]], [[1.1, 2.3], []]]
                         = 2 sessions, 2 channels each
            inplace: If True, modifies self.result and returns self.
                    If False, returns a new WindowAnalysisResult with enhanced data.
            animaldays: The animalday each entry of spikes_all belongs to, in the same order.
                    When omitted, spikes_all is taken to follow the animaldays in sorted name
                    order, which is the order AnimalOrganizer builds them in. Passing the names
                    removes that assumption, and read_sars_spikes always does.

        Returns:
            WindowAnalysisResult: WAR object with spike features added to result DataFrame.

        Raises:
            ValueError: If the number of names and spike sets differ, or an animalday in the
                result has no spike set.

        Notes:
            - Each animalday's counts are written to that animalday's own rows. They used to
              be concatenated in sorted name order and assigned to the rows by position, but
              rows are sorted by timestamp, and names sort differently from dates across some
              month boundaries ('Feb-01-2020' before 'Jan-31-2020'). For an animal recorded
              across such a boundary every row got another window's count, shifted by the
              length of its earliest animalday: about 13 h for IQSEC2 76 and 78.
            - Uses _bin_spike_df() helper to count spikes within each time window
            - Adds two new columns:
                - 'nspike': array of spike counts per channel for each window
                - 'lognspike': log-transformed spike counts via log_transform()
        """
        # Each animalday is one recording session, and its spike times are seconds from the
        # start of that session, so they are binned against that animalday's own rows.
        row_days = self.result["animalday"].to_numpy()
        days_in_rows = list(dict.fromkeys(row_days))
        if animaldays is None:
            animaldays = sorted(days_in_rows)
        animaldays = list(animaldays)
        if len(animaldays) != len(spikes_all):
            raise ValueError(
                f"{len(spikes_all)} spike sets were given for {len(animaldays)} animaldays"
            )
        spikes_by_day = dict(zip(animaldays, spikes_all))
        missing = [day for day in days_in_rows if day not in spikes_by_day]
        if missing:
            raise ValueError(f"No spike results for animalday(s) {missing}")
        unused = [day for day in spikes_by_day if day not in set(days_in_rows)]
        if unused:
            logging.warning(f"Spike results for animaldays with no windows were ignored: {unused}")
        logging.debug(f"Animal days: {days_in_rows}")

        nspike = np.empty(len(row_days), dtype=object)
        for day in days_in_rows:
            positions = np.flatnonzero(row_days == day)
            counts = _bin_spike_df(self.result.iloc[positions], spikes_channel=spikes_by_day[day])
            for position, row_counts in zip(positions, counts):
                nspike[position] = row_counts

        result = self.result.copy()
        result["nspike"] = list(nspike)
        result["lognspike"] = list(
            log_transform(np.stack(result["nspike"].tolist(), axis=0))
        )
        if inplace:
            self.result = result
            return self
        else:
            # Create a new WindowAnalysisResult
            new_war = copy.deepcopy(self)
            new_war.result = result
            return new_war
