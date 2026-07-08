"""Valid-recording iteration and sampling-rate checks.

Mixin for :class:`~neurodent.loading.animal_organizer.AnimalOrganizer`.
"""

from __future__ import annotations

import logging


class AoValidationMixin:
    """Mixin: see module docstring."""

    def _iter_valid_recordings(self):
        """Yield (index, lrec) pairs, skipping recordings with zero samples.

        This centralizes empty-recording validation so that compute_bad_channels,
        compute_windowed_analysis, and compute_frequency_domain_spike_analysis
        all share the same guard.
        """
        for i, lrec in enumerate(self.long_recordings):
            if (
                hasattr(lrec, "LongRecording")
                and lrec.LongRecording is not None
                and lrec.LongRecording.get_total_samples() == 0
            ):
                logging.warning(
                    f"Skipping recording {i} ({lrec.display_name}): 0 total samples"
                )
                continue
            yield i, lrec

    def _validate_sampling_rates(self):
        """Validate that all valid recordings share the same sampling rate.

        Inconsistent sampling rates across recordings lead to PSD arrays with
        different frequency-axis lengths, which causes downstream failures in
        ``_apply_filter`` and other operations that stack arrays across windows.

        Raises:
            ValueError: If recordings have different sampling rates.
        """
        sfreqs: dict[str, float] = {}
        for _i, lrec in self._iter_valid_recordings():
            long_rec = getattr(lrec, "LongRecording", None)
            if long_rec is None:
                logging.warning(
                    f"Skipping recording {_i} ({getattr(lrec, 'display_name', 'unknown')}): "
                    "LongRecording is None"
                )
                continue
            if not hasattr(long_rec, "get_sampling_frequency"):
                raise ValueError(
                    f"LongRecording for recording "
                    f"{getattr(lrec, 'display_name', f'index {_i}')!r} does not define "
                    "get_sampling_frequency()."
                )
            sf = long_rec.get_sampling_frequency()
            sfreqs[lrec.display_name] = sf

        if not sfreqs:
            return

        unique_rates = set(sfreqs.values())
        if len(unique_rates) > 1:
            details = ", ".join(
                f"{name}: {rate} Hz" for name, rate in sfreqs.items()
            )
            raise ValueError(
                f"All recordings must have the same sampling rate to produce "
                f"consistent feature shapes (e.g. PSD). "
                f"Found {len(unique_rates)} different rates: {details}"
            )
