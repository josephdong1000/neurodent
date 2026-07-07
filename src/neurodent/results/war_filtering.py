"""Filtering for :class:`WindowAnalysisResult` (issue #134)."""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd

from neurodent import constants
from .feature_handlers import handler_for
from .filters import (
    FILTER_REGISTRY,
    ChannelInfo,
    FilterScope,
    update_bad_channels_dict_from_config,
)

if TYPE_CHECKING:
    from .window_analysis_result import WindowAnalysisResult


_WRAPPER_METHOD_NAMES: dict[str, str] = {
    "logrms_range": "get_filter_logrms_range",
    "high_rms": "get_filter_high_rms",
    "low_rms": "get_filter_low_rms",
    "high_beta": "get_filter_high_beta",
    "reject_channels": "get_filter_reject_channels",
    "reject_channels_by_session": "get_filter_reject_channels_by_recording_session",
    "morphological_smoothing": "get_filter_morphological_smoothing",
}

class WARFilterMixin:
    """Mixin: see module docstring."""

    def _channel_info(self) -> ChannelInfo:
        """Bundle channel metadata for filter functions."""
        return ChannelInfo(
            channel_names=list(self.channel_names),
            channel_abbrevs=list(self.channel_abbrevs),
        )

    def get_filter_logrms_range(self, *, z_range=3, **kwargs):
        """Filter windows based on log(rms).

        Args:
            z_range (float, optional): The z-score range to filter by. Values outside this range will be set to NaN.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        return FILTER_REGISTRY["logrms_range"].apply(
            self.result, self._channel_info(), len(self.result), z_range=z_range
        )

    def get_filter_high_rms(self, *, max_rms=500, **kwargs):
        """Filter windows based on rms.

        Args:
            max_rms (float, optional): The maximum rms value to filter by. Values above this will be set to NaN.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        return FILTER_REGISTRY["high_rms"].apply(
            self.result, self._channel_info(), len(self.result), max_rms=max_rms
        )

    def get_filter_low_rms(self, *, min_rms=30, **kwargs):
        """Filter windows based on rms.

        Args:
            min_rms (float, optional): The minimum rms value to filter by. Values below this will be set to NaN.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        return FILTER_REGISTRY["low_rms"].apply(
            self.result, self._channel_info(), len(self.result), min_rms=min_rms
        )

    def get_filter_high_beta(self, *, max_beta_prop=0.4, **kwargs):
        """Filter windows based on beta power.

        Args:
            max_beta_prop (float, optional): The maximum beta power to filter by. Values above this will be set to NaN. Defaults to 0.4.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        return FILTER_REGISTRY["high_beta"].apply(
            self.result, self._channel_info(), len(self.result), max_beta_prop=max_beta_prop
        )

    def get_filter_reject_channels(
        self,
        *,
        bad_channels: list[str] = None,
        use_abbrevs: bool = None,
        save_bad_channels: Literal["overwrite", "union", None] = "union",
        **kwargs,
    ):
        """Filter channels to reject.

        Args:
            bad_channels (list[str]): List of channels to reject. Can be either full channel names or abbreviations.
                The method will automatically detect which format is being used. If None, no filtering is performed.
            use_abbrevs (bool, optional): Override automatic detection. If True, channels are assumed to be channel abbreviations. If False, channels are assumed to be channel names.
                If None, channels are parsed to abbreviations and matched against self.channel_abbrevs.
            save_bad_channels (Literal["overwrite", "union", None], optional): How to save bad channels to self.bad_channels_dict.
                "overwrite": Replace self.bad_channels_dict completely with bad channels applied to all sessions.
                "union": Merge bad channels with existing self.bad_channels_dict for all sessions.
                None: Don't save to self.bad_channels_dict. Defaults to "union".
                Note: When using "overwrite" mode, the bad_channels parameter and bad_channels_dict parameter
                may conflict and overwrite each other's bad channel definitions if both are provided.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        channel_info = self._channel_info()
        mask = FILTER_REGISTRY["reject_channels"].apply(
            self.result,
            channel_info,
            len(self.result),
            bad_channels=bad_channels,
            use_abbrevs=use_abbrevs,
        )

        if bad_channels is not None and save_bad_channels is not None:
            animaldays = self.result["animalday"].unique()
            self.bad_channels_dict = update_bad_channels_dict_from_config(
                self.bad_channels_dict,
                {"reject_channels": {
                    "bad_channels": bad_channels,
                    "use_abbrevs": use_abbrevs,
                    "save_bad_channels": save_bad_channels,
                }},
                channel_info,
                list(animaldays),
            )
        return mask

    def get_filter_reject_channels_by_recording_session(
        self,
        *,
        bad_channels_dict: dict[str, list[str]] = None,
        use_abbrevs: bool = None,
        save_bad_channels: Literal["overwrite", "union", None] = "union",
        **kwargs,
    ):
        """Filter channels to reject for each recording session

        Args:
            bad_channels_dict (dict[str, list[str]]): Dictionary of list of channels to reject for each recording session.
                Can be either full channel names or abbreviations. The method will automatically detect which format is being used.
                If None, the method will use the bad_channels_dict passed to the constructor.
            use_abbrevs (bool, optional): Override automatic detection. If True, channels are assumed to be channel abbreviations. If False, channels are assumed to be channel names.
                If None, channels are parsed to abbreviations and matched against self.channel_abbrevs.
            save_bad_channels (Literal["overwrite", "union", None], optional): How to save bad channels to self.bad_channels_dict.
                "overwrite": Replace self.bad_channels_dict completely with bad_channels_dict.
                "union": Merge bad_channels_dict with existing self.bad_channels_dict per session.
                None: Don't save to self.bad_channels_dict. Defaults to "union".
                Note: When using "overwrite" mode, the bad_channels parameter and bad_channels_dict parameter
                may conflict and overwrite each other's bad channel definitions if both are provided.

        Returns:
            np.ndarray: Boolean array of shape (M fragments, N channels). True = keep window, False = remove window
        """
        if bad_channels_dict is None:
            bad_channels_dict = self.bad_channels_dict.copy()
        channel_info = self._channel_info()
        mask = FILTER_REGISTRY["reject_channels_by_session"].apply(
            self.result,
            channel_info,
            len(self.result),
            bad_channels_dict=bad_channels_dict,
            use_abbrevs=use_abbrevs,
        )

        if save_bad_channels is not None and bad_channels_dict:
            animaldays = self.result["animalday"].unique()
            self.bad_channels_dict = update_bad_channels_dict_from_config(
                self.bad_channels_dict,
                {"reject_channels_by_session": {
                    "bad_channels_dict": bad_channels_dict,
                    "use_abbrevs": use_abbrevs,
                    "save_bad_channels": save_bad_channels,
                }},
                channel_info,
                list(animaldays),
            )
        return mask

    def get_filter_morphological_smoothing(
        self, filter_mask: np.ndarray, *, smoothing_seconds: float, **kwargs
    ) -> np.ndarray:
        """Apply morphological smoothing to a filter mask.

        Args:
            filter_mask (np.ndarray): Input boolean mask of shape (n_windows, n_channels)
            smoothing_seconds (float): Time window in seconds for morphological operations

        Returns:
            np.ndarray: Smoothed boolean mask
        """
        return FILTER_REGISTRY["morphological_smoothing"].apply(
            filter_mask,
            self.result,
            self._channel_info(),
            smoothing_seconds=smoothing_seconds,
        )

    def filter_all(
        self,
        df: pd.DataFrame = None,
        inplace: bool = True,
        min_valid_channels: int = 3,
        filters: list[Callable] = None,
        morphological_smoothing_seconds: float | None = None,
        bad_channels: list[str] | None = None,
        save_bad_channels: Literal["overwrite", "union", None] = "union",
        **kwargs,
    ) -> "WindowAnalysisResult":
        """Apply the default filter suite. Thin wrapper around :meth:`apply_filters`.

        Args:
            df: Deprecated; ignored (kept for signature backward compat).
            inplace: If True, mutate ``self.result`` with the filtered output.
            min_valid_channels: Minimum number of valid channels per window.
            filters: Deprecated; emits a ``DeprecationWarning`` if non-None and is
                otherwise ignored.  Use :meth:`apply_filters` with a ``filter_config``
                dict for custom filter combinations.
            morphological_smoothing_seconds: If provided, smooths the combined
                mask along the time axis with this window in seconds.
            bad_channels: If provided, adds a ``reject_channels`` filter with this list.
            save_bad_channels: How to merge into ``self.bad_channels_dict``.
            **kwargs: Per-filter overrides — currently consumed:
                ``z_range`` (default 3), ``max_rms`` (500), ``min_rms`` (50),
                ``max_beta_prop`` (0.4).  Any other keys are silently ignored.
        """
        if filters is not None:
            warnings.warn(
                "Passing `filters=` to filter_all is deprecated; use apply_filters "
                "with a filter_config dict instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        filter_config: dict = {
            "logrms_range": {"z_range": kwargs.pop("z_range", 3)},
            "high_rms":     {"max_rms": kwargs.pop("max_rms", 500)},
            "low_rms":      {"min_rms": kwargs.pop("min_rms", 50)},
            "high_beta":    {"max_beta_prop": kwargs.pop("max_beta_prop", 0.4)},
            "reject_channels_by_session": {"save_bad_channels": save_bad_channels},
        }
        if bad_channels is not None:
            filter_config["reject_channels"] = {
                "bad_channels": bad_channels,
                "save_bad_channels": save_bad_channels,
            }
        if morphological_smoothing_seconds is not None:
            filter_config["morphological_smoothing"] = {
                "smoothing_seconds": morphological_smoothing_seconds,
            }

        filtered = self.apply_filters(
            filter_config=filter_config, min_valid_channels=min_valid_channels
        )
        if inplace:
            self.result = filtered.result
            self._update_instance_vars()
        return filtered

    def _create_filtered_copy(
        self, filter_mask: np.ndarray, filter_name: str = None
    ) -> "WindowAnalysisResult":
        """Create a new WindowAnalysisResult with the filter applied.

        Args:
            filter_mask (np.ndarray): Boolean mask of shape (n_windows, n_channels)
            filter_name (str, optional): Name of the filter for logging. Defaults to None.

        Returns:
            WindowAnalysisResult: New instance with filter applied
        """
        if filter_name is not None:
            logging.info(
                f"{filter_name}: filtered {filter_mask.size - np.count_nonzero(filter_mask)}/{filter_mask.size}"
            )
        filtered_result = self._apply_filter(filter_mask)
        return type(self)._from_existing(self, filtered_result)

    def filter_logrms_range(self, z_range: float = 3) -> "WindowAnalysisResult":
        """Filter based on log(rms) z-score range.

        Args:
            z_range (float): Z-score range threshold. Defaults to 3.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_logrms_range(z_range=z_range)
        return self._create_filtered_copy(mask, filter_name="logrms_range")

    def filter_high_rms(self, max_rms: float = 500) -> "WindowAnalysisResult":
        """Filter out windows with RMS above threshold.

        Args:
            max_rms (float): Maximum RMS threshold. Defaults to 500.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_high_rms(max_rms=max_rms)
        return self._create_filtered_copy(mask, filter_name="high_rms")

    def filter_low_rms(self, min_rms: float = 50) -> "WindowAnalysisResult":
        """Filter out windows with RMS below threshold.

        Args:
            min_rms (float): Minimum RMS threshold. Defaults to 50.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_low_rms(min_rms=min_rms)
        return self._create_filtered_copy(mask, filter_name="low_rms")

    def filter_high_beta(self, max_beta_prop: float = 0.4) -> "WindowAnalysisResult":
        """Filter out windows with high beta power.

        Args:
            max_beta_prop (float): Maximum beta power proportion. Defaults to 0.4.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_high_beta(max_beta_prop=max_beta_prop)
        return self._create_filtered_copy(mask, filter_name="high_beta")

    def filter_reject_channels(
        self, bad_channels: list[str], use_abbrevs: bool = None
    ) -> "WindowAnalysisResult":
        """Filter out specified bad channels.

        Args:
            bad_channels (list[str]): List of channel names to reject
            use_abbrevs (bool, optional): Whether to use abbreviations. Defaults to None.

        Returns:
            WindowAnalysisResult: New filtered instance
        """
        mask = self.get_filter_reject_channels(
            bad_channels=bad_channels, use_abbrevs=use_abbrevs
        )
        return self._create_filtered_copy(mask, filter_name="reject_channels")

    def filter_reject_channels_by_session(
        self, bad_channels_dict: dict[str, list[str]] = None, use_abbrevs: bool = None
    ) -> "WindowAnalysisResult":
        """Filter out bad channels by recording session.

        Args:
            bad_channels_dict (dict[str, list[str]], optional): Dictionary mapping recording session
                identifiers to lists of bad channel names to reject. Session identifiers are in the
                format "{animal_id} {genotype} {day}" (e.g., "A10 WT Apr-01-2023"). Channel names
                can be either full names (e.g., "Left Auditory") or abbreviations (e.g., "LAud").
                If None, uses the bad_channels_dict from the constructor. Defaults to None.
            use_abbrevs (bool, optional): Override automatic channel name format detection. If True,
                channels are assumed to be abbreviations. If False, channels are assumed to be full
                names. If None, automatically detects format and converts to abbreviations for matching.
                Defaults to None.

        Returns:
            WindowAnalysisResult: New filtered instance with bad channels masked as NaN for their
                respective recording sessions

        Examples:
            Filter specific channels per session using abbreviations:
            >>> bad_channels = {
            ...     "A10 WT Apr-01-2023": ["LAud", "RMot"],  # Session 1: reject left auditory, right motor
            ...     "A10 WT Apr-02-2023": ["LVis"]           # Session 2: reject left visual only
            ... }
            >>> filtered_war = war.filter_reject_channels_by_session(bad_channels, use_abbrevs=True)

            Filter using full channel names:
            >>> bad_channels = {
            ...     "A12 KO May-15-2023": ["Left Motor", "Right Barrel"],
            ...     "A12 KO May-16-2023": ["Left Auditory", "Left Visual", "Right Motor"]
            ... }
            >>> filtered_war = war.filter_reject_channels_by_session(bad_channels, use_abbrevs=False)

            Auto-detect channel format (recommended):
            >>> bad_channels = {
            ...     "A15 WT Jun-10-2023": ["LMot", "RBar"],  # Will auto-detect as abbreviations
            ...     "A15 WT Jun-11-2023": ["LAud"]
            ... }
            >>> filtered_war = war.filter_reject_channels_by_session(bad_channels)

        Note:
            - Session identifiers must exactly match the "animalday" values in the result DataFrame
            - Available channel abbreviations: LAud, RAud, LVis, RVis, LHip, RHip, LBar, RBar, LMot, RMot
            - Channel names are case-insensitive and support various formats (e.g., "left aud", "Left Auditory")
            - If a session identifier is not found in bad_channels_dict, a warning is logged but processing continues
            - If a channel name is not recognized, a warning is logged but other channels are still processed
        """
        mask = self.get_filter_reject_channels_by_recording_session(
            bad_channels_dict=bad_channels_dict, use_abbrevs=use_abbrevs
        )
        return self._create_filtered_copy(mask, filter_name="reject_channels_by_session")

    def apply_filters(
        self,
        filter_config: dict = None,
        min_valid_channels: int = 3,
        morphological_smoothing_seconds: float = None,
    ) -> "WindowAnalysisResult":
        """Apply multiple filters using configuration.

        Args:
            filter_config (dict, optional): Dictionary of filter names and parameters.
                Available filters: 'logrms_range', 'high_rms', 'low_rms', 'high_beta',
                'reject_channels', 'reject_channels_by_session', 'morphological_smoothing'
            min_valid_channels (int): Minimum valid channels per window. Defaults to 3.
            morphological_smoothing_seconds (float, optional): Temporal smoothing window (deprecated, use config instead)

        Returns:
            WindowAnalysisResult: New filtered instance

        Examples:
            >>> config = {
            ...     'logrms_range': {'z_range': 3},
            ...     'high_rms': {'max_rms': 500},
            ...     'reject_channels': {'bad_channels': ['LMot', 'RMot']},
            ...     'morphological_smoothing': {'smoothing_seconds': 8.0}
            ... }
            >>> filtered_war = war.apply_filters(config)
        """
        if filter_config is None:
            filter_config = {
                "logrms_range": {"z_range": 3},
                "high_rms": {"max_rms": 500},
                "low_rms": {"min_rms": 50},
                "high_beta": {"max_beta_prop": 0.4},
                "reject_channels_by_session": {},
            }

        # Translate the legacy morphological_smoothing_seconds kwarg into the registry-driven form.
        config = dict(filter_config)
        if morphological_smoothing_seconds is not None and "morphological_smoothing" not in config:
            config["morphological_smoothing"] = {"smoothing_seconds": morphological_smoothing_seconds}

        masks: list[np.ndarray] = []
        mask_post: list[tuple[str, dict]] = []

        for name, params in config.items():
            spec = FILTER_REGISTRY.get(name)
            if spec is None:
                raise ValueError(
                    f"Unknown filter: {name}. Available: {sorted(FILTER_REGISTRY)}"
                )
            params = params or {}
            if spec.scope is FilterScope.MASK_POST:
                mask_post.append((name, params))
                continue
            # Dispatch through the wrapper method so subclass overrides (and test mocks)
            # are honoured. Fall back to the registry's pure function for any future
            # filter that doesn't ship with a WindowAnalysisResult wrapper.
            wrapper = getattr(self, _WRAPPER_METHOD_NAMES.get(name, ""), None)
            if wrapper is not None:
                mask = wrapper(**params)
            else:
                mask = spec.apply(
                    self.result, self._channel_info(), len(self.result), **params
                )
            masks.append(mask)
            logging.info(f"{name}: filtered {mask.size - np.count_nonzero(mask)}/{mask.size}")

        if masks:
            filt_bool_all = np.prod(np.stack(masks, axis=-1), axis=-1).astype(bool)
        else:
            filt_bool_all = np.ones(
                (len(self.result), len(self.channel_names)), dtype=bool
            )

        for name, params in mask_post:
            spec = FILTER_REGISTRY[name]
            wrapper_attr = _WRAPPER_METHOD_NAMES.get(name)
            if wrapper_attr and hasattr(self, wrapper_attr):
                filt_bool_all = getattr(self, wrapper_attr)(filt_bool_all, **params)
            else:
                filt_bool_all = spec.apply(
                    filt_bool_all, self.result, self._channel_info(), **params
                )
            logging.info(f"{name}: applied (post-mask)")

        # Filter windows based on minimum valid channels.
        valid_channels_per_window = np.sum(filt_bool_all, axis=1)
        window_mask = valid_channels_per_window >= min_valid_channels
        filt_bool_all = filt_bool_all & window_mask[:, np.newaxis]

        return self._create_filtered_copy(filt_bool_all)

    def _apply_filter(self, filter_tfs: np.ndarray):
        result = self.result.copy()
        filter_tfs = np.asarray(filter_tfs, dtype=bool)  # (W, C)
        for feat in constants.FEATURES:
            if feat not in result.columns:
                continue
            handler = handler_for(feat)
            result[feat] = handler.apply_mask(result[feat], filter_tfs)
        return result
