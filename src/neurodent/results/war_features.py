"""Feature extraction and averaging for :class:`WindowAnalysisResult`."""

from __future__ import annotations

from collections.abc import Sequence

import logging

import numpy as np
import pandas as pd

from neurodent import constants
from .feature_utils import average_feature, extract_linear_array


def _sanitize_feature_request(*args, **kwargs):
    from .window_analysis_result import _sanitize_feature_request as _f
    return _f(*args, **kwargs)

class WARFeatureMixin:
    """Mixin: see module docstring."""

    def get_result(
        self,
        features: Sequence[str] | str | None = None,
        exclude: Sequence[str] | str = [],
        allow_missing=False,
    ):
        """Get windowed analysis result dataframe, with helpful filters

        Args:
            features (list[str] | str | None, optional): Feature name, list of feature names,
                or None to return all features. Defaults to None (all features).
            exclude (list[str] | str, optional): Feature name or list of feature names to
                exclude from result; will override the features parameter. Defaults to [].
            allow_missing (bool, optional): If True, will return all requested features as columns regardless if they exist in result. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with features in columns and windows in rows
        """
        features = _sanitize_feature_request(features, exclude)
        if not allow_missing:
            return self.result.loc[:, self._nonfeature_columns + features]
        else:
            return self.result.reindex(columns=self._nonfeature_columns + features)

    def get_groupavg_result(
        self,
        features: Sequence[str] | str | None = None,
        exclude: Sequence[str] | str = [],
        df: pd.DataFrame = None,
        groupby="animalday",
    ):
        """Group result and average within groups. Preserves data structure and shape for each feature.

        Args:
            features (list[str] | str | None, optional): Feature name, list of feature names,
                or None to return all features. Defaults to None (all features).
            exclude (list[str] | str, optional): Feature name or list of feature names to
                exclude from result. Will override the features parameter. Defaults to [].
            df (pd.DataFrame, optional): If not None, this function will use this dataframe instead of self.result. Defaults to None.
            groupby (str, optional): Feature or list of features to group by before averaging. Passed to the `by` parameter in pd.DataFrame.groupby(). Defaults to "animalday".

        Returns:
            pd.DataFrame: Result grouped by `groupby` and averaged for each group.
        """
        result_grouped, result_validcols = self.__get_groups(
            features=features, exclude=exclude, df=df, groupby=groupby
        )
        features = _sanitize_feature_request(features, exclude)

        avg_results = []
        for f in features:
            if f in result_validcols:
                avg_result_col = result_grouped.apply(
                    average_feature, f, "duration", include_groups=False
                )
                avg_result_col.name = f
                avg_results.append(avg_result_col)
            else:
                logging.warning(f"{f} not calculated, skipping")

        return pd.concat(avg_results, axis=1)

    def __get_groups(
        self,
        features: Sequence[str] | str | None = None,
        exclude: Sequence[str] | str = [],
        df: pd.DataFrame = None,
        groupby="animalday",
    ):
        features = _sanitize_feature_request(features, exclude)
        result_win = self.result if df is None else df
        return result_win.groupby(groupby), result_win.columns

    def get_grouprows_result(
        self,
        features: Sequence[str] | str | None = None,
        exclude: Sequence[str] | str = [],
        df: pd.DataFrame = None,
        multiindex=["animalday", "animal", "genotype"],
        include=["duration", "endfile"],
    ):
        features = _sanitize_feature_request(features, exclude)
        result_win = self.result if df is None else df
        result_win = result_win.filter(features + multiindex + include)
        return result_win.set_index(multiindex)

    def get_channel_averaged_result(
        self,
        features: Sequence[str] | str | None = None,
        exclude: Sequence[str] | str = [],
        df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Get windowed analysis result with features averaged across channels.

        This method collapses the channel dimension for all requested features,
        converting multi-channel data to scalar values per time window. It handles
        three types of features differently:

        1. **Linear features** (logrms, rms, etc.): Simple average across channels
        2. **Band features** (logpsdband, logpsdfrac, etc.): Extracts each frequency
           band (delta, theta, alpha, beta, gamma) and averages across channels.
           Creates columns like: logpsdband_delta, logpsdband_theta, etc.
        3. **Matrix features** (zcohere, zimcoh, cohere, imcoh): Extracts each
           frequency band's connectivity matrix and averages the upper triangle
           (excluding diagonal). Creates columns like: zcohere_delta, zcohere_theta, etc.

        Args:
            features (list[str] | str | None, optional): Feature name, list of feature names,
                or None to return all features. Can include any combination of linear, band,
                or matrix features. Defaults to None (all features).
            exclude (list[str] | str, optional): Feature name or list of feature names to
                exclude. Defaults to [].
            df (pd.DataFrame, optional): If provided, use this dataframe instead of
                self.result. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with all features averaged to scalars per time window.
                - Non-feature columns (timestamp, animalday, etc.) are preserved
                - Band features expanded to 5 columns per feature (one per frequency band)
                - Matrix features expanded to 5 columns per feature (one per frequency band)
                - All feature values are scalars (float)

        Example:
            >>> war = WindowAnalysisResult.load_parquet_and_json(folder_path, "war.parquet", "war_metadata.json")
            >>> # Get channel-averaged zeitgeber features
            >>> df = war.get_channel_averaged_result(["logpsdband", "zcohere", "logrms"])
            >>> print(df.columns)
            ['timestamp', 'animalday', 'genotype', 'logrms',
             'logpsdband_delta', 'logpsdband_theta', 'logpsdband_alpha', 'logpsdband_beta', 'logpsdband_gamma',
             'zcohere_delta', 'zcohere_theta', 'zcohere_alpha', 'zcohere_beta', 'zcohere_gamma']
            >>> # All feature values are scalars
            >>> df['logpsdband_delta'].iloc[0]  # Returns a single float

        Note:
            This method is designed for temporal analyses (like zeitgeber) where you want
            to analyze feature trends over time without the channel dimension.
            For analyses that need channel information, use get_result() instead.

        See Also:
            - get_result(): Get features with full channel information
            - get_groupavg_result(): Average features across time windows (preserves channels)
        """
        from neurodent import constants

        features = _sanitize_feature_request(features, exclude)
        result_win = self.result if df is None else df

        # Filter to only features that exist in the dataframe
        available_features = [f for f in features if f in result_win.columns]

        # Get the base result with requested features
        df_result = result_win.loc[
            :, self._nonfeature_columns + available_features
        ].copy()

        # Classify features by type
        band_features_in_data = [
            f for f in available_features if f in constants.BAND_FEATURES
        ]
        banded_matrix_features_in_data = [
            f for f in available_features if f in constants.BANDED_MATRIX_FEATURES
        ]
        simple_matrix_features_in_data = [
            f for f in available_features if f in constants.SIMPLE_MATRIX_FEATURES
        ]
        simple_features_in_data = [
            f for f in available_features if f in constants.LINEAR_FEATURES
        ]
        linear_2d_features_in_data = [
            f for f in available_features if f in constants.LINEAR_2D_FEATURES
        ]

        # Process band features - extract all 5 bands
        for band_feature in band_features_in_data:
            if band_feature in df_result.columns:
                df_result = self._extract_band_features(
                    df_result, band_feature, constants.BAND_NAMES
                )

        # Process banded matrix features - extract all 5 bands
        for matrix_feature in banded_matrix_features_in_data:
            if matrix_feature in df_result.columns:
                df_result = self._extract_banded_matrix_features(
                    df_result, matrix_feature, constants.BAND_NAMES
                )

        # Process LINEAR_2D features - split each into one per-component column
        # (e.g. psdslope -> psdslope_slope + psdslope_intercept) so they can be
        # channel-averaged to scalars like any LINEAR feature.
        for linear_2d_feature in linear_2d_features_in_data:
            if linear_2d_feature in df_result.columns:
                df_result = self._extract_linear_2d_features(
                    df_result, linear_2d_feature
                )

        # Build list of features to average
        features_to_average = []
        features_to_average.extend(simple_features_in_data)
        features_to_average.extend(
            simple_matrix_features_in_data
        )  # pcorr, zpcorr (no bands)

        for band_feature in band_features_in_data:
            for band in constants.BAND_NAMES:
                features_to_average.append(f"{band_feature}_{band}")

        for matrix_feature in banded_matrix_features_in_data:
            for band in constants.BAND_NAMES:
                features_to_average.append(f"{matrix_feature}_{band}")

        # LINEAR_2D features: each gets one per-component expanded column.
        for linear_2d_feature in linear_2d_features_in_data:
            for component in constants.COMPONENT_LABELS.get(linear_2d_feature, []):
                features_to_average.append(f"{linear_2d_feature}_{component}")

        # Average all features across channels
        df_result = self._average_across_channels(df_result, features_to_average)

        # Drop original band/banded-matrix/linear-2d features (now that
        # components are extracted into separate columns).  These are no
        # longer needed and cannot be aggregated (contain dicts/arrays).
        features_to_drop = (
            band_features_in_data
            + banded_matrix_features_in_data
            + linear_2d_features_in_data
        )
        df_result = df_result.drop(columns=features_to_drop, errors="ignore")

        return df_result

    def _extract_band_features(
        self, df: pd.DataFrame, feature_name: str, band_names: list[str]
    ) -> pd.DataFrame:
        """Extract individual frequency bands from band features.

        Band features (logpsdband, logpsdfrac, etc.) are stored as dicts with
        band names as keys and channel arrays as values.

        Args:
            df: DataFrame containing the band feature
            feature_name: Name of the band feature column
            band_names: List of band names to extract

        Returns:
            DataFrame with new columns for each band (feature_name_bandname format)
        """
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)

        if feature_name not in df.columns:
            return df

        # Determine number of windows and channels from first element
        first_element = df[feature_name].iloc[0]
        if not isinstance(first_element, dict):
            raise ValueError(
                f"Band feature {feature_name} must be a dictionary of bands. "
                f"Got {type(first_element)}. If this is a linear feature, fix constants."
            )

        # Pre-allocate columns for all expected bands to ensure consistency
        for band_name in band_names:
            band_values = []
            for i, row_dict in enumerate(df[feature_name]):
                if not isinstance(row_dict, dict):
                    logger.warning(
                        f"Row {i} of {feature_name} is not a dict. Using NaNs."
                    )
                    band_values.append(np.full(len(self.channel_names), np.nan))
                    continue

                if band_name in row_dict:
                    val = row_dict[band_name]
                    if isinstance(val, list):
                        val = np.array(val)
                    band_values.append(val)
                else:
                    logger.warning(
                        f"Band {band_name} missing in {feature_name} at row {i}"
                    )
                    band_values.append(np.full(len(self.channel_names), np.nan))

            # Store as list of arrays/values
            df[f"{feature_name}_{band_name}"] = band_values

        return df

    def _extract_linear_2d_features(
        self, df: pd.DataFrame, feature_name: str
    ) -> pd.DataFrame:
        """Extract individual components from LINEAR_2D features.

        LINEAR_2D features (e.g. ``psdslope``) are stored as 2-D arrays of
        shape ``(n_channels, n_components)`` per row, where each channel
        has multiple components (e.g. ``[slope, intercept]`` for psdslope).
        This method splits each into one per-component column whose cells
        are per-channel arrays (length ``n_channels``), shaped exactly like
        a LINEAR feature so :meth:`_average_across_channels` can reduce it
        to a scalar per row.

        Component names come from
        :data:`neurodent.constants.COMPONENT_LABELS` (e.g.
        ``psdslope -> ["slope", "intercept"]``).  The new columns are
        ``"{feature_name}_{component}"``.

        Args:
            df: DataFrame containing the LINEAR_2D feature.
            feature_name: Name of the LINEAR_2D feature column.

        Returns:
            DataFrame with new per-component columns appended.
            Unchanged if the feature has no entry in ``COMPONENT_LABELS``.
        """
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)

        if feature_name not in df.columns:
            return df

        component_labels = constants.COMPONENT_LABELS.get(feature_name)
        if not component_labels:
            # No labels configured; can't split.  Leave as-is and let
            # downstream classification skip it.
            logger.warning(
                f"LINEAR_2D feature {feature_name!r} has no COMPONENT_LABELS entry; "
                "channel-averaging will skip it."
            )
            return df

        n_components = len(component_labels)
        for k, component in enumerate(component_labels):
            col_values = []
            for i, cell in enumerate(df[feature_name]):
                arr = np.asarray(cell)
                if arr.ndim != 2 or arr.shape[1] != n_components:
                    logger.warning(
                        f"Row {i} of {feature_name} has unexpected shape "
                        f"{arr.shape}; expected (n_channels, {n_components}). "
                        "Using NaNs."
                    )
                    col_values.append(
                        np.full(len(self.channel_names), np.nan)
                    )
                    continue
                # arr[:, k] is the k-th component across all channels —
                # same shape as a LINEAR feature's row, ready for
                # _average_across_channels.
                col_values.append(arr[:, k])
            df[f"{feature_name}_{component}"] = col_values

        return df

    def _extract_banded_matrix_features(
        self, df: pd.DataFrame, feature_name: str, band_names: list[str]
    ) -> pd.DataFrame:
        """Extract individual frequency bands from banded matrix features.

        This method handles banded matrix features (cohere, zcohere, imcoh, zimcoh)
        which are stored as dicts with band names as keys mapping to 2D matrices.

        Note: Simple matrix features (pcorr, zpcorr) should NOT be processed by this
        method - they are single 2D matrices without frequency band structure.

        Args:
            df: DataFrame containing the banded matrix feature
            feature_name: Name of the banded matrix feature column
            band_names: List of band names to extract

        Returns:
            DataFrame with new columns for each band (feature_name_bandname format)
        """
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)

        if feature_name not in df.columns:
            return df

        # Check first element to determine storage format
        first_element = df[feature_name].iloc[0]

        if isinstance(first_element, dict):
            for band_name in band_names:
                band_matrices = []
                for matrix_dict in df[feature_name]:
                    if isinstance(matrix_dict, dict) and band_name in matrix_dict:
                        matrix = matrix_dict[band_name]
                        # Convert list to numpy array if needed (legacy format)
                        if isinstance(matrix, list):
                            matrix = np.array(matrix)

                        if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                            band_matrices.append(matrix)
                        else:
                            logger.warning(
                                f"Expected 2D matrix for {feature_name}[{band_name}], "
                                f"got {type(matrix)} with shape {getattr(matrix, 'shape', 'N/A')}"
                            )
                            band_matrices.append(
                                np.full(
                                    (len(self.channel_names), len(self.channel_names)),
                                    np.nan,
                                )
                            )
                    else:
                        logger.warning(
                            f"Missing band {band_name} in {feature_name} dictionary"
                        )
                        band_matrices.append(
                            np.full(
                                (len(self.channel_names), len(self.channel_names)),
                                np.nan,
                            )
                        )

                df[f"{feature_name}_{band_name}"] = band_matrices

        elif isinstance(first_element, (np.ndarray, list)):
            if isinstance(first_element, list):
                first_element = np.array(first_element)

            if first_element.ndim == 3:
                # 3D Array format: (Bands, Ch, Ch)
                # Verify band count matches
                if first_element.shape[0] != len(band_names):
                    raise ValueError(
                        f"Matrix feature {feature_name} has {first_element.shape[0]} bands, "
                        f"but {len(band_names)} were expected ({band_names})."
                    )

                for i, band_name in enumerate(band_names):
                    band_matrices = []
                    for matrix_3d in df[feature_name]:
                        if isinstance(matrix_3d, list):
                            matrix_3d = np.array(matrix_3d)

                        if isinstance(matrix_3d, np.ndarray) and matrix_3d.ndim == 3:
                            if matrix_3d.shape[0] == len(band_names):
                                band_matrices.append(matrix_3d[i, :, :])
                            else:
                                raise ValueError(
                                    f"Band count mismatch for {feature_name}: "
                                    f"array has {matrix_3d.shape[0]} bands, expected {len(band_names)}."
                                )
                        else:
                            raise ValueError(
                                f"Expected 3D matrix for {feature_name}, "
                                f"got {type(matrix_3d)} with shape {getattr(matrix_3d, 'shape', 'N/A')}"
                            )

                    df[f"{feature_name}_{band_name}"] = band_matrices

            elif first_element.ndim == 2:
                raise ValueError(
                    f"Matrix feature {feature_name} is stored as a 2D array, but is defined as a "
                    f"banded feature. Expected a dictionary with band keys or a 3D array (Bands, Ch, Ch). "
                    f"If this feature should not have bands, add it to SIMPLE_MATRIX_FEATURES in constants."
                )
            else:
                raise ValueError(
                    f"Matrix feature {feature_name} has wrong dimensionality: {first_element.ndim}D. "
                    f"Expected 3D (Bands, Ch, Ch) or dict."
                )

        else:
            raise ValueError(
                f"Banded matrix feature {feature_name} has unexpected format: {type(first_element)}. "
                f"Expected dict with band keys or 3D array. If this is a simple matrix feature (pcorr, zpcorr), "
                f"it should not be processed by this method."
            )

        return df

    def _average_across_channels(
        self, df: pd.DataFrame, features: list[str]
    ) -> pd.DataFrame:
        """Average features across channels to produce scalar values.

        This method operates on *expanded* feature columns (e.g.
        ``cohere_delta``, ``psdband_theta``) that have already been unpacked
        from their dict-stored representation by
        :meth:`_extract_band_features` / :meth:`_extract_banded_matrix_features`.
        Because expanded names do not exist in :data:`constants.FEATURE_TYPES`,
        dispatch is based on array dimensionality rather than
        :func:`classify_feature`.

        Handles two types of features:
        - Vector features (1D arrays): Average across channels
        - Matrix features (2D arrays): Average upper triangle (excluding diagonal)

        Args:
            df: DataFrame with features as columns
            features: List of feature column names to average

        Returns:
            DataFrame with averaged features replacing original arrays
        """
        for feature in features:
            if feature not in df.columns:
                continue

            first_element = df[feature].iloc[0]

            if isinstance(first_element, (np.ndarray, list)):
                if isinstance(first_element, list):
                    first_element = np.array(first_element)

                if first_element.ndim == 1:
                    # Vector features: Mean across channels
                    try:
                        feature_arrays = extract_linear_array(df[feature])
                        feature_avg = np.nanmean(feature_arrays, axis=1)
                    except ValueError as e:
                        raise ValueError(
                            f"Feature {feature} has inconsistent channel counts across windows. "
                            f"All windows must have the same number of channels. "
                            f"This likely indicates data corruption during feature extraction. "
                            f"Original error: {e}"
                        ) from e

                    df[feature] = feature_avg

                elif first_element.ndim == 2:
                    # Matrix features: Mean of upper triangle
                    feature_avg = []
                    for matrix in df[feature].values:
                        if isinstance(matrix, list):
                            matrix = np.array(matrix)

                        # Validate matrix shape
                        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
                            logging.warning(
                                f"Expected 2D matrix for {feature}, "
                                f"got {type(matrix)} with ndim {getattr(matrix, 'ndim', 'N/A')}"
                            )
                            feature_avg.append(np.nan)
                            continue

                        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
                            # Can't get upper triangle (excluding diag) from 1x1 or smaller
                            feature_avg.append(
                                np.nanmean(matrix) if matrix.size > 0 else np.nan
                            )
                            continue

                        upper_tri_indices = np.triu_indices_from(matrix, k=1)
                        upper_tri_values = matrix[upper_tri_indices]

                        if len(upper_tri_values) == 0:
                            avg_val = np.nanmean(matrix) if matrix.size > 0 else np.nan
                        else:
                            avg_val = np.nanmean(upper_tri_values)

                        feature_avg.append(avg_val)

                    df[feature] = feature_avg

            elif isinstance(first_element, (int, float, np.number)):
                pass

        return df
