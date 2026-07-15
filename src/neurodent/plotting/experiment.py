import logging
import warnings
from collections import Counter
from typing import Literal

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statannotations.Annotator import Annotator

from neurodent.core import utils as core_utils
from neurodent.results import WindowAnalysisResult
from neurodent import constants
from neurodent.results import extract_hist_data, extract_feature, format_channel_data


class ExperimentPlotter:
    """
    A class for creating various plots from a list of multiple experimental datasets.

    This class provides methods for creating different types of plots (boxplot, violin plot,
    scatter plot, etc.) from experimental data with consistent data processing and styling.

    Plot Ordering:
        The class automatically sorts data according to predefined plot orders for columns like
        'channel', 'genotype', 'sex', 'isday', and 'band'. Users can customize this ordering
        during initialization::

            plotter = ExperimentPlotter(wars, plot_order={'channel': ['LMot', 'RMot', ...]})

        The default plot orders are defined in constants.DF_SORT_ORDER.

    Validation and Warnings:
        The class automatically validates plot order against the processed DataFrame during plotting
        and raises warnings for any mismatches. Use validate_plot_order() to explicitly validate::

            plotter.validate_plot_order(df)

    Example:
        Customize plot ordering during initialization::

            custom_order = {
                'channel': ['LMot', 'RMot', 'LBar', 'RBar'],  # Only include specific channels
                'genotype': ['WT', 'KO'],  # Standard order
                'sex': ['Female', 'Male']  # Custom order
            }
            plotter = ExperimentPlotter(wars, plot_order=custom_order)

    Args:
        wars (WindowAnalysisResult | list[WindowAnalysisResult]): Single WindowAnalysisResult or list of WindowAnalysisResult objects
        features (list[str], optional): List of features to extract. If None, defaults to ['all']
        exclude (list[str], optional): List of features to exclude from extraction
        use_abbreviations (bool, optional): Whether to use abbreviations for channel names
        plot_order (dict, optional): Dictionary mapping column names to the order of values for plotting. If None, uses constants.DF_SORT_ORDER.

    Attributes:
        results (list[WindowAnalysisResult]): List of analysis results being plotted.
        channel_names (list[list[str]]): List of channel names for each result.
        channel_to_idx (list[dict]): Mapping of channel names to indices for each result.
        all_channel_names (list[str]): Union of all channel names across results.
        df_wars (list[pd.DataFrame]): List of dataframes for each result.
        concat_df_wars (pd.DataFrame): Concatenated dataframe of all results.
    """

    def __init__(
        self,
        wars: WindowAnalysisResult | list[WindowAnalysisResult],
        features: list[str] = None,
        exclude: list[str] = None,
        use_abbreviations: bool = True,
        plot_order: dict = None,
    ):
        features = features if features else ["all"]

        if not isinstance(wars, list):
            wars = [wars]

        if not wars:
            raise ValueError("wars cannot be empty")

        self.results = wars
        if use_abbreviations:
            self.channel_names = [war.channel_abbrevs for war in wars]
        else:
            self.channel_names = [war.channel_names for war in wars]
        self.channel_to_idx = [
            {e: i for i, e in enumerate(ch_names)} for ch_names in self.channel_names
        ]
        self.all_channel_names = sorted(
            list(set([name for ch_names in self.channel_names for name in ch_names]))
        )

        # Check for inhomogeneous channel numbers/names
        if len(set(len(channels) for channels in self.channel_names)) > 1:
            warnings.warn(
                "Inhomogeneous channel numbers across WindowAnalysisResult objects, which may cause errors. Run WAR.reorder_and_pad_channels() to fix."
            )

        # Check if channel names are consistent across all results
        first_channels = set(self.channel_names[0])
        for i, channels in enumerate(self.channel_names[1:], 1):
            if set(channels) != first_channels:
                warnings.warn(
                    f"Inhomogeneous channel names between WindowAnalysisResult {wars[i].animal_id} and first result, which may cause errors. Run WAR.reorder_and_pad_channels() to fix."
                )

        logging.info(f"channel_names: {self.channel_names}")
        logging.info(f"channel_to_idx: {self.channel_to_idx}")
        logging.info(f"all_channel_names: {self.all_channel_names}")

        animal_ids = [war.animal_id for war in wars]
        counts = Counter(animal_ids)
        duplicates = [animal_id for animal_id, count in counts.items() if count > 1]
        if duplicates:
            warnings.warn(
                f"Duplicate animal IDs found: {duplicates}. Figures will still generate, but there may be data overlap. Change WAR.animal_id to avoid this issue."
            )

        # Process all data into DataFrames
        df_wars = []
        for war in wars:
            try:
                dftemp = war.get_result(
                    features=features, exclude=exclude, allow_missing=False
                )
                df_wars.append(dftemp)
            except KeyError as e:
                logging.error(
                    f"Features missing in {war}. Exclude the missing features during ExperimentPlotter init, or recompute WARs with missing features."
                )
                raise

        self.df_wars: list[pd.DataFrame] = df_wars
        self.concat_df_wars: pd.DataFrame = pd.concat(
            df_wars, axis=0, ignore_index=True
        )  # TODO this raises a warning about df wars having columns that are none I think
        self.stats = None

        self._plot_order = (
            plot_order if plot_order is not None else constants.DF_SORT_ORDER.copy()
        )

    @staticmethod
    def band_scale(plot_lib=None):
        """Return ``plot_lib.Nominal(order=BAND_NAMES)`` — the canonical band
        order for any seaborn-objects categorical scale (x / color / row / col).

        Use this in ``.scale(x=..., color=..., ...)`` for any plot that maps
        the ``band`` column to an axis or colour.  ``pd.Categorical(ordered=True)``
        on the column upstream is necessary but not sufficient: some seaborn
        stat/move transforms (``Dodge``, ``Jitter``, ``Est``) silently lose the
        order, so the explicit ``Nominal`` is the rendering-layer source of truth.

        Args:
            plot_lib: Optional reference to ``seaborn.objects``. If ``None``,
                it is imported lazily so this module doesn't require
                seaborn-objects at import time.
        """
        if plot_lib is None:
            import seaborn.objects as plot_lib
        return plot_lib.Nominal(order=list(constants.BAND_NAMES))

    def validate_plot_order(self, df: pd.DataFrame, raise_errors: bool = False) -> dict:
        """
        Validate that the current plot_order contains all necessary categories for the data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate against (should be the DataFrame that will be sorted).
        raise_errors : bool, optional
            Whether to raise errors for validation issues. Default is False.

        Returns
        -------
        dict
            Dictionary with validation results for each column
        """
        validation_results = {}

        # Only validate columns that exist in the DataFrame
        columns_to_validate = [
            col for col in self._plot_order.keys() if col in df.columns
        ]

        for col in columns_to_validate:
            categories = self._plot_order[col]
            unique_values = set(df[col].dropna().unique())
            missing_in_order = unique_values - set(categories)

            validation_results[col] = {
                "status": "valid" if not missing_in_order else "issues",
                "unique_values": list(unique_values),
                "defined_categories": categories,
                "missing_in_order": list(missing_in_order),
            }

            if missing_in_order:
                if raise_errors:
                    raise ValueError(
                        f'Plot order for column "{col}" is missing values found in data: {missing_in_order}'
                    )
                else:
                    warnings.warn(
                        f'Plot order for column "{col}" is missing values found in data: {missing_in_order}',
                        UserWarning,
                        stacklevel=2,
                    )

        return validation_results

    def pull_timeseries_dataframe(
        self,
        feature: str,
        groupby: str | list[str],
        channels: str | list[str] = "all",
        collapse_channels: bool = False,
        average_groupby: bool = False,
        strict_groupby: bool = False,
    ):
        """
        Process feature data for plotting.

        Parameters
        ----------
        feature : str
            The feature to get.
        groupby : str or list[str]
            The variable(s) to group by.
        channels : str or list[str], optional
            The channels to get. If 'all', all channels are used.
        collapse_channels : bool, optional
            Whether to average the channels to one value.
        average_groupby : bool, optional
            Whether to average the groupby variable(s).
        strict_groupby : bool, optional
            If True, raise an exception when groupby columns contain NaN values.
            If False (default), only issue a warning.

        Returns
        -------
        df : pd.DataFrame
            A DataFrame with the feature data.
        """
        if "band" in groupby or groupby == "band":
            raise ValueError(
                # NOTE band not existing is potentially confusing error message if you are just using pull_timesereies_dataframe, there must be a more elegant way to handle this
                "'band' is not supported as a groupby variable. Use 'band' as a col/row/hue/x variable instead."
            )
        logging.info(
            f"feature: {feature}, groupby: {groupby}, channels: {channels}, collapse_channels: {collapse_channels}, average_groupby: {average_groupby}"
        )
        if channels == "all":
            channels = self.all_channel_names
        elif not isinstance(channels, list):
            channels = [channels]
        df = self.concat_df_wars.copy()  # Use the combined DataFrame from __init__

        if isinstance(groupby, str):
            groupby = [groupby]

        # Validate groupby columns exist and check for NaN values
        missing_cols = [col for col in groupby if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Groupby columns not found in data: {missing_cols}. Available columns: {df.columns.tolist()}"
            )

        # Check for NaN values in groupby columns
        nan_cols = []
        for col in groupby:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                total_count = len(df)
                nan_cols.append(f"{col} ({nan_count}/{total_count} NaN values)")

        if nan_cols:
            error_msg = (
                f"Groupby columns contain NaN values: {', '.join(nan_cols)}. "
                "This may result from previous aggregation operations (e.g., aggregate_time_windows) "
                "where these columns were not included in the groupby. "
                "Consider: 1) Including these columns in your aggregation groupby, "
                "2) Filtering out NaN rows before plotting, or "
                "3) Using different groupby columns."
            )
            if strict_groupby:
                raise ValueError(error_msg)
            else:
                warnings.warn(error_msg, UserWarning, stacklevel=2)

        groups = list(df.groupby(groupby).groups.keys())
        logging.debug(f"groups: {groups}")

        # Check if grouping resulted in any groups
        if not groups:  # FIXME this isn't triggering for empty groupby
            raise ValueError(
                f"No valid groups found when grouping by {groupby}. "
                "This may be due to all values being NaN in one or more groupby columns. "
                "Check your data and groupby parameters."
            )

        # first iterate through animals, since that determines channel idx
        # then pull out feature into matrix, assign channels, and add to dataframe
        # meanwhile carrying over the groupby feature

        dataframes = []
        for i, war in enumerate(self.results):
            df_war = self.df_wars[i]
            ch_to_idx = self.channel_to_idx[i]
            ch_names = self.channel_names[i]

            if feature not in df_war.columns:
                raise ValueError(f"'{feature}' feature not found in {war}")

            ftype = constants.classify_feature(feature)

            if ftype is constants.FeatureType.HIST:
                # HIST is special: extract both coords and values
                freq_vals, psd_vals = extract_hist_data(df_war[feature])
                n_unique_freq_vals = np.unique(freq_vals, axis=0).shape[0]
                if n_unique_freq_vals > 1:
                    raise ValueError(
                        f"Multiple frequency bin values found in {feature}: {n_unique_freq_vals} values"
                    )
                logging.debug(
                    f"freq_vals.shape: {freq_vals.shape}, psd_vals.shape: {psd_vals.shape}"
                )
                channel_dict = format_channel_data(
                    psd_vals, ftype, collapse_channels,
                    ch_to_idx=ch_to_idx, channels=channels, ch_names=ch_names,
                )
                channel_dict = channel_dict | {"freq": freq_vals.tolist()}
                vals = df_war[groupby].to_dict("list") | channel_dict

            else:
                # All non-HIST feature types: extract + format channels
                extracted, _keys = extract_feature(df_war[feature], ftype)
                logging.debug(f"vals.shape: {extracted.shape}")
                channel_dict = format_channel_data(
                    extracted, ftype, collapse_channels,
                    ch_to_idx=ch_to_idx, channels=channels, ch_names=ch_names,
                )
                vals = df_war[groupby].to_dict("list") | channel_dict

            df_feature = pd.DataFrame.from_dict(vals, orient="columns")
            dataframes.append(df_feature)

        df = pd.concat(dataframes, axis=0, ignore_index=True)

        if ftype is constants.FeatureType.HIST:
            melt_groupby = groupby + ["freq"]
        else:
            melt_groupby = groupby
        feature_cols = [col for col in df.columns if col not in melt_groupby]
        df = df.melt(
            id_vars=melt_groupby,
            value_vars=feature_cols,
            var_name="channel",
            value_name=feature,
        )

        if feature == "psd":
            df = df.explode(["psd", "freq"])

        if ftype is constants.FeatureType.LINEAR_2D:
            if df[feature].isna().any():
                logging.warning(f"{feature} contains NaNs")
                df = df[df[feature].notna()]
            df[feature] = df[feature].apply(
                lambda x: x[0]
            )  # get first component (e.g. slope from [slope, intercept])
        elif ftype.is_dict_stored:
            # For dict-stored features (BAND, BANDED_MATRIX), the band axis is always
            # the last semantic axis. Use moveaxis to bring it to front for iteration.
            band_axis_idx = ftype.semantic_axes.get("bands")
            if band_axis_idx is None:
                raise ValueError(f"Feature type {ftype} is marked as dict_stored but has no 'bands' semantic axis")

            # Verify that bands is indeed the last semantic axis in the extracted shape
            # This validates the FeatureType metadata is consistent with our expectations
            expected_last_axis = len(ftype.channel_axes) + 1  # W + channel axes + band axis
            if band_axis_idx != expected_last_axis:
                raise ValueError(
                    f"Feature type {ftype} has bands at axis {band_axis_idx}, "
                    f"but expected it at axis {expected_last_axis} (last semantic axis)"
                )

            def explode_bands(x):
                """Move band axis to front and zip with band names."""
                arr = np.asarray(x)
                expected_band_count = len(constants.BAND_NAMES)

                if arr.ndim == 0:
                    raise ValueError(
                        f"Expected banded data for feature '{feature}', but received a scalar value"
                    )

                # The band axis is always the last axis in the per-row data
                # After format_channel_data + melt:
                # - For BAND (non-matrix): per-channel data is (B,), band axis at -1
                # - For BANDED_MATRIX (matrix): data is (C, C, B), band axis at -1
                actual_band_count = arr.shape[-1]
                if actual_band_count != expected_band_count:
                    raise ValueError(
                        f"Feature '{feature}' has {actual_band_count} bands at the last axis, "
                        f"but expected {expected_band_count} to match constants.BAND_NAMES. "
                        f"Data shape: {arr.shape}"
                    )

                # Move the band axis (last position) to position 0 for iteration
                arr_bands_first = np.moveaxis(arr, -1, 0)
                return list(zip(arr_bands_first.tolist(), constants.BAND_NAMES))

            df[feature] = df[feature].apply(explode_bands)
            df = df.explode(feature)
            df[[feature, "band"]] = pd.DataFrame(df[feature].tolist(), index=df.index)

        df.reset_index(drop=True, inplace=True)

        # REVIEW is this averaging correctly, i.e. animals lumped then lump together.
        # it appears to average everything at once, but animals should be averaged first?
        # is this just a user thing?
        # it is just a user thing, but users should know about this detail
        # namely, individual animaldays are proccessed so fundamentally its like an underlying "animalday" is part of groupby
        # intuitively you'd think the groupbys are applied to the aggregated full array but that's not so
        # this shouldn't be the case but maybe merging all the DFs is a memory intensive process
        # look into this
        if average_groupby:
            groupby_cols = df.columns.drop(feature).tolist()
            logging.debug(f"groupby_cols: {groupby_cols}")
            grouped = df.groupby(groupby_cols, sort=False, dropna=False)
            df = grouped[feature].apply(core_utils.nanmean_series_of_np).reset_index()

        # baseline_means = (df_base
        #                   .groupby(remaining_groupby)[feature]
        #                   .apply(nanmean_series_of_np))

        # Validate plot order against the DataFrame that will be sorted
        self.validate_plot_order(df)

        df = core_utils.sort_dataframe_by_plot_order(df, self._plot_order)

        return df

    def plot_catplot(
        self,
        feature: str,
        groupby: str | list[str],
        df: pd.DataFrame = None,
        x: str = None,
        col: str = None,
        hue: str = None,
        kind: Literal[
            "box", "boxen", "violin", "strip", "swarm", "bar", "point"
        ] = "box",
        catplot_params: dict = None,
        channels: str | list[str] = "all",
        collapse_channels: bool = False,
        average_groupby: bool = False,
        title: str = None,
        cmap: str = None,
        stat_pairs: list[tuple[str, str]] | Literal["all", "x", "hue"] = None,
        stat_test: str = "Mann-Whitney",
        norm_test: Literal[None, "D-Agostino", "log-D-Agostino", "K-S"] = None,
    ) -> sns.FacetGrid:
        """
        Create a boxplot of feature data.
        """
        ftype = constants.classify_feature(feature)
        if ftype.is_matrix and not collapse_channels:
            raise ValueError("To plot matrix features, collapse_channels must be True")
        if ftype is constants.FeatureType.HIST:
            raise ValueError(
                f"'{feature}' is a histogram feature and is not supported in plot_catplot. Use plot_psd instead."
            )

        if df is None:
            df = self.pull_timeseries_dataframe(
                feature, groupby, channels, collapse_channels, average_groupby
            )

        if isinstance(groupby, str):
            groupby = [groupby]

        # By default, just map x = groupby0, col = groupby1, hue = channel
        default_params = {
            "data": df,
            "x": groupby[0],
            "y": feature,
            "hue": "channel",
            "col": groupby[1] if len(groupby) > 1 else None,
            "kind": kind,
            "palette": cmap,
        }
        # Update default params if x, col, or hue are explicitly provided
        if x is not None:
            default_params["x"] = x
        if col is not None:
            default_params["col"] = col
        if hue is not None:
            default_params["hue"] = hue
        if catplot_params:
            default_params.update(catplot_params)

        # Check that x, col, and hue parameters exist in dataframe columns
        for param_name in ["x", "col", "hue"]:
            if default_params[param_name] == feature:
                raise ValueError(f"'{param_name}' cannot be the same as 'feature'")
            if (
                default_params[param_name] is not None
                and default_params[param_name] not in df.columns
            ):
                raise ValueError(
                    f"Parameter '{param_name}={default_params[param_name]}' not found in dataframe columns: {df.columns.tolist()}"
                )

        # # Apply ordering to x, col, and hue if not already provided
        # for param_name in ['x', 'col', 'hue']:
        #     param_order_name = 'order' if param_name == 'x' else param_name + '_order'
        #     if default_params[param_name] in constants.PLOT_ORDER and not (catplot_params is not None and param_order_name in catplot_params):
        #         default_params[param_order_name] = constants.PLOT_ORDER[default_params[param_name]]

        # Create boxplot using seaborn
        g = sns.catplot(**default_params)

        g.set_xticklabels(rotation=45, ha="right")
        g.set_titles(title)

        # Only try to modify legend if it exists
        if g.legend is not None:
            g.legend.set_loc("center left")
            g.legend.set_bbox_to_anchor((1.0, 0.5))

        # Add grid to y-axis for all subplots
        for ax in g.axes.flat:
            ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)

        groupby_test = [
            default_params[x]
            for x in ["x", "col", "hue"]
            if default_params[x] is not None
        ]
        match norm_test:
            case None:
                pass
            case "D-Agostino":
                normality_test = self._run_normaltest(df, feature, groupby_test)
                print(f"D-Agostino normality test: {normality_test}")
            case "log-D-Agostino":
                df_log = df.copy()
                df_log[feature] = np.log(df_log[feature])
                normality_test = self._run_normaltest(df_log, feature, groupby_test)
                print(f"D-Agostino log-transformed normality test: {normality_test}")
            case "K-S":
                normality_test = self._run_kstest(df, feature, groupby_test)
                print(f"K-S normality test: {normality_test}")
            case _:
                raise ValueError(f"{norm_test} is not supported")

        if stat_pairs:
            annot_params = default_params.copy()
            for (i, j, k), df in g.facet_data():
                ax = g.facet_axis(i, j)
                match stat_pairs:
                    case "all":
                        items = core_utils._get_groupby_keys(
                            df, [default_params["x"], default_params["hue"]]
                        )
                        pairs = core_utils._get_pairwise_combinations(items)
                    case "x":
                        items_x = core_utils._get_groupby_keys(df, default_params["x"])
                        pairs_x = core_utils._get_pairwise_combinations(items_x)
                        items_hue = core_utils._get_groupby_keys(
                            df, default_params["hue"]
                        )
                        pairs = [
                            ((pair[0], hue_item), (pair[1], hue_item))
                            for hue_item in items_hue
                            for pair in pairs_x
                        ]
                        logging.debug(f"pairs: {pairs}")
                    case "hue":
                        items_hue = core_utils._get_groupby_keys(
                            df, default_params["hue"]
                        )
                        pairs_hue = core_utils._get_pairwise_combinations(items_hue)
                        items_x = core_utils._get_groupby_keys(df, default_params["x"])
                        pairs = [
                            ((x_item, pair[0]), (x_item, pair[1]))
                            for x_item in items_x
                            for pair in pairs_hue
                        ]
                        logging.debug(f"pairs: {pairs}")
                    case list():
                        pairs = stat_pairs
                    case _:
                        raise ValueError(f"{stat_pairs} is not supported")

                if not pairs:
                    logging.warning("No pairs found for annotation")
                    continue

                annot_params["data"] = annot_params["data"].dropna()
                annotator = Annotator(ax, pairs, verbose=0, **annot_params)
                annotator.configure(
                    test=stat_test, text_format="star", loc="inside", verbose=1
                )
                annotator.apply_test(nan_policy="omit")
                annotator.annotate()

        plt.tight_layout()

        return g

    def plot_heatmap(
        self,
        feature: str,
        groupby: str | list[str],
        df: pd.DataFrame = None,
        col: str = None,
        row: str = None,
        channels: (
            str | list[str]
        ) = "all",  # REVIEW this might not be needed, since all channels should be visualized
        collapse_channels: bool = False,  # REVIEW Unable to plot a single cell, this parameter is not needed, if needed just use a catplot
        average_groupby: bool = False,  # REVIEW average groupby might be a redundant parameter, since matrix plotting already averages
        cmap: str = "RdBu_r",
        norm: colors.Normalize | None = None,
        height: float = 3,
        aspect: float = 1,
    ):
        """
        Create a 2D feature plot.

        Parameters:
        -----------
        cmap : str, default="RdBu_r"
            Colormap name or matplotlib colormap object
        norm : matplotlib.colors.Normalize, optional
            Normalization object. If None, will use CenteredNorm with auto-detected range.
            Common options:
            - colors.CenteredNorm(vcenter=0)  # Auto-detect range around 0
            - colors.Normalize(vmin=-1, vmax=1)  # Fixed range
            - colors.LogNorm()  # Logarithmic scale
        """
        if not constants.classify_feature(feature).is_matrix:
            raise ValueError(f"{feature} is not supported for 2D feature plots")

        if isinstance(groupby, str):
            groupby = [groupby]

        if df is None:
            df = self.pull_timeseries_dataframe(
                feature, groupby, channels, collapse_channels, average_groupby
            )

        # Create FacetGrid
        facet_vars = {
            "col": groupby[0] if len(groupby) > 0 else None,
            "row": groupby[1] if len(groupby) > 1 else None,
            "height": height,
            "aspect": aspect,
        }
        if col is not None:
            facet_vars["col"] = col
        if row is not None:
            facet_vars["row"] = row

        # Check that col and row parameters exist in dataframe columns
        for param_name in ["col", "row"]:
            if facet_vars[param_name] == feature:
                raise ValueError(f"'{param_name}' cannot be the same as 'feature'")
            if (
                facet_vars[param_name] is not None
                and facet_vars[param_name] not in df.columns
            ):
                raise ValueError(
                    f"Parameter '{param_name}={facet_vars[param_name]}' not found in dataframe columns: {df.columns.tolist()}"
                )

        g = sns.FacetGrid(df, **facet_vars)

        # Map the plotting function
        g.map_dataframe(
            self._plot_matrix, feature=feature, color_palette=cmap, norm=norm
        )

        # Adjust layout
        plt.tight_layout()

        return g

    def _get_default_pull_timeseries_params(self):
        """Get default parameters for heatmap plotting methods."""
        return {
            "channels": "all",
            "collapse_channels": False,
            "average_groupby": False,
        }

    def plot_heatmap_faceted(
        self,
        feature: str,
        groupby: str | list[str],
        facet_vars: list[str] | str,
        df: pd.DataFrame = None,
        **kwargs,
    ):
        if isinstance(groupby, str):
            groupby = [groupby]

        if df is None:
            pull_params = self._get_default_pull_timeseries_params()
            pull_params.update(
                {k: v for k, v in kwargs.items() if k in pull_params.keys()}
            )
            df = self.pull_timeseries_dataframe(
                feature=feature, groupby=groupby, **pull_params
            )

        # Among the variables present, there are a few that need modification
        # First modify groupby subtracting facetvars
        if isinstance(facet_vars, str):
            facet_vars = [facet_vars]

        # FIXME this is a very ad hoc modification, and is tied to fixing pulldataframe accepting band as a feature
        if feature in ["cohere", "zcohere", "imcoh", "zimcoh"]:
            groupby.append("band")

        subfacet_groupby = groupby.copy()
        for facet_var in facet_vars:
            if facet_var not in groupby:
                raise ValueError(
                    f"Facet variable {facet_var} must be present in groupby"
                )
            subfacet_groupby.remove(facet_var)

        # Then iterate over the dataframe facet_Vars unique groupby keys, passing them to plot_heatmap and building a list of facetgrids
        grids = []
        for name, group in df.groupby(
            facet_vars, sort=False
        ):  # TODO not related to here, but look at everywhere else that groupby is performed and consider if you should change it to sort=False
            g = self.plot_heatmap(
                feature=feature, groupby=subfacet_groupby, df=group, **kwargs
            )

            # Create title from facet variable values
            if isinstance(name, tuple):
                title = " | ".join(f"{var}={val}" for var, val in zip(facet_vars, name))
            else:
                title = f"{facet_vars[0]}={name}"

            g.figure.suptitle(title, y=1.02)
            grids.append(g)

        return grids

    def _plot_matrix(self, data, feature, color_palette="RdBu_r", norm=None, **kwargs):
        matrices = np.array(data[feature].tolist())
        avg_matrix = np.nanmean(matrices, axis=0)

        if norm is None:
            norm = colors.CenteredNorm(vcenter=0, halfrange=1)

        # Create heatmap
        plt.imshow(avg_matrix, cmap=color_palette, norm=norm)
        plt.colorbar(fraction=0.046, pad=0.04)

        # Set ticks and labels
        n_channels = avg_matrix.shape[0]
        ch_names = self.channel_names[0]
        plt.xticks(range(n_channels), ch_names, rotation=45, ha="right")
        plt.yticks(range(n_channels), ch_names)

    def plot_diffheatmap(
        self,
        feature: str,
        groupby: str | list[str],
        baseline_key: str | bool | tuple[str, ...],
        baseline_groupby: str | list[str] = None,
        operation: Literal["subtract", "divide"] = "subtract",
        remove_baseline: bool = False,
        df: pd.DataFrame = None,
        col: str = None,
        row: str = None,
        channels: str | list[str] = "all",
        collapse_channels: bool = False,
        average_groupby: bool = False,
        cmap: str = "RdBu_r",
        norm: colors.Normalize | None = None,
        height: float = 3,
        aspect: float = 1,
    ):
        """
        Create a 2D feature plot of differences between groups. Baseline is subtracted from other groups.

        Parameters:
        -----------
        cmap : str, default="RdBu_r"
            Colormap name or matplotlib colormap object
        norm : matplotlib.colors.Normalize, optional
            Normalization object. If None, will use CenteredNorm with auto-detected range.
            Common options:
            - colors.CenteredNorm(vcenter=0)  # Auto-detect range around 0
            - colors.Normalize(vmin=-1, vmax=1)  # Fixed range
            - colors.LogNorm()  # Logarithmic scale
        """
        if not constants.classify_feature(feature).is_matrix:
            raise ValueError(f"{feature} is not supported for 2D feature plots")

        if isinstance(groupby, str):
            groupby = [groupby]

        if df is None:
            df = self.pull_timeseries_dataframe(
                feature, groupby, channels, collapse_channels, average_groupby
            )

        facet_vars = {
            "col": groupby[0] if len(groupby) > 0 else None,
            "row": groupby[1] if len(groupby) > 1 else None,
            "height": height,
            "aspect": aspect,
        }
        if col is not None:
            facet_vars["col"] = col
        if row is not None:
            facet_vars["row"] = row

        # Check that col and row parameters exist in dataframe columns
        for param_name in ["col", "row"]:
            if facet_vars[param_name] == feature:
                raise ValueError(f"'{param_name}' cannot be the same as 'feature'")
            if (
                facet_vars[param_name] is not None
                and facet_vars[param_name] not in df.columns
            ):
                raise ValueError(
                    f"Parameter '{param_name}={facet_vars[param_name]}' not found in dataframe columns: {df.columns.tolist()}"
                )

        # Subtract baseline from feature
        groupby = [x for x in [facet_vars["col"], facet_vars["row"]] if x is not None]
        df = df_normalize_baseline(
            df=df,
            feature=feature,
            groupby=groupby,
            baseline_key=baseline_key,
            baseline_groupby=baseline_groupby,
            operation=operation,
            remove_baseline=remove_baseline,
        )

        if norm is None:
            norm = colors.CenteredNorm(vcenter=0, halfrange=0.5)

        # Create FacetGrid
        g = sns.FacetGrid(df, **facet_vars)

        # Map the plotting function
        g.map_dataframe(
            self._plot_matrix, feature=feature, color_palette=cmap, norm=norm
        )

        # Adjust layout
        plt.tight_layout()

        return g

    def plot_diffheatmap_faceted(
        self,
        feature: str,
        groupby: str | list[str],
        facet_vars: str | list[str],
        baseline_key: (
            str | bool | tuple[str, ...]
        ),  # NOTE these keys and groupbys only apply to the subfacets not the overall groupby
        baseline_groupby: str | list[str] = None,
        operation: Literal["subtract", "divide"] = "subtract",
        remove_baseline: bool = False,
        df: pd.DataFrame = None,
        cmap: str = "RdBu_r",
        norm: colors.Normalize | None = None,
        **kwargs,
    ):
        if isinstance(groupby, str):
            groupby = [groupby]

        if df is None:
            pull_params = self._get_default_pull_timeseries_params()
            pull_params.update(
                {k: v for k, v in kwargs.items() if k in pull_params.keys()}
            )
            df = self.pull_timeseries_dataframe(
                feature=feature, groupby=groupby, **pull_params
            )

        # Among the variables present, there are a few that need modification
        # First modify groupby subtracting facetvars
        if isinstance(facet_vars, str):
            facet_vars = [facet_vars]

        # FIXME this is a very ad hoc modification, and is tied to fixing pulldataframe accepting band as a feature
        if feature in ["cohere", "zcohere", "imcoh", "zimcoh"]:
            groupby.append("band")

        subfacet_groupby = groupby.copy()
        for facet_var in facet_vars:
            if facet_var not in groupby:
                raise ValueError(
                    f"Facet variable {facet_var} must be present in groupby"
                )
            subfacet_groupby.remove(facet_var)

        # Then iterate over the dataframe facet_Vars unique groupby keys, passing them to plot_heatmap and building a list of facetgrids
        grids = []
        for name, group in df.groupby(
            facet_vars, sort=False
        ):  # TODO look at everywhere else that groupby is performed and consider changing to sort=False
            g = self.plot_diffheatmap(
                feature=feature,
                groupby=subfacet_groupby,
                baseline_key=baseline_key,
                baseline_groupby=baseline_groupby,
                operation=operation,
                remove_baseline=remove_baseline,
                df=group,
                cmap=cmap,
                norm=norm,
                **kwargs,
            )

            # Create title from facet variable values
            if isinstance(name, tuple):
                title = " | ".join(f"{var}={val}" for var, val in zip(facet_vars, name))
            else:
                title = f"{facet_vars[0]}={name}"

            g.figure.suptitle(title, y=1.02)
            grids.append(g)

        return grids

    def plot_qqplot(
        self,
        feature: str,
        groupby: str | list[str],
        df: pd.DataFrame = None,
        col: str = None,
        row: str = None,
        log: bool = False,
        channels: str | list[str] = "all",
        collapse_channels: bool = False,
        height: float = 3,
        aspect: float = 1,
        **kwargs,
    ):
        """
        Create a QQ plot of the feature data.
        """
        ftype = constants.classify_feature(feature)
        if ftype.is_matrix and not collapse_channels:
            raise ValueError("To plot matrix features, collapse_channels must be True")
        if ftype is constants.FeatureType.HIST:
            raise ValueError(
                f"'{feature}' is a histogram feature and is not supported in plot_qqplot"
            )

        if isinstance(groupby, str):
            groupby = [groupby]

        if df is None:
            df = self.pull_timeseries_dataframe(
                feature, groupby, channels, collapse_channels, average_groupby=False
            )

        # Create FacetGrid
        facet_vars = {
            "col": groupby[0],
            "row": groupby[1] if len(groupby) > 1 else None,
            "height": height,
            "aspect": aspect,
        }
        if col is not None:
            facet_vars["col"] = col
        if row is not None:
            facet_vars["row"] = row

        # Check that col and row parameters exist in dataframe columns
        for param_name in ["col", "row"]:
            if facet_vars[param_name] == feature:
                raise ValueError(f"'{param_name}' cannot be the same as 'feature'")
            if (
                facet_vars[param_name] is not None
                and facet_vars[param_name] not in df.columns
            ):
                raise ValueError(
                    f"Parameter '{param_name}={facet_vars[param_name]}' not found in dataframe columns: {df.columns.tolist()}"
                )

        g = sns.FacetGrid(df, margin_titles=True, **facet_vars)
        g.map_dataframe(self._plot_qqplot, feature=feature, log=log, **kwargs)

        g.set_titles(row_template="{row_name}", col_template="{col_name}")

        plt.tight_layout()

        return g

    def _plot_qqplot(
        self, data: pd.DataFrame, feature: str, log: bool = False, **kwargs
    ):
        x = data[feature]
        if log:
            x = np.log(x)
        x = x[np.isfinite(x)]
        ax = plt.gca()
        pp = sm.ProbPlot(x, fit=True)
        pp.qqplot(line="45", ax=ax)

    def _run_kstest(self, df: pd.DataFrame, feature: str, groupby: str | list[str]):
        """
        Run a Kolmogorov-Smirnov test for normality on the feature data.
        This is not recommended as the test is sensitive to large values.
        """
        return df.groupby(groupby)[feature].apply(
            lambda x: stats.kstest(x, cdf="norm", nan_policy="omit")
        )

    def _run_normaltest(self, df: pd.DataFrame, feature: str, groupby: str | list[str]):
        """
        Run a D'Agostino-Pearson normality test on the feature data.
        """
        return df.groupby(groupby)[feature].apply(
            lambda x: stats.normaltest(x, nan_policy="omit")
        )


def df_normalize_baseline(
    df: pd.DataFrame,
    feature: str,
    groupby: str | list[str],
    baseline_key: str | bool | tuple[str, ...],
    baseline_groupby: str | list[str] = None,
    operation: Literal["subtract", "divide"] = "subtract",
    remove_baseline: bool = False,
    strict_groupby: bool = False,  # TODO implement strict_groupby in plot_diffheatmap
):
    """
    Subtract the baseline from the feature data.
    """
    # Handle input types
    if isinstance(groupby, str):
        groupby = [groupby]
    if baseline_groupby is None:
        # If baseline_groupby not specified, use groupby
        baseline_groupby = groupby
    if isinstance(baseline_groupby, str):
        baseline_groupby = [baseline_groupby]
    if isinstance(baseline_key, str):
        baseline_key = (baseline_key,)
    if isinstance(baseline_key, bool):
        baseline_key = (baseline_key,)
    remaining_groupby = [col for col in groupby if col not in baseline_groupby]
    logging.debug(f"Groupby: {groupby}")
    logging.debug(f"Baseline groupby: {baseline_groupby}")
    logging.debug(f"Baseline key: {baseline_key}")
    logging.debug(f"Remaining groupby: {remaining_groupby}")

    # Validate columns exist
    all_groupby_cols = list(set(groupby + baseline_groupby))
    missing_cols = [col for col in all_groupby_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Groupby columns not found in data: {missing_cols}. Available columns: {df.columns.tolist()}"
        )

    # Check for NaN values in groupby columns
    nan_cols = []
    for col in all_groupby_cols:
        if df[col].isna().any():
            nan_count = df[col].isna().sum()
            total_count = len(df)
            nan_cols.append(f"{col} ({nan_count}/{total_count} NaN values)")

    if nan_cols:
        error_msg = (
            f"Groupby columns contain NaN values: {', '.join(nan_cols)}. "
            "This may result from previous aggregation operations where these columns were not included in the groupby. "
            "Consider: 1) Including these columns in your aggregation groupby, "
            "2) Filtering out NaN rows before baseline subtraction, or "
            "3) Using different groupby columns."
        )
        if strict_groupby:
            raise ValueError(error_msg)
        else:
            warnings.warn(error_msg, UserWarning, stacklevel=2)

    # Validate baseline_key length matches baseline_groupby length
    if len(baseline_key) != len(baseline_groupby):
        raise ValueError(
            f"baseline_key length ({len(baseline_key)}) must match baseline_groupby length ({len(baseline_groupby)})"
        )

    try:
        df_base = df.groupby(baseline_groupby, as_index=False).get_group(
            baseline_key
        )  # REVIEW maybe these should use dropna=True
    except KeyError:
        available_keys = list(df.groupby(baseline_groupby).groups.keys())
        raise ValueError(
            f"Baseline key {baseline_key} not found in groupby keys: {available_keys}. "
            "Check your baseline_key and baseline_groupby parameters."
        )

    if remaining_groupby:
        baseline_means = df_base.groupby(remaining_groupby)[feature].apply(
            core_utils.nanmean_series_of_np
        )
        df_merge = df.merge(
            baseline_means, how="left", on=remaining_groupby, suffixes=("", "_baseline")
        )
    else:
        baseline_means = df_base.groupby(baseline_groupby)[feature].apply(
            core_utils.nanmean_series_of_np
        )  # Global baseline
        assert len(baseline_means) == 1
        df_merge = df.assign(
            **{f"{feature}_baseline": [baseline_means.iloc[0] for _ in range(len(df))]}
        )

    if remove_baseline:
        df_merge = df_merge.loc[
            ~(df_merge[baseline_groupby] == baseline_key).all(axis=1)
        ]
        if df_merge.empty:
            raise ValueError(f"No rows found for {groupby} != {baseline_key}")

    # Normalize the feature
    if operation == "subtract":
        df_merge[feature] = df_merge[feature].subtract(
            df_merge[f"{feature}_baseline"], fill_value=0
        )
    elif operation == "divide":
        df_merge[feature] = df_merge[feature].divide(
            df_merge[f"{feature}_baseline"], fill_value=0
        )
    else:
        raise ValueError(f"Invalid operation: {operation}")

    return df_merge
