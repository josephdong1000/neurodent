from pathlib import Path
from typing import Literal
import warnings
import logging

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gzscore, linregress, zscore

from neurodent import constants
from neurodent.constants import (
    BandFeatureName,
    LinearFeatureName,
    MatrixFeatureName,
    PlottableFeatureName,
)
from neurodent.results import WindowAnalysisResult
from neurodent.core.utils import slugify
from neurodent.results import (
    flatten_feature_for_plotting,
    extract_feature,
)


class AnimalPlotter:
    """Class for plotting results from an AnimalOrganizer/WindowAnalysisResult.

    Args:
        war (WindowAnalysisResult): The analysis results to plot.
        save_fig (bool, optional): Whether to save figures to disk. Defaults to False.
        save_path (Path, optional): Directory where figures will be saved. Defaults to None.

    Attributes:
        window_result (WindowAnalysisResult): The analysis results being plotted.
        genotype (str): Genotype of the animal.
        channel_names (list[str]): List of channel names.
        n_channels (int): Number of channels.
        channel_abbrevs (list[str]): Abbreviated channel names for plotting.
        save_fig (bool): Whether to save figures to disk.
        save_path (Path): Directory where figures will be saved.
    """

    def __init__(
        self,
        war: WindowAnalysisResult,
        save_fig: bool = False,
        save_path: Path = None,
    ) -> None:
        self.window_result = war
        self.genotype = war.genotype
        self.channel_names = war.channel_names
        self.n_channels = len(self.channel_names)
        self.channel_abbrevs = war.channel_abbrevs
        self.save_fig = save_fig
        self.save_path: Path = save_path

    def plot_coherecorr_matrix(
        self, groupby="animalday", bands=None, figsize=None, cmap="viridis", **kwargs
    ):
        avg_result = self.__get_groupavg_coherecorr(groupby, **kwargs)

        if bands is None:
            bands = constants.BAND_NAMES + ["pcorr"]
        elif isinstance(bands, str):
            bands = [bands]
        n_row = avg_result.index.size
        # rowcount = 0
        fig, ax = plt.subplots(
            n_row, len(bands), squeeze=False, figsize=figsize, **kwargs
        )

        normlist = [
            matplotlib.colors.Normalize(
                vmin=0, vmax=np.max(np.concatenate(avg_result[band].values))
            )
            for band in bands
        ]
        for i, (_, row) in enumerate(avg_result.iterrows()):
            self._plot_coherecorr_matrixgroup(
                row,
                bands,
                ax[i, :],
                show_bandname=i == 0,
                norm_list=normlist,
                cmap=cmap,
                **kwargs,
            )
            # rowcount += 1
        self._handle_figure(fig, title="coherecorr_matrix")

    def plot_coherecorr_diff(
        self, groupby="isday", bands=None, figsize=None, cmap="bwr", **kwargs
    ):
        avg_result = self.__get_groupavg_coherecorr(groupby, **kwargs)
        avg_result = avg_result.drop("cohere", axis=1, errors="ignore")
        if len(avg_result.index) != 2:
            raise ValueError(
                f"Difference can only be calculated between 2 rows. {groupby} resulted in {len(avg_result.index)} rows"
            )

        if bands is None:
            bands = constants.BAND_NAMES + ["pcorr"]
        elif isinstance(bands, str):
            bands = [bands]

        diff_result = avg_result.iloc[1] - avg_result.iloc[0]
        diff_result.name = f"{avg_result.iloc[1].name} - {avg_result.iloc[0].name}"

        fig, ax = plt.subplots(1, len(bands), squeeze=False, figsize=figsize, **kwargs)

        self._plot_coherecorr_matrixgroup(
            diff_result,
            bands,
            ax[0, :],
            show_bandname=True,
            center_cmap=True,
            cmap=cmap,
            **kwargs,
        )
        self._handle_figure(fig, title="coherecorr_diff")

    def _plot_coherecorr_matrixgroup(
        self,
        group: pd.Series,
        bands: list[str],
        ax: list[matplotlib.axes.Axes],
        show_bandname,
        center_cmap=False,
        norm_list=None,
        show_channelname=True,
        **kwargs,
    ):
        rowname = group.name
        for i, band in enumerate(bands):
            if norm_list is None:
                if center_cmap:
                    divnorm = matplotlib.colors.CenteredNorm()
                else:
                    divnorm = None
                ax[i].imshow(group[band], norm=divnorm, **kwargs)
            else:
                ax[i].imshow(group[band], norm=norm_list[i], **kwargs)

            if show_bandname:
                ax[i].set_xlabel(band, fontsize="x-large")
                ax[i].xaxis.set_label_position("top")

            if show_channelname:
                ax[i].set_xticks(
                    range(self.n_channels), self.channel_abbrevs, rotation="vertical"
                )
                ax[i].set_yticks(range(self.n_channels), self.channel_abbrevs)
            else:
                ax[i].set_xticks(range(self.n_channels), [""] * self.n_channels)
                ax[i].set_yticks(range(self.n_channels), [""] * self.n_channels)

        ax[0].set_ylabel(rowname, rotation="horizontal", ha="right")

    def __get_groupavg_coherecorr(self, groupby="animalday", **kwargs):
        avg_result = self.window_result.get_groupavg_result(
            constants.MATRIX_FEATURES.copy(), groupby=groupby
        )
        avg_coheresplit = pd.json_normalize(avg_result["cohere"]).set_index(
            avg_result.index
        )  # Split apart the cohere dictionaries
        return avg_coheresplit.join(avg_result)

    def plot_linear_temporal(
        self,
        multiindex=["animalday", "animal", "genotype"],
        features: list[Literal[LinearFeatureName, BandFeatureName]] | None = None,
        channels: list[int] = None,
        figsize=None,
        score_type="z",
        show_endfile=False,
        **kwargs,
    ):
        """Plot linear and band features over time, one subplot per feature.

        Args:
            multiindex (list[str], optional): Index levels identifying each recording session.
            features: Linear or band feature names to plot, see
                :data:`neurodent.constants.analysis.LINEAR_FEATURES` and
                :data:`neurodent.constants.analysis.BAND_FEATURES`. Defaults to the
                non-log linear and band features.
            channels (list[int], optional): Channel indices to plot. Defaults to all.
            figsize (tuple, optional): Figure size (width, height). Defaults to None.
            score_type (str, optional): Standardization applied per channel. One of "z",
                "zall", "gz", "modz", "center", or "none". Defaults to "z".
            show_endfile (bool, optional): Draw vertical lines at recording file boundaries.
                Defaults to False.
        """
        # REVIEW this breaks for plotting psdslope, which contains both slope and intercept values.
        # Perhaps split apart psdslope more cleanly into psdslope + psdintercept when computing WAR
        if features is None:
            features = constants.LINEAR_FEATURES.copy() + constants.BAND_FEATURES.copy()
            features = [x for x in features if x and not x.startswith("log")]
        if channels is None:
            channels = np.arange(self.n_channels)

        # df_featgroups = self.window_result.get_grouped(features, groupby=groupby)
        df_rowgroup = self.window_result.get_grouprows_result(
            features, multiindex=multiindex
        )
        for i, df_row in df_rowgroup.groupby(level=0):
            fig, ax = plt.subplots(
                len(features),
                1,
                figsize=figsize,
                sharex=True,
                gridspec_kw={
                    "height_ratios": [
                        constants.FEATURE_PLOT_HEIGHT_RATIOS[x] for x in features
                    ]
                },
                squeeze=False,
            )
            plt.subplots_adjust(hspace=0)

            for j, feat in enumerate(features):
                self._plot_linear_temporalgroup(
                    group=df_row,
                    feature=feat,
                    ax=ax[j, 0],
                    score_type=score_type,
                    channels=channels,
                    show_endfile=show_endfile,
                    **kwargs,
                )
            ax[-1, 0].set_xlabel("Time (s)")
            fig.suptitle(i)
            self._handle_figure(fig, title=f"linear_temporal_{i}")

    def _plot_linear_temporalgroup(
        self,
        group: pd.DataFrame,
        feature: str,
        ax: matplotlib.axes.Axes,
        channels: list[int] = None,
        score_type: str = "z",
        duration_name="duration",
        channel_y_offset=10,
        feature_y_offset=10,
        endfile_name="endfile",
        show_endfile=False,
        show_channelname=True,
        **kwargs,
    ):
        data_Z = self.__get_linear_feature(
            group=group, feature=feature, score_type=score_type
        )

        data_t = group[duration_name]
        data_T = np.cumsum(data_t)

        # Handle both 2D and 3D feature arrays
        if data_Z.ndim == 2:
            # 2D array (time, channels) - expand to 3D for consistent handling
            data_Z = np.expand_dims(data_Z, axis=-1)
        elif data_Z.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D feature array, got {data_Z.ndim}D for feature '{feature}'"
            )

        if channels is None:
            channels = np.arange(data_Z.shape[1])
        data_Z = data_Z[:, channels, :]

        n_chan = data_Z.shape[1]
        n_feat = data_Z.shape[2]
        chan_offset = np.linspace(
            0, channel_y_offset * n_chan, n_chan, endpoint=False
        ).reshape((1, -1, 1))
        feat_offset = np.linspace(
            0, feature_y_offset * n_chan * n_feat, n_feat, endpoint=False
        ).reshape((1, 1, -1))
        data_Z += chan_offset
        data_Z += feat_offset
        ytick_offset = feat_offset.squeeze() + np.mean(chan_offset.flatten())

        for i in range(n_feat):
            ax.plot(data_T, data_Z[:, :, i], c=f"C{i}", **kwargs)
        ftype = constants.classify_feature(feature)
        if ftype is constants.FeatureType.LINEAR:
            ax.set_yticks([ytick_offset], [feature])
        elif ftype is constants.FeatureType.LINEAR_2D:
            ax.set_yticks(ytick_offset, ["psdslope", "psdintercept"])
        elif ftype is constants.FeatureType.BAND:
            ax.set_yticks(ytick_offset, constants.BAND_NAMES)
        else:
            raise ValueError(
                f"Unsupported FeatureType {ftype} for ytick labels: {feature} "
                f"(expected LINEAR, LINEAR_2D, or BAND)"
            )

        if show_endfile:
            self._plot_filediv_lines(
                group=group,
                ax=ax,
                duration_name=duration_name,
                endfile_name=endfile_name,
            )

    def __get_linear_feature(
        self, group: pd.DataFrame, feature: str, score_type="z", triag=True
    ):
        ftype = constants.classify_feature(feature)

        if ftype is constants.FeatureType.HIST:
            raise ValueError(
                f"Unsupported FeatureType {ftype} for feature extraction: {feature}"
            )

        data_X, keys = extract_feature(group[feature], ftype)

        if keys is not None and list(keys) != list(constants.BAND_NAMES):
            raise AssertionError(
                f"BAND extraction for feature {feature!r} returned non-canonical "
                f"band order {list(keys)!r}; expected {list(constants.BAND_NAMES)!r}. "
                f"AnimalPlotter's BAND tick labels assume canonical order — "
                f"fix the upstream extraction, not this assertion."
            )

        data_X = flatten_feature_for_plotting(data_X, ftype, triag=triag)

        return self._calculate_standard_data(data_X, mode=score_type, axis=0)

    def _plot_filediv_lines(
        self, group: pd.DataFrame, ax: matplotlib.axes.Axes, duration_name, endfile_name
    ):
        filedivs = self.__get_filediv_times(group, duration_name, endfile_name)
        for xpos in filedivs:
            ax.axvline(xpos, ls="--", c="black", lw=1)

    def __get_filediv_times(self, group, duration_name, endfile_name):
        cumulative = group[duration_name].cumsum().shift(fill_value=0)
        # display( group[[endfile_name]].dropna().head())
        # display(cumulative.head())
        filedivs = (
            group[endfile_name].dropna() + cumulative[group[endfile_name].notna()]
        )
        return filedivs.tolist()

    def _calculate_standard_data(self, X, mode="z", axis=0):
        match mode:
            case "z":
                data_Z = zscore(X, axis=axis, nan_policy="omit")
            case "zall":
                data_Z = zscore(X, axis=None, nan_policy="omit")
            case "gz":
                data_Z = gzscore(X, axis=axis, nan_policy="omit")
            case "modz":
                data_Z = self.__calculate_modified_zscore(X, axis=axis)
            case "none" | None:
                data_Z = X
            case "center":
                data_Z = X - np.nanmean(X, axis=axis, keepdims=True)
            case _:
                raise ValueError(f"Invalid mode {mode}")
        return data_Z

    def __calculate_modified_zscore(self, X, axis=0):
        X_mid = np.nanmedian(X, axis=axis)
        X_absdev = np.nanmedian(np.abs(X - X_mid), axis=axis)
        return 0.6745 * (X - X_mid) / X_absdev

    def plot_coherecorr_spectral(
        self,
        multiindex=["animalday", "animal", "genotype"],
        features: list[MatrixFeatureName] | None = None,
        figsize=None,
        score_type="z",
        cmap="bwr",
        triag=True,
        show_endfile=False,
        duration_name="duration",
        endfile_name="endfile",
        **kwargs,
    ):
        """Plot connectivity matrix features over time as per-band heatmaps.

        Args:
            multiindex (list[str], optional): Index levels identifying each recording session.
            features: Matrix feature names to plot, see
                :data:`neurodent.constants.analysis.MATRIX_FEATURES`. Defaults to
                ``["zcohere", "zpcorr"]``.
            figsize (tuple, optional): Figure size (width, height). Defaults to None.
            score_type (str, optional): Standardization applied per channel pair. One of
                "z", "zall", "gz", "modz", "center", or "none". Defaults to "z".
            cmap (str, optional): Colormap. Defaults to "bwr".
            triag (bool, optional): Use only the upper triangle of each matrix. Defaults to True.
            show_endfile (bool, optional): Draw vertical lines at recording file boundaries.
                Defaults to False.
            duration_name (str, optional): Column holding each row's duration. Defaults to
                "duration".
            endfile_name (str, optional): Column holding each row's end-of-file marker.
                Defaults to "endfile".
        """
        if features is None:
            features = ["zcohere", "zpcorr"]
        # Use consolidated height ratios from constants (matrix features for spectral heatmaps)

        df_rowgroup = self.window_result.get_grouprows_result(
            features, multiindex=multiindex
        )
        missing = [f for f in features if f not in df_rowgroup.columns]
        for f in missing:
            warnings.warn(f"Feature {f} not found in dataframe")
        features = [f for f in features if f not in missing]

        if not features:
            raise ValueError("No valid features found for coherecorr spectral plot")

        for i, df_row in df_rowgroup.groupby(level=0):
            fig, ax = plt.subplots(
                len(features),
                1,
                figsize=figsize,
                sharex=True,
                gridspec_kw={
                    "height_ratios": [
                        constants.FEATURE_PLOT_HEIGHT_RATIOS[x] for x in features
                    ]
                },
                squeeze=False,
            )
            plt.subplots_adjust(hspace=0)
            for j, feat in enumerate(features):
                self._plot_coherecorr_spectralgroup(
                    group=df_row,
                    feature=feat,
                    ax=ax[j, 0],
                    score_type=score_type,
                    triag=triag,
                    show_endfile=show_endfile,
                    duration_name=duration_name,
                    endfile_name=endfile_name,
                    **kwargs,
                )
            ax[-1, 0].set_xlabel("Time (s)")
            fig.suptitle(i)
            self._handle_figure(fig, title=f"coherecorr_spectral_{i}")

    def _plot_coherecorr_spectralgroup(
        self,
        group: pd.DataFrame,
        feature: str,
        ax: matplotlib.axes.Axes,
        center_cmap=True,
        score_type="z",
        norm_list=None,
        show_featurename=True,
        show_endfile=False,
        duration_name="duration",
        endfile_name="endfile",
        cmap="bwr",
        triag=True,
        **kwargs,
    ):
        data_Z = self.__get_linear_feature(
            group=group, feature=feature, score_type=score_type
        )
        std_dev = np.nanstd(data_Z.flatten())

        # data_flat = data_Z.reshape(data_Z.shape[0], -1).transpose()

        if center_cmap:
            norm = matplotlib.colors.CenteredNorm(vcenter=0, halfrange=std_dev * 2)
        else:
            norm = None

        n_ch = data_Z.shape[1]
        n_bands = len(constants.BAND_NAMES)

        for i in range(data_Z.shape[-1]):
            extent = (
                0,
                data_Z.shape[0] * group["duration"].median(),
                i * n_ch,
                (i + 1) * n_ch,
            )
            ax.imshow(
                data_Z[:, :, i].transpose(),
                interpolation="none",
                aspect="auto",
                norm=norm,
                cmap=cmap,
                extent=extent,
            )

        if show_featurename:
            if feature in ["cohere", "zcohere", "imcoh", "zimcoh"]:
                ticks = n_ch * np.linspace(
                    1 / 2, n_bands + 1 / 2, n_bands, endpoint=False
                )
                ax.set_yticks(ticks=ticks, labels=constants.BAND_NAMES)
                for ypos in np.linspace(0, n_bands * n_ch, n_bands, endpoint=False):
                    ax.axhline(ypos, lw=1, ls="--", color="black")
            elif feature in ["pcorr", "zpcorr"]:
                ax.set_yticks(ticks=[1 / 2 * n_ch], labels=[feature])
            else:
                raise ValueError(f"Unknown feature name {feature}")

        if show_endfile:
            self._plot_filediv_lines(
                group=group,
                ax=ax,
                duration_name=duration_name,
                endfile_name=endfile_name,
            )

    def plot_psd_histogram(
        self,
        groupby="animalday",
        figsize=None,
        avg_channels=False,
        plot_type="loglog",
        plot_slope=True,
        xlim=None,
        **kwargs,
    ):
        avg_result = self.window_result.get_groupavg_result(["psd"], groupby=groupby)

        n_col = avg_result.index.size
        fig, ax = plt.subplots(
            1, n_col, squeeze=False, figsize=figsize, sharex=True, sharey=True, **kwargs
        )
        plt.subplots_adjust(wspace=0)
        for i, (idx, row) in enumerate(avg_result.iterrows()):
            freqs = np.asarray(row["psd"][0])
            psd = np.asarray(row["psd"][1])
            if avg_channels:
                psd = np.nanmean(psd, axis=-1, keepdims=True)
                label = "Average"
            else:
                label = self.channel_abbrevs
            match plot_type:
                case "loglog":
                    ax[0, i].loglog(freqs, psd, label=label)
                case "semilogy":
                    ax[0, i].semilogy(freqs, psd, label=label)
                case "semilogx":
                    ax[0, i].semilogy(freqs, psd, label=label)
                case "linear":
                    ax[0, i].plot(freqs, psd, label=label)
                case _:
                    raise ValueError(f"Invalid plot type {plot_type}")

            frange = np.logical_and(
                freqs >= constants.FREQ_BAND_TOTAL[0],
                freqs <= constants.FREQ_BAND_TOTAL[1],
            )
            logf = np.log10(freqs[frange])
            logpsd = np.log10(psd[frange, :])

            linfit = np.zeros((psd.shape[1], 2))
            for k in range(psd.shape[1]):
                result = linregress(logf, logpsd[:, k], "less")
                linfit[k, :] = [result.slope, result.intercept]

            for j, (m, b) in enumerate(linfit.tolist()):
                ax[0, i].plot(
                    freqs, 10 ** (b + m * np.log10(freqs)), c=f"C{j}", alpha=0.75
                )

            ax[0, i].set_title(idx)
            ax[0, i].set_xlabel("Frequency (Hz)")
        ax[0, 0].set_ylabel("PSD (uV^2/Hz)")
        ax[0, -1].legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        ax[0, -1].set_xlim(xlim)
        self._handle_figure(fig, title="psd_histogram")

    def plot_psd_spectrogram(
        self,
        multiindex=["animalday", "animal", "genotype"],
        freq_range=(1, 50),
        center_stat="mean",
        mode="z",
        figsize=None,
        cmap="magma",
        **kwargs,
    ):
        df_rowgroup = self.window_result.get_grouprows_result(
            ["psd"], multiindex=multiindex
        )
        for i, df_row in df_rowgroup.groupby(level=0):
            freqs = np.asarray(df_row.iloc[0]["psd"][0])
            psd = np.array([x[1] for x in df_row["psd"].tolist()])
            match center_stat:
                case "mean":
                    psd = np.nanmean(psd, axis=-1).transpose()
                case "median":
                    psd = np.nanmedian(psd, axis=-1).transpose()
                case _:
                    raise ValueError(
                        f"Invalid statistic {center_stat}. Pick mean or median"
                    )
            psd = np.log10(psd)
            psd = self._calculate_standard_data(psd, mode=mode, axis=-1)
            freq_mask = np.logical_and(
                (freq_range[0] <= freqs), (freqs <= freq_range[1])
            )
            freqs = freqs[freq_mask]
            psd = psd[freq_mask, :]

            extent = (
                0,
                psd.shape[1] * df_row["duration"].median(),
                np.min(freqs),
                np.max(freqs),
            )
            # print(psd.nanmin(), psd.nanmax())
            norm = matplotlib.colors.Normalize()
            # norm = matplotlib.colors.LogNorm()
            # norm = matplotlib.colors.CenteredNorm()

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            # ax.pcolormesh(psd, )
            axim = ax.imshow(
                np.flip(psd, axis=0),
                interpolation="none",
                aspect="auto",
                norm=norm,
                cmap=cmap,
                extent=extent,
            )
            cbar = fig.colorbar(axim, ax=ax)
            cbar.set_label(f"log(PSD) {mode}")

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(i)
            self._handle_figure(fig, title=f"psd_spectrogram_{i}")

    def plot_temporal_heatmap(
        self,
        features: list[PlottableFeatureName] | PlottableFeatureName | None = None,
        figsize=None,
        cmap="viridis",
        score_type=None,
        norm=None,
        component_norms=None,
        **kwargs,
    ):
        """
        Create temporal heatmap showing feature patterns over time.

        Creates a heatmap where:
        - X-axis: Time of day (timestamp mod 24h)
        - Y-axis: Days
        - Color: Feature values (flattened across channels)

        Args:
            features: Feature names to plot. Histogram features are not supported, see
                :data:`neurodent.constants.analysis.FEATURES`. Defaults to the linear
                features.
            figsize (tuple, optional): Figure size (width, height).
            cmap (str, optional): Colormap for the heatmap.
            score_type (str, optional): Standardization applied to feature values. One of
                "z", "zall", "gz", "modz", "center", or "none". Defaults to None.
            norm (matplotlib.colors.Normalize, optional): Normalization for the colormap.
                If None, uses default normalization. Common options are
                ``Normalize(vmin=0, vmax=1)`` for a fixed range, ``CenteredNorm(vcenter=0)``
                to auto-detect a range around 0, and ``LogNorm()`` for a log scale.
            component_norms (dict[str, matplotlib.colors.Normalize], optional): Per-component
                color norms keyed by semantic label (e.g. "slope" / "intercept" for psdslope,
                or band names). A component listed here overrides ``norm`` for that
                component's heatmap; components not listed fall back to ``norm``. Useful when
                a multi-component feature's components live on different scales.
            **kwargs: Additional arguments passed to matplotlib.
        """
        if features is None:
            # Use non-band linear features for temporal analysis
            features = [f for f in constants.LINEAR_FEATURES]
        if isinstance(features, str):
            features = [features]

        # Get data grouped by animalday
        df_rowgroup = self.window_result.get_grouprows_result(
            features,
            multiindex=["animal", "genotype"],
            include=["duration", "endfile", "timestamp", "animalday"],
        )

        missing = [f for f in features if f not in df_rowgroup.columns]
        for f in missing:
            warnings.warn(f"Feature {f} not found in dataframe")
        features = [f for f in features if f not in missing]

        if not features:
            raise ValueError("No valid features found for temporal heatmap")

        # Process each feature
        for feature in features:
            self._plot_temporal_heatmap_feature(
                df_rowgroup=df_rowgroup,
                feature=feature,
                figsize=figsize,
                cmap=cmap,
                score_type=score_type,
                norm=norm,
                component_norms=component_norms,
                **kwargs,
            )

    def _plot_temporal_heatmap_feature(
        self,
        df_rowgroup: pd.DataFrame,
        feature: str,
        n_bins=24 * 60,
        figsize=None,
        cmap="viridis",
        score_type="z",
        norm=None,
        component_norms=None,
        **kwargs,
    ):
        """
        Create temporal heatmap for a single feature.

        ``component_norms`` (dict label->Normalize) overrides ``norm`` per
        semantic component (e.g. {"intercept": Normalize(2, 6)} for psdslope);
        components not listed use ``norm``.
        """
        # Group by animalday to process each recording session
        for animalday, df_day in df_rowgroup.groupby(level=0):
            # Extract timestamps and convert to time of day (modulo 24h)
            timestamps = df_day["timestamp"]
            time_of_day = (
                timestamps.dt.hour
                + timestamps.dt.minute / 60.0
                + timestamps.dt.second / 3600.0
            )

            # Get feature data and flatten across channels
            # Returns shape: (n_time, n_features, n_components) via flatten_feature_for_plotting
            feature_data = self.__get_linear_feature(
                group=df_day, feature=feature, score_type=score_type
            )

            # Classify feature to get its type metadata
            ftype = constants.classify_feature(feature)

            # Collapse the middle axis (channels or channel-pairs) to get (n_time, n_components).
            # Shape transitions by feature type:
            #   LINEAR:         (n_time, n_chan, 1)        → (n_time, 1)
            #   LINEAR_2D:      (n_time, n_chan, 2)        → (n_time, 2)
            #   BAND:           (n_time, n_chan, n_bands)  → (n_time, n_bands)
            #   SIMPLE_MATRIX:  (n_time, n_pairs, 1)      → (n_time, 1)
            #   BANDED_MATRIX:  (n_time, n_pairs, n_bands) → (n_time, n_bands)
            #
            # This averages across channels/channel-pairs but PRESERVES semantic dimensions
            # (components, bands), which are handled separately below.
            feature_data = np.nanmean(feature_data, axis=1)

            # Handle features with semantic dimensions (components, bands, freq_bins).
            # For LINEAR_2D, BAND, BANDED_MATRIX: plot one heatmap per semantic dimension.
            # For HIST: reject since ~100 frequency bins would create too many plots.
            if ftype.semantic_axes:
                if "freq_bins" in ftype.semantic_axes:
                    # HIST features have too many semantic dimensions (~100 freq bins)
                    raise ValueError(
                        f"Feature '{feature}' has {ftype.semantic_axes['freq_bins']} frequency bins "
                        f"which is too many to plot as individual temporal heatmaps. "
                        f"Consider using a different visualization for HIST features."
                    )

                # After averaging over channels, the expected shapes are:
                # LINEAR_2D: (n_time, n_components)
                # BAND: (n_time, n_bands)
                # BANDED_MATRIX: (n_time, n_bands)
                # The semantic axis is at index 1 (or 0 if squeezed to 1D)

                # Handle edge case: if feature_data is 1D, it means there was only 1 semantic dimension
                # (e.g., 1 channel pair for BANDED_MATRIX or data got squeezed)
                if feature_data.ndim == 1:
                    # Only one semantic component - plot it directly
                    semantic_axis_name = list(ftype.semantic_axes.keys())[0]
                    if semantic_axis_name == "components":
                        semantic_label = "component_0"
                    elif semantic_axis_name == "bands":
                        semantic_label = constants.BAND_NAMES[0]
                    else:
                        semantic_label = f"{semantic_axis_name}_0"

                    self._plot_single_temporal_heatmap(
                        timestamps=timestamps,
                        time_of_day=time_of_day,
                        feature_data=feature_data,
                        feature=feature,
                        semantic_label=semantic_label,
                        animalday=animalday,
                        df_day=df_day,
                        score_type=score_type,
                        n_bins=n_bins,
                        figsize=figsize,
                        cmap=cmap,
                        norm=(component_norms or {}).get(semantic_label, norm),
                        **kwargs,
                    )
                else:
                    # Multiple semantic dimensions - iterate over them
                    semantic_axis_idx = 1
                    semantic_axis_name = list(ftype.semantic_axes.keys())[0]
                    n_semantic = feature_data.shape[semantic_axis_idx]

                    # Get semantic dimension labels from constants
                    if semantic_axis_name == "components":
                        # For LINEAR_2D features, use feature-specific component labels
                        semantic_labels = constants.COMPONENT_LABELS.get(
                            feature, [f"component_{i}" for i in range(n_semantic)]
                        )[:n_semantic]
                    elif semantic_axis_name == "bands":
                        # For BAND/BANDED_MATRIX features, use standard band names
                        semantic_labels = constants.BAND_NAMES[:n_semantic]
                    else:
                        # Fallback to generic labels for unknown semantic dimension types
                        semantic_labels = [f"{semantic_axis_name}_{i}" for i in range(n_semantic)]

                    # Plot one heatmap per semantic dimension
                    for i, label in enumerate(semantic_labels):
                        # Extract the i-th semantic component/band
                        feature_data_slice = feature_data.take(i, axis=semantic_axis_idx).squeeze()
                        self._plot_single_temporal_heatmap(
                            timestamps=timestamps,
                            time_of_day=time_of_day,
                            feature_data=feature_data_slice,
                            feature=feature,
                            semantic_label=label,
                            animalday=animalday,
                            df_day=df_day,
                            score_type=score_type,
                            n_bins=n_bins,
                            figsize=figsize,
                            cmap=cmap,
                            norm=(component_norms or {}).get(label, norm),
                            **kwargs,
                        )
            else:
                # No semantic dimensions (LINEAR, SIMPLE_MATRIX): plot single heatmap
                feature_data = feature_data.squeeze()
                self._plot_single_temporal_heatmap(
                    timestamps=timestamps,
                    time_of_day=time_of_day,
                    feature_data=feature_data,
                    feature=feature,
                    semantic_label=None,
                    animalday=animalday,
                    df_day=df_day,
                    score_type=score_type,
                    n_bins=n_bins,
                    figsize=figsize,
                    cmap=cmap,
                    norm=norm,
                    **kwargs,
                )

    def _plot_single_temporal_heatmap(
        self,
        timestamps,
        time_of_day,
        feature_data,
        feature,
        semantic_label,
        animalday,
        df_day,
        score_type,
        n_bins,
        figsize,
        cmap,
        norm,
        **kwargs,
    ):
        """
        Plot a single temporal heatmap for a feature (or a single semantic component/band).

        Args:
            timestamps: Series of timestamps
            time_of_day: Series of time of day values (0-24 hours)
            feature_data: 1D array of feature values (n_time,)
            feature: Feature name
            semantic_label: Label for the semantic dimension (e.g., "slope", "delta"), or None
            animalday: Animalday identifier
            df_day: DataFrame for this animalday (for boundary markers)
            score_type: Score type (e.g., "z", "none") for labeling
            n_bins: Number of time bins
            figsize: Figure size
            cmap: Colormap
            norm: Normalization for colormap
            **kwargs: Additional arguments for imshow
        """
        # Create time bins for the heatmap (24 hours)
        time_bins = np.linspace(0, 24, n_bins + 1)  # 25 edges for 24 bins

        # Create day bins (unique days)
        days = timestamps.dt.date.unique()
        days = sorted(days, reverse=True)

        # Initialize heatmap matrix
        heatmap_matrix = np.full((len(days), n_bins), np.nan)

        # Fill the heatmap matrix
        for i, day in enumerate(days):
            day_mask = timestamps.dt.date == day
            day_times = time_of_day[day_mask]
            day_values = feature_data[day_mask]

            # Bin the data by time of day
            for j, (bin_start, bin_end) in enumerate(
                zip(time_bins[:-1], time_bins[1:])
            ):
                time_mask = (day_times >= bin_start) & (day_times < bin_end)
                if np.any(time_mask):
                    heatmap_matrix[i, j] = np.nanmean(day_values[time_mask])

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize or (10, 3))

        # Create the heatmap
        im = ax.imshow(
            heatmap_matrix,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[0, 24, 0, len(days)],
            origin="lower",
            interpolation="none",
            **kwargs,
        )

        # Add red/white boundary lines between longrecording objects
        self._add_longrecording_boundaries(ax, df_day, time_of_day, days)

        # Add colorbar with full feature name and score type
        cbar = fig.colorbar(im, ax=ax)

        # Get full feature label from constants
        feature_label = constants.FEATURE_LABELS.get(feature, feature)

        # Build colorbar label
        if semantic_label:
            if score_type and score_type != "none":
                cbar.set_label(f"{feature_label} - {semantic_label} (score={score_type})")
            else:
                cbar.set_label(f"{feature_label} - {semantic_label}")
        else:
            if score_type and score_type != "none":
                cbar.set_label(f"{feature_label} (score={score_type})")
            else:
                cbar.set_label(f"{feature_label}")

        # Set labels and title
        ax.set_xlabel("Time of Day (hours)")
        ax.set_ylabel("Day")
        if semantic_label:
            ax.set_title(f"Temporal Heatmap - {feature} ({semantic_label}) - {animalday}")
        else:
            ax.set_title(f"Temporal Heatmap - {feature} - {animalday}")

        # Set x-axis ticks (every 6 hours)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])

        # Set y-axis ticks (every day) - centered in each row
        if len(days) <= 10:
            ax.set_yticks(np.arange(len(days)) + 0.5)
            ax.set_yticklabels([day.strftime("%Y-%m-%d") for day in days])
        else:
            # Show every nth day if too many days
            n = max(1, len(days) // 10)
            ax.set_yticks(np.arange(0, len(days), n) + 0.5)
            ax.set_yticklabels(
                [days[i].strftime("%Y-%m-%d") for i in range(0, len(days), n)]
            )

        # Add grid
        ax.grid(True, alpha=0.3)

        # Handle figure saving/display
        if semantic_label:
            title = f"temporal_heatmap_{feature}_{semantic_label}_{animalday}"
        else:
            title = f"temporal_heatmap_{feature}_{animalday}"
        self._handle_figure(fig, title=title)

    def _add_longrecording_boundaries(self, ax, df_day, time_of_day, days):
        """
        Add red vertical lines to indicate boundaries between longrecording objects
        and plot animalday values on top.

        Args:
            ax: matplotlib axes object
            df_day: dataframe for the current animalday
            time_of_day: array of time of day values (0-24 hours)
            days: sorted list of unique days
        """
        # Check if we have endfile column to identify longrecording boundaries
        if "endfile" not in df_day.columns:
            return

        # Find longrecording boundaries based on endfile markers
        df_day = df_day.reset_index()
        endfile_indices = df_day.index[df_day["endfile"].notna()].tolist()

        if not endfile_indices:
            return

        # For each endfile marker, draw a red line at the corresponding timestamp
        # REVIEW this logic might be faulty because of how timestamps are reported
        for idx in endfile_indices:
            if idx in df_day.index:
                timestamp = df_day.loc[idx, "timestamp"]
                day = timestamp.date()

                # Find which day row this corresponds to
                if day in days:
                    day_idx = days.index(day)
                    time_hour = (
                        timestamp.hour
                        + timestamp.minute / 60.0
                        + timestamp.second / 3600.0
                    )

                    # Draw vertical line at this time point for this day
                    ax.axvline(
                        x=time_hour,
                        ymin=(day_idx) / len(days),
                        ymax=(day_idx + 1) / len(days),
                        color="red",
                        linewidth=1,
                        linestyle="--",
                        alpha=0.8,
                    )

        # Add dotted white lines where animalday value changes
        if "animalday" in df_day.columns:
            df_day_sorted = df_day.sort_values("timestamp")
            prev_animalday = None

            for idx, row in df_day_sorted.iterrows():
                timestamp = row["timestamp"]
                animalday = row["animalday"]
                day = timestamp.date()

                if day in days and pd.notna(animalday):
                    # Check if animalday changed from previous row
                    if prev_animalday is not None and animalday != prev_animalday:
                        day_idx = days.index(day)
                        time_hour = (
                            timestamp.hour
                            + timestamp.minute / 60.0
                            + timestamp.second / 3600.0
                        )

                        # Draw dotted white vertical line at animalday boundary
                        ax.axvline(
                            x=time_hour,
                            ymin=(day_idx) / len(days),
                            ymax=(day_idx + 1) / len(days),
                            color="white",
                            linewidth=2,
                            alpha=0.8,
                        )

                    prev_animalday = animalday

    def _handle_figure(self, fig, title=None):
        if self.save_fig:
            if self.save_path is None:
                raise ValueError("save_path must be provided when save_fig is True")
            # Defensive sanitize: titles often embed animaldays/genotypes that
            # contain characters like '/' (valid display, invalid path).  See
            # `slugify` docstring for the project-wide path-safety convention.
            safe_title = slugify(title) if title else None
            save_name = (
                f"{self.save_path}_{safe_title}.png"
                if safe_title
                else f"{self.save_path}.png"
            )
            fig.savefig(save_name, bbox_inches="tight", dpi=300)
            plt.close(fig)
        else:
            plt.show()
