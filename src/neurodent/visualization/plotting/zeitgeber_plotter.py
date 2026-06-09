"""
Zeitgeber Plotter
=================

Contains logic for generating Zeitgeber temporal plots.
Refactored into a class structure to match AnimalPlotter/ExperimentPlotter patterns.
"""

import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
import seaborn as sns
from seaborn import axes_style

from neurodent import constants
from neurodent.core.utils import get_feature_label

logger = logging.getLogger(__name__)


class ZeitgeberPlotter:
    """
    Class for generating, styling, and saving Zeitgeber temporal plots.
    
    Can be initialized with either:
    - A list of ZeitgeberAnalysisResult objects (ZARs)
    - A pre-aggregated pandas DataFrame
    
    Args:
        data: Either a list of ZARs or a DataFrame containing zeitgeber-processed data.
        features: List of features to extract (only used if data is list of ZARs).
        aggregate_config: Optional config for aggregation (only used if data is list of ZARs).
    
    Examples:
        # From ZARs (interactive use)
        zars = [ZeitgeberAnalysisResult(war1, **config), ...]
        plotter = ZeitgeberPlotter(zars, features=["logpsdband", "logrms"])
        
        # From DataFrame (workflow use)
        df = pd.read_pickle("zeitgeber_features.pkl")
        plotter = ZeitgeberPlotter(df)
    """

    def __init__(
        self, 
        data,
        features: list[str] = None,
        aggregate_config: dict = None
    ):
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, list):
            # Assume list of ZARs
            self.df = self._aggregate_zars(data, features, aggregate_config)
        else:
            raise ValueError("data must be a DataFrame or list of ZeitgeberAnalysisResult objects")

    def _aggregate_zars(self, zars, features, config):
        """
        Aggregate multiple ZARs into a single plotting-ready DataFrame.

        Mirrors logic from extract_zeitgeber_features.py.

        Args:
            zars: List of ZeitgeberAnalysisResult objects.
            features: List of features to extract.
            config: Optional config dict (currently unused, for future expansion).

        Returns:
            pd.DataFrame: Aggregated 24h DataFrame ready for plotting.
                Multi-day expansion (e.g. 48h) is deferred to
                :meth:`plot_feature` so the same internal df can serve any
                requested ``n_days`` view without leftover duplicates.
        """
        if not zars:
            raise ValueError("zars list cannot be empty")

        dfs = []
        for zar in zars:
            df = zar.get_channel_averaged_result(features=features)
            df["animal"] = zar.animal_id  # Delegated via __getattr__
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # Aggregate by time bins
        group_cols = ["animal", "genotype", "sex", "gene", "zt_minutes"]
        group_cols = [c for c in group_cols if c in df.columns]

        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in group_cols]
        agg_dict = {f: "mean" for f in feature_cols}

        if group_cols and agg_dict:
            df = df.groupby(group_cols).agg(agg_dict).reset_index()

        return df

    def plot_feature(self, feature, output_path, figsize, dpi=300, n_days=2):
        """
        Create and save a zeitgeber plot for a single feature.

        Args:
            feature (str): Feature column name.
            output_path (Path): Path to save the figure.
            figsize (list): Figure size [width, height].
            dpi (int): DPI for the saved figure.
            n_days (int): Number of ZT cycles to span on the x-axis.
                Defaults to 2 (the legacy 48-hour view).  The plotter
                materialises the multi-day expansion via
                :func:`expand_zt_axis` at render time so the persisted
                24h dataframe is never duplicated on disk.
        """
        label = get_feature_label(feature)

        # Lazy import to avoid circular dependency between core.zeitgeber
        # and visualization.plotting.
        from neurodent.core.zeitgeber import expand_zt_axis
        plot_df = expand_zt_axis(self.df, n_days=n_days)

        try:
            p = (
                so.Plot(plot_df, x="zt_minutes", y=feature, color="gene")
                .facet(col="sex", row="gene")
                .add(so.Line(linewidth=2), so.Agg())
                .add(so.Dot(), so.Agg())
                .add(so.Band(), so.Est())
                .layout(size=(1, 1))
                .theme(axes_style("ticks") | sns.plotting_context("poster"))
                .label(y=label)
            )
            
            fig = mpl.figure.Figure(figsize=figsize)
            p.on(fig).plot()
            
            # Add ZT formatting and day/night shading
            for ax in fig.axes:
                # Shade night periods (12-24h and 36-48h)
                ax.axvspan(xmin=12 * 60, xmax=24 * 60, alpha=0.1, color='grey')
                ax.axvspan(xmin=36 * 60, xmax=48 * 60, alpha=0.1, color='grey')
                
                # Set ticks every 6 hours
                ax.set_xticks(np.arange(0, 49 * 60, 6 * 60))
                new_labels = [(x/60) % 24 for x in ax.get_xticks()]
                ax.set_xticklabels([f"{x:.0f}" for x in new_labels])
                ax.set_xlabel("ZT")
            
            fig.tight_layout()
            fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            
            logger.info(f"Successfully created zeitgeber plot for {feature}")
            
        except Exception as e:
            logger.error(f"Failed to create zeitgeber plot for {feature}: {str(e)}")
            raise
