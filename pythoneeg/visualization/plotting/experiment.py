import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from typing import Literal

from scipy import stats
from statannotations.Annotator import Annotator

from ... import core
from ... import visualization as viz
from pythoneeg import constants

class ExperimentPlotter():

    """
    A class for creating various plots from a list of multiple experimental datasets.
    
    This class provides methods for creating different types of plots (boxplot, violin plot,
    scatter plot, etc.) from experimental data with consistent data processing and styling.
    """


    def __init__(self, wars: viz.WindowAnalysisResult | list[viz.WindowAnalysisResult], 
                 features: list[str] = None,
                 exclude: list[str] = None,
                 use_abbreviations: bool = True):
        """
        Initialize plotter with WindowAnalysisResult object(s).
        
        Parameters
        ----------
        wars : WindowAnalysisResult or list[WindowAnalysisResult]
            Single WindowAnalysisResult or list of WindowAnalysisResult objects
        features : list[str], optional
            List of features to extract. If None, defaults to ['all']
        exclude : list[str], optional
            List of features to exclude from extraction
        use_abbreviations : bool, optional
            Whether to use abbreviations for channel names
        """
        # Handle default arguments
        features = features if features else ['all']
        
        # Convert single WindowAnalysisResult to list
        if not isinstance(wars, list):
            wars = [wars]
            
        self.results = wars
        if use_abbreviations:
            self.channel_names = [war.channel_abbrevs for war in wars]
        else:
            self.channel_names = [war.channel_names for war in wars]
        self.channel_to_idx = [{e:i for i,e in enumerate(chnames)} for chnames in self.channel_names]
        self.all_channel_names = sorted(list(set([name for chnames in self.channel_names for name in chnames])))
        logging.info(f'channel_names: {self.channel_names}')
        logging.info(f'channel_to_idx: {self.channel_to_idx}')
        logging.info(f'all_channel_names: {self.all_channel_names}')
        
        # Process all data into DataFrames
        df_wars = []
        for war in wars:
            # Get all data
            dftemp = war.get_result(features=features, exclude=exclude, allow_missing=True)
            df_wars.append(dftemp)

        self.df_wars: list[pd.DataFrame] = df_wars
        self.all = pd.concat(df_wars, axis=0, ignore_index=True)
        self.stats = None


    def _pull_timeseries_dataframe(self, feature:str, groupby:str | list[str], 
                            channels:str|list[str]='all', 
                            collapse_channels: bool=False):
        """
        Process feature data for plotting.
        """

        if 'band' in groupby or groupby == 'band':
            raise ValueError("'band' is not supported as a groupby variable. Use 'band' as a col/row/hue/x variable instead.") # REVIEW

        if channels == 'all':
            channels = self.all_channel_names
        elif not isinstance(channels, list):
            channels = [channels]
        df = self.all.copy()  # Use the combined DataFrame from __init__
        
        if isinstance(groupby, str):
            groupby = [groupby]
        groups = list(df.groupby(groupby).groups.keys())
        logging.debug(f'groups: {groups}')
        
        # first iterate through animals, since that determines channel idx
        # then pull out feature into matrix, assign channels, and add to dataframe 
        # meanwhile carrying over the groupby feature

        dataframes = []
        for i, war in enumerate(self.results):
            df_war = self.df_wars[i]
            ch_to_idx = self.channel_to_idx[i]
            ch_names = self.channel_names[i]
            match feature:
                case 'rms' | 'ampvar' | 'psdtotal' | 'psdslope' | 'psdband' | 'psdfrac':
                    if feature in constants.BAND_FEATURE:
                        df_bands = pd.DataFrame(df_war[feature].tolist())
                        vals = np.array(df_bands.values.tolist())
                        vals = vals.transpose((0, 2, 1))
                    else:
                        vals = np.array(df_war[feature].tolist())

                    if collapse_channels:
                        vals = np.nanmean(vals, axis=1)
                        logging.debug(f'vals.shape: {vals.shape}')
                        vals = {'average': vals.tolist()}
                    else:
                        logging.debug(f'vals.shape: {vals.shape}')
                        vals = {ch: vals[:, ch_to_idx[ch]].tolist() for ch in channels if ch in ch_names}
                    vals = df_war[groupby].to_dict('list') | vals

                case 'pcorr':
                    vals = np.array(df_war[feature].tolist())
                    if collapse_channels:
                        # Get lower triangular elements (excluding diagonal)
                        tril_indices = np.tril_indices(vals.shape[1], k=-1)
                        # Take mean across pairs
                        vals = np.nanmean(vals[:, tril_indices[0], tril_indices[1]], axis=-1)
                        logging.debug(f'vals.shape: {vals.shape}')
                        vals = {'average': vals.tolist()}
                    else:
                        logging.debug(f'vals.shape: {vals.shape}')
                        vals = {'all' : vals.tolist()}
                    vals = df_war[groupby].to_dict('list') | vals

                case 'cohere':
                    df_bands = pd.DataFrame(df_war[feature].tolist())
                    vals = np.array(df_bands.values.tolist())
                    logging.debug(f'vals.shape: {vals.shape}')

                    if collapse_channels:
                        tril_indices = np.tril_indices(vals.shape[1], k=-1)
                        vals = np.nanmean(vals[:, :, tril_indices[0], tril_indices[1]], axis=-1)
                        logging.debug(f'vals.shape: {vals.shape}')
                        vals = {'average': vals.tolist()}
                    else:
                        logging.debug(f'vals.shape: {vals.shape}')
                        vals = {'all' : vals.tolist()}
                    vals = df_war[groupby].to_dict('list') | vals
                    
                case _:
                    raise ValueError(f'{feature} is not supported in _pull_timeseries_dataframe')
                
            df_feature = pd.DataFrame.from_dict(vals, orient='columns')
            dataframes.append(df_feature)
            
        df = pd.concat(dataframes, axis=0, ignore_index=True)

        feature_cols = [col for col in df.columns if col not in groupby]
        df = df.melt(id_vars=groupby, value_vars=feature_cols, var_name='channel', value_name=feature)
        
        if feature == 'psdslope':
            df[feature] = df[feature].apply(lambda x: x[0]) # get slope from [slope, intercept]
        elif feature in constants.BAND_FEATURE + ['cohere']:
            df[feature] = df[feature].apply(lambda x: list(zip(x, constants.BAND_NAMES)))
            df = df.explode(feature)
            df[[feature, 'band']] = pd.DataFrame(df[feature].tolist(), index=df.index)
        
        df.reset_index(drop=True, inplace=True)

        return df


    def plot_catplot(self, feature: str, 
                    groupby: str | list[str], 
                    x: str=None,
                    col: str=None,
                    hue: str=None,
                    kind: Literal['box', 'boxen', 'violin']='box',
                    catplot_params: dict=None,
                    channels: str | list[str]='all', 
                    collapse_channels: bool=False,
                    title: str=None, 
                    cmap: str=None, 
                    stat_pairs: list[tuple[str, str]]=None,
                    ) -> sns.FacetGrid:
        """
        Create a boxplot of feature data.
        """
        if feature in constants.MATRIX_FEATURE and not collapse_channels:
            raise ValueError("To plot matrix features, collapse_channels must be True")

        df = self._pull_timeseries_dataframe(feature, groupby, channels, collapse_channels)
        
        if isinstance(groupby, str):
            groupby = [groupby]
        
        # By default, just map x = groupby0, col = groupby1, hue = channel
        default_params = {
            'data': df,
            'x': groupby[0],
            'y': feature,
            'hue': 'channel',
            'col': groupby[1] if len(groupby) > 1 else None,
            'kind': kind,
            'palette': cmap,
        }
        # Update default params if x, col, or hue are explicitly provided
        if x is not None:
            default_params['x'] = x
        if col is not None:
            default_params['col'] = col
        if hue is not None:
            default_params['hue'] = hue
        if catplot_params:
            default_params.update(catplot_params)

        # Check that x, col, and hue parameters exist in dataframe columns
        for param_name in ['x', 'col', 'hue']:
            if default_params[param_name] == feature:
                raise ValueError(f"'{param_name}' cannot be the same as 'feature'")
            if default_params[param_name] is not None and default_params[param_name] not in df.columns:
                raise ValueError(f"Parameter '{param_name}={default_params[param_name]}' not found in dataframe columns: {df.columns.tolist()}")

        # Create boxplot using seaborn
        g = sns.catplot(**default_params)

        # TODO update plot aesthetics with more pretty outline/fill

        g.set_xticklabels(rotation=45, ha='right')
        g.set_titles(title)
        
        # Only try to modify legend if it exists
        if g.legend is not None:
            g.legend.set_loc('center left')
            g.legend.set_bbox_to_anchor((1.0, 0.5))
            
        # Add grid to y-axis for all subplots
        for ax in g.axes.flat:
            ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        
        if stat_pairs:
            # annot = Annotator(None, stat_pairs)
            # plot_params = default_params.copy()
            # plot_params.pop('data')
            # plot_params.pop('hue')
            # plot_params.pop('col')
            # plot_params.pop('kind')
            # logging.debug(f'plot params: {plot_params}')
            # g.map_dataframe(annot.plot_and_annotate_facets, 
            #                 plot=f'{kind}plot', # REVIEW this might not work for all plot types
            #                 plot_params=plot_params,
            #                 configuration=dict(test='t-test_ind', text_format='star', loc='inside'),
            #                 annotation_func='apply_test')
            for ax in g.axes.flat:
                annot_params = default_params.copy()
                # annot_params['data'] = annot_params['data'].dropna()
                annotator = Annotator(ax, stat_pairs, **annot_params)
                # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
                annotator.configure(test='Brunner-Munzel', text_format='star', loc='inside')
                # annotator.apply_test('Brunner-Munzel', nan_policy='omit')
                annotator.apply_test(nan_policy='raise')
                annotator.apply_and_annotate()

        plt.tight_layout()
        
        return g


    def plot_matrixplot(self,
                        feature: str, 
                        groupby: str | list[str], 
                        col: str=None,
                        row: str=None,
                        channels: str | list[str]='all', 
                        collapse_channels: bool=False,
                        title: str=None, 
                        cmap: str='RdBu_r', 
                        figsize: tuple[float, float]=None):
        """
        Create a 2D feature plot.
        """
        if feature not in constants.MATRIX_FEATURE:
            raise ValueError(f'{feature} is not supported for 2D feature plots') # REVIEW

        if isinstance(groupby, str):
            groupby = [groupby]  
        if feature != 'cohere' and ('band' in groupby or 'band' == col or 'band' == row):
            raise ValueError(f'{feature} does not support band plots') # REVIEW
        
        df = self._pull_timeseries_dataframe(feature, groupby, channels, collapse_channels)
        
        # Create FacetGrid
        facet_vars = {
            'col': groupby[0],
            'row': groupby[1] if len(groupby) > 1 else None,
            'height': 4 if figsize is None else figsize[1],
            'aspect': 1 if figsize is None else figsize[0]/figsize[1]
        }
        if col is not None:
            facet_vars['col'] = col
        if row is not None:
            facet_vars['row'] = row
        
        # Check that x, col, and hue parameters exist in dataframe columns
        for param_name in ['col', 'row']:
            if facet_vars[param_name] == feature:
                raise ValueError(f"'{param_name}' cannot be the same as 'feature'")
            if facet_vars[param_name] is not None and facet_vars[param_name] not in df.columns:
                raise ValueError(f"Parameter '{param_name}={facet_vars[param_name]}' not found in dataframe columns: {df.columns.tolist()}")

        g = sns.FacetGrid(df, **facet_vars)

        # Map the plotting function
        g.map_dataframe(self._plot_matrix, feature=feature, color_palette=cmap)
        
        # Add titles
        if title:
            g.figure.suptitle(title, y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        
        return g

    def _plot_matrix(self, data, feature, color_palette='RdBu_r', **kwargs):
        matrices = np.array(data[feature].tolist())
        avg_matrix = np.nanmean(matrices, axis=0)
        
        # Create heatmap
        plt.imshow(avg_matrix,
                   cmap=color_palette, 
                   vmin=-1, vmax=1)
        plt.colorbar(fraction=0.046, pad=0.04)
        
        # Set ticks and labels
        n_channels = avg_matrix.shape[0]
        plt.xticks(range(n_channels), range(n_channels), rotation=45, ha='right')
        plt.yticks(range(n_channels), range(n_channels))
        
        ch_names = self.channel_names[0] # REVIEW what if the channel names are not the same for all animals?
        plt.xticks(range(n_channels), ch_names, rotation=45, ha='right')
        plt.yticks(range(n_channels), ch_names)