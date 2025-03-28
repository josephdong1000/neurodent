import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from scipy import stats

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
                case 'rms' | 'ampvar' | 'psdtotal' | 'psdslope' | 'psdband':
                    if feature == 'psdband':
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
                    raise ValueError(f'Feature {feature} is not supported')
                
            df_feature = pd.DataFrame.from_dict(vals, orient='columns')
            dataframes.append(df_feature)
            
        df = pd.concat(dataframes, axis=0, ignore_index=True)

        feature_cols = [col for col in df.columns if col not in groupby]
        df = df.melt(id_vars=groupby, value_vars=feature_cols, var_name='channel', value_name=feature)
        
        if feature == 'psdslope':
            df[feature] = df[feature].apply(lambda x: x[0]) # get slope from [slope, intercept]
        elif feature == 'psdband' or feature == 'cohere':
            df[feature] = df[feature].apply(lambda x: list(zip(x, constants.BAND_NAMES)))
            df = df.explode(feature)
            df[[feature, 'band']] = pd.DataFrame(df[feature].tolist(), index=df.index)
        
        df.reset_index(drop=True, inplace=True)

        return df


    def _process_feature_data(self, feature: str, xgroup: str | list[str], 
                              channels: str | list[str]='all', 
                              remove_outliers: str='iqr', outlier_threshold: float=3):
        """
        Process feature data for plotting.
        
        Parameters
        ----------
        feature : str
            Name of the feature to plot
        xgroup : str
            Column name to group by (e.g., 'animal', 'genotype', etc.)
        channels : str or list
            Channels to include. Can be 'all' or list of channel indices
        remove_outliers : str
            Method to remove outliers: 'iqr', 'zscore', or None
        outlier_threshold : float
            Threshold for outlier removal
        
        Returns
        -------
        dict
            Dictionary containing processed data and metadata for plotting
        """

        # REVIEW this script might be better off with a rewrite
        # Input: feature, xgroup(s)
        # Output (currently): dictionary with info about how to plot
        # Output (todo): parsed/channel-flattened dataframe with single feature + xgroup info. Tidy table
        
        # Get the data
        df = self.all.copy()  # Use the combined DataFrame from __init__
        
        # Handle channel selection
        if channels == 'all':
            max_channels = max(len(chnames) for chnames in self.channel_names) # REVIEW what if non-homogenous channels/names?
            channels = list(range(max_channels))
        elif not isinstance(channels, list):
            channels = [channels]
        
        # Get unique xgroup from the DataFrame
        unique_ids = sorted(df[xgroup].unique())  # Sort to ensure consistent ordering # REVIEW handle list xgroup, with dataframe working and seaborn probably.
        
        # Prepare data for plotting
        plot_data = []
        labels = []
        positions = []
        channel_indices = []  # Track channel indices
        channel_names = []
        
        # Calculate positions for plotting elements
        box_width = 0.8
        group_width = len(channels) * box_width
        group_positions = np.arange(len(unique_ids)) * (group_width + 1)

        # Collect and process data
        for i, id_val in enumerate(unique_ids):
            id_data = df[df[xgroup] == id_val]
            
            for j, ch in enumerate(channels):
                try:
                    # Extract feature data for this channel
                    ch_data = []
                    for _, row in id_data.iterrows():
                        feature_values = row[feature]
                        if hasattr(feature_values, '__len__') and len(feature_values) > ch: # REVIEW should be able to handle extra channels. Also will this trigger ever? 
                            # maybe flatten out and then grab features
                            ch_data.append(feature_values[ch])
                    
                    if not ch_data:
                        print(f"No data collected for channel {ch}")
                        continue
                        
                    ch_data = np.array(ch_data)
                    # Remove any NaN values
                    ch_data = ch_data[~np.isnan(ch_data)]

                    if remove_outliers:
                        clean_data, _ = self._handle_outliers(ch_data, 
                                                            remove_outliers, 
                                                            outlier_threshold)
                    else:
                        clean_data = ch_data

                    if len(clean_data) > 0:
                        plot_data.append(clean_data)
                        channel_indices.append(ch)  # Store the channel index
                        # Get channel name
                        ch_name = f"Ch{ch}"  # Default name
                        for chnames in self.channel_names:
                            if ch < len(chnames):
                                ch_name = chnames[ch].replace("Intan Input (1)/PortA ", "") # REVIEW Feels gimmicky. Have an option to by default pull out the channel names according to constants, otherwise
                                # use the war embedded channel names
                                break
                        channel_names.append(ch_name)
                        labels.append(f"{id_val}\n{ch_name}")
                        positions.append(group_positions[i] + j * box_width)
                    else:
                        print(f"No valid data after processing for channel {ch}")
                        
                except Exception as e:
                    print(f"Warning: Error processing channel {ch} for {id_val}: {e}")
                    continue
        
        if not plot_data:
            raise ValueError("No valid data to plot")
        
        return {
            'plot_data': plot_data,
            'labels': labels,
            'positions': positions,
            'unique_ids': unique_ids,
            'channel_names': list(dict.fromkeys(channel_names)),  # Remove duplicates
            'channel_indices': channel_indices,
            'group_positions': group_positions,
            'box_width': box_width,
            'group_width': group_width
        }
    

    def _handle_outliers(self, data, method, threshold):
        """
        Handle outlier removal based on specified method.
        """
        if method == 'iqr':
            return self._remove_outliers_iqr(data, threshold)
        elif method == 'zscore':
            return self._remove_outliers_zscore(data, threshold)
        else:
            return data, np.array([])

    def _remove_outliers_iqr(self, data, threshold):
        """
        Remove outliers using IQR method.
        """
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        outlier_mask = ((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR)))
        return data[~outlier_mask], data[outlier_mask]

    def _remove_outliers_zscore(self, data, threshold):
        """
        Remove outliers using z-score method.
        """
        z_scores = np.abs((data - np.nanmean(data)) / np.nanstd(data))
        outlier_mask = z_scores >= threshold
        return data[~outlier_mask], data[outlier_mask]

    def _calculate_statistics(self, plot_data, labels, original_data, outliers):
        """
        Calculate statistics for the processed data.
        """
        stats = {}
        for i, data in enumerate(plot_data):
            stats[labels[i]] = {
                'n_total': len(original_data),
                'mean': np.nanmean(data),
                'std': np.nanstd(data),
                'median': np.median(data),
                'n_outliers': len(outliers),
                'outlier_values': outliers.tolist(),
                'outlier_percentage': (len(outliers) / len(original_data) * 100) if len(original_data) > 0 else 0
            }
        return stats

    def _setup_plot(self, figsize=None, color_palette=None, num_channels=None):
        """
        Set up the plot with common elements.
        """
        fig, ax = plt.subplots(figsize=figsize)
        colors = (color_palette if color_palette is not None 
                else plt.cm.tab10(np.linspace(0, 1, num_channels)))
        return fig, ax, colors

    def _customize_plot(self, ax, data_dict, colors, title, feature, xgroup):
        """
        Apply common plot customizations.
        """
        # Set labels and title
        ax.set_xlabel(xgroup)
        ax.set_ylabel(feature)
        if title:
            ax.set_title(title, pad=20)
            
        # Set x-ticks
        ax.set_xticks(data_dict['group_positions'] + 
                     (data_dict['group_width'] - data_dict['box_width'])/2)
        ax.set_xticklabels(data_dict['unique_ids'], rotation=45, ha='right')
        
        # Create and add legend
        legend_elements = [plt.Line2D([0], [0], color=colors[i], 
                                    marker='s', linestyle='None',
                                    markersize=10, label=ch_name) 
                        for i, ch_name in enumerate(data_dict['channel_names'])]
        ax.legend(handles=legend_elements, title='Channels', 
                bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid and adjust spines
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()

    def _color_boxplot(self, bp, colors, data_dict, show_outliers):
        """
        Modified to use channel indices for color assignment
        """
        num_boxes = len(bp['boxes'])
        for i in range(num_boxes):
            channel_idx = data_dict['channel_indices'][i]  # Use the actual channel index
            
            # Set box colors
            bp['boxes'][i].set_facecolor(colors[channel_idx])
            bp['boxes'][i].set_alpha(0.7)
            bp['boxes'][i].set_edgecolor(colors[channel_idx])
            
            # Set median line colors
            bp['medians'][i].set_color(colors[channel_idx])
            
            # Set whisker colors
            bp['whiskers'][i*2].set_color(colors[channel_idx])
            bp['whiskers'][i*2+1].set_color(colors[channel_idx])
            
            # Set cap colors
            bp['caps'][i*2].set_color(colors[channel_idx])
            bp['caps'][i*2+1].set_color(colors[channel_idx])
            
            # Set flier colors
            if show_outliers and len(bp['fliers']) > i:
                bp['fliers'][i].set_markerfacecolor(colors[channel_idx])
                bp['fliers'][i].set_markeredgecolor(colors[channel_idx])


    def plot_catplot(self, feature: str, 
                    groupby: str | list[str], 
                    x: str=None,
                    col: str=None,
                    hue: str=None,
                    kind: str='box',
                    catplot_params: dict=None,
                    channels: str | list[str]='all', 
                    collapse_channels: bool=False,
                    title: str=None, color_palette: list[str]=None, figsize: tuple[float, float]=None):
        """
        Create a boxplot of the feature data.
        """
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
            'palette': color_palette,
            # 'showfliers': show_outliers,
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

        # Create boxplot using seaborn
        g = sns.catplot(**default_params)
        g.set_xticklabels(rotation=45, ha='right')
        g.set_titles(title)
        g.legend.set_loc('center left')
        g.legend.set_bbox_to_anchor((1.0, 0.5))
        # Add grid to y-axis for all subplots
        for ax in g.axes.flat:
            ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        
        plt.tight_layout()
        
        return g

    def plot_boxplot(self, feature, xgroup='animal', channels='all', 
                remove_outliers='iqr', outlier_threshold=3, show_outliers=True,
                title=None, color_palette=None, figsize=None):
            """
            [Previous docstring]
            """
            data_dict = self._process_feature_data(
                feature, xgroup, channels, remove_outliers, outlier_threshold
            )
            
            fig, ax, colors = self._setup_plot(figsize, color_palette, 
                                            max(data_dict['channel_indices']) + 1)  # Use max channel index + 1
            
            # Create boxplot
            bp = ax.boxplot(data_dict['plot_data'], positions=data_dict['positions'],
                        widths=0.7*data_dict['box_width'], showfliers=show_outliers,
                        patch_artist=True)
            
            # Color the boxes
            self._color_boxplot(bp, colors, data_dict, show_outliers)
            
            # Customize plot
            self._customize_plot(ax, data_dict, colors, title, feature, xgroup)
            
            return fig, ax, self.stats

    def plot_violin(self, feature, xgroup='animal', channels='all',
                remove_outliers='iqr', outlier_threshold=3, title=None, 
                color_palette=None, figsize=None):
        """
        Create a violin plot of the feature data.
        """
        data_dict = self._process_feature_data(
            feature, xgroup, channels, remove_outliers, outlier_threshold
        )
        
        fig, ax, colors = self._setup_plot(figsize, color_palette, 
                                        len(data_dict['channel_names']))
        
        # Create violin plot
        vp = ax.violinplot(data_dict['plot_data'], positions=data_dict['positions'])
        
        # Customize plot
        self._customize_plot(ax, data_dict, colors, title, feature, xgroup)
        
        return fig, ax, self.stats

    def plot_scatter(self, feature, xgroup='animal', channels='all',
                    remove_outliers='iqr', outlier_threshold=3, title=None, 
                    color_palette=None, figsize=None):
        """
        Create a scatter plot of the feature data.
        """
        data_dict = self._process_feature_data(
            feature, xgroup, channels, remove_outliers, outlier_threshold
        )
        
        fig, ax, colors = self._setup_plot(figsize, color_palette, 
                                        len(data_dict['channel_names']))
        
        # Create scatter plot
        for i, (data, pos) in enumerate(zip(data_dict['plot_data'], 
                                        data_dict['positions'])):
            ax.scatter([pos] * len(data), data, 
                    color=colors[i % len(colors)], alpha=0.5)
        
        # Customize plot
        self._customize_plot(ax, data_dict, colors, title, feature, xgroup)
        
        return fig, ax, self.stats
    
    def plot_2d_feature_2(self,
                        feature: str, 
                        groupby: str | list[str], 
                        col: str=None,
                        row: str=None,
                        channels: str | list[str]='all', 
                        collapse_channels: bool=False,
                        title: str=None, color_palette: list[str]=None, figsize: tuple[float, float]=None):
        
        df = self._pull_timeseries_dataframe(feature, groupby, channels, collapse_channels)

        if isinstance(groupby, str):
            groupby = [groupby]
        # REVIEW review this code, edit X/Y tick labels, enable other colormaps
        # Define the plotting function for each facet
        def plot_matrix(data, color_palette=None, **kwargs):
            matrices = np.array(data[feature].tolist())
            avg_matrix = np.nanmean(matrices, axis=0)
            
            # Create heatmap
            plt.imshow(avg_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(fraction=0.046, pad=0.04)
            
            # Set ticks and labels
            n_channels = avg_matrix.shape[0]
            plt.xticks(range(n_channels), range(n_channels), rotation=45, ha='right')
            plt.yticks(range(n_channels), range(n_channels))
            
            # Remove axis labels since they're redundant for correlation matrices
            ch_names = self.channel_names[0]
            plt.xticks(range(n_channels), ch_names, rotation=45, ha='right')
            plt.yticks(range(n_channels), ch_names)
        
        # Create FacetGrid
        facet_vars = {
            'col': groupby[0],
            'row': groupby[1] if len(groupby) > 1 else None
        }
        if col is not None:
            facet_vars['col'] = col
        if row is not None:
            facet_vars['row'] = row
        
        g = sns.FacetGrid(df, **facet_vars, height=4 if figsize is None else figsize[0]/2)
        
        # Map the plotting function
        g.map_dataframe(plot_matrix, color_palette=color_palette)
        
        # Add titles
        if title:
            g.figure.suptitle(title, y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        
        return g


    def plot_2d_feature(self, feature, xgroup='animal', channels='all', 
                       remove_outliers='iqr', outlier_threshold=3,
                       title=None, color_palette=None, figsize=None):
        """
        Create 2D plots for features that return matrices (e.g., coherence, correlation).
        
        Parameters
        ----------
        feature : str
            Name of the feature to plot (e.g., 'cohere', 'pcorr')
        xgroup : str
            Column name to group by (e.g., 'animal', 'genotype')
        channels : str or list
            Channels to include. Can be 'all' or list of channel indices
        remove_outliers : str
            Method to remove outliers: 'iqr', 'zscore', or None
        outlier_threshold : float
            Threshold for outlier removal
        title : str, optional
            Plot title
        color_palette : list, optional
            Custom color palette for the plots
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns
        -------
        tuple
            (figure, axes, stats)
        """
        # Get the data
        df = self.all.copy()
        
        # Handle channel selection
        if channels == 'all':
            max_channels = max(len(chnames) for chnames in self.channel_names)
            channels = list(range(max_channels))
        elif not isinstance(channels, list):
            channels = [channels]
            
        # Get unique xgroup values
        unique_ids = sorted(df[xgroup].unique())
        n_groups = len(unique_ids)
        
        # Create figure and axes
        if figsize is None:
            figsize = (4 * n_groups, 4)
        fig, axes = plt.subplots(1, n_groups, figsize=figsize)
        if n_groups == 1:
            axes = [axes]
            
        # Process each group
        for idx, (ax, group_id) in enumerate(zip(axes, unique_ids)):
            group_data = df[df[xgroup] == group_id]
            
            # Average the matrices in the group
            matrices = []
            for _, row in group_data.iterrows():
                matrix = row[feature]
                if isinstance(matrix, list): # FIXME hotfix, refactor this code with 2d_feat_freq perhaps
                    matrix = np.array(matrix)
                if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                    # Ensure matrix is square and matches channel dimensions
                    if matrix.shape[0] == len(channels) and matrix.shape[1] == len(channels):
                        matrices.append(matrix)

            if matrices:
                avg_matrix = np.nanmean(matrices, axis=0)
                # Create heatmap
                im = ax.imshow(avg_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                
                # Add colorbar for each subplot
                plt.colorbar(im, ax=ax, label=feature if idx == n_groups-1 else '', 
                           fraction=0.046, pad=0.04)
                
                # Customize appearance
                ax.set_title(f"{group_id}")
                ax.set_xticks(range(len(channels)))
                ax.set_yticks(range(len(channels)))
                
                # Use channel names if available
                channel_labels = []
                for ch in channels:
                    ch_name = f"Ch{ch}"
                    for chnames in self.channel_names:
                        if ch < len(chnames):
                            ch_name = chnames[ch].replace("Intan Input (1)/PortA ", "")
                            break
                    channel_labels.append(ch_name)
                
                ax.set_xticklabels(channel_labels, rotation=45, ha='right')
                ax.set_yticklabels(channel_labels)
            else:
                ax.text(0.5, 0.5, 'No valid data', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Set main title if provided
        if title:
            fig.suptitle(title, y=1.05)
            
        plt.tight_layout()
        
        return fig, axes, None
    

    def plot_2d_feature_freq(self, feature, xgroup='animal', channels='all', 
                       remove_outliers='iqr', outlier_threshold=3,
                       title=None, color_palette=None, figsize=None):
        """
        Create 2D plots for features that return matrices (e.g., coherence, correlation).
        Handles both simple 2D matrices and frequency-band specific matrices.
        
        Parameters
        ----------
        feature : str
            Name of the feature to plot (e.g., 'cohere', 'pcorr')
        xgroup : str
            Column name to group by (e.g., 'animal', 'genotype')
        channels : str or list
            Channels to include. Can be 'all' or list of channel indices
        remove_outliers : str
            Method to remove outliers: 'iqr', 'zscore', or None
        outlier_threshold : float
            Threshold for outlier removal
        title : str, optional
            Plot title
        color_palette : list, optional
            Custom color palette for the plots
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns
        -------
        tuple
            (figure, axes, stats)
        """
        # Get the data
        df = self.all.copy()
        
        # Handle channel selection
        if channels == 'all':
            max_channels = max(len(chnames) for chnames in self.channel_names)
            channels = list(range(max_channels))
        elif not isinstance(channels, list):
            channels = [channels]
            
        # Get unique xgroup values
        unique_ids = sorted(df[xgroup].unique())
        n_groups = len(unique_ids)
        
        # Check if the feature contains frequency bands
        sample_data = df.iloc[0][feature]
        has_freq_bands = isinstance(sample_data, dict)
        
        if has_freq_bands:
            freq_bands = list(sample_data.keys())
            n_bands = len(freq_bands)
            
            # Create figure and axes grid
            if figsize is None:
                figsize = (4 * n_groups, 3 * n_bands)
            fig, axes = plt.subplots(n_bands, n_groups, figsize=figsize)
            if n_groups == 1:
                axes = axes.reshape(-1, 1)
            if n_bands == 1:
                axes = axes.reshape(1, -1)
            
            # Process each frequency band and group
            for band_idx, band in enumerate(freq_bands):
                for group_idx, group_id in enumerate(unique_ids):
                    ax = axes[band_idx, group_idx]
                    group_data = df[df[xgroup] == group_id]
                    
                    # Average the matrices in the group
                    matrices = []
                    for _, row in group_data.iterrows():
                        matrix = row[feature][band]
                        if isinstance(matrix, list): # FIXME hotfix, refactor this code with 2d_feat_freq perhaps
                            matrix = np.array(matrix)
                        if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                            if matrix.shape[0] == len(channels) and matrix.shape[1] == len(channels):
                                matrices.append(matrix)
                    
                    if matrices:
                        avg_matrix = np.nanmean(matrices, axis=0)
                        
                        # Create heatmap
                        im = ax.imshow(avg_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                        
                        # Add colorbar
                        if group_idx == n_groups - 1:  # Only for last column
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        
                        # Customize appearance
                        if band_idx == 0:  # Only for first row
                            ax.set_title(f"{group_id}")
                        if group_idx == 0:  # Only for first column
                            ax.set_ylabel(band)
                        
                        # Set ticks and labels
                        ax.set_xticks(range(len(channels)))
                        ax.set_yticks(range(len(channels)))
                        
                        # Use channel names if available
                        channel_labels = []
                        for ch in channels:
                            ch_name = f"Ch{ch}"
                            for chnames in self.channel_names:
                                if ch < len(chnames):
                                    ch_name = chnames[ch].replace("Intan Input (1)/PortA ", "")
                                    break
                            channel_labels.append(ch_name)
                        
                        if band_idx == n_bands - 1:  # Only for last row
                            ax.set_xticklabels(channel_labels, rotation=45, ha='right')
                        else:
                            ax.set_xticklabels([])
                        
                        if group_idx == 0:  # Only for first column
                            ax.set_yticklabels(channel_labels)
                        else:
                            ax.set_yticklabels([])
                    else:
                        ax.text(0.5, 0.5, 'No valid data', 
                               ha='center', va='center', transform=ax.transAxes)
        
        else:
            # Original implementation for non-frequency band data
            if figsize is None:
                figsize = (4 * n_groups, 4)
            fig, axes = plt.subplots(1, n_groups, figsize=figsize)
            if n_groups == 1:
                axes = [axes]
                
            # Process each group
            for idx, (ax, group_id) in enumerate(zip(axes, unique_ids)):
                group_data = df[df[xgroup] == group_id]
                
                # Average the matrices in the group
                matrices = []
                for _, row in group_data.iterrows():
                    matrix = row[feature]
                    if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                        if matrix.shape[0] == len(channels) and matrix.shape[1] == len(channels):
                            matrices.append(matrix)
                
                if matrices:
                    avg_matrix = np.nanmean(matrices, axis=0)
                    
                    # Create heatmap
                    im = ax.imshow(avg_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, label=feature if idx == n_groups-1 else '', 
                               fraction=0.046, pad=0.04)
                    
                    # Customize appearance
                    ax.set_title(f"{group_id}")
                    ax.set_xticks(range(len(channels)))
                    ax.set_yticks(range(len(channels)))
                    
                    # Use channel names if available
                    channel_labels = []
                    for ch in channels:
                        ch_name = f"Ch{ch}"
                        for chnames in self.channel_names:
                            if ch < len(chnames):
                                ch_name = chnames[ch].replace("Intan Input (1)/PortA ", "")
                                break
                        channel_labels.append(ch_name)
                    
                    ax.set_xticklabels(channel_labels, rotation=45, ha='right')
                    ax.set_yticklabels(channel_labels)
                else:
                    ax.text(0.5, 0.5, 'No valid data', 
                           ha='center', va='center', transform=ax.transAxes)
        
        # Set main title if provided
        if title:
            fig.suptitle(title, y=1.05)
            
        plt.tight_layout()
        
        return fig, axes, None
