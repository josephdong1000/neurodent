
import pandas as pd

from ... import visualization as viz

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pythoneeg import core



class ExperimentPlotter():

    """
    A class for creating various plots from a list of multiple experimental datasets.
    
    This class provides methods for creating different types of plots (boxplot, violin plot,
    scatter plot, etc.) from experimental data with consistent data processing and styling.
    """


    def __init__(self, wars, features: list[str] = None,
                 exclude: list[str] = None):
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
        """
        # Handle default arguments
        features = features if features else ['all']
        
        # Convert single WindowAnalysisResult to list
        if not isinstance(wars, list):
            wars = [wars]
            
        self.results = wars
        self.channel_names = [war.channel_names for war in wars]
        
        # Process all data into DataFrames
        df_all = []
        for war in wars:
            # Get all data
            dftemp = war.get_result(features=features, exclude=exclude, allow_missing=True)
            df_all.append(dftemp)

        self.all = pd.concat(df_all, axis=0, ignore_index=True)
        self.stats = None


    def _process_feature_data(self, feature, xgroup, channels='all', 
                        remove_outliers='iqr', outlier_threshold=3):
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
        # Get the data
        df = self.all.copy()  # Use the combined DataFrame from __init__
        
        # Handle channel selection
        if channels == 'all':
            max_channels = max(len(chnames) for chnames in self.channel_names)
            channels = list(range(max_channels))
        elif not isinstance(channels, list):
            channels = [channels]
        
        # Get unique xgroup from the DataFrame
        unique_ids = sorted(df[xgroup].unique())  # Sort to ensure consistent ordering
        
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
                        if hasattr(feature_values, '__len__') and len(feature_values) > ch:
                            ch_data.append(feature_values[ch])
                    
                    if not ch_data:
                        print(f"No data collected for channel {ch}")
                        continue
                        
                    ch_data = np.array(ch_data)
                    # Remove any NaN values
                    ch_data = ch_data[~np.isnan(ch_data)]
                    original_data = ch_data.copy()

                    if remove_outliers:
                        clean_data, outliers = self._handle_outliers(ch_data, 
                                                                remove_outliers, 
                                                                outlier_threshold)
                    else:
                        clean_data = ch_data
                        outliers = np.array([])

                    if len(clean_data) > 0:
                        plot_data.append(clean_data)
                        channel_indices.append(ch)  # Store the channel index
                        # Get channel name
                        ch_name = f"Ch{ch}"  # Default name
                        for chnames in self.channel_names:
                            if ch < len(chnames):
                                ch_name = chnames[ch].replace("Intan Input (1)/PortA ", "")
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
    
    def _extract_channel_data(self, id_data, feature, channel, remove_outliers, outlier_threshold):
        """
        Extract and process data for a single channel.
        """
        try:
            # Extract feature data for this channel
            ch_data = []
            for _, row in id_data.iterrows():
                feature_values = row[feature]
                if hasattr(feature_values, '__len__') and len(feature_values) > channel:
                    ch_data.append(feature_values[channel])
            
            if not ch_data:
                print(f"No data collected for channel {channel}")
                return None
                
            ch_data = np.array(ch_data)
            # Remove any NaN values
            ch_data = ch_data[~np.isnan(ch_data)]
            original_data = ch_data.copy()

            # Process outliers
            clean_data, outliers = self._handle_outliers(ch_data, remove_outliers, outlier_threshold)
            
            if len(clean_data) > 0:
                return {
                    'clean_data': clean_data,
                    'original_data': original_data,
                    'outliers': outliers
                }
            else:
                print(f"No valid data after processing for channel {channel}")
                return None
                
        except Exception as e:
            print(f"Warning: Error processing channel {channel}: {e}")
            return None
        

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
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
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
                'mean': np.mean(data),
                'std': np.std(data),
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
