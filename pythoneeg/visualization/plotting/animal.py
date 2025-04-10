import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gzscore, linregress, zscore

from ... import core
from ... import visualization as viz
from ... import constants


class AnimalPlotter(viz.AnimalFeatureParser):

    def __init__(self, war: viz.WindowAnalysisResult) -> None:
        self.window_result = war
        self.genotype = war.genotype
        self.channel_names = war.channel_names
        self.n_channels = len(self.channel_names)
        self.__assume_from_number = war.assume_from_number
        self.channel_abbrevs = war.channel_abbrevs

    # REVIEW this function may not be necessary
    # def get_animalday_metadata(self, animalday) -> core.DDFBinaryMetadata:
    #     return self.window_result.meta[self.window_result.animaldays.index(animalday)]

    def _abbreviate_channel(self, ch_name:str):
        for k,v in self.CHNAME_TO_ABBREV:
            if k in ch_name:
                return v
        return ch_name

    def plot_coherecorr_matrix(self, groupby="animalday", bands=None, figsize=None, cmap='viridis', **kwargs):
        # avg_result = self.window_result.get_grouped_avg(constants.MATRIX_FEATURE, groupby=groupby)
        # avg_coheresplit = pd.json_normalize(avg_result['cohere']).set_index(avg_result.index)
        # avg_result = avg_coheresplit.join(avg_result)
        avg_result = self.__get_groupavg_coherecorr(groupby, **kwargs)

        if bands is None:
            bands = constants.BAND_NAMES + ['pcorr']
        elif isinstance(bands, str):
            bands = [bands]
        n_row = avg_result.index.size
        # rowcount = 0
        fig, ax = plt.subplots(n_row, len(bands), squeeze=False, figsize=figsize, **kwargs)

        normlist = [matplotlib.colors.Normalize(vmin=0, vmax=np.max(np.concatenate(avg_result[band].values))) for band in bands]
        for i, (_, row) in enumerate(avg_result.iterrows()):
            self._plot_coherecorr_matrixgroup(row, bands, ax[i, :], show_bandname=i == 0, norm_list=normlist, cmap=cmap, **kwargs)
            # rowcount += 1
        plt.show()

    def plot_coherecorr_diff(self, groupby="isday", bands=None, figsize=None, cmap='bwr', **kwargs):
        avg_result = self.__get_groupavg_coherecorr(groupby, **kwargs)
        avg_result = avg_result.drop('cohere', axis=1, errors='ignore')
        if len(avg_result.index) != 2:
            raise ValueError(f"Difference can only be calculated between 2 rows. {groupby} resulted in {len(avg_result.index)} rows")

        if bands is None:
            bands = constants.BAND_NAMES + ['pcorr']
        elif isinstance(bands, str):
            bands = [bands]

        diff_result = avg_result.iloc[1] - avg_result.iloc[0]
        diff_result.name = f"{avg_result.iloc[1].name} - {avg_result.iloc[0].name}"

        fig, ax = plt.subplots(1, len(bands), squeeze=False, figsize=figsize, **kwargs)

        self._plot_coherecorr_matrixgroup(diff_result, bands, ax[0, :], show_bandname=True, center_cmap=True, cmap=cmap, **kwargs)

    def _plot_coherecorr_matrixgroup(self, group:pd.Series, bands:list[str], ax:list[matplotlib.axes.Axes], show_bandname,
                                    center_cmap=False, norm_list=None, show_channelname=True, **kwargs):
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
                ax[i].set_xlabel(band, fontsize='x-large')
                ax[i].xaxis.set_label_position('top')

            if show_channelname:
                ax[i].set_xticks(range(self.n_channels), self.channel_abbrevs, rotation='vertical')
                ax[i].set_yticks(range(self.n_channels), self.channel_abbrevs)
            else:
                ax[i].set_xticks(range(self.n_channels), " ")
                ax[i].set_yticks(range(self.n_channels), " ")

        ax[0].set_ylabel(rowname, rotation='horizontal', ha='right')

    def __get_groupavg_coherecorr(self, groupby="animalday", **kwargs):
        avg_result = self.window_result.get_groupavg_result(constants.MATRIX_FEATURES, groupby=groupby)
        avg_coheresplit = pd.json_normalize(avg_result['cohere']).set_index(avg_result.index) # Split apart the cohere dictionaries
        return avg_coheresplit.join(avg_result)

    def plot_linear_temporal(self, multiindex=["animalday", "animal", "genotype"], features:list[str]=None, channels:list[int]=None, figsize=None,
                             score_type='z', show_endfile=False, **kwargs):
        if features is None:
            features = constants.LINEAR_FEATURES + constants.BAND_FEATURES
        if channels is None:
            channels = np.arange(self.n_channels)

        # df_featgroups = self.window_result.get_grouped(features, groupby=groupby)
        df_rowgroup = self.window_result.get_grouprows_result(features, multiindex=multiindex)
        for i, df_row in df_rowgroup.groupby(level=0):
            fig, ax = plt.subplots(len(features), 1, figsize=figsize, sharex=True,
                                   gridspec_kw={'height_ratios' : [constants.LINPLOT_HEIGHT_RATIOS[x] for x in features]})
            plt.subplots_adjust(hspace=0)

            for j, feat in enumerate(features):
                self._plot_linear_temporalgroup(group=df_row, feature=feat, ax=ax[j], score_type=score_type, channels=channels, show_endfile=show_endfile, **kwargs)
            ax[-1].set_xlabel("Time (s)")
            fig.suptitle(i)
            plt.show()

    def _plot_linear_temporalgroup(self, group:pd.DataFrame, feature:str, ax:matplotlib.axes.Axes, channels:list[int]=None, score_type:str='z',
                                     duration_name='duration', channel_y_offset=10, feature_y_offset=10, endfile_name='endfile', show_endfile=False, show_channelname=True, **kwargs):

        data_Z = self.__get_linear_feature(group=group, feature=feature, score_type=score_type)

        data_t = group[duration_name]
        data_T = np.cumsum(data_t)

        if channels is None:
            channels = np.arange(data_Z.shape[1])
        data_Z = data_Z[:, channels, :]

        n_chan = data_Z.shape[1]
        n_feat = data_Z.shape[2]
        chan_offset = np.linspace(0, channel_y_offset * n_chan, n_chan, endpoint=False).reshape((1, -1, 1))
        feat_offset = np.linspace(0, feature_y_offset * n_chan * n_feat, n_feat, endpoint=False).reshape((1, 1, -1))
        data_Z += chan_offset
        data_Z += feat_offset
        ytick_offset = feat_offset.squeeze() + np.mean(chan_offset.flatten())

        for i in range(n_feat):
            ax.plot(data_T, data_Z[:, :, i], c=f'C{i}', **kwargs)
        match feature: # TODO refactor this to use constants
            case 'rms' | 'ampvar' | 'psdtotal' | 'nspike' | 'logrms' | 'logampvar' | 'logpsdtotal' | 'lognspike':
                ax.set_yticks([ytick_offset], [feature])
            case 'psdslope':
                ax.set_yticks(ytick_offset, ['psdslope', 'psdintercept'])
            case 'psdband' | 'psdfrac' | 'logpsdband' | 'logpsdfrac':
                ax.set_yticks(ytick_offset, constants.BAND_NAMES)
            case _:
                raise ValueError(f"Invalid feature {feature}")

        if show_endfile:
            self._plot_filediv_lines(group=group, ax=ax, duration_name=duration_name, endfile_name=endfile_name)

    def __get_linear_feature(self, group:pd.DataFrame, feature:str, score_type='z', triag=True):
        match feature: # TODO refactor this to use constants
            case 'rms' | 'ampvar' | 'psdtotal' | 'nspike' | 'logrms' | 'logampvar' | 'logpsdtotal' | 'lognspike':
                data_X = np.array(group[feature].to_list())
                data_X = np.expand_dims(data_X, axis=-1)
            case 'psdband' | 'psdfrac' | 'logpsdband' | 'logpsdfrac':
                data_X = np.array([list(d.values()) for d in group[feature]])
                data_X = np.stack(data_X, axis=-1)
                data_X = np.transpose(data_X)
            case 'psdslope':
                data_X = np.array(group[feature].to_list())
                data_X[:, :, 0] = -data_X[:, :, 0]
            case 'cohere':
                data_X = np.array([list(d.values()) for d in group[feature]])
                data_X = np.stack(data_X, axis=-1)
                if triag:
                    tril = np.tril_indices(data_X.shape[1], k=-1)
                    data_X = data_X[:, tril[0], tril[1], :]
                data_X = data_X.reshape(data_X.shape[0], -1, data_X.shape[-1])
                data_X = np.transpose(data_X)
            case 'pcorr':
                data_X = np.stack(group[feature], axis=-1)
                if triag:
                    tril = np.tril_indices(data_X.shape[1], k=-1)
                    data_X = data_X[tril[0], tril[1], :]
                data_X = data_X.reshape(-1, data_X.shape[-1])
                data_X = data_X.transpose()
                data_X = np.expand_dims(data_X, axis=-1)
            case _:
                raise ValueError(f"Invalid feature {feature}")

        return self._calculate_standard_data(data_X, mode=score_type, axis=0)

    def _plot_filediv_lines(self, group:pd.DataFrame, ax:matplotlib.axes.Axes, duration_name, endfile_name):
        filedivs = self.__get_filediv_times(group, duration_name, endfile_name)
        for xpos in filedivs:
            ax.axvline(xpos, ls='--', c='black', lw=1)

    def __get_filediv_times(self, group, duration_name, endfile_name):
        cumulative = group[duration_name].cumsum().shift(fill_value=0)
        # display( group[[endfile_name]].dropna().head())
        # display(cumulative.head())
        filedivs = group[endfile_name].dropna() + cumulative[group[endfile_name].notna()]
        return filedivs.tolist()

    def _calculate_standard_data(self, X, mode='z', axis=0):
        match mode:
            case "z":
                data_Z = zscore(X, axis=axis, nan_policy='omit')
            case "zall":
                data_Z = zscore(X, axis=None, nan_policy='omit')
            case "gz":
                data_Z = gzscore(X, axis=axis, nan_policy='omit')
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

    def plot_coherecorr_spectral(self, multiindex=["animalday", "animal", "genotype"], features:list[str]=None, figsize=None, score_type='z', cmap='bwr', triag=True,
                                 show_endfile=False, duration_name='duration', endfile_name='endfile', **kwargs):
        if features is None:
            features = constants.MATRIX_FEATURES
        height_ratios = {'cohere' : 5,
                         'pcorr' : 1}

        df_rowgroup = self.window_result.get_grouprows_result(features, multiindex=multiindex)
        for i, df_row in df_rowgroup.groupby(level=0):
            fig, ax = plt.subplots(len(features), 1, figsize=figsize, sharex=True,
                                   gridspec_kw={'height_ratios' : [height_ratios[x] for x in features]})
            plt.subplots_adjust(hspace=0)
            for j, feat in enumerate(features):
                self._plot_coherecorr_spectralgroup(group=df_row, feature=feat, ax=ax[j], score_type=score_type, triag=triag, show_endfile=show_endfile,
                                                    duration_name=duration_name, endfile_name=endfile_name, **kwargs)
            ax[-1].set_xlabel("Time (s)")
            fig.suptitle(i)
            plt.show()

    def _plot_coherecorr_spectralgroup(self, group:pd.DataFrame, feature:str, ax:matplotlib.axes.Axes,
                                        center_cmap=True, score_type='z', norm_list=None, show_featurename=True, show_endfile=False,
                                        duration_name='duration', endfile_name='endfile', cmap='bwr', triag=True, **kwargs):

        data_Z = self.__get_linear_feature(group=group, feature=feature, score_type=score_type)
        std_dev = np.nanstd(data_Z.flatten())

        # data_flat = data_Z.reshape(data_Z.shape[0], -1).transpose()

        if center_cmap:
            norm = matplotlib.colors.CenteredNorm(halfrange=std_dev * 2)
        else:
            norm = None

        n_ch = data_Z.shape[1]
        n_bands = len(constants.BAND_NAMES)

        for i in range(data_Z.shape[-1]):
            extent = (0, data_Z.shape[0] * group['duration'].median(), i * n_ch, (i+1) * n_ch)
            ax.imshow(data_Z[:, :, i].transpose(), interpolation='none', aspect='auto', norm=norm, cmap=cmap, extent=extent)

        if show_featurename:
            match feature:
                case 'cohere':
                    ticks = n_ch * np.linspace(1/2, n_bands + 1/2, n_bands, endpoint=False)
                    ax.set_yticks(ticks=ticks, labels=constants.BAND_NAMES)
                    for ypos in np.linspace(0, n_bands * n_ch, n_bands, endpoint=False):
                        ax.axhline(ypos, lw=1, ls='--', color='black')
                case 'pcorr':
                    ax.set_yticks(ticks=[1/2 * n_ch], labels=[feature])
                case _:
                    raise ValueError(f"Unknown feature name {feature}")

        if show_endfile:
            self._plot_filediv_lines(group=group, ax=ax, duration_name=duration_name, endfile_name=endfile_name)

    def plot_psd_histogram(self, groupby='animalday', figsize=None, avg_channels=False, plot_type='loglog', plot_slope=True, xlim=None, **kwargs):
        avg_result = self.window_result.get_groupavg_result(['psd'], groupby=groupby)

        n_col = avg_result.index.size
        fig, ax = plt.subplots(1, n_col, squeeze=False, figsize=figsize, sharex=True, sharey=True, **kwargs)
        plt.subplots_adjust(wspace=0)
        for i, (idx, row) in enumerate(avg_result.iterrows()):
            freqs = row['psd'][0]
            psd = row['psd'][1]
            if avg_channels:
                psd = np.average(psd, axis=-1, keepdims=True)
                label = 'Average'
            else:
                label = self.channel_abbrevs
            match plot_type:
                case 'loglog':
                    ax[0, i].loglog(freqs, psd, label=label)
                case 'semilogy':
                    ax[0, i].semilogy(freqs, psd, label=label)
                case 'semilogx':
                    ax[0, i].semilogy(freqs, psd, label=label)
                case 'linear':
                    ax[0, i].plot(freqs, psd, label=label)
                case _:
                    raise ValueError(f"Invalid plot type {plot_type}")

            frange = np.logical_and(freqs >= constants.FREQ_BAND_TOTAL[0],
                                    freqs <= constants.FREQ_BAND_TOTAL[1])
            logf = np.log10(freqs[frange])
            logpsd = np.log10(psd[frange, :])

            linfit = np.zeros((psd.shape[1], 2))
            for k in range(psd.shape[1]):
                result = linregress(logf, logpsd[:, k], 'less')
                linfit[k, :] = [result.slope, result.intercept]

            for j, (m,b) in enumerate(linfit.tolist()):
                ax[0, i].plot(freqs, 10**(b + m * np.log10(freqs)), c=f'C{j}', alpha=0.75)

            ax[0, i].set_title(idx)
            ax[0, i].set_xlabel("Frequency (Hz)")
        ax[0, 0].set_ylabel("PSD (uV^2/Hz)")
        ax[0, -1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        ax[0, -1].set_xlim(xlim)
        plt.show()

    # STUB plot spectrogram over time, doing gaussian filter convolving when relevant, scaling logarithmically
    def plot_psd_spectrogram(self, multiindex=['animalday', 'animal', 'genotype'], freq_range=(1, 50), center_stat='mean', mode='z', figsize=None, cmap='magma', **kwargs):
        # if features is None:
        #     features = constants.MATRIX_FEATURE
        # height_ratios = {'cohere' : 5,
        #                  'pcorr' : 1}

        df_rowgroup = self.window_result.get_grouprows_result(['psd'], multiindex=multiindex)
        for i, df_row in df_rowgroup.groupby(level=0):

            freqs = df_row.iloc[0]['psd'][0]
            psd = np.array([x[1] for x in df_row['psd'].tolist()])
            match center_stat:
                case 'mean':
                    psd = np.nanmean(psd, axis=-1).transpose()
                case 'median':
                    psd = np.nanmedian(psd, axis=-1).transpose()
                case _:
                    raise ValueError(f"Invalid statistic {center_stat}. Pick mean or median")
            psd = np.log10(psd)
            psd = self._calculate_standard_data(psd, mode=mode, axis=-1)
            freq_mask = np.logical_and((freq_range[0] <= freqs), (freqs <= freq_range[1]))
            freqs = freqs[freq_mask]
            psd = psd[freq_mask, :]

            extent = (0, psd.shape[1] * df_row['duration'].median(), np.min(freqs), np.max(freqs))
            # print(psd.nanmin(), psd.nanmax())
            norm = matplotlib.colors.Normalize()
            # norm = matplotlib.colors.LogNorm()
            # norm = matplotlib.colors.CenteredNorm()


            fig, ax = plt.subplots(1, 1, figsize=figsize)
            # ax.pcolormesh(psd, )
            axim = ax.imshow(np.flip(psd, axis=0), interpolation='none', aspect='auto', norm=norm, cmap=cmap, extent=extent)
            cbar = fig.colorbar(axim, ax=ax)
            cbar.set_label(f'log(PSD) {mode}')

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(i)
            plt.show()