import os
import sys
from pathlib import Path
import time
import tempfile
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mne
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

packageroot = Path('/home/dongjp/source-code/PyEEG').resolve()
sys.path.append(str(packageroot))

from pythoneeg import core
from pythoneeg import visualization
from pythoneeg import constants

core.set_temp_directory('/scr1/users/dongjp')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG, stream=sys.stdout, force=True)
logger = logging.getLogger()


animal_ids = ['A5', 'A10', 'F22', 'G25', 'G26', 'N21', 'N22', 'N23', 'N24', 'N25']
# animal_ids = ['A5', 'A10']
base_folder = Path('/mnt/isilon/marsh_single_unit/PythonEEG Data Bins').resolve()

# TODO reject all the bad channels by eye when constructing EP
wars = []
for animal_id in animal_ids:
    war = visualization.WindowAnalysisResult.load_pickle_and_json(Path(f'/home/dongjp/source-code/PyEEG/notebooks/tests/test-wars-full/{animal_id}').resolve())
    if animal_id == 'F22':
        war.filter_all(reject_channels=['LMot', 'RBar', 'RVis', 'RAud'])
    elif animal_id == 'N21':
        war.filter_all(reject_channels=['RBar'])
    elif animal_id == 'G25':
        war.filter_all(reject_channels=['LAud', 'LHip'])

    wars.append(war)
ep = visualization.ExperimentPlotter(wars)

# CATPLOTS, as box, violin
# TODO feature vs. genotype/isday
# TODO feature vs. genotype
# TODO feature vs. genotype/isday, collapsed
# TODO feature vs. genotype, collapsed
# TODO psdband vs. genotype/isday/band, collapsed
# TODO psdband vs. genotype/band, collapsed
# TODO cohere/pcorr vs. genotype/isday, collapsed
# TODO cohere/pcorr vs. genotype, collapsed

# GRIDPLOTS
# TODO cohere vs. genotype/isday
# TODO cohere vs. genotype/band
# TODO pcorr vs. genotype/isday
# TODO pcorr vs. genotype

for feature in constants.LINEAR_FEATURE:
    for kind in ['box', 'violin']:
        for groupby in ['genotype', ['genotype', 'isday']]:
            for collapse in [False, True]:
                ep.plot_catplot(feature, groupby=groupby, kind=kind, collapse_channels=collapse)
                plt.savefig(f'/home/dongjp/Downloads/3-28-25/{feature}-{groupby}-{kind}-{collapse}.png')
for kind in ['box', 'violin']:
    ep.plot_catplot('psdband', groupby=['genotype', 'isday'], 
                    x='genotype',
                    col='isday',
                    hue='band',
                    kind=kind, collapse_channels=True)
    plt.savefig(f'/home/dongjp/Downloads/3-28-25/psdband-genotype-isday-{kind}-True.png')
    ep.plot_catplot('psdband', groupby=['genotype'], 
                    x='genotype',
                    hue='band',
                    kind=kind, collapse_channels=True)
    plt.savefig(f'/home/dongjp/Downloads/3-28-25/psdband-genotype-{kind}-True.png')

for feature in constants.MATRIX_FEATURE:
    for kind in ['box', 'violin']:
        for groupby in [['genotype', 'isday'], 'genotype']:
            ep.plot_catplot(feature, groupby=groupby, kind=kind, collapse_channels=True)
            plt.savefig(f'/home/dongjp/Downloads/3-28-25/{feature}-{groupby}-{kind}-True.png')

ep.plot_2d_feature_2('cohere', groupby=['genotype', 'isday'])
plt.savefig(f'/home/dongjp/Downloads/3-28-25/cohere-genotype-isday-matrix-False.png')
ep.plot_2d_feature_2('cohere', groupby='genotype', col='band', row='genotype')
plt.savefig(f'/home/dongjp/Downloads/3-28-25/cohere-genotype-band-matrix-False.png')

ep.plot_2d_feature_2('pcorr', groupby=['genotype', 'isday'])
plt.savefig(f'/home/dongjp/Downloads/3-28-25/pcorr-genotype-isday-matrix-False.png')
ep.plot_2d_feature_2('pcorr', groupby='genotype')
plt.savefig(f'/home/dongjp/Downloads/3-28-25/pcorr-genotype-matrix-False.png')



# features = ['rms','ampvar', 'psdtotal', 'psdslope']
# for feature in features:
#     for plot in ['boxplot', 'scatter', 'violin']:
#         for xgroup, size in zip(['animal', 'isday', 'genotype'], [(30,5), (10,5), (10,5)]):
#             plot_func = getattr(ep, f'plot_{plot}')
#             plot_func(feature, figsize=size, xgroup=xgroup)
#             plt.savefig(f'/home/dongjp/Downloads/3-28-25/{xgroup}-{feature}-{plot}.png')

# for xgroup in ['animal', 'isday', 'genotype']:

#     ep.plot_2d_feature('pcorr', xgroup=xgroup)
#     plt.savefig(f'/home/dongjp/Downloads/3-21-25/{xgroup}-pcorr-2d.png')
#     ep.plot_2d_feature_freq('cohere', xgroup=xgroup)
#     plt.savefig(f'/home/dongjp/Downloads/3-21-25/{xgroup}-cohere-2d.png')


"""
sbatch --mem 100G -c 8 -t 24:00:00 ./notebooks/examples/pipeline-warfig.sh
"""