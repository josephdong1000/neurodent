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

core.set_temp_directory('/scr1/users/dongjp')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG, stream=sys.stdout, force=True)
logger = logging.getLogger()


animal_ids = ['A5', 'A10', 'F22', 'G25', 'G26', 'N21', 'N22', 'N23', 'N24', 'N25']
# animal_ids = ['A5', 'A10']
base_folder = Path('/mnt/isilon/marsh_single_unit/PythonEEG Data Bins').resolve()

wars = []
for animal_id in animal_ids:
    war = visualization.WindowAnalysisResult.load_pickle_and_json(Path(f'/home/dongjp/source-code/PyEEG/notebooks/tests/test-wars-full/{animal_id}').resolve())
    # war = war.filter_all()
    wars.append(war)
ep = visualization.ExperimentPlotter(wars)

# features = ['rms','ampvar', 'psdtotal', 'psdslope']
# for feature in features:
#     for plot in ['boxplot', 'scatter', 'violin']:
#         for xgroup, size in zip(['animal', 'isday', 'genotype'], [(30,5), (10,5), (10,5)]):
#             plot_func = getattr(ep, f'plot_{plot}')
#             plot_func(feature, figsize=size, xgroup=xgroup)
#             plt.savefig(f'/home/dongjp/Downloads/3-21-25/{xgroup}-{feature}-{plot}.png')

for xgroup in ['animal', 'isday', 'genotype']:

    ep.plot_2d_feature('pcorr', xgroup=xgroup)
    plt.savefig(f'/home/dongjp/Downloads/3-21-25/{xgroup}-pcorr-2d.png')
    ep.plot_2d_feature_freq('cohere', xgroup=xgroup)
    plt.savefig(f'/home/dongjp/Downloads/3-21-25/{xgroup}-cohere-2d.png')


"""
sbatch --mem 100G -c 8 -t 24:00:00 ./notebooks/examples/pipeline-warfig.sh
"""