import sys
from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mne
import seaborn.objects as so
from seaborn import axes_style

from pythoneeg import core
from pythoneeg import visualization
from pythoneeg import constants

core.set_temp_directory('/scr1/users/dongjp')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger()

base_folder = Path('/mnt/isilon/marsh_single_unit/PythonEEG')
# animal_ids = ['A5', 'A10', 'F22', 'G25', 'G26', 'N21', 'N22', 'N23', 'N24', 'N25']
# animal_ids = ['A5', 'A10']
animal_ids = [p.name for p in (base_folder / 'notebooks' / 'tests' / 'test-wars-sox5-2').glob('*') if p.is_dir()]
# animal_ids = [
#     '081922_cohort10_group4_2mice_FMut_FHet FHET',
#     '062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 CF2',
# ]

def load_war(animal_id):
    logger.info(f'Loading {animal_id}')
    war = visualization.WindowAnalysisResult.load_pickle_and_json(
        Path(base_folder / 'notebooks' / 'tests' / 'test-wars-sox5-2' / f'{animal_id}').resolve()
    )
    if war.genotype == 'Unknown': # Remove pathological recordings
        logger.info(f'Skipping {animal_id} because genotype is Unknown')
        return None
    
    war.filter_all()
    war.aggregate_time_windows()
    war.add_unique_hash(4)
    war.reorder_and_pad_channels(['LMot', 'RMot', 'LBar', 'RBar', 'LAud', 'RAud', 'LVis', 'RVis', 'LHip', 'RHip'], use_abbrevs=True)

    return war

# Use multiprocessing to load WARs in parallel
num_cores = 15
with Pool(num_cores) as pool:
    wars = []
    for war in tqdm(pool.imap(load_war, animal_ids), total=len(animal_ids), desc="Loading WARs"):
        if war is not None:
            wars.append(war)


logger.info(f'{len(wars)} wars loaded')
exclude = ['nspike', 'lognspike']
ep = visualization.ExperimentPlotter(wars, exclude=exclude)

# SECTION Define parameters
save_folder = Path('/home/dongjp/Downloads/6-9 sox5 swarm marked').resolve()
if not save_folder.exists():
    save_folder.mkdir(parents=True)
with open(save_folder / 'ep.pkl', 'wb') as f:
    pickle.dump(ep, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'Saved ExperimentPlotter to {str(f)}')
# !SECTION

# SECTION use seaborn.so to plot figure
features = ['psdtotal', 'psdslope', 'logpsdtotal']
# features = ['psdtotal', 'psdslope']
for feature in features:
    df = ep.pull_timeseries_dataframe(feature=feature, groupby=['animal', 'genotype', 'isday'], collapse_channels=True)
    print(df)
    df = df.groupby(['animal', 'genotype', 'isday'])[feature].mean().reset_index()
    print(df)
    (
        so.Plot(df, x='genotype', y=feature, color='isday')
        .facet(col='isday')
        .add(so.Dot(marker='s', color='k'), so.Agg(), so.Shift(-0.2))
        .add(so.Line(color='k', linestyle='--'), so.Agg(), so.Shift(x=-0.2))
        .add(so.Range(color='k'), so.Est(errorbar='sd'), so.Shift(x=-0.2))
        .add(so.Dot(), so.Jitter(seed=42))
        .add(so.Text(halign='left'), so.Jitter(seed=42), text='animal')
        .theme(axes_style('ticks'))
        .layout(size=(20, 5), engine='tight')
        .save(save_folder / f'{feature}-genotype-isday-avgch.png', dpi=300)
    )
# !SECTION


"""
sbatch --mem 200GB -c 20 -t 24:00:00 ./notebooks/examples/pipeline.sh ./notebooks/examples/pipeline-epfig-so.py
"""