import sys
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mne

from pythoneeg import core
from pythoneeg import visualization
from pythoneeg import constants

core.set_temp_directory('/scr1/users/dongjp')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger()
# animal_ids = ['A5', 'A10', 'F22', 'G25', 'G26', 'N21', 'N22', 'N23', 'N24', 'N25']
animal_ids = ['061322_Group10 M8, M10 M8', # normal
              '062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 BM6', # sparsely disconnected
              '071321_Cohort 3_AM4_CF1_DF3_FF6 AM4', # normal
              '081922_cohort10_group4_2mice_FMut_FHet FHET', # intermittently disconnected
              '090122_group4_2mice_FMut_MMut FMUT' # very disconnected
             ] 

# /mnt/isilon/marsh_single_unit/PythonEEG Data Bins/Sox5/Dr. Lefebvre Project/061022_group 9 M1, M2, M3/group9_M2_Cage 3/061122_group9_M2_Cage 3_files0-24

base_folder = Path('/mnt/isilon/marsh_single_unit/PythonEEG').resolve()
save_folder = Path('/home/dongjp/Downloads/5-28 filtered').resolve()
if not save_folder.exists():
    save_folder.mkdir(parents=True, exist_ok=True)

for animal_id in animal_ids:

    # SECTION load in windowed analysis results
    war = visualization.WindowAnalysisResult.load_pickle_and_json(base_folder / 'notebooks' / 'tests' / 'test-wars-full' / f'{animal_id}')
    war.filter_all()
    # !SECTION

    # SECTION load into AP
    save_path = save_folder / animal_id
    if not save_path.exists():
        save_path.mkdir(parents=True)
    ap = visualization.AnimalPlotter(war, save_fig=True, save_path=save_path / animal_id)
    # !SECTION

    # SECTION plot
    ap.plot_coherecorr_spectral(figsize=(20, 5), score_type='z')
    ap.plot_psd_histogram(figsize=(10, 4), avg_channels=True, plot_type='loglog')
    ap.plot_psd_spectrogram(figsize=(20, 4), mode='none')
    
    del war
    # !SECTION

"""
sbatch --mem 100G -c 4 -t 24:00:00 ./notebooks/examples/pipeline.sh ./notebooks/examples/pipeline-apfig-fromsave.py
"""