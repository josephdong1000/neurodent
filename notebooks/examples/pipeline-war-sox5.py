import os
import sys
from pathlib import Path
import time
import tempfile
import logging
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mne
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from django.utils.text import slugify


packageroot = Path('/home/dongjp/source-code/PyEEG').resolve()
sys.path.append(str(packageroot))

from pythoneeg import core
from pythoneeg import visualization
from pythoneeg import constants

core.set_temp_directory('/scr1/users/dongjp')

# SECTION 1: Set up clusters
cluster_window = SLURMCluster(
        cores=8,
        memory='100GB',
        walltime='48:00:00',
        interface=None,
        scheduler_options={'interface': 'eth1'},
        job_extra_directives=['--output=/dev/null',
                             '--error=/dev/null']
    )
print(f"\n\n\tcluster_window.dashboard_link: {cluster_window.dashboard_link}\n\n")
cluster_spike = SLURMCluster(
        cores=1,
        memory='20GB',
        processes=1,
        walltime='12:00:00',
        interface=None,
        scheduler_options={'interface': 'eth1'}, # Look at `nmcli dev status` to find the correct interface
        job_extra_directives=['--output=/dev/null',
                             '--error=/dev/null']
    )
print(f"\n\n\tcluster_spike.dashboard_link: {cluster_spike.dashboard_link}\n\n")
cluster_window.scale(10)
cluster_spike.scale(12)
cluster_window.wait_for_workers(10)
cluster_spike.wait_for_workers(12)


# SECTION 2: Compute windowed analysis
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG, stream=sys.stdout, force=True)
logger = logging.getLogger()

with open(Path('./notebooks/tests/sox5.json').resolve(), 'r') as f:
    data = json.load(f)
data_parent_folder = Path(data['data_parent_folder'])
constants.GENOTYPE_ALIASES = data['GENOTYPE_ALIASES']
data_folders_to_animal_ids = data['data_folders_to_animal_ids']
wars = []

# constants.SORTING_PARAMS['freq_min'] = 60
# constants.SORTING_PARAMS['freq_max'] = 400

for data_folder, animal_ids in data_folders_to_animal_ids.items():
    for animal_id in animal_ids:
        ao = visualization.AnimalOrganizer(data_parent_folder / data_folder, animal_id,
                                    mode="nest", 
                                    assume_from_number=True,
                                    skip_days=['bad'])
        with Client(cluster_window) as client:
            client.upload_file(str(packageroot / 'pythoneeg.tar.gz'))

            # SECTION 1: Find bin files
            ao = visualization.AnimalOrganizer(data_parent_folder / data_folder, animal_id,
                                        mode="nest", 
                                        assume_from_number=True,
                                        skip_days=['bad'],
                                        lro_kwargs={'mode': 'bin',
                                                    'multiprocess_mode': 'dask',
                                                    'overwrite_rowbins': True}
            )
            
            ao.convert_colbins_to_rowbins(overwrite=True, multiprocess_mode='dask')
            ao.convert_rowbins_to_rec(multiprocess_mode='dask') # paralleization breaks if not enough memory
            war = ao.compute_windowed_analysis(['all'], multiprocess_mode='dask')

        with Client(cluster_spike) as client:
            client.upload_file(str(packageroot / 'pythoneeg.zip'))
            
            sars = ao.compute_spike_analysis(multiprocess_mode='dask')
            for sar in sars:
                sar.save_fif_and_json(Path(f'./notebooks/tests/test-sars-full/{data_folder} {slugify(sar.animal_day, allow_unicode=True)}').resolve(), overwrite=True) # animal_day not unique for Sox5 rec sessions, so add bin_folder_name
            war.read_sars_spikes(sars)
            
        war.save_pickle_and_json(Path(f'./notebooks/tests/test-wars-full/{data_folder} {animal_id}').resolve())
        wars.append(war)

"""
sbatch --mem 300G -c 4 -t 48:00:00 /mnt/isilon/marsh_single_unit/PythonEEG/notebooks/examples/pipeline.sh /mnt/isilon/marsh_single_unit/PythonEEG/notebooks/examples/pipeline-war-sox5.py
"""