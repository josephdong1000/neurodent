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

cluster_window = SLURMCluster(
        cores=10,
        processes=1,
        memory='30GB',
        walltime='24:00:00',
        interface=None,
        scheduler_options={'interface': 'eth1'},
        job_extra_directives=['--output=/dev/null',
                             '--error=/dev/null']
    )
cluster_window.scale(jobs=15)  # Scale to 15 workers
print(f"\n\n\tcluster_window.dashboard_link: {cluster_window.dashboard_link}\n\n")

cluster_window.wait_for_workers(15)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG, stream=sys.stdout, force=True)
logger = logging.getLogger()


animal_ids = ['A5', 'A10', 'F22', 'G25', 'G26', 'N21', 'N22', 'N23', 'N24', 'N25']
# animal_ids = ['A5']
base_folder = Path('/mnt/isilon/marsh_single_unit/PythonEEG Data Bins').resolve()

for animal_id in animal_ids:
    ao = visualization.AnimalOrganizer(base_folder, animal_id, mode="concat", assume_from_number=True)
    ao.convert_colbins_to_rowbins(overwrite=False)

    with Client(cluster_window) as client:
        client.upload_file(str(packageroot / 'pythoneeg.zip'))
        ao.convert_rowbins_to_rec(multiprocess_mode='dask') # paralleization breaks if not enough memory
        war = ao.compute_windowed_analysis(['all'], multiprocess_mode='dask')
        war.save_pickle_and_json(Path(f'/home/dongjp/source-code/PyEEG/notebooks/tests/test-wars-full/{animal_id}').resolve())

    logging.info(f"\tDone with {animal_id}")

"""
sbatch --mem 50G -c 5 -t 24:00:00 ./notebooks/examples/pipeline-war.sh
"""