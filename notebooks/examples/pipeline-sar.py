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

cluster_spike = SLURMCluster(
        cores=1,
        processes=1,
        memory='20GB',
        walltime='24:00:00',
        interface=None,
        scheduler_options={'interface': 'eth1'},
        job_extra_directives=['--output=/dev/null',
                             '--error=/dev/null']
    )
cluster_spike.scale(jobs=15)  # Scale to 15 workers
print(f"\n\n\tcluster_spike.dashboard_link: {cluster_spike.dashboard_link}\n\n")

cluster_spike.wait_for_workers(15)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG, stream=sys.stdout, force=True)
logger = logging.getLogger()


animal_ids = ['A5', 'A10', 'F22', 'G25', 'G26', 'N21', 'N22', 'N23', 'N24', 'N25']
# animal_ids = ['A5']
base_folder = Path('/mnt/isilon/marsh_single_unit/PythonEEG Data Bins').resolve()

for animal_id in animal_ids:
    ao = visualization.AnimalOrganizer(base_folder, animal_id, mode="concat", assume_from_number=True)
    ao.convert_colbins_to_rowbins(overwrite=False)

    with Client(cluster_spike) as client:
        client.upload_file(str(packageroot / 'pythoneeg.zip'))
        ao.convert_rowbins_to_rec(multiprocess_mode='dask')
        sar: list[visualization.SpikeAnalysisResult] = ao.compute_spike_analysis(multiprocess_mode='dask')
        for s in sar:
            s.save_fif_and_json(Path(f'/home/dongjp/source-code/PyEEG/notebooks/tests/test-sars-full/{s.animal_day}').resolve(),
                                overwrite=True)

    logging.info(f"\tDone with {animal_id}")

"""
sbatch --mem 100G -c 10 -t 24:00:00 ./notebooks/examples/pipeline-sar.sh
"""