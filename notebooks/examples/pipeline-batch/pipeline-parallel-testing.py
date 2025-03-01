# %% [markdown]
# # Marsh Lab Analysis Pipeline

# %%
import sys
from pathlib import Path
import tempfile
import os
import logging
import time
import subprocess

import dask
from dask.distributed import Client, LocalCluster
from dask_jobqueue.slurm import SLURMRunner, SLURMCluster

packageroot = Path('/mnt/isilon/marsh_single_unit/PythonEEG')
print(packageroot)
sys.path.append(str(packageroot))

from pythoneeg import core  # noqa: E402
from pythoneeg import visualization  # noqa: E402
from pythoneeg import constants  # noqa: E402


if __name__ == '__main__':

    # Set up logging
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger('dask')
    # logger.setLevel(logging.DEBUG)


    # print(f"portdash: {portdash}")

    # raise Exception("Stop here")

    cluster = SLURMCluster(cores=5, memory='20GB')
    cluster.scale(jobs=20)
    client = Client(cluster)

    # cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    # client = Client(cluster)

    print(f"\n\n\tclient.dashboard_link: {client.dashboard_link}\n\n")

    tempfile.tempdir = '/scr1/users/dongjp'

    base_folder = Path('/mnt/isilon/marsh_single_unit/PythonEEG Data Bins')
    output_folder = Path(__file__).parent.parent.resolve() / 'pipeline-wars'
    animal_ids = ['A10 KO', 'F22 KO']

    print("\nStarting pipeline execution...")
    for animal_id in animal_ids:
        ao = visualization.AnimalOrganizer(base_folder, animal_id, mode="concat", assume_from_number=True)
        ao.convert_colbins_to_rowbins()
        ao.convert_rowbins_to_rec()

        start_time = time.time()
        war = ao.compute_windowed_analysis(['rms'], exclude=['nspike', 'wavetemp'], multiprocess_mode='dask')
        print(time.time() - start_time)
        raise Exception("Stop here")
        # war.to_pickle_and_json(output_folder / animal_id)



"""
sbatch --mem 25G -c 4 -t 24:00:00 /mnt/isilon/marsh_single_unit/PythonEEG/notebooks/examples/pipeline-batch/pipeline-parallel-testing.sh
"""