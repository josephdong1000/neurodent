import sys
from pathlib import Path
import tempfile
import time

import dask
from dask.distributed import Client
from dask_jobqueue.slurm import SLURMCluster

packageroot = Path('/mnt/isilon/marsh_single_unit/PythonEEG')
print(packageroot)
sys.path.append(str(packageroot))

from pythoneeg import core  # noqa: E402
from pythoneeg import visualization  # noqa: E402
from pythoneeg import constants  # noqa: E402


if __name__ == '__main__':

    cluster = SLURMCluster(cores=5, memory='20GB')
    cluster.scale(jobs=20)
    client = Client(cluster)

    print(f"\n\n\tclient.dashboard_link: {client.dashboard_link}\n\n")

    tempfile.tempdir = '/scr1/users/dongjp'

    base_folder = Path('/mnt/isilon/marsh_single_unit/PythonEEG Data Bins')
    output_folder = Path(__file__).parent.parent.resolve() / 'pipeline-wars'

    animal_ids = ['A5 WT', 'A10 KO', 'F22 KO', 'G25', 'G26', 'N21', 'N22', 'N23', 'N24', 'N25']

    print("\nStarting pipeline execution...")
    for animal_id in animal_ids:
        ao = visualization.AnimalOrganizer(base_folder, animal_id, mode="concat", assume_from_number=True)
        ao.convert_colbins_to_rowbins()
        ao.convert_rowbins_to_rec()

        start_time = time.time()
        war = ao.compute_windowed_analysis(['all'], exclude=['nspike', 'wavetemp'], multiprocess_mode='dask')
        print(time.time() - start_time)
        # war.to_pickle_and_json(output_folder / animal_id)


"""
sbatch --mem 50G -c 4 -t 24:00:00 /mnt/isilon/marsh_single_unit/PythonEEG/notebooks/examples/pipeline-batch/pipeline.sh
"""