# %% [markdown]
# # Marsh Lab Analysis Pipeline

# %%
import sys
from pathlib import Path
import dask
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dask')
logger.setLevel(logging.DEBUG)

packageroot = Path('/mnt/isilon/marsh_single_unit/PythonEEG')
print(packageroot)
sys.path.append(str(packageroot))
print("pingusball")

# Log Dask/Slurm info
# logger.info(f"Number of Slurm cores: {os.environ.get('SLURM_CPUS_PER_TASK')}")
# logger.info("Current Dask configuration:")
# for key, value in dask.config.config.items():
#     logger.info(f"  {key}: {value}")



# %%
import tempfile

from pythoneeg import core
from pythoneeg import visualization
from pythoneeg import constants

tempfile.tempdir = '/scr1/users/dongjp'

# %%
base_folder = Path('/mnt/isilon/marsh_single_unit/PythonEEG Data Bins')
output_folder = Path(__file__).parent.parent.resolve() / 'pipeline-wars'
animal_ids = ['A5 WT', 'A10 KO', 'F22 KO']

# %%



for animal_id in animal_ids:
    ao = visualization.AnimalOrganizer(base_folder, animal_id, mode="concat", assume_from_number=True)
    ao.convert_colbins_to_rowbins()
    ao.convert_rowbins_to_rec()

    war = ao.compute_windowed_analysis(['all'], exclude=['nspike', 'wavetemp'], multiprocess_mode='dask-slurm')
    # war.to_pickle_and_json(output_folder / animal_id)

# %%
# Run in batch mode
"""
sbatch --mem 25G -c 20 -t 1:00:00 /mnt/isilon/marsh_single_unit/PythonEEG/notebooks/examples/pipeline-batch/pipeline-parallel-testing.sh 
"""