import os
import sys
from pathlib import Path
import logging
import json
from tqdm import tqdm

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

from neurodent import core, visualization, constants

core.set_temp_directory("/scr1/users/dongjp")

# SECTION 1: Set up clusters
cluster_window = SLURMCluster(
    cores=30,
    memory="100GB",
    walltime="48:00:00",
    interface=None,
    scheduler_options={"interface": "eth1"},  # Look at `nmcli dev status` to find the correct interface
    job_extra_directives=["--output=/dev/null", "--error=/dev/null"],
)
print(f"\n\n\tcluster_window.dashboard_link: {cluster_window.dashboard_link}\n\n")
cluster_spike = SLURMCluster(
    cores=1,
    memory="100GB",
    processes=1,
    walltime="6:00:00",
    interface=None,
    scheduler_options={"interface": "eth1"},  # Look at `nmcli dev status` to find the correct interface
    job_extra_directives=["--output=/dev/null", "--error=/dev/null"],
)
print(f"\n\n\tcluster_spike.dashboard_link: {cluster_spike.dashboard_link}\n\n")
cluster_window.scale(jobs=5)
cluster_spike.adapt(maximum_jobs=15)
# !SECTION

# SECTION 2: Setup parameters
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG, stream=sys.stdout, force=True
)
logger = logging.getLogger()

base_folder = Path("/mnt/isilon/marsh_single_unit/NeuRodent")
with open(base_folder / "notebooks" / "tests" / "sox5 combine genotypes.json", "r") as f:
    data = json.load(f)
bad_channels = data["bad_channels"]
data_parent_folder = Path(data["data_parent_folder"])
constants.GENOTYPE_ALIASES = data["GENOTYPE_ALIASES"]

# data_folders_to_animal_ids = data["data_folders_to_animal_ids"]
# Get only the second half of the dictionary items
items = list(data["data_folders_to_animal_ids"].items())
data_folders_to_animal_ids = dict(items[int(len(items) * 1 / 2) :])  # last 1/2
# data_folders_to_animal_ids = dict(items[int(len(items) * 2 / 3) :])  # last 1/3
# data_folders_to_animal_ids = data["data_folders_to_animal_ids"]

bad_folder_animalday = [
    "012322_cohort4_group6_3mice_FMUT___MMUT_MWT MHET",
    "012322_cohort4_group6_3mice_FMUT___MMUT_MWT MMUT",
    "013122_cohort4_group7_2mice both_FHET FHET(2)",
]


# !SECTION

# SECTION 3: Run pipeline
for data_folder, animal_ids in tqdm(data_folders_to_animal_ids.items(), desc="Processing data folders"):
    for animal_id in tqdm(animal_ids, desc=f"Processing animals in {data_folder}", leave=False):
        if f"{data_folder} {animal_id}" in bad_folder_animalday:
            logging.warning(f"Skipping {data_folder} {animal_id} because it is in bad_folder_animalday")
            continue
        with Client(cluster_window) as client:
            client.run(lambda: os.system(f"pip install -e {base_folder}"))

            # SECTION 1: Find bin files
            ao = visualization.AnimalOrganizer(
                data_parent_folder / data_folder,
                animal_id,
                mode="nest",
                assume_from_number=True,
                skip_days=["bad"],
                lro_kwargs={"mode": "bin", "multiprocess_mode": "dask", "overwrite_rowbins": False},
            )
            ao.compute_bad_channels()

            # SECTION 2: Make WAR
            war = ao.compute_windowed_analysis(["all"], multiprocess_mode="dask")

            if f"{data_folder} {animal_id}" in bad_channels:
                war = war.filter_reject_channels_by_session(bad_channels[f"{data_folder} {animal_id}"])
            else:
                logging.info(f"No bad channels defined for {data_folder} {animal_id}, skipping filtering")
            # !SECTION

        # SECTION 3: Make SARs, save SARs and load into WAR
        # with Client(cluster_spike) as client:
        #     client.run(lambda: os.system(f"pip install -e {base_folder}"))

        #     sars = ao.compute_spike_analysis(multiprocess_mode='dask')
        #     for sar in sars:
        #         sar.save_fif_and_json(base_folder / 'notebooks' / 'tests' / 'test-sars-full' / f'{data_folder} {slugify(sar.animal_day, allow_unicode=True)}', overwrite=True) # animal_day not unique for Sox5 rec sessions, so add bin_folder_name
        #     war.read_sars_spikes(sars)
        # !SECTION

        # SECTION 4: Save WARs and cleanup
        war.save_pickle_and_json(
            base_folder / "notebooks" / "tests" / "test-wars-sox5-9" / f"{data_folder} {animal_id}"
        )
        del war
        # del sars
        # !SECTION
# !SECTION

"""
sbatch --mem 700G -c 4 -t 48:00:00 /mnt/isilon/marsh_single_unit/NeuRodent/notebooks/examples/pipeline.sh /mnt/isilon/marsh_single_unit/NeuRodent/notebooks/examples/pipeline-war-sox5.py
"""
