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

base_folder = Path("/mnt/isilon/marsh_single_unit/Neurodent")
sys.path.append(str(base_folder))

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
cluster_window.scale(jobs=10)
# !SECTION

# SECTION 2: Setup windowed analysis
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG, stream=sys.stdout, force=True
)
logger = logging.getLogger()

with open(base_folder / "notebooks" / "tests" / "sox5 combine genotypes.json", "r") as f:
    data = json.load(f)
data_parent_folder = Path(data["data_parent_folder"])
constants.GENOTYPE_ALIASES = data["GENOTYPE_ALIASES"]
data_folders_to_animal_ids = data["data_folders_to_animal_ids"]

save_folder = Path("/home/dongjp/Downloads/5-19-25 apfig sox5").resolve()
if not save_folder.exists():
    save_folder.mkdir(parents=True)
# !SECTION

# SECTION 3: Run and plot windowed analysis

for data_folder, animal_ids in data_folders_to_animal_ids.items():
    for animal_id in animal_ids:
        with Client(cluster_window) as client:
            client.upload_file(str(base_folder / "neurodent.tar.gz"))

            # SECTION 1: Find bin files
            ao = visualization.AnimalOrganizer(
                data_parent_folder / data_folder,
                animal_id,
                mode="nest",
                assume_from_number=True,
                skip_days=["bad"],
                lro_kwargs={"mode": "bin", "multiprocess_mode": "dask", "overwrite_rowbins": True},
            )
            # !SECTION

            # SECTION 2: Make WAR
            war = ao.compute_windowed_analysis(["all"], multiprocess_mode="dask")
            # !SECTION

        # SECTION 3: load into AP
        save_path = save_folder / f"{data_folder} {animal_id}"
        if not save_path.exists():
            save_path.mkdir(parents=True)
        ap = visualization.AnimalPlotter(war, save_fig=True, save_path=save_path / animal_id)
        # !SECTION

        # SECTION 4: plot
        ap.plot_coherecorr_spectral(figsize=(20, 5), score_type="z")
        ap.plot_psd_histogram(figsize=(10, 4), avg_channels=True, plot_type="loglog")
        ap.plot_psd_spectrogram(figsize=(20, 4), mode="none")

        del war
        # !SECTION
# !SECTION

"""
sbatch --mem 300G -c 4 -t 48:00:00 /mnt/isilon/marsh_single_unit/Neurodent/notebooks/examples/pipeline.sh /mnt/isilon/marsh_single_unit/Neurodent/notebooks/examples/pipeline-apfig-sox5.py
"""
