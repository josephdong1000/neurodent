import sys
from pathlib import Path
import time
import tempfile
import logging
import json
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mne
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from django.utils.text import slugify
import joblib

base_folder = Path("/mnt/isilon/marsh_single_unit/PythonEEG")
sys.path.append(str(base_folder))

from pythoneeg import core
from pythoneeg import visualization
from pythoneeg import constants

core.set_temp_directory("/scr1/users/dongjp")


# SECTION 1: Set up clusters
try:
    cluster_window = SLURMCluster(
        cores=30,
        memory="100GB",
        walltime="48:00:00",
        interface=None,
        job_extra_directives=["--output=/dev/null", "--error=/dev/null"],
    )
except ValueError as e:
    if "interface 'eth1' doesn't have an IPv4 address" in str(e):
        cluster_window = SLURMCluster(
            cores=30,
            memory="100GB",
            walltime="48:00:00",
            interface=None,
            scheduler_options={"interface": "eth1"},
            job_extra_directives=["--output=/dev/null", "--error=/dev/null"],
        )
    else:
        raise
print(f"\n\n\tcluster_window.dashboard_link: {cluster_window.dashboard_link}\n\n")
cluster_window.scale(jobs=10)

# SECTION 2: Setup windowed analysis
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
)
logger = logging.getLogger()


from mne.io import read_raw_edf

data_folder = Path("/mnt/isilon/marsh_single_unit/PythonEEG Data Bins/Arx Rosa")
save_folder = Path("/home/dongjp/Downloads/8-13-25 war from edf")

if not save_folder.exists():
    save_folder.mkdir(parents=True)

# with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=-1):
# with joblib.parallel_config(backend="loky", n_jobs=-1):
# with joblib.parallel_config(backend="threading", n_jobs=-1):
#     # with joblib.parallel_config(backend="loky", n_jobs=-1, prefer="threads", inner_max_num_threads=3):
#     # with joblib.parallel_config(n_jobs=-1, prefer="threads"):
#     lrec = core.LongRecordingOrganizer(
#         base_folder_path=data_folder,
#         truncate=5,
#         mode="mne",
#         extract_func=read_raw_edf,
#         file_pattern="*.EDF",
#         input_type="files",
#         intermediate="bin",
#         manual_datetimes=datetime(2015, 2, 25, 8, 37, 45),
#     )


# with Client(cluster_window) as client:

#     lrec = core.LongRecordingOrganizer(
#         base_folder_path=data_folder,
#         truncate=5,
#         mode="mne",
#         extract_func=read_raw_edf,
#         file_pattern="*.EDF",
#         input_type="files",
#         intermediate="bin",
#         manual_datetimes=datetime(2015, 2, 25, 8, 37, 45),
#         multiprocess_mode="dask"
#     )

# lrec = core.LongRecordingOrganizer(
#     base_folder_path=data_folder / "Arx Rosa 1017 1015 20150224",
#     truncate=2,
#     mode="mne",
#     extract_func=read_raw_edf,
#     file_pattern="*.EDF",
#     input_type="files",
#     intermediate="bin",
#     manual_datetimes=datetime(2015, 2, 25, 8, 37, 45),
# )
# print(f"\n\n\tlrec: {lrec}\n\n")

constants.GENOTYPE_ALIASES = {"ARX": "ARX"}
constants.LR_ALIASES = {
    "L": [f"E{i}-" for i in range(1, 32, 2)],  # Odd numbered electrodes (1,3,5,...,31)
    "R": [f"E{i}-" for i in range(2, 33, 2)],  # Even numbered electrodes (2,4,6,...,32)
}
constants.CHNAME_ALIASES = {
    "Vis": ["E1-", "E2-"] + ["E9-", "E10-"] + ["E17-", "E18-"] + ["E25-", "E26-"],
    "Hip": ["E3-", "E4-"] + ["E11-", "E12-"] + ["E19-", "E20-"] + ["E27-", "E28-"],
    "Bar": ["E5-", "E6-"] + ["E13-", "E14-"] + ["E21-", "E22-"] + ["E29-", "E30-"],
    "Mot": ["E7-", "E8-"] + ["E15-", "E16-"] + ["E23-", "E24-"] + ["E31-", "E32-"],
}

for animal_id in ["1017 1015"]:
    with Client(cluster_window) as client:
        ao = visualization.AnimalOrganizer(
            data_folder,
            animal_id,
            mode="concat",
            # assume_from_number=True,
            skip_days=["bad"],
            lro_kwargs={
                # "truncate": 2,  # REVIEW put back when testing
                # REVIEW when the truncate number changes, should the user be notified and be advised to force regenerat
                "mode": "mne",
                "extract_func": read_raw_edf,
                "file_pattern": "*.EDF",
                "input_type": "files",
                "intermediate": "bin",
                "cache_policy": "force_regenerate",
                # "cache_policy": "force_regenerate",
                "manual_datetimes": datetime(2015, 2, 25, 8, 37, 45),
            },
        )

        war = ao.compute_windowed_analysis(["all"], multiprocess_mode="dask")

    save_path = save_folder / f"{animal_id}"
    if not save_path.exists():
        save_path.mkdir(parents=True)
    ap = visualization.AnimalPlotter(war, save_fig=True, save_path=save_path / animal_id)

    ap.plot_coherecorr_spectral(figsize=(20, 5), score_type="z")
    ap.plot_psd_histogram(figsize=(10, 4), avg_channels=True, plot_type="loglog")
    ap.plot_psd_spectrogram(figsize=(20, 4), mode="none")

    ap.plot_coherecorr_matrix(figsize=(12, 4))
    ap.plot_linear_temporal(features=["rms", "ampvar", "psdtotal", "psdband", "psdfrac"], figsize=(20, 40))

# with open(base_folder / "notebooks" / "tests" / "sox5 combine genotypes.json", "r") as f:
#     data = json.load(f)
# data_parent_folder = Path(data["data_parent_folder"])
# constants.GENOTYPE_ALIASES = data["GENOTYPE_ALIASES"]
# data_folders_to_animal_ids = data["data_folders_to_animal_ids"]


"""
sbatch --mem 100G -c 4 -t 48:00:00 /mnt/isilon/marsh_single_unit/PythonEEG/notebooks/examples/pipeline.sh /mnt/isilon/marsh_single_unit/PythonEEG/notebooks/examples/pipeline-war-ARXRosa.py
"""
