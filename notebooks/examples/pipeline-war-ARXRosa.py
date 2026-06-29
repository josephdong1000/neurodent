import sys
from pathlib import Path
import logging
from datetime import datetime

import matplotlib.pyplot as plt
from mne.io import read_raw_edf
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

base_folder = Path("/mnt/isilon/marsh_single_unit/NeuRodent")
sys.path.append(str(base_folder))

from neurodent import core, visualization, constants

core.set_temp_directory("/scr1/users/dongjp")


# SECTION 1: Set up clusters
# try:
#     cluster_window = SLURMCluster(
#         cores=30,
#         memory="100GB",
#         walltime="48:00:00",
#         interface="eth1",
#         job_extra_directives=["--output=/dev/null", "--error=/dev/null"],
#     )
# except ValueError as e:
#     if "interface 'eth1' doesn't have an IPv4 address" in str(e):
#         cluster_window = SLURMCluster(
#             cores=30,
#             memory="100GB",
#             walltime="48:00:00",
#             interface=None,
#             scheduler_options={"interface": "eth1"},
#             job_extra_directives=["--output=/dev/null", "--error=/dev/null"],
#         )
#     else:
#         raise
cluster_window = SLURMCluster(
    cores=30,
    memory="100GB",
    walltime="48:00:00",
    interface=None,
    scheduler_options={"interface": "eth1"},
    job_extra_directives=["--output=/dev/null", "--error=/dev/null"],
)
print(f"\n\n\tcluster_window.dashboard_link: {cluster_window.dashboard_link}\n\n")
cluster_window.scale(jobs=10)

# SECTION 2: Setup windowed analysis
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG, stream=sys.stdout, force=True
)


data_folder = Path("/mnt/isilon/marsh_single_unit/PythonEEG Data Bins/Arx Rosa")
save_folder = Path("/home/dongjp/Downloads/9-19-25 war from edf")

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
# Exact channel map (matches config/datasets/arx_rosa.yaml, the corrected Pre-Intan montage:
# Motor / Barrel / Hippo / Visual, odd electrode = Left). Channel resolution is exact.
constants.set_channel_map({
    "LMot": ["EEG E1-REF1", "EEG E9-REF2", "EEG E17-REF3", "EEG E25-REF4"],
    "RMot": ["EEG E2-REF1", "EEG E10-REF2", "EEG E18-REF3", "EEG E26-REF4"],
    "LBar": ["EEG E3-REF1", "EEG E11-REF2", "EEG E19-REF3", "EEG E27-REF4"],
    "RBar": ["EEG E4-REF1", "EEG E12-REF2", "EEG E20-REF3", "EEG E28-REF4"],
    "LHip": ["EEG E5-REF1", "EEG E13-REF2", "EEG E21-REF3", "EEG E29-REF4"],
    "RHip": ["EEG E6-REF1", "EEG E14-REF2", "EEG E22-REF3", "EEG E30-REF4"],
    "LVis": ["EEG E7-REF1", "EEG E15-REF2", "EEG E23-REF3", "EEG E31-REF4"],
    "RVis": ["EEG E8-REF1", "EEG E16-REF2", "EEG E24-REF3", "EEG E32-REF4"],
})

for animal_id in ["1017 1015"]:
    with Client(cluster_window) as client:
        ao = visualization.AnimalOrganizer(
            data_folder,
            animal_id,
            mode="nest",
            skip_days=["bad", "band data"],
            lro_kwargs={
                "mode": "mne",
                "extract_func": read_raw_edf,
                "file_pattern": "*.EDF",
                "input_type": "files",
                "intermediate": "bin",
                # "cache_policy": "force_regenerate", # Use auto mode
                "manual_datetimes": datetime(2015, 2, 25, 8, 37, 45),
            },
        )

        ao.compute_bad_channels()
        war = ao.compute_windowed_analysis(["all"], multiprocess_mode="dask")

        save_path = save_folder / f"{animal_id}"
        if not save_path.exists():
            save_path.mkdir(parents=True)
        war.save_pickle_and_json(save_path, filename="war")

    # ap = visualization.AnimalPlotter(war, save_fig=True, save_path=save_path / animal_id)

    # ap.plot_coherecorr_spectral(figsize=(20, 5), score_type="z")
    # ap.plot_psd_histogram(figsize=(10, 4), avg_channels=True, plot_type="loglog")
    # ap.plot_psd_spectrogram(figsize=(20, 4), mode="none")

    # ap.plot_coherecorr_matrix(figsize=(12, 4))
    # ap.plot_linear_temporal(features=["rms", "ampvar", "psdtotal", "psdband", "psdfrac"], figsize=(20, 40))

# with open(base_folder / "notebooks" / "tests" / "sox5 combine genotypes.json", "r") as f:
#     data = json.load(f)
# data_parent_folder = Path(data["data_parent_folder"])
# constants.GENOTYPE_ALIASES = data["GENOTYPE_ALIASES"]


"""
sbatch --mem 300G -c 4 -t 48:00:00 /mnt/isilon/marsh_single_unit/NeuRodent/notebooks/examples/pipeline.sh /mnt/isilon/marsh_single_unit/NeuRodent/notebooks/examples/pipeline-war-ARXRosa.py
"""
