import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so
from okabeito import black, blue, green, lightblue, orange, purple, red, yellow
from seaborn import axes_style
from pythoneeg import constants, core, visualization

core.set_temp_directory("/scr1/users/dongjp")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG, stream=sys.stdout, force=True
)
logger = logging.getLogger()

base_folder = Path("/mnt/isilon/marsh_single_unit/PythonEEG")
load_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-8"
save_folder = base_folder / "notebooks" / "examples"

animal_ids = [p.name for p in load_folder.glob("*") if p.is_dir()]

# bad_animal_ids = [
#     "013122_cohort4_group7_2mice both_FHET FHET(2)",
#     "012322_cohort4_group6_3mice_FMUT___MMUT_MWT MHET",
#     "012322_cohort4_group6_3mice_FMUT___MMUT_MWT MMUT",
#     "011622_cohort4_group4_3mice_MMutOLD_FMUT_FMUT_FWT OLDMMT",
#     "011322_cohort4_group3_4mice_AllM_MT_WT_HET_WT M3",
#     "012322_cohort4_group6_3mice_FMUT___MMUT_MWT FHET"
# ]
bad_animal_ids = [
    "011322_cohort4_group3_4mice_AllM_MT_WT_HET_WT M2",
    "011322_cohort4_group3_4mice_AllM_MT_WT_HET_WT M3",
    "011322_cohort4_group3_4mice_AllM_MT_WT_HET_WT M8",
    "011622_cohort4_group4_3mice_MMutOLD_FMUT_FMUT_FWT FMUT(2)",
    "011622_cohort4_group4_3mice_MMutOLD_FMUT_FMUT_FWT OLDMMT",
    "012022_cohort4_group5_3mice__FWT_MMUT_FMUT FMUT",
    "012022_cohort4_group5_3mice__FWT_MMUT_FMUT FWT",
    "012022_cohort4_group5_3mice__FWT_MMUT_FMUT MMUT",
    "012322_cohort4_group6_3mice_FMUT___MMUT_MWT FHET",
    "031021_cohort 2, group 3 and 4 #8 Cage 3A",
    "031122_cohort5_group1_2mice_FHET_MWT F4",
    "031921_cohort 2 group 5 and group 6 mouse M3 cage1A",
    "060921_Cohort 3_EM1_AM2_GF4 GF4",
    "061022_group 9 M1, M2, M3 group9_M1_Cage1",
    "061322_Group10 M8, M10 M8",
    "061722_group10 M3, F8 F8",
    "061722_group10 M3, F8 M3",
    "062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 BM6",
    "121821_cohort4_Group1_2mice M2",
    "121821_cohort4_Group1_2mice M4",
    "060921_Cohort 3_EM1_AM2_GF4 AM2",
]
animal_ids = [p for p in animal_ids if p not in bad_animal_ids]

def load_war(animal_id):
    logger.info(f"Loading {animal_id}")
    war = visualization.WindowAnalysisResult.load_pickle_and_json(Path(load_folder / f"{animal_id}"))
    if war.genotype == "Unknown":
        logger.info(f"Skipping {animal_id} because genotype is Unknown")
        return None

    # war.filter_all(bad_channels=["LHip", "RHip"])
    war.reorder_and_pad_channels(
        ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"], use_abbrevs=True
    )
    # war.filter_all(morphological_smoothing_seconds=60 * 5)
    war.filter_all(morphological_smoothing_seconds=None)

    df = war.get_result(features=["logpsdband", "logrms"])
    df["animal"] = animal_id
    del war
    return df


with Pool(10) as pool:
    dfs: list[pd.DataFrame] = []
    for df in tqdm(pool.imap(load_war, animal_ids), total=len(animal_ids), desc="Loading WARs"):
        if df is not None:
            dfs.append(df)
    df = pd.concat(dfs)

df_bands = pd.DataFrame(df["logpsdband"].tolist())
alpha_array = np.stack(df_bands["alpha"].values)
delta_array = np.stack(df_bands["delta"].values)
df["alphadelta"] = (alpha_array / delta_array).tolist()
df["delta"] = (delta_array).tolist()
df["alpha"] = (alpha_array).tolist()


# Average each feature across channels
for feature in ["alphadelta", "delta", "alpha", "logrms"]:
    feature_arrays = np.stack(df[feature].values)  # Shape: (time_points, channels)
    feature_avg = np.nanmean(feature_arrays, axis=1)  # Average across channels
    df[f"{feature}"] = feature_avg
logging.info(df.columns)

df = df[["timestamp", "animal", "genotype", "alphadelta", "delta", "alpha", "logrms"]]
df["hour"] = df["timestamp"].dt.hour.copy()
df["minute"] = df["timestamp"].dt.minute.copy()
df["total_minutes"] = 60 * (round((df["hour"] * 60 + df["minute"]) / 60) % 24)
logging.info(df.columns)

df = (
    df.groupby(["animal", "genotype", "total_minutes"])
    .agg({"alphadelta": "mean", "delta": "mean", "alpha": "mean", "logrms": "mean"})
    .reset_index()
)
# df = df.set_index("total_minutes")
# df = df.copy()

logging.debug(df)
logging.debug(df.shape)

df.to_pickle(save_folder / "alphadelta_avg_delta_alpha_rms.pkl")
# df.to_pickle(save_folder / "alphadelta_avg_delta_alpha_minimally_filtered.pkl")
# df.to_pickle(save_folder / "alphadelta_avg_delta_alpha_nolog.pkl")


"""
sbatch --mem 200GB -c 11 -t 24:00:00 ./notebooks/examples/pipeline.sh ./notebooks/examples/pipeline-alphadelta.py
"""
