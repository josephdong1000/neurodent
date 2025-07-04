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
import mne
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so
import pingouin as pg
from okabeito import black, blue, green, lightblue, orange, purple, red, yellow
from seaborn import axes_style
from pythoneeg import constants, core, visualization

core.set_temp_directory("/scr1/users/dongjp")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG, stream=sys.stdout, force=True
)
logger = logging.getLogger()

base_folder = Path("/mnt/isilon/marsh_single_unit/PythonEEG")
load_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-6"
save_folder = base_folder / "notebooks" / "examples"

animal_ids = [p.name for p in load_folder.glob("*") if p.is_dir()]


def load_war(animal_id):
    logger.info(f"Loading {animal_id}")
    war = visualization.WindowAnalysisResult.load_pickle_and_json(Path(load_folder / f"{animal_id}").resolve())
    if war.genotype == "Unknown":
        logger.info(f"Skipping {animal_id} because genotype is Unknown")
        return None

    war.filter_all(bad_channels=["LHip", "RHip"])
    war.reorder_and_pad_channels(
        ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis", "LHip", "RHip"], use_abbrevs=True
    )
    war.add_unique_hash()
    df = war.get_result(features=["psdband"])
    del war
    return df


with Pool(10) as pool:
    dfs: list[pd.DataFrame] = []
    for df in tqdm(pool.imap(load_war, animal_ids), total=len(animal_ids), desc="Loading WARs"):
        if df is not None:
            dfs.append(df)
    df = pd.concat(dfs)

df_bands = pd.DataFrame(df["psdband"].tolist())
alpha_array = np.stack(df_bands["alpha"].values)
delta_array = np.stack(df_bands["delta"].values)
df["alphadelta"] = core.log_transform(alpha_array / delta_array).tolist()

alphadelta_arrays = np.stack(df["alphadelta"].values)  # Shape: (time_points, channels)
alphadelta_avg = np.nanmean(alphadelta_arrays, axis=1)  # Average across channels
df["alphadelta_avg"] = alphadelta_avg
logging.info(df.columns)

df = df[["timestamp", "animal", "genotype", "alphadelta_avg"]]
df["hour"] = df["timestamp"].dt.hour.copy()
df["minute"] = df["timestamp"].dt.minute.copy()
df["total_minutes"] = 60 * round((df["hour"] * 60 + df["minute"]) / 60)
logging.info(df.columns)

df = df.groupby(["animal", "genotype", "total_minutes"]).agg({"alphadelta_avg": "mean"}).reset_index()
# df = df.set_index("total_minutes")
# df = df.copy()

logging.debug(df)
logging.debug(df.shape)
logger.setLevel(logging.WARNING)

df.to_pickle(save_folder / "alphadelta_avg.pkl")


"""
sbatch --mem 200GB -c 11 -t 24:00:00 ./notebooks/examples/pipeline.sh ./notebooks/examples/pipeline-alphadelta.py
"""
