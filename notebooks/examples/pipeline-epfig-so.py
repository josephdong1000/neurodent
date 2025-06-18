import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn.objects as so
from seaborn import axes_style
from tqdm import tqdm

from pythoneeg import constants, core, visualization

core.set_temp_directory("/scr1/users/dongjp")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
)
logger = logging.getLogger()

base_folder = Path("/mnt/isilon/marsh_single_unit/PythonEEG")
load_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-3"
save_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-collapsed-3"
# animal_ids = ['A5', 'A10', 'F22', 'G25', 'G26', 'N21', 'N22', 'N23', 'N24', 'N25']
# animal_ids = ['A5', 'A10']
animal_ids = [p.name for p in load_folder.glob("*") if p.is_dir()]
# bad_animal_ids = [
#     "013122_cohort4_group7_2mice both_FHET FHET(2)",
#     "012322_cohort4_group6_3mice_FMUT___MMUT_MWT MHET",
#     "012322_cohort4_group6_3mice_FMUT___MMUT_MWT MMUT",
#     "011622_cohort4_group4_3mice_MMutOLD_FMUT_FMUT_FWT OLDMMT",
#     "011322_cohort4_group3_4mice_AllM_MT_WT_HET_WT M3",
# ]
bad_animal_ids = []
animal_ids = [animal_id for animal_id in animal_ids if animal_id not in bad_animal_ids]
if not save_folder.exists():
    save_folder.mkdir(parents=True)


def load_war(animal_id):
    logger.info(f"Loading {animal_id}")
    war = visualization.WindowAnalysisResult.load_pickle_and_json(Path(load_folder / f"{animal_id}").resolve())
    if war.genotype == "Unknown":  # Remove pathological recordings
        logger.info(f"Skipping {animal_id} because genotype is Unknown")
        return None

    war.filter_all()
    war.aggregate_time_windows()
    war.add_unique_hash(4)
    war.reorder_and_pad_channels(
        ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis", "LHip", "RHip"], use_abbrevs=True
    )
    war.save_pickle_and_json(save_folder / f"{animal_id} {war.animal_id}")

    return war


# Use multiprocessing to load WARs in parallel
with Pool(14) as pool:
    wars: list[visualization.WindowAnalysisResult] = []
    for war in tqdm(pool.imap(load_war, animal_ids), total=len(animal_ids), desc="Loading WARs"):
        if war is not None:
            wars.append(war)

logger.info(f"{len(wars)} wars loaded")
exclude = ["nspike", "lognspike"]
ep = visualization.ExperimentPlotter(wars, exclude=exclude)


# SECTION use seaborn.so to plot figure
features = ["psdtotal", "psdslope", "logpsdtotal"]
for feature in features:
    df = ep.pull_timeseries_dataframe(feature=feature, groupby=["animal", "genotype", "isday"], collapse_channels=True)
    print(df)
    df = df.groupby(["animal", "genotype", "isday"])[feature].mean().reset_index()
    print(df)
    (
        so.Plot(df, x="genotype", y=feature, color="isday")
        .facet(col="isday")
        .add(so.Dash(color="k"), so.Agg())
        .add(so.Line(color="k", linestyle="--"), so.Agg())
        .add(so.Range(color="k"), so.Est(errorbar="sd"))
        .add(so.Dot(), so.Jitter(seed=42))
        .add(so.Text(halign="left"), so.Jitter(seed=42), text="animal")
        .theme(axes_style("ticks"))
        .layout(size=(20, 5), engine="tight")
        .save(save_folder / f"{feature}-genotype-isday-avgch.png", dpi=300)
    )
# !SECTION


"""
sbatch --mem 200GB -c 15 -t 24:00:00 ./notebooks/examples/pipeline.sh ./notebooks/examples/pipeline-epfig-so.py
"""
