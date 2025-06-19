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
save_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-collapsed-4"
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
war_bad_channels = {
    # 'FHet' : {
    #     'A-F5' : ['LAud', 'LVis', 'LBar'],
    #     'A-F6' : ['LAud', 'LVis', 'LBar'],
    #     'D-F8' : ['LAud', 'LVis'],
    #     'G-F4' : ['LAud', 'LVis'],
    # }
    'FWT' : {
        'C-F2' : ['RVis', 'RAud'],
        'B-F3' : ['RVis', 'RAud'],
        'D-F3' : ['LAud', 'LVis'],
        'FWT_cohort4_group4' : ['LAud', 'LVis', 'LHip', 'RHip', 'RAud']
    },
    'FHet' : {
        'A-F5' : ['LAud', 'LVis', 'LBar'],
        'A-F6' : ['LAud', 'LVis', 'LBar'],
        'D-F8': ['LAud', 'LVis'],
        'G-F4': ['LAud', 'LVis'],
        'I-F5' : ['LAud', 'RVis', 'RAud'],
        'F-F6' : ['LAud', 'LVis', 'LBar', 'RVis'],
        'FHET(1)_cohort4_group7' : ['LAud', 'LVis'],
        'FHET(2)_cohort4_group7' : ['LAud', 'LVis', 'RAud'],
        'F4_FHET_cohort5_group1' : ['LAud', 'LVis'],
    },
    'FMut' : {
        'C-F10': ['LAud', 'LVis'],
        'C-F9': ['LAud', 'LBar', 'RAud'],
        'FMUT_cohort4_group6': ['LAud', 'LVis', 'LHip', 'LBar'],
        'FMUT_cohort4_group5': ['LHip', 'RHip'],
        'FMUT(2)_cohort4_group4': ['LVis', 'LHip', 'RHip'],
        'FMUT_cohort4_group4': ['LAud', 'LHip', 'RBar', 'RHip', 'RAud'],
        'FMUT_cohort10_group4': ['RAud'],
        'FMUT(3)cohort10_group4': ['RAud'],
        'F9_FMUT_cohort10_cage3': ['RAud']
    },
    'MWT': {
        'B-M6': ['LAud', 'LVis', 'RAud'],
        'C-M9': ['LAud', 'LVis', 'LBar'],
        'C-M5': ['LAud', 'LVis', 'LHip', 'LBar', 'LMot', 'RMot', 'RBar', 'RHip', 'RVis', 'RAud'],
        'A-M4': ['LAud', 'LVis', 'LBar'],
        'M7_MWT_cohort6_group1': ['LVis', 'LBar', 'RMot', 'RBar', 'RHip', 'RVis', 'RAud'],
        'MWT_cohort4_group3': ['LMot'],
        'M10_MWT_cohort4_group2': ['LAud', 'LVis', 'LHip', 'LBar', 'LMot', 'RMot', 'RBar', 'RHip', 'RVis', 'RAud'],
        'M2_group9_cage3': ['LAud', 'LVis', 'LHip', 'LBar', 'LMot', 'RMot', 'RBar', 'RHip', 'RVis', 'RAud']
    },
    'MHet': {
        'E-M1': ['LAud', 'LVis'],
        'B-M3': ['LAud', 'LVis'],
        'C-M3': ['LAud', 'LVis'],
        'D-M3': ['LAud', 'LVis'],
        'C-M2': ['LAud', 'LVis'],
        'MHET_cohort4_group3': ['RAud'],
        'M3_group9_cage4': ['RAud']
    },
    'MMut':{
        'B-M2': ['LHip'],
        'C-M8': ['LAud', 'LVis'],
        'C-M5': ['LAud', 'LBar'],
        'D-M5': ['LAud', 'LVis', 'LHip', 'LBar', 'LMot', 'RMot', 'RBar', 'RHip', 'RVis', 'RAud'],
        'MMUT(1)_cohort6_group1': ['LAud', 'LVis'],
        'MMUT(2)_cohort6_group1': ['LAud', 'LVis', 'LHip', 'LBar', 'LMot', 'RMot', 'RBar', 'RHip', 'RVis', 'RAud'],
        'OLDMMT_cohort4_group4': ['LAud', 'LVis', 'LHip', 'LBar', 'LMot', 'RMot', 'RBar', 'RHip', 'RVis', 'RAud'],
        'MMT_cohort4_group3': ['LAud', 'LVis', 'LMot'],
        'MMUT(2)_cohort10_group4': ['LVis'],
        'M8_MMT_group10_cage1': ['LAud', 'LVis', 'LHip', 'LBar', 'RAud']
    }
}

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
