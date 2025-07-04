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
load_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-6"
save_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-collapsed-6-isday"
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
owen_name_to_animal_id = {
    "FWT": {
        "D-F3": "071321_Cohort 3_AM4_CF1_DF3_FF6 DF3",
        "FWT_cohort4_group5": "012022_cohort4_group5_3mice__FWT_MMUT_FMUT FWT",
        "B-F3": "062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 BF3",
        "C-F2": "062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 CF2",
        "FWT_cohort4_group4": "011622_cohort4_group4_3mice_MMutOLD_FMUT_FMUT_FWT FWT",
        "C-F1": "071321_Cohort 3_AM4_CF1_DF3_FF6 CF1",
    },
    "FHet": {
        "D-F6": "032221_cohort 2, Group 3, Mouse 6 Cage 2A Re-Recording",
        "FHET(2)_cohort4_group7": "013122_cohort4_group7_2mice both_FHET FHET(2)",
        "F-F6": "071321_Cohort 3_AM4_CF1_DF3_FF6 FF6",
        "G-F4": "060921_Cohort 3_EM1_AM2_GF4 GF4",
        "FHET(1)_cohort4_group7": "013122_cohort4_group7_2mice both_FHET FHET(1)",
        "I-F5": "062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 IF5",
        "A-F5": "031621_cohort 2 group 2 and group 5 F5",
        "FHET_cohort10_group4": "081922_cohort10_group4_2mice_FMut_FHet FHET",
        "D-F8": "031021_cohort 2, group 3 and 4 #8 Cage 1A",
        "A-F6": "031621_cohort 2 group 2 and group 5 F6",
        "F4_FHET_cohort5_group1": "031122_cohort5_group1_2mice_FHET_MWT F4",
    },
    "FMut": {
        "FMUT_cohort4_group4": "011622_cohort4_group4_3mice_MMutOLD_FMUT_FMUT_FWT FMUT_",
        "FMUT_cohort10_group4": "081922_cohort10_group4_2mice_FMut_FHet FMUT",
        "FMUT_cohort4_group5": "012022_cohort4_group5_3mice__FWT_MMUT_FMUT FMUT",
        "FMUT_cohort4_group6": "012322_cohort4_group6_3mice_FMUT___MMUT_MWT FMUT",
        "FMUT(2)_cohort4_group4": "011622_cohort4_group4_3mice_MMutOLD_FMUT_FMUT_FWT FMUT(2)",
        "FMUT(2)_cohort10_group4": "082922_group4_2mice_MMUT_FMUT FMUT",
        "FMUT(3)cohort10_group4": "090122_group4_2mice_FMut_MMut FMUT",
        "C-F10": "031621_cohort 2 group 2 and group 5 F10",
        "F9_FMUT_cohort10_cage3": "062122_group 10_2mice_F7Het_F9Mut F9",
        "C-F9": "040221_Group 1 Mouse 9 Recording",
    },
    "MWT": {
        "M7_MWT_cohort6_group1": "031722_cohort_6_group1_3mice_MMUT_MMUT_MWT M7",
        "M2_group9_cage3": "061022_group 9 M1, M2, M3 group9_M2_Cage 3",
        "C-M5": "062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 CM5",
        "MWT_cohort4_group3": "011322_cohort4_group3_4mice_AllM_MT_WT_HET_WT M4",
        "M10_MWT_cohort4_group2": "010822_cohort4_group2_2mice_MWT_MHET M10",
        "A-M2": "060921_Cohort 3_EM1_AM2_GF4 AM2",
        "A-M4": "071321_Cohort 3_AM4_CF1_DF3_FF6 AM4",
        "M2_MWT_cohort5_group1": "031122_cohort5_group1_2mice_FHET_MWT M2",
        "B-M6": "062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 BM6",
        "C-M9": "062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 CM9",
    },
    "MHet": {
        "M3_MHET_cohort4_group2": "010822_cohort4_group2_2mice_MWT_MHET M3",
        "B-M3": "031921_cohort 2 group 5 and group 6 mouse M3 cage3A",
        "M3_group9_cage4": "061022_group 9 M1, M2, M3 group9_M3_Cage 4",
        "C-M2": "031621_cohort 2 group 2 and group 5 M2",
        "A-M5": "062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 AM5",
        "MHET_cohort4_group6": "012322_cohort4_group6_3mice_FMUT___MMUT_MWT MHET",
        "E-M1": "060921_Cohort 3_EM1_AM2_GF4 EM1",
        "C-M3": "031921_cohort 2 group 5 and group 6 mouse M3 cage1A",
        "MHET_cohort4_group3": "011322_cohort4_group3_4mice_AllM_MT_WT_HET_WT M8",
        "D-M3": "031021_cohort 2, group 3 and 4 #3 Cage 4A",
    },
    "MMut": {
        "MMUT_cohort4_group6": "012322_cohort4_group6_3mice_FMUT___MMUT_MWT MMUT",
        "MMUT_cohort4_group5": "012022_cohort4_group5_3mice__FWT_MMUT_FMUT MMUT",
        "C-M5": "031921_cohort 2 group 5 and group 6 mouse M5 cage2A",
        "MMUT(2)_cohort6_group1": "031722_cohort_6_group1_3mice_MMUT_MMUT_MWT MMUT(2)",
        "MMUT_cohort10_group4": "082922_group4_2mice_MMUT_FMUT MMUT",
        "OLDMMT_cohort4_group4": "011622_cohort4_group4_3mice_MMutOLD_FMUT_FMUT_FWT OLDMMT",
        "MMUT(2)_cohort10_group4": "090122_group4_2mice_FMut_MMut MMUT",
        "D-M5": "031921_cohort 2 group 5 and group 6 mouse M5 cage4A",
        "M8_MMT_group10_cage1": "061322_Group10 M8, M10 M8",
        "MMT_cohort4_group3": "011322_cohort4_group3_4mice_AllM_MT_WT_HET_WT M3",
        "C-M8": "031021_cohort 2, group 3 and 4 #8 Cage 3A",
        "MMUT(1)_cohort6_group1": "031722_cohort_6_group1_3mice_MMUT_MMUT_MWT MMUT(1)",
        "B-M2": "031021_cohort 2, group 3 and 4 #2 Cage 2A",
    },
}
animal_id_to_owen_name = {
    genotype: {v: k for k, v in dictionary.items()} for genotype, dictionary in owen_name_to_animal_id.items()
}
logging.info(animal_id_to_owen_name)

war_bad_channels = {
    "FWT": {
        "C-F2": ["RVis", "RAud"],
        "B-F3": ["RVis", "RAud"],
        "D-F3": ["LAud", "LVis"],
        "FWT_cohort4_group4": ["LAud", "LVis", "LHip", "RHip", "RAud"],
    },
    "FHet": {
        "A-F5": ["LAud", "LVis", "LBar"],
        "A-F6": ["LAud", "LVis", "LBar"],
        "D-F8": ["LAud", "LVis"],
        "G-F4": ["LAud", "LVis"],
        "I-F5": ["LAud", "RVis", "RAud"],
        "F-F6": ["LAud", "LVis", "LBar", "RVis"],
        "FHET(1)_cohort4_group7": ["LAud", "LVis"],
        "FHET(2)_cohort4_group7": ["LAud", "LVis", "RAud"],
        "F4_FHET_cohort5_group1": ["LAud", "LVis"],
    },
    "FMut": {
        "C-F10": ["LAud", "LVis"],
        "C-F9": ["LAud", "LBar", "RAud"],
        "FMUT_cohort4_group6": ["LAud", "LVis", "LHip", "LBar"],
        "FMUT_cohort4_group5": ["LHip", "RHip"],
        "FMUT(2)_cohort4_group4": ["LVis", "LHip", "RHip"],
        "FMUT_cohort4_group4": ["LAud", "LHip", "RBar", "RHip", "RAud"],
        "FMUT_cohort10_group4": ["RAud"],
        "FMUT(3)cohort10_group4": ["RAud"],
        "F9_FMUT_cohort10_cage3": ["RAud"],
    },
    "MWT": {
        "B-M6": ["LAud", "LVis", "RAud"],
        "C-M9": ["LAud", "LVis", "LBar"],
        "C-M5": ["LAud", "LVis", "LHip", "LBar", "LMot", "RMot", "RBar", "RHip", "RVis", "RAud"],
        "A-M4": ["LAud", "LVis", "LBar"],
        "M7_MWT_cohort6_group1": ["LVis", "LBar", "RMot", "RBar", "RHip", "RVis", "RAud"],
        "MWT_cohort4_group3": ["LMot"],
        "M10_MWT_cohort4_group2": ["LAud", "LVis", "LHip", "LBar", "LMot", "RMot", "RBar", "RHip", "RVis", "RAud"],
        "M2_group9_cage3": ["LAud", "LVis", "LHip", "LBar", "LMot", "RMot", "RBar", "RHip", "RVis", "RAud"],
    },
    "MHet": {
        "E-M1": ["LAud", "LVis"],
        "B-M3": ["LAud", "LVis"],
        "C-M3": ["LAud", "LVis"],
        "D-M3": ["LAud", "LVis"],
        "C-M2": ["LAud", "LVis"],
        "MHET_cohort4_group3": ["RAud"],
        "M3_group9_cage4": ["RAud"],
    },
    "MMut": {
        "B-M2": ["LHip"],
        "C-M8": ["LAud", "LVis"],
        "C-M5": ["LAud", "LBar"],
        "D-M5": ["LAud", "LVis", "LHip", "LBar", "LMot", "RMot", "RBar", "RHip", "RVis", "RAud"],
        "MMUT(1)_cohort6_group1": ["LAud", "LVis"],
        "MMUT(2)_cohort6_group1": ["LAud", "LVis", "LHip", "LBar", "LMot", "RMot", "RBar", "RHip", "RVis", "RAud"],
        "OLDMMT_cohort4_group4": ["LAud", "LVis", "LHip", "LBar", "LMot", "RMot", "RBar", "RHip", "RVis", "RAud"],
        "MMT_cohort4_group3": ["LAud", "LVis", "LMot"],
        "MMUT(2)_cohort10_group4": ["LVis"],
        "M8_MMT_group10_cage1": ["LAud", "LVis", "LHip", "LBar", "RAud"],
    },
}

if not save_folder.exists():
    save_folder.mkdir(parents=True)


def load_war(animal_id):
    logger.info(f"Loading {animal_id}")
    war = visualization.WindowAnalysisResult.load_pickle_and_json(Path(load_folder / f"{animal_id}").resolve())
    if war.genotype == "Unknown":  # Remove pathological recordings
        logger.info(f"Skipping {animal_id} because genotype is Unknown")
        return None

    # try:
    #     owen_name = animal_id_to_owen_name[war.genotype][animal_id.strip()]
    # except KeyError as e:
    #     logging.warning(f"{animal_id} not found in animal_id_to_owen_name for {war.genotype}")
    #     logging.error(e)
    #     owen_name = None

    # if owen_name is None:
    #     bad_channels = None
    # else:
    #     try:
    #         bad_channels = war_bad_channels[war.genotype][owen_name]
    #     except KeyError:
    #         logging.warning(f"{animal_id} not found in war_bad_channels for {war.genotype}")
    #         bad_channels = None

    # Ensure LHip and RHip are always in bad_channels
    required_channels = {"LHip", "RHip"}
    # if bad_channels is not None:
    #     bad_channels = list(set(bad_channels) | required_channels)
    # else:
    #     bad_channels = list(required_channels)
    bad_channels = list(required_channels)

    war.filter_all(bad_channels=bad_channels)
    war.aggregate_time_windows(groupby=["animalday", "isday"])  # Not grouping by isday to increase statistical power
    # war.add_unique_hash(4)
    war.reorder_and_pad_channels(
        ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis", "LHip", "RHip"], use_abbrevs=True
    )
    war.save_pickle_and_json(save_folder / f"{animal_id}")

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
# features = ["psdtotal", "psdslope", "logpsdtotal"]
# for feature in features:
#     df = ep.pull_timeseries_dataframe(feature=feature, groupby=["animal", "genotype", "isday"], collapse_channels=True)
#     print(df)
#     df = df.groupby(["animal", "genotype", "isday"])[feature].mean().reset_index()
#     print(df)
#     (
#         so.Plot(df, x="genotype", y=feature, color="isday")
#         .facet(col="isday")
#         .add(so.Dash(color="k"), so.Agg())
#         .add(so.Line(color="k", linestyle="--"), so.Agg())
#         .add(so.Range(color="k"), so.Est(errorbar="sd"))
#         .add(so.Dot(), so.Jitter(seed=42))
#         .add(so.Text(halign="left"), so.Jitter(seed=42), text="animal")
#         .theme(axes_style("ticks"))
#         .layout(size=(20, 5), engine="tight")
#         .save(save_folder / f"{feature}-genotype-isday-avgch.png", dpi=300)
#     )
# !SECTION


"""
sbatch --mem 400GB -c 15 -t 24:00:00 ./notebooks/examples/pipeline.sh ./notebooks/examples/pipeline-epfig-so.py
"""
