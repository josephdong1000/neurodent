import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
from tqdm import tqdm

from pythoneeg import constants, core, visualization

core.set_temp_directory("/scr1/users/dongjp")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
)
logger = logging.getLogger()

base_folder = Path("/mnt/isilon/marsh_single_unit/PythonEEG")
load_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-7"
save_folder = Path("/home/dongjp/Downloads/7-30 artifact figs 4 zscore")
if not save_folder.exists():
    save_folder.mkdir(parents=True, exist_ok=True)

animal_ids = [p.name for p in load_folder.glob("*") if p.is_dir()]
bad_animal_ids = [
    "013122_cohort4_group7_2mice both_FHET FHET(2)",
    "012322_cohort4_group6_3mice_FMUT___MMUT_MWT MHET",
    "012322_cohort4_group6_3mice_FMUT___MMUT_MWT MMUT",
    "011622_cohort4_group4_3mice_MMutOLD_FMUT_FMUT_FWT OLDMMT",
    "011322_cohort4_group3_4mice_AllM_MT_WT_HET_WT M3",
    "012322_cohort4_group6_3mice_FMUT___MMUT_MWT FHET",
]
animal_ids = [p for p in animal_ids if p not in bad_animal_ids]

def plot_animal(animal_id):
    logger.info(f"Plotting {animal_id}")
    war = visualization.WindowAnalysisResult.load_pickle_and_json(load_folder / f"{animal_id}")
    if war.genotype == "Unknown":
        logger.info(f"Skipping {animal_id} because genotype is Unknown")
        return None

    # save_path = save_folder / animal_id
    # if not save_path.exists():
    #     save_path.mkdir(parents=True)
    save_path = save_folder

    ap = visualization.AnimalPlotter(war, save_fig=True, save_path=save_path / f"{animal_id}")
    # ap.plot_temporal_heatmap(features="rms", figsize=(10, 3), norm=matplotlib.colors.Normalize(vmin=50, vmax=300))
    ap.plot_temporal_heatmap(
        features="rms",
        figsize=(10, 3),
        cmap="seismic",
        score_type="z",
        norm=matplotlib.colors.CenteredNorm(halfrange=2),
    )

    # ap.plot_temporal_heatmap(
    #     features="psdtotal", figsize=(10, 3), norm=matplotlib.colors.Normalize(vmin=50**2, vmax=500**2)
    # )
    # ap.plot_temporal_heatmap(features="zcohere", figsize=(10, 5), norm=matplotlib.colors.Normalize(vmin=0, vmax=1.5))
    # ap.plot_temporal_heatmap(
    #     features="zpcorr", figsize=(10, 3), norm=matplotlib.colors.Normalize(vmin=-1.5, vmax=1.5), cmap="RdBu_r"
    # )
    ap.plot_temporal_heatmap(
        features="zpcorr", figsize=(10, 3), cmap="RdBu_r", score_type="z", norm=matplotlib.colors.CenteredNorm()
    )

    # Filter using default features
    war = war.filter_all()
    ap = visualization.AnimalPlotter(war, save_fig=True, save_path=save_path / f"{animal_id}_filtered")
    # ap.plot_temporal_heatmap(features="rms", figsize=(10, 3), norm=matplotlib.colors.Normalize(vmin=50, vmax=300))
    # ap.plot_temporal_heatmap(
    #     features="zpcorr", figsize=(10, 3), norm=matplotlib.colors.Normalize(vmin=-1.5, vmax=1.5), cmap="RdBu_r"
    # )
    ap.plot_temporal_heatmap(
        features="rms",
        figsize=(10, 3),
        cmap="seismic",
        score_type="z",
        norm=matplotlib.colors.CenteredNorm(halfrange=2),
    )
    ap.plot_temporal_heatmap(
        features="zpcorr", figsize=(10, 3), cmap="RdBu_r", score_type="z", norm=matplotlib.colors.CenteredNorm()
    )

    return animal_id


with Pool(10) as pool:
    list(tqdm(pool.imap(plot_animal, animal_ids), total=len(animal_ids), desc="Processing animals"))


"""
sbatch --mem 300G -c 11 -t 24:00:00 ./notebooks/examples/pipeline.sh ./notebooks/examples/pipeline-apfig-artifact.py
"""
