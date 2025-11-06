import logging
import sys
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from neurodent import core, visualization

core.set_temp_directory("/scr1/users/dongjp")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
)
logger = logging.getLogger()

base_folder = Path("/mnt/isilon/marsh_single_unit/NeuRodent")
load_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-9"
save_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-collapsed-9-isday"

animal_ids = [p.name for p in load_folder.glob("*") if p.is_dir()]

if not save_folder.exists():
    save_folder.mkdir(parents=True)


def load_war(animal_id):
    logger.info(f"Loading {animal_id}")
    war = visualization.WindowAnalysisResult.load_pickle_and_json(Path(load_folder / f"{animal_id}").resolve())
    if war.genotype == "Unknown":
        logger.info(f"Skipping {animal_id} because genotype is Unknown")
        return None

    bad_channels = ["LHip", "RHip"]

    war.filter_all(bad_channels=bad_channels)
    # war.filter_all(bad_channels=bad_channels, morphological_smoothing_seconds=60 * 5)
    war.aggregate_time_windows(groupby=["animalday", "isday"])

    war.reorder_and_pad_channels(["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis"], use_abbrevs=True)
    war.save_pickle_and_json(save_folder / f"{animal_id}")

    return war


with Pool(14) as pool:
    wars: list[visualization.WindowAnalysisResult] = []
    for war in tqdm(pool.imap(load_war, animal_ids), total=len(animal_ids), desc="Loading WARs"):
        if war is not None:
            wars.append(war)

logger.info(f"{len(wars)} wars loaded")
ep = visualization.ExperimentPlotter(wars)


"""
sbatch --mem 400GB -c 15 -t 24:00:00 ./notebooks/examples/pipeline.sh ./notebooks/examples/pipeline-epfig-so.py
"""
