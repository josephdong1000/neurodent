import logging
import sys
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from pythoneeg import constants, core, visualization

core.set_temp_directory("/scr1/users/dongjp")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
)
logger = logging.getLogger()

base_folder = Path("/mnt/isilon/marsh_single_unit/PythonEEG").resolve()
load_folder = base_folder / "notebooks" / "tests" / "test-wars-sox5-3"
save_folder = Path("/home/dongjp/Downloads/6-17 APfig").resolve()
if not save_folder.exists():
    save_folder.mkdir(parents=True, exist_ok=True)

# animal_ids = ['A5', 'A10', 'F22', 'G25', 'G26', 'N21', 'N22', 'N23', 'N24', 'N25']
# animal_ids = ['061322_Group10 M8, M10 M8', # normal
#               '062921_Cohort 3_AM3_AM5_CM9_BM6_CM5_CF2_IF5_BF3 BM6', # sparsely disconnected
#               '071321_Cohort 3_AM4_CF1_DF3_FF6 AM4', # normal
#               '081922_cohort10_group4_2mice_FMut_FHet FHET', # intermittently disconnected
#               '090122_group4_2mice_FMut_MMut FMUT' # very disconnected
#              ]
# animal_ids = [p.name for p in load_folder.glob("*") if p.is_dir()]
animal_ids = [
    "012022_cohort4_group5_3mice__FWT_MMUT_FMUT MMUT",
    "031921_cohort 2 group 5 and group 6 mouse M3 cage1A",
    "060921_Cohort 3_EM1_AM2_GF4 AM2",
]

def plot_animal(animal_id):
    war = visualization.WindowAnalysisResult.load_pickle_and_json(load_folder / f"{animal_id}")
    war.filter_all()

    save_path = save_folder / animal_id
    if not save_path.exists():
        save_path.mkdir(parents=True)
    ap = visualization.AnimalPlotter(war, save_fig=True, save_path=save_path / animal_id)

    ap.plot_coherecorr_spectral(figsize=(20, 5), score_type="z")
    ap.plot_psd_histogram(figsize=(10, 4), avg_channels=True, plot_type="loglog")
    ap.plot_psd_spectrogram(figsize=(20, 4), mode="none")

    del war
    return animal_id


with Pool(10) as pool:
    list(tqdm(pool.imap(plot_animal, animal_ids), total=len(animal_ids), desc="Processing animals"))


"""
sbatch --mem 300G -c 11 -t 24:00:00 ./notebooks/examples/pipeline.sh ./notebooks/examples/pipeline-apfig-fromsave.py
"""
