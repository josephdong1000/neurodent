import json
import logging
import random
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import spikeinterface.preprocessing as spre

from pythoneeg import constants, core, visualization

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout, force=True
)
logger = logging.getLogger()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
base_folder = Path("/mnt/isilon/marsh_single_unit/PythonEEG")
genotype_json_path = base_folder / "notebooks" / "tests" / "sox5 combine genotypes.json"
save_folder = base_folder / "notebooks" / "examples" / "eeg_snippets_plots"
save_folder.mkdir(parents=True, exist_ok=True)

# Genotypes to process
GENOTYPES = ["MWT", "MHet", "MMut", "FWT", "FHet", "FMut"]
SNIPPETS_PER_ANIMAL = 3
ANIMALS_PER_GENOTYPE = 5
SNIPPET_DURATION = 5.0  # seconds
SAMPLING_RATE = constants.GLOBAL_SAMPLING_RATE  # Use global sampling rate from constants

# Channel order for consistent plotting
CHANNEL_ORDER = ["LMot", "RMot", "LBar", "RBar", "LAud", "RAud", "LVis", "RVis", "LHip", "RHip"]


def load_genotype_mapping() -> Tuple[Dict[str, List[str]], Path]:
    """Load genotype mapping from JSON file and set constants."""
    with open(genotype_json_path, "r") as f:
        genotype_data = json.load(f)

    data_parent_folder = Path(genotype_data["data_parent_folder"])
    genotype_aliases = genotype_data["GENOTYPE_ALIASES"]
    data_folders_to_animal_ids = genotype_data["data_folders_to_animal_ids"]

    # IMPORTANT: Set the constants.GENOTYPE_ALIASES so AnimalOrganizer can find genotypes
    constants.GENOTYPE_ALIASES = genotype_aliases
    logger.info("Updated constants.GENOTYPE_ALIASES with Sox5 genotype mappings")

    # Create reverse mapping from animal ID to genotype
    animal_to_genotype = {}
    for genotype, aliases in genotype_aliases.items():
        for alias in aliases:
            animal_to_genotype[alias] = genotype

    # Build genotype to animals mapping with folder context
    genotype_to_animals = {genotype: [] for genotype in GENOTYPES}

    for data_folder, animal_ids in data_folders_to_animal_ids.items():
        for animal_id in animal_ids:
            if animal_id in animal_to_genotype:
                genotype = animal_to_genotype[animal_id]
                if genotype in GENOTYPES:  # Only include target genotypes
                    genotype_to_animals[genotype].append((animal_id, data_folder))

    # Log the mapping
    for genotype, animals in genotype_to_animals.items():
        logger.info(
            f"{genotype}: {len(animals)} animals found - {[a[0] for a in animals[:5]]}{'...' if len(animals) > 5 else ''}"
        )

    return genotype_to_animals, data_parent_folder


def load_and_sample_animal_data(animal_info: Tuple[str, str, str, Path]) -> Dict:
    """Load animal data and extract random snippets.

    Args:
        animal_info: Tuple of (animal_id, genotype, data_folder, data_parent_folder)

    Returns:
        Dictionary containing snippets and metadata
    """
    animal_id, genotype, data_folder, data_parent_folder = animal_info

    try:
        logger.info(f"Processing {animal_id} ({genotype})")

        # Create AnimalOrganizer with error handling
        try:
            ao = visualization.AnimalOrganizer(
                data_parent_folder / data_folder,
                animal_id,
                mode="nest",
                assume_from_number=True,
                skip_days=["bad"],
                # truncate=3,
                lro_kwargs={
                    "mode": "bin",
                    "multiprocess_mode": "serial",  # Use serial for individual animals
                    "overwrite_rowbins": False,
                    "truncate": 5,
                },
            )
        except Exception as e:
            logger.error(f"Failed to create AnimalOrganizer for {animal_id}: {str(e)}")
            return {
                "animal_id": animal_id,
                "genotype": genotype,
                "snippets": [],
                "success": False,
                "error": f"Failed to create AnimalOrganizer: {str(e)}",
            }

        snippets = []

        # Process each long recording
        for lro_idx, lro in enumerate(ao.long_recordings):
            try:
                # Check total duration
                total_duration = 0
                valid_recs = []
                rec = lro.LongRecording
                rec = spre.notch_filter(rec, freq=60, q=100)
                try:
                    duration = rec.get_duration()
                    if duration > 0:
                        total_duration += duration
                        valid_recs.append(rec)
                except Exception as e:
                    logger.warning(f"Failed to get duration for {animal_id} LRO {lro_idx} rec: {str(e)}")
                    continue

                if total_duration < SNIPPET_DURATION:
                    logger.warning(f"Recording too short for {animal_id} LRO {lro_idx}: {total_duration:.2f}s")
                    continue

                if not valid_recs:
                    logger.warning(f"No valid recordings for {animal_id} LRO {lro_idx}")
                    continue

                # Extract snippets from this long recording (use valid_recs instead of lro.recs)
                recording_snippets = extract_snippets_from_lro_safe(valid_recs, animal_id, genotype, lro_idx)
                snippets.extend(recording_snippets)

                # Stop if we have enough snippets
                if len(snippets) >= SNIPPETS_PER_ANIMAL:
                    break
            except Exception as e:
                logger.warning(f"Error processing LRO {lro_idx} for {animal_id}: {str(e)}")
                continue

        # Randomly sample the requested number of snippets
        if len(snippets) > SNIPPETS_PER_ANIMAL:
            snippets = random.sample(snippets, SNIPPETS_PER_ANIMAL)

        return {"animal_id": animal_id, "genotype": genotype, "snippets": snippets, "success": True}

    except Exception as e:
        logger.error(f"Error processing {animal_id}: {str(e)}")
        return {"animal_id": animal_id, "genotype": genotype, "snippets": [], "success": False, "error": str(e)}


def extract_snippets_from_lro_safe(valid_recs: List, animal_id: str, genotype: str, lro_idx: int) -> List[Dict]:
    """Extract random snippets from a list of valid recordings."""
    snippets = []

    for rec_idx, rec in enumerate(valid_recs):
        try:
            duration = rec.get_duration()
            if duration < SNIPPET_DURATION:
                continue

            # Calculate number of samples for snippet
            snippet_samples = int(SNIPPET_DURATION * rec.sampling_frequency)
            max_start_sample = rec.get_num_samples() - snippet_samples

            if max_start_sample <= 0:
                continue

            # Extract a random snippet
            start_sample = random.randint(0, max_start_sample)
            end_sample = start_sample + snippet_samples

            # Get the data with error handling
            try:
                data = rec.get_traces(start_frame=start_sample, end_frame=end_sample, return_scaled=True)
                if data is None or data.size == 0:
                    logger.warning(f"Empty data for {animal_id} LRO {lro_idx} rec {rec_idx}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to get traces for {animal_id} LRO {lro_idx} rec {rec_idx}: {str(e)}")
                continue

            # Get channel names
            try:
                channel_names = list(rec.channel_ids)
                if not channel_names:
                    logger.warning(f"No channel names for {animal_id} LRO {lro_idx} rec {rec_idx}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to get channel names for {animal_id} LRO {lro_idx} rec {rec_idx}: {str(e)}")
                continue

            # Create time axis
            time_axis = np.arange(data.shape[0]) / rec.sampling_frequency

            snippet_info = {
                "animal_id": animal_id,
                "genotype": genotype,
                "lro_idx": lro_idx,
                "rec_idx": rec_idx,
                "data": data,
                "time_axis": time_axis,
                "channel_names": channel_names,
                "sampling_frequency": rec.sampling_frequency,
                "start_sample": start_sample,
                "duration": SNIPPET_DURATION,
            }

            snippets.append(snippet_info)
            logger.debug(f"Successfully extracted snippet from {animal_id} LRO {lro_idx} rec {rec_idx}")

        except Exception as e:
            logger.warning(f"Error extracting snippet from {animal_id} LRO {lro_idx} rec {rec_idx}: {str(e)}")
            continue

    return snippets


def plot_snippets(all_snippets: List[Dict], save_folder: Path):
    """Create plots for all snippets organized by genotype."""

    # Organize snippets by genotype
    genotype_snippets = {}
    for result in all_snippets:
        if result["success"] and result["snippets"]:
            genotype = result["genotype"]
            if genotype not in genotype_snippets:
                genotype_snippets[genotype] = []
            genotype_snippets[genotype].extend(result["snippets"])

    # Plot each genotype with both color schemes
    for genotype, snippets in genotype_snippets.items():
        if not snippets:
            continue

        logger.info(f"Plotting {len(snippets)} snippets for {genotype}")
        plot_genotype_snippets(genotype, snippets, save_folder, color_scheme="color")
        plot_genotype_snippets(genotype, snippets, save_folder, color_scheme="black")


def plot_genotype_snippets(genotype: str, snippets: List[Dict], save_folder: Path, color_scheme: str = "color"):
    """Plot 5×3 subplot layout: 5 animals (rows) × 3 samples (columns) with improved formatting."""

    if not snippets:
        return

    # Organize snippets by animal
    animal_snippets = {}
    for snippet in snippets:
        animal_id = snippet["animal_id"]
        if animal_id not in animal_snippets:
            animal_snippets[animal_id] = []
        animal_snippets[animal_id].append(snippet)

    # Take up to 5 animals and ensure each has 3 snippets
    animals = list(animal_snippets.keys())[:5]

    # Prepare data for 5×3 grid
    plot_data = []
    for animal_id in animals:
        animal_data = animal_snippets[animal_id][:3]  # Take up to 3 snippets
        # Pad with None if less than 3 snippets
        while len(animal_data) < 3:
            animal_data.append(None)
        plot_data.append(animal_data)

    # Pad with empty rows if less than 5 animals
    while len(plot_data) < 5:
        plot_data.append([None, None, None])

    # Calculate global y-limits for consistent scaling
    all_data = []
    for row_data in plot_data:
        for snippet in row_data:
            if snippet is not None:
                all_data.append(snippet["data"])

    # if all_data:
    #     # Find global min/max for consistent scaling
    #     global_min = np.min([np.min(data) for data in all_data])
    #     global_max = np.max([np.max(data) for data in all_data])
    #     data_range = global_max - global_min
    #     # Add 10% padding
    #     global_min -= 0.1 * data_range
    #     global_max += 0.1 * data_range
    # else:
    #     global_min, global_max = 0, 1
    global_max = 100
    global_min = -global_max

    # Create 5×3 subplot grid with 1:1 aspect ratio - no sharing for independent control
    fig_width = 15
    fig_height = 20  # Make it square for 1:1 aspect ratio
    fig, axes = plt.subplots(5, 3, figsize=(fig_width, fig_height), sharex=False, sharey=False)

    # Adjust spacing to prevent text overlap
    # plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.08, hspace=0.4, wspace=0.3)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    fig.suptitle(
        f"EEG Snippets - {genotype} ({'Color' if color_scheme == 'color' else 'Black'})",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )

    # Color scheme selection
    if color_scheme == "color":
        colors = plt.cm.tab10(np.linspace(0, 1, len(CHANNEL_ORDER)))
    else:
        colors = ["black"] * len(CHANNEL_ORDER)

    # Channel spacing for stacked display
    # channel_spacing = (global_max - global_min) / (len(CHANNEL_ORDER) + 1)
    channel_spacing = global_max - global_min

    for row in range(5):
        for col in range(3):
            ax = axes[row, col]
            snippet = plot_data[row][col] if row < len(plot_data) else None

            if snippet is None:
                # Empty subplot
                ax.text(
                    0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray"
                )
                # Set consistent limits and labels for empty subplots
                ax.set_xlim([0, SNIPPET_DURATION])  # Extra space for scalebar
                ax.set_ylim([global_min, global_max])
                # ax.set_xticks([0, 1, 2, 3, 4, 5])
                # ax.set_xticklabels([0, 1, 2, 3, 4, 5], fontsize=7)
                # ax.set_yticks([])  # No y-ticks for empty subplots
            else:
                # Plot all channels on this subplot
                data = snippet["data"]
                time_axis = snippet["time_axis"]
                channel_names = snippet["channel_names"]
                animal_id = snippet["animal_id"]

                # Get abbreviated channel names in original order
                channel_indices = list(range(len(channel_names)))
                plot_channel_abbrevs = []

                for ch_name in channel_names:
                    try:
                        abbrev = core.parse_chname_to_abbrev(ch_name, assume_from_number=True, strict_matching=False)
                        plot_channel_abbrevs.append(abbrev)
                    except Exception:
                        plot_channel_abbrevs.append(str(ch_name)[:4])  # Fallback: first 4 chars

                # Calculate y-offsets for stacked channels - start with spacing from bottom
                y_offset = global_min + channel_spacing
                y_labels = []
                y_positions = []

                for ch_idx, (data_ch_idx, ch_abbrev) in enumerate(zip(channel_indices, plot_channel_abbrevs)):
                    # Plot the channel data with offset
                    channel_data = data[:, data_ch_idx]
                    # Normalize to channel spacing and add offset - reduced amplitude
                    normalized_data = channel_data - np.mean(channel_data)
                    channel_data_plot = normalized_data + y_offset

                    color_idx = ch_idx % len(colors)  # Cycle through colors if more channels than colors
                    ax.plot(time_axis, channel_data_plot, color=colors[color_idx], linewidth=0.8, alpha=0.8)
                    y_labels.append(ch_abbrev)
                    y_positions.append(y_offset)

                    y_offset += channel_spacing

                # Set consistent axis parameters for all data subplots
                ax.set_xlim([0, SNIPPET_DURATION])  # Extra space for scalebar
                ax.set_ylim([global_min, y_offset])
                # ax.set_xticks([0, 1, 2, 3, 4, 5])
                # ax.set_xticklabels([0, 1, 2, 3, 4, 5], fontsize=7)
                ax.set_yticks(y_positions)
                ax.set_yticklabels(y_labels, fontsize=7)

                # Style the plot
                ax.grid(True, alpha=0.2)
                ax.tick_params(labelsize=7)

                # Force 1:1 aspect ratio
                ax.set_aspect("auto", adjustable="box")

            # Set labels and titles
            if row == 0:
                ax.set_title(f"Sample {col + 1}", fontsize=10, fontweight="bold", pad=10)

            # Set xlabel on all subplots
            ax.set_xlabel("Time (s)", fontsize=8)

            # Add animal ID on the left with better positioning
            if row < len(animals) and col == 0:
                ax.text(
                    -0.25,
                    0.5,
                    animals[row][:15],
                    rotation=90,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    fontweight="bold",
                )

    # Add scale bar to all subplots that have data - positioned outside plot area
    for row in range(5):
        for col in range(3):
            if plot_data[row][col] is not None:
                ax_scale = axes[row, col]

                # Position scalebar in bottom-right corner, outside the data area
                # ANCHOR position the scale bar
                scale_x_start = SNIPPET_DURATION + 0.2  # Just outside the right edge
                scale_y_start = global_min - 3 * channel_spacing  # Near bottom
                scale_width = 1.0  # 1 second
                scale_height = channel_spacing  # Voltage scale

                # Horizontal line for time
                line_h = ax_scale.plot(
                    [scale_x_start, scale_x_start + scale_width], [scale_y_start, scale_y_start], "k-", linewidth=2
                )
                line_h[0].set_clip_on(False)  # Disable clipping so line shows outside plot

                text_h = ax_scale.text(
                    scale_x_start + scale_width / 2,
                    scale_y_start - channel_spacing * 0.1,
                    "1 s",
                    ha="center",
                    va="top",
                    fontsize=7,
                    fontweight="bold",
                )
                text_h.set_clip_on(False)  # Disable clipping so text shows outside plot

                # Vertical line for voltage
                line_v = ax_scale.plot(
                    [scale_x_start, scale_x_start], [scale_y_start, scale_y_start + scale_height], "k-", linewidth=2
                )
                line_v[0].set_clip_on(False)  # Disable clipping so line shows outside plot

                text_v = ax_scale.text(
                    scale_x_start - 0.05,
                    scale_y_start + scale_height / 2,
                    f"{int(scale_height)} μV",
                    ha="right",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    rotation=90,
                )
                text_v.set_clip_on(False)  # Disable clipping so text shows outside plot

                # X-axis limits already set to accommodate scalebar

    # Save the plot
    color_suffix = "_color" if color_scheme == "color" else "_black"
    filename = f"eeg_snippets_{genotype}_5x3{color_suffix}.tiff"
    filepath = save_folder / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white", format="tiff")
    plt.close()

    logger.info(f"Saved {genotype} 5×3 {color_scheme} plot to {filepath}")


def main():
    """Main function to orchestrate the EEG snippet plotting."""

    logger.info("Starting EEG snippet plotting script")
    logger.info(f"Target: {SNIPPETS_PER_ANIMAL} snippets × {ANIMALS_PER_GENOTYPE} animals per genotype")
    logger.info(f"Snippet duration: {SNIPPET_DURATION} seconds")

    # Load genotype mapping from JSON
    genotype_to_animals, data_parent_folder = load_genotype_mapping()

    # Select animals for processing
    animals_to_process = []
    for genotype in GENOTYPES:
        available_animals = genotype_to_animals[genotype]
        if len(available_animals) < ANIMALS_PER_GENOTYPE:
            logger.warning(
                f"Only {len(available_animals)} animals available for {genotype}, need {ANIMALS_PER_GENOTYPE}"
            )
            selected_animals = available_animals
        else:
            selected_animals = random.sample(available_animals, ANIMALS_PER_GENOTYPE)

        for animal_id, data_folder in selected_animals:
            animals_to_process.append((animal_id, genotype, data_folder, data_parent_folder))

    logger.info(f"Processing {len(animals_to_process)} animals total")

    # Process animals in parallel
    start_time = time.time()

    with Pool(processes=min(10, len(animals_to_process))) as pool:
        results = list(
            tqdm(
                pool.imap(load_and_sample_animal_data, animals_to_process),
                total=len(animals_to_process),
                desc="Processing animals",
            )
        )

    # Filter successful results
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    logger.info(f"Successfully processed {len(successful_results)} animals")
    if failed_results:
        logger.warning(f"Failed to process {len(failed_results)} animals:")
        for failed in failed_results:
            logger.warning(f"  {failed['animal_id']}: {failed.get('error', 'Unknown error')}")

    # Generate plots
    if successful_results:
        logger.info("Generating plots...")
        plot_snippets(successful_results, save_folder)

        # Summary statistics
        total_snippets = sum(len(r["snippets"]) for r in successful_results)
        logger.info(f"Generated plots for {total_snippets} total snippets")

        # Per-genotype summary
        genotype_counts = {}
        for result in successful_results:
            genotype = result["genotype"]
            genotype_counts[genotype] = genotype_counts.get(genotype, 0) + len(result["snippets"])

        for genotype, count in genotype_counts.items():
            logger.info(f"  {genotype}: {count} snippets")
    else:
        logger.error("No animals were successfully processed!")

    elapsed_time = time.time() - start_time
    logger.info(f"Script completed in {elapsed_time:.1f} seconds")


if __name__ == "__main__":
    main()

"""
To run this script on SLURM:

sbatch --mem 200GB -c 11 -t 12:00:00 /mnt/isilon/marsh_single_unit/PythonEEG/notebooks/examples/pipeline.sh /mnt/isilon/marsh_single_unit/PythonEEG/notebooks/examples/plot_eeg_snippets.py
"""
