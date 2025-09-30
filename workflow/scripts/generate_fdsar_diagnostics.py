#!/usr/bin/env python
"""
FDSAR Diagnostics Script
========================

Generate diagnostic plots from frequency-domain spike analysis results (FDSARs).
This script creates spike-averaged trace plots and saves epoch data for validation.

Input: FDSAR results directory (contains .fif and .json files)
Output: Spike-averaged plots and epoch .fif files
"""

import json
import logging
import sys
import warnings
from pathlib import Path

from pythoneeg.visualization.frequency_domain_results import FrequencyDomainSpikeAnalysisResult


def load_fdsar_results(fdsar_base_dir: Path):
    """
    Load all FDSAR results from a directory containing per-animalday subdirectories.

    Expected structure:
        fdsar_base_dir/
        ├── animal-genotype-day1/
        │   ├── animal-genotype-day1.json
        │   └── animal-genotype-day1-raw.fif
        └── animal-genotype-day2/
            ├── animal-genotype-day2.json
            └── animal-genotype-day2-raw.fif
    """
    fdsar_base_dir = Path(fdsar_base_dir)

    if not fdsar_base_dir.exists():
        raise ValueError(f"FDSAR directory does not exist: {fdsar_base_dir}")

    fdsar_list = []

    # Find all subdirectories that contain FDSAR results
    subdirs = [d for d in fdsar_base_dir.iterdir() if d.is_dir()]

    if not subdirs:
        logging.warning(f"No subdirectories found in {fdsar_base_dir}")
        return fdsar_list

    logging.info(f"Found {len(subdirs)} potential FDSAR subdirectories in {fdsar_base_dir}")

    for subdir in sorted(subdirs):
        # Check if this subdirectory contains FDSAR files
        try:
            logging.info(f"Loading FDSAR from {subdir.name}")
            fdsar = FrequencyDomainSpikeAnalysisResult.load_fif_and_json(subdir)
            fdsar_list.append(fdsar)
        except Exception as e:
            logging.error(f"Failed to load FDSAR from {subdir}: {e}")
            raise

    return fdsar_list


def generate_diagnostics(fdsar_list, output_dir: Path):
    """Generate diagnostic plots for all FDSAR results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not fdsar_list:
        logging.warning("No FDSAR results to process")
        return

    logging.info(f"Generating diagnostics for {len(fdsar_list)} FDSAR results")

    total_spikes = 0
    for i, fdsar in enumerate(fdsar_list):
        logging.info(f"Processing FDSAR {i+1}/{len(fdsar_list)}: {fdsar.animal_id} - {fdsar.animal_day}")

        spike_counts = fdsar.get_spike_counts_per_channel()
        session_total = sum(spike_counts)
        total_spikes += session_total

        logging.info(f"  Spikes detected: {session_total} across {len(spike_counts)} channels")

        if session_total == 0:
            logging.warning(f"  No spikes detected, skipping diagnostic plots")
            continue

        # Generate spike-averaged traces
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                returned_counts = fdsar.plot_spike_averaged_traces(
                    tmin=-0.5,
                    tmax=0.5,
                    baseline=None,
                    save_dir=output_dir,
                    animal_id=f"{fdsar.animal_id}_{fdsar.animal_day}",
                    save_epoch=True
                )

            logging.info(f"  Generated plots for {len([c for c in returned_counts if c > 0])} channels with spikes")

        except Exception as e:
            logging.error(f"  Failed to generate diagnostic plots: {e}")
            raise

    logging.info(f"Total spikes across all sessions: {total_spikes}")


def main():
    """Main execution function"""
    global snakemake

    with open(snakemake.log[0], "w") as f:
        sys.stderr = sys.stdout = f

        # Set up logging to the redirected stdout
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.DEBUG,
            stream=sys.stdout,
            force=True,
        )

        logging.info("FDSAR diagnostics script started")

        # Load FDSAR results
        fdsar_dir = Path(snakemake.input.fdsar_dir)
        output_dir = Path(snakemake.output.diagnostics_dir)

        logging.info(f"Loading FDSAR results from: {fdsar_dir}")
        fdsar_list = load_fdsar_results(fdsar_dir)

        if not fdsar_list:
            logging.warning("No FDSAR results found, creating empty output directory")
            output_dir.mkdir(parents=True, exist_ok=True)
            return

        # Generate diagnostics
        logging.info(f"Generating diagnostics to: {output_dir}")
        generate_diagnostics(fdsar_list, output_dir)

        logging.info("FDSAR diagnostics generation completed successfully")


if __name__ == "__main__":
    main()
