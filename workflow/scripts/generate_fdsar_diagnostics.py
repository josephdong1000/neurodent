#!/usr/bin/env python
"""
FDSAR Diagnostics Script
========================

Generate diagnostic plots from frequency-domain spike analysis results (FDSARs).
This script creates spike-averaged trace plots and saves epoch data for validation.

Input: FDSAR results directory (contains .fif and .json files)
Output: Spike-averaged plots and epoch .fif files
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from neurodent.visualization.frequency_domain_results import FrequencyDomainSpikeAnalysisResult
from neurodent.workflow import setup_snakemake_logging, apply_samples_config


def load_fdsar_results(fdsar_base_dir: Path):
    """
    Yield FDSAR results one at a time from a directory of per-animalday subdirectories.

    Yields one :class:`FrequencyDomainSpikeAnalysisResult` at a time so that
    only a single FDSAR is live in memory during iteration.  Each result's
    ``.fif`` file is opened with ``preload=False`` (memory-mapped), so peak
    RAM is proportional to epoch-window size rather than full-recording size.

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

    # Find all subdirectories that contain FDSAR results
    subdirs = [d for d in fdsar_base_dir.iterdir() if d.is_dir()]

    if not subdirs:
        logging.warning(f"No subdirectories found in {fdsar_base_dir}")
        return

    logging.info(f"Found {len(subdirs)} potential FDSAR subdirectories in {fdsar_base_dir}")

    for subdir in sorted(subdirs):
        try:
            logging.info(f"Loading FDSAR from {subdir.name}")
            yield FrequencyDomainSpikeAnalysisResult.load_fif_and_json(subdir)
        except Exception as e:
            logging.error(f"Failed to load FDSAR from {subdir}: {e}")
            raise


def generate_diagnostics(fdsar_iter, output_dir: Path, sat_config: dict):
    """Generate diagnostic plots for all FDSAR results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_spikes = 0
    i = 0
    for i, fdsar in enumerate(fdsar_iter, start=1):
        logging.info(f"Processing FDSAR {i}: {fdsar.animal_id} - {fdsar.animal_day}")

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
                    tmin=sat_config.get("tmin", -0.5),
                    tmax=sat_config.get("tmax", 0.5),
                    baseline=sat_config.get("baseline", None),
                    save_dir=output_dir,
                    animal_id=f"{fdsar.animal_id}_{fdsar.animal_day}",
                    # Note: the config key is 'save_epochs' (plural) to match YAML convention,
                    # while the function parameter is 'save_epoch' (singular).
                    save_epoch=sat_config.get("save_epochs", True),
                )

            logging.info(f"  Generated plots for {len([c for c in returned_counts if c > 0])} channels with spikes")

        except Exception as e:
            logging.error(f"  Failed to generate diagnostic plots: {e}")
            raise

    if not i:
        logging.warning("No FDSAR results to process")
        return

    logging.info(f"Total spikes across all sessions: {total_spikes}")


def main():
    """Main execution function"""
    global snakemake

    logger = setup_snakemake_logging(snakemake)

    logger.info("FDSAR diagnostics script started")

    # Load FDSAR results
    fdsar_dir = Path(snakemake.params.fdsar_dir)
    output_dir = Path(snakemake.output.diagnostics_dir)
    config = snakemake.params.config
    samples_config = snakemake.params.samples_config

    # Inject aliases
    apply_samples_config(samples_config)

    # Extract spike-averaged-traces parameters from config
    sat_config = (
        config
        .get("analysis", {})
        .get("frequency_domain_spike_detection", {})
        .get("spike_averaged_traces", {})
    )

    logger.info(f"Loading FDSAR results from: {fdsar_dir}")
    fdsar_iter = load_fdsar_results(fdsar_dir)

    # Generate diagnostics
    # Note: load_fdsar_results() returns a generator (always truthy), so
    # emptiness is handled inside generate_diagnostics() via the `if not i`
    # guard after iteration completes.
    logger.info(f"Generating diagnostics to: {output_dir}")
    generate_diagnostics(fdsar_iter, output_dir, sat_config)

    logger.info("FDSAR diagnostics generation completed successfully")


if __name__ == "__main__":
    main()
