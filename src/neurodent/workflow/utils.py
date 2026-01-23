"""
Workflow utility functions.

This module provides utilities that reduce boilerplate in Snakemake workflow scripts.
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurodent.visualization import WindowAnalysisResult


def setup_snakemake_logging(snakemake) -> logging.Logger:
    """Configure logging to write to the Snakemake log file.

    This replaces the common boilerplate pattern in workflow scripts that
    redirects stdout/stderr to the log file specified in the Snakemake rule.

    Args:
        snakemake: The global snakemake object injected by Snakemake.
            Must have a ``log`` attribute containing the log file path.

    Returns:
        logging.Logger: A configured logger instance.

    Example:
        In a Snakemake script::

            from neurodent.workflow import setup_snakemake_logging

            def main():
                logger = setup_snakemake_logging(snakemake)
                logger.info("Starting processing...")

            if __name__ == "__main__":
                main()

    Note:
        The log file path is determined by the ``log:`` directive in your
        Snakemake rule. For example::

            rule my_rule:
                log: "logs/my_rule.log"
                script: "scripts/my_script.py"

        The logger will write to ``logs/my_rule.log``.
    """
    log_path = snakemake.log[0]
    log_file = open(log_path, "w")

    # Redirect stdout and stderr to the log file
    sys.stdout = log_file
    sys.stderr = log_file

    # Configure logging to use the redirected stdout
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
        force=True,
    )

    return logging.getLogger(__name__)


def inject_config_aliases(samples_config: dict):
    """Inject aliases from samples_config into the global neurodent.constants.

    This ensures that custom aliases for genotypes, channel names, and L/R labels
    are available across all modules in the pipeline. This should be called at the
    beginning of every Snakemake script that loads WindowAnalysisResults or uses
    channel name parsing.

    Args:
        samples_config (dict): Configuration dictionary loaded from samples_jess.json
    """
    from neurodent import constants

    if "GENOTYPE_ALIASES" in samples_config:
        constants.GENOTYPE_ALIASES = samples_config["GENOTYPE_ALIASES"]
    if "CHNAME_ALIASES" in samples_config:
        constants.CHNAME_ALIASES = samples_config["CHNAME_ALIASES"]
    if "LR_ALIASES" in samples_config:
        constants.LR_ALIASES = samples_config["LR_ALIASES"]


def load_wars(
    pkl_paths: list[str | Path],
    json_paths: list[str | Path] | None = None,
) -> list["WindowAnalysisResult"]:
    """Load multiple WindowAnalysisResult objects from pickle/json file pairs.

    General-purpose utility for loading WAR files. Works with any list of paths,
    not tied to Snakemake.

    Args:
        pkl_paths: Paths to .pkl files containing WindowAnalysisResult data.
        json_paths: Optional paths to corresponding .json metadata files.
            If None, assumes json files are in the same directory as pkl files
            with the same basename but .json extension.

    Returns:
        List of loaded WindowAnalysisResult objects.

    Raises:
        FileNotFoundError: If a pkl or json file does not exist.
        RuntimeError: If no WARs could be loaded.

    Example:
        Load WARs from explicit paths::

            from neurodent.workflow import load_wars

            wars = load_wars(
                pkl_paths=["data/animal1/war.pkl", "data/animal2/war.pkl"],
                json_paths=["data/animal1/war.json", "data/animal2/war.json"],
            )

        Load WARs with auto-detected json paths::

            wars = load_wars(pkl_paths=["data/animal1/war.pkl"])
            # Automatically looks for data/animal1/war.json
    """
    from neurodent import visualization

    # If json_paths not provided, derive from pkl_paths
    if json_paths is None:
        json_paths = [Path(p).with_suffix(".json") for p in pkl_paths]

    if len(pkl_paths) != len(json_paths):
        raise ValueError(
            f"pkl_paths ({len(pkl_paths)}) and json_paths ({len(json_paths)}) "
            "must have the same length"
        )

    wars = []
    for pkl_path, json_path in zip(pkl_paths, json_paths):
        pkl_path = Path(pkl_path)
        json_path = Path(json_path)

        war = visualization.WindowAnalysisResult.load_pickle_and_json(
            folder_path=pkl_path.parent,
            pickle_name=pkl_path.name,
            json_name=json_path.name,
        )
        wars.append(war)

    if not wars:
        raise RuntimeError("No WARs were successfully loaded")

    return wars
