#!/usr/bin/env python3
"""
EP Heatmap Generation Script
===========================

Generate experiment-level correlation/coherence matrix heatmaps using ExperimentPlotter.
Based on the heatmap pipeline from notebooks/tests/ep figures example.py.

Input: Flattened WAR pickle and JSON files from all animals
Output: Heatmap matrix files (TIF) and CSV data exports
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.colors as colors

from neurodent import visualization, constants
from neurodent.workflow import (
    setup_snakemake_logging,
    load_wars,
    apply_samples_config,
    extend_plot_order_from_attr,
)


def generate_regular_heatmaps(ep, features, output_dir, data_dir, ep_config):
    """Generate regular correlation/coherence heatmaps"""

    logger = logging.getLogger(__name__)

    # Get format parameters from config
    figure_format = ep_config.get("figure_format", "png")
    data_format = ep_config.get("data_format", "pkl")
    dpi = ep_config.get("dpi", 300)

    for feature in features:
        logger.info(f"Generating regular heatmap for {feature}")

        try:
            # Pull data for this feature
            df = ep.pull_timeseries_dataframe(feature, ["genotype", "isday"], average_groupby=True)

            # Save data in configured format
            if data_format == "csv":
                df.to_csv(data_dir / f"{feature}.csv", index=False)
            else:  # default to pkl
                df.to_pickle(data_dir / f"{feature}.pkl")

            if feature in ["cohere", "imcoh", "zcohere", "zimcoh"]:
                # Band-based features - use faceted heatmaps
                if feature.startswith("z"):
                    # Z-transformed features use centered normalization
                    gs = ep.plot_heatmap_faceted(
                        feature,
                        groupby=["genotype", "isday"],
                        facet_vars="band",
                        norm=colors.CenteredNorm(vcenter=0, halfrange=2),
                    )
                else:
                    # Regular features
                    gs = ep.plot_heatmap_faceted(feature, groupby=["genotype", "isday"], facet_vars="band")

                # Save each subplot
                for i, g in enumerate(gs):
                    g.savefig(output_dir / f"matrix-{feature}-{i}.{figure_format}", bbox_inches="tight", dpi=dpi)

            elif feature in ["pcorr", "zpcorr"]:
                # Non-band features - single heatmap
                if feature.startswith("z"):
                    # Z-transformed features use centered normalization
                    g = ep.plot_heatmap(
                        feature, groupby=["genotype", "isday"], norm=colors.CenteredNorm(vcenter=0, halfrange=2)
                    )
                else:
                    # Regular features
                    g = ep.plot_heatmap(feature, groupby=["genotype", "isday"])

                g.savefig(output_dir / f"matrix-{feature}.{figure_format}", bbox_inches="tight", dpi=dpi)

            logger.info(f"Successfully generated regular heatmap for {feature}")

        except Exception as e:
            logger.error(f"Failed to generate regular heatmap for {feature}: {str(e)}")
            raise


def filter_wars_by_sex(wars, sex):
    """
    Filter WARs by sex using the war.sex attribute.

    Args:
        wars: List of WindowAnalysisResult objects.
        sex: Sex string to filter by (e.g. "Male", "Female").

    Returns:
        List of WARs matching the given sex.
    """
    return [war for war in wars if war.sex == sex]


def determine_baseline_key(found_genotypes, sex):
    """Determine the baseline genotype key for difference maps.

    Prefer the first genotype in ``DF_SORT_ORDER["genotype"]`` that is actually
    present (typically "WT"); fall back to the first observed genotype.
    """
    for g in constants.DF_SORT_ORDER.get("genotype", []):
        if g in found_genotypes:
            return g
    return found_genotypes[0] if found_genotypes else None


def generate_difference_heatmaps(wars, features, output_dir, config):
    """Generate difference heatmaps (baseline comparison)"""

    logger = logging.getLogger(__name__)

    # Get baseline configuration
    ep_config = config["analysis"]["ep_heatmaps"]
    baseline_type = ep_config.get("baseline_type", "sex_specific")  # "sex_specific" or "global"
    figure_format = ep_config.get("figure_format", "png")
    dpi = ep_config.get("dpi", 300)

    if baseline_type == "sex_specific":
        # Create separate EPs for male and female, compare to sex-specific WT
        for sex in constants.DF_SORT_ORDER.get("sex", ["Male", "Female"]):
            logger.info(f"Generating difference heatmaps for {sex} vs WT")

            # Filter wars by sex
            sex_wars = filter_wars_by_sex(wars, sex)

            if not sex_wars:
                logger.warning(f"No wars found for sex {sex}")
                continue

            # Extend genotype plot order with any values observed on the
            # sex-filtered WARs that aren't in the default DF_SORT_ORDER.
            plot_order = constants.DF_SORT_ORDER.copy()
            plot_order["genotype"] = extend_plot_order_from_attr(
                sex_wars, "genotype", constants.DF_SORT_ORDER.get("genotype", [])
            )

            ep = visualization.ExperimentPlotter(
                wars=sex_wars,
                plot_order=plot_order,
            )

            # Determine baseline key from the genotypes actually present in this sex bucket
            found_genotypes = sorted({war.genotype for war in sex_wars})
            baseline_key = determine_baseline_key(found_genotypes, sex)
            if baseline_key not in found_genotypes:
                 logger.warning(f"Could not find exact baseline genotype for {sex}. Using {baseline_key} (might fail). Found: {found_genotypes}")

            for feature in features:
                logger.info(f"Generating difference heatmap for {feature} ({sex} vs {baseline_key})")

                try:
                    if feature in ["cohere", "imcoh", "zcohere", "zimcoh"]:
                        # Band-based features
                        g = ep.plot_diffheatmap_faceted(
                            feature,
                            groupby=["genotype", "isday"],
                            baseline_key=baseline_key,
                            baseline_groupby="genotype",
                            facet_vars="band",
                            norm=colors.CenteredNorm(vcenter=0, halfrange=0.5),
                        )
                        for i, figure in enumerate(g):
                            figure.savefig(
                                output_dir / f"diffmatrix-{feature}-{sex}-{i}.{figure_format}",
                                bbox_inches="tight",
                                dpi=dpi,
                            )

                    elif feature in ["pcorr", "zpcorr"]:
                        # Non-band features
                        g = ep.plot_diffheatmap(
                            feature,
                            groupby=["genotype", "isday"],
                            baseline_key=baseline_key,
                            baseline_groupby="genotype",
                            norm=colors.CenteredNorm(vcenter=0, halfrange=0.5),
                        )
                        g.savefig(
                            output_dir / f"diffmatrix-{feature}-{sex}.{figure_format}", bbox_inches="tight", dpi=dpi
                        )

                    logger.info(f"Successfully generated difference heatmap for {feature} ({sex})")

                except Exception as e:
                    logger.error(f"Failed to generate difference heatmap for {feature} ({sex}): {str(e)}")
                    raise

    else:
        # Global baseline (e.g., compare all to FWT or MWT)
        logger.info("Generating global baseline difference heatmaps")
        # Implementation for global baseline if needed
        logger.warning("Global baseline difference heatmaps not yet implemented")


def main():
    """Main EP heatmaps generation function"""
    global snakemake
    logger = setup_snakemake_logging(snakemake)
    logger.info("EP heatmap generation started")

    # Get parameters from snakemake
    war_parquet_files = snakemake.input.war_parquet
    war_json_files = snakemake.input.war_json
    config = snakemake.params.config
    samples_config = snakemake.params.samples_config

    # Inject aliases
    apply_samples_config(samples_config)

    # Create output directories
    output_dir = Path(snakemake.output.heatmap_dir)
    data_dir = Path(snakemake.output.heatmap_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {len(war_parquet_files)} flattened WARs")

    # Load WARs using the workflow utility
    wars = load_wars(war_parquet_files, war_json_files)
    for war in wars:
        logger.info(f"Loaded WAR for {war.animal_id} ({war.genotype})")

    logger.info(f"Successfully loaded {len(wars)} WARs")

    # Get EP heatmap configuration
    ep_config = config["analysis"]["ep_heatmaps"]
    features = ep_config["matrix_features"]

    # Extend the genotype/sex plot orders with any values observed on the
    # loaded WARs that aren't in the default DF_SORT_ORDER.  Without this,
    # datasets like arxrosa (every animal has sex='Unknown', genotype='UNKNOWN')
    # fail strict plot_order validation in sort_dataframe_by_plot_order.
    plot_order = constants.DF_SORT_ORDER.copy()
    plot_order["genotype"] = extend_plot_order_from_attr(
        wars, "genotype", constants.DF_SORT_ORDER.get("genotype", [])
    )
    plot_order["sex"] = extend_plot_order_from_attr(
        wars, "sex", constants.DF_SORT_ORDER.get("sex", [])
    )

    # Create ExperimentPlotter for regular heatmaps
    logger.info("Creating ExperimentPlotter for regular heatmaps")
    ep = visualization.ExperimentPlotter(
        wars=wars,
        plot_order=plot_order,
    )

    # Generate regular heatmaps
    logger.info("Generating regular heatmaps")
    generate_regular_heatmaps(ep, features, output_dir, data_dir, ep_config)

    # Generate difference heatmaps
    logger.info("Generating difference heatmaps")
    generate_difference_heatmaps(wars, features, output_dir, config)
    logger.info(f"Successfully generated EP heatmaps for {len(features)} features")


if __name__ == "__main__":
    main()
