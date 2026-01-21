"""
Split Joint Recordings Script
=============================

This script is called by the split_joint_recordings Snakemake rule.
It uses AnimalOrganizer.split() to split a joint multi-animal recording
into per-animal zarr files.

Usage (via Snakemake):
    Receives parameters via snakemake object:
    - snakemake.params.session: Session folder name
    - snakemake.params.joint_config: Channel mapping dict
    - snakemake.params.data_parent: Parent data folder
    - snakemake.params.split_config: AO/LRO configuration dict
    - snakemake.output[0]: Output directory
"""

from pathlib import Path

from neurodent.visualization import AnimalOrganizer
from neurodent.workflow import setup_snakemake_logging


def main():
    """Main entry point for the split_joint_recordings script."""
    
    logger = setup_snakemake_logging(snakemake)
    logger.info("Split joint recordings script started")
    
    # Extract parameters from Snakemake
    session = snakemake.params.session
    joint_config = snakemake.params.joint_config
    data_parent = snakemake.params.data_parent
    split_config = snakemake.params.split_config
    output_base = Path(snakemake.output[0])
    
    logger.info(f"Splitting joint session: {session}")
    logger.info(f"Channel groups: {list(joint_config.keys())}")
    logger.info(f"Output directory: {output_base}")

    # Build path to joint session folder
    session_folder = Path(data_parent) / session
    
    if not session_folder.exists():
        raise FileNotFoundError(f"Session folder not found: {session_folder}")
    
    logger.info(f"Loading joint recording from: {session_folder}")
    
    # Create AnimalOrganizer for the joint recording
    # Use a generic animal_id since this is multi-animal
    ao = AnimalOrganizer(
        base_folder_path=session_folder,
        anim_id="joint",
        mode=split_config["mode"],
        file_pattern=split_config.get("file_pattern"),
        day_sep=split_config.get("day_sep"),
        assume_from_number=split_config.get("assume_from_number", True),
        lro_kwargs=split_config.get("lro_kwargs", {}),
    )
    
    logger.info(f"Loaded {len(ao.long_recordings)} days of recordings")
    logger.info(f"Channels: {ao.channel_names}")
    
    # Split by channel groups
    output_format = split_config.get("output_format", "zarr")
    splits = ao.split(
        groups=joint_config,
        persist_base=output_base,
        format=output_format,
    )
    
    logger.info(f"Split into {len(splits)} animals:")
    for animal_name, child_ao in splits.items():
        logger.info(
            f"  - {animal_name}: {len(child_ao.long_recordings)} days, "
            f"{len(child_ao.channel_names)} channels"
        )
    
    logger.info("Joint recording split complete!")


if __name__ == "__main__":
    main()
