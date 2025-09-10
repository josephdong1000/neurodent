"""
WAR Quality Filter Rules
========================

Rules for filtering WARs based on quality criteria like genotype validation.
This creates a clean separation between WAR generation (all animals) and 
downstream analysis (only good quality WARs).
"""


checkpoint war_quality_filter:
    """
    Filter WARs based on quality criteria and symlink good ones to filtered directory
    """
    input:
        war_pkl="results/wars/{animal}/war.pkl",
        war_json="results/wars/{animal}/war.json",
    output:
        directory("results/wars_quality_filtered/{animal}"),
    log:
        "logs/war_quality_filter/{animal}.log",
    threads: 1
    retries: 1
    resources:
        time=config["cluster"]["war_quality_filter"]["time"],
        mem_mb=increment_memory(config["cluster"]["war_quality_filter"]["mem_mb"]),
        nodes=config["cluster"]["war_quality_filter"]["nodes"],
    run:
        import logging
        from pathlib import Path
        import sys
        import json
        from datetime import datetime

        # Set up logging for this animal
        log_file = Path(log[0])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=log[0], level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filemode="w"
        )
        logger = logging.getLogger()

        sys.path.insert(0, str(Path("pythoneeg").resolve()))

        animal = wildcards.animal
        logger.info(f"Starting quality filtering for animal: {animal}")


        # Helper function to clean up existing files
        def cleanup_existing_files(destination_dir, input_files):
            """Remove any existing files for rejected animals"""
            for file_path in [Path(x) for x in input_files]:
                if file_path.is_file():
                    destination_file = destination_dir / file_path.name
                    if destination_file.exists():
                        destination_file.unlink()  # Remove existing file if present
                        logging.info(f"Removed existing file: {destination_file}")

                # Get bad animaldays from samples config via config parameter


        with open(config["samples"]["samples_file"], "r") as f:
            samples_config = json.load(f)
        bad_folder_animalday = samples_config.get("bad_folder_animalday", [])

        # Get quality filtering parameters from config
        exclude_bad_animaldays = config["samples"]["quality_filter"]["exclude_bad_animaldays"]
        exclude_unknown_genotypes = config["samples"]["quality_filter"]["exclude_unknown_genotypes"]

        logger.info(
            f"Quality filter settings - exclude_bad_animaldays: {exclude_bad_animaldays}, exclude_unknown_genotypes: {exclude_unknown_genotypes}"
        )

        # Create filtered directory and clean up any existing files
        destination_dir = Path(output[0])
        destination_dir.mkdir(parents=True, exist_ok=True)
        cleanup_existing_files(destination_dir, input)

        # Check if this is a bad animalday first
        combined_name = SLUGIFIED_TO_ORIGINAL[animal]
        logger.info(f"Original animal name: {combined_name}")

        if exclude_bad_animaldays and combined_name in bad_folder_animalday:
            logger.warning(f"REJECTED: Animal {animal} ({combined_name}) is in bad_folder_animalday list")
            return

        try:
            # Apply quality filters
            with open(input.war_json, "r") as f:
                data = json.load(f)
                genotype = data.get("genotype", "Unknown")
                logger.info(f"Animal genotype: {genotype}")

                if exclude_unknown_genotypes and genotype == "Unknown":
                    logger.warning(f"REJECTED: Animal {animal} has unknown genotype")
                    return

            logger.info(f"ACCEPTED: Animal {animal} passed all quality filters")
            logger.info(f"Copying files to: {destination_dir}")

            # Copy all files from the original WAR directory
            import shutil
            copied_files = []
            for file_path in [Path(x) for x in input]:
                if file_path.is_file():
                    destination_file = destination_dir / file_path.name
                    shutil.copy2(file_path, destination_file)
                    copied_files.append(file_path.name)
                    logger.info(f"Copied file: {file_path.name} -> {destination_file}")

            logger.info(f"Successfully copied {len(copied_files)} files: {copied_files}")

        except Exception as e:
            logger.error(f"ERROR: Failed to process animal {animal}: {str(e)}")
            raise
