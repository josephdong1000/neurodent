"""
Command-line interface for NeuRodent.

Provides the ``neurodent`` command with sub-commands for managing the
Snakemake pipeline.
"""

import argparse
import shutil
import sys
from importlib.resources import files
from pathlib import Path


def _pipeline_source() -> Path:
    """Return the path to the bundled pipeline package data."""
    return Path(str(files("neurodent.pipeline")))


def cmd_init_pipeline(args: argparse.Namespace) -> int:
    """Copy the bundled Snakemake pipeline files to the working directory.

    Parameters
    ----------
    args:
        Parsed CLI arguments.  Relevant attributes:

        ``target`` – destination directory (defaults to current directory).
        ``overwrite`` – if ``True``, overwrite any existing files.

    Returns
    -------
    int
        Exit code (0 on success, non-zero on error).
    """
    target = Path(args.target).resolve()
    overwrite = args.overwrite
    source = _pipeline_source()

    items_to_copy = [
        ("Snakefile", target / "Snakefile"),
        ("workflow", target / "workflow"),
        ("config", target / "config"),
    ]

    for src_name, dst_path in items_to_copy:
        src_path = source / src_name
        if not src_path.exists():
            print(
                f"[neurodent] ERROR: bundled pipeline item not found: {src_name}",
                file=sys.stderr,
            )
            return 1

        if dst_path.exists():
            if not overwrite:
                print(
                    f"[neurodent] Skipping '{dst_path.name}' (already exists). "
                    "Use --overwrite to replace.",
                )
                continue
            # Remove the existing file/directory before copying
            if dst_path.is_dir():
                shutil.rmtree(dst_path)
            else:
                dst_path.unlink()

        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

        print(f"[neurodent] Copied '{src_name}' → '{dst_path}'")

    print(
        "\n[neurodent] Pipeline initialised.  Next steps:\n"
        f"  1. Edit '{target / 'config' / 'config.yaml'}' for your dataset.\n"
        "  2. Run:  snakemake --cores <N>\n"
        "  3. Optional cluster support:  pip install neurodent[pipeline]"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="neurodent",
        description="NeuRodent command-line tools",
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="<command>",
    )

    # ------------------------------------------------------------------ #
    # init-pipeline                                                        #
    # ------------------------------------------------------------------ #
    init_parser = subparsers.add_parser(
        "init-pipeline",
        help="Copy the bundled Snakemake pipeline files to a directory",
        description=(
            "Copy the NeuRodent Snakemake pipeline (Snakefile, workflow/, config/) "
            "into the target directory so the pipeline can be run without a git clone."
        ),
    )
    init_parser.add_argument(
        "target",
        nargs="?",
        default=".",
        metavar="DIR",
        help="Destination directory (default: current directory)",
    )
    init_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files/directories",
    )
    init_parser.set_defaults(func=cmd_init_pipeline)

    return parser


def main() -> None:
    """Entry point for the ``neurodent`` CLI command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
