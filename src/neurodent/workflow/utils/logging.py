"""Snakemake runtime glue: log-file setup and memory-retry resources."""

import logging
import os
import sys


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
        Snakemake rule. Log files use ``.out`` and ``.err`` extensions and
        are organized under ``logs/<rule_group>/``. For example::

            rule my_rule:
                log:
                    stdout="logs/my_rule/{animal}.out",
                    stderr="logs/my_rule/{animal}.err",
                script: "scripts/my_script.py"

        The logger will write to ``logs/my_rule/{animal}.out``.
    """
    log_path = snakemake.log[0]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
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


def increment_memory(base_memory):
    """Return a callable ``mem(wildcards, attempt)`` that doubles on each retry.

    Used by Snakemake rules to exponentially increase memory on retries::

        resources:
            mem_mb=increment_memory(4000),
    """
    def mem(wildcards, attempt):
        return base_memory * (2 ** (attempt - 1))
    return mem
