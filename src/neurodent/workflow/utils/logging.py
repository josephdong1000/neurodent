"""Snakemake runtime glue: log-file setup and memory-retry resources."""

import contextlib
import logging
import os
import signal
import socket
import sys
import threading
import traceback
from datetime import datetime

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def log_paths(log):
    """Return the stdout and stderr log paths of a rule's ``log`` directive.

    The rules of the pipeline name them ``stdout`` and ``stderr``. A rule with a single log file
    sends both streams to it, and a rule with two unnamed ones uses them in that order.

    Args:
        log: The rule's log files, ``snakemake.log`` in a script or ``log`` in a ``run:`` block.

    Returns:
        tuple[str, str]: The stdout and stderr paths, the same path twice for a single log.

    Raises:
        ValueError: If the rule declares no log file.
    """
    if len(log) == 0:
        raise ValueError("The rule declares no log file to write to")
    stdout = getattr(log, "stdout", None) or log[0]
    stderr = getattr(log, "stderr", None) or (log[1] if len(log) > 1 else stdout)
    return str(stdout), str(stderr)


def attempt_header(rule, wildcards, attempt):
    """Return the line that opens one attempt of a job in its log files.

    It names the rule, the wildcards, the attempt, the SLURM job and node, and the start time,
    so the attempts of a job that was retried can be told apart in the same log.

    Args:
        rule (str): The rule's name.
        wildcards: The job's wildcards (a Snakemake ``Wildcards`` or a dict).
        attempt (int | None): The attempt number, from the rule's ``attempt`` resource, or None
            if the rule does not pass one.

    Returns:
        str: One line, without a trailing newline.
    """
    items = wildcards.items() if hasattr(wildcards, "items") else []
    named = ", ".join(f"{name}={value}" for name, value in items) or "no wildcards"
    job = os.environ.get("SLURM_JOB_ID")
    node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    return (
        f"=== {rule} ({named}), attempt {attempt if attempt is not None else 'unknown'}, "
        f"{f'SLURM job {job}' if job else 'not a SLURM job'}, node {node}, started {started} ==="
    )


def _exit_on_sigterm(signum, frame):
    """Turn SIGTERM into SystemExit, so ``finally`` blocks and context managers run.

    SLURM sends SIGTERM when a job reaches its time limit, or is cancelled, and SIGKILL 30
    seconds later. By default SIGTERM ends Python at once, skipping all cleanup. A second
    SIGTERM is ignored, so it cannot interrupt the cleanup the first one started.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(
        f"Received SIGTERM at {datetime.now().astimezone().isoformat(timespec='seconds')} "
        "(a time limit or a cancel). Exiting after cleanup.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(128 + signum)


def _append_line(paths, line):
    for path in dict.fromkeys(paths):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def setup_snakemake_logging(snakemake) -> logging.Logger:
    """Send a Snakemake script's output to its rule's log files, keeping earlier attempts.

    Appends to the rule's stdout and stderr logs, starting with a line from attempt_header in
    each, then points file descriptors 1 and 2 at them. Everything the job writes lands in
    them: the script's prints and logging (stdout log), warnings and tracebacks (stderr log),
    and the output of child processes such as Dask workers, which inherit the descriptors.

    It also installs a SIGTERM handler that raises ``SystemExit(143)``, so ``finally`` blocks
    and context managers run when SLURM ends the job at its time limit: the temporary zarr
    cache of ``compute_windowed_analysis`` is removed, and the Dask cluster is closed. SLURM
    sends SIGKILL 30 seconds after SIGTERM. A second SIGTERM is ignored.

    Before, the stdout log was opened in write mode, so a retried job erased the log of the
    attempt that failed, and both Python streams went to it while the stderr log stayed empty.
    Only Python-level output was redirected, so Dask workers wrote to the SLURM job's own
    output files instead, and SIGTERM ended the script without any cleanup.

    Args:
        snakemake: The global snakemake object injected by Snakemake. Its ``log`` names the
            log files (see log_paths); ``rule``, ``wildcards`` and the ``attempt`` resource,
            which rule_resources gives every rule, go into the header.

    Returns:
        logging.Logger: A logger whose records go to the stdout log at INFO and above.

    Example:
        In a Snakemake script::

            from neurodent.workflow import setup_snakemake_logging

            def main():
                logger = setup_snakemake_logging(snakemake)
                logger.info("Starting processing...")

            if __name__ == "__main__":
                main()

    Note:
        The log file paths come from the ``log:`` directive in your Snakemake rule. Log files
        use ``.out`` and ``.err`` extensions and are organized under ``logs/<rule_group>/``::

            rule my_rule:
                log:
                    stdout="logs/my_rule/{animal}.out",
                    stderr="logs/my_rule/{animal}.err",
                script: "scripts/my_script.py"
    """
    stdout_path, stderr_path = log_paths(snakemake.log)
    attempt = getattr(snakemake.resources, "attempt", None)
    _append_line((stdout_path, stderr_path), attempt_header(snakemake.rule, snakemake.wildcards, attempt))

    # Point the process's own descriptors at the logs, so that child processes and output
    # written below Python (C extensions, faulthandler) land there too
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.flush()
    for fd, path in ((1, stdout_path), (2, stderr_path)):
        # O_BINARY, on Windows only, so the text streams below translate newlines once
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        log_fd = os.open(path, flags, 0o644)
        os.dup2(log_fd, fd)
        os.close(log_fd)
    # Line buffered, so the logs stay current while the job runs and lose nothing if it is killed
    sys.stdout = open(1, "w", buffering=1, encoding="utf-8", errors="backslashreplace", closefd=False)
    sys.stderr = open(2, "w", buffering=1, encoding="utf-8", errors="backslashreplace", closefd=False)

    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, stream=sys.stdout, force=True)

    # Only the main thread may install a signal handler
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _exit_on_sigterm)

    return logging.getLogger(__name__)


@contextlib.contextmanager
def rule_log(log, rule, wildcards, attempt=None):
    """Log a ``run:`` block to its rule's log files, keeping every earlier attempt.

    The counterpart of setup_snakemake_logging for a rule with a ``run:`` block. It appends a
    line from attempt_header to the stdout and stderr logs and yields a logger that appends to
    the stdout log. An exception that leaves the block is written to the stderr log, with its
    traceback, before it propagates to Snakemake.

    Unlike setup_snakemake_logging it leaves file descriptors 1 and 2 alone, because a ``run:``
    block executes inside Snakemake's own process: in a thread of it with
    ``--force-use-threads``, which every cluster job uses. Redirecting them would also send
    Snakemake's own output, and that of any other job in the process, to this rule's logs.

    Args:
        log: The block's ``log``.
        rule (str): The block's ``rule``, the rule's name.
        wildcards: The block's ``wildcards``.
        attempt (int, optional): ``resources.attempt``, which rule_resources gives every rule.

    Yields:
        logging.Logger: A logger, of its own and outside the logging hierarchy, that writes to
        the stdout log at INFO and above.

    Example:
        In a rule::

            run:
                with rule_log(log, rule, wildcards, resources.attempt) as logger:
                    logger.info("Starting")
    """
    stdout_path, stderr_path = log_paths(log)
    _append_line((stdout_path, stderr_path), attempt_header(rule, wildcards, attempt))
    logger = logging.Logger(f"{rule} run block", level=logging.INFO)
    handler = logging.FileHandler(stdout_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    try:
        yield logger
    except BaseException:
        with open(stderr_path, "a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


def increment_memory(base_memory):
    """Return a callable ``mem(wildcards, attempt)`` that doubles on each retry.

    The pipeline's rules no longer call it directly: ``rule_resources`` uses it for a rule whose
    memory does not depend on its inputs. In a rule of your own::

        resources:
            mem_mb=increment_memory(4000),
    """
    def mem(wildcards, attempt):
        return base_memory * (2 ** (attempt - 1))
    return mem
