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


def attempts_path(path):
    """Return the file beside a declared log that keeps the output of every attempt.

    Snakemake deletes a rule's declared log files before each attempt of a job: when it submits
    the attempt (``ClusterExecutor._run``) and again inside the job before the rule runs
    (``Job.prepare``), both through ``Job.remove_existing_output``. A retry therefore erases
    the declared logs of the attempt that failed. Snakemake does not know of this file, so it
    never deletes it. It has ``.attempts`` before the extension of the declared log:
    ``logs/war_generation/A1.out`` is kept in ``logs/war_generation/A1.attempts.out``.

    Args:
        path (str): A declared log file.

    Returns:
        str: The attempts file of that log.
    """
    root, extension = os.path.splitext(str(path))
    return f"{root}.attempts{extension}"


def _kept_in(kept_stdout, kept_stderr, children):
    """The line that names, in a declared log, the attempts files that keep every attempt."""
    files = kept_stdout if kept_stdout == kept_stderr else f"{kept_stdout} and {kept_stderr}"
    tail = ", with the output of its child processes such as Dask workers" if children else ""
    return f"Every attempt of this job is kept in {files}{tail}."


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


def _append_lines(paths, lines):
    for path in dict.fromkeys(paths):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.writelines(line + "\n" for line in lines)


class _Tee:
    """A text stream that writes to several text streams.

    The first stream is the one on the process's file descriptor. Anything the tee does not
    define itself, such as ``fileno``, ``encoding`` or ``isatty``, comes from that stream.
    """

    def __init__(self, descriptor_stream, *copies):
        self._descriptor_stream = descriptor_stream
        self._streams = (descriptor_stream, *copies)

    def write(self, text):
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        for stream in self._streams:
            if not stream.closed:
                stream.flush()

    def __getattr__(self, name):
        if name in ("_descriptor_stream", "_streams"):
            raise AttributeError(name)
        return getattr(self._descriptor_stream, name)


def _line_buffered(target):
    """Open a descriptor, or a file for appending, as line-buffered text.

    Line buffering keeps the logs current while the job runs, and loses nothing if it is killed.
    """
    mode, closefd = ("w", False) if isinstance(target, int) else ("a", True)
    return open(
        target, mode, buffering=1, encoding="utf-8", errors="backslashreplace", closefd=closefd
    )


def setup_snakemake_logging(snakemake) -> logging.Logger:
    """Send a Snakemake script's output to its rule's log files, and keep every attempt's.

    Snakemake deletes the rule's declared log files before each attempt of the job, so they
    hold the current attempt only. Each attempt is also appended to the attempts files beside
    them (see attempts_path), which Snakemake never deletes, after a line from attempt_header
    that names the attempt, its SLURM job and node, and its start time:

    - The script's prints and logging go to the stdout log and its attempts file, and its
      warnings and tracebacks to the stderr log and its attempts file.
    - File descriptors 1 and 2 point at the attempts files. Output written to the descriptors
      directly lands there only, not in the declared logs: that of child processes such as
      Dask workers, which inherit the descriptors, of code below Python, and of logging
      handlers made before this call, such as Dask's own.
    - The declared logs start with the header and a line naming the attempts files. The job's
      own stderr, which SLURM writes to the job's output file, gets the same two lines.

    It also installs a SIGTERM handler that raises ``SystemExit(143)``, so ``finally`` blocks
    and context managers run when SLURM ends the job at its time limit: the temporary zarr
    cache of ``compute_windowed_analysis`` is removed, and the Dask cluster is closed. SLURM
    sends SIGKILL 30 seconds after SIGTERM. A second SIGTERM is ignored.

    Before, the stdout log was opened in write mode and both Python streams went to it, while
    the stderr log stayed empty. A retry lost the failed attempt's log, which Snakemake deletes
    anyway. Only Python-level output was redirected, so Dask workers wrote to the SLURM job's
    own output files, and SIGTERM ended the script without any cleanup.

    Args:
        snakemake: The global snakemake object injected by Snakemake. Its ``log`` names the
            log files (see log_paths); ``rule``, ``wildcards`` and the ``attempt`` resource,
            which rule_resources gives every rule, go into the header.

    Returns:
        logging.Logger: A logger whose records go to the stdout log and its attempts file at
        INFO and above.

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
    kept_stdout, kept_stderr = attempts_path(stdout_path), attempts_path(stderr_path)
    attempt = getattr(snakemake.resources, "attempt", None)
    header = attempt_header(snakemake.rule, snakemake.wildcards, attempt)
    kept_in = _kept_in(kept_stdout, kept_stderr, children=True)
    _append_lines((kept_stdout, kept_stderr), [header])
    _append_lines((stdout_path, stderr_path), [header, kept_in])
    if sys.stderr is not None:
        print(header, kept_in, sep="\n", file=sys.stderr, flush=True)

    # Point the process's own descriptors at the attempts files, so that child processes and
    # output written below Python (C extensions, faulthandler) are kept there too
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.flush()
    for fd, path in ((1, kept_stdout), (2, kept_stderr)):
        # O_BINARY, on Windows only, so the text streams below translate newlines once
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        kept_fd = os.open(path, flags, 0o644)
        os.dup2(kept_fd, fd)
        os.close(kept_fd)
    declared_stdout = _line_buffered(stdout_path)
    declared_stderr = declared_stdout if stderr_path == stdout_path else _line_buffered(stderr_path)
    sys.stdout = _Tee(_line_buffered(1), declared_stdout)
    sys.stderr = _Tee(_line_buffered(2), declared_stderr)

    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, stream=sys.stdout, force=True)

    # Only the main thread may install a signal handler
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _exit_on_sigterm)

    return logging.getLogger(__name__)


@contextlib.contextmanager
def rule_log(log, rule, wildcards, attempt=None):
    """Log a ``run:`` block to its rule's log files, and keep every attempt's.

    The counterpart of setup_snakemake_logging for a rule with a ``run:`` block. Snakemake
    deletes the declared logs before each attempt, so they hold the current attempt only, and
    each attempt is also appended to the attempts files beside them (see attempts_path). It
    writes a line from attempt_header to all four files, and to the declared logs a line naming
    the attempts files, then yields a logger that writes to the stdout log and its attempts
    file. An exception that leaves the block is written, with its traceback, to the stderr log
    and its attempts file before it propagates to Snakemake.

    Unlike setup_snakemake_logging it leaves file descriptors 1 and 2 alone, because a ``run:``
    block executes inside Snakemake's own process: in a thread of it with
    ``--force-use-threads``, which every cluster job uses. Redirecting them would also send
    Snakemake's own output, and that of any other job in the process, to this rule's logs.
    The block's prints, and the output of its child processes, go to Snakemake's own streams.

    Args:
        log: The block's ``log``.
        rule (str): The block's ``rule``, the rule's name.
        wildcards: The block's ``wildcards``.
        attempt (int, optional): ``resources.attempt``, which rule_resources gives every rule.

    Yields:
        logging.Logger: A logger, of its own and outside the logging hierarchy, that writes to
        the stdout log and its attempts file at INFO and above.

    Example:
        In a rule::

            run:
                with rule_log(log, rule, wildcards, resources.attempt) as logger:
                    logger.info("Starting")
    """
    stdout_path, stderr_path = log_paths(log)
    kept_stdout, kept_stderr = attempts_path(stdout_path), attempts_path(stderr_path)
    header = attempt_header(rule, wildcards, attempt)
    kept_in = _kept_in(kept_stdout, kept_stderr, children=False)
    _append_lines((kept_stdout, kept_stderr), [header])
    _append_lines((stdout_path, stderr_path), [header, kept_in])
    logger = logging.Logger(f"{rule} run block", level=logging.INFO)
    handlers = [
        logging.FileHandler(path, mode="a", encoding="utf-8") for path in (stdout_path, kept_stdout)
    ]
    for handler in handlers:
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    try:
        yield logger
    except BaseException:
        _append_lines((stderr_path, kept_stderr), [traceback.format_exc().rstrip("\n")])
        raise
    finally:
        for handler in handlers:
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
