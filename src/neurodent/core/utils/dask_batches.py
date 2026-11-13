"""Compute many Dask tasks a batch at a time, each batch under a deadline.

One ``dask.compute`` over a whole recording day's windows ends in a single gather: each worker
sends every result it holds in one reply, and the scheduler, which gathers on the client's
behalf, decodes those replies on its event loop. On 2026-11-11, a 51,000-window day of
IQSEC2_80 (replies of about 12,750 results each) stalled that loop for 32.7 to 97.4 s,
together with the garbage collections the decoding set off. The workers' sends timed out
after 30 s, and the client asked for every missing result again, round after round, with no
limit (scripts/diagnostics/gather_hang.py). That is how war_generation hung for 12 hours.

Here each batch's results are collected into one list on a worker, and the client fetches that
one result with a deadline. The scheduler then relays one key per batch, the next batch
computes while the client fetches the current one, and a batch that does not finish in time
raises instead of running out the job's clock.
"""

import asyncio
import logging
import time
from typing import Optional, Sequence

import dask

FIRST_TIMEOUT_S = 3600.0
"""Seconds the first batch may take. Nothing has been measured yet, and it also waits for the
workers to start and read their first zarr chunk."""

TIMEOUT_FACTOR = 10.0
"""Later batches may take this many times their tasks at the slowest rate seen so far."""

MIN_TIMEOUT_S = 600.0
"""No batch after the first gets less than this many seconds."""

logger = logging.getLogger(__name__)


class BatchTimeoutError(TimeoutError):
    """A batch of Dask tasks did not finish within its deadline."""


def batch_timeout(
    n_tasks: int,
    slowest_s_per_task: Optional[float],
    first_timeout_s: float = FIRST_TIMEOUT_S,
    timeout_factor: float = TIMEOUT_FACTOR,
    min_timeout_s: float = MIN_TIMEOUT_S,
) -> float:
    """Seconds a batch of ``n_tasks`` may take.

    Args:
        n_tasks (int): Tasks in the batch.
        slowest_s_per_task (float or None): The most seconds per task any earlier batch took,
            or None before any batch has finished.
        first_timeout_s (float): The deadline while nothing has been measured.
        timeout_factor (float): How many times its expected duration a batch may take.
        min_timeout_s (float): The least deadline of a measured batch.

    Returns:
        float: ``first_timeout_s`` for the first batch, then
        ``max(min_timeout_s, timeout_factor * n_tasks * slowest_s_per_task)``. A dataset with
        more channels or longer windows is slower per task, so its batches get longer.
    """
    if slowest_s_per_task is None:
        return first_timeout_s
    return max(min_timeout_s, timeout_factor * n_tasks * slowest_s_per_task)


def _collect(*results):
    """The results of one batch, as one list, so that the client fetches a single key."""
    return list(results)


def compute_in_batches(
    tasks: Sequence,
    batch_size: int,
    *,
    first_timeout_s: Optional[float] = FIRST_TIMEOUT_S,
    timeout_factor: float = TIMEOUT_FACTOR,
    min_timeout_s: float = MIN_TIMEOUT_S,
    label: str = "task",
) -> list:
    """Compute Dask tasks in consecutive batches and return their results in order.

    With a ``dask.distributed`` client, each batch is submitted as one task that collects its
    results on a worker, and the next batch is submitted before the current one is waited on,
    so the workers stay busy. Each batch must finish within :func:`batch_timeout` seconds of
    the previous one, or every batch still running is cancelled and
    :class:`BatchTimeoutError` names the batch. Without a client, each batch is computed with
    ``dask.compute`` and no deadline, since nothing is gathered over a network.

    The results are the same as one ``dask.compute(*tasks)``: each task is computed once, on
    its own inputs, whatever the batch size.

    Args:
        tasks (Sequence): Dask delayed objects, or anything ``dask.compute`` accepts.
        batch_size (int): Tasks per batch, at least 1.
        first_timeout_s (float or None): The first batch's deadline, see
            :func:`batch_timeout`. None computes every batch without a deadline.
        timeout_factor (float): See :func:`batch_timeout`.
        min_timeout_s (float): See :func:`batch_timeout`.
        label (str): What a task is, for log and error messages, such as "window".

    Returns:
        list: One result per task, in the order of ``tasks``.

    Raises:
        ValueError: If ``batch_size`` is less than 1.
        BatchTimeoutError: If a batch passes its deadline.
        Exception: Whatever a task raises.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    tasks = list(tasks)
    bounds = [(start, min(start + batch_size, len(tasks))) for start in range(0, len(tasks), batch_size)]
    if not bounds:
        return []
    try:
        from distributed import get_client

        client = get_client()
    except (ImportError, ValueError):
        client = None
    if client is None:
        results = []
        for start, stop in bounds:
            results.extend(dask.compute(*tasks[start:stop]))
        return results
    return _compute_on_client(
        client, tasks, bounds, first_timeout_s, timeout_factor, min_timeout_s, label
    )


def _compute_on_client(client, tasks, bounds, first_timeout_s, timeout_factor, min_timeout_s, label):
    """The ``dask.distributed`` path of :func:`compute_in_batches`, two batches in flight."""

    def submit(i):
        start, stop = bounds[i]
        return client.compute(dask.delayed(_collect)(*tasks[start:stop]))

    n = len(bounds)
    pending = {0: submit(0)}
    if n > 1:
        pending[1] = submit(1)
    results, slowest, started = [], None, time.monotonic()
    try:
        for i, (start, stop) in enumerate(bounds):
            timeout = (
                None if first_timeout_s is None
                else batch_timeout(stop - start, slowest, first_timeout_s, timeout_factor, min_timeout_s)
            )
            try:
                batch = pending[i].result(timeout=timeout)
            # On Python 3.10 a Future's timeout is asyncio's TimeoutError, not the builtin
            except (TimeoutError, asyncio.TimeoutError) as e:
                raise BatchTimeoutError(
                    f"Batch {i + 1} of {n} ({label}s {start} to {stop - 1}) did not finish within "
                    f"{timeout:.0f} s. The batches still running were cancelled."
                ) from e
            del pending[i]
            if i + 2 < n:
                pending[i + 2] = submit(i + 2)
            finished = time.monotonic()
            per_task = (finished - started) / (stop - start)
            slowest = per_task if slowest is None else max(slowest, per_task)
            started = finished
            results.extend(batch)
            log = logger.info if (i + 1) % 10 == 0 or i + 1 == n else logger.debug
            log(f"Batch {i + 1} of {n}: {stop - start} {label}s in {per_task * (stop - start):.1f} s")
    finally:
        if pending:
            client.cancel(list(pending.values()))
    return results
