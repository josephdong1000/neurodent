"""Settings for the local Dask cluster that war_generation starts inside its SLURM job."""

import logging

WORKER_MEMORY_SHARE = 0.5
"""Share of the job's memory given to the Dask workers, split evenly between them.

The other half is left to the main process, which holds the recordings, gathers the result of
every window and runs spike detection.
"""


def local_cluster_kwargs(n_workers, mem_mb):
    """Return keyword arguments for ``dask.distributed.LocalCluster``, sized from the job.

    - ``n_workers`` workers of one thread each, as before.
    - ``memory_limit``: ``WORKER_MEMORY_SHARE`` of the job's memory, split across the workers,
      so that Dask spills, pauses or restarts a worker before SLURM kills the job. Before, the
      cluster had no limit of its own and Dask divided the memory it found on the node between
      the workers, about 245 GiB each on the nodes of the IQSEC2 run, far above the job's limit.
    - ``dashboard_address`` ":0": the dashboard takes a free port. Before, it was fixed at 8787,
      so two jobs on one node collided.
    - ``silence_logs`` INFO: Dask's INFO messages reach the job's log. Before, only WARN and
      above did.

    Args:
        n_workers (int): Workers to start, the job's ``threads``.
        mem_mb (int): The job's memory in MB, its ``mem_mb`` resource. The profile passes it to
            SLURM's ``--mem``, which counts in units of 2**20 bytes.

    Returns:
        dict: Keyword arguments for ``LocalCluster``, with ``memory_limit`` in bytes.

    Raises:
        ValueError: If ``n_workers`` is less than 1 or ``mem_mb`` is not positive.
    """
    if n_workers < 1:
        raise ValueError(f"A Dask cluster needs at least one worker, got n_workers={n_workers}")
    if not mem_mb or mem_mb <= 0:
        raise ValueError(f"The job's memory must be positive to size the workers, got mem_mb={mem_mb}")
    return {
        "n_workers": n_workers,
        "threads_per_worker": 1,
        "memory_limit": int(mem_mb * 2**20 * WORKER_MEMORY_SHARE / n_workers),
        "dashboard_address": ":0",
        "silence_logs": logging.INFO,
    }
