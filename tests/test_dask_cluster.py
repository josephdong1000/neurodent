"""The local Dask cluster of war_generation is sized from its SLURM job.

The cluster used to take Dask's defaults: no memory limit of its own, so each worker took a
share of the whole node's memory, a dashboard fixed at port 8787, so two jobs on one node
collided, and only WARN messages and above in the log.
"""

import logging

import pytest

from neurodent.workflow.utils import local_cluster_kwargs
from neurodent.workflow.utils.dask_cluster import WORKER_MEMORY_SHARE


@pytest.mark.parametrize(
    "n_workers, mem_mb", [(4, 80_000), (1, 4_000), (8, 160_000), (3, 50_000)]
)
def test_half_the_job_is_split_between_the_workers(n_workers, mem_mb):
    kwargs = local_cluster_kwargs(n_workers, mem_mb)

    assert WORKER_MEMORY_SHARE == 0.5
    assert kwargs["n_workers"] == n_workers
    assert kwargs["threads_per_worker"] == 1
    assert kwargs["memory_limit"] == int(mem_mb * 2**20 * 0.5 / n_workers)
    assert kwargs["memory_limit"] * n_workers <= mem_mb * 2**20 / 2


def test_the_iqsec2_job_gives_each_worker_ten_gb():
    # war_generation: 4 threads and 80,000 MB
    assert local_cluster_kwargs(4, 80_000)["memory_limit"] == 10_485_760_000


def test_the_dashboard_takes_a_free_port_and_info_reaches_the_log():
    kwargs = local_cluster_kwargs(4, 80_000)
    assert kwargs["dashboard_address"] == ":0"
    assert kwargs["silence_logs"] == logging.INFO


@pytest.mark.parametrize("n_workers, mem_mb", [(0, 80_000), (4, 0), (4, None), (4, -1)])
def test_a_cluster_that_cannot_be_sized_is_refused(n_workers, mem_mb):
    with pytest.raises(ValueError):
        local_cluster_kwargs(n_workers, mem_mb)


def test_two_clusters_on_one_node_start_with_the_limit():
    from dask.distributed import Client, LocalCluster

    kwargs = local_cluster_kwargs(1, 1_000)
    # In-process workers keep the test light; the settings are the same ones a job passes
    with (
        LocalCluster(**kwargs, processes=False) as first,
        LocalCluster(**kwargs, processes=False) as second,
        Client(first) as client,
    ):
        (worker,) = client.scheduler_info()["workers"].values()
        assert worker["memory_limit"] == kwargs["memory_limit"]
        ports = {cluster.scheduler.http_server.port for cluster in (first, second)}
        assert len(ports) == 2 and 8787 not in ports
