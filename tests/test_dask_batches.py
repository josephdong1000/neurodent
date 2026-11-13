"""Dask tasks computed a batch at a time, each batch under a deadline.

compute_windowed_analysis used to gather every window of a recording in one dask.compute. On
a 51,000-window day the scheduler stalled decoding the workers' replies, their sends timed out,
and the client asked for everything again without end. compute_in_batches gathers one batch
at a time and gives each batch a deadline.
"""

import time

import numpy as np
import pytest

import dask
from distributed import Client, LocalCluster

from neurodent.core.utils import BatchTimeoutError, compute_in_batches
from neurodent.core.utils.dask_batches import batch_timeout

N = 23


@pytest.fixture
def client():
    with (
        LocalCluster(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None) as cluster,
        Client(cluster) as client,
    ):
        yield client


def _result(i, delay=0.0):
    """A result shaped like a window's: a dict of arrays and scalars that depend on i."""
    time.sleep(delay)
    rng = np.random.default_rng(i)
    return {"i": i, "rms": rng.random(4), "psd": (np.arange(5.0), rng.random((5, 4))), "x": float(i) ** 0.5}


def _tasks(n=N, delays=None):
    delays = delays or [0.0] * n
    return [dask.delayed(_result)(i, d) for i, d in zip(range(n), delays)]


def _assert_same(results, expected):
    assert len(results) == len(expected)
    for got, want in zip(results, expected):
        assert got["i"] == want["i"] and got["x"] == want["x"]
        np.testing.assert_array_equal(got["rms"], want["rms"])
        for a, b in zip(got["psd"], want["psd"]):
            np.testing.assert_array_equal(a, b)


def _scheduler_tasks(client):
    return client.run_on_scheduler(lambda dask_scheduler: len(dask_scheduler.tasks))


def _wait_for_no_tasks(client, seconds=10):
    deadline = time.monotonic() + seconds
    while _scheduler_tasks(client) and time.monotonic() < deadline:
        time.sleep(0.05)
    return _scheduler_tasks(client)


@pytest.mark.parametrize("batch_size", [1, 3, 7, N, N + 5])
def test_batches_give_the_results_of_one_compute(client, batch_size):
    expected = dask.compute(*_tasks())

    results = compute_in_batches(_tasks(), batch_size)

    _assert_same(results, expected)
    assert _wait_for_no_tasks(client) == 0


@pytest.mark.parametrize("batch_size", [1, 4, N + 5])
def test_without_a_client_each_batch_is_one_dask_compute(batch_size, monkeypatch):
    expected = dask.compute(*_tasks(), scheduler="sync")
    calls = []
    real_compute = dask.compute

    def counting_compute(*args, **kwargs):
        calls.append(len(args))
        return real_compute(*args, scheduler="sync")

    monkeypatch.setattr(dask, "compute", counting_compute)
    results = compute_in_batches(_tasks(), batch_size)

    _assert_same(results, expected)
    assert calls == [min(batch_size, N - start) for start in range(0, N, batch_size)]


def test_results_keep_the_order_of_the_tasks(client):
    """Later tasks finish first here, and the results still follow the tasks."""
    delays = [0.02 * (N - i) for i in range(N)]

    results = compute_in_batches(_tasks(delays=delays), 5)

    assert [r["i"] for r in results] == list(range(N))


def test_a_batch_past_its_deadline_raises_fast_and_leaves_no_work(client):
    """Batch 3 holds a task far slower than the deadline its predecessors set."""
    delays = [0.0] * 8
    delays[5] = 4.0
    tasks = _tasks(8, delays)

    started = time.monotonic()
    with pytest.raises(BatchTimeoutError, match=r"Batch 3 of 4 \(tasks 4 to 5\) did not finish within 1 s"):
        compute_in_batches(tasks, 2, first_timeout_s=60, timeout_factor=1.0, min_timeout_s=1.0)
    elapsed = time.monotonic() - started

    assert elapsed < 3.5, f"took {elapsed:.1f} s to give up"
    assert _wait_for_no_tasks(client) == 0


def test_without_deadlines_a_slow_batch_finishes(client):
    delays = [0.0] * 6
    delays[4] = 1.5

    results = compute_in_batches(_tasks(6, delays), 2, first_timeout_s=None, min_timeout_s=0.01)

    assert [r["i"] for r in results] == list(range(6))


def test_an_error_in_a_task_propagates_and_cancels_the_rest(client):
    def fails(i):
        raise RuntimeError(f"window {i} failed")

    tasks = _tasks(10)
    tasks[4] = dask.delayed(fails)(4)

    with pytest.raises(RuntimeError, match="window 4 failed"):
        compute_in_batches(tasks, 3)
    assert _wait_for_no_tasks(client) == 0


def test_no_tasks_give_no_results(client):
    assert compute_in_batches([], 5) == []


def test_a_batch_size_below_one_raises():
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        compute_in_batches(_tasks(3), 0)


class TestBatchTimeout:
    def test_the_first_batch_gets_the_first_timeout(self):
        assert batch_timeout(720, None, first_timeout_s=3600) == 3600

    def test_later_batches_scale_with_their_tasks_and_the_slowest_rate(self):
        assert batch_timeout(720, 0.25, timeout_factor=10, min_timeout_s=600) == 1800
        assert batch_timeout(1440, 0.25, timeout_factor=10, min_timeout_s=600) == 3600

    def test_a_fast_batch_still_gets_the_least_timeout(self):
        assert batch_timeout(720, 0.01, timeout_factor=10, min_timeout_s=600) == 600
