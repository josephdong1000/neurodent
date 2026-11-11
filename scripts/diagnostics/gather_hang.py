#!/usr/bin/env python3
"""Reproduce the Dask gather hang seen in war_generation, with instruments that show its cause.

The first war_generation attempts for IQSEC2_80 and IQSEC2_81 hung for 12 hours on m-09 nodes
and their retries finished in 3.5 hours on c-11. The worker traceback failed in
Worker.get_data: the main process did not read the results within the 30 s connection timeout,
Dask's gather kept nothing it had received, and the client asked for every key again, with no
limit. One gather moved all 51,000 window results of one recording day, about 6 GB.

Two modes run the same LocalCluster as workflow/scripts/generate_wars.py (one thread per
worker, the default memory limit and dashboard):

- synthetic: N tasks that each return about --result-kb of nested numpy arrays, gathered at
  once or in batches of --batch. No NeuRodent I/O.
- windowed: one animal of a dataset config, optionally one of its sources, loaded as
  war_generation loads it, then compute_windowed_analysis in dask mode with the configured
  features and chunk size. LOF is skipped unless --with-lof is given.

Instruments, all under --out:

- stacks.txt: every thread's stack, each minute and on a fatal signal such as a segfault
  (faulthandler, which needs no GIL).
- events.jsonl: each dask.compute call's start and end, garbage collections over 0.5 s,
  event-loop lag over 1 s on the cluster's loop, and log records from distributed about
  gathers, unresponsive loops, failed transfers and lost workers.
- scheduler.jsonl: from inside the scheduler, the results held in memory and their bytes,
  every --every results, and each graph submitted.
- summary.json: timings, counts, the settings the workers saw, and how the run ended.

A watchdog ends a run that stays stuck. When a dask.compute call has seen no new result for
--stuck-min minutes, or the whole run passes --deadline-min, it dumps the stacks, writes the
summary and exits with code 3.

Dask settings are passed through the environment before dask is imported, so the worker
processes see them too, and the summary records what the workers report.

Usage, inside a cluster job (see gather_hang.sbatch):

    python scripts/diagnostics/gather_hang.py synthetic --out DIR --n-tasks 51000
    python scripts/diagnostics/gather_hang.py synthetic --out DIR --n-tasks 51000 --batch 720
    python scripts/diagnostics/gather_hang.py windowed --out DIR --dataset iqsec2 \\
        --animal IQSEC2_80 --source 1
"""

import argparse
import faulthandler
import gc
import json
import logging
import os
import socket
import sys
import threading
import time
import warnings
from collections import Counter
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]

LOG_PATTERNS = (
    "Couldn't gather",
    "unresponsive",
    "failed during get data",
    "Removing worker",
    "Lost connection",
    "Timed out",
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["synthetic", "windowed"])
    ap.add_argument("--out", required=True, help="Directory for the instruments' files.")
    ap.add_argument("--workers", type=int, default=4, help="Worker processes, one thread each.")
    ap.add_argument("--tcp-timeout", default="30s", help="distributed.comm.timeouts.tcp")
    ap.add_argument("--tick-limit", default="1s", help="distributed.admin.tick.limit")
    ap.add_argument("--cluster-log-level", default="INFO", help="LocalCluster silence_logs level.")
    ap.add_argument("--stuck-min", type=float, default=30, help="Minutes without a new result.")
    ap.add_argument("--deadline-min", type=float, default=300, help="Minutes for the whole run.")
    ap.add_argument("--every", type=int, default=500, help="Results between scheduler records.")
    ap.add_argument("--n-tasks", type=int, default=51000, help="synthetic: tasks.")
    ap.add_argument("--result-kb", type=int, default=120, help="synthetic: size of each result.")
    ap.add_argument("--batch", type=int, default=0, help="synthetic: tasks per gather; 0 is all.")
    ap.add_argument("--dataset", help="windowed: dataset config name.")
    ap.add_argument("--animal", help="windowed: animal id as the config writes it.")
    ap.add_argument("--source", type=int, help="windowed: keep only this index of the animal's sources.")
    ap.add_argument("--temp-dir", help="windowed: temp directory for the zarr cache; default OUT/tmp.")
    ap.add_argument("--with-lof", action="store_true", help="windowed: run compute_bad_channels first.")
    args = ap.parse_args(argv)
    if args.mode == "windowed" and not (args.dataset and args.animal):
        ap.error("windowed needs --dataset and --animal")
    return args


class Recorder:
    """Appends timestamped events to a JSON lines file from any thread, and counts them."""

    def __init__(self, path):
        self.fh = open(path, "a", buffering=1, encoding="utf-8")
        self.lock = threading.Lock()
        self.t0 = time.monotonic()
        self.counts = Counter()

    def event(self, kind, **fields):
        record = {"t": round(time.monotonic() - self.t0, 3), "wall": time.time(), "kind": kind, **fields}
        with self.lock:
            self.counts[kind] += 1
            self.fh.write(json.dumps(record, default=str) + "\n")


class PatternCounter(logging.Handler):
    """Records log records from distributed whose message holds one of LOG_PATTERNS."""

    def __init__(self, recorder):
        super().__init__(level=logging.INFO)
        self.recorder = recorder
        self.counts = Counter()

    def emit(self, record):
        message = record.getMessage()
        for pattern in LOG_PATTERNS:
            if pattern in message:
                self.counts[pattern] += 1
                self.recorder.event("log", pattern=pattern, logger=record.name, message=message[:300])
                break


def scheduler_plugin_class():
    """The scheduler plugin, built after dask is imported so the environment settings apply."""
    from distributed.diagnostics.plugin import SchedulerPlugin

    class ResultsInMemory(SchedulerPlugin):
        """Records, inside the scheduler, the results held in memory and their bytes."""

        name = "gather-hang-results-in-memory"

        def __init__(self, path, every):
            self.path = str(path)
            self.every = every

        async def start(self, scheduler):
            self.scheduler = scheduler
            self.n_memory = 0
            self.bytes_memory = 0
            self.n_total = 0
            self.fh = open(self.path, "a", buffering=1, encoding="utf-8")

        def _nbytes(self, key):
            ts = self.scheduler.tasks.get(key)
            nbytes = getattr(ts, "nbytes", None)
            return nbytes if isinstance(nbytes, int) and nbytes > 0 else 0

        def _write(self, kind, **fields):
            record = {"wall": time.time(), "kind": kind, "n_memory": self.n_memory,
                      "bytes_memory": self.bytes_memory, "n_total": self.n_total, **fields}
            self.fh.write(json.dumps(record) + "\n")

        def transition(self, key, start, finish, *args, stimulus_id=None, **kwargs):
            if finish == "memory":
                self.n_memory += 1
                self.n_total += 1
                self.bytes_memory += self._nbytes(key)
                if self.n_total % self.every == 0:
                    self._write("results")
            elif start == "memory":
                self.n_memory -= 1
                self.bytes_memory -= self._nbytes(key)

        def update_graph(self, scheduler, *args, keys=None, **kwargs):
            self._write("graph", n_keys=len(keys) if keys is not None else None)

    return ResultsInMemory


def make_result(index, rows):
    """A window-like result: nested arrays of about rows * 64 bytes, as feature dicts are."""
    import numpy as np

    rng = np.random.default_rng(index)
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    return {
        "psd": (np.arange(rows, dtype=float), rng.random((rows, 8))),
        "cohere": {band: rng.random((8, 8)) for band in bands},
        "rms": rng.random(8),
    }


def run_synthetic(args, recorder):
    import dask
    from dask import delayed

    rows = max(1, args.result_kb * 1024 // 64)
    tasks = [delayed(make_result)(i, rows) for i in range(args.n_tasks)]
    batch = args.batch or len(tasks)
    n_results = 0
    for start in range(0, len(tasks), batch):
        results = dask.compute(*tasks[start : start + batch])
        n_results += len(results)
        del results
    recorder.event("synthetic_done", n_results=n_results)
    return {"n_results": n_results, "rows_per_result": rows}


def run_windowed(args, recorder):
    from neurodent import core
    from neurodent.analysis import AnimalAnalyzer
    from neurodent.workflow import apply_samples_config
    from neurodent.workflow.utils import load_animal_recordings
    from neurodent.workflow.utils.config import (
        expand_animals_config,
        load_dataset_config,
        resolve_samples_config,
    )

    os.chdir(TREE)  # dataset configs are read relative to the tree
    config = load_dataset_config(args.dataset)
    samples = resolve_samples_config(config)
    entries = [entry for entry in samples["animals"] if entry["id"] == args.animal]
    if len(entries) != 1:
        raise SystemExit(f"{args.animal} appears {len(entries)} times in {args.dataset}")
    entry = dict(entries[0])
    if args.source is not None:
        sources = entry.get("sources") or []
        if not 0 <= args.source < len(sources):
            raise SystemExit(f"{args.animal} has {len(sources)} sources; --source {args.source} is out of range")
        entry["sources"] = [sources[args.source]]
    samples["animals"] = [entry]
    samples = expand_animals_config(samples)
    apply_samples_config(samples)
    core.utils.set_temp_directory(args.temp_dir or Path(args.out) / "tmp")

    recorder.event("load_start")
    ao = load_animal_recordings(
        samples, config, [("", args.animal, "")], args.animal,
        channel_subset=samples.get("_animal_channel_subsets", {}).get(args.animal),
    )
    recorder.event("load_end", animaldays=list(ao.animaldays))
    az = AnimalAnalyzer(ao)
    if args.with_lof:
        lof = config["analysis"]["channel_filter_config"]["lof"]
        az.compute_bad_channels(
            lof_threshold=lof.get("reject_lof_threshold"),
            lof_chunk_duration_s=lof.get("lof_chunk_duration_s", 60),
        )
    cwa = config["analysis"]["war_generation"].get("compute_windowed_analysis", {})
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*fmin=.*Spectrum estimate will be unreliable.*", category=RuntimeWarning
        )
        war = az.compute_windowed_analysis(
            cwa.get("features", ["all"]),
            multiprocess_mode="dask",
            chunk_duration_s=cwa.get("chunk_duration_s", 3600),
        )
    recorder.event("windowed_done", windows=len(war.result))
    return {"windows": len(war.result), "animaldays": list(ao.animaldays)}


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Before dask is imported, here and in the worker processes, which inherit the environment
    os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP"] = args.tcp_timeout
    os.environ["DASK_DISTRIBUTED__ADMIN__TICK__LIMIT"] = args.tick_limit

    recorder = Recorder(out / "events.jsonl")
    stacks = open(out / "stacks.txt", "a", buffering=1, encoding="utf-8")
    faulthandler.enable(file=stacks, all_threads=True)
    faulthandler.dump_traceback_later(60, repeat=True, file=stacks)
    state = {"in_compute": False, "computes": [], "max_loop_lag": 0.0, "max_gc": 0.0, "status": "running"}
    summary_path = out / "summary.json"
    scheduler_path = out / "scheduler.jsonl"

    def write_summary(extra=None):
        summary = {
            "mode": args.mode, "args": vars(args), "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "status": state["status"],
            "elapsed_s": round(time.monotonic() - recorder.t0, 1), "computes": state["computes"],
            "max_loop_lag_s": round(state["max_loop_lag"], 3), "max_gc_s": round(state["max_gc"], 3),
            "event_counts": dict(recorder.counts), "log_counts": dict(pattern_counter.counts),
            **(extra or {}),
        }
        summary_path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

    gc_start = {}

    def on_gc(phase, info):
        if phase == "start":
            gc_start["t"] = time.perf_counter()
        elif "t" in gc_start:
            seconds = time.perf_counter() - gc_start.pop("t")
            state["max_gc"] = max(state["max_gc"], seconds)
            if seconds > 0.5:
                recorder.event("gc", seconds=round(seconds, 3), generation=info.get("generation"),
                               collected=info.get("collected"))

    gc.callbacks.append(on_gc)

    import dask
    from dask.distributed import Client, LocalCluster
    from tornado.ioloop import PeriodicCallback

    pattern_counter = PatternCounter(recorder)
    logging.getLogger("distributed").addHandler(pattern_counter)

    real_compute = dask.compute

    def timed_compute(*args_, **kwargs):
        start = time.monotonic()
        state["compute_started"] = time.time()  # before in_compute, which the watchdog reads first
        state["in_compute"] = True
        recorder.event("compute_start", n_args=len(args_))
        try:
            return real_compute(*args_, **kwargs)
        finally:
            state["in_compute"] = False
            seconds = round(time.monotonic() - start, 1)
            state["computes"].append({"n_args": len(args_), "seconds": seconds})
            recorder.event("compute_end", n_args=len(args_), seconds=seconds)

    dask.compute = timed_compute  # animal_analyzer calls dask.compute through the module

    stop = threading.Event()

    def watchdog():
        while not stop.wait(30):
            elapsed_min = (time.monotonic() - recorder.t0) / 60
            reason = None
            if elapsed_min > args.deadline_min:
                reason = "deadline"
            elif state["in_compute"]:
                last = scheduler_path.stat().st_mtime if scheduler_path.exists() else state["compute_started"]
                last = max(last, state["compute_started"])
                if (time.time() - last) / 60 > args.stuck_min:
                    reason = "stuck"
            if reason:
                state["status"] = f"watchdog_{reason}"
                recorder.event("watchdog", reason=reason, elapsed_min=round(elapsed_min, 1))
                faulthandler.dump_traceback(file=stacks, all_threads=True)
                write_summary()
                stacks.flush()
                os._exit(3)

    threading.Thread(target=watchdog, name="gather-hang-watchdog", daemon=True).start()

    result = {}
    try:
        with (
            LocalCluster(
                n_workers=args.workers,
                threads_per_worker=1,
                interface=None,
                silence_logs=getattr(logging, args.cluster_log_level.upper()),
            ) as cluster,
            Client(cluster) as client,
        ):
            seen = client.run(lambda: {
                "tcp": dask.config.get("distributed.comm.timeouts.tcp"),
                "tick_limit": dask.config.get("distributed.admin.tick.limit"),
            })
            recorder.event("cluster", dashboard=cluster.dashboard_link, workers_saw=seen,
                           memory_limits={w: info.get("memory_limit")
                                          for w, info in client.scheduler_info()["workers"].items()})
            client.register_plugin(scheduler_plugin_class()(scheduler_path, args.every))

            def start_probe():
                last = [time.monotonic()]

                def tick():
                    now = time.monotonic()
                    lag = now - last[0] - 1.0
                    last[0] = now
                    state["max_loop_lag"] = max(state["max_loop_lag"], lag)
                    if lag > 1.0:
                        recorder.event("loop_lag", seconds=round(lag, 3))

                state["probe"] = PeriodicCallback(tick, 1000)
                state["probe"].start()

            cluster.loop.add_callback(start_probe)
            result = run_synthetic(args, recorder) if args.mode == "synthetic" else run_windowed(args, recorder)
            state["status"] = "completed"
            result["workers_saw"] = seen
    except BaseException as e:
        state["status"] = f"error: {type(e).__name__}: {e}"
        raise
    finally:
        stop.set()
        faulthandler.cancel_dump_traceback_later()
        write_summary(result)
    print(json.dumps({"status": state["status"], "summary": str(summary_path)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
