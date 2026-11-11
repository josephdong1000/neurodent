"""Every attempt of a job keeps its log.

setup_snakemake_logging used to open the stdout log in write mode, so a retried job erased the
log of the attempt that failed, and it redirected only Python's streams, so Dask workers wrote
to the SLURM job's own files. It now appends, opens each attempt with a header line, and points
file descriptors 1 and 2 at the logs. It changes the process's own descriptors, so these tests
run it in a subprocess and never touch pytest's streams. rule_log, its counterpart for a run:
block, leaves the descriptors alone and is tested in process.
"""

import logging
import os
import re
import subprocess
import sys
import textwrap

import pytest

from neurodent.workflow.utils import logging as snakemake_logging
from neurodent.workflow.utils import rule_log, rule_resources
from neurodent.workflow.utils.logging import attempt_header, log_paths

HEADER = re.compile(
    r"^=== (?P<rule>\w+) \((?P<wildcards>[^)]*)\), attempt (?P<attempt>\w+), "
    r"(?P<job>SLURM job \d+|not a SLURM job), node (?P<node>\S+), "
    r"started \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\S* ===$"
)

DRIVER = textwrap.dedent(
    """
    import importlib.util, os, subprocess, sys, warnings
    from types import SimpleNamespace

    # The module alone, from its file: it needs only the standard library, and importing the
    # whole package in every subprocess would only slow the tests down
    spec = importlib.util.spec_from_file_location("snakemake_logging", sys.argv[5])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setup_snakemake_logging = module.setup_snakemake_logging

    class Log(list):
        def __init__(self, *paths, **named):
            super().__init__(paths or named.values())
            for name, path in named.items():
                setattr(self, name, path)

    stdout_path, stderr_path, attempt, mode = sys.argv[1:5]
    log = Log(stdout_path) if stderr_path == "-" else Log(stdout=stdout_path, stderr=stderr_path)
    snakemake = SimpleNamespace(
        log=log, rule="rule_x", wildcards={"animal": "a1"},
        resources=SimpleNamespace(attempt=int(attempt)),
    )
    logger = setup_snakemake_logging(snakemake)
    logger.info(f"logged by attempt {attempt}")
    print(f"printed by attempt {attempt}")
    warnings.warn(f"warned by attempt {attempt}")
    os.write(1, f"descriptor 1 by attempt {attempt}\\n".encode())
    if mode == "child":
        subprocess.run([sys.executable, "-c", (
            "import sys; print('child out {0}'); print('child err {0}', file=sys.stderr)"
        ).format(attempt)], check=True)
    if mode == "fail":
        raise RuntimeError(f"attempt {attempt} failed")
    """
)
DRIVER_LINES = ("logged by", "printed by", "warned by", "descriptor 1 by", "failed")


class _Log(list):
    """A rule's log files as Snakemake passes them: a list whose entries also have names."""

    def __init__(self, **named):
        super().__init__(named.values())
        for name, path in named.items():
            setattr(self, name, path)


def _env(**variables):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM")}
    env.update(variables)
    return env


def _run(tmp_path, stdout_path, stderr_path, attempt, mode="ok", env=None):
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(driver), str(stdout_path), str(stderr_path), str(attempt), mode,
         snakemake_logging.__file__],
        capture_output=True,
        text=True,
        env=env if env is not None else _env(),
        timeout=120,
    )


def _headers(text):
    return [HEADER.match(line) for line in text.splitlines() if line.startswith("=== ")]


def test_a_retry_keeps_the_failed_attempts_log(tmp_path):
    out, err = tmp_path / "logs" / "a1.out", tmp_path / "logs" / "a1.err"
    env = _env(SLURM_JOB_ID="4242", SLURMD_NODENAME="m-09-03")

    first = _run(tmp_path, out, err, 1, mode="fail", env=env)
    second = _run(tmp_path, out, err, 2, env=env)

    assert first.returncode != 0 and second.returncode == 0, second.stderr
    out_text, err_text = out.read_text(encoding="utf-8"), err.read_text(encoding="utf-8")
    for text in (out_text, err_text):
        headers = _headers(text)
        assert [header["attempt"] for header in headers] == ["1", "2"]
        assert {header["rule"] for header in headers} == {"rule_x"}
        assert {header["wildcards"] for header in headers} == {"animal=a1"}
        assert {header["job"] for header in headers} == {"SLURM job 4242"}
        assert {header["node"] for header in headers} == {"m-09-03"}
    # The first attempt's output and its traceback survive the second attempt
    assert out_text.index("logged by attempt 1") < out_text.index("logged by attempt 2")
    assert "RuntimeError: attempt 1 failed" in err_text
    assert err_text.index("Traceback") < err_text.index("attempt 2")
    # Nothing the job wrote reached its own streams, where SLURM would have caught it
    for result in (first, second):
        assert not [line for line in DRIVER_LINES if line in result.stdout + result.stderr]


def test_each_stream_goes_to_its_own_log(tmp_path):
    out, err = tmp_path / "a1.out", tmp_path / "a1.err"
    result = _run(tmp_path, out, err, 1)

    assert result.returncode == 0, result.stderr
    out_text, err_text = out.read_text(encoding="utf-8"), err.read_text(encoding="utf-8")
    for line in ("logged by attempt 1", "printed by attempt 1", "descriptor 1 by attempt 1"):
        assert line in out_text and line not in err_text
    assert "warned by attempt 1" in err_text and "warned by attempt 1" not in out_text
    assert _headers(out_text)[0]["job"] == "not a SLURM job"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows child processes inherit the console handles, not C file descriptors",
)
def test_child_processes_such_as_dask_workers_write_to_the_logs(tmp_path):
    out, err = tmp_path / "a1.out", tmp_path / "a1.err"
    result = _run(tmp_path, out, err, 3, mode="child")

    assert result.returncode == 0, result.stderr
    assert "child out 3" in out.read_text(encoding="utf-8")
    assert "child err 3" in err.read_text(encoding="utf-8")
    assert "child" not in result.stdout + result.stderr


def test_a_single_log_takes_both_streams(tmp_path):
    log = tmp_path / "a1.log"
    result = _run(tmp_path, log, "-", 1)

    assert result.returncode == 0, result.stderr
    text = log.read_text(encoding="utf-8")
    assert len(_headers(text)) == 1
    assert "logged by attempt 1" in text and "warned by attempt 1" in text


def test_log_paths_read_named_and_unnamed_logs():
    assert log_paths(_Log(stdout="x.out", stderr="x.err")) == ("x.out", "x.err")
    assert log_paths(["a.log"]) == ("a.log", "a.log")
    assert log_paths(["a.out", "a.err"]) == ("a.out", "a.err")
    with pytest.raises(ValueError, match="no log file"):
        log_paths([])


def test_the_header_without_an_attempt_or_wildcards(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("SLURMD_NODENAME", "c-11-25")
    header = HEADER.match(attempt_header("war_zeitgeber", {}, None))

    assert header["wildcards"] == "no wildcards"
    assert header["attempt"] == "unknown"
    assert header["job"] == "not a SLURM job"
    assert header["node"] == "c-11-25"


def test_every_rule_receives_its_attempt_number():
    resources = rule_resources({"time": "1h", "mem_mb": 1000, "nodes": 1})
    assert [resources["attempt"](None, attempt) for attempt in (1, 2, 3)] == [1, 2, 3]


def _block_log(tmp_path):
    return _Log(stdout=str(tmp_path / "q.out"), stderr=str(tmp_path / "q.err"))


def test_rule_log_appends_each_attempt(tmp_path):
    log = _block_log(tmp_path)
    for attempt in (1, 2):
        with rule_log(log, "war_quality_filter", {"animal": "a1"}, attempt) as logger:
            logger.info(f"kept by attempt {attempt}")

    out_text = (tmp_path / "q.out").read_text(encoding="utf-8")
    assert [header["attempt"] for header in _headers(out_text)] == ["1", "2"]
    assert out_text.index("kept by attempt 1") < out_text.index("kept by attempt 2")
    err_lines = (tmp_path / "q.err").read_text(encoding="utf-8").splitlines()
    assert len(err_lines) == 2 and all(HEADER.match(line) for line in err_lines)


def test_rule_log_writes_the_traceback_of_a_failure_and_reraises(tmp_path):
    log = _block_log(tmp_path)

    with pytest.raises(KeyError, match="genotype"):
        with rule_log(log, "war_quality_filter", {"animal": "a1"}, 1):
            raise KeyError("genotype")

    err_text = (tmp_path / "q.err").read_text(encoding="utf-8")
    assert "Traceback" in err_text and "KeyError: 'genotype'" in err_text


def test_rule_log_leaves_the_process_streams_and_root_logger_alone(tmp_path, capfd):
    log = _block_log(tmp_path)
    root_handlers = list(logging.getLogger().handlers)

    with rule_log(log, "war_quality_filter", {"animal": "a1"}, 1) as logger:
        print("still on the process's stdout")
        logger.info("only in the log")

    captured = capfd.readouterr()
    assert "still on the process's stdout" in captured.out
    assert "only in the log" not in captured.out + captured.err
    assert "only in the log" in (tmp_path / "q.out").read_text(encoding="utf-8")
    assert logging.getLogger().handlers == root_handlers
    assert not logger.handlers
