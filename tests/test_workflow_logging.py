"""Every attempt of a job keeps its log, although Snakemake deletes the declared logs.

Snakemake deletes a rule's declared log files before each attempt of a job, when it submits the
attempt and again inside the job, so a log kept in them cannot outlive a retry. Each attempt is
therefore also appended to the attempts files beside the declared logs, which Snakemake does not
know of, after a header line naming the attempt. setup_snakemake_logging points file descriptors
1 and 2 at the attempts files, so child processes such as Dask workers write there too, and it
turns SIGTERM, which SLURM sends at a job's time limit, into SystemExit so that cleanup runs.

setup_snakemake_logging used to open the stdout log in write mode and leave the descriptors
alone, so Dask workers wrote to the SLURM job's own files. It changes the process's own
descriptors and signal handlers, so these tests run it in a subprocess and never touch pytest's
streams. Between two attempts they delete the declared logs, as Snakemake does. rule_log, its
counterpart for a run: block, leaves the descriptors alone and is tested in process.
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
from neurodent.workflow.utils.logging import attempt_header, attempts_path, log_paths

HEADER = re.compile(
    r"^=== (?P<rule>\w+) \((?P<wildcards>[^)]*)\), attempt (?P<attempt>\w+), "
    r"(?P<job>SLURM job \d+|not a SLURM job), node (?P<node>\S+), "
    r"started \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\S* ===$"
)
KEPT_IN = re.compile(
    r"^Every attempt of this job is kept in (?P<files>.+?)(?P<children>, with .+)?\.$"
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

    stdout_path, stderr_path, attempt, actions = sys.argv[1:5]
    actions = actions.split(",")
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
    if "child" in actions:
        subprocess.run([sys.executable, "-c", (
            "import sys; print('child out {0}'); print('child err {0}', file=sys.stderr)"
        ).format(attempt)], check=True)
    if "sigterm" in actions:
        import signal, time
        cache = os.path.join(os.path.dirname(stdout_path), "cache")
        os.makedirs(cache)
        try:
            os.kill(os.getpid(), signal.SIGTERM)  # as SLURM does at the time limit
            time.sleep(30)
            print("not reached")
        finally:
            os.kill(os.getpid(), signal.SIGTERM)  # a second one must not cut the cleanup short
            os.rmdir(cache)
            print("cleanup finished")
    if "fail" in actions:
        raise RuntimeError(f"attempt {attempt} failed")
    """
)
DRIVER_LINES = ("logged by", "printed by", "warned by", "descriptor 1 by", "child out",
                "child err", "RuntimeError", "Traceback", "cleanup finished", "Received SIGTERM")
ON_WINDOWS_CHILDREN_KEEP_THE_CONSOLE = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows child processes inherit the console handles, not C file descriptors",
)


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


def _run(tmp_path, stdout_path, stderr_path, attempt, actions="ok", env=None):
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(driver), str(stdout_path), str(stderr_path), str(attempt), actions,
         snakemake_logging.__file__],
        capture_output=True,
        text=True,
        env=env if env is not None else _env(),
        timeout=120,
    )


def _delete_declared_logs(*paths):
    """Delete the declared logs between two attempts, as Snakemake does before each attempt.

    Snakemake 7 calls Job.remove_existing_output, which removes every declared log, from
    ClusterExecutor._run when it submits an attempt and from Job.prepare inside the job.
    """
    for path in paths:
        os.remove(path)


def _read(path):
    return path.read_text(encoding="utf-8")


def _headers(text):
    return [HEADER.match(line) for line in text.splitlines() if line.startswith("=== ")]


def _kept_in(text):
    return [match for match in map(KEPT_IN.match, text.splitlines()) if match]


def _split_at_attempt(text, attempt):
    """Split an attempts file at the header of ``attempt``: what came before it, and the rest."""
    starts = [
        line.start()
        for line in re.finditer(r"^=== .*$", text, flags=re.MULTILINE)
        if HEADER.match(line.group())["attempt"] == str(attempt)
    ]
    assert len(starts) == 1, text
    return text[: starts[0]], text[starts[0]:]


def _assert_the_job_streams_only_say_where_the_output_went(result, kept_out, kept_err):
    """The job's own streams, which SLURM writes to the job's own files, get two lines."""
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert len(lines) == 2 and HEADER.match(lines[0]), result.stderr
    assert KEPT_IN.match(lines[1])["files"] == f"{kept_out} and {kept_err}"
    assert not [line for line in DRIVER_LINES if line in result.stderr]


@pytest.mark.parametrize(
    "with_child", [False, pytest.param(True, marks=ON_WINDOWS_CHILDREN_KEEP_THE_CONSOLE)]
)
def test_a_retry_keeps_the_failed_attempt_although_snakemake_deletes_the_logs(tmp_path, with_child):
    out, err = tmp_path / "logs" / "a1.out", tmp_path / "logs" / "a1.err"
    kept_out = tmp_path / "logs" / "a1.attempts.out"
    kept_err = tmp_path / "logs" / "a1.attempts.err"
    child = "child," if with_child else ""

    first = _run(
        tmp_path, out, err, 1, f"{child}fail",
        env=_env(SLURM_JOB_ID="4242", SLURMD_NODENAME="m-09-03"),
    )
    assert first.returncode != 0
    assert "RuntimeError: attempt 1 failed" in _read(err)
    _delete_declared_logs(out, err)
    second = _run(
        tmp_path, out, err, 2, f"{child}ok",
        env=_env(SLURM_JOB_ID="4243", SLURMD_NODENAME="c-11-25"),
    )
    assert second.returncode == 0, _read(kept_err)

    kept_out_text, kept_err_text = _read(kept_out), _read(kept_err)
    for text in (kept_out_text, kept_err_text):
        headers = _headers(text)
        assert [header["attempt"] for header in headers] == ["1", "2"]
        assert {header["rule"] for header in headers} == {"rule_x"}
        assert {header["wildcards"] for header in headers} == {"animal=a1"}
        assert [header["job"] for header in headers] == ["SLURM job 4242", "SLURM job 4243"]
        assert [header["node"] for header in headers] == ["m-09-03", "c-11-25"]

    # The first attempt's output, its traceback and its child's output outlive the retry
    first_out, second_out = _split_at_attempt(kept_out_text, 2)
    first_err, second_err = _split_at_attempt(kept_err_text, 2)
    for line in ("logged by attempt 1", "printed by attempt 1", "descriptor 1 by attempt 1"):
        assert line in first_out and line not in second_out
    for line in ("logged by attempt 2", "printed by attempt 2", "descriptor 1 by attempt 2"):
        assert line in second_out
    assert "warned by attempt 1" in first_err and "warned by attempt 2" in second_err
    assert "RuntimeError: attempt 1 failed" in first_err and "Traceback" not in second_err
    if with_child:
        assert "child out 1" in first_out and "child out 2" in second_out
        assert "child err 1" in first_err and "child err 2" in second_err

    # The declared logs hold the current attempt only, and name the attempts files
    out_text, err_text = _read(out), _read(err)
    for text in (out_text, err_text):
        assert [header["attempt"] for header in _headers(text)] == ["2"]
        assert "attempt 1" not in text
        assert [match["files"] for match in _kept_in(text)] == [f"{kept_out} and {kept_err}"]
    assert "logged by attempt 2" in out_text and "warned by attempt 2" in err_text
    # Output written to the descriptors, by the script or by a child, is kept only in the
    # attempts files
    assert "descriptor 1 by attempt 2" not in out_text
    assert "child out" not in out_text and "child err" not in err_text

    for result in (first, second):
        _assert_the_job_streams_only_say_where_the_output_went(result, kept_out, kept_err)


def test_each_stream_goes_to_its_own_log(tmp_path):
    out, err = tmp_path / "a1.out", tmp_path / "a1.err"
    kept_out, kept_err = tmp_path / "a1.attempts.out", tmp_path / "a1.attempts.err"
    result = _run(tmp_path, out, err, 1)

    assert result.returncode == 0, result.stderr
    out_text, err_text = _read(out), _read(err)
    kept_out_text, kept_err_text = _read(kept_out), _read(kept_err)
    for line in ("logged by attempt 1", "printed by attempt 1"):
        assert line in out_text and line in kept_out_text
        assert line not in err_text + kept_err_text
    assert "descriptor 1 by attempt 1" in kept_out_text
    assert "descriptor 1 by attempt 1" not in out_text + err_text + kept_err_text
    assert "warned by attempt 1" in err_text and "warned by attempt 1" in kept_err_text
    assert "warned by attempt 1" not in out_text + kept_out_text
    assert _headers(out_text)[0]["job"] == "not a SLURM job"
    _assert_the_job_streams_only_say_where_the_output_went(result, kept_out, kept_err)


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill on Windows terminates at once")
def test_sigterm_runs_the_cleanup_before_the_job_exits(tmp_path):
    out, err = tmp_path / "a1.out", tmp_path / "a1.err"
    result = _run(tmp_path, out, err, 1, "sigterm")

    assert result.returncode == 143, result.stderr
    assert not (tmp_path / "cache").exists()
    for path in (out, tmp_path / "a1.attempts.out"):
        text = _read(path)
        assert "cleanup finished" in text and "not reached" not in text
    for path in (err, tmp_path / "a1.attempts.err"):
        assert "Received SIGTERM" in _read(path)


def test_a_single_log_takes_both_streams(tmp_path):
    log, kept = tmp_path / "a1.log", tmp_path / "a1.attempts.log"
    result = _run(tmp_path, log, "-", 1)

    assert result.returncode == 0, result.stderr
    for path in (log, kept):
        text = _read(path)
        assert len(_headers(text)) == 1
        assert "logged by attempt 1" in text and "warned by attempt 1" in text
    assert [match["files"] for match in _kept_in(_read(log))] == [str(kept)]
    assert "descriptor 1 by attempt 1" in _read(kept)


@pytest.mark.parametrize(
    ("declared", "kept"),
    [
        ("logs/war_generation/iqsec2_80.out", "logs/war_generation/iqsec2_80.attempts.out"),
        ("logs/war_zeitgeber/war_zeitgeber.err", "logs/war_zeitgeber/war_zeitgeber.attempts.err"),
        ("logs/v1.2/A1 WT.log", "logs/v1.2/A1 WT.attempts.log"),
        ("logs/rule/a1", "logs/rule/a1.attempts"),
    ],
)
def test_the_attempts_file_sits_beside_the_declared_log(declared, kept):
    assert attempts_path(declared) == kept


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


def test_rule_log_keeps_every_attempt_although_snakemake_deletes_the_logs(tmp_path):
    log = _block_log(tmp_path)
    with pytest.raises(KeyError, match="genotype"):
        with rule_log(log, "war_quality_filter", {"animal": "a1"}, 1) as logger:
            logger.info("kept by attempt 1")
            raise KeyError("genotype")
    _delete_declared_logs(tmp_path / "q.out", tmp_path / "q.err")
    with rule_log(log, "war_quality_filter", {"animal": "a1"}, 2) as logger:
        logger.info("kept by attempt 2")

    kept_out_text = _read(tmp_path / "q.attempts.out")
    kept_err_text = _read(tmp_path / "q.attempts.err")
    assert [header["attempt"] for header in _headers(kept_out_text)] == ["1", "2"]
    assert kept_out_text.index("kept by attempt 1") < kept_out_text.index("kept by attempt 2")
    assert [header["attempt"] for header in _headers(kept_err_text)] == ["1", "2"]
    first_err, second_err = _split_at_attempt(kept_err_text, 2)
    assert "KeyError: 'genotype'" in first_err and "Traceback" not in second_err

    out_text, err_text = _read(tmp_path / "q.out"), _read(tmp_path / "q.err")
    for text in (out_text, err_text):
        assert [header["attempt"] for header in _headers(text)] == ["2"]
        assert "attempt 1" not in text
        # A run: block keeps the descriptors, so its child processes write to Snakemake's
        # own streams, and the line does not claim them
        assert [(match["files"], match["children"]) for match in _kept_in(text)] == [
            (f"{tmp_path / 'q.attempts.out'} and {tmp_path / 'q.attempts.err'}", None)
        ]
    assert "kept by attempt 2" in out_text


def test_rule_log_writes_the_traceback_of_a_failure_and_reraises(tmp_path):
    log = _block_log(tmp_path)

    with pytest.raises(KeyError, match="genotype"):
        with rule_log(log, "war_quality_filter", {"animal": "a1"}, 1):
            raise KeyError("genotype")

    for name in ("q.err", "q.attempts.err"):
        text = _read(tmp_path / name)
        assert "Traceback" in text and "KeyError: 'genotype'" in text


def test_rule_log_leaves_the_process_streams_and_root_logger_alone(tmp_path, capfd):
    log = _block_log(tmp_path)
    root_handlers = list(logging.getLogger().handlers)

    with rule_log(log, "war_quality_filter", {"animal": "a1"}, 1) as logger:
        print("still on the process's stdout")
        logger.info("only in the log")

    captured = capfd.readouterr()
    assert "still on the process's stdout" in captured.out
    assert "only in the log" not in captured.out + captured.err
    for name in ("q.out", "q.attempts.out"):
        assert "only in the log" in _read(tmp_path / name)
    assert logging.getLogger().handlers == root_handlers
    assert not logger.handlers
