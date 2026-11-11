"""
Rule Declaration Scans
======================

File scans of ``workflow/Snakefile``, ``workflow/rules/*.smk`` and the scripts and notebooks
they run, for conventions every rule must follow and that a dry run cannot check.

- A rule passes ``analysis_config(config)`` as its ``config`` param, never the whole config.
  Snakemake reruns a job when its params change, so passing the whole config made every edit
  to job sizes rerun every job.
- A script reads the execution-only keys (``cluster``, ``temp_directory``, ``active_dataset``)
  from ``snakemake.config`` only, because ``snakemake.params.config`` no longer holds them.

Running
-------
::

    uv run pytest .tests/unit/test_rule_declarations.py -v

"""

import ast
import json
import re
from pathlib import Path

import pytest

from neurodent.workflow.utils.config import EXECUTION_ONLY_KEYS

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / "workflow"
RULE_FILES = [WORKFLOW / "Snakefile", *sorted((WORKFLOW / "rules").glob("*.smk"))]
JOB_FILES = sorted((WORKFLOW / "scripts").glob("*.py")) + sorted(
    (WORKFLOW / "notebooks").glob("*.ipynb")
)

DIRECTIVES = {
    "input", "output", "params", "threads", "resources", "log", "benchmark", "message",
    "shell", "run", "script", "notebook", "wrapper", "conda", "container", "retries",
    "priority", "group", "shadow", "localrule", "cache", "default_target", "handover",
    "envmodules", "wildcard_constraints", "version", "name", "template_engine",
}
RULE_HEADER = re.compile(r"^(?P<indent>[ \t]*)(?:rule|checkpoint)\s+(?P<name>\w+)\s*:")
DIRECTIVE_LINE = re.compile(r"^(?P<indent>[ \t]+)(?P<name>[a-z_]+)\s*:(?P<rest>.*)$")


def _indent(line):
    return len(line) - len(line.lstrip())


def _directives(body):
    """Map each directive of a rule body to its text, skipping the rule's docstring."""
    found = {}
    current = None
    depth = None
    in_docstring = False
    for line in body:
        stripped = line.strip()
        if in_docstring or stripped.startswith(('"""', "'''")):
            quotes = stripped.count('"""') + stripped.count("'''")
            if in_docstring:
                in_docstring = quotes % 2 == 0
            else:
                in_docstring = quotes % 2 == 1
            continue
        if not stripped or stripped.startswith("#"):
            continue
        match = DIRECTIVE_LINE.match(line)
        if match and match["name"] in DIRECTIVES and (depth is None or _indent(line) <= depth):
            depth = _indent(line)
            current = match["name"]
            found[current] = match["rest"].strip()
        elif current is not None:
            found[current] = f"{found[current]}\n{stripped}".strip()
    return found


def parse_rules(path):
    """Map each rule and checkpoint in a Snakemake file to its directives and their text."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rules = {}
    i = 0
    while i < len(lines):
        match = RULE_HEADER.match(lines[i])
        i += 1
        if not match:
            continue
        base = len(match["indent"])
        body = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and not line.lstrip().startswith("#") and _indent(line) <= base:
                break
            body.append(line)
            i += 1
        rules[match["name"]] = _directives(body)
    return rules


ALL_RULES = {
    f"{path.name}:{name}": directives
    for path in RULE_FILES
    for name, directives in parse_rules(path).items()
}


def test_the_scan_finds_the_pipeline_rules():
    names = {key.split(":", 1)[1] for key in ALL_RULES}
    assert {"war_generation", "war_quality_filter", "war_zeitgeber", "notebook", "all"} <= names
    assert ALL_RULES["war_generation.smk:war_generation"]["script"] == '"../scripts/generate_wars.py"'


def _config_params(directives):
    """The value of every ``config=`` param a rule passes."""
    params = directives.get("params", "")
    return [
        value.strip()
        for value in re.findall(r"(?<![\w.])config\s*=\s*([^\n]+?),?$", params, flags=re.MULTILINE)
    ]


def test_the_params_scan_sees_a_whole_config(tmp_path):
    smk = tmp_path / "bad.smk"
    smk.write_text(
        'rule bad:\n    """Doc: params: config=config"""\n    params:\n'
        "        samples_config=samples_config,\n        config=config,\n"
        '    script:\n        "x.py"\n',
        encoding="utf-8",
    )
    assert _config_params(parse_rules(smk)["bad"]) == ["config"]


@pytest.mark.parametrize("rule", sorted(ALL_RULES))
def test_no_rule_passes_the_whole_config(rule):
    for value in _config_params(ALL_RULES[rule]):
        assert value == "analysis_config(config)", (
            f"{rule} passes config={value} as a param. Pass config=analysis_config(config), "
            "so that editing a job size does not rerun the job."
        )


def _is_snakemake_config(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "config"
        and isinstance(node.value, ast.Name)
        and node.value.id == "snakemake"
    )


def _execution_key_reads(tree):
    """Yield (line, key, the expression read from) for every read of an execution-only key."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in EXECUTION_ONLY_KEYS
        ):
            yield node.lineno, node.slice.value, node.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("get", "pop", "setdefault")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in EXECUTION_ONLY_KEYS
        ):
            yield node.lineno, node.args[0].value, node.func.value


def _job_sources(path):
    """Python source of a workflow script, or of each code cell of a workflow notebook."""
    if path.suffix == ".py":
        return [path.read_text(encoding="utf-8")]
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    sources = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        # IPython magics and shell escapes are not Python
        lines = [line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))]
        sources.append("\n".join(lines))
    return sources


def test_the_job_scan_sees_the_scripts():
    assert WORKFLOW / "scripts" / "generate_wars.py" in JOB_FILES
    reads = [
        read
        for source in _job_sources(WORKFLOW / "scripts" / "generate_wars.py")
        for read in _execution_key_reads(ast.parse(source))
    ]
    assert {key for _, key, _ in reads} == {"cluster", "temp_directory"}


def _reads_outside_snakemake_config(source):
    """(line, key, expression) for each execution-only key read from anything but snakemake.config."""
    return [
        (line, key, ast.unparse(read_from))
        for line, key, read_from in _execution_key_reads(ast.parse(source))
        if not _is_snakemake_config(read_from)
    ]


def test_the_job_scan_sees_a_read_from_params():
    source = (
        "config = snakemake.params.config\n"
        "tmp = config['temp_directory']\n"
        "iface = snakemake.params.config.get('cluster')\n"
        "ok = snakemake.config['cluster']\n"
    )
    assert _reads_outside_snakemake_config(source) == [
        (2, "temp_directory", "config"),
        (3, "cluster", "snakemake.params.config"),
    ]


@pytest.mark.parametrize("path", JOB_FILES, ids=lambda path: path.name)
def test_jobs_read_execution_only_keys_from_snakemake_config(path):
    for source in _job_sources(path):
        bad = _reads_outside_snakemake_config(source)
        assert not bad, (
            f"{path.name} reads execution-only keys from something other than "
            f"snakemake.config: {bad}. Rules pass analysis_config(config) as params, which "
            "leaves them out."
        )
