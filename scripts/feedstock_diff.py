"""Diff pyproject runtime deps against the conda-forge feedstock recipe.

Compares ``[project.dependencies]`` in ``pyproject.toml`` against the
``requirements: run:`` list of the live ``conda-forge/neurodent-feedstock``
recipe and reports what the recipe must change: ADD (in pyproject, missing from
the recipe), REMOVE (in the recipe, not in pyproject - the case the recipe's
``pip check`` test does not catch), and PIN-CHANGE. Optional ``pipeline`` /
``readers`` extras are pip-only for conda users and are out of scope.

Names differ between PyPI and conda-forge, so an expected-name map is applied to
the pyproject side (matplotlib -> matplotlib-base; dask[distributed] -> dask-core
+ distributed). It is the project's declared translation, kept explicit here.

Exit 0 in sync, 3 on drift, 2 on fetch error. ``--format markdown`` emits an
issue body; ``--recipe PATH`` reads a local recipe instead of fetching.

    uv run python scripts/feedstock_diff.py
"""
import argparse
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 (tomli is in the dev extra)

try:
    # Verify TLS against certifi's CA bundle; Python on macOS often ships without
    # a usable system trust store (SSL: CERTIFICATE_VERIFY_FAILED otherwise).
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ModuleNotFoundError:
    _SSL_CONTEXT = None

RECIPE_URL = "https://raw.githubusercontent.com/conda-forge/neurodent-feedstock/main/recipe/meta.yaml"

# PyPI name -> conda-forge name. dask also contributes a separate `distributed`
# (from its [distributed] extra); see EXTRA_RUN below.
NAME_MAP = {"matplotlib": "matplotlib-base", "dask": "dask-core"}
EXTRA_RUN = {"dask": ("distributed", "")}  # PyPI dep -> extra conda run dep it pulls

# conda-only pseudo-deps that never appear in pyproject; ignore on the recipe side.
IGNORE = {"python", "pip"}

_REQ = re.compile(r"^([A-Za-z0-9._-]+)\s*(\[[^\]]*\])?\s*(.*)$")


def parse_pyproject(path):
    data = tomllib.loads(Path(path).read_text())
    proj = data["project"]
    return proj["version"], proj["dependencies"]


def expected_run(deps):
    """Map pyproject deps to the conda run: names/pins we expect on the recipe."""
    run, flagged = {}, []
    for dep in deps:
        spec, sep, marker = dep.partition(";")
        if sep:
            flagged.append(dep.strip())
        m = _REQ.match(spec.strip())
        name, pin = m.group(1).lower(), m.group(3).replace(" ", "")
        run[NAME_MAP.get(name, name)] = pin
        if name in EXTRA_RUN:
            extra_name, extra_pin = EXTRA_RUN[name]
            run[extra_name] = extra_pin
    return run, flagged


def parse_recipe_run(text):
    """Extract {name: pin} from the run: block of a Jinja-templated meta.yaml."""
    lines = text.splitlines()
    start, seen_req = None, False
    for idx, line in enumerate(lines):
        s = line.strip()
        if s == "requirements:":
            seen_req = True
        elif seen_req and s == "run:":
            start = idx + 1
            break
    if start is None:
        return {}
    run, item_indent = {}, None
    for line in lines[start:]:
        s = line.strip()
        if not s:
            continue
        indent = len(line) - len(line.lstrip())
        if not s.startswith("- "):
            break  # dedented out of the run: list
        if item_indent is None:
            item_indent = indent
        elif indent < item_indent:
            break
        parts = s[2:].split(None, 1)
        name = parts[0].lower()
        pin = parts[1].replace(" ", "") if len(parts) > 1 else ""
        if name not in IGNORE:
            run[name] = pin
    return run


def fetch_recipe(url):
    with urllib.request.urlopen(url, context=_SSL_CONTEXT) as resp:
        return resp.read().decode()


def diff(expected, live):
    adds = sorted(n for n in expected if n not in live)
    removes = sorted(n for n in live if n not in expected)
    pins = sorted(
        (n, live[n], expected[n]) for n in expected if n in live and expected[n] != live[n]
    )
    return adds, removes, pins


def render(version, adds, removes, pins, flagged, markdown):
    in_sync = not (adds or removes or pins or flagged)
    if in_sync:
        return f"conda-forge recipe matches pyproject.toml (v{version})."
    h = "## " if markdown else ""
    b = "- " if markdown else "  - "
    out = [f"conda-forge recipe out of sync with pyproject.toml (v{version})", ""]
    if adds:
        out.append(f"{h}ADD (in pyproject, missing from recipe)")
        out += [f"{b}{n}" for n in adds] + [""]
    if removes:
        out.append(f"{h}REMOVE (in recipe, not in pyproject)")
        out += [f"{b}{n}" for n in removes] + [""]
    if pins:
        out.append(f"{h}PIN CHANGE (recipe vs pyproject)")
        out += [f"{b}{n}: recipe `{lv or 'unpinned'}` vs pyproject `{ev or 'unpinned'}`" for n, lv, ev in pins] + [""]
    if flagged:
        out.append(f"{h}NEEDS REVIEW (marker'd dep may need a recipe selector)")
        out += [f"{b}{d}" for d in flagged] + [""]
    out.append("name-map applied: matplotlib->matplotlib-base, dask[distributed]->dask-core + distributed")
    out.append("Apply under requirements: run: on the autotick bot's PR; the drift check clears once in sync.")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pyproject", default="pyproject.toml")
    p.add_argument("--version", help="override the version shown in the report")
    p.add_argument("--recipe", help="read a local recipe file instead of fetching")
    p.add_argument("--format", choices=["text", "markdown"], default="text")
    args = p.parse_args()

    version, deps = parse_pyproject(args.pyproject)
    if args.version:
        version = args.version
    expected, flagged = expected_run(deps)

    try:
        text = Path(args.recipe).read_text() if args.recipe else fetch_recipe(RECIPE_URL)
    except (OSError, urllib.error.URLError) as e:
        print(f"error: could not read recipe: {e}", file=sys.stderr)
        return 2

    live = parse_recipe_run(text)
    adds, removes, pins = diff(expected, live)
    print(render(version, adds, removes, pins, flagged, args.format == "markdown"))
    return 3 if (adds or removes or pins or flagged) else 0


if __name__ == "__main__":
    sys.exit(main())
