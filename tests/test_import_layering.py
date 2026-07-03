"""Import-layer + public-namespace contract (issue #110).

The package is organized as flat analysis-stage packages:

    constants  <  core (shared helpers)  <  loading / analysis / results  <  plotting

``plotting`` is the top visualization leaf: no lower-layer package may import it
at import time, and a bare ``import neurodent`` must stay lazy (it may not
eager-load any stage package). Each import check runs in a fresh interpreter so
it is immune to import pollution from other tests.
"""

import importlib.util
import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def _assert_no_leak(importing: str, forbidden_prefix: str):
    code = (
        f"import {importing}, sys; "
        f"leaked = sorted(m for m in sys.modules "
        f"if m == '{forbidden_prefix}' or m.startswith('{forbidden_prefix}.')); "
        f"assert not leaked, leaked"
    )
    r = _run(code)
    assert r.returncode == 0, (
        f"import-time layer violation: importing {importing} pulled in {forbidden_prefix}:\n"
        f"{r.stdout}\n{r.stderr}"
    )


def test_stage_packages_do_not_import_plotting():
    """No non-plotting stage package may import neurodent.plotting at import time."""
    for pkg in ("neurodent.core", "neurodent.loading", "neurodent.analysis", "neurodent.results"):
        _assert_no_leak(pkg, "neurodent.plotting")


def test_import_neurodent_is_lazy():
    """A bare `import neurodent` must not eager-load any stage package."""
    code = (
        "import neurodent, sys; "
        "eager = sorted(m for m in sys.modules if any(m == p or m.startswith(p + '.') "
        "for p in ('neurodent.loading','neurodent.analysis','neurodent.results','neurodent.plotting'))); "
        "assert not eager, eager"
    )
    r = _run(code)
    assert r.returncode == 0, f"`import neurodent` is not lazy:\n{r.stdout}\n{r.stderr}"


# Canonical module each headline class resolves to (also the lazy-export contract).
_HEADLINE_MODULES = {
    "AnimalOrganizer": "neurodent.loading.animal_organizer",
    "LongRecordingOrganizer": "neurodent.loading.long_recording_organizer",
    "LongRecordingAnalyzer": "neurodent.analysis.long_recording_analyzer",
    "WindowAnalysisResult": "neurodent.results.window_analysis_result",
    "FrequencyDomainSpikeAnalysisResult": "neurodent.results.frequency_domain_results",
    "ZeitgeberAnalysisResult": "neurodent.results.zeitgeber",
    "AnimalPlotter": "neurodent.plotting.animal",
    "ExperimentPlotter": "neurodent.plotting.experiment",
    "ZeitgeberPlotter": "neurodent.plotting.zeitgeber_plotter",
}


def test_lazy_headline_exports_resolve():
    """Every lazily-exported headline class resolves to its canonical module."""
    import neurodent

    for name, module in _HEADLINE_MODULES.items():
        obj = getattr(neurodent, name)
        assert obj.__module__ == module, f"{name}.__module__ == {obj.__module__!r}, expected {module!r}"


def test_top_level_namespace_is_curated():
    """The top-level namespace is a hand-curated allowlist, not a wildcard dump.

    Adding a name to ``_LAZY_EXPORTS`` (or otherwise leaking an internal symbol
    to the top level) fails here until this expected set is deliberately updated.
    """
    import neurodent

    expected = {
        "__version__", "__author__", "__email__", "__license__",
        "__title__", "__summary__", "__uri__",
        "set_channel_map",
        *_HEADLINE_MODULES,
    }
    assert set(neurodent.__all__) == expected, (
        "neurodent.__all__ drifted from the curated headline set:\n"
        f"  unexpected: {sorted(set(neurodent.__all__) - expected)}\n"
        f"  missing:    {sorted(expected - set(neurodent.__all__))}"
    )


def test_removed_shims_are_gone():
    """The pre-flatten module paths were hard-cut and must no longer resolve."""
    removed = [
        "neurodent.visualization",
        "neurodent.core.core",
        "neurodent.core.analysis",
        "neurodent.core.analyze_frag",
        "neurodent.core.frequency_domain_spike_detection",
        "neurodent.core.discovery",
        "neurodent.core.loading",
        "neurodent.core.results",
        "neurodent.core.zeitgeber",
    ]
    for name in removed:
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        assert spec is None, f"{name} still resolves; it should have been removed in the flatten"
