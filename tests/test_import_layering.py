"""Import-layer contract (issue #110).

``neurodent.core`` (primitives + loading + result containers) must never import
``neurodent.visualization`` (plotting) at import time. Each check runs in a fresh
interpreter so it is immune to import pollution from other tests.
"""

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_core_does_not_import_visualization():
    """Importing neurodent.core must not pull in neurodent.visualization."""
    code = (
        "import neurodent.core, sys; "
        "leaked = sorted(m for m in sys.modules "
        "if m == 'neurodent.visualization' or m.startswith('neurodent.visualization.')); "
        "assert not leaked, leaked"
    )
    r = _run(code)
    assert r.returncode == 0, (
        "neurodent.core import-time layer violation (loaded visualization):\n"
        f"{r.stdout}\n{r.stderr}"
    )


def test_import_neurodent_is_lazy():
    """A bare `import neurodent` must not eager-load core.loading or visualization."""
    code = (
        "import neurodent, sys; "
        "eager = sorted(m for m in sys.modules "
        "if m.startswith('neurodent.core.loading') or m.startswith('neurodent.visualization')); "
        "assert not eager, eager"
    )
    r = _run(code)
    assert r.returncode == 0, f"`import neurodent` is not lazy:\n{r.stdout}\n{r.stderr}"


def test_lazy_headline_exports_resolve():
    """The lazily-exported headline classes resolve to their canonical modules."""
    import neurodent

    assert neurodent.AnimalOrganizer.__module__ == "neurodent.core.loading.animal_organizer"
    assert neurodent.WindowAnalysisResult.__module__ == "neurodent.core.results.window_analysis_result"
    assert neurodent.ZeitgeberAnalysisResult.__module__ == "neurodent.core.results.zeitgeber"
