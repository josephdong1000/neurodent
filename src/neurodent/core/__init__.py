"""Shared low-level helpers used across every analysis stage.

``neurodent.core`` holds only cross-cutting utilities (:mod:`neurodent.core.utils`)
and animal-metadata resolution (:mod:`neurodent.core.metadata`). It is the shared
floor of the package: every stage may import it, and it imports nothing above
``constants``.

The stage classes live in their own top-level packages:

- :mod:`neurodent.loading` — ``LongRecordingOrganizer``, ``AnimalOrganizer``
- :mod:`neurodent.analysis` — ``LongRecordingAnalyzer``, ``AnalysisPipeline``, spike detection
- :mod:`neurodent.results` — ``WindowAnalysisResult``, ``FrequencyDomainSpikeAnalysisResult``, ``ZeitgeberAnalysisResult``
- :mod:`neurodent.plotting` — ``AnimalPlotter``, ``ExperimentPlotter``, ``ZeitgeberPlotter``

The headline classes are also lazily importable from the top level, e.g.
``from neurodent import AnimalOrganizer``.
"""

import os
import tempfile

# Ensure a usable temporary directory is available for downstream modules
if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = tempfile.gettempdir()

from . import utils  # noqa: E402,F401
from . import metadata  # noqa: E402,F401

__all__ = ["utils", "metadata"]
