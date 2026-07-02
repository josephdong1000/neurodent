"""Backward-compat shim; canonical home is neurodent.core.results.zeitgeber (#110)."""
from .results.zeitgeber import *  # noqa: F401,F403
from .results.zeitgeber import _load_war_for_zeitgeber, _compute_daynight  # noqa: F401
