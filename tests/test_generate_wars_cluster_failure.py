"""A Dask cluster that fails to start must surface its own error.

generate_war_for_animal opened the cluster in a 'with' block and also closed it in a 'finally'.
When LocalCluster raised during construction, 'cluster' was never bound, so the 'finally' raised
UnboundLocalError on top of the real failure, and that is what the job reported. The 'with' block
already closes the cluster on every exit, so the 'finally' did nothing useful and is gone.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parent.parent / "workflow" / "scripts" / "generate_wars.py"


@pytest.fixture
def generate_wars(monkeypatch):
    spec = importlib.util.spec_from_file_location("generate_wars_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is guarded by __name__, so nothing runs
    module.snakemake = SimpleNamespace(threads=1)
    monkeypatch.setattr(module.core.utils, "set_temp_directory", lambda *_: None)
    monkeypatch.setattr(module, "apply_samples_config", lambda *_: None)
    return module


def test_cluster_startup_failure_is_reported_as_itself(generate_wars, monkeypatch):
    class StartupFailure(RuntimeError):
        pass

    def failing_cluster(*args, **kwargs):
        raise StartupFailure("no free port for the scheduler")

    monkeypatch.setattr(generate_wars, "LocalCluster", failing_cluster)
    config = {"temp_directory": "/unused", "cluster": {"war_generation": {"interface": None}}}

    with pytest.raises(StartupFailure, match="no free port"):
        generate_wars.generate_war_for_animal(
            samples_config={}, config=config, animal_folders=[("", "A1", "")],
            animal_id="A1", channel_subset=None,
            logger=SimpleNamespace(info=lambda *_: None, error=lambda *_: None),
        )
