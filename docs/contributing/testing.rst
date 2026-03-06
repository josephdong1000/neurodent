Testing & Development
=====================

This guide covers testing, documentation building, and code style for NeuRodent.

Running Tests
-------------

Run the test suite with pytest:

.. code-block:: bash

   make test
   # or: uv run pytest

Run with coverage:

.. code-block:: bash

   uv run pytest --cov=neurodent

Run integration tests only:

.. code-block:: bash

   uv run pytest tests/integration/ -v -m integration


Pipeline Testing Strategy
-------------------------

The Snakemake pipeline is validated at three levels, all integrated into
CI/CD via GitHub Actions:

1. **Component-level integration tests (pytest)**
   These tests exercise the core pipeline building blocks — ``FileDiscoverer``,
   ``AnimalOrganizer``, WAR generation, plotters, and FDSAR — against a tiny
   synthetic NWB dataset generated on the fly.  They run as part of the
   normal ``uv run pytest`` invocation and are the fastest feedback loop.

2. **Snakemake DAG dry-run (pytest + subprocess)**
   ``TestSnakemakeDryRun`` in ``tests/integration/test_snakemake_flow.py``
   invokes ``snakemake --dryrun`` as a subprocess for both the ``example`` and
   ``mini_real`` datasets.  This validates the Snakefile, config files, sample
   JSONs, wildcard resolution, and rule definitions without processing any data.

3. **Snakemake real run (GitHub Actions, mini_real)**
   The ``test-build-docs.yml`` workflow runs a real full-pipeline
   ``snakemake`` execution on the committed mini_real dataset.  This
   validates end-to-end processing (data loading, discovery, analysis,
   filtering, figures) on every push/PR.  The ``example`` dataset uses a
   DAG dry-run since its synthetic NWB data is generated at test-fixture
   time, not committed to the repo.

.. code-block:: bash

   # Quick: validate DAG only (seconds)
   NEURODENT_DATASET=example uv run snakemake --snakefile workflow/Snakefile --dryrun --cores 1

   # Real run: execute full pipeline on committed mini data
   NEURODENT_DATASET=mini_real uv run snakemake --snakefile workflow/Snakefile --cores 1

   # Component integration tests (seconds)
   uv run pytest tests/integration/ -v -m integration


Example Dataset for Pipeline Testing
-------------------------------------

The repository includes infrastructure for testing the Snakemake pipeline
without processing full production-scale recordings.

**Programmatic synthetic data (CI-friendly)**

A ``create_synthetic_dataset()`` helper in ``tests/data/generate.py``
builds a tiny directory tree with 8-channel NWB files that can be read back
by SpikeInterface.  The ``example_dataset`` pytest fixture in
``tests/conftest.py`` wraps this for convenient use:

.. code-block:: python

   def test_my_pipeline_step(example_dataset):
       data_root = example_dataset["data_root"]
       samples_config = example_dataset["samples_config"]
       # ...

Integration tests in ``tests/integration/test_snakemake_flow.py`` demonstrate
discovery, filtering, and config-alias injection against this data.

**User-provided real data (local development)**

Place small real recordings under ``tests/data/raw/`` following the
directory layout described in ``tests/data/README.md``, then run:

.. code-block:: bash

   NEURODENT_DATASET=example snakemake --snakefile workflow/Snakefile --cores 1 --dryrun

See ``config/datasets/example.yaml`` and ``config/samples_example.json`` for
the corresponding configuration.

**Mini real dataset (committed bin/csv recordings)**

Small real recordings are committed under ``tests/data/raw/`` and exercised
by pytest integration tests in ``TestMiniRealDataset``.  The dataset uses
``{animal}/{index}`` placeholders with paired ``.bin`` / ``.csv`` files:

.. code-block:: bash

   # Run mini-real integration tests
   uv run pytest tests/integration/ -v -k TestMiniRealDataset

The full Snakemake run shown in the Pipeline Testing Strategy section above
also exercises this dataset end-to-end.  See ``config/datasets/mini_real.yaml``
and ``config/samples_mini_real.json`` for the corresponding configuration.


Building Documentation
----------------------

Build docs locally:

.. code-block:: bash

   make docs
   # or: cd docs && uv run sphinx-build -b html . _build/html

Build with live reload (auto-refresh on changes):

.. code-block:: bash

   make docs-live


Validation Scripts
------------------

NeuRodent includes validation scripts to verify correctness and performance on real data.
These scripts are located in the ``scripts/`` directory and should be run on the cluster.

Dask vs Serial Spike Detection Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that dask and serial multiprocess modes produce identical spike detection results:

.. code-block:: bash

   # Request 10 cores for dask processing
   srun -c 10 --pty bash
   cd /mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_Joseph_devtree

   # Run validation (default: 30 minutes of data)
   uv run python scripts/validate_dask_serial_spike_consistency.py \
       --recording-folder "/path/to/real/recording" \
       --verbose

This script validates exact spike indices (sample-level precision) and measures performance.
Run this after modifying ``src/neurodent/core/frequency_domain_spike_detection.py``.

**When to run:**

- After modifying spike detection algorithm
- Before releasing major versions
- When investigating spike count discrepancies
- Periodically (e.g., quarterly) as a sanity check

**Expected output:**

- ✅ All channels match perfectly at sample-level precision
- Performance comparison (dask should be 1.5-2x faster)
- Exit code 0 if identical, 1 if different

For full usage: ``python scripts/validate_dask_serial_spike_consistency.py --help``


Writing Validation Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating validation scripts for the neurodent project:

**Structure:**

- Location: ``/scripts/`` directory
- CLI interface via ``argparse`` with ``--help`` output
- Comprehensive module docstring (30-40 lines) with usage examples
- Exit code 0 for success, 1 for failure
- Use ``uv run python scripts/<name>.py`` in documentation

**Docstring requirements:**

- Brief description of what the script validates
- Important warnings (e.g., "Run on cluster only")
- Installation/setup requirements
- Usage examples with real data paths (or placeholders)
- Expected output description

**Code style:**

- Follow project style (ruff/PEP8)
- Helper functions with Google-style docstrings
- Structured output with visual separators
- Informative error messages

**Documentation:**

- Add section to ``/docs/contributing/testing.rst``
- Update CLAUDE.md if establishing new patterns
- Reference in README.md if user-facing

**Example structure:**

.. code-block:: python

   #!/usr/bin/env python3
   """
   Brief description.

   Detailed explanation.

   IMPORTANT:
   - Requirements/warnings

   Usage:
       uv run python scripts/script_name.py [options]

   Example:
       uv run python scripts/script_name.py --input /path/to/data
   """

   import argparse
   import sys

   def helper_function():
       """Docstring."""
       pass

   def main():
       parser = argparse.ArgumentParser(description="...")
       # ...
       sys.exit(0 if success else 1)

   if __name__ == "__main__":
       main()


Code Style
----------

We follow standard Python conventions:

- **Formatting**: Use `ruff <https://docs.astral.sh/ruff/>`_ for formatting and linting
- **Style**: Follow PEP 8 guidelines
- **Types**: Add type hints where appropriate
- **Docstrings**: Use NumPy style
