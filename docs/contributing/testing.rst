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


Example Dataset for Pipeline Testing
-------------------------------------

The repository includes infrastructure for testing the Snakemake pipeline
without processing full production-scale recordings.

**Programmatic synthetic data (CI-friendly)**

A ``create_synthetic_dataset()`` helper in ``tests/example_data/generate.py``
builds a tiny nest-mode directory tree with 8-channel float32 binary files and
companion ``_Meta.csv`` files.  The ``example_dataset`` pytest fixture in
``tests/conftest.py`` wraps this for convenient use:

.. code-block:: python

   def test_my_pipeline_step(example_dataset):
       data_root = example_dataset["data_root"]
       samples_config = example_dataset["samples_config"]
       # ...

Integration tests in ``tests/integration/test_snakemake_flow.py`` demonstrate
discovery, filtering, and config-alias injection against this data.

**User-provided real data (local development)**

Place small real recordings under ``tests/example_data/raw/`` following the
nest layout described in ``tests/example_data/README.md``, then run:

.. code-block:: bash

   NEURODENT_DATASET=example snakemake --cores 1 --dryrun

See ``config/datasets/example.yaml`` and ``config/samples_example.json`` for
the corresponding configuration.


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
