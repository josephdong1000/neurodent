Development Setup
=================

This guide covers the complete development environment setup for NeuRodent.

Prerequisites
-------------

- **Python 3.10+**
- **uv** - Fast Python package manager (`install uv <https://docs.astral.sh/uv/getting-started/installation/>`_)
- **make** (optional, for convenience commands)

Quick Start
-----------

.. code-block:: bash

   git clone https://github.com/josephdong1000/neurodent.git
   cd neurodent
   make setup

This runs ``uv sync --all-extras`` (installs all dependencies) and ``uv run pre-commit install`` (sets up git hooks).

Manual Setup
------------

If you don't have ``make``:

.. code-block:: bash

   git clone https://github.com/josephdong1000/neurodent.git
   cd neurodent
   uv sync --all-extras
   uv run pre-commit install

If you don't have ``uv``:

.. code-block:: bash

   git clone https://github.com/josephdong1000/neurodent.git
   cd neurodent
   pip install -e ".[dev]"
   pre-commit install

Makefile Commands
-----------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Command
     - Description
   * - ``make setup``
     - Install dependencies + pre-commit hooks
   * - ``make test``
     - Run test suite with pytest
   * - ``make docs``
     - Build documentation to ``docs/_build/html``
   * - ``make docs-live``
     - Build docs with live reload (auto-refresh on changes)
   * - ``make clean``
     - Remove build artifacts

Pre-commit Hooks
----------------

NeuRodent uses `pre-commit <https://pre-commit.com/>`_ hooks to maintain code quality.
These run automatically before each ``git commit``.

**nbstripout**
   Strips output cells from Jupyter notebooks before commits.
   This keeps git history clean and avoids merge conflicts from cell outputs and metadata.

   .. note::
   
      Notebooks in ``notebooks/`` are excluded from stripping (for development use).
      Only ``docs/`` notebooks are stripped.

To run hooks manually on all files:

.. code-block:: bash

   uv run pre-commit run --all-files

Notebook Documentation
----------------------

Documentation notebooks in ``docs/`` are executed during Sphinx builds to generate figures.
Outputs are stripped from git but regenerated at build time.

If a notebook cell fails during build, execution stops at that cell and preceding outputs are shown.
To skip execution of specific cells, add the ``skip-execution`` tag in JupyterLab.

