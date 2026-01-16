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


Building Documentation
----------------------

Build docs locally:

.. code-block:: bash

   make docs
   # or: cd docs && uv run sphinx-build -b html . _build/html

Build with live reload (auto-refresh on changes):

.. code-block:: bash

   make docs-live


Code Style
----------

We follow standard Python conventions:

- **Formatting**: Use `ruff <https://docs.astral.sh/ruff/>`_ for formatting and linting
- **Style**: Follow PEP 8 guidelines
- **Types**: Add type hints where appropriate
- **Docstrings**: Use NumPy style
