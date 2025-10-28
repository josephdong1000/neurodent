Contributing
============

We welcome contributions to Neurodent! This guide will help you get started with developing and contributing to the project.

Development Setup
-----------------

We recommend using `uv <https://docs.astral.sh/uv/>`_ for managing the development environment:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/josephdong1000/neurodent.git
   cd neurodent

   # Install with development dependencies
   uv sync --all-groups

Running Tests
-------------

To run the test suite:

.. code-block:: bash

   pytest

To run tests with coverage:

.. code-block:: bash

   pytest --cov=neurodent

Building Documentation
----------------------

To build the documentation locally:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

For live-reloading during development:

.. code-block:: bash

   sphinx-autobuild docs docs/_build

Code Style
----------

We follow standard Python code style conventions:

- Use `ruff` for code formatting and linting
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Write docstrings in NumPy style

Pull Request Process
---------------------

1. Fork the repository and create a new branch for your feature or bug fix
2. Make your changes and add tests if applicable
3. Ensure all tests pass and documentation builds successfully
4. Submit a pull request with a clear description of the changes

Questions?
----------

If you have questions about contributing, please open an issue on the `GitHub repository <https://github.com/josephdong1000/neurodent>`_.
