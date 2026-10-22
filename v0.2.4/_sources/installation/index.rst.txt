Installation
============

Prerequisites
-------------

NeuRodent requires Python 3.10 or higher. If you don't have Python installed, visit `python.org <https://www.python.org/downloads/>`_ to download and install it.

To check if Python is installed on your system, open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:

.. code-block:: bash

   python --version

Quick Installation
------------------

Install NeuRodent using pip:

.. code-block:: bash

   pip install neurodent

.. note::

   **New to Python?** Pip is Python's package installer that comes bundled with Python. Open your terminal and copy-paste the command above, then press Enter. This will automatically download and install NeuRodent and all its dependencies.

Verifying Installation
----------------------

To verify that NeuRodent is installed correctly, open a Python interpreter by typing ``python`` in your terminal, then try importing ``neurodent``:

.. code-block:: python

   import neurodent
   print(neurodent.__version__)

If this runs without errors, you're ready to start using NeuRodent!

Installing with ``uv``
-----------------------------------

Though not required, we recommend using `uv <https://docs.astral.sh/uv/getting-started/>`_, a Python package and project manager that's faster than pip and handles virtual environments automatically.

.. tip::

   **Why use uv?** uv is an all-in-one tool that simplifies Python development by handling:
   
   - **Virtual environment creation and management** - Automatically creates isolated environments for each project
   - **Package management** - Installs and manages dependencies faster than pip
   - **Python version management** - Can install and switch between different Python versions
   - **Dependency resolution** - Ensures all packages work together without conflicts
   
   Learn more about uv's features `here <https://docs.astral.sh/uv/getting-started/features/>`_.

First, install uv by following the instructions at `docs.astral.sh/uv <https://docs.astral.sh/uv/getting-started/installation/>`_.

Then create a new project with NeuRodent:

.. code-block:: bash

   uv init yourprojectname
   cd yourprojectname
   uv add neurodent

This creates a new directory, sets up a virtual environment, and installs NeuRodent—all in one go.

Installing with Pipeline Support
---------------------------------

NeuRodent includes a Snakemake workflow for automated analysis pipelines. If you want to use this workflow, you'll need to install the optional pipeline dependencies:

**Using pip:**

.. code-block:: bash

   pip install neurodent[pipeline]

**Using uv:**

.. code-block:: bash

   uv add neurodent[pipeline]

.. note::

   The ``pipeline`` extra includes Snakemake and related dependencies needed for running the automated analysis workflow. If you only need the core NeuRodent library for Python-based analysis, the basic installation is sufficient.

Development Installation
------------------------

If you want to contribute to NeuRodent or modify the source code, install it in editable mode:

**Using uv (recommended):**

.. code-block:: bash

   git clone https://github.com/josephdong1000/neurodent
   cd neurodent
   uv sync --extra pipeline

**Using pip:**

.. code-block:: bash

   git clone https://github.com/josephdong1000/neurodent
   cd neurodent
   pip install -e .[pipeline]

Next Steps
----------

Check out the :doc:`../quickstart/index` guide to learn how to use NeuRodent for EEG analysis.
