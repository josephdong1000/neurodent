Quick Installation
==================

Prerequisites
-------------

NeuRodent requires Python 3.10 or higher. If you don't have Python installed, visit `python.org <https://www.python.org/downloads/>`_ to download and install it.

To check if Python is installed on your system, open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:

.. code-block:: bash

   python --version

Install with pip
----------------

Install NeuRodent using pip:

.. code-block:: bash

   pip install neurodent

.. note::

   **New to Python?** Pip is Python's package installer that comes bundled with Python. Open your terminal and copy-paste the command above, then press Enter. This will automatically download and install NeuRodent and all its dependencies.

Install with conda
------------------

If you use Anaconda or Miniconda, you can install NeuRodent from conda-forge:

.. code-block:: bash

   conda install -c conda-forge neurodent

.. note::

   **New to conda?** Conda is a cross-platform package and environment manager. If you don't have it installed, visit `Miniconda <https://docs.anaconda.com/miniconda/>`_ to get started with a minimal installation.

To create a dedicated environment for NeuRodent:

.. code-block:: bash

   conda create -n neurodent-env python=3.10 neurodent -c conda-forge
   conda activate neurodent-env

Verifying Installation
----------------------

To verify that NeuRodent is installed correctly, open a Python interpreter by typing ``python`` in your terminal, then try importing ``neurodent``:

.. code-block:: python

   import neurodent
   print(neurodent.__version__)

If this runs without errors, you're ready to start using NeuRodent!

Install with ``uv`` (Recommended)
---------------------------------

We recommend using `uv <https://docs.astral.sh/uv/getting-started/>`_, a Python package and project manager that's faster than pip and handles virtual environments automatically.

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

This creates a new directory, sets up a virtual environment, and installs NeuRodent.

Next Steps
----------

- See :doc:`extras` for optional dependency groups
- See :doc:`pipeline` for Snakemake workflow setup
- Check out the :doc:`../quickstart/index` guide to learn how to use NeuRodent for EEG analysis
