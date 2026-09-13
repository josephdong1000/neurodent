Installation
============

Prerequisites
-------------

Neurodent requires Python 3.10 or higher. If you don't have Python installed, visit `python.org <https://www.python.org/downloads/>`_ to download and install it.

To check if Python is installed on your system, open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:

.. code-block:: bash

   python --version

Quick Installation
------------------

Install Neurodent using pip:

.. code-block:: bash

   pip install neurodent

.. note::

   **New to Python?** Pip is Python's package installer that comes bundled with Python. Open your terminal and copy-paste the command above, then press Enter. This will automatically download and install Neurodent and all its dependencies.

Verifying Installation
----------------------

To verify that Neurodent is installed correctly, open a Python interpreter by typing ``python`` in your terminal, then try importing Neurodent:

.. code-block:: python

   import neurodent
   print(neurodent.__version__)

If this runs without errors, you're ready to start using Neurodent!

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

Then create a new project with Neurodent:

.. code-block:: bash

   uv init yourprojectname
   cd yourprojectname
   uv add neurodent

This creates a new directory, sets up a virtual environment, and installs Neurodent—all in one go.

Next Steps
----------

Check out the :doc:`../quickstart/index` guide to learn how to use Neurodent for EEG analysis.
