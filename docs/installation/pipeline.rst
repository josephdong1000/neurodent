Pipeline & Cluster Setup
========================

NeuRodent includes a Snakemake workflow for automated analysis pipelines. This page covers installation and cluster configuration.

Quick Start (pip install workflow)
----------------------------------

No git clone required.  The full pipeline is bundled with the Python package and
can be deployed with a single command:

.. code-block:: bash

   # 1. Install NeuRodent with pipeline support
   pip install "neurodent[pipeline]"

   # 2. Create a new analysis directory and deploy the pipeline files
   mkdir my_analysis && cd my_analysis
   neurodent init-pipeline

   # 3. Edit the configuration for your data
   # (see Dataset Configuration guide)

   # 4. Run the pipeline
   snakemake --cores 4

The ``neurodent init-pipeline`` command copies ``Snakefile``, ``workflow/``, and
``config/`` into the current directory (or a specified target directory).

.. code-block:: bash

   # Deploy to a specific directory
   neurodent init-pipeline /path/to/my/project

   # Overwrite existing files
   neurodent init-pipeline --overwrite

.. tip::

   Run ``neurodent init-pipeline --help`` for a full list of options.

Installing Pipeline Dependencies
--------------------------------

Install the optional pipeline dependencies:

**Using pip:**

.. code-block:: bash

   pip install "neurodent[pipeline]"

**Using uv:**

.. code-block:: bash

   uv add neurodent[pipeline]

.. note::

   The ``pipeline`` extra includes Snakemake and related dependencies needed for running the automated analysis workflow. If you only need the core NeuRodent library for Python-based analysis, the basic installation is sufficient.

SLURM Cluster Configuration
---------------------------

If you're running the Snakemake workflow on a SLURM cluster, the setup depends on your Snakemake version.

Snakemake 7.x (Python 3.10)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use a `Snakemake SLURM profile <https://github.com/Snakemake-Profiles/slurm>`_ generated with cookiecutter.

**Recommended log path setting:** To place SLURM job logs alongside Snakemake logs (making debugging easier), update your profile's ``CookieCutter.py``:

.. code-block:: python

   # ~/.config/snakemake/your-profile/CookieCutter.py
   def get_cluster_logpath() -> str:
       return "logs/%r/slurm_%j"  # puts slurm_{jobid}.{out,err} in logs/<rule>/

Run the workflow with:

.. code-block:: bash

   snakemake --profile your-profile

Snakemake 8+ (Python 3.11+)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Snakemake 8 uses a native `SLURM executor plugin <https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html>`_ instead of cookiecutter profiles.

Install the plugin:

.. code-block:: bash

   pip install snakemake-executor-plugin-slurm

Run the workflow with the ``--executor`` flag:

.. code-block:: bash

   snakemake --executor slurm --default-resources --jobs 30

The native plugin automatically:

- Deletes SLURM log files for successful jobs (reduces clutter)
- Preserves logs for failed jobs for 10 days
- Uses ``--slurm-logdir`` to customize log location

To customize log directory, add to your profile:

.. code-block:: yaml

   # ~/.config/snakemake/your-profile/config.yaml
   executor: slurm
   slurm-logdir: "logs/slurm"

See the `plugin documentation <https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html>`_ for full configuration options.

Local Configuration Overrides
-----------------------------

You can override any setting from ``config/config.yaml`` using a local configuration file. This is useful for adjusting analysis parameters or file paths for your specific environment without modifying the main configuration file (which is tracked by git).

To use local overrides:

1.  Create a file named ``config/config.local.yaml``.
2.  Add the specific configuration keys you wish to override. You do *not* need to copy the entire configuration file; Snakemake performs a "deep merge", so only the keys you specify will be updated.

**Example:**

If you want to change the analysis sampling rate but keep all other settings:

.. code-block:: yaml

   # config/config.local.yaml
   analysis:
     sampling_rate: 2000

The ``config/config.local.yaml`` file is included in ``.gitignore`` and will not be pushed to the repository.

Next Steps
----------

For detailed pipeline configuration guides:

- **Snakemake Setup**: See :doc:`../quickstart/snakemake_setup` for running the pipeline, SLURM configuration, and workflow commands
- **Dataset Configuration**: See :doc:`../quickstart/dataset_configuration` for switching between datasets and file formats
