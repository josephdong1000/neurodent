Snakemake Pipeline Setup
=========================

This guide covers setting up and configuring the Snakemake workflow for automated analysis pipelines.

Deploying the Pipeline
-----------------------

The pipeline is bundled with the NeuRodent Python package, so no git clone is
required.  Deploy the pipeline files to your working directory with:

.. code-block:: bash

   neurodent init-pipeline

This copies ``Snakefile``, ``workflow/``, and ``config/`` into the current
directory.  To deploy to a different location:

.. code-block:: bash

   neurodent init-pipeline /path/to/my/analysis
   neurodent init-pipeline --overwrite   # replace existing files

Installing Pipeline Dependencies
---------------------------------

Install the optional pipeline dependencies:

**Using uv:**

.. code-block:: bash

   uv add neurodent[pipeline]

**Using pip:**

.. code-block:: bash

   pip install "neurodent[pipeline]"

.. note::

   The ``pipeline`` extra includes Snakemake and related dependencies needed for running the automated analysis workflow. If you only need the core NeuRodent library for Python-based analysis, the basic installation is sufficient.

SLURM Cluster Configuration
----------------------------

If you're running the Snakemake workflow on a SLURM cluster, the setup depends on your Snakemake version.

Snakemake 7.x (Python 3.10)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use a `Snakemake SLURM profile <https://github.com/Snakemake-Profiles/slurm>`_ generated with cookiecutter.

**Recommended log path setting:** To place SLURM job logs alongside Snakemake logs (making debugging easier), update your profile's ``CookieCutter.py``:

.. code-block:: python

   # ~/.config/snakemake/your-profile/CookieCutter.py
   def get_cluster_logpath() -> str:
       return "logs/%r/slurm_%j"  # puts slurm_{jobid}.{out,err} in logs/<rule>/

Run the workflow with:

.. code-block:: bash

   uv run snakemake --profile your-profile

Snakemake 8+ (Python 3.11+)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Snakemake 8 uses a native `SLURM executor plugin <https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html>`_ instead of cookiecutter profiles.

Install the plugin:

.. code-block:: bash

   pip install snakemake-executor-plugin-slurm

Run the workflow with the ``--executor`` flag:

.. code-block:: bash

   uv run snakemake --executor slurm --default-resources --jobs 30

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
------------------------------

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

Testing with a Subset of Animals
---------------------------------

When testing the pipeline, you may want to run only a small number of animals instead of the full dataset. Use the ``truncate_animals`` setting under ``samples`` to limit processing to the first *N* animals in the samples file:

.. code-block:: yaml

   # config/config.local.yaml
   samples:
     truncate_animals: 2   # only process the first 2 animals

Set ``truncate_animals`` to ``null`` (the default) to process all animals.

.. tip::

   Combine this with a fast dataset such as ``mini_real`` for quick smoke-testing:

   .. code-block:: bash

      NEURODENT_DATASET=mini_real uv run snakemake --cores 1

Running the Pipeline
--------------------

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   # Dry run to see what would be executed
   uv run snakemake --dry-run

   # Run pipeline locally (for testing)
   uv run snakemake --cores 4

   # Run on SLURM cluster
   uv run snakemake --profile your-profile

Useful Commands
^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate workflow visualization
   uv run snakemake --rulegraph | dot -Tpng > workflow.png

   # Clean results (be careful!)
   uv run snakemake --delete-all-output

   # Unlock workflow (if interrupted)
   uv run snakemake --unlock

   # Force re-run specific rule
   uv run snakemake --forcerun rule_name

Alternative: Deploying with ``snakedeploy``
--------------------------------------------

The `Snakemake documentation <https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html>`_
recommends distributing workflows via the
`Snakemake module system <https://snakemake.readthedocs.io/en/stable/snakefiles/modularization.html>`_
and `snakedeploy <https://snakedeploy.readthedocs.io>`_.  This approach references
a specific tagged release on GitHub rather than copying files, which eliminates
duplication and makes version pinning explicit.

Install ``snakedeploy``:

.. code-block:: bash

   pip install snakedeploy

Deploy a tagged release of the NeuRodent pipeline:

.. code-block:: bash

   snakedeploy deploy-workflow \
       https://github.com/josephdong1000/neurodent \
       /path/to/my/analysis \
       --tag v0.2.4

This generates a thin ``workflow/Snakefile`` that references the remote
workflow and copies the ``config/`` directory as a local template.  You can then
customise the configuration and run the pipeline as usual.

.. note::

   Well-established Snakemake workflows in the
   `Snakemake Workflow Catalog <https://snakemake.github.io/snakemake-workflow-catalog/>`_
   (e.g. ``snakemake-workflows/rna-seq-star-deseq2``) use this approach
   exclusively.  For fully reproducible analyses, ``snakedeploy`` is preferred
   over ``neurodent init-pipeline`` because it pins the exact workflow version.

See also: :doc:`dataset_configuration` for selecting and configuring different datasets.
