Dataset Configuration
=====================

NeuRodent supports multiple datasets and file formats through a flexible configuration system. This guide shows how to select and configure different datasets for your analyses.

Quick Start
-----------

**Switch datasets using environment variable:**

.. code-block:: bash

   # Run with Sox5 binary dataset
   NEURODENT_DATASET=sox5_bin uv run snakemake --profile your-profile

   # Run with AP3B2 NWB dataset
   NEURODENT_DATASET=ap3b2_nwb uv run snakemake --profile your-profile

   # Run with AP3B2 RHD dataset
   NEURODENT_DATASET=ap3b2_rhd uv run snakemake --profile your-profile

How It Works
------------

**File structure:**

.. code-block:: text

   config/
   ├── config.yaml               # Main config (shared settings)
   ├── config.local.yaml         # Local overrides (gitignored)
   ├── datasets/                 # Dataset-specific configs
   │   ├── sox5_bin.yaml        # Sox5 project, binary format
   │   ├── ap3b2_nwb.yaml       # AP3B2 project, NWB format
   │   └── ap3b2_rhd.yaml       # AP3B2 project, RHD format
   └── samples*.json            # Sample metadata files

**How dataset selection works:**

1. Main config (``config.yaml``) contains shared settings for all datasets
2. Active dataset is specified via ``active_dataset`` parameter or ``NEURODENT_DATASET`` environment variable
3. Dataset config is loaded from ``config/datasets/{active_dataset}.yaml``
4. Dataset config is deep-merged into main config
5. Local overrides (``config.local.yaml``) are applied last

**Merge order** (later configs override earlier ones)::

   config.yaml → datasets/{active}.yaml → config.local.yaml

Available Datasets
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Dataset ID
     - Config File
     - Samples File
     - Format
   * - ``sox5_bin``
     - ``datasets/sox5_bin.yaml``
     - ``samples.json``
     - Binary
   * - ``ap3b2_nwb``
     - ``datasets/ap3b2_nwb.yaml``
     - ``samples_jess.json``
     - NWB
   * - ``ap3b2_rhd``
     - ``datasets/ap3b2_rhd.yaml``
     - ``samples_jess_rhd.json``
     - RHD (raw Intan)

Switching Datasets
------------------

There are three methods to select a dataset, each suited for different use cases.

Method 1: Environment Variable (Recommended for Cluster)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for:** Switching datasets per job, CI/CD pipelines, cluster batch jobs

.. code-block:: bash

   NEURODENT_DATASET=ap3b2_rhd uv run snakemake --profile your-profile

**Pros:** No file editing, easy to switch, perfect for scripts

**Cons:** Must specify for every command

Method 2: Edit config.yaml
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best for:** Setting team-wide default dataset

Edit ``config/config.yaml`` and change the ``active_dataset`` line:

.. code-block:: yaml

   # config/config.yaml
   active_dataset: "ap3b2_rhd"  # Change this line

**Pros:** Set once, git-tracked

**Cons:** Affects all users who pull your changes

Method 3: Local Override
^^^^^^^^^^^^^^^^^^^^^^^^

**Best for:** Personal dataset preference

Edit ``config/config.local.yaml``:

.. code-block:: yaml

   # config/config.local.yaml
   active_dataset: "ap3b2_rhd"

**Pros:** Local-only (gitignored), doesn't affect team

**Cons:** Can be forgotten

Priority Order
--------------

If multiple methods are used, this is the priority:

1. **Environment variable** (``NEURODENT_DATASET``) - **highest priority**
2. **Local config** (``config.local.yaml``)
3. **Main config** (``config.yaml``) - **lowest priority**

Verification
------------

Verify which dataset is active:

.. code-block:: bash

   uv run snakemake --dry-run 2>&1 | head -20

Expected output:

.. code-block:: text

   ✓ Using dataset: sox5_bin
     Config: config/datasets/sox5_bin.yaml
     Samples: config/samples.json
     Format: *.dat
     Mode: bin

Adding New Datasets
-------------------

Step 1: Create Samples JSON
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``config/samples_mydata.json``:

.. code-block:: json

   {
       "data_parent_folder": "/path/to/your/data",
       "GENOTYPE_ALIASES": {...},
       "data_folders_to_animal_ids": {...},
       "joint_sessions": {}
   }

Step 2: Create Dataset Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``config/datasets/mydata_nwb.yaml``:

.. code-block:: yaml

   # My Data - NWB Format

   samples:
     samples_file: "config/samples_mydata.json"

   analysis:
     war_generation:
       mode: "concat"
       file_pattern: "*.nwb"
       lro_kwargs:
         mode: "si"

You can override **any** config parameter using the same hierarchy as the main config.

Step 3: Use the New Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   NEURODENT_DATASET=mydata_nwb uv run snakemake --profile your-profile

Common Use Cases
----------------

Team Default Dataset
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Edit config.yaml: active_dataset: "sox5_bin"
   git add config/config.yaml
   git commit -m "Set default to Sox5"
   git push

Personal Dataset Preference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Edit config.local.yaml: active_dataset: "ap3b2_nwb"
   # This is gitignored - won't affect team

Cluster Batch Script
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=neurodent
   #SBATCH --time=24:00:00

   export NEURODENT_DATASET=sox5_bin
   uv run snakemake --profile slurm

Parallel Analysis
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Terminal 1
   NEURODENT_DATASET=sox5_bin uv run snakemake --profile slurm

   # Terminal 2
   NEURODENT_DATASET=ap3b2_nwb uv run snakemake --profile slurm

Troubleshooting
---------------

Dataset Not Found
^^^^^^^^^^^^^^^^^

**Error:**

.. code-block:: text

   FileNotFoundError: Dataset config file not found:
   config/datasets/mydata.yaml

**Solutions:**

- Check spelling (case-sensitive)
- Verify file exists: ``ls config/datasets/``
- Create the missing dataset config

Wrong Dataset Active
^^^^^^^^^^^^^^^^^^^^

**Check priority order:**

.. code-block:: bash

   echo $NEURODENT_DATASET        # Check env var
   grep active_dataset config/config.local.yaml
   grep active_dataset config/config.yaml

   # Clear env var if needed
   unset NEURODENT_DATASET

See also: :doc:`snakemake_setup` for pipeline setup and SLURM configuration.
