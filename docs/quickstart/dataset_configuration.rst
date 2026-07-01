Dataset Configuration
=====================

NeuRodent supports multiple datasets and file formats through a flexible configuration system. This guide shows how to select and configure different datasets for your analyses.

Quick Start
-----------

**Switch datasets using environment variable:**

.. code-block:: bash

   # Run with Sox5 binary dataset
   NEURODENT_DATASET=sox5_bin uv run snakemake --profile your-profile

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
   │   └── ap3b2_rhd.yaml       # AP3B2 project, RHD format
   └── samples*.json            # Sample metadata files

**How dataset selection works:**

1. Main config (``config.yaml``) contains shared settings for all datasets
2. Local overrides (``config.local.yaml``) are merged if the file exists (can set ``active_dataset``)
3. Active dataset is resolved via ``NEURODENT_DATASET`` env var or the ``active_dataset`` key
4. Dataset config is loaded from ``config/datasets/{active_dataset}.yaml``
5. Dataset config is deep-merged on top of the combined config

**Merge order** (later configs override earlier ones)::

   config.yaml → config.local.yaml → datasets/{active}.yaml

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

Expected output (shows the dataset-specific overrides applied):

.. code-block:: text

   [ok] Using dataset: sox5_bin
     Config file: config/datasets/sox5_bin.yaml

     Dataset configuration overrides:
       samples:
         samples_file: "config/samples.json"
       analysis:
         war_generation:
           ...

Adding New Datasets
-------------------

Step 1: Create Samples JSON
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The samples JSON file defines the animals in your experiment and their metadata.
Create ``config/samples_mydata.json`` using the unified ``animals`` list format:

.. code-block:: json

   {
       "data_root": "/path/to/your/data",
       "animals": [
           {"id": "M1", "genotype": "WT", "sex": "M"},
           {"id": "F3", "genotype": "KO", "sex": "F"}
       ]
   }

.. _animals-parameter-reference:

Animals Parameter Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each entry in the ``animals`` list is a dictionary. The following table
describes all available parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 55

   * - Parameter
     - Type
     - Required
     - Description
   * - ``id``
     - string
     - Yes
     - Unique identifier for the animal (e.g. ``"M1"``, ``"AP3B2homo-240-M"``).
       Used as the primary key throughout the pipeline.
   * - ``genotype``
     - string
     - Yes
     - Genotype label (e.g. ``"WT"``, ``"KO"``, ``"Het"``); surfaces as the
       ``genotype`` column for downstream grouping. The legacy key ``gene`` is
       still accepted as an alias. Optionally normalized to a short label via
       ``GENOTYPE_MAP`` (exact match; when that map is set it is authoritative —
       an uncovered value raises at load).
   * - ``sex``
     - string
     - Yes
     - Sex of the animal (e.g. ``"M"``, ``"F"``, ``"Male"``, ``"Female"``).
   * - ``manual_datetime``
     - string
     - No
     - Recording **start** datetime. Accepts any standard datetime format
       including ISO 8601 (e.g. ``"2025-05-10T10:00:00"``) and
       ``"YYYY-MM-DD HH:MM:SS"``.
       Use this when automatic datetime parsing from filenames is unreliable
       or when files lack embedded timestamps. See :ref:`manual-datetimes`.
   * - ``datetimes_are_start``
     - bool
     - No
     - Whether ``manual_datetime`` represents the recording **start** time
       (``true``, default) or **end** time (``false``).
       See :ref:`manual-datetimes`.
   * - ``bad_channels``
     - list or dict
     - No
     - Channels to exclude from analysis. Accepts two formats:

       * **List** — channels bad across *all* sessions:
         ``["LHip", "RHip"]``
       * **Dict** — per-session bad channels:
         ``{"Session1": ["LHip"], "Session2": ["RMot"]}``

       See :ref:`bad-channels` for details.
   * - ``pattern``
     - string
     - No
     - Per-animal file discovery pattern, overriding the global
       ``pattern`` in the dataset config. Supports ``{data_root}``,
       ``{animal}``, and ``{index}`` placeholders
       (e.g. ``"{data_root}/custom/{animal}_{index}.rhd"``).
   * - ``lro_kwargs``
     - dict
     - No
     - Per-animal keyword arguments passed to
       ``LongRecordingOrganizer``, overriding the global ``lro_kwargs``.
       Useful when an animal's files require different loading parameters
       (e.g. ``{"mode": "si", "extract_func": "read_intan"}``).
   * - ``day_parse_kwargs``
     - dict
     - No
     - Per-animal keyword arguments for day/date parsing from filenames,
       overriding the global ``day_parse_kwargs``
       (e.g. ``{"date_patterns": [["\\d{6}", "%y%m%d"]]}``).
   * - ``exclude``
     - bool
     - No
     - If ``true``, the animal is skipped entirely by the pipeline.
       The entry is retained in the config for documentation but no
       rules are executed for it. Default: ``false``.

Any additional keys (beyond those listed above) are passed through to
``ANIMAL_METADATA`` and are available for custom downstream processing.

Top-Level Config Keys
~~~~~~~~~~~~~~~~~~~~~

In addition to ``animals``, the samples JSON supports these top-level keys:

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - Key
     - Description
   * - ``data_root``
     - **Required.** Root path containing the raw data directories.
   * - ``channels``
     - Mapping of each canonical channel abbreviation to the **exact** raw channel
       names that appear in the data (e.g. ``{"LAud": ["0"], "RAud": ["5"], ...}``).
       Channel resolution is an exact lookup against these names — no inference.
   * - ``GENOTYPE_MAP``
     - Optional exact ``{canonical_label: [accepted spellings]}`` map normalizing
       the per-animal ``genotype`` value (e.g. collapsing full strings to ``"KO"`` /
       ``"WT"``). Omitted = passthrough (raw values used as-is). When set it is
       authoritative: a value it does not cover raises at load. ``SEX_MAP`` works
       the same way for ``sex``.
   * - ``bad_channels``
     - Legacy top-level bad-channel dict (see :ref:`bad-channels`).
       Prefer per-animal ``bad_channels`` in the ``animals`` list.

.. _bad-channels:

Bad Channels
~~~~~~~~~~~~

Bad channels can be specified per animal in two ways.

**Channels bad across all sessions** — use a list:

.. code-block:: json

   {
       "animals": [
           {
               "id": "M1", "genotype": "WT", "sex": "M",
               "bad_channels": ["LHip", "RHip"]
           }
       ]
   }

This is the simplest approach when the same channels are consistently
noisy for a given animal.

**Per-session bad channels** — use a dict mapping session identifiers to
channel lists:

.. code-block:: json

   {
       "animals": [
           {
               "id": "M1", "genotype": "WT", "sex": "M",
               "bad_channels": {
                   "Session1": ["LHip", "RHip"],
                   "Session2": ["LHip", "RHip", "LMot"]
               }
           }
       ]
   }

You can also combine both approaches: use the list for channels that are
broadly bad across sessions and add per-session entries for channels that
are only bad in specific recordings. When both ``_all`` (from a list) and
per-session entries are present, the pipeline merges them automatically.

.. _manual-datetimes:

Manual Datetimes
~~~~~~~~~~~~~~~~

Recording timestamps are specified via the ``manual_datetime`` field on
each animal entry. **By default this is the recording start time.**

.. code-block:: json

   {
       "animals": [
           {
               "id": "M1", "genotype": "WT", "sex": "M",
               "manual_datetime": "2025-05-10T10:00:00"
           }
       ]
   }

This manual approach avoids the complexity and fragility of automatic
datetime parsing from heterogeneous filename formats. The value is parsed
by ``dateutil.parser.parse``, so any standard format is accepted:

* ISO 8601: ``"2025-05-10T10:00:00"`` (recommended)
* Spaced: ``"2025-05-10 10:00:00"``
* Date only: ``"2025-05-10"`` (midnight assumed)

**Start vs. end time.** By default ``manual_datetime`` is treated as the
recording **start** time.  To indicate an **end** time instead, set
``datetimes_are_start`` to ``false`` on the same animal entry:

.. code-block:: json

   {
       "animals": [
           {
               "id": "M1", "genotype": "WT", "sex": "M",
               "manual_datetime": "2025-05-10T22:00:00",
               "datetimes_are_start": false
           }
       ]
   }

Alternatively, ``datetimes_are_start`` can be set inside the ``lro_kwargs``
dict on the animal entry or in the global ``lro_kwargs`` of the dataset
config YAML.

Per-session anchors and ``null`` cumulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a recording is split into multiple sessions (the ``{session}``
placeholder in the ``pattern``), ``manual_datetime`` may be a **dict keyed by
session**. Each value is either an explicit start, or ``null`` to *cumulate
forward* — i.e. start where the previous session ended (assumes the sessions
are contiguous). Only the first session needs a known time:

.. code-block:: json

   "manual_datetime": { "_0_": "2011-06-07 16:04:14", "_1_": null, "_2_": null }

Dict **insertion order** is the canonical chronological order. Explicit values
act as resets; ``null`` only works in start-time mode.

Per-file explicit starts (gap recovery)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a session is **not** internally contiguous — e.g. one of its
``Selection{index}`` files is missing, leaving a gap — contiguous cumulation
would place every later file in that session too early. Give that session a
**list** of one explicit start per file (in ``{index}`` order); the remaining
sessions can stay scalar or ``null``:

.. code-block:: json

   "manual_datetime": {
       "_0_": ["2011-06-07 16:04:14", "2011-06-07 20:04:12", "2011-06-08 00:04:12",
               "2011-06-08 08:04:13", "2011-06-08 12:04:14"],
       "_1_": "2011-06-08 15:59:48",
       "_2_": null
   }

Here ``_0_`` is missing ``Selection4``, so its fourth and fifth *present* files
(``Selection5``/``Selection6``) are anchored at their true post-gap starts
rather than 4 h early. The list length must equal the number of files
discovered for that session, and entries are matched to files in ``{index}``
(natural-sort) order. A following ``null`` session cumulates from the last
listed file's end. Lists are start-time only.

Full Example
~~~~~~~~~~~~

A complete samples JSON file with all available parameters:

.. code-block:: json

   {
       "data_root": "/mnt/data/project",
       "animals": [
           {"id": "AM3", "genotype": "WT", "sex": "Male"},
           {"id": "AM5", "genotype": "Het", "sex": "Male",
            "bad_channels": ["LHip", "RHip"]},
           {"id": "AP3B2homo-240-M", "genotype": "HOMO", "sex": "Male",
            "pattern": "{data_root}/PortA-*PortB-*/{animal}*_ColMajor_{index}.rhd",
            "manual_datetime": "2025-11-27T15:39:05",
            "lro_kwargs": {"mode": "si"},
            "bad_channels": {
                "Session_Nov27": ["LMot"],
                "Session_Nov28": ["LMot", "RAud"]
            }}
       ],
       "channels": {"LAud": ["0"], "RAud": ["5"], "LVis": ["1"], "RVis": ["6"],
                    "LHip": ["2"], "RHip": ["7"], "LBar": ["3"], "RBar": ["8"],
                    "LMot": ["4"], "RMot": ["9"]}
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
       pattern: "{animal}/{session}/{index}.nwb"
       lro_kwargs:
         mode: "si"
         extract_func: "read_nwb_recording"

You can override **any** config parameter using the same hierarchy as the main config.

Step 3: Use the New Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   NEURODENT_DATASET=mydata_nwb uv run snakemake --profile your-profile

Session-Specific Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some datasets contain mixed file formats that require different processing parameters per recording session. Use the ``overrides.by_session`` section in your dataset config to override any parameter on a per-session basis.

**Example:** Dataset with both EDF and RHD formats

.. code-block:: yaml

   # config/datasets/mixed_format.yaml
   samples:
     samples_file: "config/samples_mixed.json"

   analysis:
     war_generation:
       pattern: "{index}"
       lro_kwargs:
         mode: "si"

   overrides:
     by_session:
       "Session_EDF":
         "analysis.war_generation.pattern": "{index}.EDF"
         "analysis.war_generation.lro_kwargs.mode": "mne"
         "analysis.war_generation.lro_kwargs.extract_func": "read_raw_edf"
       "Session_RHD":
         "analysis.war_generation.pattern": "{index}.rhd"
         "analysis.war_generation.lro_kwargs.extract_func": "read_intan"
         "analysis.war_generation.lro_kwargs.stream_id": "0"

**How it works:**

- Use dotted paths to specify which parameter to override (e.g., ``"analysis.war_generation.pattern"``)
- Can override **any** config parameter, not just war_generation settings
- Overrides are applied via deep merge before session processing
- Falls back to global config if no override specified for a session

Session-specific settings override the dataset config for that session only. This allows processing mixed formats in a single pipeline run while keeping all animals in unified analysis outputs.

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

   # Edit config.local.yaml: active_dataset: "ap3b2_rhd"
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
   NEURODENT_DATASET=ap3b2_rhd uv run snakemake --profile slurm

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
