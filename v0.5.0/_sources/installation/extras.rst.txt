Installation Extras
===================

NeuRodent provides optional dependency groups for different use cases. Install any extra with:

.. code-block:: bash

   pip install neurodent[extra_name]
   # or with uv:
   uv add neurodent[extra_name]

.. note::

   **Using conda?** Conda does not support pip-style extras. To use optional dependency groups, install NeuRodent via pip or uv instead of conda.

Available Extras
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Description
   * - ``[pipeline]``
     - Snakemake and dependencies for running automated analysis workflows
   * - ``[readers]``
     - Readers for EDF and NWB files (``edfio``, ``pyedflib``, ``pynwb``)
   * - ``[dev]``
     - Development tools plus the pipeline and readers extras (for contributors)
   * - ``[all]``
     - Installs all optional runtime dependencies

``[pipeline]`` Extra
^^^^^^^^^^^^^^^^^^^^

The pipeline extra includes Snakemake and related dependencies for running the automated analysis workflow on single machines or HPC clusters.

.. code-block:: bash

   pip install neurodent[pipeline]

See :doc:`pipeline` for SLURM cluster configuration.

``[readers]`` Extra
^^^^^^^^^^^^^^^^^^^

Readers for formats not covered by the core dependencies. ``.edf`` and ``.nwb`` files need this
extra; the documentation tutorials use it for their single-file loading example.

.. code-block:: bash

   pip install neurodent[readers]

``[dev]`` Extra
^^^^^^^^^^^^^^^

The development extra includes everything needed to contribute to NeuRodent:

.. code-block:: bash

   pip install neurodent[dev]

See the
`canonical CONTRIBUTING.md guide <https://github.com/josephdong1000/neurodent/blob/main/CONTRIBUTING.md>`_
for development setup instructions.

``[all]`` Extra
^^^^^^^^^^^^^^^

Installs ``[pipeline]`` and ``[readers]`` together.

.. code-block:: bash

   pip install neurodent[all]
