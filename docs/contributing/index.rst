Contributing
============

We welcome contributions to NeuRodent! This guide will help you get started.

.. toctree::
   :maxdepth: 1
   :hidden:

   setup
   testing
   naming

Quick Start
-----------

.. code-block:: bash

   git clone https://github.com/josephdong1000/neurodent.git
   cd neurodent
   make setup  # Requires uv

.. grid:: 2

   .. grid-item-card:: :octicon:`tools` Setup Guide
      :link: setup
      :link-type: doc

      Environment setup, prerequisites, pre-commit hooks, and Makefile commands.

   .. grid-item-card:: :octicon:`beaker` Testing & Development
      :link: testing
      :link-type: doc

      Running tests, building documentation, and code style guidelines.

   .. grid-item-card:: :octicon:`book` Naming Conventions
      :link: naming
      :link-type: doc

      The verb/noun canon: which verb for which operation, and the noun rules
      (``_MAP`` vs ``_ALIASES``, config levels, channel terminology).

Pull Request Process
--------------------

1. Fork the repository and create a new branch for your feature or bug fix
2. Make your changes and add tests if applicable
3. Ensure all tests pass and documentation builds successfully
4. Submit a pull request with a clear description of the changes

Questions?
----------

If you have questions, please open an issue on the `GitHub repository <https://github.com/josephdong1000/neurodent>`_.
