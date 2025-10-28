# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../src'))

# Read project metadata from pyproject.toml
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python 3.10 and below

pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project_info = pyproject_data["project"]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = project_info["name"].title()  # "neurodent" -> "Neurodent"
release = project_info["version"]
author = ", ".join([a["name"] for a in project_info["authors"]])
copyright = f'2025, {author}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_nb",
    "sphinx_design",
    "sphinx_multiversion",
]

# MyST parser configuration (for markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # Map common type aliases for better cross-referencing
    'array-like': ':term:`array-like <numpy:array_like>`',
    'array_like': ':term:`array-like <numpy:array_like>`',
    'ndarray': '~numpy.ndarray',
    'DataFrame': '~pandas.DataFrame',
    'Series': '~pandas.Series',
    'Raw': '~mne.io.Raw',
    'Epochs': '~mne.Epochs',
}
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autosummary_generate = True

# Configure sphinx_autodoc_typehints to not link built-in types
typehints_defaults = 'braces'
always_use_bars_union = True
typehints_use_signature = True
typehints_use_signature_return = True

# Ensure all modules are imported for module index
autodoc_mock_imports = []

# Intersphinx mapping for cross-referencing external packages
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'mne': ('https://mne.tools/stable/', None),
    'spikeinterface': ('https://spikeinterface.readthedocs.io/en/stable/', None),
}

# Speed up intersphinx by setting timeout and cache
intersphinx_timeout = 10  # seconds - faster timeout for slow sites
intersphinx_cache_limit = 5  # days - cache inventories

# Disable intersphinx for built-in types to avoid linking int, str, etc.
intersphinx_disabled_reftypes = ['*']

# MyST-NB configuration (replaces nbsphinx)
nb_execution_mode = 'off'  # Don't execute notebooks during build
nb_execution_allow_errors = False
nb_execution_timeout = 600  # 10 minutes timeout for notebook execution

# Enable MyST extensions for notebooks
myst_url_schemes = ['http', 'https', 'mailto']
myst_heading_anchors = 3  # Auto-generate anchors for headings

# Make MyST use default_role for inline code
# This allows `ClassName` in markdown to auto-link to API docs
nb_custom_formats = {
    '.ipynb': ['jupytext.reads', {'fmt': 'ipynb'}]
}

# Configure MyST to process inline code as cross-references
# when they match known Python objects
myst_dmath_double_inline = True

# Exclude patterns
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
]

templates_path = ['_templates']

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master document
master_doc = 'index'

# Default role for inline code (allows `code` to auto-link to Python objects)
default_role = 'py:obj'

# -- Sphinx-multiversion configuration ---------------------------------------
# Whitelist pattern for tags (Sphinx will build docs for tags matching this pattern)
# NOTE: Old tags may not have Sphinx docs, so they might be skipped
smv_tag_whitelist = r"^v\d+\.\d+.*$"  # Matches v0.1.0, v0.2.0, etc.

# Whitelist pattern for branches - only match local branches (not remotes)
smv_branch_whitelist = r"^main$|^develop$"

# Whitelist pattern for remotes - None means only use local branches
smv_remote_whitelist = None

# Pattern for released versions (tags only, not branches)
smv_released_pattern = r"^refs/tags/v\d+\.\d+.*$"

# Output directory for versioned docs (use ref.name for simple naming)
smv_outputdir_format = "{ref.name}"

# Prefer latest branch if available
smv_prefer_remote_refs = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
    "github_url": "https://github.com/josephdong1000/neurodent",
    "show_nav_level": 2,
    "navigation_depth": 4,
    "show_toc_level": 2,
    "navbar_align": "left",
    "logo": {
        "text": "Neurodent",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/neurodent/",
            "icon": "fab fa-python",
        },
    ],
    "switcher": {
        "json_url": "https://josephdong1000.github.io/neurodent/_static/switcher.json",
        "version_match": release,
    },
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
}

# Force all pages to use the main navigation sidebar
html_sidebars = {
    "**": ["sidebar-nav-bs.html"]
}

# HTML context for version information
html_context = {
    "github_user": "josephdong1000",
    "github_repo": "neurodent",
    "github_version": "main",
    "doc_path": "docs",
}

# Output file base name for HTML help builder
htmlhelp_basename = f'{project}doc'

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {}

# Grouping the document tree into LaTeX files
latex_documents = [
    (master_doc, f'{project}.tex', f'{project} Documentation',
     author, 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, project.lower(), f'{project} Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, project, f'{project} Documentation',
     author, project, project_info["description"],
     'Miscellaneous'),
]
