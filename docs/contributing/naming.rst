Naming Conventions
==================

NeuRodent uses a small, predictable naming vocabulary so that a reader fluent in the
scientific-Python ecosystem (MNE-Python, scikit-learn, pandas) can predict an API from its
name. The rule of thumb: **one verb per kind of operation, one noun per concept.** When you
add or rename a function, constant, parameter, or config key, follow the canon below.

Verb canon
----------

Each operation has a single canonical verb, matched to an established precedent.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Operation
     - Verb
     - Notes / precedent
   * - Mutate a single global/config value
     - ``set_``
     - sklearn ``set_params``, MNE ``set_config``/``set_montage``. E.g.
       :func:`~neurodent.set_channel_map`. Pairs with ``get_``.
   * - Read a single global/config value
     - ``get_``
     - sklearn ``get_params``, MNE ``get_config``. No side effects, no disk I/O.
   * - Install a whole config dict into globals
     - ``apply_``
     - Distributes one config across many globals. ``apply_samples_config`` completes the
       ``load_samples_config`` → ``resolve_samples_config`` → ``apply_samples_config`` trio.
   * - Read from disk (format-named parser)
     - ``read_``
     - pandas ``read_parquet``, MNE ``read_raw_edf``. Use when the suffix names the format.
   * - Reconstruct a known object from its files
     - ``load_``
     - joblib ``load``; repo ``load_wars``. Use when rebuilding a known object from its
       own paired files.
   * - Numeric/statistical derivation (pure)
     - ``compute_``
     - A real calculation (RMS, PSD, coherence, …).
   * - Cheap accessor / lookup
     - ``get_``
     - Returns a stored or trivially-derived value.
   * - Pull a sub-object out of a container
     - ``extract_``
     - Reserve for "carve a piece out of a larger structure."
   * - Construct an object
     - ``create_``
     - The single construct verb. Do **not** introduce ``make_`` or ``build_``.
   * - Produce a Snakemake rule result
     - ``generate_``
     - **Pipeline scripts only** (``workflow/scripts/generate_*.py``). This is rule/log
       vocabulary, not a general object factory.
   * - Canonicalize one identifier value
     - ``resolve_``
     - E.g. :func:`~neurodent.core.resolve_channel` (raw name → canonical abbrev).
   * - Rename labels on an object, in place
     - ``rename_``
     - MNE ``rename_channels``. E.g. ``rename_mne_channels``.
   * - Format / dtype conversion
     - ``convert_``
     - Strictly format/type changes — never identifier mapping (that is ``resolve_``).
   * - String → identifier extraction
     - ``parse_``
     - E.g. ``parse_str_to_genotype``. Not a disk-I/O verb.
   * - Spelling
     - US ``normalize``
     - NumPy/SciPy/sklearn spelling. Never ``normalise``.

``get_`` / ``compute_`` / ``extract_`` are already well-partitioned across the codebase;
the boundary above is the rule to keep them that way.

Noun canon
----------

**``_MAP`` vs ``_ALIASES``.** The suffix encodes the resolution semantics:

* ``_MAP`` — an **exact** ``{key: [values]}`` lookup. :data:`~neurodent.constants.CHANNEL_MAP`
  maps a canonical channel abbreviation to the exact raw spellings that resolve to it
  (matched exactly by :func:`~neurodent.core.resolve_channel`).
* ``_ALIASES`` — a **fuzzy**, variant-spelling family matched by substring. ``GENOTYPE_ALIASES``,
  ``SEX_ALIASES``, ``GENE_ALIASES`` map a canonical label to the spellings it may appear as
  in filenames/metadata.

Never name an exact map ``*_ALIASES`` or a fuzzy variant-list ``*_MAP``.

**Channel terminology.** Three distinct concepts:

* *raw name* — the channel label as it appears in the source data (e.g. ``"EEG E1-REF1"``,
  ``"A-014"``).
* ``channel_abbrevs`` — the canonical abbreviations (``"LMot"``, ``"LHip"``, …), always
  derived from the raw names via :func:`~neurodent.core.resolve_channel`.
* ``channel_names`` — the *current working* labels on a result: the raw names at
  construction, or the abbreviations after standardization with ``use_abbrevs=True``.

**Config levels.** "Config" spans three nesting levels; keep them distinct in prose and
variable names:

* *pipeline config* — the merged Snakemake ``config`` object (framework-owned; not renamed).
* *dataset config* — a ``config/datasets/*.yaml`` file.
* *samples config* — the in-memory inventory dict (animals, channels, metadata) that
  ``apply_samples_config`` installs. The inline YAML key for it stays ``samples_data``.

**The two ``channels`` keys.** A samples config has a top-level ``channels:`` map
(``{abbrev: [raw names]}``, the montage) and, per animal in a joint recording, a
``channel_subset:`` list (the raw names belonging to that animal). They are different
concepts — do not reuse one key for both.

Channel API entry points
------------------------

Both the package and the pipeline funnel through the same single source of truth
(:data:`~neurodent.constants.CHANNEL_MAP`):

* **Package users** call :func:`neurodent.set_channel_map` once with
  ``{abbrev: [raw names]}``.
* **Pipeline users** declare ``channels:`` in the samples config; each Snakemake script
  calls ``apply_samples_config`` (which calls ``set_channel_map`` internally).

Resolution is always exact (:func:`~neurodent.core.resolve_channel`); an unmapped raw name
raises loudly rather than being inferred.

Migration policy
----------------

Renames are **hard cuts** — no backwards-compatibility aliases. A stale caller should fail
loudly (``ImportError``/``NameError``) rather than silently resolve to deprecated behavior.
Renames never touch serialized on-disk data (the persisted ``channel_names`` JSON key is
intentionally stable), so already-saved results keep loading.
