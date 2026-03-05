# Example Dataset for Pipeline Integration Testing

This directory contains infrastructure for testing the Snakemake pipeline
**without** processing full production-scale recordings.

## Quick start

There are two ways to use example data for pipeline tests:

### 1. Programmatic synthetic data (CI-friendly, no files to commit)

The test fixtures in `tests/conftest.py` provide an
`example_dataset` fixture that **generates** a tiny synthetic NWB
dataset on the fly inside a `tmp_path` directory.  Tests in
`tests/integration/test_snakemake_flow.py` use this fixture.

Run integration tests:

```bash
uv run pytest tests/integration/ -v -m integration
```

### 2. Mini real dataset (committed bin/csv recordings)

Small real recordings are committed under `tests/data/raw/` for local
smoke-testing.  The layout uses `{animal}/{index}` placeholders:

```
tests/data/
└── raw/
    ├── A10/
    │   ├── Cage 2 A10-0_ColMajor.bin
    │   └── Cage 2 A10-0_Meta.csv
    └── F22/
        ├── Cage 3 F22-0_ColMajor.bin
        └── Cage 3 F22-0_Meta.csv
```

Run with the `mini_real` dataset:

```bash
NEURODENT_DATASET=mini_real snakemake --cores 1 --dryrun
```

The matching configuration files are:
- `config/datasets/mini_real.yaml`  — dataset config (bin/csv multi-pattern)
- `config/samples_mini_real.json`   — sample mapping

## File size guidelines

To keep the repository lean:

- **Raw EEG files** should be ≤ 5 MB per animal-day.  A few seconds of
  8-channel 1 kHz `float32` data is sufficient.
- **WAR pickle files** are generated, not committed.
- Files in `tests/data/raw/` are tracked by git.
  Keep individual files small where possible.

## Adding your own example data

1. Place raw files under `tests/data/raw/` in session folders.
2. Create or update a `config/samples_*.json` with the correct folder→animal
   mapping and `ANIMAL_METADATA` entries.
3. Create or update a `config/datasets/*.yaml` with the correct pattern and
   loader settings.
4. Run with `NEURODENT_DATASET=<your_dataset> snakemake --cores 1`.
