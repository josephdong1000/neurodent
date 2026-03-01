# Example Dataset for Pipeline Integration Testing

This directory contains (or will contain) a minimal example dataset used for
integration-testing the Snakemake pipeline **without** processing full
production-scale recordings.

## Quick start

There are two ways to use example data for pipeline tests:

### 1. Programmatic synthetic data (CI-friendly, no files to commit)

The test fixtures in `tests/conftest.py` provide a
`example_dataset` fixture that **generates** a tiny synthetic
dataset on the fly inside a `tmp_path` directory.  Tests in
`tests/integration/test_snakemake_flow.py` use this fixture.

Run integration tests:

```bash
uv run pytest tests/integration/ -v -m integration
```

### 2. User-provided real data (for local development)

If you have a small real recording you'd like to test with, place it here
following the **nest** directory convention used by the default `sox5_bin`
dataset:

```
tests/example_data/
└── raw/
    └── session_folder/          # e.g. "example_session"
        └── AnimalA/             # animal ID
            └── day1/            # session / day folder
                ├── rec_ColMajor.bin   # column-major binary data
                └── rec_Meta.csv       # metadata CSV
```

Then set `NEURODENT_DATASET=example` before running the pipeline:

```bash
NEURODENT_DATASET=example snakemake --cores 1 --dryrun
```

The matching configuration files are:
- `config/datasets/example.yaml`  — dataset config
- `config/samples_example.json`   — sample mapping

## File size guidelines

To keep the repository lean:

- **Raw EEG files** should be ≤ 5 MB per animal-day.  A few seconds of
  8-channel 1 kHz `float32` data (~32 kB/s) is sufficient.
- **WAR pickle files** are generated, not committed.
- Binary files (`*.bin`, `*.pkl`, `*.npy.gz`) in this directory are
  git-ignored by default.  If you want to track specific files, add an
  exception to `.gitignore` (similar to the `notebooks/tests/` pattern).

## Adding your own example data

1. Place raw files under `tests/example_data/raw/` using the structure above.
2. Update `config/samples_example.json` with the correct folder→animal mapping
   and `ANIMAL_METADATA` entries.
3. Run with `NEURODENT_DATASET=example snakemake --cores 1`.
