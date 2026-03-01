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

### 2. User-provided real data (for local development)

If you have a small real recording you'd like to test with, place it here
following the nested directory layout:

```
tests/data/
└── raw/
    └── session_folder/          # e.g. "example_session"
        └── AnimalA/             # animal ID
            └── day1/            # session / day folder
                └── recording.nwb    # NWB file
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
  8-channel 1 kHz `float32` data is sufficient.
- **WAR pickle files** are generated, not committed.
- Binary files (`*.nwb`, `*.pkl`, `*.npy.gz`) in this directory are
  git-ignored by default.

## Adding your own example data

1. Place raw files under `tests/data/raw/` using the structure above.
2. Update `config/samples_example.json` with the correct folder→animal mapping
   and `ANIMAL_METADATA` entries.
3. Run with `NEURODENT_DATASET=example snakemake --cores 1`.
