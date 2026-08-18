# Sample recordings

Two rodent EEG recordings used by the NeuRodent documentation tutorials and shipped
with the package so the examples run without a download.

| Animal | Genotype | Sex | Channels | Rate | `.bin` length | `.edf` length |
|---|---|---|---|---|---|---|
| A10 | WT | Male | 10 | 1000 Hz | 60 s | 5 s |
| F22 | KO | Female | 10 | 1000 Hz | 60 s | 5 s |

Each directory holds a paired ColMajor `.bin` and Meta `.csv`, which is what the
tutorials analyse, plus a shorter single-file `.edf` used only to demonstrate loading
a standard format.

These are real intracranial recordings acquired on an Intan system (Port C, channels
C-009 to C-022). The `.bin` files store float32 samples in column-major order: every
sample of channel 0, then channel 1, and so on.
`neurodent.readers.read_bin_csv_pair` derives the sample count from file size, so the
metadata is unchanged from the full-length originals.

The `.bin` files here are the first 60 seconds of longer recordings that live in
`.tests/integration/data/` in the source repository, where the full-length versions
drive the Snakemake pipeline integration tests. The `.edf` files are copied from there
unchanged. Regenerate this directory with:

    python scripts/make_sample_dataset.py
