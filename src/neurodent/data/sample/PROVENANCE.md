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

These are real intracranial recordings acquired on an Intan system: A10 on port C
(channels C-009 to C-022) and F22 on port D. The `.bin` files store float32 samples in
column-major order: every sample of channel 0, then channel 1, and so on.
`neurodent.readers.read_bin_csv_pair` derives the sample count from file size, so the
sample counts and rates are unchanged from the full-length originals.

One metadata field is deliberately not verbatim. The acquisition software writes the
`ProbeInfo` column as `Intan Input (1)/PortC C-009`, whose prefix repeats the port
letter already present in the channel id and adds nothing but length. Because these
recordings are what the tutorials read, that column is shortened here and in
`.tests/integration/data/` to the bare id. `config/datasets/sox5_bin.yaml` maps against
unmodified export names, and `tests/test_readers.py` covers the prefixed form directly.

The `.bin` files here are the first 60 seconds of longer recordings that live in
`.tests/integration/data/` in the source repository, where the full-length versions
drive the Snakemake pipeline integration tests. The `.edf` files are copied from there
unchanged. Regenerate this directory with:

    python scripts/make_sample_dataset.py
