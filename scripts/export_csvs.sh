#!/usr/bin/env bash
# Export analysis CSVs (and accompanying files) from a finished pipeline
# run into a single zip in ~/Downloads, ready to pull off the cluster.
#
# By default bundles two CSV-bearing directories:
#   - results/ep_data/             (EP figures' per-feature pivot CSVs)
#   - results/zeitgeber_plot_data/ (zeitgeber per-animal processed CSVs)
#
# Both contain CSVs you'd typically pull together for downstream plotting
# on your local machine.
#
# Usage:
#   scripts/export_csvs.sh arxrosa
#     → ~/Downloads/arxrosa-csvs-<YYYY-MM-DD>.zip
#   scripts/export_csvs.sh sox5
#     → ~/Downloads/sox5-csvs-<YYYY-MM-DD>.zip
#   DEST=/tmp scripts/export_csvs.sh arxrosa
#     → /tmp/arxrosa-csvs-<YYYY-MM-DD>.zip
#   SRC="results/ep_data" scripts/export_csvs.sh foo
#     → ~/Downloads/foo-csvs-<...>.zip (only ep_data)
#
# A run name is REQUIRED — the script refuses to run without one.  Each
# export's filename includes the current date so re-exports on different
# days never overwrite each other; you build up a dated history of CSV
# snapshots in DEST.  Same-day re-exports replace that day's snapshot.
#
# Before running:
#   Make sure results/{ep_data,zeitgeber_plot_data}/ hold the run you want.
#   If you've run the pytest integration suite since your real pipeline
#   run, those dirs may have stale mini_real outputs (A10/F22) — rerun
#   ep_figures + zeitgeber_plots against your real dataset first.
#   Recommended split (zeitgeber_plots in its own invocation avoids
#   snakemake #823):
#     # 1) ep + war_zeitgeber pkl
#     NEURODENT_DATASET=arx_rosa uv run snakemake --snakefile workflow/Snakefile \
#       --forcerun war_zeitgeber ep_figures ep_heatmaps \
#       --until war_zeitgeber ep_figures ep_heatmaps \
#       --profile slurm.pyeeg.yydemo
#     # 2) zeitgeber plots (reads the pkl from step 1)
#     NEURODENT_DATASET=arx_rosa uv run snakemake --snakefile workflow/Snakefile \
#       --forcerun zeitgeber_plots --until zeitgeber_plots \
#       --profile slurm.pyeeg.yydemo
set -euo pipefail

if [[ $# -lt 1 || -z "${1:-}" ]]; then
    echo "usage: $(basename "$0") <run_name>" >&2
    echo "  e.g. $(basename "$0") arxrosa  → ~/Downloads/arxrosa-csvs.zip" >&2
    exit 2
fi

RUN_NAME="$1"
DEST_DIR="${DEST:-$HOME/Downloads}"
# Space-separated list of source dirs.  Override via SRC env var.
SRC_DIRS="${SRC:-results/ep_data results/zeitgeber_plot_data}"

# Validate every src dir exists before touching the destination.
missing=()
for d in $SRC_DIRS; do
    [[ -d "$d" ]] || missing+=("$d")
done
if (( ${#missing[@]} > 0 )); then
    echo "error: source dir(s) not found: ${missing[*]}" >&2
    echo "  (run ep_figures + zeitgeber_plots first; see header for command)" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
TODAY="$(date +%Y-%m-%d)"
OUT="$DEST_DIR/${RUN_NAME}-csvs-${TODAY}.zip"

# -r recurses into each src dir.  Paths are preserved so the zip unpacks
# to ep_data/ and zeitgeber_plot_data/ side by side (no -j flag).
# `rm -f` first so a same-day re-export replaces that day's snapshot
# rather than appending to it.
rm -f "$OUT"
zip -r "$OUT" $SRC_DIRS

echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
