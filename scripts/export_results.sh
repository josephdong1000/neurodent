#!/usr/bin/env bash
# Export finished-pipeline results into dated zip(s) in ~/Downloads, ready to
# pull off the cluster.  Bundles whole result dirs, preserving their paths.
#
# Two kinds:
#   csvs  -> results/ep_data/  results/zeitgeber_plot_data/
#   plots -> results/{ep_figures,ep_heatmaps,zeitgeber_plots,relfreq_plots,
#                     diagnostic_figures,fdsar_diagnostics,
#                     filtering_comparison_plots,lof_evaluation}/
#
# Usage:
#   scripts/export_results.sh <run_name> [csvs|plots|all]   (default: all)
#     scripts/export_results.sh arxrosa          -> arxrosa-csvs-<date>.zip + arxrosa-plots-<date>.zip
#     scripts/export_results.sh arxrosa plots    -> arxrosa-plots-<date>.zip only
#   DEST=/tmp scripts/export_results.sh arxrosa  -> write zips to /tmp instead
#   CSV_DIRS / PLOT_DIRS env vars override the source-dir lists.
#
# A run name is REQUIRED.  The filename includes the date, so re-exports on
# different days don't overwrite; same-day re-exports replace that snapshot.
# Dirs that don't exist yet (a stage hasn't run) are skipped with a note.
#
# Before exporting plots/csvs: make sure the relevant dirs hold THIS run (not
# stale mini_real pytest outputs).  If needed, regenerate first, e.g.:
#   NEURODENT_DATASET=arx_rosa uv run snakemake \
#     --forcerun war_zeitgeber ep_figures ep_heatmaps \
#     --until   war_zeitgeber ep_figures ep_heatmaps --profile slurm.pyeeg.yydemo
#   NEURODENT_DATASET=arx_rosa uv run snakemake \
#     --forcerun zeitgeber_plots --until zeitgeber_plots --profile slurm.pyeeg.yydemo
set -euo pipefail

if [[ $# -lt 1 || -z "${1:-}" ]]; then
    echo "usage: $(basename "$0") <run_name> [csvs|plots|all]" >&2
    echo "  e.g. $(basename "$0") arxrosa  -> ~/Downloads/arxrosa-{csvs,plots}-<date>.zip" >&2
    exit 2
fi

RUN_NAME="$1"
KIND="${2:-all}"
DEST_DIR="${DEST:-$HOME/Downloads}"
CSV_DIRS="${CSV_DIRS:-results/ep_data results/zeitgeber_plot_data}"
PLOT_DIRS="${PLOT_DIRS:-results/ep_figures results/ep_heatmaps results/zeitgeber_plots results/relfreq_plots results/diagnostic_figures results/fdsar_diagnostics results/filtering_comparison_plots results/lof_evaluation}"

# bundle <label> <mode: dir|img> <space-separated dirs> -> one dated zip.
# mode=dir : zip whole dirs (CSV dirs are clean data dirs).
# mode=img : zip only image files -- the figure dirs ALSO hold large .fif
#            diagnostic-epoch data we don't want in a "plots" bundle.
bundle() {
    local label="$1" mode="$2" dirs="$3" existing=() d
    for d in $dirs; do
        if [[ -d "$d" ]]; then existing+=("$d"); else echo "  skip (not found): $d" >&2; fi
    done
    if (( ${#existing[@]} == 0 )); then
        echo "  no $label dirs exist yet -- skipping $label" >&2
        return
    fi
    mkdir -p "$DEST_DIR"
    local out="$DEST_DIR/${RUN_NAME}-${label}-$(date +%Y-%m-%d).zip"
    rm -f "$out"   # same-day re-export replaces rather than appends
    if [[ "$mode" == img ]]; then
        find "${existing[@]}" -type f \
            \( -iname '*.png' -o -iname '*.pdf' -o -iname '*.svg' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
            -exec zip -q "$out" {} +
    else
        zip -qr "$out" "${existing[@]}"
    fi
    if [[ -f "$out" ]]; then
        echo "wrote $out  ($(du -h "$out" | cut -f1))"
    else
        echo "  no $label files found under: ${existing[*]}" >&2
    fi
}

case "$KIND" in
    csvs)  bundle csvs  dir "$CSV_DIRS" ;;
    plots) bundle plots img "$PLOT_DIRS" ;;
    all)   bundle csvs dir "$CSV_DIRS"; bundle plots img "$PLOT_DIRS" ;;
    *)     echo "error: kind must be csvs|plots|all (got '$KIND')" >&2; exit 2 ;;
esac
