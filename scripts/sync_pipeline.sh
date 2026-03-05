#!/usr/bin/env bash
# sync_pipeline.sh – Keep the bundled pipeline in src/neurodent/pipeline/
# in sync with the canonical root-level Snakefile, config/, and workflow/ files.
#
# Usage:
#   scripts/sync_pipeline.sh          # copy root → bundled
#   scripts/sync_pipeline.sh --check  # exit non-zero if out of sync (for CI)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINE_DIR="$REPO_ROOT/src/neurodent/pipeline"

# Files/dirs to sync (relative to repo root → pipeline dir)
ITEMS=(
    "Snakefile"
    "config"
    "workflow"
)

check_mode=false
if [[ "${1:-}" == "--check" ]]; then
    check_mode=true
fi

drift=0

for item in "${ITEMS[@]}"; do
    src="$REPO_ROOT/$item"
    dst="$PIPELINE_DIR/$item"

    if [[ ! -e "$src" ]]; then
        echo "WARN: root item '$item' does not exist – skipping"
        continue
    fi

    if [[ -d "$src" ]]; then
        # Compare directories recursively
        if ! diff -rq "$src" "$dst" > /dev/null 2>&1; then
            if $check_mode; then
                echo "DRIFT: $item/ is out of sync"
                diff -rq "$src" "$dst" || true
                drift=1
            else
                rm -rf "$dst"
                cp -a "$src" "$dst"
                echo "Synced $item/"
            fi
        else
            echo "OK: $item/"
        fi
    else
        # Compare single file
        if ! diff -q "$src" "$dst" > /dev/null 2>&1; then
            if $check_mode; then
                echo "DRIFT: $item is out of sync"
                drift=1
            else
                cp "$src" "$dst"
                echo "Synced $item"
            fi
        else
            echo "OK: $item"
        fi
    fi
done

if $check_mode && [[ $drift -ne 0 ]]; then
    echo ""
    echo "ERROR: Bundled pipeline is out of sync with root files."
    echo "Run 'scripts/sync_pipeline.sh' to fix."
    exit 1
fi

echo ""
echo "Pipeline sync complete."
