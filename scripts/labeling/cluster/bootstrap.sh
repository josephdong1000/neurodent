#!/bin/bash
# Create or verify the two worktrees of the autoreject evaluation (scripts/labeling/CLUSTER_RUNBOOK.md, step 0).
#
#     bash bootstrap.sh MAIN TIP [PORT [JULY]]
#
# MAIN is the owner's clone that holds the neurodent_wt worktrees; TIP is the full hash of the commit of
# origin/feat/autoreject-port that the handoff names. PORT (default .../neurodent_wt/autoreject) becomes a
# worktree of the branch feat/autoreject-port at TIP, and JULY (default .../neurodent_wt/july) a detached
# worktree at 413535429de06a92341471495678e8c8dab7ee14.
#
# It fetches origin's feat/autoreject-port and feature/autoreject into remote-tracking refs only, then creates
# each tree that does not exist and verifies each that does. A tree passes when it is a worktree of MAIN, at
# its commit (PORT may also be TIP plus commits that change only config/labeling/recording_map.csv), with a
# clean status and no config/config.local.yaml. It stops at the first thing it did not expect: origin at
# another commit, an existing path or branch that does not pass, or a tree that is not clean. It never checks
# out, stashes, resets or commits in MAIN, and it never uses --force. Exit 0 when both trees are ready.

set -euo pipefail

WT_ROOT=/mnt/isilon/marsh_single_unit/YY_PyEEG/neurodent_wt
BRANCH=feat/autoreject-port
JULY_COMMIT=413535429de06a92341471495678e8c8dab7ee14
MAP=config/labeling/recording_map.csv

die() {
    echo "bootstrap: $*" >&2
    echo "bootstrap: stopped; nothing after this point was changed. Report this message to the owner." >&2
    exit 1
}

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "usage: bash bootstrap.sh MAIN TIP [PORT [JULY]]" >&2
    exit 2
fi
MAIN=$1
TIP=$2
PORT=${3:-$WT_ROOT/autoreject}
JULY=${4:-$WT_ROOT/july}

[[ $TIP =~ ^[0-9a-f]{40}$ ]] || die "TIP must be a full 40-character commit hash, not '$TIP'"
git -C "$MAIN" rev-parse --git-dir >/dev/null 2>&1 || die "$MAIN is not a git repository"

# common_dir DIR: the absolute git directory DIR's repository shares, or nothing.
common_dir() {
    (cd "$1" 2>/dev/null && cd "$(git rev-parse --git-common-dir 2>/dev/null)" 2>/dev/null && pwd -P) || true
}
COMMON=$(common_dir "$MAIN")
main_head=$(git -C "$MAIN" rev-parse HEAD)
main_branch=$(git -C "$MAIN" symbolic-ref -q --short HEAD || echo "(detached)")
main_stashes=$(git -C "$MAIN" stash list | wc -l | tr -d ' ')

echo "git: $(git --version)"
echo "MAIN: $MAIN (git directory $COMMON), HEAD $main_head on $main_branch, $main_stashes stash entries"
echo "origin: $(git -C "$MAIN" config --get remote.origin.url || echo 'not set')"
git -C "$MAIN" worktree list || true

git -C "$MAIN" fetch origin \
    "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" \
    "+refs/heads/feature/autoreject:refs/remotes/origin/feature/autoreject"
remote_tip=$(git -C "$MAIN" rev-parse "refs/remotes/origin/$BRANCH")
git -C "$MAIN" merge-base --is-ancestor "$JULY_COMMIT" refs/remotes/origin/feature/autoreject \
    || die "$JULY_COMMIT is not on origin/feature/autoreject"

# map_only BASE COMMIT: true when COMMIT is BASE, or descends from it and changes only the recording map.
map_only() {
    [ "$2" = "$1" ] && return 0
    git -C "$MAIN" merge-base --is-ancestor "$1" "$2" || return 1
    [ "$(git -C "$MAIN" diff --name-only "$1" "$2")" = "$MAP" ]
}

# check_tree DIR: DIR is the top of a worktree of MAIN, with a clean status and no config.local.yaml.
check_tree() {
    [ "$(common_dir "$1")" = "$COMMON" ] || die "$1 exists but is not a worktree of $MAIN"
    [ "$(cd "$1" && pwd -P)" = "$(git -C "$1" rev-parse --show-toplevel)" ] \
        || die "$1 lies inside a worktree of $MAIN but is not the top of one"
    local status
    status=$(git -C "$1" status --porcelain)
    [ -z "$status" ] || die "$1 has uncommitted changes or untracked files: $(echo "$status" | head -5 | tr '\n' ' ')"
    [ ! -e "$1/config/config.local.yaml" ] || die "$1 has a config/config.local.yaml"
}

if [ -e "$PORT" ]; then
    map_only "$TIP" "$remote_tip" \
        || die "origin/$BRANCH is at $remote_tip, which is neither $TIP nor $TIP plus the recording map"
    check_tree "$PORT"
    on=$(git -C "$PORT" symbolic-ref -q --short HEAD || echo "(detached)")
    [ "$on" = "$BRANCH" ] || die "$PORT is on $on, not $BRANCH"
    head=$(git -C "$PORT" rev-parse HEAD)
    map_only "$TIP" "$head" || die "$PORT is at $head, which is neither $TIP nor $TIP plus the recording map"
    echo "PORT verified: $PORT at $head on $BRANCH"
else
    [ "$remote_tip" = "$TIP" ] || die "origin/$BRANCH is at $remote_tip, not at the handed-off tip $TIP"
    if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$BRANCH"; then
        die "branch $BRANCH already exists in $MAIN, at $(git -C "$MAIN" rev-parse "refs/heads/$BRANCH")," \
            "but $PORT does not"
    fi
    git -C "$MAIN" worktree add -b "$BRANCH" "$PORT" "origin/$BRANCH"
    [ "$(git -C "$PORT" rev-parse HEAD)" = "$TIP" ] || die "$PORT was created at another commit than $TIP"
    check_tree "$PORT"
    echo "PORT created: $PORT at $TIP on $BRANCH"
fi

if [ -e "$JULY" ]; then
    check_tree "$JULY"
    if git -C "$JULY" symbolic-ref -q HEAD >/dev/null; then
        die "$JULY is on a branch, not detached at $JULY_COMMIT"
    fi
    head=$(git -C "$JULY" rev-parse HEAD)
    [ "$head" = "$JULY_COMMIT" ] || die "$JULY is at $head, not $JULY_COMMIT"
    echo "JULY verified: $JULY detached at $JULY_COMMIT"
else
    git -C "$MAIN" worktree add --detach "$JULY" "$JULY_COMMIT"
    check_tree "$JULY"
    echo "JULY created: $JULY detached at $JULY_COMMIT"
fi

[ "$(git -C "$MAIN" rev-parse HEAD)" = "$main_head" ] || die "the HEAD of $MAIN changed during this run"
[ "$(git -C "$MAIN" stash list | wc -l | tr -d ' ')" = "$main_stashes" ] \
    || die "the stash list of $MAIN changed during this run"
echo
echo "ready: PORT $PORT, JULY $JULY; MAIN unchanged at $main_head on $main_branch"
git -C "$MAIN" worktree list || true
