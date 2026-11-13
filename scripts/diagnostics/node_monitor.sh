#!/bin/bash
# Record what this node is doing every 5 seconds until killed, for reading a hang afterwards.
#
#     bash node_monitor.sh OUT_DIR
#
# Writes OUT_DIR/node_once.txt (CPU layout, transparent huge pages, swap, the node's and the
# job's SLURM records, the node's other jobs) and appends to OUT_DIR/node_monitor.log, every 5 s:
# vmstat (steal, swap in and out), /proc/vmstat counters for major faults, swapping, reclaim
# stalls and compaction, the job's cgroup memory and CPU counters (cgroup v2 or v1), for every
# Python thread of this user its run and run-queue wait times, state and wait channel, each
# Python process's resident and swapped memory, and the established loopback TCP connections
# with their queues.

set -u
OUT=${1:?usage: node_monitor.sh OUT_DIR}
mkdir -p "$OUT"
NODE=$(hostname -s)

{
    echo "date: $(date -Is)"; echo "node: $NODE"
    lscpu
    echo "thp enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>&1)"
    echo "thp defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag 2>&1)"
    swapon --show 2>&1
    scontrol show node "$NODE" 2>&1
    [ -n "${SLURM_JOB_ID:-}" ] && scontrol show job "$SLURM_JOB_ID" 2>&1
    squeue -w "$NODE" -o '%i %u %j %C %m %T' 2>&1
} > "$OUT/node_once.txt"

# The cgroup this job runs in: one line "0::path" under v2, a "memory" line under v1
CG2=$(awk -F: '$1 == "0" && $2 == "" {print $3}' /proc/self/cgroup)
CG1_MEM=$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3}' /proc/self/cgroup)

while true; do
    {
        echo "=== $(date -Is)"
        vmstat 1 2 | tail -n 1
        grep -E '^(pgmajfault|pswpin|pswpout|allocstall_normal|allocstall_movable|compact_stall|compact_fail|thp_fault_alloc|thp_fault_fallback) ' \
            /proc/vmstat | tr '\n' ' '; echo
        if [ -n "$CG2" ] && [ -r "/sys/fs/cgroup$CG2/memory.current" ]; then
            D=/sys/fs/cgroup$CG2
            echo "cgroup memory.current $(cat "$D/memory.current")"
            echo "cgroup memory.events $(tr '\n' ' ' < "$D/memory.events")"
            echo "cgroup memory.stat $(grep -E '^(anon|file|kernel|sock|shmem|pgmajfault) ' "$D/memory.stat" | tr '\n' ' ')"
            echo "cgroup cpu.stat $(tr '\n' ' ' < "$D/cpu.stat")"
        elif [ -n "$CG1_MEM" ] && [ -r "/sys/fs/cgroup/memory$CG1_MEM/memory.usage_in_bytes" ]; then
            D=/sys/fs/cgroup/memory$CG1_MEM
            echo "cgroup memory.usage_in_bytes $(cat "$D/memory.usage_in_bytes") failcnt $(cat "$D/memory.failcnt")"
            echo "cgroup memory.stat $(grep -E '^(total_rss|total_cache|total_swap|total_pgmajfault) ' "$D/memory.stat" | tr '\n' ' ')"
        fi
        for pid in $(pgrep -u "$(id -u)" -f python); do
            for task in /proc/"$pid"/task/*; do
                read -r run_ns wait_ns _ < "$task/schedstat" 2>/dev/null || continue
                echo "thread $pid/${task##*/} $(cat "$task/comm" 2>/dev/null) run_ns=$run_ns wait_ns=$wait_ns" \
                    "state=$(awk '/^State/ {print $2}' "$task/status" 2>/dev/null) wchan=$(cat "$task/wchan" 2>/dev/null)"
            done
            echo "process $pid $(grep -E '^(VmRSS|VmSwap)' /proc/"$pid"/status 2>/dev/null | tr -s ' \t\n' ' ')"
        done
        ss -tni state established '( src 127.0.0.1 or dst 127.0.0.1 )' 2>/dev/null | head -n 60
    } >> "$OUT/node_monitor.log" 2>&1
    sleep 5
done
