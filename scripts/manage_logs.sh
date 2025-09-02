#!/bin/bash
# SLURM Log Management Script
# ===========================
# 
# This script helps manage SLURM and Snakemake log files following best practices:
# - Organizes logs by date
# - Maps job IDs to animal names
# - Automatically cleans up old logs
# - Provides utilities for log analysis

set -euo pipefail

LOG_DIR="logs/slurm"
PIPELINE_LOGS="logs/war_generation"
RETENTION_DAYS=${1:-10}  # Default 10 days retention

# Function to clean up old logs
cleanup_old_logs() {
    echo "🧹 Cleaning up SLURM logs older than $RETENTION_DAYS days..."
    find "$LOG_DIR" -name "*.out" -o -name "*.err" | while read -r logfile; do
        if [ -f "$logfile" ] && [ $(($(date +%s) - $(stat -c %Y "$logfile"))) -gt $((RETENTION_DAYS * 86400)) ]; then
            echo "Removing old log: $logfile"
            rm -f "$logfile"
        fi
    done
    
    # Remove empty directories
    find "$LOG_DIR" -type d -empty -delete 2>/dev/null || true
    
    echo "🧹 Cleaning up pipeline logs older than $RETENTION_DAYS days..."
    find "$PIPELINE_LOGS" -name "*.log" | while read -r logfile; do
        if [ -f "$logfile" ] && [ $(($(date +%s) - $(stat -c %Y "$logfile"))) -gt $((RETENTION_DAYS * 86400)) ]; then
            echo "Removing old pipeline log: $logfile"
            rm -f "$logfile"
        fi
    done
}

# Function to show active jobs and their logs
show_active_jobs() {
    echo "🔍 Active SLURM jobs and their log files:"
    squeue -u $USER -o "%.8i %.10P %.20j %.8u %.2t %.10M %.6D %R" | while IFS= read -r line; do
        if [[ "$line" == *"JOBID"* ]]; then
            echo "$line"
            continue
        fi
        
        job_id=$(echo "$line" | awk '{print $1}')
        if [[ "$job_id" =~ ^[0-9]+$ ]]; then
            # Find corresponding log files
            log_files=$(find "$LOG_DIR" -name "*${job_id}*" 2>/dev/null || true)
            if [ -n "$log_files" ]; then
                echo "$line"
                echo "  📄 Logs: $log_files"
            else
                echo "$line"
                echo "  📄 Logs: Not found yet"
            fi
        fi
    done
}

# Function to find logs for a specific animal
find_animal_logs() {
    local animal_pattern="$1"
    echo "🔍 Searching for logs containing: $animal_pattern"
    
    echo "SLURM job logs:"
    find "$LOG_DIR" -name "*.out" -o -name "*.err" | xargs grep -l "$animal_pattern" 2>/dev/null | while read -r logfile; do
        echo "  📄 $logfile"
        if [[ "$logfile" == *.err ]]; then
            echo "    ERRORS:"
            tail -10 "$logfile" | sed 's/^/      /'
        else
            echo "    OUTPUT:"
            tail -5 "$logfile" | sed 's/^/      /'
        fi
        echo
    done
    
    echo "Snakemake rule logs:"  
    find logs/war_generation -name "*.log" 2>/dev/null | grep -i "$animal_pattern" | while read -r logfile; do
        echo "  📄 $logfile"
        echo "    CONTENTS:"
        tail -15 "$logfile" | sed 's/^/      /' || echo "      (empty or unreadable)"
        echo
    done
}

# Function to list recent runs
list_runs() {
    echo "📋 Recent Snakemake runs:"
    if [ -d "$LOG_DIR" ]; then
        # List run directories sorted by timestamp
        find "$LOG_DIR" -maxdepth 1 -type d -name "*_*" | sort | while read -r run_dir; do
            run_name=$(basename "$run_dir")
            job_count=$(find "$run_dir" -name "job_*.out" 2>/dev/null | wc -l)
            error_count=$(find "$run_dir" -name "job_*.err" -size +0 2>/dev/null | wc -l)
            echo "  📁 $run_name (Jobs: $job_count, Errors: $error_count)"
        done
    else
        echo "  No runs found in $LOG_DIR"
    fi
}

# Function to tail recent logs
tail_recent_logs() {
    echo "📊 Most recent SLURM logs:"
    find "$LOG_DIR" -name "*.err" -printf "%T@ %p\n" | sort -n | tail -5 | cut -d' ' -f2- | while read -r logfile; do
        echo "=== $logfile ==="
        tail -10 "$logfile" 2>/dev/null || echo "Cannot read $logfile"
        echo
    done
}

# Function to show failed jobs with details
show_failed_jobs() {
    echo "❌ Failed SLURM jobs analysis:"
    
    # Get recent failed jobs from sacct
    echo "Recent failed jobs:"
    sacct --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS,AveRSS --state=FAILED --starttime=$(date -d '1 day ago' +%Y-%m-%d) | head -20
    
    echo
    echo "Error patterns in recent logs:"
    find "$LOG_DIR" -name "*.err" -mtime -1 | while read -r logfile; do
        if [ -s "$logfile" ]; then  # Only non-empty error files
            job_id=$(basename "$logfile" .err | sed 's/job_//')
            echo "Job $job_id errors:"
            # Show unique error patterns
            grep -E "(Error|Exception|Failed|Traceback)" "$logfile" 2>/dev/null | head -3 || echo "  No clear error patterns found"
        fi
    done
}

# Function to get job statistics
show_job_stats() {
    echo "📈 Job statistics from last 24 hours:"
    
    start_time=$(date -d '1 day ago' +%Y-%m-%d)
    
    echo "Job state summary:"
    sacct --format=State --noheader --starttime="$start_time" | sort | uniq -c | sort -nr
    
    echo
    echo "Resource utilization (top 10 memory users):"
    sacct --format=JobID,JobName,MaxRSS,Elapsed,State --starttime="$start_time" | sort -k3 -hr | head -10
}

# Function to debug specific job
debug_job() {
    local job_id="$1"
    echo "🔍 Debugging job $job_id:"
    
    # Get job details from sacct
    echo "Job information:"
    sacct --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS,ReqMem,ReqCPUS,NodeList -j "$job_id"
    
    echo
    echo "Job output files:"
    find "$LOG_DIR" -name "*${job_id}*" | while read -r logfile; do
        echo "📄 $logfile:"
        if [[ "$logfile" == *.err ]]; then
            echo "  ERROR LOG:"
            tail -20 "$logfile" 2>/dev/null | sed 's/^/    /' || echo "    Cannot read error log"
        else
            echo "  OUTPUT LOG:"
            tail -10 "$logfile" 2>/dev/null | sed 's/^/    /' || echo "    Cannot read output log"
        fi
        echo
    done
}

# Main command handling
case "${1:-help}" in
    "cleanup")
        cleanup_old_logs
        ;;
    "active")
        show_active_jobs
        ;;
    "find")
        if [ $# -lt 2 ]; then
            echo "Usage: $0 find <animal_pattern>"
            exit 1
        fi
        find_animal_logs "$2"
        ;;
    "tail")
        tail_recent_logs
        ;;
    "runs")
        list_runs
        ;;
    "failed")
        show_failed_jobs
        ;;
    "stats")
        show_job_stats
        ;;
    "debug")
        if [ $# -lt 2 ]; then
            echo "Usage: $0 debug <job_id>"
            exit 1
        fi
        debug_job "$2"
        ;;
    "help"|*)
        echo "SLURM Log Management Script"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  cleanup [days]     Clean up logs older than [days] (default: 10)"
        echo "  active            Show active jobs and their log files" 
        echo "  find <pattern>    Find logs containing animal pattern"
        echo "  tail              Show tail of most recent error logs"
        echo "  runs              List recent Snakemake runs with job counts"
        echo "  failed            Show failed jobs with error analysis"
        echo "  stats             Show job statistics from last 24 hours"
        echo "  debug <job_id>    Debug specific job with detailed info"
        echo "  help              Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 cleanup 7                    # Clean logs older than 7 days"
        echo "  $0 find \"cage3A\"                # Find logs for cage3A"
        echo "  $0 active                       # Show active jobs"
        echo "  $0 tail                         # Show recent error logs"
        echo "  $0 runs                         # List recent runs"
        echo "  $0 failed                       # Analyze failed jobs"
        echo "  $0 stats                        # Show job statistics"
        echo "  $0 debug 12345                  # Debug job 12345"
        ;;
esac