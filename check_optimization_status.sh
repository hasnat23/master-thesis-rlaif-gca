#!/bin/bash
# Real-time monitoring of GCA optimization job

JOB_ID="1335975"
WORK_DIR="/fshpc/muhhas01/thesis_git"

echo "🔍 GCA Optimization Job Monitor"
echo "========================================"
echo "Job ID: $JOB_ID"
echo "Timestamp: $(date)"
echo ""

# Function to check job status
check_status() {
    ssh mogon "cd $WORK_DIR && sacct -j $JOB_ID --format=State,ExitCode --noheader 2>/dev/null" | head -1
}

# Function to get elapsed time
get_runtime() {
    ssh mogon "squeue -j $JOB_ID -h -o '%8M' 2>/dev/null" | head -1 || echo "?"
}

# Function to show recent log lines
show_logs() {
    echo ""
    echo "📋 Recent log output:"
    echo "---------"
    ssh mogon "tail -30 $WORK_DIR/logs/gca_opt_v3_*.log 2>/dev/null" | tail -20
    echo "---------"
}

# Function to check for intermediate files
check_progress() {
    echo ""
    echo "📁 File generation progress:"
    ssh mogon "cd $WORK_DIR && {
        echo -n '  Preferences: '
        if [ -f 'data/preferences_1000_opt_alpha_0.0/holistic_reward_preferences_1000_opt_alpha_0.0.jsonl' ]; then
            echo \"✓ $(wc -l < data/preferences_1000_opt_alpha_0.0/holistic_reward_preferences_1000_opt_alpha_0.0.jsonl) lines\"
        else
            echo '✗ Not yet'
        fi
        echo -n '  RM summary: '
        if [ -f 'outputs/reward_models_1000_opt_alpha_0.0/rm_training_summary.json' ]; then
            echo '✓ Generated'
        else
            echo '✗ Not yet'
        fi
    }" 2>/dev/null
}

# Get status
STATUS=$(check_status)
RUNTIME=$(get_runtime)

echo "Status: $STATUS"
echo "Runtime: $RUNTIME minutes"

# Show progress
check_progress

# Show logs
show_logs

# Final status
case "$STATUS" in
    *"COMPLETED"*)
        echo ""
        echo "✅ Job completed! Running analysis..."
        ssh mogon "cd $WORK_DIR && python3 analysis/analyze_optimization_results.py 2>&1 | tail -80"
        ;;
    *"FAILED"*)
        echo ""
        echo "❌ Job failed! Checking errors..."
        ssh mogon "cat $WORK_DIR/logs/gca_opt_v3_*.err 2>/dev/null"
        ;;
    *"RUNNING"*)
        echo ""
        echo "⏳ Job still running... check back in ~10 minutes"
        echo ""
        echo "To re-run this monitor:"
        echo "  bash monitor_optimization.sh"
        ;;
    *)
        echo ""
        echo "? Unknown status: $STATUS"
        ;;
esac

echo ""
echo "========================================"
echo "Monitor end: $(date)"
