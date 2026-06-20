#!/bin/bash
# GCA Optimization Monitoring Script
# Checks job status, analyzes results when ready, and triggers next phases

set -e

WORK_DIR="/fshpc/muhhas01/thesis_git"
JOB_ID="1335973"

echo "=========================================="
echo "GCA Optimization Monitor"
echo "=========================================="
echo "Time: $(date)"
echo "Working directory: $WORK_DIR"
echo "Main job ID: $JOB_ID"
echo ""

# Check job status
echo "1. Checking job status..."
JOB_STATUS=$(ssh mogon "sacct -j $JOB_ID --format=State --noheader" 2>/dev/null | head -1 | xargs || echo "UNKNOWN")
echo "   Status: $JOB_STATUS"

case "$JOB_STATUS" in
    "RUNNING")
        echo "   ✓ Job is running..."
        RUNTIME=$(ssh mogon "squeue -j $JOB_ID -o '%8M' --noheader" 2>/dev/null | head -1 || echo "?")
        echo "   Runtime: $RUNTIME"
        echo ""
        echo "   Next update in ~30 minutes. Checking logs..."
        ssh mogon "tail -50 $WORK_DIR/logs/gca_opt_v2_*.log 2>/dev/null | head -30" || true
        ;;
    "COMPLETED")
        echo "   ✓ Job completed!"
        echo ""
        echo "2. Analyzing results..."
        ssh mogon "cd $WORK_DIR && python3 analysis/analyze_optimization_results.py" || {
            echo "   ERROR: Result analysis failed"
            echo "   Checking for error logs..."
            ssh mogon "cat $WORK_DIR/logs/gca_opt_v2_*.err" || true
        }
        ;;
    "FAILED")
        echo "   ✗ Job failed!"
        echo ""
        echo "2. Checking error logs..."
        ssh mogon "cat $WORK_DIR/logs/gca_opt_v2_*.err 2>/dev/null | tail -50" || true
        echo ""
        echo "3. Resubmitting with diagnostics..."
        ssh mogon "cd $WORK_DIR && sbatch slurm/submit_gca_opt_v2.sh" || true
        ;;
    *)
        echo "   ? Status unknown: $JOB_STATUS"
        ;;
esac

echo ""
echo "=========================================="
echo "Monitor completed: $(date)"
echo "=========================================="
