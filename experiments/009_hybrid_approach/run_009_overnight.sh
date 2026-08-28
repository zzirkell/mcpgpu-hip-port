#!/bin/bash
set -e

# expects "nvidia" oder "amd" as argument
HW_CHOICE=${1:-"all"}

echo "==============================================="
echo "=== Start Hybrid Approach Final Evaluation  ==="
echo "=== System: $HW_CHOICE"
echo "==============================================="

cd "$HOME/mpcgpu_project/mcpgpu-hip-port/experiments/009_hybrid_approach"

echo ""
echo "--- 1. Generating Heatmap Data ---"
python3 -u run_heatmaps.py --hw "$HW_CHOICE"

echo ""
echo "--- 2. Plotting Heatmaps ---"
python3 plot_heatmaps.py

if [ "$HW_CHOICE" == "nvidia" ] || [ "$HW_CHOICE" == "all" ]; then
    echo ""
    echo "--- 3. Generating Timing Data (Hybrid Component Analysis) ---"
    python3 -u run_timing.py
    
    echo ""
    echo "--- 4. Plotting Gantt Charts ---"
    python3 plot_gantt.py
fi

echo ""
echo "=================================================================="
echo "=== Data Collection & Plotting for $HW_CHOICE finished!        ==="
echo "=================================================================="

# commands to run:

# nohup ./run_009_overnight.sh nvidia > evaluation_nvidia.log 2>&1 &

# nohup ./run_009_overnight.sh amd > evaluation_amd.log 2>&1 &