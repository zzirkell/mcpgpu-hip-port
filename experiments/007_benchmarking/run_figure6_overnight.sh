#!/bin/bash
set -e # Abort the script if an error occurs

echo "=== Start Figure 6 Benchmark ==="

# 1. Navigate to the benchmarking directory
cd "$HOME/mpcgpu_project/mcpgpu-hip-port/experiments/007_benchmarking"

# 2. Run the benchmarking script (Hardware detection is automatic now)
python3 -u run_benchmarks.py

echo "=== Benchmarks completed! ==="
echo "=== Generate Plots ==="

# 3. Generate the heatmaps
python3 -u run_figure6.py

echo "=== Completed!!! ==="

# chmod +x run_figure6_overnight.sh

# nohup ./run_figure6_overnight.sh > benchmark_figure6_output.log 2>&1 &
# tail -f benchmark_figure6_output.log

# for restarting: 

# pkill -9 -f run_figure6_overnight.sh
# pkill -9 -f run_benchmarks.py
# killall -9 pcg.exe qdldl.exe make
# rm benchmark_figure6_output.log

