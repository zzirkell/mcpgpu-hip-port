#!/bin/bash
set -e

echo "=== Start Fine-Grained Timing Analysis ==="

# Wechsle in den richtigen Benchmark-Ordner
cd "$HOME/mpcgpu_project/mcpgpu-hip-port/experiments/008_component_analysis"

# Starte das neue Python-Skript (-u sorgt dafür, dass nohup den Log sofort schreibt)
python3 -u run_timing_analysis.py --hw "nvidia"

echo "=== Timing Analysis completed! ==="

# chmod +x run_figure6_overnight.sh

# nohup ./run_figure6_overnight.sh > benchmark_figure6_output.log 2>&1 &
# tail -f benchmark_figure6_output.log

# for restarting: 

# pkill -9 -f run_figure6_overnight.sh
# pkill -9 -f run_benchmarks.py
# killall -9 pcg.exe qdldl.exe make
# rm benchmark_figure6_output.log