#!/bin/bash

echo "=== Start Figure 6 Benchmark ==="
# Führt das Benchmark-Skript aus
#cd mpcgpu_project/mcpgpu-hip-port/experiments/007_benchmarking
python -u run_benchmarks.py --hw amd

echo "=== Benchmarks completed! ==="
echo "=== Generate Plots ==="

python -u run_figure6.py 

echo "=== Completed!!! ==="

# chmod +x run_figure6_overnight_amd.sh

# nohup ./run_figure6_overnight_amd.sh > benchmark_figure6_output_amd.log 2>&1 &