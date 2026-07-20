#!/bin/bash

echo "=== Start Figure 6 Benchmark ==="
# Führt das Benchmark-Skript aus
#cd mpcgpu_project/mcpgpu-hip-port/experiments/007_benchmarking
python -u run_benchmarks.py --hw nvidia

echo "=== Benchmarks completed! ==="
echo "=== Generate Plots ==="

python -u run_figure6.py

echo "=== Completed ==="

# chmod +x run_figure6_overnight.sh

# nohup ./run_figure6_overnight.sh > benchmark_figure6_output.log 2>&1 &