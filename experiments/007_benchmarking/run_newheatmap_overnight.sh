#!/bin/bash
set -e

echo "=== Start New Heatmap Benchmarks ==="

cd "$HOME/mpcgpu_project/mcpgpu-hip-port/experiments/007_benchmarking"

python3 -u run_benchmarks_newheatmap.py --hw "nvidia"

echo "=== Benchmarks completed! ==="

#  nohup ./run_newheatmap_overnight.sh > newheatmap_output.log 2>&1 &

# tail -f newheatmap_output.log