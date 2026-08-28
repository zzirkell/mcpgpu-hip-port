#!/usr/bin/env python3
import os
import subprocess
import shutil
import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw", type=str, choices=["nvidia", "amd", "all"], default="all")
    return parser.parse_args()

FREQUENCIES = {250: 4000, 500: 2000, 1000: 1000}
KNOTS = [32, 64, 128, 256, 512]

VARIANTS = [
    {"arch": "cuda",    "solver": "hybrid"},
    {"arch": "cuda",    "solver": "pcg"},
    {"arch": "cuda",    "solver": "qdldl"},
    {"arch": "nv_hip",  "solver": "pcg"},
    {"arch": "nv_hip",  "solver": "qdldl"},
    {"arch": "amd_hip", "solver": "pcg"},
    {"arch": "amd_hip", "solver": "qdldl"}
]

def main():
    args = parse_arguments()
    
    active_archs = []
    if args.hw in ["nvidia", "all"]:
        active_archs.extend(["cuda", "nv_hip"])
    if args.hw in ["amd", "all"]:
        active_archs.append("amd_hip")

    script_dir = Path(__file__).resolve().parent
    bash_script = script_dir / "run_backend.sh"
    active_repo = script_dir.parent.parent
    
    run_id = 1
    for variant in VARIANTS:
        if variant["arch"] not in active_archs:
            run_id += 1
            continue

        for hz, budget in FREQUENCIES.items():
            for knot in KNOTS:
                print(f"\n=== Executing Run {run_id} | {variant['arch'].upper()} {variant['solver'].upper()} | {hz}Hz | Knots: {knot} | WS: ON ===")
                
                src_dir = active_repo.parent / "MPCGPU" if variant["arch"] == "cuda" else active_repo / "raw_hip_port"
                tmp_results = src_dir / "tmp" / "results"
                
                if tmp_results.exists():
                    shutil.rmtree(tmp_results)
                tmp_results.mkdir(parents=True, exist_ok=True)
                
                cmd = [
                    str(bash_script), variant["arch"], str(knot), str(budget),
                    "-DUSE_SQP_WORKSPACE=1", variant["solver"], "20", "5e-5"
                ]
                subprocess.run(cmd, cwd=src_dir)
                
                archive_name = f"Run_{run_id}_{variant['arch']}_{variant['solver']}_K{knot}_B{budget}_WS_ON"
                archive_folder = script_dir / "heatmap_data" / archive_name
                archive_folder.mkdir(parents=True, exist_ok=True)
                
                if tmp_results.exists():
                    for csv_file in tmp_results.glob("*.csv"):
                        shutil.copy(csv_file, archive_folder)
                run_id += 1

if __name__ == "__main__":
    main()