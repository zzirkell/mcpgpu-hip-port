#!/usr/bin/env python3
import os
import subprocess
import shutil
import argparse
from pathlib import Path

# ==========================================
# 1. HARDWARE DETECTION & CLI ARGUMENTS
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run benchmarks based on new heatmap plan.")
    parser.add_argument("--hw", type=str, choices=["nvidia", "amd", "all"], default="all",
                        help="Select which hardware architectures to benchmark.")
    return parser.parse_args()

def detect_hardware(hw_choice):
    archs = []
    
    if hw_choice in ["nvidia", "all"]:
        try:
            subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            archs.extend(["cuda", "nv_hip"])
            print("[+] NVIDIA GPU found -> Enabled: 'cuda', 'nv_hip'")
        except: pass
        
    if hw_choice in ["amd", "all"]:
        try:
            subprocess.run(["rocm-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            archs.append("amd_hip")
            print("[+] AMD GPU found -> Enabled: 'amd_hip'")
        except: pass
        
    return archs

# ==========================================
# 2. RUN CONFIGURATIONS
# ==========================================
RUNS = [
    # --- PCG Baseline (Runs 1-8) | iters: 20, tol: 5e-5 ---
    {"run": 1, "arch": "cuda",    "solver": "pcg", "ws": False, "knots": 64,  "budget": 5000, "iters": 20, "tol": "5e-5", "group": "baseline"},
    {"run": 2, "arch": "cuda",    "solver": "pcg", "ws": False, "knots": 128, "budget": 5000, "iters": 20, "tol": "5e-5", "group": "baseline"},
    {"run": 3, "arch": "nv_hip",  "solver": "pcg", "ws": False, "knots": 64,  "budget": 5000, "iters": 20, "tol": "5e-5", "group": "baseline"},
    {"run": 4, "arch": "nv_hip",  "solver": "pcg", "ws": False, "knots": 128, "budget": 5000, "iters": 20, "tol": "5e-5", "group": "baseline"},
    {"run": 5, "arch": "amd_hip", "solver": "pcg", "ws": False, "knots": 64,  "budget": 5000, "iters": 20, "tol": "5e-5", "group": "baseline"},
    {"run": 6, "arch": "amd_hip", "solver": "pcg", "ws": False, "knots": 128, "budget": 5000, "iters": 20, "tol": "5e-5", "group": "baseline"},
    {"run": 7, "arch": "amd_hip", "solver": "pcg", "ws": True,  "knots": 64,  "budget": 5000, "iters": 20, "tol": "5e-5", "group": "baseline"},
    {"run": 8, "arch": "amd_hip", "solver": "pcg", "ws": True,  "knots": 128, "budget": 5000, "iters": 20, "tol": "5e-5", "group": "baseline"},

    # --- QDLDL Sanity Check (Runs 9-14) | iters: 20, tol: 5e-5 ---
    {"run": 9,  "arch": "nv_hip",  "solver": "qdldl", "ws": False, "knots": 128, "budget": 5000,  "iters": 20, "tol": "5e-5", "group": "sanity_qdldl"},
    {"run": 10, "arch": "nv_hip",  "solver": "qdldl", "ws": False, "knots": 128, "budget": 10000, "iters": 20, "tol": "5e-5", "group": "sanity_qdldl"},
    {"run": 11, "arch": "nv_hip",  "solver": "qdldl", "ws": False, "knots": 128, "budget": 50000, "iters": 20, "tol": "5e-5", "group": "sanity_qdldl"},
    {"run": 12, "arch": "amd_hip", "solver": "qdldl", "ws": False, "knots": 128, "budget": 5000,  "iters": 20, "tol": "5e-5", "group": "sanity_qdldl"},
    {"run": 13, "arch": "amd_hip", "solver": "qdldl", "ws": False, "knots": 128, "budget": 10000, "iters": 20, "tol": "5e-5", "group": "sanity_qdldl"},
    {"run": 14, "arch": "amd_hip", "solver": "qdldl", "ws": False, "knots": 128, "budget": 50000, "iters": 20, "tol": "5e-5", "group": "sanity_qdldl"},

    # --- Stress Tests (Runs 15-20) ---
    # 128 Knots (Tighter PCG Tolerance: 1e-5)
    {"run": 15, "arch": "nv_hip",  "solver": "pcg", "ws": False, "knots": 128, "budget": 5000, "iters": 20, "tol": "1e-5", "group": "stress_128"},
    {"run": 16, "arch": "amd_hip", "solver": "pcg", "ws": False, "knots": 128, "budget": 5000, "iters": 20, "tol": "1e-5", "group": "stress_128"},
    {"run": 17, "arch": "amd_hip", "solver": "pcg", "ws": True,  "knots": 128, "budget": 5000, "iters": 20, "tol": "1e-5", "group": "stress_128"},
    # 256 Knots (Reduced iterations: 10, Standard Tolerance: 5e-5)
    {"run": 18, "arch": "nv_hip",  "solver": "pcg", "ws": False, "knots": 256, "budget": 5000, "iters": 10, "tol": "5e-5", "group": "stress_256"},
    {"run": 19, "arch": "amd_hip", "solver": "pcg", "ws": False, "knots": 256, "budget": 5000, "iters": 10, "tol": "5e-5", "group": "stress_256"},
    {"run": 20, "arch": "amd_hip", "solver": "pcg", "ws": True,  "knots": 256, "budget": 5000, "iters": 10, "tol": "5e-5", "group": "stress_256"},
]

def run_benchmark_iteration(src_dir, bash_script, config):
    workspace_flag = "-DUSE_SQP_WORKSPACE=0" if config["ws"] else "-DUSE_SQP_WORKSPACE=1"
    
    cmd = [
        str(bash_script),
        config["arch"],
        str(config["knots"]),
        str(config["budget"]),
        workspace_flag,
        config["solver"],
        str(config["iters"]),
        config["tol"]
    ]
    
    env = os.environ.copy()
    if config["arch"] in ["cuda", "nv_hip"]:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["HIP_VISIBLE_DEVICES"] = "0"
    elif config["arch"] == "amd_hip":
        env["HIP_VISIBLE_DEVICES"] = "0"

    result = subprocess.run(cmd, cwd=src_dir, env=env)
    return result.returncode == 0

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    args = parse_arguments()
    
    script_dir = Path(__file__).resolve().parent
    bash_script = script_dir / "run_backend.sh"
    active_repo = script_dir.parent.parent
    
    print("=== Hardware Detection ===")
    available_archs = detect_hardware(args.hw)
    if not available_archs:
        print(f"[!] No compatible hardware found for choice '{args.hw}'.")
        return
    print("==========================\n")

    for config in RUNS:
        if config["arch"] not in available_archs:
            continue
            
        print(f"\n{'='*60}")
        print(f"Executing Run {config['run']} ({config['group'].upper()})")
        print(f"Arch: {config['arch']} | Solver: {config['solver']} | Knots: {config['knots']} | Budget: {config['budget']}us")
        print(f"WS_OFF: {config['ws']} | Iters: {config['iters']} | Tolerance: {config['tol']}")
        print(f"{'='*60}")
        
        if config["arch"] == "cuda":
            src_dir = active_repo.parent / "MPCGPU" 
        else:
            src_dir = active_repo / "raw_hip_port"
            
        tmp_results = src_dir / "tmp" / "results"
        if tmp_results.exists():
            shutil.rmtree(tmp_results)
        tmp_results.mkdir(parents=True, exist_ok=True)
        
        success = run_benchmark_iteration(src_dir, bash_script, config)
        
        if not success:
            print(f"[!] Run {config['run']} failed.")
            continue
            
        # Ergebnisse archivieren (NUR .csv Dateien)
        ws_label = "OFF" if config["ws"] else "ON"
        archive_name = f"Run_{config['run']:02d}_{config['arch']}_{config['solver']}_K{config['knots']}_B{config['budget']}_WS{ws_label}"
        archive_folder = script_dir / f"newheatmap_results_{config['group']}" / archive_name
        archive_folder.mkdir(parents=True, exist_ok=True)
        
        if tmp_results.exists():
            # Sucht gezielt nach allen .csv Dateien im tmp/results Ordner
            csv_files_copied = 0
            for csv_file in tmp_results.glob("*.csv"):
                shutil.copy(csv_file, archive_folder)
                csv_files_copied += 1
                
            print(f"[+] Successfully copied {csv_files_copied} .csv files to: {archive_folder.name}")

if __name__ == "__main__":
    main()