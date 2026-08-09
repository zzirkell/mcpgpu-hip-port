#!/usr/bin/env python3
import os
import subprocess
import shutil
import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fine-grained timing analysis.")
    parser.add_argument("--hw", type=str, choices=["nvidia", "amd", "all"], default="all")
    return parser.parse_args()

def detect_hardware(hw_choice):
    # On clusters, nvidia-smi/rocm-smi can be unavailable or restricted.
    # If the user explicitly selects a backend family, trust the Slurm allocation.
    if hw_choice == "nvidia":
        print("[+] Forced NVIDIA mode -> Enabled: 'cuda', 'nv_hip'")
        return ["cuda", "nv_hip"]

    if hw_choice == "amd":
        print("[+] Forced AMD mode -> Enabled: 'amd_hip'")
        return ["amd_hip"]

    archs = []

    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        archs.extend(["cuda", "nv_hip"])
        print("[+] NVIDIA GPU found -> Enabled: 'cuda', 'nv_hip'")
    except Exception:
        pass

    try:
        subprocess.run(["rocm-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        archs.append("amd_hip")
        print("[+] AMD GPU found -> Enabled: 'amd_hip'")
    except Exception:
        pass

    return archs

# Erweiterte Testreihe: Inklusive NV_HIP und Workspace ON/OFF
TIMING_RUNS = [
    # --- CUDA ---
    {"run": 1, "arch": "cuda",    "solver": "pcg", "ws_off": False, "knots": 32,  "budget": 50000, "iters": 20},
    {"run": 2, "arch": "cuda",    "solver": "pcg", "ws_off": False, "knots": 64, "budget": 50000, "iters": 20},
    {"run": 3, "arch": "cuda",    "solver": "pcg", "ws_off": False, "knots": 128, "budget": 50000, "iters": 20},
    {"run": 4, "arch": "cuda",    "solver": "qdldl", "ws_off": False, "knots": 32,  "budget": 50000, "iters": 20},
    {"run": 5, "arch": "cuda",    "solver": "qdldl", "ws_off": False, "knots": 64, "budget": 50000, "iters": 20},
    {"run": 6, "arch": "cuda",    "solver": "qdldl", "ws_off": False, "knots": 128, "budget": 50000, "iters": 20},
    
    # --- NV_HIP  ---
    {"run": 7, "arch": "nv_hip",    "solver": "pcg", "ws_off": False, "knots": 32,  "budget": 50000, "iters": 20},
    {"run": 8, "arch": "nv_hip",    "solver": "pcg", "ws_off": False, "knots": 64, "budget": 50000, "iters": 20},
    {"run": 9, "arch": "nv_hip",    "solver": "pcg", "ws_off": False, "knots": 128, "budget": 50000, "iters": 20},
    {"run": 10, "arch": "nv_hip",    "solver": "qdldl", "ws_off": False, "knots": 32,  "budget": 50000, "iters": 20},
    {"run": 11, "arch": "nv_hip",    "solver": "qdldl", "ws_off": False, "knots": 64, "budget": 50000, "iters": 20},
    {"run": 12, "arch": "nv_hip",    "solver": "qdldl", "ws_off": False, "knots": 128, "budget": 50000, "iters": 20},

    # --- AMD_HIP  ---
    {"run": 13, "arch": "amd_hip",    "solver": "pcg", "ws_off": False, "knots": 32,  "budget": 50000, "iters": 20},
    {"run": 14, "arch": "amd_hip",    "solver": "pcg", "ws_off": False, "knots": 64, "budget": 50000, "iters": 20},
    {"run": 15, "arch": "amd_hip",    "solver": "pcg", "ws_off": False, "knots": 128, "budget": 50000, "iters": 20},
    {"run": 16, "arch": "amd_hip",    "solver": "qdldl", "ws_off": False, "knots": 32,  "budget": 50000, "iters": 20},
    {"run": 17, "arch": "amd_hip",    "solver": "qdldl", "ws_off": False, "knots": 64, "budget": 50000, "iters": 20},
    {"run": 18, "arch": "amd_hip",    "solver": "qdldl", "ws_off": False, "knots": 128, "budget": 50000, "iters": 20},
]

def run_timing_iteration(src_dir, bash_script, config):
    # Wenn ws_off == True, setze USE_SQP_WORKSPACE=0 (OFF), ansonsten 1 (ON)
    workspace_flag = "-DUSE_SQP_WORKSPACE=0" if config["ws_off"] else "-DUSE_SQP_WORKSPACE=1"
    
    cmd = [
        str(bash_script),
        config["arch"],
        str(config["knots"]),
        str(config["budget"]),
        workspace_flag,
        config["solver"],
        str(config["iters"]),
        "5e-5"
    ]
    
    env = os.environ.copy()
    if config["arch"] in ["cuda", "nv_hip"]:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["HIP_VISIBLE_DEVICES"] = "0"
    elif config["arch"] == "amd_hip":
        env["HIP_VISIBLE_DEVICES"] = "0"

    result = subprocess.run(cmd, cwd=src_dir, env=env)
    return result.returncode == 0

def main():
    args = parse_arguments()
    script_dir = Path(__file__).resolve().parent
    
    bash_script = script_dir / "run_backend_timing.sh"
    active_repo = script_dir.parent.parent
    
    print("=== Hardware Detection ===")
    available_archs = detect_hardware(args.hw)
    if not available_archs:
        print(f"[!] No compatible hardware found.")
        return
    print("==========================\n")

    for config in TIMING_RUNS:
        if config["arch"] not in available_archs:
            continue
            
        ws_label = "OFF" if config["ws_off"] else "ON"
        print(f"\n{'='*60}")
        print(f"Executing TIMING Run {config['run']} | Arch: {config['arch']} | Knots: {config['knots']} | Workspace: {ws_label}")
        print(f"{'='*60}")
        
        if config["arch"] == "cuda":
            src_dir = active_repo.parent / "MPCGPU" 
        else:
            src_dir = active_repo / "raw_hip_port"
            
        tmp_results = src_dir / "tmp" / "results"
        if tmp_results.exists():
            shutil.rmtree(tmp_results)
        tmp_results.mkdir(parents=True, exist_ok=True)
        
        success = run_timing_iteration(src_dir, bash_script, config)
        
        if not success:
            print(f"[!] Timing Run {config['run']} failed.")
            continue
            
        # Archivieren: Erzeugt klare Ordnernamen, z.B. Timing_nv_hip_K64_WS_ON
        archive_name = f"Timing_{config['arch']}_{config['solver']}_K{config['knots']}"
        archive_folder = script_dir / "timing_results" / archive_name
        archive_folder.mkdir(parents=True, exist_ok=True)
        
        if tmp_results.exists():
            files_copied = 0
            # Sucht gezielt nach .result (und sicherheitshalber nach .csv)
            for file_path in tmp_results.glob("*.*"):
                if file_path.suffix in ['.csv', '.result']:
                    shutil.copy(file_path, archive_folder)
                    files_copied += 1
                
            print(f"[+] Successfully copied {files_copied} files to: {archive_folder.name}")

if __name__ == "__main__":
    main()