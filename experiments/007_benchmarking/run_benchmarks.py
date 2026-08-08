#!/usr/bin/env python3
import os
import subprocess
import shutil
from pathlib import Path

# ==========================================
# 1. HARDWARE DETECTION & LOGGING
# ==========================================
def log_amd_gpu_info():
    """Detects active AMD hardware via rocminfo and logs it for the report."""
    print("=== AMD Hardware Validation ===")
    try:
        result = subprocess.run(["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        gpu_name = "Unknown"
        gpu_isa = "Unknown"
        
        for line in result.stdout.splitlines():
            if "Marketing Name:" in line and "Radeon" in line:
                gpu_name = line.split(":", 1)[1].strip()
            elif "Name: " in line and "gfx" in line:
                gpu_isa = line.split(":", 1)[1].strip()
                
        print(f"[+] Found GPU Hardware : {gpu_name}")
        print(f"[+] AMD ISA Architecture : {gpu_isa}")
        
        if "gfx1151" in gpu_isa or "Radeon 8060S" in gpu_name:
            print("[+] STATUS: Correct AMD Strix-Halo APU is active!")
        else:
            print("[!] WARNING: Different GPU detected.")
    except Exception as e:
        print(f"[!] Could not read rocminfo: {e}")
    print("===============================\n")


def detect_hardware():
    """Checks which GPUs are available in the system and enables the corresponding architectures."""
    archs = []
    print("=== Hardware Detection ===")
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("[+] NVIDIA GPU found -> Enabling: 'cuda', 'nv_hip'")
        archs.extend(["cuda", "nv_hip"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    try:
        subprocess.run(["rocm-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("[+] AMD GPU found    -> Enabling: 'amd_hip'")
        archs.append("amd_hip")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
        
    print("==========================\n")
    return archs


# ==========================================
# 2. BENCHMARK ITERATION (Via Bash Script)
# ==========================================
def run_benchmark_iteration(src_dir: Path, bash_script: Path, arch, knots, rate_hz, workspace_off):
    """
    Calls the secure bash script exactly inside the correct source folder.
    """
    max_time_us = int(1000000 / rate_hz)
    workspace_flag = "-DUSE_SQP_WORKSPACE=0" if workspace_off else "-DUSE_SQP_WORKSPACE=1"
    
    try:
        os.chmod(bash_script, 0o755)
    except Exception:
        pass

    cmd = [
        str(bash_script),
        arch,
        str(knots),
        str(max_time_us),
        workspace_flag
    ]
    
    # Correctly set the visibility of the GPUs
    env = os.environ.copy()
    if arch in ["cuda", "nv_hip"]:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["HIP_VISIBLE_DEVICES"] = "0"
    elif arch == "amd_hip":
        env["HIP_VISIBLE_DEVICES"] = "0"

    # Execute compile and run in the specific source directory (Original or HIP port)
    result = subprocess.run(cmd, cwd=src_dir, env=env)
    return result.returncode == 0


# ==========================================
# 3. MAIN LOOP
# ==========================================
def main():
    script_dir = Path(__file__).resolve().parent
    bash_script = script_dir / "run_backend.sh"
    active_repo = script_dir.parent.parent  # Resolves to mcpgpu-hip-port/
    
    knots_list = [32, 64, 128, 256, 512]
    rates_list = [250, 500, 1000]
    workspace_variants = [False, True]
    
    log_amd_gpu_info()
    archs = detect_hardware()
    
    if not archs:
        print("[!] No compatible hardware found. Aborting.")
        return

    for arch in archs:
        # 1. Determine the exact source folder for the architecture
        if arch == "cuda":
            # Points to the original mpcgpu repo (assuming it is parallel to mcpgpu-hip-port)
            src_dir = active_repo.parent / "MPCGPU" 
        else:
            # Points to the raw hip port inside the current repo
            src_dir = active_repo / "raw_hip_port"

        if not src_dir.exists():
            print(f"[!] CRITICAL: Directory {src_dir} not found! Check your paths.")
            continue

        for workspace_off in workspace_variants:
            ws_label = "OFF" if workspace_off else "ON"
            print(f"\n\n{'='*60}")
            print(f"STARTING TEST SUITE: Arch={arch} | Workspace={ws_label}")
            print(f"{'='*60}")
            
            for knots in knots_list:
                for rate_hz in rates_list:
                    print(f"\n[*] RUN | Knots: {knots} | Rate: {rate_hz}Hz")
                    
                    # 2. Clean up tmp_results inside the active src_dir
                    tmp_results = src_dir / "tmp" / "results"
                    if tmp_results.exists():
                        shutil.rmtree(tmp_results)

                    tmp_results.mkdir(parents=True, exist_ok=True)
                    
                    # 3. Delegate to the bash script
                    success = run_benchmark_iteration(src_dir, bash_script, arch, knots, rate_hz, workspace_off)
                    
                    if not success:
                        print(f"    [!] Error (Compile or Runtime) at {knots} knots and {rate_hz}Hz.")
                        print(f"    [!] Skipping remaining rates for {knots} knots.")
                        break
                    
                    # 4. Save results to the archive folder
                    archive_folder = script_dir / f"benchmark_archive_{arch}_ws_{ws_label.lower()}" / f"K{knots}_{rate_hz}Hz"
                    archive_folder.mkdir(parents=True, exist_ok=True)
                    
                    if tmp_results.exists():
                        subprocess.run(["rsync", "-a", f"{tmp_results}/", f"{archive_folder}/"])
                        print(f"    [+] Saved results to: {archive_folder}")
                        
    print("\n=== All benchmarks completed successfully! ===")

if __name__ == "__main__":
    main()