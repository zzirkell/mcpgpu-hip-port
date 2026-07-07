import subprocess
import os
import shutil
import itertools
import argparse


def detect_hardware():
    archs = []
    print("=== Hardware Detection ===")
    
    # Check NVIDIA
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("[+] NVIDIA GPU found -> Enabling: 'cuda', 'nv_hip'")
        archs.extend(["cuda", "nv_hip"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Check AMD
    try:
        subprocess.run(["rocm-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("[+] AMD GPU found    -> Enabling: 'amd_hip'")
        archs.append("amd_hip")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
        
    print("==========================\n")
    return archs


def compile_project(active_repo, knots, rate_hz, hide_warnings):
    """Compiles the project with injected parameters."""
    max_time_us = int(1000000 / rate_hz)
    compiler_flags = (
        f"-DKNOT_POINTS={knots} "
        f"-DSQP_MAX_TIME_US={max_time_us} "
        f"-DSAVE_DATA=1 "
        f"-DTEST_ITERS=8" 
    )

    if hide_warnings:
        compiler_flags += " -w "
    
    env = os.environ.copy()
    # Inject for both C and C++ Makefiles
    env["CFLAGS"] = f"-O3 {compiler_flags}" 
    env["CXXFLAGS"] = f"-O3 {compiler_flags}" 
    
    subprocess.run(["make", "clean"], stdout=subprocess.DEVNULL, cwd=active_repo, env=env)
    result = subprocess.run(["make", "examples"], stdout=subprocess.DEVNULL, cwd=active_repo, env=env)
    
    return result.returncode == 0

def run_solver(active_repo, arch, solver_name):
    """Executes the solver with correct library paths and GPU masking."""
    os.makedirs(os.path.join(active_repo, "tmp", "results"), exist_ok=True)
    exe_path = f"./examples/{solver_name.lower()}.exe"
    env = os.environ.copy()
    
    # Set library paths
    path_hip = f"{active_repo}/qdldl/temp_build/out"
    path_cuda = f"{active_repo}/qdldl/build/out"
    env["LD_LIBRARY_PATH"] = f"{path_hip}:{path_cuda}:{env.get('LD_LIBRARY_PATH', '')}"
    
    # Isolate GPUs
    if arch in ["cuda", "nv_hip"]:
        env["CUDA_VISIBLE_DEVICES"] = "0" 
        env["HIP_VISIBLE_DEVICES"] = "0"
    elif arch == "amd_hip":
        env["HIP_VISIBLE_DEVICES"] = "0" 

    subprocess.run([exe_path], stdout=subprocess.DEVNULL, cwd=active_repo, env=env)

def archive_results(active_repo, archive_base, arch, solver_name, knots, rate_hz):
    """Moves result logs to the archive folder."""
    tmp_dir = os.path.join(active_repo, "tmp", "results")
    archive_dir = os.path.join(archive_base, arch, solver_name, f"knots_{knots}_rate_{rate_hz}")
    
    if os.path.exists(tmp_dir) and os.listdir(tmp_dir):
        shutil.copytree(tmp_dir, archive_dir, dirs_exist_ok=True)
        print(f"       [+] Data saved to: {archive_dir}")
        shutil.rmtree(tmp_dir)
    else:
        print("       [!] No log files found!")


def main():
    # CLI Setup
    parser = argparse.ArgumentParser(description="MPCGPU Benchmarking Tool")
    parser.add_argument("--hw", choices=["auto", "nvidia", "amd"], default="auto", help="Force hardware target.")
    parser.add_argument("--solvers", nargs="+", default=["PCG", "QDLDL"], help="Target solvers.")
    parser.add_argument("--knots", nargs="+", type=int, default=[32, 64, 128, 256, 512], help="Knot points.")
    parser.add_argument("--rates", nargs="+", type=int, default=[250, 500, 1000], help="Control rates (Hz).")
    parser.add_argument("--hide-compiler-warnings", action="store_true", help="Surpresses the warnings from the compiler for MPCGPU and the hip aquivalent.")
    args = parser.parse_args()

    # Determine architecture
    if args.hw == "auto":
        architectures = detect_hardware()
    elif args.hw == "nvidia":
        architectures = ["cuda", "nv_hip"]
    else:
        architectures = ["amd_hip"]

    if not architectures:
        print("[!] No compatible hardware found. Exiting.")
        return

    # Path Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    archive_base = os.path.join(script_dir, "benchmark_archive")
    os.makedirs(archive_base, exist_ok=True)

    repo_paths = {
        "cuda": os.path.join(project_root, "MPCGPU"),
        "nv_hip": os.path.join(project_root, "mcpgpu-hip-port", "raw_hip_port"),
        "amd_hip": os.path.join(project_root, "mcpgpu-hip-port", "raw_hip_port")
    }

    # Run Benchmark Matrix
    for arch in architectures:
        active_repo = repo_paths[arch]
        if not os.path.exists(active_repo):
            print(f"[!] Warning: Path {active_repo} missing. Skipping {arch}.")
            continue

        for knots, rate_hz in itertools.product(args.knots, args.rates):
            print(f"\n[*] COMPILING {arch.upper()} | Knots: {knots} | Rate: {rate_hz}Hz")
            
            if not compile_project(active_repo, knots, rate_hz, args.hide_compiler_warnings):
                print(f"    [!] Compile error for {knots} knots. Skipping.")
                continue

            for solver_name in args.solvers:
                print(f"    -> Running: {solver_name}")
                run_solver(active_repo, arch, solver_name)
                archive_results(active_repo, archive_base, arch, solver_name, knots, rate_hz)

if __name__ == "__main__":
    main()