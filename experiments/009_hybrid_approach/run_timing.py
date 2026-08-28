#!/usr/bin/env python3
import os
import subprocess
import shutil
from pathlib import Path


def resolve_cuda_repo(active_repo: Path) -> Path:
    """Return CUDA MPCGPU source tree.

    Preferred layout after this patch:
      mcpgpu-hip-port/MPCGPU

    Also supports the old colleague layout:
      mpcgpu_project/MPCGPU
      mpcgpu_project/mcpgpu-hip-port
    """
    candidates = [
        active_repo / "MPCGPU",
        active_repo.parent / "MPCGPU",
    ]
    for candidate in candidates:
        if (candidate / "examples").exists() and (candidate / "include").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find CUDA MPCGPU folder. Expected either "
        f"{candidates[0]} or {candidates[1]}."
    )

def main():
    script_dir = Path(__file__).resolve().parent
    bash_script = script_dir / "run_backend.sh"
    active_repo = script_dir.parent.parent
    
    KNOTS = [32, 64, 128]
    
    for knot in KNOTS:
        print(f"\n=== Executing TIMING | CUDA HYBRID | Knots: {knot} | WS: ON ===")
        src_dir = resolve_cuda_repo(active_repo)
        
        tmp_results = src_dir / "tmp" / "results"
        if tmp_results.exists():
            shutil.rmtree(tmp_results)
        tmp_results.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(bash_script), "cuda", str(knot), "50000",
            "-DUSE_SQP_WORKSPACE=1", "hybrid", "20", "5e-5", 
            "1" 
        ]
        
        subprocess.run(cmd, cwd=src_dir, check=True)
        
        archive_name = f"Timing_cuda_hybrid_K{knot}"
        archive_folder = script_dir / "timing_data" / archive_name
        archive_folder.mkdir(parents=True, exist_ok=True)
        
        if tmp_results.exists():
            for file_path in tmp_results.glob("*.*"):
                if file_path.suffix in ['.csv', '.result']:
                    shutil.copy(file_path, archive_folder)

if __name__ == "__main__":
    main()