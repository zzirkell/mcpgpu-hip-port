#!/usr/bin/env python3
import os
import subprocess
import shutil
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    bash_script = script_dir / "run_backend.sh"
    active_repo = script_dir.parent.parent
    
    KNOTS = [32, 64, 128]
    
    for knot in KNOTS:
        print(f"\n=== Executing TIMING | CUDA HYBRID | Knots: {knot} | WS: ON ===")
        src_dir = active_repo.parent / "MPCGPU"
        
        tmp_results = src_dir / "tmp" / "results"
        if tmp_results.exists():
            shutil.rmtree(tmp_results)
        tmp_results.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(bash_script), "cuda", str(knot), "50000",
            "-DUSE_SQP_WORKSPACE=1", "hybrid", "20", "5e-5"
        ]
        
        subprocess.run(cmd, cwd=src_dir)
        
        # Ordnername endet jetzt auf WS_ON
        archive_name = f"Timing_cuda_hybrid_K{knot}"
        archive_folder = script_dir / "timing_data" / archive_name
        archive_folder.mkdir(parents=True, exist_ok=True)
        
        if tmp_results.exists():
            for file_path in tmp_results.glob("*.*"):
                if file_path.suffix in ['.csv', '.result']:
                    shutil.copy(file_path, archive_folder)

if __name__ == "__main__":
    main()