#!/usr/bin/env python3
"""
Figure 5-style linear-system benchmark runner for MPCGPU.

Backends supported:
  - cuda       : native/original CUDA baseline, built from an older CUDA-source commit via git worktree
  - hip-nvidia : current raw_hip_port HIP code on NVIDIA backend
  - hip-amd    : current raw_hip_port HIP code on AMD backend

For smoke tests use --test-iters 1. For real data use higher --test-iters.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TOLERANCES_FOR_128: Dict[int, float] = {
    2: 1e-4,
    1: 5e-5,
    0: 1e-5,
}
EPS_TO_INDEX_FOR_128: Dict[str, int] = {
    "0.000100": 2,
    "0.000050": 1,
    "0.000010": 0,
}


def sh(cmd: List[str], *, cwd: Path, env: dict | None = None, log_file: Path | None = None, timeout: int | None = None) -> None:
    print("$", " ".join(cmd))
    env = env or os.environ.copy()
    if log_file is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True, timeout=timeout)
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    log_file.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def project_root_default() -> Path:
    here = Path(__file__).resolve()
    # expected location: experiments/007_benchmarking/run_figure5_linsys_backend.py
    if len(here.parents) >= 3 and (here.parents[2] / "raw_hip_port").exists():
        return here.parents[2]
    cwd = Path.cwd()
    if (cwd / "raw_hip_port").exists():
        return cwd
    if cwd.name == "raw_hip_port":
        return cwd.parent
    return cwd


def raw_repo(project_root: Path) -> Path:
    return project_root / "raw_hip_port"


def clear_tmp_results(repo: Path) -> None:
    tmp = repo / "tmp" / "results"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)


def copy_results(repo: Path, archive: Path) -> None:
    tmp = repo / "tmp" / "results"
    if archive.exists():
        shutil.rmtree(archive)
    if tmp.exists():
        shutil.copytree(tmp, archive)
    else:
        archive.mkdir(parents=True, exist_ok=True)


def read_linsys_times(results_dir: Path) -> List[float]:
    values: List[float] = []
    for path in sorted(results_dir.glob("*_linsys_times.result")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    values.append(float(s))
    return values


def read_linsys_times_matching(results_dir: Path, pattern: str) -> List[float]:
    values: List[float] = []
    for path in sorted(results_dir.glob(pattern)):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    values.append(float(s))
    return values


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: float("nan") for k in ["count", "min", "mean", "median", "q1", "q3", "p90", "p99", "max"]}
    xs = sorted(values)

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = (len(xs) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(xs) - 1)
        frac = idx - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    return {
        "count": float(len(xs)),
        "min": xs[0],
        "mean": sum(xs) / len(xs),
        "median": pct(0.50),
        "q1": pct(0.25),
        "q3": pct(0.75),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": xs[-1],
    }


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_cdf(out_png: Path, series: Dict[str, List[float]], title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, values in series.items():
        xs = sorted(values)
        if not xs:
            continue
        ys = [(i + 1) * 100.0 / len(xs) for i in range(len(xs))]
        ax.plot(xs, ys, label=label)

    ax.set_xlabel("Linear system solve time (us)")
    ax.set_ylabel("Percentage of solves (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def add_rows(raw_rows: List[dict], summary_rows: List[dict], *, backend: str, knots: int, sqp_budget_us: int,
             test_iters: int, solver: str, label: str, values: List[float], tol_index: int | str = "",
             pcg_exit_tol: float | str = "") -> None:
    for v in values:
        raw_rows.append({
            "backend": backend,
            "knot_points": knots,
            "sqp_budget_us": sqp_budget_us,
            "test_iters": test_iters,
            "solver": solver,
            "label": label,
            "tol_index": tol_index,
            "pcg_exit_tol": pcg_exit_tol,
            "linsys_time_us": v,
        })
    summary_rows.append({
        "backend": backend,
        "knot_points": knots,
        "sqp_budget_us": sqp_budget_us,
        "test_iters": test_iters,
        "solver": solver,
        "label": label,
        "tol_index": tol_index,
        "pcg_exit_tol": pcg_exit_tol,
        **summarize(values),
    })
    print("Summary:", summary_rows[-1])


def common_define_flags(args: argparse.Namespace, *, linsys_solve: int) -> List[str]:
    flags = [
        f"-DKNOT_POINTS={args.knots}",
        f"-DLINSYS_SOLVE={linsys_solve}",
        "-DTIME_LINSYS=1",
        "-DSAVE_DATA=1",
        f"-DTEST_ITERS={args.test_iters}",
        "-DCONST_UPDATE_FREQ=1",
        f"-DSQP_MAX_TIME_US={args.sqp_budget_us}",
        f"-DREMOVE_JITTERS={1 if args.remove_jitters else 0}",
    ]
    # Only current HIP code knows these macros. They are harmless for HIP and intentionally not used for old CUDA.
    if args.extra_defines:
        flags += args.extra_defines.split()
    return flags


def build_qdldl_library(repo: Path, out_dir: Path, env: dict, args: argparse.Namespace) -> None:
    build_dir = repo / "qdldl" / "build"
    cmd_config = [
        "cmake",
        "-S", "qdldl",
        "-B", "qdldl/build",
        "-DQDLDL_FLOAT=true",
        "-DQDLDL_LONG=false",
        "-DQDLDL_BUILD_SHARED_LIB=OFF",
        "-DQDLDL_BUILD_DEMO_EXE=OFF",
        "-DQDLDL_UNITTESTS=OFF",
    ]
    cmd_build = ["cmake", "--build", "qdldl/build", "--parallel"]
    sh(cmd_config, cwd=repo, env=env, log_file=out_dir / "build_qdldl_cmake_config.log", timeout=args.timeout)
    sh(cmd_build, cwd=repo, env=env, log_file=out_dir / "build_qdldl_cmake_build.log", timeout=args.timeout)
    if not (build_dir / "out" / "libqdldl.a").exists():
        print("WARNING: expected qdldl/build/out/libqdldl.a was not found. Link may still find another qdldl library.")


def detect_amd_arch() -> str | None:
    try:
        proc = subprocess.run(["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
    except Exception:
        return None
    matches = re.findall(r"gfx[0-9a-zA-Z]+", proc.stdout)
    for m in matches:
        if m != "gfx000":
            return m
    return matches[0] if matches else None


def hip_compile_common(repo: Path, backend: str, args: argparse.Namespace, env: dict) -> Tuple[List[str], List[str]]:
    includes = [
        "-Iinclude", "-Iinclude/common", "-Iinclude/utils", "-Iinclude/pcg", "-Iinclude/qdldl",
        "-Iinclude/dynamics", "-Iinclude/dynamics/iiwa", "-IGLASS", "-IGBD-PCG/include", "-Iqdldl/include",
    ]
    link = []
    if backend == "hip-nvidia":
        cuda_root = args.cuda_root or env.get("CUDA_HOME") or env.get("CUDA_PATH") or "/usr/local/cuda"
        env["HIP_PLATFORM"] = "nvidia"
        includes.append(f"-I{cuda_root}/include")
        link += [f"-L{cuda_root}/lib64", "-lcublas", "-lcudart"]
    elif backend == "hip-amd":
        rocm_root = args.rocm_root or env.get("ROCM_ROOT") or env.get("ROCM_PATH") or "/opt/rocm"
        arch = args.amd_arch
        if arch == "auto":
            detected = detect_amd_arch()
            if not detected:
                raise RuntimeError("Could not auto-detect AMD gfx arch. Pass --amd-arch gfxXYZ manually.")
            arch = detected
            print(f"Detected AMD arch: {arch}")
        includes += [f"--offload-arch={arch}", f"-I{rocm_root}/include/hipblas"]
        link += [f"-L{rocm_root}/lib", "-lhipblas", "-lrocblas"]
    else:
        raise ValueError(backend)
    return includes, link


def run_hip_backend(project_root: Path, backend: str, args: argparse.Namespace) -> Path:
    repo = raw_repo(project_root).resolve()
    if not (repo / "examples" / "track_iiwa_pcg.hip.cpp").exists():
        raise FileNotFoundError(f"HIP PCG example not found: {repo}")
    if not (repo / "examples" / "track_iiwa_qdldl.hip.cpp").exists() and not args.skip_qdldl:
        raise FileNotFoundError(f"HIP QDLDL example not found: {repo}")

    qdldl_tag = "pcg_only" if args.skip_qdldl else "with_qdldl"
    out_dir = (args.out / f"{backend}_K{args.knots}_budget{args.sqp_budget_us}_iters{args.test_iters}_{qdldl_tag}").resolve()
    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    includes, link = hip_compile_common(repo, backend, args, env)

    raw_rows: List[dict] = []
    summary_rows: List[dict] = []
    plot_series: Dict[str, List[float]] = {}

    print(f"\n================ {backend} ================")
    print(f"Repo: {repo}")
    print(f"Output: {out_dir}")

    pcg_exe = build_dir / "pcg_figure5.exe"
    pcg_cmd = [
        "hipcc", "-std=c++17", "-O3", *common_define_flags(args, linsys_solve=1),
        "-DMPCGPU_FIXED_SIM_TIME=0", "-DMPCGPU_TRACE_UPDATE_TIME=0",
        *includes,
        "examples/track_iiwa_pcg.hip.cpp", "-o", str(pcg_exe), *link,
    ]
    sh(pcg_cmd, cwd=repo, env=env, log_file=out_dir / "build_pcg.log", timeout=args.timeout)

    for tol_index in args.tol_indices:
        tol_value = TOLERANCES_FOR_128[tol_index]
        label = f"GBD-PCG eps={tol_value:g}"
        print(f"\n=== {backend}: PCG tolerance index {tol_index} ({label}) ===")
        clear_tmp_results(repo)
        sh([str(pcg_exe), str(tol_index)], cwd=repo, env=env,
           log_file=out_dir / f"run_pcg_tol_index_{tol_index}.log", timeout=args.timeout)
        archive = out_dir / f"pcg_tol_index_{tol_index}"
        copy_results(repo, archive)
        values = read_linsys_times(archive)
        plot_series[label] = values
        add_rows(raw_rows, summary_rows, backend=backend, knots=args.knots, sqp_budget_us=args.sqp_budget_us,
                 test_iters=args.test_iters, solver="GBD-PCG", label=label, values=values,
                 tol_index=tol_index, pcg_exit_tol=tol_value)

    if not args.skip_qdldl:
        print(f"\n=== {backend}: CPU QDLDL baseline ===")
        build_qdldl_library(repo, out_dir, env, args)
        qdldl_exe = build_dir / "qdldl_figure5.exe"
        qdldl_cmd = [
            "hipcc", "-std=c++17", "-O3", *common_define_flags(args, linsys_solve=0),
            "-DMPCGPU_FIXED_SIM_TIME=0", "-DMPCGPU_TRACE_UPDATE_TIME=0",
            *includes,
            "examples/track_iiwa_qdldl.hip.cpp", "-o", str(qdldl_exe),
            "-Lqdldl/build/out", "-lqdldl", *link,
        ]
        sh(qdldl_cmd, cwd=repo, env=env, log_file=out_dir / "build_qdldl.log", timeout=args.timeout)
        clear_tmp_results(repo)
        qdldl_env = env.copy()
        qdldl_env["LD_LIBRARY_PATH"] = f"{repo / 'qdldl' / 'build' / 'out'}:{qdldl_env.get('LD_LIBRARY_PATH', '')}"
        sh([str(qdldl_exe)], cwd=repo, env=qdldl_env, log_file=out_dir / "run_qdldl.log", timeout=args.timeout)
        archive = out_dir / "qdldl"
        copy_results(repo, archive)
        values = read_linsys_times(archive)
        label = "CPU QDLDL"
        plot_series[label] = values
        add_rows(raw_rows, summary_rows, backend=backend, knots=args.knots, sqp_budget_us=args.sqp_budget_us,
                 test_iters=args.test_iters, solver="QDLDL", label=label, values=values)

    write_outputs(out_dir, raw_rows, summary_rows, plot_series, backend, args)
    return out_dir


def ensure_cuda_worktree(project_root: Path, args: argparse.Namespace, out_base: Path) -> Path:
    worktree = (args.cuda_worktree or (out_base / "_cuda_original_worktree")).resolve()
    if args.refresh_cuda_worktree and worktree.exists():
        shutil.rmtree(worktree)
    if not worktree.exists():
        worktree.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "worktree", "add", "--detach", str(worktree), args.cuda_commit],
           cwd=project_root, env=os.environ.copy(), log_file=out_base / "cuda_worktree_add.log", timeout=args.timeout)
    cuda_repo = worktree / "raw_hip_port"
    if not (cuda_repo / "examples" / "track_iiwa_pcg.cu").exists():
        raise FileNotFoundError(f"Native CUDA source not found in worktree: {cuda_repo}")
    return cuda_repo


def run_cuda_backend(project_root: Path, args: argparse.Namespace) -> Path:
    backend = "cuda"
    out_dir = (args.out / f"cuda_K{args.knots}_budget{args.sqp_budget_us}_iters{args.test_iters}_with_qdldl").resolve()
    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    repo = ensure_cuda_worktree(project_root, args, args.out)
    env = os.environ.copy()
    cuda_root = args.cuda_root or env.get("CUDA_HOME") or env.get("CUDA_PATH") or "/usr/local/cuda"
    env["PATH"] = f"{cuda_root}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_root}/lib64:{env.get('LD_LIBRARY_PATH', '')}"

    raw_rows: List[dict] = []
    summary_rows: List[dict] = []
    plot_series: Dict[str, List[float]] = {}

    print(f"\n================ native CUDA ================")
    print(f"Repo/worktree: {repo}")
    print(f"Output: {out_dir}")

    build_qdldl_library(repo, out_dir, env, args)

    cuda_includes = ["-Iinclude", "-Iinclude/common", "-IGLASS", "-IGBD-PCG/include", "-Iqdldl/include"]
    cuda_link = ["-Lqdldl/build/out", "-lqdldl", f"-L{cuda_root}/lib64", "-lcublas"]
    cuda_common = ["nvcc", "-std=c++17", "--compiler-options", "-Wall", "-O3"]

    pcg_exe = build_dir / "pcg_figure5_cuda.exe"
    pcg_cmd = [
        *cuda_common, *common_define_flags(args, linsys_solve=1), *cuda_includes,
        "examples/track_iiwa_pcg.cu", "-o", str(pcg_exe), *cuda_link,
    ]
    sh(pcg_cmd, cwd=repo, env=env, log_file=out_dir / "build_pcg.log", timeout=args.timeout)

    print("\n=== native CUDA: PCG all tolerances in one original run ===")
    clear_tmp_results(repo)
    sh([str(pcg_exe)], cwd=repo, env=env, log_file=out_dir / "run_pcg_all_tolerances.log", timeout=args.timeout)
    pcg_archive = out_dir / "pcg_all_tolerances"
    copy_results(repo, pcg_archive)

    for tol_index in args.tol_indices:
        tol_value = TOLERANCES_FOR_128[tol_index]
        eps_key = f"{tol_value:.6f}"
        label = f"GBD-PCG eps={tol_value:g}"
        values = read_linsys_times_matching(pcg_archive, f"*PCG_{eps_key}_linsys_times.result")
        if not values:
            print(f"WARNING: no CUDA PCG linsys values for eps={eps_key}")
        plot_series[label] = values
        add_rows(raw_rows, summary_rows, backend=backend, knots=args.knots, sqp_budget_us=args.sqp_budget_us,
                 test_iters=args.test_iters, solver="GBD-PCG", label=label, values=values,
                 tol_index=tol_index, pcg_exit_tol=tol_value)

    if not args.skip_qdldl:
        qdldl_exe = build_dir / "qdldl_figure5_cuda.exe"
        qdldl_cmd = [
            *cuda_common, *common_define_flags(args, linsys_solve=0), *cuda_includes,
            "examples/track_iiwa_qdldl.cu", "-o", str(qdldl_exe), *cuda_link,
        ]
        sh(qdldl_cmd, cwd=repo, env=env, log_file=out_dir / "build_qdldl.log", timeout=args.timeout)
        print("\n=== native CUDA: CPU QDLDL baseline ===")
        clear_tmp_results(repo)
        qdldl_env = env.copy()
        qdldl_env["LD_LIBRARY_PATH"] = f"{repo / 'qdldl' / 'build' / 'out'}:{qdldl_env.get('LD_LIBRARY_PATH', '')}"
        sh([str(qdldl_exe)], cwd=repo, env=qdldl_env, log_file=out_dir / "run_qdldl.log", timeout=args.timeout)
        qdldl_archive = out_dir / "qdldl"
        copy_results(repo, qdldl_archive)
        values = read_linsys_times(qdldl_archive)
        label = "CPU QDLDL"
        plot_series[label] = values
        add_rows(raw_rows, summary_rows, backend=backend, knots=args.knots, sqp_budget_us=args.sqp_budget_us,
                 test_iters=args.test_iters, solver="QDLDL", label=label, values=values)

    write_outputs(out_dir, raw_rows, summary_rows, plot_series, backend, args)
    return out_dir


def write_outputs(out_dir: Path, raw_rows: List[dict], summary_rows: List[dict], plot_series: Dict[str, List[float]],
                  backend: str, args: argparse.Namespace) -> None:
    raw_fields = ["backend", "knot_points", "sqp_budget_us", "test_iters", "solver", "label", "tol_index", "pcg_exit_tol", "linsys_time_us"]
    summary_fields = raw_fields[:-1] + ["count", "min", "mean", "median", "q1", "q3", "p90", "p99", "max"]
    write_csv(out_dir / "figure5_linsys_raw_times.csv", raw_fields, raw_rows)
    write_csv(out_dir / "figure5_linsys_summary.csv", summary_fields, summary_rows)
    plot_cdf(out_dir / "figure5_linsys_cdf.png", plot_series,
             title=f"Figure 5-style linear-system CDF ({backend}, K={args.knots}, budget={args.sqp_budget_us}us)")
    print("\nDONE:")
    print(f"  Summary: {out_dir / 'figure5_linsys_summary.csv'}")
    print(f"  Plot:    {out_dir / 'figure5_linsys_cdf.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Figure 5-style smoke/full benchmark for CUDA, HIP NVIDIA, and HIP AMD.")
    parser.add_argument("--project-root", type=Path, default=project_root_default())
    parser.add_argument("--backends", nargs="+", choices=["cuda", "hip-nvidia", "hip-amd"], required=True)
    parser.add_argument("--knots", type=int, default=128)
    parser.add_argument("--tol-indices", nargs="+", type=int, default=[2, 1, 0])
    parser.add_argument("--test-iters", type=int, default=1)
    parser.add_argument("--sqp-budget-us", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-qdldl", action="store_true")
    parser.add_argument("--remove-jitters", action="store_true")
    parser.add_argument("--cuda-root", default=os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda")
    parser.add_argument("--rocm-root", default=None)
    parser.add_argument("--amd-arch", default="auto", help="Use auto, gfx942, gfx90a, gfx1030, etc.")
    parser.add_argument("--cuda-commit", default="0f566b3", help="Commit containing original/native CUDA raw_hip_port sources.")
    parser.add_argument("--cuda-worktree", type=Path, default=None)
    parser.add_argument("--refresh-cuda-worktree", action="store_true")
    parser.add_argument("--extra-defines", default="")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if not (project_root / ".git").exists():
        print(f"ERROR: project root does not look like a git repo: {project_root}", file=sys.stderr)
        return 2
    if args.out is None:
        args.out = project_root / "experiments" / "007_benchmarking" / "figure5_results"
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Output root:  {args.out}")
    print(f"Backends:     {args.backends}")
    print(f"Smoke/full:   TEST_ITERS={args.test_iters}, K={args.knots}, budget={args.sqp_budget_us}us")

    produced: List[Path] = []
    for backend in args.backends:
        if backend == "cuda":
            produced.append(run_cuda_backend(project_root, args))
        else:
            produced.append(run_hip_backend(project_root, backend, args))

    print("\nALL REQUESTED BACKENDS DONE")
    for p in produced:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
