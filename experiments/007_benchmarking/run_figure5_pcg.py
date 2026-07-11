#!/usr/bin/env python3
"""
Figure 5-style linear-system benchmark runner for MPCGPU HIP port.

What it does:
  1. Builds raw_hip_port/examples/track_iiwa_pcg.hip.cpp for one backend.
  2. Runs K=128 PCG tolerance indices separately.
  3. Optionally builds/runs raw_hip_port/examples/track_iiwa_qdldl.hip.cpp.
  4. Reads tmp/results/*_linsys_times.result.
  5. Writes combined raw CSV, summary CSV, and CDF plot.

Important:
  - PCG curves use the timed `linsys_times` values produced by MPCGPU.
    In the current PCG code those are PCG solve timings, not Schur+PCG.
  - QDLDL is CPU linear solve baseline inside the same MPCGPU simulation context.
  - For AMD, use --sqp-budget-us 5000 first. Use 2000 only when testing the
    strict original real-time deadline.
"""

from __future__ import annotations

import argparse
import csv
import os
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


def run_checked(
    cmd: List[str],
    *,
    cwd: Path,
    env: dict,
    log_file: Path | None = None,
    timeout: int | None = None,
) -> None:
    print("$", " ".join(cmd))
    if log_file is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True, timeout=timeout)
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        f.write(proc.stdout)
        print(proc.stdout)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def repo_default() -> Path:
    # If placed at experiments/007_benchmarking/run_figure5_pcg.py:
    # parents[2] is project root, raw_hip_port is below it.
    here = Path(__file__).resolve()
    project_root = here.parents[2] if len(here.parents) >= 3 else Path.cwd()
    candidate = project_root / "raw_hip_port"
    return candidate if candidate.exists() else Path.cwd()


def common_flags(args: argparse.Namespace, *, linsys_solve: int) -> List[str]:
    flags = [
        "-std=c++17",
        "-O3",
        f"-DKNOT_POINTS={args.knots}",
        f"-DLINSYS_SOLVE={linsys_solve}",
        "-DTIME_LINSYS=1",
        "-DSAVE_DATA=1",
        f"-DTEST_ITERS={args.test_iters}",
        "-DCONST_UPDATE_FREQ=1",
        f"-DSQP_MAX_TIME_US={args.sqp_budget_us}",
        "-DMPCGPU_FIXED_SIM_TIME=0",
        "-DMPCGPU_TRACE_UPDATE_TIME=0",
        f"-DREMOVE_JITTERS={1 if args.remove_jitters else 0}",
    ]
    if args.extra_cxxflags:
        flags += args.extra_cxxflags.split()
    return flags


def common_includes() -> List[str]:
    return [
        "-Iinclude",
        "-Iinclude/common",
        "-Iinclude/utils",
        "-Iinclude/pcg",
        "-Iinclude/qdldl",
        "-Iinclude/dynamics",
        "-Iinclude/dynamics/iiwa",
        "-IGLASS",
        "-IGBD-PCG/include",
        "-Iqdldl/include",
    ]


def backend_link_flags(backend: str, args: argparse.Namespace, env: dict) -> Tuple[List[str], List[str]]:
    """Return (backend-specific compile flags, backend-specific link flags)."""
    if backend == "hip-amd":
        rocm_root = args.rocm_root or env.get("ROCM_ROOT") or env.get("ROCM_PATH") or "/opt/rocm"
        compile_flags = [f"--offload-arch={args.amd_arch}", f"-I{rocm_root}/include/hipblas"]
        link_flags = [f"-L{rocm_root}/lib", "-lhipblas", "-lrocblas"]
        return compile_flags, link_flags

    if backend == "hip-nvidia":
        cuda_root = args.cuda_root or env.get("CUDA_HOME") or env.get("CUDA_PATH") or "/usr/local/cuda"
        env["HIP_PLATFORM"] = "nvidia"
        compile_flags = [f"-I{cuda_root}/include"]
        link_flags = [f"-L{cuda_root}/lib64", "-lcublas", "-lcudart"]
        return compile_flags, link_flags

    raise ValueError(f"Unsupported backend: {backend}")


def build_pcg_command(repo: Path, backend: str, output_exe: Path, args: argparse.Namespace) -> Tuple[List[str], dict]:
    env = os.environ.copy()
    backend_compile_flags, backend_link_flags_ = backend_link_flags(backend, args, env)
    cmd = [
        "hipcc",
        *common_flags(args, linsys_solve=1),
        *backend_compile_flags,
        *common_includes(),
        "examples/track_iiwa_pcg.hip.cpp",
        "-o",
        str(output_exe),
        *backend_link_flags_,
    ]
    return cmd, env


def build_qdldl_commands(repo: Path, backend: str, output_exe: Path, args: argparse.Namespace, build_dir: Path) -> Tuple[List[List[str]], dict]:
    """Build QDLDL executable.

    QDLDL itself is plain C, so we compile qdldl/src/qdldl.c into an object with cc
    and link that object into the HIP example. This avoids depending on a prebuilt
    qdldl/temp_build library path.
    """
    env = os.environ.copy()
    backend_compile_flags, backend_link_flags_ = backend_link_flags(backend, args, env)
    qdldl_obj = build_dir / "qdldl.o"
    cc = args.qdldl_cc or env.get("CC") or "cc"

    compile_qdldl_obj = [
        cc,
        "-O3",
        "-std=c99",
        "-Iqdldl/include",
        "-c",
        "qdldl/src/qdldl.c",
        "-o",
        str(qdldl_obj),
    ]

    link_qdldl_exe = [
        "hipcc",
        *common_flags(args, linsys_solve=0),
        *backend_compile_flags,
        *common_includes(),
        "examples/track_iiwa_qdldl.hip.cpp",
        str(qdldl_obj),
        "-o",
        str(output_exe),
        *backend_link_flags_,
    ]
    return [compile_qdldl_obj, link_qdldl_exe], env


def clear_tmp_results(repo: Path) -> None:
    tmp = repo / "tmp" / "results"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)


def read_linsys_times(results_dir: Path) -> List[float]:
    values: List[float] = []
    for path in sorted(results_dir.glob("*_linsys_times.result")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values.append(float(line))
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


def add_series_rows(
    *,
    raw_rows: List[dict],
    summary_rows: List[dict],
    backend: str,
    knots: int,
    sqp_budget_us: int,
    test_iters: int,
    solver: str,
    label: str,
    values: List[float],
    tol_index: int | str = "",
    pcg_exit_tol: float | str = "",
) -> None:
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

    summary = summarize(values)
    summary_rows.append({
        "backend": backend,
        "knot_points": knots,
        "sqp_budget_us": sqp_budget_us,
        "test_iters": test_iters,
        "solver": solver,
        "label": label,
        "tol_index": tol_index,
        "pcg_exit_tol": pcg_exit_tol,
        **summary,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Figure 5-style GBD-PCG + QDLDL CDF benchmark.")
    parser.add_argument("--repo", type=Path, default=repo_default(), help="Path to raw_hip_port.")
    parser.add_argument("--backend", choices=["hip-amd", "hip-nvidia"], required=True)
    parser.add_argument("--knots", type=int, default=128)
    parser.add_argument("--tol-indices", nargs="+", type=int, default=[2, 1, 0],
                        help="For K=128: 2=1e-4, 1=5e-5, 0=1e-5.")
    parser.add_argument("--test-iters", type=int, default=8)
    parser.add_argument("--sqp-budget-us", type=int, default=5000,
                        help="Use 2000 for strict 500Hz/original deadline; use 5000 for stable AMD data.")
    parser.add_argument("--remove-jitters", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rocm-root", default=None)
    parser.add_argument("--cuda-root", default=os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda")
    parser.add_argument("--amd-arch", default="gfx942", help="AMD GPU architecture, e.g. gfx942 for MI300A.")
    parser.add_argument("--extra-cxxflags", default="")
    parser.add_argument("--skip-qdldl", action="store_true", help="Only run PCG tolerance curves, no CPU QDLDL curve.")
    parser.add_argument("--only-qdldl", action="store_true", help="Only run CPU QDLDL curve, no PCG curves.")
    parser.add_argument("--qdldl-cc", default=None, help="C compiler for qdldl/src/qdldl.c, default: $CC or cc.")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not (repo / "examples" / "track_iiwa_pcg.hip.cpp").exists():
        print(f"ERROR: repo does not look like raw_hip_port: {repo}", file=sys.stderr)
        return 2
    if not (repo / "examples" / "track_iiwa_qdldl.hip.cpp").exists() and not args.skip_qdldl:
        print(f"ERROR: QDLDL example not found in repo: {repo}", file=sys.stderr)
        return 2

    out_base = args.out or (repo.parent / "experiments" / "007_benchmarking" / "figure5_results")
    qdldl_tag = "pcg_only" if args.skip_qdldl else "with_qdldl"
    if args.only_qdldl:
        qdldl_tag = "qdldl_only"
    run_name = f"{args.backend}_K{args.knots}_budget{args.sqp_budget_us}_iters{args.test_iters}_{qdldl_tag}"
    out_dir = (out_base / run_name).resolve()
    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    pcg_exe = build_dir / "pcg_figure5.exe"
    qdldl_exe = build_dir / "qdldl_figure5.exe"

    print(f"Repo: {repo}")
    print(f"Output: {out_dir}")
    print(f"Backend: {args.backend}")
    print(f"Knot points: {args.knots}")
    print(f"SQP budget: {args.sqp_budget_us} us")
    print(f"TEST_ITERS: {args.test_iters}")
    print(f"QDLDL included: {not args.skip_qdldl}")

    raw_rows: List[dict] = []
    summary_rows: List[dict] = []
    plot_series: Dict[str, List[float]] = {}

    if not args.only_qdldl:
        cmd, env = build_pcg_command(repo, args.backend, pcg_exe, args)
        run_checked(cmd, cwd=repo, env=env, log_file=out_dir / "build_pcg.log")

        for tol_index in args.tol_indices:
            tol_value = TOLERANCES_FOR_128.get(tol_index, float("nan"))
            tol_label = f"GBD-PCG eps={tol_value:g}"
            print(f"\n=== Running tolerance index {tol_index} ({tol_label}) ===")

            clear_tmp_results(repo)
            log_path = out_dir / f"run_pcg_tol_index_{tol_index}.log"
            run_checked([str(pcg_exe), str(tol_index)], cwd=repo, env=env, log_file=log_path, timeout=args.timeout)

            tmp_results = repo / "tmp" / "results"
            archive = out_dir / f"pcg_tol_index_{tol_index}"
            if archive.exists():
                shutil.rmtree(archive)
            shutil.copytree(tmp_results, archive)

            values = read_linsys_times(archive)
            if not values:
                print(f"WARNING: no *_linsys_times.result files for tolerance index {tol_index}")
            plot_series[tol_label] = values

            add_series_rows(
                raw_rows=raw_rows,
                summary_rows=summary_rows,
                backend=args.backend,
                knots=args.knots,
                sqp_budget_us=args.sqp_budget_us,
                test_iters=args.test_iters,
                solver="GBD-PCG",
                label=tol_label,
                values=values,
                tol_index=tol_index,
                pcg_exit_tol=tol_value,
            )
            print("Summary:", summary_rows[-1])

    if not args.skip_qdldl:
        print("\n=== Building/running CPU QDLDL baseline ===")
        qdldl_cmds, qdldl_env = build_qdldl_commands(repo, args.backend, qdldl_exe, args, build_dir)
        run_checked(qdldl_cmds[0], cwd=repo, env=qdldl_env, log_file=out_dir / "build_qdldl_obj.log")
        run_checked(qdldl_cmds[1], cwd=repo, env=qdldl_env, log_file=out_dir / "build_qdldl.log")

        clear_tmp_results(repo)
        run_checked([str(qdldl_exe)], cwd=repo, env=qdldl_env, log_file=out_dir / "run_qdldl.log", timeout=args.timeout)

        tmp_results = repo / "tmp" / "results"
        archive = out_dir / "qdldl"
        if archive.exists():
            shutil.rmtree(archive)
        shutil.copytree(tmp_results, archive)

        qdldl_values = read_linsys_times(archive)
        if not qdldl_values:
            print("WARNING: no *_linsys_times.result files for QDLDL")
        qdldl_label = "CPU QDLDL"
        plot_series[qdldl_label] = qdldl_values

        add_series_rows(
            raw_rows=raw_rows,
            summary_rows=summary_rows,
            backend=args.backend,
            knots=args.knots,
            sqp_budget_us=args.sqp_budget_us,
            test_iters=args.test_iters,
            solver="QDLDL",
            label=qdldl_label,
            values=qdldl_values,
        )
        print("Summary:", summary_rows[-1])

    raw_fields = [
        "backend",
        "knot_points",
        "sqp_budget_us",
        "test_iters",
        "solver",
        "label",
        "tol_index",
        "pcg_exit_tol",
        "linsys_time_us",
    ]
    summary_fields = [
        "backend",
        "knot_points",
        "sqp_budget_us",
        "test_iters",
        "solver",
        "label",
        "tol_index",
        "pcg_exit_tol",
        "count",
        "min",
        "mean",
        "median",
        "q1",
        "q3",
        "p90",
        "p99",
        "max",
    ]

    write_csv(out_dir / "figure5_linsys_raw_times.csv", raw_fields, raw_rows)
    write_csv(out_dir / "figure5_linsys_summary.csv", summary_fields, summary_rows)

    plot_cdf(
        out_dir / "figure5_linsys_cdf.png",
        plot_series,
        title=f"Figure 5-style linear-system CDF ({args.backend}, K={args.knots}, budget={args.sqp_budget_us}us)",
    )

    print("\nDONE")
    print(f"Raw CSV:     {out_dir / 'figure5_linsys_raw_times.csv'}")
    print(f"Summary CSV: {out_dir / 'figure5_linsys_summary.csv'}")
    print(f"Plot:        {out_dir / 'figure5_linsys_cdf.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())