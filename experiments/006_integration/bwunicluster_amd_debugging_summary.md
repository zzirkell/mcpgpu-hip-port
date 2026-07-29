  # bwUniCluster AMD Debugging Summary

  ## Goal

  Validate the raw CUDA-to-HIP PCG port on a real AMD accelerator, identify the
  cause of incorrect full MPC tracking behavior, and compare the HIP AMD path with
  the HIP NVIDIA/CUDA-backend path.

  ## Test Platforms

  - HIP NVIDIA: NVIDIA RTX 5060 Laptop GPU, HIP using the CUDA backend.
  - HIP AMD: AMD Instinct MI300A on bwUniCluster, `gfx942`, ROCm 7.0.
  - Important environment note: AMD runtime tests must be run inside a `gpu_mi300`
    allocation, for example on `uc3n083`. Login nodes such as `uc3n990` do not
    expose a ROCm-capable GPU and produce:
    `GPUassert: no ROCm-capable device is detected`.

  ## Confirmed Working Components

  The following stages were validated on MI300A and produced finite results that
  matched or closely followed the HIP NVIDIA path:

  - Cooperative kernel launch on MI300A.
  - Standalone GBD-PCG compilation/runtime checks.
  - PCG linear-system setup: `S`, `Pinv`, and `gamma`.
  - Real-dimension PCG solve and `lambda`.
  - `compute_dz`.
  - KKT outputs: `G`, `C`, `g`, and `c`.
  - Updated `xu`.
  - Sequential and cooperative merit calculations.
  - Full one-tolerance real-time MPC trajectory after workspace/reset fixes.

  This ruled out the earlier theory that the MI300A failure was caused by the M7
  cooperative kernel itself.

  ## Bugs Found And Fixed

  ### 1. SQP rho became zero on AMD

  The full SQP trace showed:

  ```text
  iteration 0: valid result
  iteration 1: rho = 0
  later iterations: PCG reached its limit and produced NaNs
  ```

  The code used unqualified `min(...)` and `max(...)` expressions for floating
  point rho updates. They did not behave portably under the AMD HIP build.

  The fix used typed standard-library operations and explicit casts:

  ```cpp
  drho = std::max<T>(drho * rho_factor, rho_factor);
  rho = std::max<T>(rho * drho, rho_min);

  drho = std::min<T>(
      drho / rho_factor,
      static_cast<T>(1) / rho_factor
  );
  rho = std::max<T>(rho * drho, rho_min);
  ```

  After the fix, the AMD SQP trace followed the NVIDIA rho sequence, PCG no
  longer failed, and the final lambda remained finite.

  ### 2. PCG exit tolerance was corrupted on AMD

  The full AMD run printed impossible values such as:

  ```text
  Exit tol: 3.9785e+27
  ```

  The example used a variable-length stack array, which is not standard C++:

  ```cpp
  uint32_t num_exit_vals = 5;
  float pcg_exit_vals[num_exit_vals];
  ```

  It was replaced with:

  ```cpp
  constexpr uint32_t num_exit_vals = 5;
  std::array<linsys_t, num_exit_vals> pcg_exit_vals{};
  ```

  The selected value is now read as:

  ```cpp
  const linsys_t pcg_exit_tol = pcg_exit_vals.at(pcg_exit_ind);
  ```

  Debug prints confirmed that the tolerance remained unchanged before and after
  `simulateMPC`.

  ### 3. Tracking comparison depended on measured execution time

  With `CONST_UPDATE_FREQ=0`, the simulator used the measured SQP runtime as the
  amount of simulated plant time:

  ```cpp
  simulation_time = sqp_solve_time_us;
  ```

  This made tracking accuracy depend directly on platform speed. The slower AMD
  execution advanced the simulated robot further after each control update,
  causing poor tracking even though isolated calculations matched NVIDIA.

  A controlled comparison kept 20 SQP iterations but used the same simulation
  period on both platforms:

  ```cpp
  #if CONST_UPDATE_FREQ || MPCGPU_FIXED_SIM_TIME
      simulation_time = SIMULATION_PERIOD;
  #else
      simulation_time = sqp_solve_time_us;
  #endif
  ```

  This comparison is for validation. It does not by itself prove that the AMD
  implementation meets the real-time deadline.

  ### 4. ROCm resource-lifecycle overhead dominated early real-time tests

  Profiling the AMD real-time run showed that most of the time was not spent in
  kernel math. The large cost came from repeated runtime resource management:

  - stream creation/destruction,
  - BLAS handle creation/destruction,
  - repeated device allocations/frees,
  - synchronization and runtime overhead.

  A temporary static-resource experiment showed that reusing streams and BLAS
  handles brought AMD update times much closer to the NVIDIA/HIP path. This
  confirmed that the slowdown was primarily a resource-lifecycle problem exposed
  by ROCm, not a PCG algorithm error.

  ### 5. Final `SqpWorkspace` introduced deterministic resource ownership

  A backend-neutral `SqpWorkspace<T>` was introduced for the SQP path. It owns and
  reuses:

  - HIP streams,
  - cuBLAS/hipBLAS handle,
  - merit buffers,
  - KKT/linsys buffers,
  - PCG scratch buffers,
  - solver status buffers.

  The workspace is constructed once per simulation and reused across MPC updates.
  This replaced the temporary static-resource experiment with explicit ownership.

  ### 6. Reused workspace buffers needed explicit reset

  After adding `SqpWorkspace`, NVIDIA worked, but AMD real-time behavior was still
  unstable. The reason was that some scratch buffers were no longer fresh
  allocations on every SQP call. They were reused, so AMD could observe stale
  state unless the buffers were explicitly cleared.

  The reset patch clears reused scratch/status buffers at the start of
  `sqpSolvePcg()`, including:

  ```text
  d_merit_initial
  d_merit_news
  d_merit_temp
  d_S
  d_gamma
  d_dz
  d_Pinv
  d_r
  d_p
  d_v_temp
  d_eta_new_temp
  d_pcg_iters
  d_pcg_exit
  ```

  After this reset, the short 20-update AMD and NVIDIA real-time tests produced
  almost identical tracking errors:

  ```text
  AMD run 1 tracking average:    0.00164031
  AMD run 2 tracking average:    0.00164136
  NVIDIA tracking average:       0.00162743
  ```

  A full one-tolerance AMD real-time run also became stable:

  ```text
  AMD full one-tolerance run 1:
  tracking average: 0.095847
  final error:      0.0121808
  SQP average:      0.674673

  AMD full one-tolerance run 2:
  tracking average: 0.100903
  final error:      0.033339
  SQP average:      0.670254
  ```

  ## Correctness Comparison Under Equal Simulation Conditions

  Average tracking error with fixed simulation time:

  | PCG tolerance | HIP NVIDIA | HIP AMD MI300A |
  |---|---:|---:|
  | `5e-6` | `0.115502` | `0.111579` |
  | `7.5e-6` | `0.118749` | `0.123287` |
  | `5e-6` repeated | `0.115502` | `0.111579` |
  | `2.5e-6` | `0.112850` | `0.109568` |
  | `1e-6` | `0.089192` | `0.0924532` |

  These results confirm that the corrected HIP AMD implementation reproduces the
  HIP NVIDIA tracking behavior under equal simulation conditions.

  ## Final Real-Time Budget Findings

  The original real-time benchmark uses `CONST_UPDATE_FREQ=1` and a strict
  `SQP_MAX_TIME_US=2000` deadline.

  After the correctness and workspace fixes, the AMD path still showed unstable
  behavior with the 2 ms SQP budget. The failure mode was not NaNs or wrong
  linear algebra; the solver often exited before completing useful SQP iterations:

  ```text
  Bad 2 ms runs:
  SQP average approximately 0.003 - 0.017
  tracking average approximately 1.2 - 1.35
  ```

  Increasing the SQP time budget showed that the remaining problem is performance
  and jitter, not correctness.

  ### 5 ms SQP budget

  All five tolerance runs passed on MI300A:

  | PCG tolerance | Tracking average | Final tracking error | Average SQP iterations |
  |---|---:|---:|---:|
  | `5e-6` | `0.117501` | `0.0590143` | `3.18793` |
  | `7.5e-6` | `0.123608` | `0.0196055` | `3.38182` |
  | `5e-6` repeated | `0.114170` | `0.0597235` | `3.18486` |
  | `2.5e-6` | `0.102909` | `0.0799025` | `2.44043` |
  | `1e-6` | `0.0940199` | `0.0636329` | `2.50346` |

  ### 10 ms SQP budget

  All five tolerance runs passed on MI300A:

  | PCG tolerance | Tracking average | Final tracking error | Average SQP iterations |
  |---|---:|---:|---:|
  | `5e-6` | `0.113888` | `0.038703` | `6.86414` |
  | `7.5e-6` | `0.117582` | `0.0554984` | `8.40008` |
  | `5e-6` repeated | `0.106101` | `0.0539713` | `7.97732` |
  | `2.5e-6` | `0.103926` | `0.0325244` | `7.81860` |
  | `1e-6` | `0.088786` | `0.0163681` | `6.79958` |

  ## Final Status

  ```text
  Numerical correctness: validated.
  Cooperative launch on MI300A: validated.
  Major AMD correctness bugs: fixed.
  Resource lifecycle issue: identified and mostly addressed with SqpWorkspace.
  Workspace stale-buffer issue: fixed with explicit resets.
  Full AMD runs at 5 ms and 10 ms SQP budgets: validated.
  Original strict 2 ms real-time deadline on AMD: not yet stable.
  ```

  The remaining open issue is no longer a HIP correctness bug. It is the
  performance/jitter gap under the original 2 ms real-time deadline on AMD/ROCm.

  ## Manual Portability Changes Confirmed During This Work

  - Full six-argument HIP cooperative-launch calls.
  - Full MPCGPU GLASS used for the MPCGPU linsys/SQP path.
  - Missing direct standard-library includes added where required.
  - HIP runtime return values checked where practical.
  - Dynamic shared memory made type-independent when float and double kernels
    were instantiated in one translation unit.
  - Typed `std::min<T>` and `std::max<T>` used for SQP rho updates.
  - Variable-length PCG tolerance array replaced with `std::array`.
  - Fixed simulation-time mode added for platform-independent correctness tests.
  - `SqpWorkspace<T>` added for reusable SQP resources.
  - Reused SQP workspace scratch buffers explicitly reset per solve.
  - Example modified to allow tolerance-index selection for separate-process
    AMD benchmark tests.

  ## Remaining Work

  1. Transfer the confirmed fixes from `raw_hip_port` to final `hip_port`.
  2. Remove temporary debug prints and test-only instrumentation.
  3. Clean up remaining warnings:
    - VLA warning in `mpcsim.hip.hpp` for `h_xs[state_size]`;
    - ignored `hipFuncSetAttribute` return values in generated dynamics code.
  4. Add guard logic for empty result vectors in short performance tests.
  5. Re-run final NVIDIA and AMD correctness tests from the cleaned `hip_port`.
  6. Decide with the supervisor whether the original 2 ms deadline is a hard
    requirement for AMD, or whether the current port is accepted as numerically
    correct with remaining AMD-specific performance optimization work.
  7. Continue performance work only after the cleaned port is committed and the
    remaining 2 ms bottleneck is measured precisely.
