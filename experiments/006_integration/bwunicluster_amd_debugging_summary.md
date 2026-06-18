# bwUniCluster AMD Debugging Summary


## Goal

Validate the raw CUDA-to-HIP PCG port on a real AMD accelerator, identify the
cause of the incorrect full MPC tracking result, and compare it with the HIP
NVIDIA build.

## Test Platforms

- HIP NVIDIA: NVIDIA RTX 5060 Laptop GPU, HIP using the CUDA backend.
- HIP AMD: AMD Instinct MI300A on bwUniCluster, `gfx942`, ROCm 7.0.

## Confirmed Working Components

The following stages ran successfully on MI300A and produced finite results
that closely matched HIP NVIDIA:

- Cooperative kernel launch.
- Standalone GBD-PCG, float and double.
- PCG linear-system setup: `S`, `Pinv`, and `gamma`.
- Real-dimension PCG solve and `lambda`.
- `compute_dz`.
- KKT outputs: `G`, `C`, `g`, and `c`.
- Updated `xu`.
- Sequential and cooperative merit calculations.

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

This comparison is for validation. It does not by itself prove that the
current AMD implementation meets the real-time deadline.

## Final Correctness Comparison

Average tracking error:

| PCG tolerance | HIP NVIDIA | HIP AMD MI300A |
|---|---:|---:|
| `5e-6` | `0.115502` | `0.111579` |
| `7.5e-6` | `0.118749` | `0.123287` |
| `5e-6` repeated | `0.115502` | `0.111579` |
| `2.5e-6` | `0.112850` | `0.109568` |
| `1e-6` | `0.089192` | `0.0924532` |

The repeated `5e-6` test was deterministic on each platform. These results
confirm that the corrected HIP AMD implementation reproduces the HIP NVIDIA
tracking behavior under equal simulation conditions.

## Performance Status

Compilation time is not a real-time metric because compilation happens before
deployment.

The reported linear-system timing also does not explain the earlier full-run
slowdown:

- NVIDIA median linear-system time was roughly `180-196 us`.
- AMD median linear-system time was roughly `121-123 us`.

The AMD median linear-system time was therefore not worse. The remaining
performance question concerns the complete SQP/MPC update, including KKT,
merit, dynamics, allocations, copies, synchronization, and kernel-launch
overhead.

Current conclusion:

```text
Numerical correctness: validated.
Cooperative launch on MI300A: validated.
Real-time deadline: not yet validated.
Performance profiling and optimization: still required.
```

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

## Remaining Work

1. Transfer the confirmed fixes from `raw_hip_port` to final `hip_port`.
2. Remove temporary tolerance and SQP trace prints.
3. Measure complete SQP update time on NVIDIA and AMD.
4. Profile AMD with ROCprofiler to locate time spent in kernels, memory
   allocation, copies, and synchronization.
5. Optimize only after the measured bottleneck is known.
6. Re-run final correctness and real-time tests from the cleaned `hip_port`.
