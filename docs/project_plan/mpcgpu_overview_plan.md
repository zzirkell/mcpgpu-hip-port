# MPCGPU CUDA → HIP Porting Scope and Test Plan

Goal: port enough of the original MPCGPU project to run the main PCG-based example:

```text
examples/track_iiwa_pcg.cu
```

Secondary goal, useful for comparison later:

```text
examples/track_iiwa_qdldl.cu
```

This document is **not only an API list**. It is a concrete work plan for:

```text
1. what must be tested in isolation,
2. what must be tested inside the real MPCGPU repository,
3. who implements each test,
4. who depends on each result,
5. how we can work in parallel without blocking each other.
```

---

## 0. Main project structure

The main PCG example has this rough call structure:

```text
examples/track_iiwa_pcg.cu
  -> include/mpcsim.cuh
      -> SQP loop
          -> common MPC kernels
             integrator / KKT / merit / dz
          -> robot dynamics kernels
             GRiD-generated iiwa dynamics code
          -> linear system setup
             include/pcg/linsys_setup.cuh
          -> GPU PCG solver
             GBD-PCG/include/pcg.cuh
      -> simulate state
      -> shift trajectory
      -> repeat
```

Important folders:

```text
examples/                    # runnable examples
include/mpcsim.cuh            # high-level MPC simulation wrapper
include/common/               # integrator, KKT, merit, dz, settings
include/pcg/                  # MPCGPU PCG-based SQP path
include/qdldl/                # QDLDL baseline path
include/dynamics/iiwa/        # GRiD-generated robot dynamics code
include/utils/                # matrix / CSR / experiment helpers
GBD-PCG/include/              # core GPU PCG solver
GBD-PCG/GLASS/                # GLASS copy used by GBD-PCG submodule
GLASS/                        # GLASS copy used by main MPCGPU build
qdldl/                        # CPU sparse solver library
```

The port should **not** start by blindly hipifying the whole repository. The safe order is:

```text
1. collect CUDA features used by the repo,
2. test every important feature in a tiny CUDA → HIP example,
3. test real project components one by one,
4. only then try the full example.
```

---

## 1. Static scan result

A source scan over the uploaded MPCGPU project found the following CUDA/HIP-relevant usage.

Approximate scanned size:

```text
79 source/header files
~21,053 lines total
```

Summary:

| Category | Unique items | Total occurrences | Meaning |
|---|---:|---:|---|
| CUDA runtime names | 30 | 877 | memory, streams, cooperative launch, device queries, etc. |
| cuBLAS names | 6 | 13 | only AXPY appears, but it is important |
| cooperative groups names | 6 | 91 | block/group/grid synchronization |
| CUDA kernel keywords / built-ins | 12 | 4177 | massive use in generated dynamics code |

Most important CUDA runtime / cuBLAS tokens:

```text
cudaDeviceSynchronize                         139
cudaMemcpy                                    118
cudaMalloc                                    114
cudaFree                                       97
cudaMemcpyHostToDevice                         85
cudaMemcpyDeviceToHost                         78
cudaMemcpyAsync                                58
cudaStream_t                                   52
cudaPeekAtLastError                            34
cudaFuncSetAttribute                           17
cudaFuncAttributeMaxDynamicSharedMemorySize    17
cudaMemcpyDeviceToDevice                       14
cudaMemset                                      7
cudaLaunchCooperativeKernel                     5
cudaStreamDestroy                               4
cudaDeviceGetStreamPriorityRange                2
cudaStreamCreateWithPriority                    2
cudaStreamCreate                                2
cudaMemcpy2D                                    1
cudaOccupancyMaxActiveBlocksPerMultiprocessor   1
cudaGetDeviceProperties                         1
cudaDeviceGetAttribute                          1

cublasCreate                                    2
cublasSaxpy                                     2
cublasDaxpy                                     2
cublasDestroy                                   2
```

Most important CUDA language / device-code tokens:

```text
blockDim                                      1582
threadIdx                                     1089
__syncthreads                                  607
__device__                                     263
__shared__                                     142
extern __shared__                               67
__host__                                       101
__global__                                      95
dim3                                           116
gridDim                                         62
blockIdx                                        51
atomicAdd                                        2
```

Conclusion:

```text
The port is not just replacing cudaMalloc with hipMalloc.
The hard parts are cooperative kernels, grid-wide sync, dynamic shared memory,
cuBLAS/HIPBLAS, GLASS device math, and the large GRiD-generated dynamics code.
```

---

## 2. Key ownership decision

There are two different meanings of ownership:

| Term | Meaning |
|---|---|
| **Implementer** | Person who creates the isolated test, hipifies it, runs it, and documents the result. |
| **Needed by** | Person who later depends on this test result. |

Some foundation tests are needed by both people, but they do **not** need to be implemented twice.

Tobias is **not blocked** by PCG. He cannot run the full `track_iiwa_pcg` end-to-end until the PCG path is ported, but he can independently test dynamics, common kernels, streams, shared memory, utilities, and the NMPC wrapper.

---

## 3. High-level division

### 3.1 Masha: core PCG / solver / GPU math path

Masha owns the solver core:

```text
GBD-PCG/include/*
GBD-PCG/GLASS/*
GLASS/*
include/pcg/linsys_setup.cuh
include/pcg/sqp.cuh
```

Main responsibility:

```text
given a prepared linear system -> solve it on GPU with PCG
```

This includes:

```text
PCG iterations
preconditioner
Schur / linear-system setup
cooperative launch
grid.sync()
GLASS math helpers
cuBLAS -> HIPBLAS replacement
```

### 3.2 Tobias: outer NMPC loop / dynamics / generated code path

Tobias owns the high-level MPC flow and the largest generated code volume:

```text
examples/track_iiwa_pcg.cu
include/mpcsim.cuh
include/common/*
include/dynamics/*
include/utils/*
```

Main responsibility:

```text
given robot state + reference trajectory -> build/update the MPC problem
```

This includes:

```text
NMPC simulation wrapper
SQP outer-loop orchestration
robot dynamics / GRiD-generated code
KKT submatrix generation
integrator kernels
merit / line search support
tracking error
memory copies, streams, atomics, dynamic shared memory
```

### 3.3 Grey zone: SQP

SQP is split by responsibility:

| SQP part | Owner |
|---|---|
| Host wrapper / loop structure / calls to dynamics / trajectory update | Tobias |
| `include/pcg/linsys_setup.cuh`, Schur setup, `Pinv`, PCG call preparation | Masha |
| Full `include/pcg/sqp.cuh` integration | Masha first, then joint integration |

Schur/preconditioner should stay with Masha because it prepares the reduced system that PCG solves.

---

## 4. Canonical isolated test table

Each isolated test should have this pattern where possible:

```bash
make run-cuda          # CUDA on NVIDIA
make hipify            # generate HIP code
make run-hip-nvidia    # HIP syntax, NVIDIA backend
make run-hip-amd       # HIP syntax, AMD backend
```

Success criteria:

```text
1. CUDA version compiles and runs.
2. hipify-perl produces readable HIP code.
3. HIP NVIDIA backend compiles and gives the same result.
4. HIP AMD backend compiles and gives the same result.
5. The result is checked numerically when possible.
```

### 4.1 Full test assignment table

| Test name | What this test proves | Concrete CUDA / project names | HIP / port names | Implementer | Needed by |
|---|---|---|---|---|---|
| `001_basic_memory` | Basic GPU allocation, copy, and free work. | `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice` | `hipMalloc`, `hipMemcpy`, `hipFree`, HIP copy directions | Masha | Masha + Tobias |
| `002_error_sync` | Runtime errors can be checked and CPU can wait for GPU work. | `cudaError_t`, `cudaSuccess`, `cudaGetErrorString`, `cudaPeekAtLastError`, `cudaDeviceSynchronize` | `hipError_t`, `hipSuccess`, `hipGetErrorString`, `hipPeekAtLastError`, `hipDeviceSynchronize` | Masha | Masha + Tobias |
| `003_basic_kernel_launch_indexing` | Normal kernel launches and indexing work. | `kernel<<<blocks, threads>>>`, `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, `dim3` | same HIP syntax / same names | Masha | Masha + Tobias |
| `004_cuda_language_qualifiers` | CUDA function qualifiers survive HIPIFY. | `__global__`, `__device__`, `__host__`, `__host__ __device__`, `__forceinline__` | mostly same names, check HIP support | Masha | Masha + Tobias |
| `005_shared_memory_block_sync` | Static/dynamic shared memory and block sync work. | `__shared__`, `extern __shared__`, `__syncthreads()`, `kernel<<<blocks, threads, shared_mem>>>` | same HIP syntax / same names | Masha | Masha + Tobias |
| `006_stream_async_copy` | Basic streams and async copy work. | `cudaStream_t`, `cudaStreamCreate`, `cudaMemcpyAsync`, `cudaStreamDestroy` | `hipStream_t`, `hipStreamCreate`, `hipMemcpyAsync`, `hipStreamDestroy` | Tobias | Masha + Tobias |
| `007_stream_priority` | Priority streams can be queried/created. | `cudaDeviceGetStreamPriorityRange`, `cudaStreamCreateWithPriority`, `cudaStreamNonBlocking` | HIP equivalents | Tobias | Tobias |
| `008_memset_memcpy2d` | GPU memory set and 2D copies work. | `cudaMemset`, `cudaMemcpy2D` | `hipMemset`, `hipMemcpy2D` | Tobias | Masha + Tobias |
| `009_device_props_occupancy` | Device properties, cooperative support, and occupancy calculation work. | `cudaDeviceProp`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, `cudaDevAttrCooperativeLaunch`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `hipDeviceProp_t`, `hipGetDeviceProperties`, `hipDeviceGetAttribute`, HIP cooperative attr, `hipOccupancyMaxActiveBlocksPerMultiprocessor` | Masha | Masha |
| `010_cooperative_groups_block` | Block-level cooperative groups work. | `cooperative_groups`, `cooperative_groups::this_thread_block()`, `cooperative_groups::thread_block`, `cooperative_groups::thread_group`, `block.sync()` | same or HIP-supported equivalent | Masha | Masha + Tobias |
| `011_cooperative_groups_grid_sync` | Full-grid cooperative group and `grid.sync()` work. | `cooperative_groups::this_grid()`, `grid.sync()` | same if supported | Masha | Masha |
| `012_cooperative_kernel_launch` | Cooperative kernel launch works from host. | `cudaLaunchCooperativeKernel` | `hipLaunchCooperativeKernel` or workaround | Masha | Masha |
| `013_func_set_attribute_dynamic_smem` | Kernel function attributes and large dynamic shared memory work. | `cudaFuncSetAttribute`, `cudaFuncAttributeMaxDynamicSharedMemorySize` | `hipFuncSetAttribute`, `hipFuncAttributeMaxDynamicSharedMemorySize` or workaround | Tobias | Tobias |
| `014_atomic_add` | Atomic accumulation works. | `atomicAdd` | same / check type support | Tobias | Tobias |
| `015_hip_headers` | Header translation strategy is correct. | `cuda_runtime.h`, `cuda_runtime_api.h`, `cublas_v2.h` | `hip/hip_runtime.h`, HIP runtime API, `hipblas/hipblas.h` | Masha | Masha + Tobias |
| `016_blas_axpy` | BLAS AXPY replacement works. | `cublasHandle_t`, `cublasCreate`, `cublasSaxpy`, `cublasDaxpy`, `cublasDestroy` | `hipblasHandle_t`, `hipblasCreate`, `hipblasSaxpy`, `hipblasDaxpy`, `hipblasDestroy` | Masha | Masha |
| `017_glass_l1_helpers` | GLASS L1 vector helpers work. | `axpy`, `axpby`, `copy`, `dot`, `reduce`, `scal`, `swap`, `loadIdentity`, `addI`, `clip`, `set_const`, `infnorm`, `l2norm` | hipified device functions | Masha | Masha |
| `018_glass_l2_l3_helpers` | GLASS matrix helpers work. | `gemv`, `gemm`, `invertMatrix`, `cholDecomp_InPlace_c`, `chol_InPlace` | hipified device functions | Masha | Masha |
| `019_pcg_core_kernel` | Main PCG kernel compiles and can be called in isolation. | `pcg`, `solvePCG` | hipified kernel/function | Masha | Masha |
| `020_pcg_linsys_setup` | PCG linear-system setup runs on small fake input. | `form_S_gamma_Pinv_kernel` | hipified kernel | Masha | Masha |
| `021_pcg_sqp_skeleton` | PCG-side SQP path compiles enough to call solver-facing code. | `sqpSolvePcg` / PCG SQP functions | hipified function path | Masha | Masha + Tobias |
| `022_kkt_dz_common_kernels` | KKT and dz common kernels compile/run with tiny input. | `generate_kkt_submatrices`, `compute_dz_kernel` | hipified kernels | Tobias | Tobias |
| `023_integrator_kernels` | Integrator kernels compile/run with tiny input. | `integrator_kernel`, `simple_integrator_kernel` | hipified kernels | Tobias | Tobias |
| `024_merit_tracking_kernels` | Merit, line-search merit, and tracking error kernels compile/run. | `ls_gato_compute_merit`, `compute_merit`, `compute_tracking_error_kernel` | hipified kernels | Tobias, Masha reviews cooperative part | Masha + Tobias |
| `025_utils_csr_matrix` | Utility kernels and matrix operations compile/run. | `prep_csr`, matrix helpers, `cudaMemcpy2D` usage | hipified utilities | Tobias | Tobias |
| `026_grid_dynamics_core` | Core GRiD-generated dynamics kernels compile/run. | `inverse_dynamics_kernel`, `direct_minv_kernel`, `forward_dynamics_kernel` | hipified kernels | Tobias | Tobias |
| `027_grid_dynamics_gradients` | GRiD-generated gradient kernels compile/run. | `inverse_dynamics_gradient_kernel`, `forward_dynamics_gradient_kernel` | hipified kernels | Tobias | Tobias |
| `028_grid_end_effector` | GRiD-generated end-effector kernels compile/run. | `end_effector_positions_kernel`, `end_effector_positions_gradient_kernel` | hipified kernels | Tobias | Tobias |
| `029_qdldl_secondary` | QDLDL alternative path compiles/runs if needed later. | `track_iiwa_qdldl.cu`, `form_schur_qdl_kernel`, `include/qdldl/*` | hipified QDLDL path | Later / optional | Later / optional |
| `030_full_track_iiwa_pcg` | Full PCG NMPC example compiles and runs end-to-end. | `examples/track_iiwa_pcg.cu` full path | full HIP path | Both | Masha + Tobias |

---

## 5. Short implementer summary

| Implementer | Tests |
|---|---|
| **Masha first, shared result** | `001`, `003`, `004`, `005`, `015` |
| **Masha** | `002`, `009`, `010`, `011`, `012`, `016`, `017`, `018`, `019`, `020`, `021` |
| **Tobias** | `006`, `007`, `008`, `013`, `014`, `022`, `023`, `024`, `025`, `026`, `027`, `028` |
| **Both together** | `030` |
| **Later / optional** | `029` |

Important interpretation:

```text
The shared foundation tests are implemented by Masha because she starts earlier and needs them for PCG.
Tobias depends on them but does not need to repeat them.
```

---

## 6. Real-repository test plan

The isolated tests prove that individual CUDA/HIP features work. After that, we need to test real MPCGPU components.

### 6.1 Masha real-repo path

Masha should not jump directly from tiny API tests to the full MPCGPU example.

Order:

```text
Foundation tests
  -> cooperative kernel tests
  -> BLAS replacement test
  -> GLASS tests
  -> GBD-PCG standalone example
  -> solvePCG tiny interface test
  -> PCG linsys setup test
  -> PCG SQP compile test
  -> full track_iiwa_pcg.cu integration
```

Concrete real-repo tests:

| ID | Test | Purpose |
|---|---|---|
| `M1` | Compile hipified `GLASS/GTests/test.cu` or equivalent minimal GLASS test | Checks real GLASS include path and helper functions. |
| `M2` | Compile hipified `GBD-PCG/examples/pcg_solve.cu` | First real PCG-only compile test. |
| `M3` | Run GBD-PCG PCG example on HIP NVIDIA backend | Tests PCG before AMD-specific issues. |
| `M4` | Run GBD-PCG PCG example on HIP AMD backend | Tests actual AMD target. |
| `M5` | Create minimal `solvePCG` interface test | Tests `GBD-PCG/include/interface.cuh` without full robot stack. |
| `M6` | Compile `include/pcg/linsys_setup.cuh` in isolation | Tests Schur / preconditioner setup compile. |
| `M7` | Run tiny `form_S_gamma_Pinv_kernel` test | Tests PCG linear-system setup with fake small input. |
| `M8` | Compile `include/pcg/sqp.cuh` after cuBLAS/HIPBLAS replacement | Tests solver-facing SQP path. |
| `M9` | Join full `track_iiwa_pcg.cu` integration | Final integration with Tobias's outer path. |

The first meaningful real-repo target for Masha is:

```text
GBD-PCG/examples/pcg_solve.cu
```

Why:

```text
- It avoids the whole robot dynamics stack.
- It tests the actual PCG solver.
- It tests cooperative launch and grid-wide sync.
- It tests GLASS inside the real PCG code.
```

If this does not work, the full MPCGPU example cannot work.

### 6.2 Tobias real-repo path

Tobias can test his part without waiting for the full PCG path.

Order:

```text
Foundation tests already available from Masha
  -> streams / memset / memcpy2D / function attributes / atomics
  -> utils/matrix + csr tests
  -> common kernel tests
  -> generated dynamics compile-only
  -> generated dynamics small runtime tests
  -> integrator test
  -> mpcsim compile test
  -> full track_iiwa_pcg.cu integration
```

Concrete real-repo tests:

| ID | Test | Purpose |
|---|---|---|
| `T1` | Hipify and compile `include/utils/matrix.cuh` test | Checks matrix utility and `cudaMemcpy2D` usage. |
| `T2` | Hipify and compile `include/common/dz.cuh` test | Checks trajectory update kernel. |
| `T3` | Hipify and compile `include/common/kkt.cuh` test | Checks KKT submatrix generation. |
| `T4` | Hipify and compile `include/dynamics/iiwa/iiwa_grid.cuh` compile-only test | Checks large generated dynamics file. |
| `T5` | Hipify and compile `include/dynamics/iiwa/iiwa_eepos_grid.cuh` compile-only test | Checks large generated eepos dynamics file. |
| `T6` | Run tiny `end_effector_positions_kernel` test | First generated dynamics runtime test. |
| `T7` | Run tiny `forward_dynamics_kernel` test | Tests dynamics compute path. |
| `T8` | Run tiny `integrator_kernel` test | Tests dynamics through integrator. |
| `T9` | Run tiny `compute_merit` test | Tests merit/cost path. |
| `T10` | Compile `include/mpcsim.cuh` after common + dynamics work | Tests high-level wrapper without final PCG integration. |
| `T11` | Join full `track_iiwa_pcg.cu` integration | Final integration with Masha's PCG path. |

The first meaningful real-repo target for Tobias is a tiny program that includes:

```cpp
#include "dynamics/rbd_plant.cuh"
```

and calls one generated dynamics function with small/dummy input.

Why:

```text
- It avoids PCG while testing the generated GRiD code.
- It catches shared-memory, stream, and cudaFuncSetAttribute issues early.
- It proves the largest source files are portable before full integration.
```

Important cross-boundary item:

```text
include/common/merit.cuh uses cooperative_groups::this_grid().sync()
```

So Tobias owns the file, but Masha should review the cooperative-launch part.

---

## 7. Full integration order

Recommended order:

```text
Step 1: CUDA baseline still runs.
Step 2: Foundation isolated tests pass.
Step 3: HIPIFY tiny tests run on NVIDIA and AMD.
Step 4: Cooperative groups tests pass on NVIDIA and AMD.
Step 5: HIPBLAS AXPY replacement works.
Step 6: GLASS tests pass on NVIDIA and AMD.
Step 7: GBD-PCG standalone runs on NVIDIA and AMD.
Step 8: Generated dynamics compile and small runtime tests pass.
Step 9: Common MPC kernels pass small tests.
Step 10: include/pcg/sqp.cuh compiles with HIPBLAS replacement.
Step 11: examples/track_iiwa_pcg.cu compiles.
Step 12: examples/track_iiwa_pcg.cu runs and produces tracking results.
Step 13: Compare CUDA result vs HIP NVIDIA result vs HIP AMD result.
```

Expected final comparison file:

```text
docs/results_comparison.md
```

It should record:

```text
CUDA NVIDIA output
HIP NVIDIA output
HIP AMD output
tracking error stats
runtime if measured
known warnings
known numerical differences
```

---

## 8. What is not first priority

Useful, but not first-week blockers:

```text
examples/track_iiwa_qdldl.cu
include/qdldl/*
qdldl/ full cleanup
Docker finalization
performance optimization
pretty architecture refactor
```

Reason:

```text
The main requested path is PCG. QDLDL is useful as a baseline, but it can be handled after the PCG path starts compiling.
```

---

## 10. Biggest risks

| Risk | Why it matters | Owner |
|---|---|---|
| Cooperative grid sync | PCG correctness depends on `grid.sync()` | Masha |
| Cooperative launch support on AMD backend | `hipLaunchCooperativeKernel` may behave differently or have stricter limits | Masha |
| Occupancy calculation | PCG checks whether all blocks can be resident | Masha |
| Large dynamic shared memory | Generated dynamics and PCG use dynamic shared memory | Masha + Tobias |
| `cudaFuncSetAttribute` mapping | Generated GRiD code relies on it | Tobias |
| cuBLAS to HIPBLAS | SQP update uses AXPY | Masha |
| Two GLASS copies | Easy to port one copy but include another | Masha |
| Generated GRiD source size | Largest amount of CUDA syntax and shared-memory code | Tobias |
| WSL AMD backend warnings | May be driver/setup-related, not code-related | Shared |

---

## 11. Commands to keep for repeated scans

Use these from the original MPCGPU root.

List CUDA/HIP-relevant external API calls:

```bash
grep -R "cuda[A-Za-z0-9_]*\|cublas[A-Za-z0-9_]*" -n \
  examples include GBD-PCG GLASS \
  --include='*.cu' --include='*.cuh' --include='*.h' --include='*.cpp' \
  | sort
```

List CUDA language/device features:

```bash
grep -R "__global__\|__device__\|__host__\|__shared__\|__syncthreads\|threadIdx\|blockIdx\|blockDim\|gridDim\|atomicAdd\|cooperative_groups" -n \
  examples include GBD-PCG GLASS \
  --include='*.cu' --include='*.cuh' --include='*.h' --include='*.cpp' \
  | sort
```

Find cooperative-kernel launch sites:

```bash
grep -R "cudaLaunchCooperativeKernel\|this_grid\|grid.sync" -n \
  examples include GBD-PCG GLASS \
  --include='*.cu' --include='*.cuh'
```

Find cuBLAS usage:

```bash
grep -R "cublas" -n \
  examples include GBD-PCG GLASS \
  --include='*.cu' --include='*.cuh' --include='*.h'
```

Keep these commands in the repo because they make the scope reproducible.

---

## 12. Summary

```text
We should split by architecture, not by random files.

Masha starts by implementing the shared foundation tests, because she starts earlier
and needs them for the PCG port. Then she owns the core solver path: GBD-PCG,
cooperative kernels, GLASS, PCG linear-system setup, PCG SQP, and cuBLAS/HIPBLAS.

Tobias reuses the shared foundation tests and owns the outer MPC / robot path:
examples/track_iiwa_pcg.cu, mpcsim.cuh, common kernels, generated GRiD dynamics,
and matrix/CSR utilities.

Tobias is not blocked by PCG: he can test dynamics/common/wrapper components independently.
The full track_iiwa_pcg example is the final joint integration target.
```
