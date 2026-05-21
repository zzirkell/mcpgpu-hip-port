# MPCGPU CUDA → HIP porting scope and test plan

Goal: port enough of the original MPCGPU project to run:

```text
examples/track_iiwa_pcg.cu
```

Secondary goal, useful for comparison later:

```text
examples/track_iiwa_qdldl.cu
```

This document is not only an API list. It is a work plan for deciding what must be tested in isolation, what must be tested inside the real repository, and how the work is divided.

---

## 0. Current understanding of the project

The main PCG example has this rough structure:

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

Structure and Important folders:

```text
examples/                    # runnable examples
include/mpcsim.cuh            # high-level MPC simulation wrapper
include/common/               # integrator, KKT, merit, dz, settings
include/pcg/                  # MPCGPU PCG-based SQP path
include/qdldl/                # QDLDL baseline path
include/dynamics/iiwa/        # GRiD-generated robot dynamics code
include/utils/                # matrix/CSR/experiment helpers
GBD-PCG/include/              # core GPU PCG solver
GBD-PCG/GLASS/                # GLASS copy used by GBD-PCG submodule
GLASS/                        # GLASS copy used by main MPCGPU build
qdldl/                        # CPU sparse solver library
```

The port should not start by blindly hipifying the full repository. The safer order is:

```text
1. collect CUDA features used by the repo
2. test every feature in a tiny isolated CUDA → HIP example
3. test real project components one by one
4. only then try the full example
```

---

## 1. Static scan result

A full source scan was done over `.cu`, `.cuh`, `.c`, `.h`, `.cpp` files from the uploaded MPCGPU project, excluding `.git` internals.

Approximate scanned size:

```text
79 source/header files
~20,974 lines total
```

CUDA/HIP-relevant occurrence summary:

| Category | Unique items | Total occurrences | Meaning |
|---|---:|---:|---|
| CUDA runtime API names | 30 | 877 | memory, streams, cooperative launch, device queries, etc. |
| cuBLAS API names | 6 | 13 | only AXPY appears, but it is important |
| cooperative groups names | 5 | 50 | block/group/grid synchronization |
| CUDA kernel keywords / built-ins | 10 | 4061 | massive use in generated dynamics code |

Most important tokens found:

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

__global__                                     95
__device__                                    263
__host__                                      101
__shared__                                    209
__syncthreads                                 607
threadIdx                                    1089
blockIdx                                      51
blockDim                                    1582
gridDim                                       62
atomicAdd                                      2
```

Conclusion: the project is not only a small set of external API calls. A lot of the work is also about whether generated CUDA kernels, cooperative groups, dynamic shared memory, GLASS device helpers, and PCG synchronization survive HIP correctly.

---

## 2. What must be tested

There are two kinds of things to test.

### 2.1 External CUDA/HIP API calls

These are explicit runtime/library functions such as:

```text
cudaMalloc
cudaMemcpy
cudaLaunchCooperativeKernel
cublasSaxpy
...
```

For these, isolated tests are straightforward: write a small `.cu`, run it with CUDA, hipify it, compile with HIP for NVIDIA and AMD, compare output.

### 2.2 CUDA language/device-code features

These are not normal API calls, but they are still critical:

```text
__global__
__device__
__host__
__shared__
extern __shared__
__syncthreads()
threadIdx / blockIdx / blockDim / gridDim
cooperative_groups::this_thread_block()
cooperative_groups::this_grid()
grid.sync()
atomicAdd
kernel<<<blocks, threads, shared_mem, stream>>>()
```

These must also be tested. If an API translates correctly but the kernel synchronization or shared memory behavior is wrong, the full MPCGPU example can still fail.

---

## 3. Complete feature inventory for porting

### Group A — basic memory management

| CUDA feature | Expected HIP direction | Main project locations | Owner |
|---|---|---|---|
| `cudaMalloc` | `hipMalloc` | dynamics, PCG SQP, QDLDL SQP, GBD-PCG interface, GLASS tests | Masha + Tobias baseline |
| `cudaFree` | `hipFree` | same as above | Masha + Tobias baseline |
| `cudaMemcpy` | `hipMemcpy` | dynamics, `mpcsim.cuh`, GBD-PCG interface, SQP | Masha + Tobias baseline |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | examples, dynamics, GLASS tests | Masha + Tobias baseline |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | dynamics, `mpcsim.cuh`, GLASS tests, SQP | Masha + Tobias baseline |
| `cudaMemcpyDeviceToDevice` | `hipMemcpyDeviceToDevice` | `mpcsim.cuh`, integrator, SQP | Tobias |
| `cudaMemset` | `hipMemset` | `mpcsim.cuh`, SQP | Tobias |
| `cudaMemcpy2D` | `hipMemcpy2D` | `include/utils/matrix.cuh` | Tobias |

Isolated tests:

```text
001_basic_malloc_memcpy_free
002_memset
003_device_to_device_copy
004_memcpy2d_matrix_copy
```

Real repo tests:

```text
compile and run tiny wrapper using include/utils/matrix.cuh
compile and run basic allocation path from include/pcg/sqp.cuh or include/qdldl/sqp.cuh
```

---

### Group B — synchronization and error handling

| CUDA feature | Expected HIP direction | Main project locations | Owner |
|---|---|---|---|
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | everywhere | Shared |
| `cudaPeekAtLastError` | `hipPeekAtLastError` | SQP, `mpcsim.cuh`, GBD-PCG interface, GLASS tests | Shared |
| `cudaError_t` | `hipError_t` | error macros | Masha |
| `cudaSuccess` | `hipSuccess` | error macros | Masha |
| `cudaGetErrorString` | `hipGetErrorString` | error macros | Masha |

Isolated tests:

```text
005_kernel_error_checking
006_gpuErrchk_macro_cuda_to_hip
```

Real repo tests:

```text
port GBD-PCG/include/gpuassert.cuh
make sure every failed HIP call prints useful file/line information
```

---

### Group C — normal kernel launches and indexing

| CUDA feature | Expected HIP direction | Main project locations | Owner |
|---|---|---|---|
| `kernel<<<blocks, threads>>>()` | same launch syntax usually works with HIP | all kernels | Tobias |
| `kernel<<<blocks, threads, smem>>>()` | same syntax | dynamics, common kernels, GLASS | Tobias |
| `kernel<<<blocks, threads, smem, stream>>>()` | same syntax | dynamics, streams, SQP | Tobias |
| `threadIdx`, `blockIdx`, `blockDim`, `gridDim` | same names | generated GRiD code, GLASS, common kernels | Tobias |
| grid-stride loops | same logic | many kernels | Tobias |

Isolated tests:

```text
007_normal_kernel_launch
008_grid_stride_loop
009_2d_block_thread_indexing
```

Real repo tests:

```text
compile selected generated GRiD kernels
compile common MPC kernels: dz, kkt, integrator, merit
```

---

### Group D — shared memory and block synchronization

| CUDA feature | Expected HIP direction | Main project locations | Owner |
|---|---|---|---|
| `__shared__` static memory | same keyword | dynamics, GLASS, merit, integrator | Tobias |
| `extern __shared__` dynamic memory | same keyword | PCG, linsys setup, merit, dz, integrator | Masha + Tobias |
| `__syncthreads()` | same call | dynamics, PCG, GLASS, linsys setup | Masha + Tobias |
| `cooperative_groups::this_thread_block()` | likely similar but must test | GLASS, merit, integrator, PCG utils | Masha + Tobias |
| `block.sync()` | likely similar but must test | GLASS, merit, integrator | Masha + Tobias |

Isolated tests:

```text
010_static_shared_memory_reduce
011_dynamic_shared_memory_reduce
012_thread_block_sync
```

Real repo tests:

```text
Masha: GLASS reduce/dot/copy tests
Tobias: dynamics kernels using static shared arrays
```

---

### Group E — streams and async copies

| CUDA feature | Expected HIP direction | Main project locations | Owner |
|---|---|---|---|
| `cudaStream_t` | `hipStream_t` | generated dynamics, SQP | Tobias |
| `cudaStreamCreate` | `hipStreamCreate` | SQP | Tobias |
| `cudaStreamDestroy` | `hipStreamDestroy` | dynamics, SQP | Tobias |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | generated dynamics, SQP | Tobias |
| `cudaDeviceGetStreamPriorityRange` | HIP equivalent, must test | generated dynamics | Tobias |
| `cudaStreamCreateWithPriority` | HIP equivalent, must test | generated dynamics | Tobias |
| `cudaStreamNonBlocking` | HIP equivalent, must test | generated dynamics | Tobias |

Isolated tests:

```text
013_basic_stream_async_copy
014_stream_kernel_launch
015_priority_stream_create_destroy
```

Real repo tests:

```text
compile and run one generated dynamics host wrapper that creates streams
check whether priority streams are actually supported on AMD backend
```

---

### Group F — function attributes and large dynamic shared memory

| CUDA feature | Expected HIP direction | Main project locations | Owner |
|---|---|---|---|
| `cudaFuncSetAttribute` | `hipFuncSetAttribute` or manual workaround | generated dynamics, GLASS timing test | Tobias |
| `cudaFuncAttributeMaxDynamicSharedMemorySize` | HIP equivalent or workaround | generated dynamics, GLASS timing test | Tobias |

This is a risk area because generated GRiD code sets large dynamic shared-memory limits before launching kernels.

Isolated tests:

```text
016_func_set_attribute_dynamic_smem
017_large_dynamic_shared_memory_kernel
```

Real repo tests:

```text
compile iiwa_grid.cuh / iiwa_eepos_grid.cuh after hipify
run one simple dynamics function using real generated code
```

---

### Group G — cooperative kernels and grid-wide synchronization

| CUDA feature | Expected HIP direction | Main project locations | Owner |
|---|---|---|---|
| `cudaLaunchCooperativeKernel` | `hipLaunchCooperativeKernel` or workaround | PCG, linsys setup, SQP, merit, QDLDL SQP | Masha |
| `cooperative_groups::this_grid()` | HIP cooperative groups support must be tested | `GBD-PCG/include/pcg.cuh`, `include/common/merit.cuh` | Masha |
| `grid.sync()` | must work for PCG correctness | `GBD-PCG/include/pcg.cuh` | Masha |
| cooperative launch capability query | HIP equivalent required | `GBD-PCG/include/pcg.cuh` | Masha |

Exact locations found:

```text
GBD-PCG/include/interface.cuh      cudaLaunchCooperativeKernel
include/pcg/linsys_setup.cuh       cudaLaunchCooperativeKernel
include/pcg/sqp.cuh                cudaLaunchCooperativeKernel
include/qdldl/sqp.cuh              cudaLaunchCooperativeKernel
include/common/merit.cuh           cooperative_groups::this_grid().sync()
GBD-PCG/include/pcg.cuh            grid.sync() inside PCG loop
```

This is one of the biggest port risks. PCG uses grid-wide synchronization between blocks. This is not the same as `__syncthreads()`. `__syncthreads()` only synchronizes threads inside one block. `grid.sync()` synchronizes blocks in the same cooperative kernel launch.

Isolated tests:

```text
018_cooperative_kernel_launch_basic
019_grid_sync_counter
020_grid_sync_two_phase_array_update
```

Real repo tests:

```text
GBD-PCG/examples/pcg_solve.cu       # first real PCG-only test
GBD-PCG/examples/pcg_solve_dp.cu    # double precision PCG-only test, if needed
GBD-PCG/include/interface.cuh       # call solvePCG directly from tiny test
include/pcg/linsys_setup.cuh        # test form_S_gamma_Pinv_kernel separately
include/pcg/sqp.cuh                 # test SQP-level PCG call after linsys works
```

---

### Group H — device properties and occupancy

| CUDA feature | Expected HIP direction | Main project locations | Owner |
|---|---|---|---|
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` | `GBD-PCG/include/pcg.cuh` | Masha |
| `cudaDeviceGetAttribute` | `hipDeviceGetAttribute` | `GBD-PCG/include/pcg.cuh` | Masha |
| `cudaDevAttrCooperativeLaunch` | HIP attr equivalent | `GBD-PCG/include/pcg.cuh` | Masha |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `hipOccupancyMaxActiveBlocksPerMultiprocessor` | `GBD-PCG/include/pcg.cuh` | Masha |

Isolated tests:

```text
021_device_properties_print
022_cooperative_launch_attribute_check
023_occupancy_calculation_for_dummy_kernel
```

Real repo tests:

```text
port and run checkPcgOccupancy() from GBD-PCG/include/pcg.cuh
verify it does not reject AMD incorrectly
verify it chooses a valid launch configuration
```

---

### Group I — cuBLAS to HIPBLAS / rocBLAS

| CUDA feature | Expected HIP direction | Main project locations | Owner |
|---|---|---|---|
| `cublasHandle_t` | `hipblasHandle_t` | `include/pcg/sqp.cuh`, `include/qdldl/sqp.cuh` | Masha |
| `cublasCreate` | `hipblasCreate` | same | Masha |
| `cublasSaxpy` | `hipblasSaxpy` | same | Masha |
| `cublasDaxpy` | `hipblasDaxpy` | same | Masha |
| `cublasDestroy` | `hipblasDestroy` | same | Masha |
| `<cublas_v2.h>` | `<hipblas/hipblas.h>` or chosen wrapper | `mpcsim.cuh`, SQP files | Masha |

Only AXPY is used, but it is inside the SQP path, so it matters.

Isolated tests:

```text
024_cublas_saxpy_to_hipblas_saxpy
025_cublas_daxpy_to_hipblas_daxpy
026_single_source_axpy_wrapper
```

Real repo tests:

```text
port cublas usage in include/pcg/sqp.cuh
compile SQP path without full example first
then run full example
```

Preferred strategy:

```text
Do not scatter #ifdefs everywhere.
Create a small compatibility wrapper later, for example:
include/compat/gpu_blas_compat.h
```

The wrapper can hide whether the backend is CUDA/cuBLAS or HIP/HIPBLAS.

---

### Group J — GLASS device math helpers

GLASS is not an external API, but it is core device math used by PCG, linsys setup, integrator, dynamics, and utilities.

Important files:

```text
GLASS/glass.cuh
GLASS/src/L1/axpy.cuh
GLASS/src/L1/clip.cuh
GLASS/src/L1/copy.cuh
GLASS/src/L1/dot.cuh
GLASS/src/L1/ident.cuh
GLASS/src/L1/infnorm.cuh
GLASS/src/L1/l2norm.cuh
GLASS/src/L1/reduce.cuh
GLASS/src/L1/scal.cuh
GLASS/src/L1/set_const.cuh
GLASS/src/L1/swap.cuh
GLASS/src/L2/gemv.cuh
GLASS/src/L3/chol_InPlace.cuh
GLASS/src/L3/gemm.cuh
GLASS/src/L3/inv.cuh
```

Also present inside submodule:

```text
GBD-PCG/GLASS/glass.cuh
GBD-PCG/GLASS/src/L1/*
GBD-PCG/GLASS/src/L2/*
GBD-PCG/GLASS/src/L3/*
```

Risk: there are two GLASS copies. The include path decides which one is used. The main Makefile used both:

```text
-IGLASS -IGBD-PCG/include
```

So the top-level `GLASS/` copy is likely the one included by `glass.cuh` in the main MPCGPU build. But GBD-PCG also has its own nested copy. We should keep this visible and not silently port only one copy.

Isolated tests:

```text
027_glass_copy
028_glass_scal_axpy
029_glass_reduce
030_glass_dot
031_glass_gemv
032_glass_gemm
033_glass_matrix_inverse_small
```

Real repo tests:

```text
first compile GLASS/GTests/test.cu after hipify
then compile only the GLASS functions used by PCG
then compile GBD-PCG examples
```

Owner: Masha.

---

### Group K — generated GRiD robot dynamics code

These files are huge compared with the rest of the project:

```text
include/dynamics/iiwa/iiwa_grid.cuh          ~4842 lines
include/dynamics/iiwa/iiwa_eepos_grid.cuh    ~5731 lines
include/dynamics/iiwa/iiwa_plant.cuh          ~332 lines
include/dynamics/iiwa/iiwa_eepos_plant.cuh    ~406 lines
include/dynamics/rbd_plant.cuh                  5 lines
```

This is the biggest source-volume area. It contains many kernels and many shared-memory sections. Even if it was generated by GRiD, the generated CUDA output still has to compile and run under HIP for the final example.

Important kernel families:

```text
inverse_dynamics_kernel
inverse_dynamics_gradient_kernel
direct_minv_kernel
forward_dynamics_kernel
forward_dynamics_gradient_kernel
end_effector_positions_kernel
end_effector_positions_gradient_kernel
```

Important CUDA features inside this area:

```text
__global__ / __device__ / __host__
threadIdx / blockIdx / blockDim / gridDim
__shared__
__syncthreads()
cudaMalloc / cudaMemcpy / cudaFree
cudaMemcpyAsync
cudaStream_t
cudaDeviceGetStreamPriorityRange
cudaStreamCreateWithPriority
cudaFuncSetAttribute
cudaFuncAttributeMaxDynamicSharedMemorySize
```

Isolated tests:

```text
034_generated_kernel_compile_only
035_forward_dynamics_small_run
036_end_effector_positions_small_run
037_inverse_dynamics_small_run
038_streamed_dynamics_call
039_large_shared_memory_dynamics_call
```

Real repo tests:

```text
port include/dynamics/iiwa/*
compile a tiny program that includes rbd_plant.cuh
call one plant wrapper with small dummy input
then call through integrator.cuh
then call through mpcsim.cuh
```

Owner: Tobias.

---

### Group L — common MPC kernels

Files:

```text
include/common/dz.cuh
include/common/integrator.cuh
include/common/kkt.cuh
include/common/merit.cuh
include/common/settings.cuh
```

Main kernels:

```text
compute_dz_kernel
integrator_kernel
simple_integrator_kernel
generate_kkt_submatrices
ls_gato_compute_merit
compute_merit
```

Important note: `include/common/merit.cuh` is mostly Tobias's outer-loop/common area, but it contains a grid-wide cooperative sync in `ls_gato_compute_merit`. That part overlaps with Masha's cooperative-kernel risk area.

Isolated tests:

```text
040_compute_dz_small
041_integrator_compile_and_small_run
042_kkt_submatrices_small
043_compute_merit_small
044_ls_merit_cooperative_launch_small
```

Real repo tests:

```text
compile common kernels after hipify
run them with tiny dimensions
then integrate them into mpcsim.cuh
```

Owner: Tobias, with Masha reviewing cooperative launch in `merit.cuh`.

---

### Group M — PCG/SQP core path

Files:

```text
include/pcg/linsys_setup.cuh
include/pcg/sqp.cuh
GBD-PCG/include/constants.cuh
GBD-PCG/include/gpu_pcg.cuh
GBD-PCG/include/gpuassert.cuh
GBD-PCG/include/interface.cuh
GBD-PCG/include/pcg.cuh
GBD-PCG/include/types.cuh
GBD-PCG/include/utils.cuh
```

This contains the actual GPU PCG solver and the PCG-based SQP path. It is the logical core of the project.

Main risky features:

```text
cudaLaunchCooperativeKernel
cooperative_groups::this_grid()
grid.sync()
extern __shared__
GLASS dot/reduce/copy/gemv/gemm helpers
cudaOccupancyMaxActiveBlocksPerMultiprocessor
cudaGetDeviceProperties
cudaDeviceGetAttribute
cublasSaxpy / cublasDaxpy
```

Isolated tests:

```text
045_pcg_like_grid_sync_kernel
046_block_tridiagonal_vector_load_store
047_glass_inside_cooperative_kernel
048_occupancy_plus_cooperative_launch
049_hipblas_axpy_inside_sqp_style_update
```

Real repo tests, in order:

```text
1. hipify/port GBD-PCG/include/*
2. build GBD-PCG/examples/pcg_solve.cu equivalent
3. build GBD-PCG/examples/pcg_solve_dp.cu equivalent, optional
4. create tiny test calling solvePCG from GBD-PCG/include/interface.cuh
5. port include/pcg/linsys_setup.cuh
6. create tiny test for form_S_gamma_Pinv_kernel
7. port include/pcg/sqp.cuh enough to compile
8. compile full examples/track_iiwa_pcg.cu
```

Owner: Masha.

---

### Group N — QDLDL baseline path

Files:

```text
examples/track_iiwa_qdldl.cu
include/qdldl/linsys_setup.cuh
include/qdldl/sqp.cuh
qdldl/
```

This is secondary because the main goal is the PCG path. However, it is useful as a comparison path and uses many of the same outer-loop/common pieces.

Important note: `qdldl/` itself is CPU C code. It is not the main HIP problem. The CUDA/HIP work is mainly in `include/qdldl/*.cuh` and the example wrapper.

Owner: later / secondary. Could be Tobias if PCG path is already progressing.

---

## 4. Division of work

### 4.1 Masha — core PCG / solver / GPU math path

Masha owns the part where porting is most likely to fail because of cooperative grid synchronization and solver internals.

Files:

```text
GBD-PCG/include/constants.cuh
GBD-PCG/include/gpu_pcg.cuh
GBD-PCG/include/gpuassert.cuh
GBD-PCG/include/interface.cuh
GBD-PCG/include/pcg.cuh
GBD-PCG/include/types.cuh
GBD-PCG/include/utils.cuh

GBD-PCG/GLASS/glass.cuh
GBD-PCG/GLASS/src/L1/*
GBD-PCG/GLASS/src/L2/*
GBD-PCG/GLASS/src/L3/*

GLASS/glass.cuh
GLASS/src/L1/*
GLASS/src/L2/*
GLASS/src/L3/*

include/pcg/linsys_setup.cuh
include/pcg/sqp.cuh
```

Concrete isolated tests for Masha:

```text
006_gpuErrchk_macro_cuda_to_hip
018_cooperative_kernel_launch_basic
019_grid_sync_counter
020_grid_sync_two_phase_array_update
021_device_properties_print
022_cooperative_launch_attribute_check
023_occupancy_calculation_for_dummy_kernel
024_cublas_saxpy_to_hipblas_saxpy
025_cublas_daxpy_to_hipblas_daxpy
027_glass_copy
028_glass_scal_axpy
029_glass_reduce
030_glass_dot
031_glass_gemv
032_glass_gemm
033_glass_matrix_inverse_small
045_pcg_like_grid_sync_kernel
046_block_tridiagonal_vector_load_store
047_glass_inside_cooperative_kernel
048_occupancy_plus_cooperative_launch
049_hipblas_axpy_inside_sqp_style_update
```

Concrete real-repo tests for Masha:

```text
M1. compile hipified GLASS/GTests/test.cu
M2. compile hipified GBD-PCG/examples/pcg_solve.cu
M3. run GBD-PCG PCG example on NVIDIA backend
M4. run GBD-PCG PCG example on AMD backend
M5. create minimal solvePCG interface test
M6. compile include/pcg/linsys_setup.cuh in isolation
M7. run tiny form_S_gamma_Pinv_kernel test
M8. compile include/pcg/sqp.cuh after cuBLAS/HIPBLAS replacement
M9. integrate with full track_iiwa_pcg.cu after Tobias's common/dynamics path compiles
```

### 4.2 Tobias — outer MPC loop / dynamics / generated code path

Tobias owns the part with the largest generated code volume and the high-level example flow.

Files:

```text
examples/track_iiwa_pcg.cu
include/mpcsim.cuh

include/common/dz.cuh
include/common/integrator.cuh
include/common/kkt.cuh
include/common/merit.cuh
include/common/settings.cuh

include/dynamics/rbd_plant.cuh
include/dynamics/iiwa/iiwa_grid.cuh
include/dynamics/iiwa/iiwa_plant.cuh
include/dynamics/iiwa/iiwa_eepos_grid.cuh
include/dynamics/iiwa/iiwa_eepos_plant.cuh

include/utils/csr.cuh
include/utils/matrix.cuh
include/utils/experiment.cuh
```

Concrete isolated tests for Tobias:

```text
001_basic_malloc_memcpy_free
002_memset
003_device_to_device_copy
004_memcpy2d_matrix_copy
007_normal_kernel_launch
008_grid_stride_loop
009_2d_block_thread_indexing
010_static_shared_memory_reduce
011_dynamic_shared_memory_reduce
012_thread_block_sync
013_basic_stream_async_copy
014_stream_kernel_launch
015_priority_stream_create_destroy
016_func_set_attribute_dynamic_smem
017_large_dynamic_shared_memory_kernel
034_generated_kernel_compile_only
035_forward_dynamics_small_run
036_end_effector_positions_small_run
037_inverse_dynamics_small_run
038_streamed_dynamics_call
039_large_shared_memory_dynamics_call
040_compute_dz_small
041_integrator_compile_and_small_run
042_kkt_submatrices_small
043_compute_merit_small
044_ls_merit_cooperative_launch_small
```

Concrete real-repo tests for Tobias:

```text
T1. hipify and compile include/utils/matrix.cuh test
T2. hipify and compile include/common/dz.cuh test
T3. hipify and compile include/common/kkt.cuh test
T4. hipify and compile include/dynamics/iiwa/iiwa_grid.cuh compile-only test
T5. hipify and compile include/dynamics/iiwa/iiwa_eepos_grid.cuh compile-only test
T6. run tiny end_effector_positions_kernel test
T7. run tiny forward_dynamics_kernel test
T8. run tiny integrator_kernel test
T9. run tiny compute_merit test
T10. compile include/mpcsim.cuh after common + dynamics work
T11. compile examples/track_iiwa_pcg.cu after Masha's PCG path is available
```

Important cross-boundary item:

```text
include/common/merit.cuh uses cooperative_groups::this_grid().sync()
```

So Tobias owns the file, but Masha should review the cooperative-launch part.

---

## 5. Suggested repository structure for tests

Inside the porting repository:

```text
experiments/
  000_cuda_basics/
  001_api_calls_mpcgpu/
  002_cooperative_groups/
  003_glass_tests/
  004_streams_and_shared_memory/
  005_grid_generated_dynamics/
  006_pcg_core/
  007_full_repo_smoke/

ported_mpcgpu/
  # later: hipified/ported copy of the real MPCGPU source

docs/
  setup_cuda_wsl.md
  setup_hip_wsl.md
  setup_rocm_amd_wsl.md
  porting_scope_plan.md
```

Each experiment should have the same internal pattern:

```text
experiment_name/
  cuda/
    test_name.cu
  hipify_generated/
    test_name.hip.cpp
  bin/
  Makefile
  README.md
```

Each test should be runnable in three modes where possible:

```bash
make run-cuda          # CUDA on NVIDIA
make run-hip-nvidia    # HIP syntax, NVIDIA backend
make run-hip-amd       # HIP syntax, AMD backend
```

For a test to count as successful, it should satisfy all of these:

```text
1. CUDA version compiles and runs.
2. hipify-perl produces readable HIP code.
3. HIP NVIDIA backend compiles and gives same result.
4. HIP AMD backend compiles and gives same result.
5. The result is checked numerically, not only by visual output.
```

---

## 6. How to test after isolated API calls

### 6.1 Masha's path after API calls

Masha should not jump directly from tiny API tests to the full MPCGPU example. The intermediate real-repo path should be:

```text
API tests
  -> GLASS tests
  -> cooperative kernel tests
  -> GBD-PCG standalone example
  -> solvePCG tiny interface test
  -> PCG linsys setup test
  -> PCG SQP compile test
  -> full track_iiwa_pcg.cu
```

The first meaningful real-repo target for Masha is:

```text
GBD-PCG/examples/pcg_solve.cu
```

Why this is good:

```text
- It avoids the whole robot dynamics stack.
- It tests the actual PCG solver.
- It tests cooperative launch and grid-wide sync.
- It tests GLASS inside the real PCG code.
```

If this does not work, the full MPCGPU example cannot work.

### 6.2 Tobias's path after API calls

Tobias's real-repo path should be:

```text
API tests
  -> utils/matrix + csr tests
  -> common kernel tests
  -> generated dynamics compile-only
  -> generated dynamics small runtime tests
  -> integrator test
  -> mpcsim compile test
  -> full track_iiwa_pcg.cu
```

The first meaningful real-repo target for Tobias is a small program that includes:

```cpp
#include "dynamics/rbd_plant.cuh"
```

and then calls a minimal generated dynamics function with small/dummy input.

Why this is good:

```text
- It avoids PCG while testing the generated GRiD code.
- It catches shared-memory, stream, and cudaFuncSetAttribute issues early.
- It proves the largest source files are portable before full integration.
```

---

## 7. Full integration order

Recommended order:

```text
Step 1: CUDA baseline still runs.
Step 2: HIPIFY tiny tests run on NVIDIA and AMD.
Step 3: GLASS runs on NVIDIA and AMD.
Step 4: cooperative groups tests run on NVIDIA and AMD.
Step 5: GBD-PCG standalone runs on NVIDIA and AMD.
Step 6: generated dynamics compile and small runtime tests pass.
Step 7: common MPC kernels pass small tests.
Step 8: include/pcg/sqp.cuh compiles with HIPBLAS replacement.
Step 9: examples/track_iiwa_pcg.cu compiles.
Step 10: examples/track_iiwa_pcg.cu runs and produces tracking results.
Step 11: compare CUDA result vs HIP NVIDIA result vs HIP AMD result.
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

## 8. What is probably not first priority

These are useful but not first-week blockers:

```text
examples/track_iiwa_qdldl.cu
include/qdldl/*
qdldl/ full cleanup
Docker finalization
performance optimization
pretty architecture refactor
```

Reason: the main requested path is PCG. QDLDL is useful as a baseline, but it can be handled after the PCG path starts compiling.

---

## 9. Biggest risks

| Risk | Why it matters | Owner |
|---|---|---|
| Cooperative grid sync | PCG correctness depends on `grid.sync()` | Masha |
| Cooperative launch support on AMD backend | `hipLaunchCooperativeKernel` may behave differently or have stricter limits | Masha |
| Occupancy calculation | PCG checks whether all blocks can be resident | Masha |
| Large dynamic shared memory | generated dynamics and PCG use dynamic shared memory | Masha + Tobias |
| `cudaFuncSetAttribute` mapping | generated GRiD code relies on it | Tobias |
| cuBLAS to HIPBLAS | SQP update uses AXPY | Masha |
| Two GLASS copies | easy to port one copy but include another | Masha |
| Generated GRiD source size | largest amount of CUDA syntax and shared-memory code | Tobias |
| WSL AMD backend warnings | may be driver/setup-related, not code-related | Shared |

---

## 10. Commands to keep for repeated scans

Use these from the original MPCGPU root:

```bash
# List CUDA/HIP-relevant external API calls

grep -R "cuda[A-Za-z0-9_]*\|cublas[A-Za-z0-9_]*" -n \
  examples include GBD-PCG GLASS \
  --include='*.cu' --include='*.cuh' --include='*.h' --include='*.cpp' \
  | sort

# List CUDA language/device features

grep -R "__global__\|__device__\|__host__\|__shared__\|__syncthreads\|threadIdx\|blockIdx\|blockDim\|gridDim\|atomicAdd\|cooperative_groups" -n \
  examples include GBD-PCG GLASS \
  --include='*.cu' --include='*.cuh' --include='*.h' --include='*.cpp' \
  | sort

# Find cooperative-kernel launch sites

grep -R "cudaLaunchCooperativeKernel\|this_grid\|grid.sync" -n \
  examples include GBD-PCG GLASS \
  --include='*.cu' --include='*.cuh'

# Find cuBLAS usage

grep -R "cublas" -n \
  examples include GBD-PCG GLASS \
  --include='*.cu' --include='*.cuh' --include='*.h'
```

Keep these commands in the repo because they make the scope reproducible.

---

## 11. Summary  

```text
We should split by architecture, not by random files.

Masha takes the core solver path:
- GBD-PCG
- cooperative kernels
- GLASS
- PCG linsys setup
- PCG SQP
- cuBLAS/HIPBLAS replacement

Tobias takes the outer MPC / robot path:
- examples/track_iiwa_pcg.cu
- mpcsim.cuh
- common kernels
- generated GRiD dynamics
- matrix/CSR utilities

Both first test tiny isolated CUDA features.
Then Masha tests GBD-PCG standalone.
Tobias tests generated dynamics standalone.
Only after both sides work do we try the full track_iiwa_pcg.cu example.
```

This split is sane because it separates the two hardest independent risks:

```text
Masha: cooperative PCG solver correctness
Tobias: generated robot dynamics portability
```
