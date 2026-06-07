### Portability notes!!!

General CUDA → HIP findings that may affect the real MPCGPU port are collected in: portability_notes.md

## COMMANDS TO RUN ALL TESTS AT ONCE ON AMD!
```bash
make clean
make clean-generated
make run-all-hip-amd-tests
```

# MPCGPU CUDA API Call Collection

This folder contains small isolated tests for CUDA API calls that are relevant for CUDA → HIP porting.

The goal is to check one small API group at a time:

```bash
CUDA source #make run-cuda
  -> hipify-perl #make hipify
  -> HIP source
  -> compile/run on NVIDIA backend #make run-hip-nvidia
  -> compile/run on AMD backend #make-hip-run-amd
```

## Test 001: basic malloc / memcpy / free

Files:

```text
cuda/001_basic_malloc_memcpy_free.cu
hipify_generated/001_basic_malloc_memcpy_free.hip.cpp
```

This test checks the simplest GPU memory workflow:

```text
allocate GPU memory
copy one value from CPU to GPU
copy the value from GPU back to CPU
free GPU memory
check that the copied value is still correct
```

### API calls tested

| CUDA call                | HIPIFY result           | Meaning                                        |
| ------------------------ | ----------------------- | ---------------------------------------------- |
| `cudaMalloc`             | `hipMalloc`             | Allocates memory on the GPU/device.            |
| `cudaMemcpy`             | `hipMemcpy`             | Copies memory between CPU/host and GPU/device. |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | Copy direction: CPU → GPU.                     |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | Copy direction: GPU → CPU.                     |
| `cudaFree`               | `hipFree`               | Frees GPU/device memory.                       |
| `cudaError_t`            | `hipError_t`            | Error/status type returned by API calls.       |
| `cudaSuccess`            | `hipSuccess`            | Successful API result.                         |
| `cudaGetErrorString`     | `hipGetErrorString`     | Converts an error code to readable text.       |

## Test 002: error checking and synchronization

Files:

```text
cuda/002_error_sync.cu
hipify_generated/002_error_sync.hip.cpp
```
This test checks that a kernel launch can be checked for errors and synchronized before the CPU reads the result.
```
allocate GPU memory
launch a tiny kernel
check launch error
wait for GPU completion
copy result back to CPU
free GPU memory
```
### API Calls tested
| CUDA / feature           | HIPIFY result           | Meaning                                                |
| ------------------------ | ----------------------- | ------------------------------------------------------ |
| `__global__`             | same                    | Marks a GPU kernel callable from CPU.                  |
| `kernel<<<1, 1>>>()`     | same HIP syntax         | Launches one GPU kernel with one block and one thread. |
| `cudaPeekAtLastError`    | `hipPeekAtLastError`    | Checks whether kernel launch configuration failed.     |
| `cudaDeviceSynchronize`  | `hipDeviceSynchronize`  | Waits until GPU work has finished.                     |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | Copies result from GPU to CPU.                         |
| `cudaError_t`            | `hipError_t`            | Error/status type returned by API calls.               |
| `cudaSuccess`            | `hipSuccess`            | Successful API result.                                 |
| `cudaGetErrorString`     | `hipGetErrorString`     | Converts GPU error code to readable text.              |

## Test 003: basic kernel launch and indexing

Files:

```text
cuda/003_basic_kernel_launch_indexing.cu
hipify_generated/003_basic_kernel_launch_indexing.hip.cpp
```
This test checks whether normal CUDA kernel launch syntax and GPU thread indexing work after HIPIFY.
```
create a 2D block configuration
create a 2D grid configuration
launch one kernel
each GPU thread computes its own global index
write global indices into GPU memory
copy results back to CPU
check that every index is correct
```
### API Calls tested
| CUDA / feature              | HIPIFY result   | Meaning                                             |
| --------------------------- | --------------- | --------------------------------------------------- |
| `dim3`                      | same            | CUDA/HIP type for 1D/2D/3D grid and block sizes.    |
| `kernel<<<grid, block>>>()` | same HIP syntax | Launches a GPU kernel.                              |
| `threadIdx.x`               | same            | Thread index inside a block in x direction.         |
| `threadIdx.y`               | same            | Thread index inside a block in y direction.         |
| `blockIdx.x`                | same            | Block index inside the grid in x direction.         |
| `blockIdx.y`                | same            | Block index inside the grid in y direction.         |
| `blockDim.x`                | same            | Number of threads in a block in x direction.        |
| `blockDim.y`                | same            | Number of threads in a block in y direction.        |
| `gridDim.x`                 | same            | Number of blocks in the grid in x direction.        |
| `gridDim.y`                 | same            | Number of blocks in the grid in y direction.        |
| `__global__`                | same            | Marks a function as a GPU kernel callable from CPU. |

## Test 004: CUDA language qualifiers

Files:

```text
cuda/004_cuda_language_qualifiers.cu
hipify_generated/004_cuda_language_qualifiers.hip.cpp
```
This test checks whether CUDA function qualifiers compile and run correctly after HIPIFY.
```
call a normal host-only helper from CPU
call a host-device helper from CPU
launch a GPU kernel
inside the kernel, call device-only helpers
inside the kernel, call a host-device helper
copy results back to CPU
check all computed values
```
## CUDA  features tested:
| CUDA / feature        | HIPIFY result            | Meaning                                              |
| --------------------- | ------------------------ | ---------------------------------------------------- |
| `__global__`          | same                     | Marks a function as a GPU kernel callable from CPU.  |
| `__device__`          | same                     | Marks a function callable from GPU code.             |
| `__host__`            | same                     | Marks a function callable from CPU code.             |
| `__host__ __device__` | same                     | Function can be compiled for both CPU and GPU.       |
| `__forceinline__`     | same / backend dependent | Suggests strongly inlining a device helper function. |
| `kernel<<<1, n>>>()`  | same HIP syntax          | Launches a GPU kernel.
                               |
## Test 005: shared memory and block synchronization

Files:

```text
cuda/005_shared_memory_block_sync.cu
hipify_generated/005_shared_memory_block_sync.hip.cpp
```
This test checks whether static shared memory, dynamic shared memory, and block-level synchronization work after HIPIFY.
```
launch a kernel with multiple blocks and multiple threads
each thread writes values into shared memory
synchronize all threads inside the block
each thread reads another thread's value from shared memory
copy results back to CPU
check all values
```

### CUDA features tested
| CUDA / feature                                      | HIPIFY result   | Meaning                                                 |
| --------------------------------------------------- | --------------- | ------------------------------------------------------- |
| `__shared__`                                        | same            | Static shared memory allocated per block.               |
| `extern __shared__`                                 | same            | Dynamic shared memory size passed during kernel launch. |
| `__syncthreads()`                                   | same            | Synchronizes all threads inside one block.              |
| `kernel<<<blocks, threads, shared_memory_size>>>()` | same HIP syntax | Launches a kernel with dynamic shared memory.           |
| `threadIdx.x`                                       | same            | Thread index inside the block.                          |
| `blockIdx.x`                                        | same            | Block index inside the grid.                            |
| `blockDim.x`                                        | same            | Number of threads in each block.                        |

## Test 009: device properties and occupancy

Files:

```text
cuda/009_device_props_occupancy.cu
hipify_generated/009_device_props_occupancy.hip.cpp
```
This test checks whether device property queries, device attribute queries, and occupancy calculation work after HIPIFY.
```
query visible GPU device count
query current GPU device
read full device properties
query selected device attributes
query cooperative launch support
calculate theoretical kernel occupancy
check that returned values are valid
```
### CUDA API / features tested
| CUDA / feature                                  | HIPIFY result                                  | Meaning                                         |
| ----------------------------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| `cudaDeviceProp`                                | `hipDeviceProp_t`                              | Structure containing GPU properties.            |
| `cudaGetDeviceProperties`                       | `hipGetDeviceProperties`                       | Reads full device property structure.           |
| `cudaDeviceGetAttribute`                        | `hipDeviceGetAttribute`                        | Reads one specific device capability.           |
| `cudaDevAttrMaxThreadsPerBlock`                 | HIP equivalent                                 | Maximum number of threads allowed in one block. |
| `cudaDevAttrMultiProcessorCount`                | HIP equivalent                                 | Number of GPU compute units / SMs.              |
| `cudaDevAttrCooperativeLaunch`                  | HIP equivalent                                 | Whether cooperative kernel launch is supported. |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `hipOccupancyMaxActiveBlocksPerMultiprocessor` | Estimates active blocks per SM/CU for a kernel. |
## Test 012: cooperative launch with dynamic shared memory

Files:

```text
cuda/012_cooperative_launch_shared_memory.cu
hipify_generated/012_cooperative_launch_shared_memory.hip.cpp
```

This test checks cooperative kernel launch together with dynamic shared memory.

```
check whether cooperative launch is supported
if not supported, skip cleanly
allocate GPU memory for block sums and final result
launch cooperative kernel with dynamic shared memory
each block reduces values in shared memory
grid.sync() synchronizes all blocks
one thread combines block sums into final result
copy result back to CPU
check expected sum
```
### CUDA API / features tested
| CUDA / feature                        | HIPIFY result                | Meaning                                        |
| ------------------------------------- | ---------------------------- | ---------------------------------------------- |
| `extern __shared__`                   | same                         | Dynamic shared memory inside kernel.           |
| `__syncthreads()`                     | same                         | Synchronizes threads inside one block.         |
| `cooperative_groups::this_grid()`     | HIP equivalent               | Creates full-grid cooperative group.           |
| `grid.sync()`                         | HIP equivalent               | Synchronizes all blocks in cooperative launch. |
| `cudaLaunchCooperativeKernel`         | `hipLaunchCooperativeKernel` | Explicit cooperative kernel launch.            |
| `shared_memory_bytes` launch argument | same role                    | Controls dynamic shared memory size.           |

## Test 015: HIP header translation

Files:

```text
cuda/015_hip_headers.cu
hipify_generated/015_hip_headers.hip.cpp
```
This test checks whether CUDA runtime headers and simple runtime API calls are translated correctly by HIPIFY.
```
include CUDA runtime headers
query number of GPU devices
query current device
set current device again
check all return values
```
### CUDA API / headers tested
| CUDA / feature       | HIPIFY result           | Meaning                                   |
| -------------------- | ----------------------- | ----------------------------------------- |
| `cuda_runtime.h`     | `hip/hip_runtime.h`     | Main CUDA/HIP runtime header.             |
| `cuda_runtime_api.h` | `hip/hip_runtime_api.h` | Runtime API declarations.                 |
| `cudaError_t`        | `hipError_t`            | Error/status type returned by API calls.  |
| `cudaSuccess`        | `hipSuccess`            | Successful API result.                    |
| `cudaGetErrorString` | `hipGetErrorString`     | Converts GPU error code to readable text. |
| `cudaGetDeviceCount` | `hipGetDeviceCount`     | Returns number of visible GPU devices.    |
| `cudaGetDevice`      | `hipGetDevice`          | Returns currently selected GPU device.    |
| `cudaSetDevice`      | `hipSetDevice`          | Selects active GPU device.                |
## Test 16: blas_axpy usage

### Purpose

This test checks CUDA BLAS usage from cuBLAS and its HIPIFY translation to hipBLAS.

It is important because MPCGPU uses NVIDIA libraries in addition to normal CUDA runtime calls. These library calls are not kernels written by us, but they still have to be translated and linked correctly for HIP.

### CUDA features tested

| CUDA / cuBLAS item | HIPIFY result | Meaning |
| --- | --- | --- |
| `#include <cublas_v2.h>` | `#include <hipblas.h>` | BLAS library header |
| `cublasHandle_t` | `hipblasHandle_t` | BLAS context/handle |
| `cublasCreate` | `hipblasCreate` | Creates BLAS handle |
| `cublasDestroy` | `hipblasDestroy` | Destroys BLAS handle |
| `cublasSaxpy` | `hipblasSaxpy` | Single-precision AXPY: `y = alpha*x + y` |
| `cublasDaxpy` | `hipblasDaxpy` | Double-precision AXPY: `y = alpha*x + y` |
| `CUBLAS_STATUS_SUCCESS` | `HIPBLAS_STATUS_SUCCESS` | Success status check |
| `cudaMalloc`, `cudaMemcpy`, `cudaFree` | `hipMalloc`, `hipMemcpy`, `hipFree` | Device memory management |

### Result

| Backend | Result | Notes |
| --- | --- | --- |
| CUDA / NVIDIA | PASS | cuBLAS works correctly. |
| HIP / NVIDIA | SKIPPED | Mixed HIP NVIDIA + hipBLAS setup is not reliable locally. |
| HIP / AMD WSL / gfx1103 | COMPILES, RUNTIME FAILS | Program builds, but crashes with segmentation fault inside the hipBLAS/rocBLAS runtime path. |
## Test 17: glass vector helper pattern

This test checks a GLASS-like L1 vector helper pattern:

```cpp
template <typename T>
__device__
void glass_like_copy(const T* src, T* dst, int n)
```
GLASS-style helper functions are used in the solver-side code. This test verifies that a small templated __device__ helper can be translated by HIPIFY and executed on CUDA, HIP NVIDIA, and HIP AMD.
### CUDA/HIP features tested
| Feature                    | Meaning                                                |
| -------------------------- | ------------------------------------------------------ |
| `template <typename T>`    | Same helper works for `float` and `double`.            |
| `__device__` helper        | Function runs on the GPU and is called by a kernel.    |
| `__global__` kernel        | Entry point launched from CPU.                         |
| Grid-stride loop           | Lets all GPU threads cover a vector of arbitrary size. |
| `cudaMemcpy` / `hipMemcpy` | Copies input/output data between CPU and GPU.          |
## Test 18: glass axpy_scal
tested:

| CUDA / feature | HIPIFY result | Meaning |
|---|---|---|
| device helper `scal` | same | Scales vector values on GPU. |
| device helper `axpy` | same | Computes `y = alpha * x + y` on GPU. |
| `float` and `double` templates | same | Both precision types compile and run. |
| grid-stride loop inside helper | same | Multiple GPU threads process the vector. |

Conclusion: basic GLASS L1 arithmetic helpers are portable on CUDA, HIP NVIDIA, and HIP AMD.
## Test 019: GLASS-like L1 reduction / dot / norms

Files:

```text
cuda/019_glass_l1_reduce_dot_norm.cu
hipify_generated/019_glass_l1_reduce_dot_norm.hip.cpp
```
This test checks GLASS-like L1 reduction helpers. tests:
```
dot product
L2 norm
infinity norm
float
double
block-level reduction logic
```
### CUDA / HIP features tested
| CUDA / feature                           | HIPIFY result            | Meaning                                   |
| ---------------------------------------- | ------------------------ | ----------------------------------------- |
| `__device__` helper functions            | same                     | Small GPU-only functions used by kernels. |
| `__shared__`                             | same                     | Shared memory used for block reduction.   |
| `__syncthreads()`                        | same                     | Synchronizes threads during reduction.    |
| `threadIdx.x`                            | same                     | Thread index inside block.                |
| `blockDim.x`                             | same                     | Number of threads in block.               |
| `atomicAdd` or final block write pattern | same / backend-dependent | Combines partial reduction results.       |
| `float` / `double` math                  | same                     | Confirms both precisions work.            |
Yes, for your **Masha part**, `021_glass_l3_gemm` is the last planned main test. After this, only `029` is optional/later, and `030` is for both of you together.

Append this to `api_collection.md` after Test 019. I checked the current uploaded md structure before writing this. 

## Test 020: GLASS-like L2 GEMV

Files:
```text
cuda/020_glass_l2_gemv.cu
hipify_generated/020_glass_l2_gemv.hip.cpp
```

This test checks a GLASS-like L2 matrix-vector multiplication pattern.
It is relevant because solver code often needs operations of the form:
y = A * x where `A` is a matrix, `x` is a vector, and `y` is the output vector.

The test checks:

```text
float matrix-vector multiplication
double matrix-vector multiplication
2D data stored in flat 1D arrays
GPU kernel with one output element per thread
CUDA → HIPIFY translation
execution on CUDA, HIP NVIDIA, and HIP AMD
```

### CUDA / HIP features tested

| CUDA / feature                           | HIPIFY result          | Meaning                                |
| ---------------------------------------- | ---------------------- | -------------------------------------- |
| `__global__` kernel                      | same                   | GPU kernel launched from CPU.          |
| `__device__` helper logic                | same                   | Matrix-vector computation runs on GPU. |
| Flat matrix indexing                     | same                   | Matrix is stored as a 1D array.        |
| `float` / `double` templates             | same                   | Both precision types compile and run.  |
| `cudaMalloc` / `cudaMemcpy` / `cudaFree` | HIP equivalents        | Device memory management.              |
| `cudaDeviceSynchronize`                  | `hipDeviceSynchronize` | Waits for GPU computation to finish.   |

---

## Test 021: GLASS-like L3 GEMM

Files:

```text
cuda/021_glass_l3_gemm.cu
hipify_generated/021_glass_l3_gemm.hip.cpp
```

This test checks a GLASS-like L3 matrix-matrix multiplication pattern.

It is relevant because solver-side code may need operations of the form:
C = A * B where all matrices are stored in flat GPU arrays.

The test checks:

```text
float matrix-matrix multiplication
double matrix-matrix multiplication
2D matrix indexing in flat arrays
GPU kernel computing matrix output elements
CUDA → HIPIFY translation
execution on CUDA, HIP NVIDIA, and HIP AMD
```

### CUDA / HIP features tested

| CUDA / feature                           | HIPIFY result          | Meaning                                |
| ---------------------------------------- | ---------------------- | -------------------------------------- |
| `__global__` kernel                      | same                   | GPU kernel launched from CPU.          |
| `__device__` helper logic                | same                   | Matrix-matrix computation runs on GPU. |
| Flat matrix indexing                     | same                   | Matrices are stored as 1D arrays.      |
| `float` / `double` templates             | same                   | Both precision types compile and run.  |
| `dim3` grid/block setup                  | same                   | 2D GPU launch configuration.           |
| `threadIdx` / `blockIdx`                 | same                   | Used to choose matrix output element.  |
| `cudaMalloc` / `cudaMemcpy` / `cudaFree` | HIP equivalents        | Device memory management.              |
| `cudaDeviceSynchronize`                  | `hipDeviceSynchronize` | Waits for GPU computation to finish.   |
