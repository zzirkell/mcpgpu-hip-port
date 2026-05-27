### Portability notes!!!

General CUDA → HIP findings that may affect the real MPCGPU port are collected in: portability_notes.md
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
| `cudaSetDevice`      | `hipSetDevice`          | Selects active GPU device.
                |
## 016_blas_axpy

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

