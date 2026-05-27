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

## API calls tested

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
## API Calls tested
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
## API Calls tested
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
