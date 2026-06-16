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

## Test 006: stream and async copy

Files:
    cuda/006_stream_async_copy.cu
    hipify_generated/006_stream_async_copy.hip.cpp

This test checks whether basic stream creation, asynchronous memory copies, and stream destruction work after HIPIFY. Streams are critical for overlapping computation and memory transfers.

### API calls / features tested
| CUDA / feature | HIPIFY result | Meaning |
|---|---|---|
| `cudaStream_t` | `hipStream_t` | Handle for managing an asynchronous stream of GPU operations. |
| `cudaStreamCreate` | `hipStreamCreate` | Creates a new asynchronous stream. |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | Copies memory asynchronously on a specific stream. |
| `cudaStreamDestroy` | `hipStreamDestroy` | Destroys the stream and frees associated resources. |

## Test 007: stream priority

Files:
    cuda/007_stream_priority.cu
    hipify_generated/007_stream_priority.hip.cpp

This test ensures that streams can be created with specific priorities and queried for supported priority ranges on the AMD backend.

### API calls / features tested
| CUDA / feature | HIPIFY result | Meaning |
|---|---|---|
| `cudaDeviceGetStreamPriorityRange` | `hipDeviceGetStreamPriorityRange` | Queries the valid priority range for the current device. |
| `cudaStreamCreateWithPriority` | `hipStreamCreateWithPriority` | Creates a stream with a specific scheduling priority. |
| `cudaStreamNonBlocking` | `hipStreamNonBlocking` | Flag to create a stream that does not synchronize with the default stream. |

## Test 008: memset and memcpy2d

Files:
    cuda/008_memset_memcpy2d.cu
    hipify_generated/008_memset_memcpy2d.hip.cpp

This test verifies operations for setting GPU memory directly and performing pitched 2D memory copies, which are heavily used in the utility and matrix headers.

### API calls / features tested
| CUDA / feature | HIPIFY result | Meaning |
|---|---|---|
| `cudaMemset` | `hipMemset` | Fills device memory with a constant byte value. |
| `cudaMemcpy2D` | `hipMemcpy2D` | Copies a 2D memory matrix with specific pitch (stride). |


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
## Test 010: cooperative groups block synchronization

Files:

```text
cuda/010_cooperative_groups_block.cu
hipify_generated/010_cooperative_groups_block.hip.cpp
```
This test checks whether CUDA cooperative groups work for block-level synchronization after HIPIFY.
```
include cooperative groups header
create a thread block group inside the kernel
each thread gets its rank inside the block
each thread writes one value into shared memory
synchronize the whole block using block.sync()
one thread per block sums the shared values
copy results back to CPU
check block size and block sums
```
### CUDA features tested
| CUDA / feature                            | HIPIFY result                 | Meaning                                         |
| ----------------------------------------- | ----------------------------- | ----------------------------------------------- |
| `cooperative_groups.h`                    | HIP cooperative groups header | Header for cooperative groups API.              |
| `cooperative_groups::this_thread_block()` | HIP equivalent                | Creates a group representing the current block. |
| `cooperative_groups::thread_block`        | HIP equivalent                | Type representing one thread block.             |
| `block.thread_rank()`                     | same / HIP equivalent         | Thread rank inside the block group.             |
| `block.size()`                            | same / HIP equivalent         | Number of threads in the block group.           |
| `block.sync()`                            | same / HIP equivalent         | Synchronizes all threads in the block group.    |
| `__shared__`                              | same                          | Shared memory used by threads in one block.     |

## Test 011: cooperative groups grid synchronization

Files:

```text
cuda/011_cooperative_groups_grid_sync.cu
hipify_generated/011_cooperative_groups_grid_sync.hip.cpp
```
This test checks whether grid-level cooperative groups and cooperative kernel launch work after HIPIFY.
```
query whether cooperative launch is supported
if not supported, skip the test cleanly
allocate GPU memory
launch a cooperative kernel using cudaLaunchCooperativeKernel
inside the kernel, write data from all threads
synchronize the whole grid using grid.sync()
one thread sums the data after grid synchronization
copy result back to CPU
check the final sum
```
### CUDA API / features tested
| CUDA / feature                    | HIPIFY result                | Meaning                                                        |
| --------------------------------- | ---------------------------- | -------------------------------------------------------------- |
| `cooperative_groups::this_grid()` | HIP equivalent               | Creates a group representing the whole launched grid.          |
| `cooperative_groups::grid_group`  | HIP equivalent               | Type representing all threads in the cooperative grid.         |
| `grid.sync()`                     | HIP equivalent               | Synchronizes all threads in the cooperative grid.              |
| `cudaDeviceGetAttribute`          | `hipDeviceGetAttribute`      | Used to check whether cooperative launch is supported.         |
| `cudaDevAttrCooperativeLaunch`    | HIP equivalent               | Device capability flag for cooperative launch.                 |
| `cudaLaunchCooperativeKernel`     | `hipLaunchCooperativeKernel` | Launches a kernel that supports grid-wide synchronization.     |
| `cudaStream_t`                    | `hipStream_t`                | Stream type used by the explicit cooperative launch signature. |

## Test 013: function attributes and dynamic shared memory

Files:
    cuda/013_func_set_attribute_dynamic_smem.cu
    hipify_generated/013_func_set_attribute_dynamic_smem.hip.cpp

This test confirms that function attributes can be modified at runtime. This is critical for the generated GRiD dynamics, which require opting into larger dynamic shared memory allocations than the default limit.

### API calls / features tested
| CUDA / feature | HIPIFY result | Meaning |
|---|---|---|
| `cudaFuncSetAttribute` | `hipFuncSetAttribute` | Sets a specific attribute for a device function. |
| `cudaFuncAttributeMaxDynamicSharedMemorySize` | `hipFuncAttributeMaxDynamicSharedMemorySize` | Attribute flag to unlock maximum dynamic shared memory for a kernel. |

## Test 014: atomic operations

Files:
    cuda/014_atomic_add.cu
    hipify_generated/014_atomic_add.hip.cpp

This test ensures that atomic accumulation works correctly on the AMD backend, ensuring no race conditions occur when multiple threads write to the same memory location.

### API calls / features tested
| CUDA / feature | HIPIFY result | Meaning |
|---|---|---|
| `atomicAdd` | `atomicAdd` | Performs an atomic read-modify-write addition directly in device or shared memory. |

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

---

## Real Repository Component Tests: Dynamics and Utilities

### Tests 023 & 024: Integrator, Merit, and Tracking

Files:
    include/common/integrator.cuh -> integrator.hip.hpp
    include/common/merit.cuh -> merit.hip.hpp

These tests isolate the compilation and runtime execution of the physics integration step and the SQP cost evaluations.
* **023:** Validates `integrator_kernel` and `simple_integrator_kernel` with tiny dummy inputs.
* **024:** Validates `ls_gato_compute_merit`, `compute_merit`, and `compute_tracking_error_kernel`. The merit kernel also successfully tests the port of `cooperative_groups::this_grid().sync()` on the AMD architecture.

### Test 025: CSR Matrix Utilities

Files:
    include/utils/matrix.cuh -> matrix.hip.hpp

This test validates the core matrix manipulation routines and Sparse CSR (Compressed Sparse Row) preparations. It heavily relies on the successful port of `hipMemcpy2D` (proven in Test 008) and custom error-checking macros.

### Tests 026, 027 & 028: GRiD-Generated Robot Dynamics

Files:
    include/dynamics/iiwa/iiwa_grid.cuh -> iiwa_grid.hip.hpp
    include/dynamics/iiwa/iiwa_eepos_grid.cuh -> iiwa_eepos_grid.hip.hpp

These tests form the foundation of the physical simulation. They verify that the massive, automatically generated CUDA code blocks port successfully to HIP and run without illegal memory accesses.
* **026:** Validates `inverse_dynamics_kernel`, `direct_minv_kernel`, and `forward_dynamics_kernel`.
* **027:** Validates the analytical gradients via `inverse_dynamics_gradient_kernel` and `forward_dynamics_gradient_kernel`.
* **028:** Validates the kinematics mapping via `end_effector_positions_kernel` and its respective gradient kernel.

## Test 022: KKT and dz common kernels

Files:
    cuda/022_kkt_dz_common_kernels.cu
    hipify_generated/022_kkt_dz_common_kernels.hip.cpp
    (Tested directly via include/common/dz.cuh and kkt.cuh)

This test checks the compilation and isolated execution of the trajectory update (`dz`) and KKT submatrix generation kernels.

    allocate large dummy memory buffers to prevent out-of-bounds access
    calculate exact dynamic shared memory requirements for the KKT kernel using get_kkt_smem_size()
    launch template-based __global__ kernels (generate_kkt_submatrices)
    launch host-wrapped kernels (compute_dz)
    verify underlying matrix/vector math relying on hipBLAS
    check custom error handling macros

### API calls / features tested
| CUDA / feature | HIPIFY result | Meaning |
|---|---|---|
| `extern __shared__` | same | Dynamic shared memory used heavily by the KKT generator. |
| `kernel<<<b, t, s>>>()` | same HIP syntax | Launching kernels with dynamically calculated shared memory size. |
| Template Kernels | same HIP syntax | Executing `__global__` functions with `<float, 0, false>` template arguments. |
| `cudaMemset` | `hipMemset` | Filling allocated memory with zeros to prevent NaN calculations. |
| `cublasSaxpy` / Math | `hipblas` Math | Ensuring underlying linear algebra calls link correctly via `-lhipblas`. |

## Test 030: full track_iiwa_pcg integration (T11 / T30)

Files:
    examples/track_iiwa_pcg.cu
    examples/track_iiwa_pcg.hip.cpp

This test checks the full end-to-end integration of the NMPC wrapper, generated dynamics, and the PCG linear system solver on the AMD backend.

    load precomputed trajectory files (.traj / .csv)
    execute the full SQP (Sequential Quadratic Programming) loop
    utilize cooperative grid synchronization inside the solver
    calculate final tracking errors
    handle strict CPU-side C++ standard library bounds checking

### API calls / features tested
| CUDA / feature | HIPIFY result | Meaning |
|---|---|---|
| End-to-End Build | HIP / `hipcc` | Compiling the entire dependency tree (mpcsim, pcg, dynamics) into one executable. |
| Host / Device Sync | same | Ensuring CPU trajectory loading feeds correctly into GPU memory. |
| Full Grid Sync | HIP cooperative groups | Executing the PCG algorithm across the full AMD GPU grid. |