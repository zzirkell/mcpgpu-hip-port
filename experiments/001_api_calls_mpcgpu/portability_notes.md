# CUDA → HIP portability notes

This file collects small findings from isolated CUDA/HIP tests that may later require changes in the real MPCGPU code.

The goal is not only to test whether HIPIFY translates names correctly, but also to notice code patterns that compile with CUDA/NVCC but may be problematic with HIP/Clang or AMD backend.

---

## P001: Always check GPU runtime API return values

### Found in

```text
001_basic_malloc_memcpy_free
002_error_sync
003_basic_kernel_launch_indexing
````

### Pattern

CUDA/HIP runtime calls return an error status:

```cpp
cudaError_t err = cudaMalloc(...);
```

After HIPIFY this becomes:

```cpp
hipError_t err = hipMalloc(...);
```

### Why it matters

HIP marks many runtime API return values as important. If we call functions like this:

```cpp
hipMalloc(...);
hipDeviceSynchronize();
hipFree(...);
```

HIP/Clang may warn that the return value is ignored.

More importantly, without checking return values, failed memory allocations, bad kernel launches, or synchronization errors can be missed.

### Porting rule

Use an error-checking macro/wrapper for GPU API calls:

```cpp
#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err));      \
            return 1;                                                       \
        }                                                                  \
    } while (0)
```

HIPIFY translates this to the HIP version automatically:

```cpp
hipError_t
hipSuccess
hipGetErrorString
```

### Possible MPCGPU impact

When porting real MPCGPU code, direct unchecked calls should be reviewed, especially:

```text
cudaMalloc
cudaFree
cudaMemcpy
cudaMemcpyAsync
cudaMemset
cudaDeviceSynchronize
cudaPeekAtLastError
cudaLaunchCooperativeKernel
cudaFuncSetAttribute
cudaStreamCreate
cudaStreamDestroy
```

---

## P002: Avoid variable length arrays in C++

### Found in

```text
003_basic_kernel_launch_indexing
```

### Problematic pattern

This compiled with NVCC but produced a warning with HIP AMD/Clang:

```cpp
const int n = block.x * block.y * grid.x * grid.y;
int h_out[n];
```

The warning was:

```text
variable length arrays in C++ are a Clang extension
```

### Why it matters

Variable length arrays are normal in C, but not standard C++.

NVCC may accept them, but HIP AMD uses Clang more strictly. This can create warnings or errors when porting CUDA C++ code.

### Porting rule

Use `std::vector` for runtime-sized CPU arrays:

```cpp
std::vector<int> h_out(n);
cudaMemcpy(h_out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
```

Or use compile-time constants when the size is truly fixed:

```cpp
constexpr int n = 48;
int h_out[n];
```

### Possible MPCGPU impact

When porting real MPCGPU code, check for runtime-sized stack arrays in host-side C++ code.

Useful search patterns:

```bash
grep -R "int .*\\[.*\\]" -n include examples GBD-PCG GLASS
grep -R "float .*\\[.*\\]" -n include examples GBD-PCG GLASS
grep -R "double .*\\[.*\\]" -n include examples GBD-PCG GLASS
```
Not every result is bad. Fixed-size arrays are fine. The risky case is an array whose size depends on a runtime variable.

---

## P003: AMD WSL ROCDXG may report SharedSignalPool resource leaks

### Found in

```text
HIP AMD backend runs with HSA_ENABLE_DXG_DETECTION=1
```

### Observed message

```text
Warning: Resource leak detected by SharedSignalPool, 2 Signals leaked.
```

### Meaning

This warning comes from the AMD ROCm/HSA/ROCDXG runtime layer on WSL, not from CUDA `__shared__` memory.

A "signal" is an internal HSA synchronization object used by the AMD runtime. The warning means the runtime detected that some internal synchronization objects were still allocated during shutdown.

### Current interpretation

If the test prints its expected `OK` output and exits normally, this warning is treated as a backend/runtime warning, not as a failed porting test.

### Porting rule

Still make sure real code correctly cleans up all explicit resources:

```text
hipFree / cudaFree
hipStreamDestroy / cudaStreamDestroy
hipblasDestroy / cublasDestroy
hipDeviceSynchronize / cudaDeviceSynchronize where needed
```

For debugging, we can also try adding:

```cpp
cudaDeviceReset();
```

which HIPIFY translates to:

```cpp
hipDeviceReset();
```

### Possible MPCGPU impact

This warning may appear during HIP AMD testing on WSL even when the program result is correct. It should be documented separately from real correctness failures.

---

## P004: AMD WSL iGPU reports no cooperative launch support

### Found in

```text
009_device_props_occupancy
```

### Observed result

```text
CUDA native / NVIDIA: cooperativeLaunch = 1
HIP NVIDIA / NVIDIA: cooperativeLaunch = 1
HIP AMD / AMD Radeon 780M Graphics through WSL/ROCDXG: cooperativeLaunch = 0
```

### Meaning

The AMD WSL/ROCDXG backend currently reports that cooperative kernel launch is not supported on this local AMD iGPU setup.

This does not necessarily mean that AMD GPUs in general cannot support the port. It only describes this specific local backend:

```text
AMD Radeon 780M Graphics
WSL
ROCDXG
HSA_ENABLE_DXG_DETECTION=1
```

### Why it matters

MPCGPU's PCG solver uses cooperative kernel launch and grid-level synchronization. These features are part of the high-risk PCG porting area.

### Porting consequence

Tests for cooperative groups and cooperative kernel launch should still be implemented, but we should expect:

```text
NVIDIA CUDA: pass
HIP NVIDIA: pass
HIP AMD WSL/iGPU: may fail or need to be skipped
```

If AMD cooperative launch is unavailable, the PCG solver may need:

```text
an alternative non-cooperative implementation
a different synchronization strategy
testing on another AMD/Linux setup with better cooperative launch support
```

---

## P005: Cooperative kernel launch needs explicit full signature

### Found in

```text
011_cooperative_groups_grid_sync
```

### Problem

CUDA accepted the short form:

```cpp
cudaLaunchCooperativeKernel(
    (void*)grid_sync_kernel,
    blocks,
    threads_per_block,
    kernel_args
);
```

After HIPIFY, HIP compilation failed because `hipLaunchCooperativeKernel` expects the full form.

### Porting rule

Write the full CUDA form before HIPIFY:

```cpp
const unsigned int shared_memory_bytes = 0;
cudaStream_t stream = 0;

CHECK_GPU(cudaLaunchCooperativeKernel(
    (void*)grid_sync_kernel,
    dim3(blocks),
    dim3(threads_per_block),
    kernel_args,
    shared_memory_bytes,
    stream
));
```

HIPIFY then translates it cleanly to `hipStream_t` and `hipLaunchCooperativeKernel`.

### MPCGPU impact

MPCGPU/GBD-PCG uses cooperative kernel launch, so real code should be checked for short `cudaLaunchCooperativeKernel` calls.

Search:

```bash
grep -R "cudaLaunchCooperativeKernel" -n include examples GBD-PCG GLASS
```

### Current result

```text
CUDA native / NVIDIA: OK
HIP NVIDIA: OK
HIP AMD WSL/iGPU: SKIP, because cooperativeLaunch = 0
```

This AMD result is a local backend/device limitation, not a HIPIFY failure. The same test should be run on the real AMD Linux target.
## P006: cuBLAS to hipBLAS needs separate build and runtime validation

### Found in

`016_blas_axpy`

### Pattern

CUDA BLAS code uses cuBLAS:

`cublasHandle_t`, `cublasCreate`, `cublasSaxpy`, `cublasDaxpy`, `cublasDestroy`

HIPIFY translates this to hipBLAS:

`hipblasHandle_t`, `hipblasCreate`, `hipblasSaxpy`, `hipblasDaxpy`, `hipblasDestroy`

### Build notes

On ROCm 7.2, HIPIFY generated:

`#include <hipblas.h>`

but the actual header was located at:

`/opt/rocm-7.2.0/include/hipblas/hipblas.h`

So the HIP build needed:

`-I$(ROCM_PATH)/include/hipblas`

For AMD linking, the test used:

`-L$(ROCM_PATH)/lib -lhipblas -lrocblas`

For CUDA linking, the test used:

`-L$(CUDA_NATIVE)/lib64 -lcublas`

### Runtime notes

CUDA/NVIDIA passed.

HIP/AMD on local WSL with AMD Radeon 780M compiled, but crashed at runtime with a segmentation fault after loading the AMD WSL GPU bridge.

This means the issue is probably not normal HIPIFY translation anymore. It is more likely related to the local AMD WSL/iGPU hipBLAS/rocBLAS runtime path.

### Porting rule

Do not mix normal CUDA runtime tests and BLAS-library tests in the same category.

For MPCGPU:

- normal kernels and runtime calls can be tested locally with HIP AMD;
- cuBLAS to hipBLAS translation can be checked locally for compilation;
- final hipBLAS runtime validation should be done on a native Linux AMD GPU/server.