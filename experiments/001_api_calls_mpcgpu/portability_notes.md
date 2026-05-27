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