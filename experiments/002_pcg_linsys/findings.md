# Experiment 002: PCG linear-system setup isolation

## Goal

This experiment starts the next real MPCGPU porting task after the standalone `GBD-PCG` solver port.

The goal of this step is to isolate and compile the real MPCGPU PCG linear-system setup path:

```text
include/pcg/linsys_setup.cuh
```

This corresponds to the M6 task:

```text
M6: compile/port include/pcg/linsys_setup.cuh in isolation
```

At this stage, the test is compile-only. The binary is not meant to be run yet.

## Why this step matters

The previous standalone PCG experiment proved that `GBD-PCG` itself can solve a small system on HIP AMD.

This experiment checks the previous stage of the MPCGPU pipeline: the code that forms the PCG linear system data, especially:

```text
S
gamma
Pinv
```

So the intended progression is:

```text
Experiment 003: given S/gamma/lambda, GBD-PCG solves.
Experiment 002: MPCGPU linsys code can form S/gamma/Pinv.
```

## Files copied for CUDA baseline

The first isolated CUDA copy contains:

```text
src_cuda/include/pcg/linsys_setup.cuh
src_cuda/include/utils/matrix.cuh
src_cuda/GLASS/
src_cuda/GBD-PCG/include/
tests/m6_compile_linsys_setup.cu
```

The full `GBD-PCG/include/` directory was copied because the minimal initial set was missing required transitive headers.

## Manual finding 1: missing `constants.cuh`

The first CUDA compile attempt failed because `types.cuh` includes `constants.cuh`:

```text
fatal error: constants.cuh: No such file or directory
```

Cause:

```text
src_cuda/GBD-PCG/include/types.cuh
```

depends on:

```text
src_cuda/GBD-PCG/include/constants.cuh
```

Fix:

Instead of copying only a few selected `GBD-PCG/include` headers, the whole include directory was copied:

```bash
rsync -a "$ORIG/GBD-PCG/include/" src_cuda/GBD-PCG/include/
```

This avoids chasing transitive `GBD-PCG` header dependencies one by one.

## Manual finding 2: `matrix.cuh` is not self-contained

The next CUDA compile attempt failed in:

```text
src_cuda/include/utils/matrix.cuh
```

with errors such as:

```text
incomplete type "std::ofstream" is not allowed
namespace "std" has no member "setprecision"
namespace "std" has no member "endl"
namespace "std" has no member "cerr"
```

Cause:

`matrix.cuh` uses standard C++ I/O utilities but does not include the required standard headers directly. In the full MPCGPU project, those headers may be included indirectly by another file, but in this isolated test the header must be self-contained.

Fix added to `src_cuda/include/utils/matrix.cuh`:

```cpp
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
```

This makes the isolated compile path explicit and portable.

## CUDA compile command

The current CUDA compile-only command is:

```bash
nvcc \
  -std=c++17 -O2 \
  -DSCHUR_THREADS=128 \
  -I src_cuda/include \
  -I src_cuda/GLASS \
  -I src_cuda/GBD-PCG/include \
  tests/m6_compile_linsys_setup.cu \
  -o bin/m6_compile_linsys_setup_cuda \
  2>&1 | tee logs/m6_cuda_compile.txt
```

## HIP plan

For the HIP version of this experiment, the already cleaned standalone HIP port should be reused:

```text
hip_port/GBD-PCG/
```

This folder already contains manually fixed HIP files with clean extensions:

```text
.hip.cpp
.hip.hpp
```

Therefore, for M6 HIP work, we should not re-hipify `GBD-PCG`. We should only hipify the new MPCGPU linsys-related files:

```text
src_hip/include/pcg/linsys_setup.cuh
src_hip/include/utils/matrix.cuh
tests/m6_compile_linsys_setup.cu
```

and reuse:

```text
hip_port/GBD-PCG/
```

as the known working HIP dependency.

## Manual finding 3: templated dynamic shared memory conflict

After fixing the missing standard includes in `matrix.cuh`, the CUDA compile reached `linsys_setup.cuh` and failed at the dynamic shared-memory declaration:

```cpp
extern __shared__ T s_temp[]; matrix.cuh include fix.
```
form_S_gamma_Pinv_kernel is templated on T. The compile-only test instantiates both float and double in the same translation unit. CUDA then sees the same dynamic shared-memory symbol s_temp with two different types: float and double. fix:
```cpp
extern __shared__ __align__(16) unsigned char s_temp_raw[];
T* s_temp = reinterpret_cast<T*>(s_temp_raw);
```
## M6 CUDA compile result

After the manual fixes, the CUDA compile-only baseline succeeded.
## Manual finding 4: HIP cooperative launch signature

After HIPIFY, `linsys_setup.hip.hpp` still used the short CUDA-style cooperative launch form:

```cpp
hipLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel), knot_points, 128, args, s_temp_size)
```
fix:
```cpp
dim3 schur_grid(knot_points);
dim3 schur_block(SCHUR_THREADS);
hipStream_t schur_stream = 0;

gpuErrchk(hipLaunchCooperativeKernel(
    reinterpret_cast<const void*>(kernel),
    schur_grid,
    schur_block,
    args,
    static_cast<unsigned int>(s_temp_size),
    schur_stream
));
```
Remaining warning:

matrix.hip.hpp ignores the return value of hipMemcpy2D.

Fix:
```cpp
gpuErrchk(hipMemcpy2D(h_matrix, pitch, d_matrix, pitch, pitch, rows, hipMemcpyDeviceToHost));
```

works both on NVIDIA and AMD without errors.