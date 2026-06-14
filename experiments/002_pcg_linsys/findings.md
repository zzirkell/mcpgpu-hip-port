# M6/M7 Report: PCG Linear-System Setup Isolation

## Goal

This experiment isolates the real MPCGPU PCG linear-system setup path:

```text
include/pcg/linsys_setup.cuh
```

The goal is to test the code path that forms the PCG data:

```text
S
Pinv
gamma
```

This is the stage before the standalone `GBD-PCG` solver is called.

The planned tasks covered here are:

```text
M6: compile/port include/pcg/linsys_setup.cuh in isolation
M7: run a tiny form_S_gamma_Pinv_kernel / form_schur_system test
```

## M6: Compile linsys setup in isolation

### Files copied

The isolated CUDA test uses:

```text
src_cuda/include/pcg/linsys_setup.cuh
src_cuda/include/utils/matrix.cuh
src_cuda/GLASS/
src_cuda/GBD-PCG/include/
tests/m6_compile_linsys_setup.cu
```

The whole `GBD-PCG/include/` folder was copied because selected individual headers were not enough.

### Finding 1: missing transitive GBD-PCG header

The first CUDA compile failed with:

```text
fatal error: constants.cuh: No such file or directory
```

Cause:

```text
types.cuh includes constants.cuh
```

Fix:

```bash
rsync -a "$ORIG/GBD-PCG/include/" src_cuda/GBD-PCG/include/
```

Instead of copying only individual headers, the complete `GBD-PCG/include/` folder is used.

### Finding 2: matrix.cuh was not self-contained

The next CUDA compile failed in:

```text
src_cuda/include/utils/matrix.cuh
```

Errors included:

```text
incomplete type "std::ofstream" is not allowed
namespace "std" has no member "setprecision"
namespace "std" has no member "cerr"
namespace "std" has no member "endl"
```

Cause:

`matrix.cuh` uses standard C++ I/O utilities but does not include the required standard headers directly.

Fix added to `matrix.cuh`:

```cpp
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
```

### Finding 3: templated dynamic shared memory conflict

After the include fixes, the CUDA compile reached the kernel in `linsys_setup.cuh` and failed at:

```cpp
extern __shared__ T s_temp[];
```

Error:

```text
declaration is incompatible with previous "s_temp"
a host variable("s_temp") redeclared with __shared__
```

Cause:

The compile-only test instantiates the templated kernel for both `float` and `double` in the same translation unit. CUDA then sees the same dynamic shared-memory symbol with two different types.

Fix:

```cpp
extern __shared__ __align__(16) unsigned char s_temp_raw[];
T* s_temp = reinterpret_cast<T*>(s_temp_raw);
```

This keeps the rest of the kernel unchanged while making the shared-memory declaration type-independent.

### M6 result

After these fixes, the CUDA compile-only test succeeded.

The HIP version also compiled successfully for:

```text
HIP NVIDIA
HIP AMD
```

Conclusion:

```text
M6 is complete.
```

## M7: Runtime linsys setup test

### CUDA baseline

A tiny runtime test was added:

```text
tests/m7_run_linsys_setup.cu
```

The test creates small valid input arrays for:

```text
G
C
g
c
```

and calls:

```text
form_schur_system
```

The test then copies back:

```text
S
Pinv
gamma
```

and checks that all values are finite.

CUDA result:

```text
float
OK: linsys setup produced finite S, Pinv, and gamma

double
OK: linsys setup produced finite S, Pinv, and gamma
```

Conclusion:

```text
The CUDA linsys runtime baseline works.
```

## M7 HIP NVIDIA

The HIP NVIDIA version initially compiled and ran, but its output did not fully match CUDA.

Automatic comparison showed:

```text
float:
S:    4 mismatches
Pinv: 24 mismatches
gamma: 0 mismatches

double:
S:    4 mismatches
Pinv: 24 mismatches
gamma: 0 mismatches
```

### Finding 4: wrong GLASS dependency caused numerical mismatch

The mismatch was caused by using the standalone `GBD-PCG/GLASS` dependency from the cleaned PCG port.

That GLASS copy is sufficient for standalone `GBD-PCG`, but it is not the correct dependency for the full MPCGPU linsys path.

Initial HIP include path used:

```text
src_hip/GBD-PCG/GLASS
```

but the CUDA baseline used:

```text
src_cuda/GLASS
```

Fix:

The exact MPCGPU GLASS copy used by the CUDA baseline was copied and hipified:

```bash
rm -rf src_hip/GLASS
rsync -a src_cuda/GLASS/ src_hip/GLASS/
```

Then HIP was compiled with:

```text
-I src_hip/GLASS
-I src_hip/GBD-PCG/include
```

After this fix, HIP NVIDIA matched CUDA for:

```text
S
Pinv
gamma
```

Conclusion:

```text
MPCGPU linsys/SQP must use the full MPCGPU GLASS port, not the smaller standalone GBD-PCG GLASS subset.
```

## M7 HIP AMD

The corrected HIP version compiled successfully on the AMD server.

However, runtime failed at the cooperative launch in:

```text
src_hip/include/pcg/linsys_setup.hip.hpp
```

Runtime error:

```text
GPUassert: the operation cannot be performed in the present state
```

The failing call is:

```cpp
hipLaunchCooperativeKernel(...)
```

The kernel cannot simply be changed to a normal kernel launch because it contains a grid-wide synchronization:

```cpp
cgrps::this_grid().sync();
```

A check confirmed the grid synchronization is really present:

```text
602: cgrps::this_grid().sync();
```

### AMD cooperative-launch debug

The AMD device reported cooperative launch support and enough theoretical capacity:

```text
cooperativeLaunch=1
knot_points=3
block=128
shmem=224
multiProcessorCount=27
maxBlocksPerSM=16
maxCoopBlocks=432
```

So the device claims cooperative launch support, and the test grid is much smaller than the theoretical limit.

Additional attempt:

```text
Using HIP_KERNEL_NAME(form_S_gamma_Pinv_kernel<T>) instead of reinterpret_cast<const void*>(kernel)
```

did not fix the AMD runtime failure.

### M7 AMD conclusion

The M7 HIP AMD version:

```text
compiles successfully
fails at runtime at hipLaunchCooperativeKernel
```

This is currently left as an integration/discussion issue.

Possible future directions:

```text
1. Refactor the cooperative kernel into two safe non-cooperative phases.
2. Investigate ROCm cooperative-groups restrictions for this specific kernel.
3. Discuss whether this linsys setup path should be rewritten during MPCGPU integration.
```

## Final M6/M7 status

```text
M6 CUDA compile-only:       PASS
M6 HIP NVIDIA compile-only: PASS
M6 HIP AMD compile-only:    PASS

M7 CUDA runtime:            PASS
M7 HIP NVIDIA runtime:      PASS after using the correct MPCGPU GLASS
M7 HIP AMD runtime:         BLOCKED at cooperative launch
```

## Transferable result

The following result is safe to transfer into `hip_port`:

```text
Full MPCGPU GLASS HIP port
```

The following files can also be stored in `hip_port` as current WIP port files, but should be documented as not fully AMD-runtime validated yet:

```text
MPCGPU/include/pcg/linsys_setup.hip.hpp
MPCGPU/include/utils/matrix.hip.hpp
```

Important note:

```text
standalone GBD-PCG/GLASS is not enough for MPCGPU linsys/SQP.
The full MPCGPU GLASS port must exist separately as hip_port/GLASS.
```
