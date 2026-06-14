# Experiment 005: SQP Compile Isolation

## Goal

This experiment covers the M8 task from the porting plan.

The goal is to compile the real MPCGPU SQP/PCG layer in isolation, without running the full `track_iiwa_pcg` example yet.

The main target header is:

`include/pcg/sqp.cuh`

This layer sits above the PCG linear-system setup and standalone GBD-PCG solver. It pulls together the SQP path, PCG solver calls, line-search/merit logic, dynamics headers, and BLAS usage.

M8 is intentionally compile-only. Runtime validation is postponed because M7 already found an AMD runtime blocker in the cooperative launch used by `linsys_setup`.

## Scope

The isolated CUDA source copy includes the required dependency chain for `sqp.cuh`:

`src_cuda/include/pcg/sqp.cuh`

`src_cuda/include/pcg/linsys_setup.cuh`

`src_cuda/include/common/`

`src_cuda/include/dynamics/`

`src_cuda/include/utils/matrix.cuh`

`src_cuda/GLASS/`

`src_cuda/GBD-PCG/include/`

The compile test is:

`tests/m8_compile_sqp.cu`

The HIP version is:

`tests/m8_compile_sqp.hip.cpp`

The test only includes the SQP header and returns from `main`. It is used to force compilation of the SQP/PCG layer and its dependencies.

## CUDA baseline

The first CUDA compile failed because `sqp.cuh` includes:

`common/kkt.cuh`

Fix:

The `include/common/` folder was copied from the original MPCGPU source tree.

The next compile failed because `common/kkt.cuh` includes:

`dynamics/rbd_plant.cuh`

Fix:

The `include/dynamics/` folder was copied from the original MPCGPU source tree.

The next compile failed because the dynamics headers include:

`settings.cuh`

The file is located in the original project at:

`include/common/settings.cuh`

Some headers include it directly as `settings.cuh`, so the isolated compile needs this additional include path:

`-I src_cuda/include/common`

## Repeated finding: matrix.cuh is not self-contained

The CUDA compile also failed in:

`src_cuda/include/utils/matrix.cuh`

Errors included missing `std::ofstream` and `std::cerr`.

This is the same self-contained-header issue already seen in M6.

Fix added to `matrix.cuh`:

`#include <fstream>`

`#include <iomanip>`

`#include <iostream>`

`#include <limits>`

After these fixes, the CUDA compile-only baseline succeeded.

M8 CUDA result:

`PASS`

## HIP setup

The HIP working copy was created from the CUDA isolated source.

The tested HIP dependencies from earlier experiments were reused where possible:

`hip_port/GLASS/`

`hip_port/GBD-PCG/`

This is important because M7 showed that the standalone `GBD-PCG/GLASS` subset is not sufficient for the full MPCGPU linsys/SQP path. The SQP path needs the full MPCGPU GLASS copy.

The MPCGPU headers in this experiment were hipified and renamed to `.hip.hpp`.

The test file was converted to:

`tests/m8_compile_sqp.hip.cpp`

## HIPBLAS dependency

The first HIP NVIDIA compile failed because HIPIFY translated cuBLAS usage into:

`#include <hipblas.h>`

but the HIPBLAS header was not found by the default include paths.

Fix:

The HIPBLAS include path was added:

`-I /opt/rocm-7.2.0/include/hipblas`

For HIP NVIDIA linking, the following libraries were used:

`-lhipblas -lcublas`

For HIP AMD linking, the following libraries were used:

`-lhipblas -lrocblas`

Finding:

M8 confirms that the SQP/PCG layer depends on the cuBLAS to hipBLAS migration path.

## Cooperative launch signature fixes

HIPIFY kept several short CUDA-style cooperative launch calls.

These failed to compile because HIP requires the explicit six-argument cooperative launch form with:

`dim3 grid`

`dim3 block`

`void** args`

`shared memory size`

`stream`

### Fix 1: linsys_setup cooperative launch

In:

`src_hip/include/pcg/linsys_setup.hip.hpp`

The short cooperative launch was replaced with the full HIP form using:

`dim3 schur_grid(knot_points)`

`dim3 schur_block(SCHUR_THREADS)`

`hipStream_t schur_stream = 0`

and a full `hipLaunchCooperativeKernel` call with explicit shared-memory size and stream.

This is the same portability issue already found in M6/M7 and in the earlier cooperative launch API tests.

### Fix 2: PCG solver launch inside sqp.hip.hpp

In:

`src_hip/include/pcg/sqp.hip.hpp`

The short cooperative launch of the PCG kernel was replaced with the full HIP form using:

`dim3 pcg_grid(knot_points)`

`dim3 pcg_block(PCG_NUM_THREADS)`

`hipStream_t pcg_stream = 0`

### Fix 3: line-search merit launch inside sqp.hip.hpp

A second cooperative launch in `sqp.hip.hpp`, used for the line-search merit kernel, was also replaced with the full HIP form using:

`dim3 merit_grid(knot_points)`

`dim3 merit_block(MERIT_THREADS)`

The existing stream `streams[p]` was kept.

## Warning cleanup

Some HIP runtime calls produced warnings because their return values were ignored.

The following calls were cleaned where practical:

`hipStreamCreate(...)`

`hipMemcpy(...)`

`hipMemcpy2D(...)`

by wrapping them with:

`gpuErrchk(...)`

Remaining warnings in the AMD build come from:

`src_hip/include/dynamics/iiwa/iiwa_eepos_grid.hip.hpp`

Specifically, several `hipFuncSetAttribute(...)` calls ignore the returned `hipError_t`.

These warnings are not compile blockers. They should be cleaned later by wrapping those calls in `gpuErrchk(...)`.

## HIP NVIDIA compile result

After adding the HIPBLAS include path, linking against HIPBLAS/cuBLAS, and fixing the cooperative launch signatures, the HIP NVIDIA compile-only test succeeded.

M8 HIP NVIDIA result:

`PASS`

## HIP AMD compile result

The same M8 compile-only test was compiled on the AMD server for:

`gfx1101`

The compile succeeded with warnings from `iiwa_eepos_grid.hip.hpp` about ignored `hipError_t` return values from `hipFuncSetAttribute(...)`.

These warnings were emitted for both device and host compilation, but they did not stop the build.

M8 HIP AMD result:

`PASS with warnings`

## Why runtime was not attempted

M8 is compile-only by design.

Runtime was not attempted because M7 already found an AMD runtime blocker in the lower linsys setup layer.

The relevant M7 blocker is:

`linsys_setup.hip.hpp` uses `cgrps::this_grid().sync()`

Therefore, the kernel requires cooperative launch.

On AMD, the cooperative launch failed at runtime with:

`GPUassert: the operation cannot be performed in the present state`

even though the device reported cooperative launch support and enough theoretical capacity.

This issue should be discussed during integration before full M9 runtime work.

## Final M8 status

`M8 CUDA compile: PASS`

`M8 HIP NVIDIA compile: PASS`

`M8 HIP AMD compile: PASS with warnings`

Runtime status:

`Not attempted`

Reason:

`M8 is compile-only, and M7 already exposed a lower-level AMD cooperative-launch runtime blocker.`

## Files suitable for transfer to hip_port

After M8, the following HIP files/directories are useful for the root `hip_port` result:

`hip_port/MPCGPU/include/pcg/`

`hip_port/MPCGPU/include/common/`

`hip_port/MPCGPU/include/dynamics/`

`hip_port/MPCGPU/include/utils/`

The full MPCGPU GLASS HIP port should remain separate from the standalone GBD-PCG GLASS subset:

`hip_port/GLASS/`

The standalone GBD-PCG port remains:

`hip_port/GBD-PCG/`

Important architecture note:

The standalone `GBD-PCG/GLASS` copy is valid for the standalone GBD-PCG solver, but the full MPCGPU SQP/linsys path must use the full MPCGPU GLASS port.

## Remaining work

The remaining major task is M9, which should be done together with Tobias.

M9 will integrate the ported SQP/PCG path into the full MPCGPU example, especially:

`track_iiwa_pcg`

The known M7 AMD cooperative-launch blocker should be carried into M9 as a discussion item because it directly affects the real runtime integration path.
