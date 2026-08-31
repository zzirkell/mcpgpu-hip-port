# K32 AMD PCG Optimization Notes

## Goal

The goal of this patch is to reduce the AMD-specific slowdown in the PCG linear-system solve for `KNOT_POINTS=32`.

Before this patch, the AMD PCG linsys path was much slower than CUDA. The main suspected reason was the use of a cooperative kernel with `grid.sync()` inside the PCG solver. On MI300, cooperative launches and grid-wide synchronization were measured to be expensive.

The optimization therefore focuses on removing the cooperative-kernel dependency for the small K32 case.

---

## Scope

This patch is intentionally limited to the K32 PCG linsys path.

Optimized path:

```text
KNOT_POINTS = 32
state_size  = 14
solver      = PCG
backend     = AMD HIP
```

Non-optimized fallback path:

```text
K64 / K128 / K256 / K512 -> original cooperative PCG kernel
macro disabled           -> original cooperative PCG kernel
QDLDL                    -> unchanged
line search              -> uses existing normal-reduce optimization when enabled
```

---

## Files changed

```text
hip_port/GBD-PCG/include/pcg_k32_singleblock.hip.hpp
hip_port/GBD-PCG/include/pcg.hip.hpp
hip_port/include/pcg/sqp.hip.hpp
```

---

## Main idea

The original PCG solver launched one cooperative grid over the knot points.

Conceptually:

```text
K32 original:
32 knot blocks
global grid synchronization between PCG stages
hipLaunchCooperativeKernel
grid.sync()
```

For K32, the full PCG vector is small:

```text
state_size * knot_points = 14 * 32 = 448 scalar entries
```

This fits into one HIP block.

So the optimized path does this instead:

```text
K32 optimized:
1 normal HIP block
448 active threads
shared memory for PCG vectors
__syncthreads() instead of grid.sync()
hipLaunchKernelGGL instead of hipLaunchCooperativeKernel
```

This removes the expensive global cooperative synchronization for K32.

---

## Patch 1: Add K32 single-block PCG kernel

Added a new header:

```text
GBD-PCG/include/pcg_k32_singleblock.hip.hpp
```

This header implements:

```cpp
launchPCGK32SingleBlock<T>(...)
```

Internally it launches:

```cpp
pcg_k32_singleblock_kernel<T, 14, 32>
```

The kernel stores the main PCG vectors in shared memory:

```text
s_lambda
s_r
s_rt
s_p
s_sp
s_red
```

The kernel supports only:

```text
state_size  = 14
knot_points = 32
```

Any other size must fall back to the original cooperative PCG path.

---

## Patch 2: Include the K32 kernel in PCG header

Modified:

```text
GBD-PCG/include/pcg.hip.hpp
```

Added include:

```cpp
#include "pcg_k32_singleblock.hip.hpp"
```

This makes `launchPCGK32SingleBlock<T>()` visible to the SQP PCG code.

---

## Patch 3: Guarded launch in PCG SQP path

Modified:

```text
include/pcg/sqp.hip.hpp
```

Added macro guard:

```cpp
#ifndef MPCGPU_PCG_K32_SINGLEBLOCK
#define MPCGPU_PCG_K32_SINGLEBLOCK 0
#endif
```

Then replaced the active PCG cooperative launch with:

```cpp
#if MPCGPU_PCG_K32_SINGLEBLOCK
    if (state_size == 14 && knot_points == 32) {
        launchPCGK32SingleBlock<T>(...);
    } else {
        hipLaunchCooperativeKernel(...);
    }
#else
    hipLaunchCooperativeKernel(...);
#endif
```

This means:

```text
macro off -> original behavior
macro on + K32 -> optimized single-block kernel
macro on + non-K32 -> original cooperative fallback
```

This is important because the first temporary prototype aborted for K64/K128. The final guarded version does not.

---

## Patch 4: Direct thread indexing

The first K32 prototype used strided loops:

```cpp
for (uint32_t idx = tid; idx < N; idx += blockDim.x) {
    ...
}
```

For K32:

```text
N = 448
```

So each active thread only needs to handle one scalar entry.

The kernel was changed to direct indexing:

```cpp
if (tid < N) {
    ...
}
```

This reduced overhead in the common one-iteration PCG path.

Result:

```text
Before direct indexing:
mean   ≈ 108 us
median ≈ 59 us
p95    ≈ 381 us

After direct indexing:
mean   ≈ 81 us
median ≈ 53 us
p95    ≈ 234 us
```

---

## Patch 5: Fast wavefront/block reduction

The original single-block prototype used a full shared-memory block reduction with many `__syncthreads()` calls.

A faster optional reduction path was added:

```cpp
#ifndef MPCGPU_PCG_K32_FAST_REDUCE
#define MPCGPU_PCG_K32_FAST_REDUCE 1
#endif
```

The fast path first reduces within a wavefront using shuffle operations, then reduces the wave sums.

Conceptually:

```text
old reduction:
all threads reduce through shared memory

new reduction:
wavefront-local reduction
then one small cross-wave reduction
```

Measured effect:

```text
fast_reduce=0:
mean   81.649 us
median 53.190 us
p95    234.012 us

fast_reduce=1:
mean   79.060 us
median 51.780 us
p95    228.352 us
```

The improvement is small but consistent, so the final default is:

```cpp
#define MPCGPU_PCG_K32_FAST_REDUCE 1
```

---

## Patch 6: Block-size tuning

Several block sizes were tested:

```text
128
256
448
512
1024
```

With direct indexing and fast reduction:

```text
448 threads:
mean   78.440 us
median 54.550 us
p95    211.772 us

512 threads:
mean   79.131 us
median 55.021 us
p95    212.312 us

1024 threads:
mean   78.482 us
median 52.340 us
p95    228.982 us
```

Although 1024 had the best median, 448 had the best mean and p95. Since MPC performance is sensitive to tail latency, 448 was selected as the safer default.

Final default:

```cpp
#define MPCGPU_PCG_K32_BLOCK_THREADS 448
```

---

## Patch 7: Rejected pad16 mapping

A padded 16-lane-per-knot mapping was tested to avoid repeated `tid / 14` and `tid % 14`.

Results:

```text
pad16=0:
mean   79.510 us
median 52.201 us
p95    228.842 us

pad16=1:
mean   96.952 us
median 55.630 us
p95    327.683 us
```

The padded layout was slower, probably because it introduces inactive padded rows and forces a 512-thread mapping.

Final default:

```cpp
#define MPCGPU_PCG_K32_PAD16 0
```

---

## Final selected configuration

The final optimized K32 configuration is:

```text
MPCGPU_PCG_K32_SINGLEBLOCK = 1
MPCGPU_PCG_K32_FAST_REDUCE = 1
MPCGPU_PCG_K32_BLOCK_THREADS = 448
MPCGPU_PCG_K32_PAD16 = 0
```

The final validation command used:

```bash
bash "$EXP/run_backend_timing.sh" \
  amd_hip 32 50000 \
  "-DUSE_SQP_WORKSPACE=1 -DMPCGPU_LS_NORMAL_REDUCE=1 -DMPCGPU_PCG_K32_SINGLEBLOCK=1" \
  pcg 1 1e-5
```

Final validation result:

```text
linsys_us:
n      = 103956
mean   = 77.221 us
median = 53.660 us
p95    = 206.532 us

pcg_iters:
mean   = 5.387
median = 1

tracking:
mean   = 0.144
median = 0.100
p95    = 0.366
```

---

## Performance comparison

Old AMD K32 cooperative PCG baseline:

```text
linsys mean   ≈ 221 us
linsys median ≈ 122 us
```

New AMD K32 optimized PCG:

```text
linsys mean   ≈ 77 us
linsys median ≈ 54 us
```

Speedup:

```text
mean speedup   ≈ 221 / 77 ≈ 2.9x
median speedup ≈ 122 / 54 ≈ 2.3x
```

Compared to the earlier CUDA K32 PCG component timing:

```text
CUDA K32 PCG linsys mean ≈ 69.5 us
AMD K32 optimized mean   ≈ 77.2 us
```

So the optimized AMD K32 mean linsys time is now close to CUDA. The median is still somewhat slower than the CUDA diagnostic median, but the previous large AMD-specific slowdown has mostly been removed.

---

## What changed conceptually

Before:

```text
K32 PCG used a cooperative grid.
Many blocks synchronized globally with grid.sync().
This was expensive on MI300.
```

After:

```text
K32 PCG uses one normal HIP block.
The whole K32 linear system fits into shared memory.
Synchronization is local to the block via __syncthreads().
No cooperative launch is needed for K32.
```

This is why the K32 linsys time improved.

---

## What did not change

This patch does not optimize K64/K128/K256/K512.

For larger knot counts:

```text
K64   -> 14 * 64  = 896 entries
K128  -> 14 * 128 = 1792 entries
K256  -> 14 * 256 = 3584 entries
K512  -> 14 * 512 = 7168 entries
```

K64 may still fit into a similar one-block prototype, but K128 and larger likely need a different design:

```text
normal multi-kernel PCG
explicit reductions
no cooperative launch
no grid.sync()
```

The current patch is intentionally conservative and keeps the original cooperative PCG fallback for non-K32 cases.

---

## Report wording

Suggested wording:

```text
The original AMD PCG linsys path used a cooperative kernel with grid-wide synchronization. A MI300 microbenchmark showed that cooperative launches and grid.sync() calls are expensive on this backend. For K32, the full linear-system vector contains only 14 × 32 = 448 scalar entries, so the PCG solve was rewritten as a specialized single-block HIP kernel using shared memory and block-local synchronization. Additional tuning replaced strided indexing with direct thread indexing, added a wavefront-based block reduction, and selected 448 block threads as the best tail-latency tradeoff. This reduced the K32 AMD linsys median from about 122 us to about 54 us and the mean from about 221 us to about 77 us.
```

---

## Next possible work

Possible next steps:

```text
1. Try a K64 single-block variant.
   K64 has 14 * 64 = 896 entries, which can still fit in one 1024-thread block.

2. For K128 and larger, design a normal multi-kernel PCG implementation:
   - matvec kernel
   - reduction kernel
   - update kernel
   - no cooperative launch
   - no grid.sync()

3. Compare optimized AMD K32 against CUDA K32 using the same timing script and same tolerance.
```
