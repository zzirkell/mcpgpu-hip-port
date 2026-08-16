# MPCGPU AMD Port Patches: QDLDL Stability and Line-Search Optimization

Date: 2026-08-17

## Scope

This note documents the two relevant patch groups used for the final MI300 component-analysis runs:

1. **QDLDL AMD stability fix** for the rho update / `min` / `max` issue.
2. **AMD-friendly line-search merit evaluation** for both QDLDL and PCG.

The final line-search patch keeps the same algorithmic behavior: all 8 alpha candidates are still evaluated and the selected alpha remains determined by the same merit comparison. The implementation changes how the alpha merit values are computed on AMD.

---

## 1. QDLDL AMD Stability Patch

### Problem

The AMD HIP QDLDL path originally became numerically unstable after the first SQP iteration. A cross-backend trace showed that the first QDLDL solve was close to the NVIDIA result, but after the first SQP iteration AMD set `rho` to zero. The following Schur-system formation then produced NaNs in intermediate values such as `h_val` / `h_gamma`, and the QDLDL solution became invalid.

### Root Cause

The rho update logic used unqualified `min` / `max` expressions with mixed types. On AMD/HIP this resolved differently enough to corrupt the rho update and allow `rho = 0`.

### Fix

Use typed `std::min<T>` and `std::max<T>` with explicit casts for mixed-type expressions, so the rho update remains positive and backend-portable.

### Result

After the fix, AMD QDLDL became tracking-valid. For example, MI300 QDLDL K128 with a 50 ms SQP budget produced stable tracking comparable to NVIDIA:

```text
Average final tracking error: approximately 0.0002
QDLDL linsys timing remained stable
No NaN propagation in the Schur/QDLDL path
```

---

## 2. Original Line-Search Bottleneck

### Original Implementation

The original line-search merit evaluation launched one cooperative merit kernel per alpha candidate:

```text
for alpha candidate p = 0..7:
    launch cooperative merit kernel
        compute merit partials over knots
        use grid-wide synchronization
        reduce to merit(alpha_p)

synchronize
copy 8 merit values GPU -> CPU
select alpha
apply update
```

This implementation is CUDA-friendly, because NVIDIA overlapped the alpha-candidate kernels reasonably well. On AMD, the cooperative kernels behaved much closer to sequential execution, creating a large line-search bottleneck.

### Evidence Before Patch

For MI300 QDLDL K128:

```text
Baseline line_search mean: 601.620 us
Grace Hopper CUDA QDLDL K128 line_search: about 66.4 us
AMD/CUDA gap: about 9x
```

For MI300 PCG K128:

```text
Baseline line_search mean: 638.751 us
```

---

## 3. New AMD-Friendly Line-Search Patch

### New Common Kernels

The patch adds reusable kernels in:

```text
raw_hip_port/include/common/merit.hip.hpp
```

Conceptually:

```cpp
ls_gato_compute_merit_partials<T>(...)
ls_gato_reduce_merit_partials<T>(...)
```

The new partial kernel uses a normal grid layout:

```text
blockIdx.x = knot
blockIdx.y = alpha candidate
```

So all alpha/knot merit partials are exposed to AMD as one normal parallel kernel instead of 8 separate cooperative kernels.

### New Pipeline

```text
1 normal partial-merit kernel:
    compute merit partials for alpha x knot

1 reduction kernel:
    reduce partial merits into 8 merit(alpha) values

synchronize
copy 8 merit values GPU -> CPU
select alpha
apply update
```

### Compile-Time Switch

The new path is guarded by:

```cpp
#ifndef MPCGPU_LS_NORMAL_REDUCE
#define MPCGPU_LS_NORMAL_REDUCE 0
#endif
```

Enable with:

```text
-DMPCGPU_LS_NORMAL_REDUCE=1
```

The switch was added to both:

```text
raw_hip_port/include/qdldl/sqp.hip.hpp
raw_hip_port/include/pcg/sqp.hip.hpp
```

---

## 4. QDLDL MI300 Result

Fair timing comparison, MI300 QDLDL K128, TEST_ITERS=1, 50 ms SQP budget:

| Stage | Baseline mean [us] | Normal-reduce mean [us] | Speedup |
|---|---:|---:|---:|
| KKT | 65.956 | 63.597 | 1.04x |
| Schur | 92.068 | 89.470 | 1.03x |
| Linsys | 533.474 | 510.698 | 1.04x |
| DZ | 20.476 | 17.785 | 1.15x |
| **Line search** | **601.620** | **74.426** | **8.08x** |

Tracking remained equivalent:

```text
Baseline final tracking error:      0.000209898
Normal-reduce final tracking error: 0.000203639
```

Interpretation:

```text
The abnormal AMD line-search bottleneck was removed.
The new MI300 QDLDL line-search time is close to Grace Hopper CUDA line-search time.
```

---

## 5. PCG MI300 Result

Fair timing comparison, MI300 PCG K128, TEST_ITERS=1, 50 ms SQP budget, tolerance 5e-5:

| Stage | Baseline mean [us] | Normal-reduce mean [us] | Speedup |
|---|---:|---:|---:|
| KKT | 68.792 | 65.839 | 1.04x |
| Schur | 220.707 | 220.025 | 1.00x |
| Linsys | 2230.009 | 2104.209 | 1.06x |
| DZ | 22.581 | 19.764 | 1.14x |
| **Line search** | **638.751** | **68.908** | **9.27x** |

Tracking remained valid:

```text
Baseline final tracking error:      0.00934559
Normal-reduce final tracking error: 0.00679862
```

Interpretation:

```text
The PCG line-search bottleneck is also fixed.
However, PCG still has a large AMD-specific linsys bottleneck at K128.
The remaining PCG performance gap is therefore no longer line search, but the PCG linear-system solve itself.
```

---

## 6. Final Interpretation

The line-search optimization should be enabled for both QDLDL and PCG component-analysis runs:

```text
-DUSE_SQP_WORKSPACE=1 -DMPCGPU_LS_NORMAL_REDUCE=1
```

Final conclusions:

```text
QDLDL:
  AMD stability is fixed.
  AMD line search is fixed.
  MI300 QDLDL is now close to Grace Hopper overall.

PCG:
  AMD line search is fixed.
  The remaining major AMD performance issue is the PCG linsys stage, especially at K128.
```

