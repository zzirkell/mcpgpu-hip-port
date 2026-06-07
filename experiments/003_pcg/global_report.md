# Experiment 003: Standalone GBD-PCG CUDA to HIP port

For this phase, port only this folder from the original MPCGPU repo:

```text
MPCGPU/GBD-PCG/
```

---

## Exact files in scope

### 0. Example entry points

```text
GBD-PCG/examples/pcg_solve.cu
GBD-PCG/examples/pcg_solve_dp.cu
GBD-PCG/examples/Makefile
```

Purpose: small standalone PCG examples for float and double.

### 1. PCG public/interface headers

```text
GBD-PCG/include/gpu_pcg.cuh
GBD-PCG/include/interface.cuh
GBD-PCG/include/pcg.cuh
GBD-PCG/include/utils.cuh
GBD-PCG/include/types.cuh
GBD-PCG/include/constants.cuh
GBD-PCG/include/gpuassert.cuh
```

Purpose: actual solver API, cooperative kernel launch, PCG kernel, device helpers, constants, error checking.

### 2. GLASS dependency used by GBD-PCG

```text
GBD-PCG/GLASS/glass.cuh
GBD-PCG/GLASS/src/L1/axpy.cuh
GBD-PCG/GLASS/src/L1/copy.cuh
GBD-PCG/GLASS/src/L1/dot.cuh
GBD-PCG/GLASS/src/L1/ident.cuh
GBD-PCG/GLASS/src/L1/reduce.cuh
GBD-PCG/GLASS/src/L1/scal.cuh
GBD-PCG/GLASS/src/L1/swap.cuh
GBD-PCG/GLASS/src/L2/gemv.cuh
GBD-PCG/GLASS/src/L3/chol_InPlace.cuh
GBD-PCG/GLASS/src/L3/gemm.cuh
GBD-PCG/GLASS/src/L3/inv.cuh
```

Purpose: GLASS math helpers included by the PCG solver.

Even if some L3 files are not heavily used by the small PCG example, they are included through `glass.cuh`, so they should be translated together.

---

## Translation/testing order

### Step 00: Inventory check

Goal: confirm that the copied `GBD-PCG` folder contains all files needed for the standalone PCG experiment.

Commands:

```bash
cd ~/mcpgpu-hip-port/experiments/003_pcg/src_cuda/GBD-PCG

pwd

find . -maxdepth 3 -type f | sort | sed -n '1,160p'

find . -maxdepth 3 \( -name "*.cu" -o -name "*.cuh" -o -name "Makefile" \) -print | sort
```

## Step 01: Fix CUDA double-example Makefile target

Goal: fix the obvious baseline issue where `pcg_dp.exe` was built from the float source file.

```text
pcg_dp.exe:
    $(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve.cu -o pcg_dp.exe
#fix:
pcg_dp.exe:
    $(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve_dp.cu -o pcg_dp.exe
```
## Step 02: Fix CUDA baseline NaN from empty preconditioner

Goal: fix the original CUDA standalone example so it produces finite values before HIP translation.

### Problem

After fixing the double Makefile target, both CUDA examples still built and ran, but returned `nan`:

```text
GBD-PCG returned in 1 iters.
Lambda:
nan nan nan nan nan nan
```
The no-preconditioner API uses config.empty_pinv = 1 by default. However, in include/interface.cuh, d_Pinv was allocated but not initialized:
```cpp
T *d_Pinv;
gpuErrchk(cudaMalloc(&d_Pinv, 3*states_sq*knotPoints*sizeof(T)));
```
Then d_Pinv was passed into the PCG kernel. Inside include/pcg.cuh, the kernel always used Pinv:
```cpp
// r_tilde = Pinv * r
loadbdVec<T, state_size, knot_points-1>(s_r, block_id, &d_r[block_x_statesize]);
__syncthreads();
bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, block_id);
__syncthreads();
```
Fix: Use identity preconditioning when empty_pinv == 1. (commit fix_1)
Now:
```
GBD-PCG returned in 1 iters.
Lambda:
-303.708 -46.4161 -315.182 -14.898 -298.795 13.5047
```