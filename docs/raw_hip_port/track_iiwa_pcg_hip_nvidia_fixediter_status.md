# Raw HIP Port: track_iiwa_pcg HIP NVIDIA Fixed-Iteration Status

The raw HIP port of the full MPCGPU PCG example now builds and runs on HIP NVIDIA.

Configuration:
- LINSYS_SOLVE=1
- REMOVE_JITTERS=0
- CONST_UPDATE_FREQ=0
- TEST_ITERS=1
- SQP exits by constant iterations
- Direct cuBLAS is used on HIP NVIDIA because hipBLAS-on-NVIDIA failed at hipblasCreate.

Result:
- The HIP NVIDIA fixed-iteration run completes successfully.
- Tracking error is in the same range as the CUDA fixed-iteration baseline.
- Linsys timing is also in the same approximate range.
- This is currently the strongest working validation of the raw HIP port.

Manual fixes after HIPIFY:
- Full hipLaunchCooperativeKernel signatures in linsys_setup.hip.hpp and sqp.hip.hpp.
- Wrapped ignored HIP runtime return values with gpuErrchk.
- Conditional BLAS backend:
  - HIP NVIDIA uses cuBLAS directly.
  - HIP AMD keeps hipBLAS.
