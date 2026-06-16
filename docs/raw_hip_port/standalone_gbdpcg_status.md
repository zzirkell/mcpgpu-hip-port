# Standalone GBD-PCG Status

Standalone GBD-PCG was HIPIFY-converted inside raw_hip_port.

Result:
- CUDA standalone GBD-PCG returns NaN lambda.
- HIP NVIDIA standalone GBD-PCG also returns NaN lambda.
- Therefore the NaN is not currently treated as a HIP regression.

Manual HIP fixes applied:
- Added missing HIP runtime include for gpuassert.hip.hpp.
- Changed hipLaunchCooperativeKernel call to the full HIP signature.

Decision:
- Do not fix the standalone NaN yet.
- Revisit only if the full MPCGPU PCG path later shows lambda NaNs caused by missing or empty Pinv/preconditioner handling.
