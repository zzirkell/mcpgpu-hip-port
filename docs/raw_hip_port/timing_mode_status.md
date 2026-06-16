# Timing-Mode Status

The raw HIP port was also tested in the default timing-mode configuration.

Configuration:
- LINSYS_SOLVE=1
- TEST_ITERS=1
- Default CONST_UPDATE_FREQ behavior
- Default jitter behavior
- SQP exits by constant time

Observed HIP NVIDIA result:
- Average tracking error around 1.235
- Average final tracking error around 0.999
- Program then terminates in stats printing with vector::_M_range_check because one stats vector is empty.

Interpretation:
- This matches the local CUDA timing-mode behavior.
- Therefore this is not treated as a HIP-specific regression.
- Fixed-iteration mode is the meaningful local correctness baseline for CUDA-vs-HIP comparison.
