# MPCGPU CUDA API Call Collection

This folder contains small isolated tests for CUDA API calls that are relevant for CUDA → HIP porting.

The goal is to check one small API group at a time:

```bash
CUDA source #make run-cuda
  -> hipify-perl #make hipify
  -> HIP source
  -> compile/run on NVIDIA backend #make run-hip-nvidia
  -> compile/run on AMD backend #make-hip-run-amd
```

## Test 000: basic memory API

Files:

```text
cuda/test.cu
hipify_generated/test.hip.cpp
```

This test checks the simplest GPU memory workflow:

```text
allocate GPU memory
copy value from CPU to GPU
copy value from GPU back to CPU
free GPU memory
```

## API calls tested

| CUDA call                | HIPIFY result           | Meaning                                        |
| ------------------------ | ----------------------- | ---------------------------------------------- |
| `cudaMalloc`             | `hipMalloc`             | Allocates memory on the GPU/device.            |
| `cudaMemcpy`             | `hipMemcpy`             | Copies memory between CPU/host and GPU/device. |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | Copy direction: CPU → GPU.                     |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | Copy direction: GPU → CPU.                     |
| `cudaFree`               | `hipFree`               | Frees GPU/device memory.                       |
| `cudaError_t`            | `hipError_t`            | Error/status type returned by API calls.       |
| `cudaSuccess`            | `hipSuccess`            | Successful API result.                         |
| `cudaGetErrorString`     | `hipGetErrorString`     | Converts an error code to readable text.       |

## Test 001: ...