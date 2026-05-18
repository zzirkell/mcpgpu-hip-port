# MPCGPU CUDA to HIP Porting

This repository contains setup notes, small experiments, and later the MPCGPU CUDA → HIP porting work.

### CUDA setup and basic programs

To start working with CUDA, look at:

```bash
docs/setup_cuda_wsl.md      # setup and compile/run
experiments/000_cuda_basics/ # real working CUDA examples
````

### HIP setup and HIPIFY

To start working with HIP, look at:

```bash
docs/setup_hip_wsl.md        # HIP setup, HIPIFY, compile/run
experiments/000_cuda_basics/ # HIPIFY-generated HIP examples
```

### ROCm AMD backend on WSL

To run HIP code on the AMD GPU/iGPU from WSL, look at:

```bash
docs/setup_rocm_amd_wsl.md # AMD ROCm/ROCDXG setup and AMD HIP compile/run

#_amd in ./bin compiled files mean that it was HIP compiled for AMD backend
```

### CUDA API call tests from MPCGPU

To test concrete CUDA API calls and their HIPIFY translation, look at:
```bash
001_api_calls_mpcgpu/        # isolated CUDA API call translation tests
001_api_calls_mpcgpu/api_collection.md
```

### Docker setup

We have decided to provide Dockerfiles for configuring both NVIDIA/AMD environment. 

**BUT:**

Docker can package compilers, HIPIFY, ROCm/CUDA userspace, Makefiles, dependencies.
Docker **CANNOT** package the Windows GPU driver, WSL kernel, or /dev/dxg GPU bridge.