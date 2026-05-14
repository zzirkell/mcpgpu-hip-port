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

#_amd in ./bim compiled files mean that it was HIP compiled for AMD backend
```
