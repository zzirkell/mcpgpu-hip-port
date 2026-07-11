# MPCGPU CUDA to HIP Porting

This repository contains setup notes, small experiments, and later the MPCGPU CUDA → HIP porting work.

### HIP port folder

Cleaned HIP ports are stored in:

```bash
raw_hip_port/
```
The ``hip_port`` folder is a port without making any changes to the HIP code (only some syntaxis but it was absolutely not efficient)
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

### MPCGPU CUDA/HIP feature tests

To test concrete CUDA API calls, CUDA language features, and their HIPIFY translation, look at:

```bash
experiments/001_api_calls_mpcgpu/                # isolated CUDA/HIP feature translation tests
experiments/001_api_calls_mpcgpu/api_collection.md # tested calls/features and results
```

This folder is for small isolated tests before touching the real MPCGPU code.

### Docker setup

We have decided to provide Dockerfiles for configuring both NVIDIA/AMD environment. 

**BUT:**

Docker can package compilers, HIPIFY, ROCm/CUDA userspace, Makefiles, dependencies.
Docker **CANNOT** package the Windows GPU driver, WSL kernel, or /dev/dxg GPU bridge.

### Project plan and work separation

The detailed porting plan, ownership split, and scan findings are in:
```bash
docs/project_plan/mpcgpu_overview_plan.md              # main detailed project plan
docs/project_plan/final_table_separation.md            # concrete test/feature ownership table
docs/project_plan/tobias_grep_findings/                # grep/static-scan findings for CUDA usage
```

### Standalone GBD-PCG CUDA → HIP port

To see the first real MPCGPU component port, look at:

```bash
experiments/003_pcg/                 # step-by-step CUDA baseline and HIP porting experiment
experiments/003_pcg/global_report.md        # notes, fixes, commands, and results
hip_port/GBD-PCG/                    # cleaned standalone HIP AMD port of GBD-PCG
```
GBD-PCG was copied, fixed, translated with HIPIFY, compiled, and tested. use to run:
```cd hip_port/GBD-PCG/examples
make run-hip
```
