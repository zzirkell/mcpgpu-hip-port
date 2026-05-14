## What this Makefile does
> [!NOTE]
> The Makefile quotes `PATH` and `LD_LIBRARY_PATH` inside the HIP backend environments.
>
> This is needed on WSL because the inherited Windows `PATH` can contain folders with spaces, for example `Program Files`.
> Without quotes, `env` can fail with:
>
> ```text
> env: ‘Files/...’: No such file or directory
> ```
CUDA native       -> NVIDIA GPU through nvcc
HIP NVIDIA backend -> NVIDIA GPU through hipcc + nvcc
HIP AMD backend    -> AMD GPU through hipcc + ROCm/ROCDXG

`make` Builds only the CUDA examples by default.

`make cuda` Compiles:

```text
cuda/hello.cu        -> bin/hello_cuda
cuda/unified_add.cu  -> bin/unified_add_cuda
````

`make run-cuda` Builds and runs both CUDA examples.

`make hipify` Runs `hipify-perl` and generates:

```text
cuda/hello.cu        -> hipify_generated/hello.hip.cpp
cuda/unified_add.cu  -> hipify_generated/unified_add.hip.cpp
```

`make hip` Builds HIP for the NVIDIA backend by default. Same as:

```bash
make hip-nvidia
```

`make hip-nvidia` Compiles the HIPIFY-generated files for the NVIDIA GPU:

```text
hipify_generated/hello.hip.cpp        -> bin/hello_hip_nvidia
hipify_generated/unified_add.hip.cpp  -> bin/unified_add_hip_nvidia
```

It forces:

```text
HIP_PLATFORM=nvidia
CUDA used by hipcc: /usr/local/cuda-12.8
```

`make run-hip` Runs the NVIDIA HIP version by default. Same as:

```bash
make run-hip-nvidia
```

`make run-hip-nvidia` Builds and runs the HIP examples on the NVIDIA backend.

`make hip-amd` Compiles the same HIPIFY-generated files for the AMD backend:

```text
hipify_generated/hello.hip.cpp        -> bin/hello_hip_amd
hipify_generated/unified_add.hip.cpp  -> bin/unified_add_hip_amd
```

It uses:

```text
HIP_PLATFORM=amd
HSA_ENABLE_DXG_DETECTION=1
--offload-arch=gfx1103
```

`make run-hip-amd` Builds and runs the HIP examples on the AMD Radeon 780M iGPU.

`make clean` Deletes compiled binaries:

```text
bin/
```

It does **not** delete `hipify_generated/`.

## Normal usage

Run CUDA:

```bash
make clean
make run-cuda
```

Generate HIP files:

```bash
make hipify
```

Run HIP on NVIDIA:

```bash
make clean
make run-hip-nvidia
```

Run HIP on AMD:

```bash
make clean
make run-hip-amd
```

Run default HIP backend:

```bash
make run-hip
```

Currently default HIP backend is NVIDIA. (you can change it)

```
