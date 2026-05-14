## What this Makefile does

`make`
Builds only the CUDA examples by default.

`make cuda`
Compiles:

```text
cuda/hello.cu          → bin/hello_cuda
cuda/unified_add.cu    → bin/unified_add_cuda
```

`make run-cuda`
Builds and runs both CUDA examples.

`make hipify`
Runs `hipify-perl` and generates:

```text
cuda/hello.cu          → hipify_generated/hello.hip.cpp
cuda/unified_add.cu    → hipify_generated/unified_add.hip.cpp
```

`make hip`
First hipifies the CUDA files if needed, then compiles:

```text
hipify_generated/hello.hip.cpp        → bin/hello_hip
hipify_generated/unified_add.hip.cpp  → bin/unified_add_hip
```

It forces HIP to use:

```text
CUDA 12.8
HIP_PLATFORM=nvidia
hipcc
```

`make run-hip`
Builds and runs both HIP examples.

`make clean`
Deletes compiled binaries:

```text
bin/
```

It does **not** delete `hipify_generated/`.