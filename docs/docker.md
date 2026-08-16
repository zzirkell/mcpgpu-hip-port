# Docker Setup
> [!NOTE]
> Some imports are heavy (30GB for NVIDIA and ~20GB for AMD).
> Use after looking inside Docker files!!

This repository uses two Docker images:

```bash
mcpgpu-hip:nvidia   # CUDA 12.8 + HIP NVIDIA backend
mcpgpu-hip:amd-wsl  # ROCm AMD backend for WSL/ROCDXG
```
## Build NVIDIA image
```bash
docker build -f docker/nvidia.Dockerfile -t mcpgpu-hip:nvidia .
docker run --rm --gpus all mcpgpu-hip:nvidia nvidia-smi #test
#run example
docker run --rm -it --gpus all \
  -v "$PWD":/work \
  -w /work/experiments/000_cuda_basics \
  mcpgpu-hip:nvidia \
  bash -lc "make clean && make run-cuda && make run-hip-nvidia"
```
## Build AMD image
```bash
docker build -f docker/amd.Dockerfile -t mcpgpu-hip:amd .
#Test AMD GPU visibility
docker run --rm -it \
  --device=/dev/dxg \
  --security-opt seccomp=unconfined \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  -e HIP_PLATFORM=amd \
  -v /usr/local/lib/librocdxg.so:/usr/local/lib/librocdxg.so:ro \
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro \
  mcpgpu-hip:amd \
  bash -lc 'rocminfo | grep -Ei "Agent|Name:|Marketing Name|gfx" | head -80'
  #example
  docker run --rm -it \
  --device=/dev/dxg \
  --security-opt seccomp=unconfined \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  -e HIP_PLATFORM=amd \
  -v /usr/local/lib/librocdxg.so:/usr/local/lib/librocdxg.so:ro \
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro \
  -v "$PWD":/work \
  -w /work/experiments/000_cuda_basics \
  mcpgpu-hip:amd-wsl \
  bash -lc "make clean && make run-hip-amd"
```
| Group                            | Hardware                         | Backend / method | Workspace | Knot points | PCG tolerance | SQP budget | Iterations | Purpose                                  |
| -------------------------------- | -------------------------------- | ---------------- | --------: | ----------: | ------------: | ---------: | ---------: | ---------------------------------------- |
| Stress                           | c3po RTX 5060                    | HIP-NVIDIA + PCG |        ON |         128 |        `1e-5` |  `5000 us` |         20 | Tight tolerance, NVIDIA reference        |
| Stress                           | namira / Strix Halo Radeon 8060S | HIP-AMD + PCG    |        ON |         128 |        `1e-5` |  `5000 us` |         20 | Tight tolerance, AMD comparison          |
| Stress                           | namira / Strix Halo Radeon 8060S | HIP-AMD + PCG    |       OFF |         128 |        `1e-5` |  `5000 us` |         20 | Show workspace importance under pressure |
| Stress                           | c3po RTX 5060                    | HIP-NVIDIA + PCG |        ON |         256 |        `5e-5` |  `5000 us` |         10 |                                          |
| Larger horizon, NVIDIA reference |                                  |                  |           |             |               |            |            |                                          |
| Stress                           | namira / Strix Halo Radeon 8060S | HIP-AMD + PCG    |        ON |         256 |        `5e-5` |  `5000 us` |         10 
| Stress                           | namira / Strix Halo Radeon 8060S | HIP-AMD + PCG    |       OFF |         256 |        `5e-5` |  `5000 us` |         10 | Expected to expose overhead/failure      |
