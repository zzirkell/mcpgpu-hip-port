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