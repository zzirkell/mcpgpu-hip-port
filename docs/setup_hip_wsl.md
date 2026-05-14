# HIP Setup on WSL Ubuntu
> [!NOTE]
> For AMD backend setup, see:
> docs/setup_rocm_amd_wsl.md 

## Required system

Use WSL Ubuntu **22.04 or newer**, hipcc does not work on lower verisons!!!

Tested setup:

```text
Ubuntu 22.04.5 LTS
HIP 6.2.4
HIP_PLATFORM=nvidia for NVIDIA backend
HIP_PLATFORM=amd for AMD backend
CUDA used by HIP: 12.8
````

## Basic checks

```bash
which hipcc
hipcc --version
hipconfig --full
```

Expected output:

```text
HIP_PLATFORM: nvidia
HIP_COMPILER: nvcc
HIP_RUNTIME: cuda
CUDA version used by hipcc: 12.8
```

## Install HIP for NVIDIA backend

```bash
cd ~

wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb
sudo apt install -y ./amdgpu-install_6.2.60204-1_all.deb

sudo apt update
sudo apt install -y hip-runtime-nvidia hip-dev
```

## Set NVIDIA HIP environment

```bash
nano ~/.bashrc
export CUDA_PATH=/usr/local/cuda-12.8
export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/opt/rocm/bin:/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export HIP_PLATFORM=nvidia
source ~/.bashrc #reload
```

Check:

```bash
which hipcc
hipcc --version
hipconfig --full
```

> [!IMPORTANT]
> HIP 6.2.4 did not compile correctly with CUDA 13.x headers on this machine.
> Use CUDA 12.8 for HIP NVIDIA backend compilation.

## HIPIFY translator installation

```bash
sudo apt update
sudo apt install -y git perl
mkdir -p ~/tools
cd ~/tools
git clone https://github.com/ROCm/HIPIFY.git #Clone HIPIFY
```

Expose `hipify-perl`:

```bash
mkdir -p ~/.local/bin
HIPIFY_PERL_PATH=$(find ~/tools/HIPIFY -name "hipify-perl" -type f | head -n 1)
chmod +x "$HIPIFY_PERL_PATH"
#Add local bin to PATH if needed:
ln -sf "$HIPIFY_PERL_PATH" ~/.local/bin/hipify-perl
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Check:

```bash
which hipify-perl
hipify-perl --help | head
```

## Generate HIP code

From the CUDA basics experiment:

```bash
cd ~/mcpgpu-hip-port/experiments/000_cuda_basics
mkdir -p hipify_generated

hipify-perl cuda/hello.cu > hipify_generated/hello.hip.cpp
hipify-perl cuda/unified_add.cu > hipify_generated/unified_add.hip.cpp
```

Compare changes:

```bash
diff -u cuda/hello.cu hipify_generated/hello.hip.cpp || true
diff -u cuda/unified_add.cu hipify_generated/unified_add.hip.cpp || true
```

## Compile and run HIP examples
```bash
cd ~/mpcgpu_hip_port/experiments/000_cuda_basics
make
make hip
make run-hip
#don't forget
make clean
```
### Hipify your programs yourself
```bash
hipify-perl path/to/input.cu > path/to/output.hip.cpp
```

### Compile and run your programs yourself
NVIDIA backend:

```bash
HIP_PLATFORM=nvidia hipcc -std=c++17 -O2 path/to/file.hip.cpp -o output
./output
```

AMD backend:

```bash
HIP_PLATFORM=amd HSA_ENABLE_DXG_DETECTION=1 \
hipcc -std=c++17 -O2 --offload-arch=gfx1103 path/to/file.hip.cpp -o output

HSA_ENABLE_DXG_DETECTION=1 ./output
```
> [!NOTE]
> For AMD backend setup, see:
> docs/setup_rocm_amd_wsl.md 