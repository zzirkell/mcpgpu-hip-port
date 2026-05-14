# ROCm AMD Backend Setup on WSL

This setup is for running HIP code on the AMD GPU/iGPU from WSL.

Tested GPU:

```text
AMD Radeon 780M Graphics
ROCm architecture: gfx1103
````

## Required system

Use WSL Ubuntu **22.04 or newer**.

Check:

```bash
lsb_release -a
ls /dev/dxg
```

Expected:

```text
Ubuntu 22.04 or newer
/dev/dxg
```

> [!NOTE]
> `/dev/dxg` is the WSL GPU bridge used by ROCDXG.

## Install AMD ROCm WSL packages

```bash
cd ~
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb

sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
sudo apt install -y rocminfo rocm-device-libs
```

## Install Windows SDK

ROCDXG needs Windows SDK headers.

In Windows PowerShell:

```powershell
winget install -e --id Microsoft.WindowsSDK.10.0.26100
```

Back in WSL, check:

```bash
ls -d "/mnt/c/Program Files (x86)/Windows Kits/10/Include/"*
```

## Build ROCDXG

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config

mkdir -p ~/tools
cd ~/tools

git clone https://github.com/ROCm/librocdxg.git
cd librocdxg

export win_sdk=$(find "/mnt/c/Program Files (x86)/Windows Kits/10/Include" -maxdepth 1 -mindepth 1 -type d | sort -V | tail -n 1)

echo "$win_sdk"
ls "$win_sdk/shared"

mkdir -p build
cd build

cmake .. -DWIN_SDK="${win_sdk}/shared"
make -j$(nproc)
sudo make install
```

## Set AMD ROCm environment

For AMD backend tests:

```bash
unset CUDA_PATH
unset CUDA_HOME
unset CUDACXX

export HIP_PLATFORM=amd
export HSA_ENABLE_DXG_DETECTION=1
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

Check HIP backend:

```bash
hipconfig --full | grep -E "HIP_PLATFORM|HIP_COMPILER|HIP_RUNTIME|ROCM_PATH|HIP_PATH"
```

Expected:

```text
HIP_PLATFORM: amd
HIP_COMPILER: clang
HIP_RUNTIME: rocclr
```

## Verify AMD GPU visibility

```bash
rocminfo | grep -Ei "Agent|Name:|Marketing Name|gfx" | head -80
```

## Compile HIP for AMD manually

Example with basic hello:

```bash
cd ~/mcpgpu-hip-port/experiments/000_cuda_basics
mkdir -p bin

HIP_PLATFORM=amd HSA_ENABLE_DXG_DETECTION=1 \
hipcc -std=c++17 -O2 --offload-arch=gfx1103 \
hipify_generated/hello.hip.cpp \
-o bin/hello_hip_amd

HSA_ENABLE_DXG_DETECTION=1 ./bin/hello_hip_amd
```
