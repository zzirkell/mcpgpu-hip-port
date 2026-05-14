# CUDA Setup on WSL Ubuntu

## Basic checks

We expect these packages to be installed and working correctly. Run inside WSL/Linux:

```bash
nvidia-smi #CUDA there does not mean you have it installed (check the version your NVIDIA needs)
nvcc --version
gcc --version
make --version
```

## If something is missing
```bash
#basics
sudo apt update
sudo apt install -y build-essential git cmake wget make

#for CUDA 13-1
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-1

#setup PATH smth like:
nano ~/.bashrc
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

#check how it went:
nvcc --version
```

### CUDA note
CUDA 13/libcu++ may require C++17 for some projects (including MPCGPU):
```bash
-std=c++17 # example for MPCGPU:
NVCC_FLAGS = -std=c++17 -O2
```
You can test your CUDA setup by tunning experiments from ```./mcpgpu-hip-port/experiments/000_cuda_basics``` folder using Makefile:
```bash
cd ~/mpcgpu_hip_port/experiments/000_cuda_basics
make
make run-hello #or
make run-add
#don't forget
make clean
```