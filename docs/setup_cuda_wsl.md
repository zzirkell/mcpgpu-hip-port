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

**NOTE!** This project uses **CUDA 12.8** because the HIP NVIDIA backend needs it in our setup.

```bash
#basics
sudo apt update
sudo apt install -y build-essential git cmake wget make

#for CUDA 12-8
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8

#setup PATH smth like:
nano ~/.bashrc
export CUDA_PATH=/usr/local/cuda-12.8
export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
source ~/.bashrc #reload
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

### Compile and run your programs yourself
```bash
nvcc [file_to_compile] -o [output]
./[output] #to run
```