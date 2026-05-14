CUDA_NATIVE ?= /usr/local/cuda-12.8
CUDA_FOR_HIP ?= /usr/local/cuda-12.8

NVCC ?= $(CUDA_NATIVE)/bin/nvcc
HIPCC ?= hipcc
HIPIFY ?= hipify-perl

NVCC_FLAGS = -std=c++17 -O2
HIPCC_FLAGS = -std=c++17 -O2

BIN_DIR = bin
HIPIFY_DIR = hipify_generated

# AMD Radeon 780M iGPU architecture.
AMD_ARCH ?= gfx1103

HIP_NVIDIA_ENV = env CUDA_PATH=$(CUDA_FOR_HIP) CUDA_HOME=$(CUDA_FOR_HIP) CUDACXX=$(CUDA_FOR_HIP)/bin/nvcc HIP_PLATFORM=nvidia PATH="/opt/rocm/bin:$(CUDA_FOR_HIP)/bin:$$PATH" LD_LIBRARY_PATH="$(CUDA_FOR_HIP)/lib64:$$LD_LIBRARY_PATH"

HIP_AMD_ENV = env -u CUDA_PATH -u CUDA_HOME -u CUDACXX HIP_PLATFORM=amd HSA_ENABLE_DXG_DETECTION=1 PATH="/opt/rocm/bin:$$PATH" LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib:$$LD_LIBRARY_PATH"
CUDA_HELLO_SRC = cuda/hello.cu
CUDA_HELLO_BIN = $(BIN_DIR)/hello_cuda

CUDA_ADD_SRC = cuda/unified_add.cu
CUDA_ADD_BIN = $(BIN_DIR)/unified_add_cuda

HIP_HELLO_SRC = $(HIPIFY_DIR)/hello.hip.cpp
HIP_ADD_SRC = $(HIPIFY_DIR)/unified_add.hip.cpp

HIP_HELLO_NVIDIA_BIN = $(BIN_DIR)/hello_hip_nvidia
HIP_ADD_NVIDIA_BIN = $(BIN_DIR)/unified_add_hip_nvidia

HIP_HELLO_AMD_BIN = $(BIN_DIR)/hello_hip_amd
HIP_ADD_AMD_BIN = $(BIN_DIR)/unified_add_hip_amd

.PHONY: all cuda hipify hip hip-nvidia hip-amd run-cuda run-hello run-add run-hip run-hip-nvidia run-hip-amd clean

all: cuda

cuda: $(CUDA_HELLO_BIN) $(CUDA_ADD_BIN)

hipify: $(HIP_HELLO_SRC) $(HIP_ADD_SRC)

# Default HIP target is NVIDIA because this laptop has NVIDIA dGPU available.
hip: hip-nvidia

hip-nvidia: $(HIP_HELLO_NVIDIA_BIN) $(HIP_ADD_NVIDIA_BIN)

hip-amd: $(HIP_HELLO_AMD_BIN) $(HIP_ADD_AMD_BIN)

$(CUDA_HELLO_BIN): $(CUDA_HELLO_SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_HELLO_SRC) -o $(CUDA_HELLO_BIN)

$(CUDA_ADD_BIN): $(CUDA_ADD_SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ADD_SRC) -o $(CUDA_ADD_BIN)

$(HIP_HELLO_SRC): $(CUDA_HELLO_SRC)
	mkdir -p $(HIPIFY_DIR)
	$(HIPIFY) $(CUDA_HELLO_SRC) > $(HIP_HELLO_SRC)

$(HIP_ADD_SRC): $(CUDA_ADD_SRC)
	mkdir -p $(HIPIFY_DIR)
	$(HIPIFY) $(CUDA_ADD_SRC) > $(HIP_ADD_SRC)

$(HIP_HELLO_NVIDIA_BIN): $(HIP_HELLO_SRC)
	mkdir -p $(BIN_DIR)
	$(HIP_NVIDIA_ENV) $(HIPCC) $(HIPCC_FLAGS) $(HIP_HELLO_SRC) -o $(HIP_HELLO_NVIDIA_BIN)

$(HIP_ADD_NVIDIA_BIN): $(HIP_ADD_SRC)
	mkdir -p $(BIN_DIR)
	$(HIP_NVIDIA_ENV) $(HIPCC) $(HIPCC_FLAGS) $(HIP_ADD_SRC) -o $(HIP_ADD_NVIDIA_BIN)

$(HIP_HELLO_AMD_BIN): $(HIP_HELLO_SRC)
	mkdir -p $(BIN_DIR)
	$(HIP_AMD_ENV) $(HIPCC) $(HIPCC_FLAGS) --offload-arch=$(AMD_ARCH) $(HIP_HELLO_SRC) -o $(HIP_HELLO_AMD_BIN)

$(HIP_ADD_AMD_BIN): $(HIP_ADD_SRC)
	mkdir -p $(BIN_DIR)
	$(HIP_AMD_ENV) $(HIPCC) $(HIPCC_FLAGS) --offload-arch=$(AMD_ARCH) $(HIP_ADD_SRC) -o $(HIP_ADD_AMD_BIN)

run-cuda: cuda
	./$(CUDA_HELLO_BIN)
	./$(CUDA_ADD_BIN)

run-hello: $(CUDA_HELLO_BIN)
	./$(CUDA_HELLO_BIN)

run-add: $(CUDA_ADD_BIN)
	./$(CUDA_ADD_BIN)

# Default HIP run target is NVIDIA.
run-hip: run-hip-nvidia

run-hip-nvidia: hip-nvidia
	./$(HIP_HELLO_NVIDIA_BIN)
	./$(HIP_ADD_NVIDIA_BIN)

run-hip-amd: hip-amd
	HSA_ENABLE_DXG_DETECTION=1 ./$(HIP_HELLO_AMD_BIN)
	HSA_ENABLE_DXG_DETECTION=1 ./$(HIP_ADD_AMD_BIN)

clean:
	rm -rf $(BIN_DIR)