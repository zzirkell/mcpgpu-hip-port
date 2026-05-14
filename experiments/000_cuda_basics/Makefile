NVCC ?= nvcc
HIPCC ?= hipcc
HIPIFY ?= hipify-perl

NVCC_FLAGS = -std=c++17 -O2
HIPCC_FLAGS = -std=c++17 -O2

BIN_DIR = bin
HIPIFY_DIR = hipify_generated

# HIP 6.2.4 + NVIDIA backend must use CUDA 12.8 in this setup.
CUDA_FOR_HIP = /usr/local/cuda-12.8
HIP_ENV = CUDA_PATH=$(CUDA_FOR_HIP) CUDA_HOME=$(CUDA_FOR_HIP) CUDACXX=$(CUDA_FOR_HIP)/bin/nvcc HIP_PLATFORM=nvidia PATH=/opt/rocm/bin:$(CUDA_FOR_HIP)/bin:$$PATH LD_LIBRARY_PATH=$(CUDA_FOR_HIP)/lib64:$$LD_LIBRARY_PATH

CUDA_HELLO_SRC = cuda/hello.cu
CUDA_HELLO_BIN = $(BIN_DIR)/hello_cuda

CUDA_ADD_SRC = cuda/unified_add.cu
CUDA_ADD_BIN = $(BIN_DIR)/unified_add_cuda

HIP_HELLO_SRC = $(HIPIFY_DIR)/hello.hip.cpp
HIP_HELLO_BIN = $(BIN_DIR)/hello_hip

HIP_ADD_SRC = $(HIPIFY_DIR)/unified_add.hip.cpp
HIP_ADD_BIN = $(BIN_DIR)/unified_add_hip

.PHONY: all cuda hip hipify run-cuda run-hello run-add run-hip run-hip-hello run-hip-add clean

all: cuda

cuda: $(CUDA_HELLO_BIN) $(CUDA_ADD_BIN)

hipify: $(HIP_HELLO_SRC) $(HIP_ADD_SRC)

hip: $(HIP_HELLO_BIN) $(HIP_ADD_BIN)

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

$(HIP_HELLO_BIN): $(HIP_HELLO_SRC)
	mkdir -p $(BIN_DIR)
	$(HIP_ENV) $(HIPCC) $(HIPCC_FLAGS) $(HIP_HELLO_SRC) -o $(HIP_HELLO_BIN)

$(HIP_ADD_BIN): $(HIP_ADD_SRC)
	mkdir -p $(BIN_DIR)
	$(HIP_ENV) $(HIPCC) $(HIPCC_FLAGS) $(HIP_ADD_SRC) -o $(HIP_ADD_BIN)

run-cuda: cuda
	./$(CUDA_HELLO_BIN)
	./$(CUDA_ADD_BIN)

run-hello: $(CUDA_HELLO_BIN)
	./$(CUDA_HELLO_BIN)

run-add: $(CUDA_ADD_BIN)
	./$(CUDA_ADD_BIN)

run-hip: hip
	./$(HIP_HELLO_BIN)
	./$(HIP_ADD_BIN)

run-hip-hello: $(HIP_HELLO_BIN)
	./$(HIP_HELLO_BIN)

run-hip-add: $(HIP_ADD_BIN)
	./$(HIP_ADD_BIN)

clean:
	rm -rf $(BIN_DIR)