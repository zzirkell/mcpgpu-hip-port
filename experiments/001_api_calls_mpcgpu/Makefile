CUDA_NATIVE ?= /usr/local/cuda-12.8
CUDA_FOR_HIP ?= /usr/local/cuda-12.8

NVCC ?= $(CUDA_NATIVE)/bin/nvcc
HIPCC ?= hipcc
HIPIFY ?= hipify-perl

NVCC_FLAGS = -std=c++17 -O2
HIPCC_FLAGS = -std=c++17 -O2

BIN_DIR = bin
HIPIFY_DIR = hipify_generated
CUDA_DIR = cuda

AMD_ARCH ?= gfx1103

HIP_NVIDIA_ENV = env CUDA_PATH=$(CUDA_FOR_HIP) CUDA_HOME=$(CUDA_FOR_HIP) CUDACXX=$(CUDA_FOR_HIP)/bin/nvcc HIP_PLATFORM=nvidia PATH="/opt/rocm/bin:$(CUDA_FOR_HIP)/bin:$$PATH" LD_LIBRARY_PATH="$(CUDA_FOR_HIP)/lib64:$$LD_LIBRARY_PATH"

HIP_AMD_ENV = env -u CUDA_PATH -u CUDA_HOME -u CUDACXX HIP_PLATFORM=amd HSA_ENABLE_DXG_DETECTION=1 PATH="/opt/rocm/bin:$$PATH" LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib:$$LD_LIBRARY_PATH"

TEST_NAME ?= 001_basic_malloc_memcpy_free

CUDA_SRC = $(CUDA_DIR)/$(TEST_NAME).cu
HIP_SRC = $(HIPIFY_DIR)/$(TEST_NAME).hip.cpp

CUDA_BIN = $(BIN_DIR)/$(TEST_NAME)_cuda
HIP_NVIDIA_BIN = $(BIN_DIR)/$(TEST_NAME)_hip_nvidia
HIP_AMD_BIN = $(BIN_DIR)/$(TEST_NAME)_hip_amd

TESTS = $(basename $(notdir $(wildcard $(CUDA_DIR)/*.cu)))

.PHONY: all list cuda hipify hip hip-nvidia hip-amd run-cuda run-hip run-hip-nvidia run-hip-amd run-all diff run-all-tests clean clean-generated

all: cuda

list:
	@echo "Available tests:"
	@for t in $(TESTS); do echo "  $$t"; done

cuda: $(CUDA_BIN)

hipify: $(HIP_SRC)

hip: hip-nvidia

hip-nvidia: $(HIP_NVIDIA_BIN)

hip-amd: $(HIP_AMD_BIN)

$(CUDA_BIN): $(CUDA_SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_SRC) -o $(CUDA_BIN)

$(HIP_SRC): $(CUDA_SRC)
	mkdir -p $(HIPIFY_DIR)
	$(HIPIFY) $(CUDA_SRC) > $(HIP_SRC)

$(HIP_NVIDIA_BIN): $(HIP_SRC)
	mkdir -p $(BIN_DIR)
	$(HIP_NVIDIA_ENV) $(HIPCC) $(HIPCC_FLAGS) $(HIP_SRC) -o $(HIP_NVIDIA_BIN)

$(HIP_AMD_BIN): $(HIP_SRC)
	mkdir -p $(BIN_DIR)
	$(HIP_AMD_ENV) $(HIPCC) $(HIPCC_FLAGS) --offload-arch=$(AMD_ARCH) $(HIP_SRC) -o $(HIP_AMD_BIN)

run-cuda: cuda
	./$(CUDA_BIN)

run-hip: run-hip-nvidia

run-hip-nvidia: hip-nvidia
	./$(HIP_NVIDIA_BIN)

run-hip-amd: hip-amd
	HSA_ENABLE_DXG_DETECTION=1 ./$(HIP_AMD_BIN)

run-all: run-cuda run-hip-nvidia run-hip-amd

diff: hipify
	diff -u $(CUDA_SRC) $(HIP_SRC) || true

run-all-tests:
	@set -e; \
	for t in $(TESTS); do \
		echo ""; \
		echo "=============================="; \
		echo "Running $$t"; \
		echo "=============================="; \
		$(MAKE) run-all TEST_NAME=$$t; \
	done

clean:
	rm -rf $(BIN_DIR)

clean-generated:
	rm -rf $(HIPIFY_DIR)