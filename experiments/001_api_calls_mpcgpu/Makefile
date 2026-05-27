CUDA_NATIVE ?= /usr/local/cuda
CUDA_FOR_HIP ?= $(CUDA_NATIVE)
ROCM_PATH ?= /opt/rocm-7.2.0

NVCC ?= $(CUDA_NATIVE)/bin/nvcc
HIPCC ?= hipcc
HIPIFY ?= hipify-perl

NVCC_FLAGS = -std=c++17 -O2
HIPCC_FLAGS = -std=c++17 -O2

BIN_DIR = bin
HIPIFY_DIR = hipify_generated
CUDA_DIR = cuda

AMD_ARCH ?= gfx1103

TEST_NAME ?= 001_basic_malloc_memcpy_free

CUDA_SRC = $(CUDA_DIR)/$(TEST_NAME).cu
HIP_SRC = $(HIPIFY_DIR)/$(TEST_NAME).hip.cpp

CUDA_BIN = $(BIN_DIR)/$(TEST_NAME)_cuda
HIP_NVIDIA_BIN = $(BIN_DIR)/$(TEST_NAME)_hip_nvidia
HIP_AMD_BIN = $(BIN_DIR)/$(TEST_NAME)_hip_amd

TESTS = $(basename $(notdir $(wildcard $(CUDA_DIR)/*.cu)))

# Default: most tests do not need extra libraries.
CUDA_LDLIBS =
HIP_NVIDIA_LDLIBS =
HIP_AMD_LDLIBS =
SKIP_HIP_NVIDIA = 0

# Test-specific libraries.
# 016 tests cuBLAS -> hipBLAS translation.
ifeq ($(TEST_NAME),016_blas_axpy)
CUDA_LDLIBS += -L$(CUDA_NATIVE)/lib64 -lcublas

# HIPIFY generates: #include <hipblas.h>
# On ROCm 7.2 this file is inside:
# $(ROCM_PATH)/include/hipblas/hipblas.h
HIPCC_FLAGS += -I$(ROCM_PATH)/include/hipblas

HIP_NVIDIA_LDLIBS += -L$(ROCM_PATH)/lib -L$(CUDA_FOR_HIP)/lib64 -lhipblas -lcublas
HIP_AMD_LDLIBS += -L$(ROCM_PATH)/lib -lhipblas -lrocblas

# Local ROCm apt hipBLAS is not useful for NVIDIA backend testing here.
SKIP_HIP_NVIDIA = 1
endif

HIP_NVIDIA_ENV = env \
	CUDA_PATH=$(CUDA_FOR_HIP) \
	CUDA_HOME=$(CUDA_FOR_HIP) \
	CUDACXX=$(CUDA_FOR_HIP)/bin/nvcc \
	HIP_PLATFORM=nvidia \
	PATH="$(ROCM_PATH)/bin:$(CUDA_FOR_HIP)/bin:$$PATH" \
	LD_LIBRARY_PATH="$(ROCM_PATH)/lib:$(CUDA_FOR_HIP)/lib64"

HIP_AMD_ENV = env \
	-u CUDA_PATH \
	-u CUDA_HOME \
	-u CUDACXX \
	-u HIP_COMPILER \
	-u HIP_RUNTIME \
	HIP_PLATFORM=amd \
	HSA_ENABLE_DXG_DETECTION=1 \
	ROCM_PATH=$(ROCM_PATH) \
	HIP_PATH=$(ROCM_PATH) \
	ROCBLAS_TENSILE_LIBPATH="$(ROCM_PATH)/lib/rocblas/library" \
	PATH="$(ROCM_PATH)/bin:$$PATH" \
	LD_LIBRARY_PATH="$(ROCM_PATH)/lib:/usr/lib/wsl/lib"

.PHONY: all list \
	cuda hipify hip hip-nvidia hip-amd \
	run-cuda run-hip run-hip-nvidia run-hip-amd run-all \
	check-hip-amd check-hip-nvidia \
	diff run-all-tests \
	clean clean-generated clean-all

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
	$(NVCC) $(NVCC_FLAGS) $(CUDA_SRC) -o $(CUDA_BIN) $(CUDA_LDLIBS)

$(HIP_SRC): $(CUDA_SRC)
	mkdir -p $(HIPIFY_DIR)
	$(HIPIFY) $(CUDA_SRC) > $(HIP_SRC)

$(HIP_NVIDIA_BIN): $(HIP_SRC)
	mkdir -p $(BIN_DIR)
	$(HIP_NVIDIA_ENV) $(HIPCC) $(HIPCC_FLAGS) $(HIP_SRC) -o $(HIP_NVIDIA_BIN) $(HIP_NVIDIA_LDLIBS)

$(HIP_AMD_BIN): $(HIP_SRC)
	mkdir -p $(BIN_DIR)
	$(HIP_AMD_ENV) $(HIPCC) $(HIPCC_FLAGS) --offload-arch=$(AMD_ARCH) $(HIP_SRC) -o $(HIP_AMD_BIN) $(HIP_AMD_LDLIBS)

run-cuda: cuda
	./$(CUDA_BIN)

run-hip: run-hip-nvidia

run-hip-nvidia:
ifeq ($(SKIP_HIP_NVIDIA),1)
	@echo "SKIP: HIP NVIDIA run disabled for $(TEST_NAME)"
else
	$(MAKE) hip-nvidia TEST_NAME=$(TEST_NAME)
	$(HIP_NVIDIA_ENV) ./$(HIP_NVIDIA_BIN)
endif

run-hip-amd: hip-amd
	$(HIP_AMD_ENV) ./$(HIP_AMD_BIN)

run-all: run-cuda run-hip-nvidia run-hip-amd

check-hip-amd:
	$(HIP_AMD_ENV) hipconfig --full | sed -n '1,100p'
	$(HIP_AMD_ENV) rocminfo | grep -Ei "Agent|Name:|Marketing Name|Device Type|gfx" | head -100

check-hip-nvidia:
	$(HIP_NVIDIA_ENV) hipconfig --full | sed -n '1,100p'

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

clean-all: clean clean-generated