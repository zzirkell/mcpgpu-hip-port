NVCC ?= nvcc
NVCC_FLAGS = -std=c++17 -O2

BIN_DIR = bin

HELLO_SRC = cuda/hello.cu
HELLO_BIN = $(BIN_DIR)/hello_cuda

ADD_SRC = cuda/unified_add.cu
ADD_BIN = $(BIN_DIR)/unified_add_cuda

.PHONY: all hello add run-hello run-add clean

all: hello add

hello: $(HELLO_BIN)

add: $(ADD_BIN)

$(HELLO_BIN): $(HELLO_SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(HELLO_SRC) -o $(HELLO_BIN)

$(ADD_BIN): $(ADD_SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(ADD_SRC) -o $(ADD_BIN)

run-hello: hello
	./$(HELLO_BIN)

run-add: add
	./$(ADD_BIN)

clean:
	rm -rf $(BIN_DIR)