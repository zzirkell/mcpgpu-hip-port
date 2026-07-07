# Makefile

# Compiler and compiler flags

CXX = /opt/rocm/bin/hipcc

PROJECT_INCS = $(shell find include examples GBD-PCG/include GLASS -type d 2>/dev/null | sed s/^/-I/)
EXT_INCS = -Iqdldl/include -I/opt/rocm-7.2.0/include/hipblas

ALL_CXXFLAGS = -std=c++17 -O3 $(PROJECT_INCS) $(EXT_INCS) $(CXXFLAGS)

LDFLAGS = -Lqdldl/temp_build/out -lqdldl -lhipblas

EXAMPLES = examples/pcg.exe examples/qdldl.exe

.PHONY: all clean examples

all: examples

examples: $(EXAMPLES)

examples/pcg.exe: examples/track_iiwa_pcg.hip.cpp
	$(CXX) $(ALL_CXXFLAGS) $< -o $@ $(LDFLAGS)

examples/qdldl.exe: examples/track_iiwa_qdldl.hip.cpp
	$(CXX) $(ALL_CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f examples/*.exe
	rm -rf tmp/results/*