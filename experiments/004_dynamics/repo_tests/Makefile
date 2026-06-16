
HIPCC = /opt/rocm/bin/hipcc

# -I../src_hip/include ist der Basis-Ordner. 
# Mit `find ... -printf` ziehen wir alle Unterordner dynamisch dazu:
INC_DIRS := $(shell find ../src_hip -type d -printf "-I%p ")
# -include erzwingt, dass diese Header-Datei in JEDE Quelldatei eingebunden wird
CXXFLAGS = -std=c++17 -O2 $(INC_DIRS)

AMD_ARCH ?= gfx1101

all: t1 t2 t3 t4t5

# ------------------------------------------
# T1: matrix.cuh & cudaMemcpy2D Test
# ------------------------------------------
t1: t1_matrix_test.exe

t1_matrix_test.exe: t1_matrix_test.hip.cpp
	$(HIPCC) $(CXXFLAGS) t1_matrix_test.hip.cpp -o t1_matrix_test.exe

run-t1: t1
	./t1_matrix_test.exe

# ------------------------------------------
# T2: dz.cuh (Trajectory update kernel compile test)
# ------------------------------------------
t2: t2_dz_test.exe

t2_dz_test.exe: t2_dz_test.hip.cpp
	$(HIPCC) $(CXXFLAGS) t2_dz_test.hip.cpp -o t2_dz_test.exe

run-t2: t2
	./t2_dz_test.exe

# ------------------------------------------
# T3: kkt.cuh (KKT submatrix generation compile test)
# ------------------------------------------
t3: t3_kkt_test.exe

t3_kkt_test.exe: t3_kkt_test.hip.cpp
	$(HIPCC) $(CXXFLAGS) t3_kkt_test.hip.cpp -o t3_kkt_test.exe

run-t3: t3
	./t3_kkt_test.exe

# T4 & T5: Compile-only
t4t5: 
	$(HIPCC) $(CXXFLAGS) -c t4_t5_compile_tests.hip.cpp -o t4t5.o


run-all: run-t1 run-t2 run-t3


clean:
	rm -f *.exe

.PHONY: all clean t1 run-t1 t2 run-t2 t3 run-t3 run-all