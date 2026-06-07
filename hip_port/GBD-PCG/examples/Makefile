NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -I../include -I../GLASS  

examples: pcg.exe pcg_dp.exe


pcg.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve.cu -o pcg.exe 

pcg_dp.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve_dp.cu -o pcg_dp.exe

clean:
	rm -rf *.exe
	
HIPCC = /opt/rocm/bin/hipcc
HIPFLAGS = -std=c++17 -O3 --offload-arch=gfx1101 -I../include -I../GLASS -DKNOT_POINTS=3 -DSTATE_SIZE=2

pcg_hip.exe:
	$(HIPCC) $(HIPFLAGS) pcg_solve.hip.cpp -o pcg_hip.exe

pcg_dp_hip.exe:
	$(HIPCC) $(HIPFLAGS) pcg_solve_dp.hip.cpp -o pcg_dp_hip.exe

hip: pcg_hip.exe pcg_dp_hip.exe

run-hip: hip
	./pcg_hip.exe
	./pcg_dp_hip.exe