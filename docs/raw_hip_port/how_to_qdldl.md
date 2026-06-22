Summary of the Porting Process

We successfully ported, linked, and executed your part of the MPCGPU project (NMPC wrapper, GRiD dynamics) alongside the C-based QDLDL solver on the AMD HIP backend.

Encountered Errors & Solutions:

    Missing Headers (gpuassert.hip.hpp, glass.hip.hpp): The compiler could not find these files. Solution: We added their respective directories to the compiler's include path (-I).

    UTF-8 Compiler Error with libqdldl.a: The compiler tried to read the compiled binary library as text source code. Solution: We removed the file from the source arguments and correctly passed it as a linker flag (-L... -lqdldl).

    Shared Library Runtime Error (libqdldl.so): Linux could not find the dynamic library when starting the program. Solution: We exported the path to the library for the current session (export LD_LIBRARY_PATH=...).

    QDLDL Type Mismatch: The C library defaulted to 64-bit types (double/long long), but the GPU code expected 32-bit types (float/int). Solution: We deleted the old build and forced CMake to use 32-bit C++ types by passing -DQDLDL_FLOAT=ON -DQDLDL_LONG=OFF.

Step-by-Step Guide (For Next Time)

If you ever need to port or update a C/C++ solver like QDLDL in the future, follow this exact sequence:

Step 1: Build the C library with the correct data types
Navigate to the QDLDL directory, create a clean build folder, and configure CMake to match your GPU's data types (e.g., 32-bit):
Bash

mkdir temp_build && cd temp_build
cmake -DQDLDL_FLOAT=ON -DQDLDL_LONG=OFF ..
make

Step 2: Hipify the CUDA example
Return to the project root directory and translate the execution code:
Bash

hipify-perl examples/track_iiwa_qdldl.cu > examples/track_iiwa_qdldl.hip.cpp

Step 3: Setup the Include network
Dynamically collect all directories containing header files:
Bash

INC_DIRS=$(find include examples GBD-PCG/include GLASS -type d -printf "-I%p ")

Step 4: Compile and Link
Run the compiler. Important: Explicitly pass the include paths (-I), the library path (-L), and the library name (-lqdldl):
Bash

/opt/rocm/bin/hipcc -DLINSYS_SOLVE=2 -std=c++17 -O2 $INC_DIRS -Iqdldl/include -I/opt/rocm-7.2.0/include/hipblas examples/track_iiwa_qdldl.hip.cpp -o track_iiwa_qdldl.exe -Lqdldl/temp_build/out -lqdldl -lhipblas

Step 5: Set the environment variable and run
Tell the system where the compiled .so library is located before starting the executable:
Bash

export LD_LIBRARY_PATH=$PWD/qdldl/temp_build/out:$LD_LIBRARY_PATH
./track_iiwa_qdldl.exe