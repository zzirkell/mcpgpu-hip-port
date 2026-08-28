#!/usr/bin/env bash
set -euo pipefail

ARCH=$1
KNOTS=$2
BUDGET_US=$3
WORKSPACE_FLAG=$4
SOLVER=$5
TEST_ITERS=$6
TOLERANCE=$7

echo "============================================================"
echo "TIMING ANALYSIS: Backend: $ARCH | Knots: $KNOTS | Solver: $SOLVER"
echo "============================================================"

rm -rf tmp/results
mkdir -p tmp/results

# 1. LOAD ENVIRONMENT
if [ "$ARCH" == "cuda" ]; then
    source "$HOME/mpcgpu_project/env_cuda_c3po.sh"
elif [ "$ARCH" == "nv_hip" ]; then
    source "$HOME/mpcgpu_project/env_nvhip_c3po.sh"
    export HIP_PLATFORM="nvidia"
elif [ "$ARCH" == "amd_hip" ]; then
    if [ -f "$HOME/mpcgpu_project/env_amdhip_mi300_ftp.sh" ]; then
        source "$HOME/mpcgpu_project/env_amdhip_mi300_ftp.sh"
    fi
    export HIP_PLATFORM="amd"
    : "${ROCM_ROOT:=${ROCM_PATH:-/opt/rocm-6.4.1}}"
    export ROCM_ROOT ROCM_PATH HIP_PATH
fi

AMD_HIP_ARCH_FLAGS=""
if [ "$ARCH" == "amd_hip" ]; then
    AMD_HIP_ARCH_FLAGS="--offload-arch=gfx942"
fi

DEFINES="-DKNOT_POINTS=${KNOTS} -DSQP_MAX_TIME_US=${BUDGET_US} -DSAVE_DATA=1 -DTEST_ITERS=${TEST_ITERS} -DTIME_LINSYS=1 -DFINE_GRAINED_TIMING=1 ${WORKSPACE_FLAG}" 

if [ "$SOLVER" == "pcg" ]; then
    DEFINES="$DEFINES -DLINSYS_SOLVE=1"
elif [ "$SOLVER" == "qdldl" ]; then
    DEFINES="$DEFINES -DLINSYS_SOLVE=0"
elif [ "$SOLVER" == "hybrid" ]; then
    DEFINES="$DEFINES -DUSE_HYBRID=1"
else
    echo "Unknown solver: $SOLVER"
    exit 1
fi

export LD_LIBRARY_PATH="$(pwd)/qdldl/build/out:$(pwd)/qdldl/build:$LD_LIBRARY_PATH"

# 2. COMPILE & RUN
if [ "$ARCH" == "cuda" ]; then
    cmake -S qdldl -B qdldl/build -DQDLDL_FLOAT=true -DQDLDL_LONG=false -DQDLDL_BUILD_SHARED_LIB=OFF -DQDLDL_BUILD_DEMO_EXE=OFF -DQDLDL_UNITTESTS=OFF
    cmake --build qdldl/build --parallel

    nvcc -std=c++17 -O3 --compiler-options -Wall $DEFINES \
        -Iinclude -Iinclude/common -IGLASS -IGBD-PCG/include -Iqdldl/include \
        examples/track_iiwa_${SOLVER}.cu -o examples/${SOLVER}_cuda_timing.exe \
        -Lqdldl/build/out -lqdldl -L"$CUDA_HOME/lib64" -lcublas
    
    ./examples/${SOLVER}_cuda_timing.exe $TOLERANCE

else
    cmake -S qdldl -B qdldl/build -DQDLDL_FLOAT=true -DQDLDL_LONG=false -DQDLDL_BUILD_SHARED_LIB=OFF -DQDLDL_BUILD_DEMO_EXE=OFF -DQDLDL_UNITTESTS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build qdldl/build --parallel

    INCLUDES="-Iinclude -Iinclude/common -Iinclude/utils -Iinclude/pcg -Iinclude/qdldl -Iinclude/dynamics -Iinclude/dynamics/iiwa -IGLASS -IGBD-PCG/include -Iqdldl/include"
    
    if [ "$ARCH" == "nv_hip" ]; then
        INCLUDES="$INCLUDES -I$CUDA_HOME/include -I$ROCM_ROOT/include/hipblas -I$ROCM_ROOT/include"
        LINKS="-Lqdldl/build/out -lqdldl -L$CUDA_HOME/lib64 -lcublas -lcudart"
    else
        : "${ROCM_ROOT:=${ROCM_PATH:-/opt/rocm-6.4.1}}"
        INCLUDES="$INCLUDES -I$ROCM_ROOT/include -I$ROCM_ROOT/include/hipblas"
        LINKS="-Lqdldl/build/out -lqdldl -L$ROCM_ROOT/lib -lhipblas -lrocblas"
    fi

    hipcc -std=c++17 -O3 $AMD_HIP_ARCH_FLAGS $DEFINES \
        $INCLUDES \
        examples/track_iiwa_${SOLVER}.hip.cpp -o examples/${SOLVER}_hip_timing.exe \
        $LINKS
        
    ./examples/${SOLVER}_hip_timing.exe $TOLERANCE
fi