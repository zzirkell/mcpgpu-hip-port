#include "hip/hip_runtime.h"
#include <iostream>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                            \
        if (err != hipSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, hipGetErrorString(err));     \
            return 1;                                                      \
        }                                                                  \
    } while (0)

__global__ void dynamicSmemKernel(int *out) {
    extern __shared__ int shared_data[];
    int tid = threadIdx.x;
    shared_data[tid] = tid;
    __syncthreads();
    out[tid] = shared_data[tid];
}

int main() {
    const int num_threads = 256;
    const int smem_size = num_threads * sizeof(int);
    
    int h_out[num_threads] = {0};
    int *d_out;

    CHECK_GPU(hipMalloc(&d_out, smem_size));

    CHECK_GPU(hipFuncSetAttribute((const void*)dynamicSmemKernel, hipFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dynamicSmemKernel<<<1, num_threads, smem_size>>>(d_out);
    CHECK_GPU(hipGetLastError());
    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(h_out, d_out, smem_size, hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < num_threads; ++i) {
        if (h_out[i] != i) {
            success = false;
        }
    }

    if (success) {
        std::cout << "OK: 013_func_set_attribute_dynamic_smem" << std::endl;
    } else {
        std::cout << "FAIL: 013_func_set_attribute_dynamic_smem" << std::endl;
    }

    CHECK_GPU(hipFree(d_out));
    return 0;
}