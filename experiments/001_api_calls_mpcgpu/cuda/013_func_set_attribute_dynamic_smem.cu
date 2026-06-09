#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err));     \
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

    CHECK_GPU(cudaMalloc(&d_out, smem_size));

    CHECK_GPU(cudaFuncSetAttribute((const void*)dynamicSmemKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dynamicSmemKernel<<<1, num_threads, smem_size>>>(d_out);
    CHECK_GPU(cudaGetLastError());
    CHECK_GPU(cudaDeviceSynchronize());

    CHECK_GPU(cudaMemcpy(h_out, d_out, smem_size, cudaMemcpyDeviceToHost));

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

    CHECK_GPU(cudaFree(d_out));
    return 0;
}