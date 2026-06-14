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

__global__ void atomicAddKernel(int *counter) {
    atomicAdd(counter, 1);
}

int main() {
    const int num_threads = 256;
    const int num_blocks = 4;
    const int total_threads = num_threads * num_blocks;

    int h_counter = 0;
    int *d_counter;

    CHECK_GPU(hipMalloc(&d_counter, sizeof(int)));
    CHECK_GPU(hipMemcpy(d_counter, &h_counter, sizeof(int), hipMemcpyHostToDevice));

    atomicAddKernel<<<num_blocks, num_threads>>>(d_counter);
    CHECK_GPU(hipGetLastError());
    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(&h_counter, d_counter, sizeof(int), hipMemcpyDeviceToHost));

    if (h_counter == total_threads) {
        std::cout << "OK: 014_atomic_add" << std::endl;
    } else {
        std::cout << "FAIL: 014_atomic_add Expected " << total_threads << " but got " << h_counter << std::endl;
    }

    CHECK_GPU(hipFree(d_counter));

    return 0;
}