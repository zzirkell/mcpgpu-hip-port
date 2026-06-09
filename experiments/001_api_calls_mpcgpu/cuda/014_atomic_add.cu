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

__global__ void atomicAddKernel(int *counter) {
    atomicAdd(counter, 1);
}

int main() {
    const int num_threads = 256;
    const int num_blocks = 4;
    const int total_threads = num_threads * num_blocks;

    int h_counter = 0;
    int *d_counter;

    CHECK_GPU(cudaMalloc(&d_counter, sizeof(int)));
    CHECK_GPU(cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice));

    atomicAddKernel<<<num_blocks, num_threads>>>(d_counter);
    CHECK_GPU(cudaGetLastError());
    CHECK_GPU(cudaDeviceSynchronize());

    CHECK_GPU(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_counter == total_threads) {
        std::cout << "OK: 014_atomic_add" << std::endl;
    } else {
        std::cout << "FAIL: 014_atomic_add Expected " << total_threads << " but got " << h_counter << std::endl;
    }

    CHECK_GPU(cudaFree(d_counter));

    return 0;
}