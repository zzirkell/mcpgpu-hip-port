#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err));      \
            return 1;                                                       \
        }                                                                  \
    } while (0)

__global__
void hello_from_gpu() {
    printf("Hello from GPU! block=%d thread=%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello_from_gpu<<<2, 4>>>();

    CHECK_GPU(cudaGetLastError());
    CHECK_GPU(cudaDeviceSynchronize());

    return 0;
}