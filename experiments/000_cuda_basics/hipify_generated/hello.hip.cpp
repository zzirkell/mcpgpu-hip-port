#include <cstdio>
#include <hip/hip_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                             \
        if (err != hipSuccess) {                                           \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, hipGetErrorString(err));      \
            return 1;                                                       \
        }                                                                  \
    } while (0)

__global__
void hello_from_gpu() {
    printf("Hello from GPU! block=%d thread=%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello_from_gpu<<<2, 4>>>();

    CHECK_GPU(hipGetLastError());
    CHECK_GPU(hipDeviceSynchronize());

    return 0;
}