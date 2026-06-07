#include "hip/hip_runtime.h"
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
void write_value_kernel(int *out) {
    out[0] = 123;
}

int main() {
    int h_out = 0;
    int *d_out = nullptr;

    CHECK_GPU(hipMalloc((void**)&d_out, sizeof(int)));

    write_value_kernel<<<1, 1>>>(d_out);

    CHECK_GPU(hipPeekAtLastError());
    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost));
    CHECK_GPU(hipFree(d_out));

    if (h_out != 123) {
        std::fprintf(stderr, "Wrong value: expected 123, got %d\n", h_out);
        return 1;
    }

    std::printf("OK: error/sync test passed, value = %d\n", h_out);
    return 0;
}

