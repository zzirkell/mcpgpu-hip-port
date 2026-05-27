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
void write_value_kernel(int *out) {
    out[0] = 123;
}

int main() {
    int h_out = 0;
    int *d_out = nullptr;

    CHECK_GPU(cudaMalloc((void**)&d_out, sizeof(int)));

    write_value_kernel<<<1, 1>>>(d_out);

    CHECK_GPU(cudaPeekAtLastError());
    CHECK_GPU(cudaDeviceSynchronize());

    CHECK_GPU(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_GPU(cudaFree(d_out));

    if (h_out != 123) {
        std::fprintf(stderr, "Wrong value: expected 123, got %d\n", h_out);
        return 1;
    }

    std::printf("OK: error/sync test passed, value = %d\n", h_out);
    return 0;
}

