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

int main() {
    int h_in = 42;
    int h_out = 0;
    int *d_value = nullptr;

    CHECK_GPU(cudaMalloc((void**)&d_value, sizeof(int)));

    CHECK_GPU(cudaMemcpy(d_value, &h_in, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_GPU(cudaMemcpy(&h_out, d_value, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_GPU(cudaFree(d_value));

    if (h_out != h_in) {
        std::fprintf(stderr, "Wrong result: expected %d, got %d\n", h_in, h_out);
        return 1;
    }

    std::printf("OK: memory API test passed, value = %d\n", h_out);
    return 0;
}