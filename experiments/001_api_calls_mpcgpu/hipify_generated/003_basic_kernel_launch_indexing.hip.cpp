#include "hip/hip_runtime.h"
#include <cstdio>
#include <vector>
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
void write_global_indices_kernel(int *out, int *meta, int n) {
    int local_thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    int block_index = blockIdx.y * gridDim.x + blockIdx.x;
    int global_thread_index = block_index * threads_per_block + local_thread_index;

    if (global_thread_index < n) {
        out[global_thread_index] = global_thread_index;
    }

    if (global_thread_index == 0) {
        meta[0] = blockDim.x;
        meta[1] = blockDim.y;
        meta[2] = gridDim.x;
        meta[3] = gridDim.y;
    }
}

int main() {
    dim3 block(4, 2); // 8 threads per block
    dim3 grid(3, 2);  // 6 blocks

    const int n = block.x * block.y * grid.x * grid.y;

    std::vector<int> h_out(n);
    std::vector<int> h_meta(4);

    int *d_out = nullptr;
    int *d_meta = nullptr;

    CHECK_GPU(hipMalloc((void**)&d_out, n * sizeof(int)));
    CHECK_GPU(hipMalloc((void**)&d_meta, 4 * sizeof(int)));

    write_global_indices_kernel<<<grid, block>>>(d_out, d_meta, n);

    CHECK_GPU(hipPeekAtLastError());
    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(h_out.data(), d_out, n * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_GPU(hipMemcpy(h_meta.data(), d_meta, 4 * sizeof(int), hipMemcpyDeviceToHost));

    CHECK_GPU(hipFree(d_out));
    CHECK_GPU(hipFree(d_meta));

    for (int i = 0; i < n; i++) {
        if (h_out[i] != i) {
            std::fprintf(stderr, "Wrong value at %d: expected %d, got %d\n",
                         i, i, h_out[i]);
            return 1;
        }
    }

    if (h_meta[0] != 4 || h_meta[1] != 2 || h_meta[2] != 3 || h_meta[3] != 2) {
        std::fprintf(stderr,
                     "Wrong metadata: block=(%d,%d), grid=(%d,%d)\n",
                     h_meta[0], h_meta[1], h_meta[2], h_meta[3]);
        return 1;
    }

    std::printf("OK: kernel launch/indexing test passed, n = %d\n", n);
    std::printf("Observed block=(%d,%d), grid=(%d,%d)\n",
                h_meta[0], h_meta[1], h_meta[2], h_meta[3]);

    return 0;
}