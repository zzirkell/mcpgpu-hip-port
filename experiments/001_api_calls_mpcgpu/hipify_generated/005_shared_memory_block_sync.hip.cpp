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
void shared_memory_kernel(int *out, int n) {
    __shared__ int static_shared[256];
    extern __shared__ int dynamic_shared[];

    int tid = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_index < n) {
        static_shared[tid] = global_index;
        dynamic_shared[tid] = global_index * 10;
    }

    __syncthreads();

    if (global_index < n) {
        int reversed_tid = blockDim.x - 1 - tid;

        int a = static_shared[reversed_tid];
        int b = dynamic_shared[reversed_tid];

        out[global_index] = a + b;
    }
}

int main() {
    const int blocks = 2;
    const int threads_per_block = 16;
    const int n = blocks * threads_per_block;

    const size_t dynamic_shared_bytes = threads_per_block * sizeof(int);

    std::vector<int> h_out(n);

    int *d_out = nullptr;

    CHECK_GPU(hipMalloc((void**)&d_out, n * sizeof(int)));

    shared_memory_kernel<<<blocks, threads_per_block, dynamic_shared_bytes>>>(d_out, n);

    CHECK_GPU(hipPeekAtLastError());
    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(h_out.data(), d_out, n * sizeof(int), hipMemcpyDeviceToHost));

    CHECK_GPU(hipFree(d_out));

    for (int block = 0; block < blocks; block++) {
        for (int tid = 0; tid < threads_per_block; tid++) {
            int global_index = block * threads_per_block + tid;
            int reversed_tid = threads_per_block - 1 - tid;
            int source_index = block * threads_per_block + reversed_tid;

            int expected = source_index + source_index * 10;

            if (h_out[global_index] != expected) {
                std::fprintf(stderr,
                             "Wrong value at global_index=%d: expected %d, got %d\n",
                             global_index, expected, h_out[global_index]);
                return 1;
            }
        }
    }

    std::printf("OK: shared memory/block sync test passed, n = %d\n", n);
    return 0;
}
