#include "hip/hip_runtime.h"
#include <cstdio>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

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
void cooperative_block_kernel(int *block_sums, int *block_sizes) {
    cg::thread_block block = cg::this_thread_block();

    __shared__ int values[256];

    int rank = block.thread_rank();
    int size = block.size();

    values[rank] = blockIdx.x * 100 + rank + 1;

    block.sync();

    if (rank == 0) {
        int sum = 0;

        for (int i = 0; i < size; i++) {
            sum += values[i];
        }

        block_sums[blockIdx.x] = sum;
        block_sizes[blockIdx.x] = size;
    }
}

int main() {
    const int blocks = 3;
    const int threads_per_block = 16;

    std::vector<int> h_sums(blocks);
    std::vector<int> h_sizes(blocks);

    int *d_sums = nullptr;
    int *d_sizes = nullptr;

    CHECK_GPU(hipMalloc((void**)&d_sums, blocks * sizeof(int)));
    CHECK_GPU(hipMalloc((void**)&d_sizes, blocks * sizeof(int)));

    cooperative_block_kernel<<<blocks, threads_per_block>>>(d_sums, d_sizes);

    CHECK_GPU(hipPeekAtLastError());
    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(h_sums.data(), d_sums, blocks * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_GPU(hipMemcpy(h_sizes.data(), d_sizes, blocks * sizeof(int), hipMemcpyDeviceToHost));

    CHECK_GPU(hipFree(d_sums));
    CHECK_GPU(hipFree(d_sizes));

    for (int block = 0; block < blocks; block++) {
        int expected_sum = 0;

        for (int rank = 0; rank < threads_per_block; rank++) {
            expected_sum += block * 100 + rank + 1;
        }

        if (h_sizes[block] != threads_per_block) {
            std::fprintf(stderr,
                         "Wrong block size for block %d: expected %d, got %d\n",
                         block, threads_per_block, h_sizes[block]);
            return 1;
        }

        if (h_sums[block] != expected_sum) {
            std::fprintf(stderr,
                         "Wrong sum for block %d: expected %d, got %d\n",
                         block, expected_sum, h_sums[block]);
            return 1;
        }
    }

    std::printf("OK: cooperative groups block test passed, blocks = %d, threads/block = %d\n",
                blocks, threads_per_block);

    return 0;
}
