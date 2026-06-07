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
void cooperative_shared_kernel(int *block_sums, int *result, int n) {
    extern __shared__ int shared[];

    cg::grid_group grid = cg::this_grid();

    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.x;

    if (global_index < n) {
        shared[local_index] = global_index + 1;
    } else {
        shared[local_index] = 0;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_index < stride) {
            shared[local_index] += shared[local_index + stride];
        }
        __syncthreads();
    }

    if (local_index == 0) {
        block_sums[blockIdx.x] = shared[0];
    }

    grid.sync();

    if (global_index == 0) {
        int total = 0;

        for (int i = 0; i < gridDim.x; i++) {
            total += block_sums[i];
        }

        result[0] = total;
    }
}

int main() {
    int device = 0;
    CHECK_GPU(hipGetDevice(&device));

    int cooperative_launch = 0;
    hipError_t attr_err = hipDeviceGetAttribute(
        &cooperative_launch,
        hipDeviceAttributeCooperativeLaunch,
        device
    );

    if (attr_err != hipSuccess || cooperative_launch == 0) {
        std::printf("SKIP: cooperative launch is not supported on this device/backend\n");
        return 0;
    }

    const int blocks = 2;
    const int threads_per_block = 32;
    const int n = blocks * threads_per_block;

    int *d_block_sums = nullptr;
    int *d_result = nullptr;

    std::vector<int> h_result(1);

    CHECK_GPU(hipMalloc((void**)&d_block_sums, blocks * sizeof(int)));
    CHECK_GPU(hipMalloc((void**)&d_result, sizeof(int)));

    void *kernel_args[] = {
        (void*)&d_block_sums,
        (void*)&d_result,
        (void*)&n
    };

    const unsigned int shared_memory_bytes = threads_per_block * sizeof(int);
    hipStream_t stream = 0;

    CHECK_GPU(hipLaunchCooperativeKernel(
        (void*)cooperative_shared_kernel,
        dim3(blocks),
        dim3(threads_per_block),
        kernel_args,
        shared_memory_bytes,
        stream
    ));

    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(h_result.data(), d_result, sizeof(int), hipMemcpyDeviceToHost));

    CHECK_GPU(hipFree(d_block_sums));
    CHECK_GPU(hipFree(d_result));

    int expected = n * (n + 1) / 2;

    if (h_result[0] != expected) {
        std::fprintf(stderr,
                     "Wrong result: expected %d, got %d\n",
                     expected, h_result[0]);
        return 1;
    }

    std::printf("OK: cooperative launch + shared memory test passed, n = %d\n", n);
    return 0;
}
