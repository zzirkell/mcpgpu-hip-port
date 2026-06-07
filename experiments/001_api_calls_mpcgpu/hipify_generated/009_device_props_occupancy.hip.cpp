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
void occupancy_test_kernel(int *out) {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_index == 0) {
        out[0] = 1;
    }
}

int main() {
    int device = 0;
    int device_count = 0;

    CHECK_GPU(hipGetDeviceCount(&device_count));

    if (device_count <= 0) {
        std::fprintf(stderr, "No GPU devices found\n");
        return 1;
    }

    CHECK_GPU(hipGetDevice(&device));

    hipDeviceProp_t prop;
    CHECK_GPU(hipGetDeviceProperties(&prop, device));

    int max_threads_per_block_attr = 0;
    int multiprocessor_count_attr = 0;
    int cooperative_launch_attr = -1;

    CHECK_GPU(hipDeviceGetAttribute(
        &max_threads_per_block_attr,
        hipDeviceAttributeMaxThreadsPerBlock,
        device
    ));

    CHECK_GPU(hipDeviceGetAttribute(
        &multiprocessor_count_attr,
        hipDeviceAttributeMultiprocessorCount,
        device
    ));

    hipError_t coop_err = hipDeviceGetAttribute(
        &cooperative_launch_attr,
        hipDeviceAttributeCooperativeLaunch,
        device
    );

    if (coop_err != hipSuccess) {
        std::printf("NOTE: cooperative launch attribute query failed: %s\n",
                    hipGetErrorString(coop_err));
        cooperative_launch_attr = -1;
    }

    const int block_size = 128;
    const size_t dynamic_shared_bytes = 0;

    int active_blocks_per_sm = 0;

    CHECK_GPU(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        occupancy_test_kernel,
        block_size,
        dynamic_shared_bytes
    ));

    if (prop.maxThreadsPerBlock <= 0) {
        std::fprintf(stderr, "Invalid prop.maxThreadsPerBlock\n");
        return 1;
    }

    if (prop.multiProcessorCount <= 0) {
        std::fprintf(stderr, "Invalid prop.multiProcessorCount\n");
        return 1;
    }

    if (max_threads_per_block_attr <= 0) {
        std::fprintf(stderr, "Invalid hipDeviceAttributeMaxThreadsPerBlock\n");
        return 1;
    }

    if (multiprocessor_count_attr <= 0) {
        std::fprintf(stderr, "Invalid hipDeviceAttributeMultiprocessorCount\n");
        return 1;
    }

    if (active_blocks_per_sm <= 0) {
        std::fprintf(stderr, "Invalid occupancy result\n");
        return 1;
    }

    std::printf("OK: device properties/occupancy test passed\n");
    std::printf("Device name: %s\n", prop.name);
    std::printf("Device id: %d / device count: %d\n", device, device_count);
    std::printf("prop.maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
    std::printf("prop.multiProcessorCount = %d\n", prop.multiProcessorCount);
    std::printf("attr.maxThreadsPerBlock = %d\n", max_threads_per_block_attr);
    std::printf("attr.multiProcessorCount = %d\n", multiprocessor_count_attr);
    std::printf("attr.cooperativeLaunch = %d\n", cooperative_launch_attr);
    std::printf("occupancy active blocks per SM = %d for block size %d\n",
                active_blocks_per_sm, block_size);

    return 0;
}
