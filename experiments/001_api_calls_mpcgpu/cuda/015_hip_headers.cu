#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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
    int device_count = 0;
    int current_device = -1;

    CHECK_GPU(cudaGetDeviceCount(&device_count));

    if (device_count <= 0) {
        std::fprintf(stderr, "No GPU devices found\n");
        return 1;
    }

    CHECK_GPU(cudaGetDevice(&current_device));

    if (current_device < 0 || current_device >= device_count) {
        std::fprintf(stderr,
                     "Invalid current device: %d, device_count = %d\n",
                     current_device, device_count);
        return 1;
    }

    CHECK_GPU(cudaSetDevice(current_device));

    std::printf("OK: CUDA runtime header translation test passed\n");
    std::printf("Device count = %d, current device = %d\n",
                device_count, current_device);

    return 0;
}
