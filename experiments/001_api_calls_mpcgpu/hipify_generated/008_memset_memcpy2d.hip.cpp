#include <iostream>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                            \
        if (err != hipSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, hipGetErrorString(err));     \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main() {
    const int rows = 4;
    const int cols = 4;
    const int pitch_host = cols * sizeof(float);
    
    float h_data[rows][cols];
    float h_result[rows][cols];
    
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            h_data[r][c] = 1.0f;
        }
    }

    float* d_data;
    size_t pitch_device;
    
    CHECK_GPU(hipMallocPitch(&d_data, &pitch_device, cols * sizeof(float), rows));

    CHECK_GPU(hipMemset2D(d_data, pitch_device, 0, cols * sizeof(float), rows));

    CHECK_GPU(hipMemcpy2D(d_data, pitch_device, h_data, pitch_host, 
                           cols * sizeof(float), rows, hipMemcpyHostToDevice));

    CHECK_GPU(hipMemcpy2D(h_result, pitch_host, d_data, pitch_device, 
                           cols * sizeof(float), rows, hipMemcpyDeviceToHost));

    bool success = true;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (h_result[r][c] != 1.0f) {
                success = false;
            }
        }
    }

    if (success) {
        std::cout << "OK: 008_memset_memcpy2d" << std::endl;
    } else {
        std::cout << "FAIL: 008_memset_memcpy2d" << std::endl;
    }

    CHECK_GPU(hipFree(d_data));

    return 0;
}