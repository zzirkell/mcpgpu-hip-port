#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err));     \
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
    
    CHECK_GPU(cudaMallocPitch(&d_data, &pitch_device, cols * sizeof(float), rows));

    CHECK_GPU(cudaMemset2D(d_data, pitch_device, 0, cols * sizeof(float), rows));

    CHECK_GPU(cudaMemcpy2D(d_data, pitch_device, h_data, pitch_host, 
                           cols * sizeof(float), rows, cudaMemcpyHostToDevice));

    CHECK_GPU(cudaMemcpy2D(h_result, pitch_host, d_data, pitch_device, 
                           cols * sizeof(float), rows, cudaMemcpyDeviceToHost));

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

    CHECK_GPU(cudaFree(d_data));

    return 0;
}