#include <iostream>
#include <cstdio>
#include <hip/hip_runtime.h>
#include "utils/matrix.cuh" 

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                             \
        if (err != hipSuccess) {                                           \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, hipGetErrorString(err));      \
            std::cout << "FAIL: T1_matrix_test" << std::endl;              \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

int main() {
    std::cout << "Running T1: matrix.cuh & hipMemcpy2D..." << std::endl;

    float h_matrix[4][4] = {{1.0f}};
    float *d_matrix;
    size_t pitch;
    
    CHECK_GPU(hipMallocPitch(&d_matrix, &pitch, 4 * sizeof(float), 4));
    CHECK_GPU(hipMemcpy2D(d_matrix, pitch, h_matrix, 4 * sizeof(float), 
                           4 * sizeof(float), 4, hipMemcpyHostToDevice));

    CHECK_GPU(hipDeviceSynchronize());
    CHECK_GPU(hipFree(d_matrix));

    std::cout << "OK: T1 passed!" << std::endl << std::endl;
    return 0;
}