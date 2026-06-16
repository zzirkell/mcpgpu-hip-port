#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_GPU(call)                                                        \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, hipGetErrorString(err));          \
            std::cout << "FAIL: T2_dz_test" << std::endl;                      \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#ifndef DZ_THREADS
#define DZ_THREADS 32
#endif


#include "../../../hip_port/include/common/dz.hip.hpp"

int main() {
    std::cout << "Starting T2: Trajectory Update (dz) Test" << std::endl;

    uint32_t state_size = 12;
    uint32_t control_size = 6;
    uint32_t knot_points = 10;

    size_t buffer_size = knot_points * (state_size + control_size) * (state_size + control_size) * sizeof(float);

    float *d_G_dense, *d_C_dense, *d_g_val, *d_lambda, *d_dz;

    CHECK_GPU(hipMalloc(&d_G_dense, buffer_size));
    CHECK_GPU(hipMalloc(&d_C_dense, buffer_size));
    CHECK_GPU(hipMalloc(&d_g_val, buffer_size));
    CHECK_GPU(hipMalloc(&d_lambda, buffer_size));
    CHECK_GPU(hipMalloc(&d_dz, buffer_size));

    CHECK_GPU(hipMemset(d_G_dense, 0, buffer_size));
    CHECK_GPU(hipMemset(d_C_dense, 0, buffer_size));
    CHECK_GPU(hipMemset(d_g_val, 0, buffer_size));
    CHECK_GPU(hipMemset(d_lambda, 0, buffer_size));
    CHECK_GPU(hipMemset(d_dz, 0, buffer_size));

    compute_dz<float>(
        state_size, 
        control_size, 
        knot_points, 
        d_G_dense, 
        d_C_dense, 
        d_g_val, 
        d_lambda, 
        d_dz
    );


    CHECK_GPU(hipDeviceSynchronize());
    std::cout << "OK: T2 (compute_dz) executed successfully!" << std::endl;


    CHECK_GPU(hipFree(d_G_dense));
    CHECK_GPU(hipFree(d_C_dense));
    CHECK_GPU(hipFree(d_g_val));
    CHECK_GPU(hipFree(d_lambda));
    CHECK_GPU(hipFree(d_dz));

    return 0;
}