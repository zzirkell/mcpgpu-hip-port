#include <hip/hip_runtime.h>
#include <iostream>


#define CHECK_GPU(call)                                                        \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, hipGetErrorString(err));          \
            std::cout << "FAIL: T3_kkt_test" << std::endl;                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)


#include "../../../hip_port/include/common/kkt.hip.hpp"

int main() {
    std::cout << "Starting T3: KKT Submatrices Test..." << std::endl;

    uint32_t state_size = 14;   
    uint32_t control_size = 7;  
    uint32_t knot_points = 10;
    float timestep = 0.01f;

    size_t huge_buffer = 10 * 1024 * 1024; 

    float *d_G_dense, *d_C_dense, *d_g, *d_c, *d_eePos_traj, *d_xs, *d_xu;
    void *d_dynMem_const;

    CHECK_GPU(hipMalloc(&d_G_dense, huge_buffer));
    CHECK_GPU(hipMalloc(&d_C_dense, huge_buffer));
    CHECK_GPU(hipMalloc(&d_g, huge_buffer));
    CHECK_GPU(hipMalloc(&d_c, huge_buffer));
    CHECK_GPU(hipMalloc(&d_dynMem_const, huge_buffer));
    CHECK_GPU(hipMalloc(&d_eePos_traj, huge_buffer));
    CHECK_GPU(hipMalloc(&d_xs, huge_buffer));
    CHECK_GPU(hipMalloc(&d_xu, huge_buffer));

    CHECK_GPU(hipMemset(d_G_dense, 0, huge_buffer));
    CHECK_GPU(hipMemset(d_C_dense, 0, huge_buffer));
    CHECK_GPU(hipMemset(d_g, 0, huge_buffer));
    CHECK_GPU(hipMemset(d_c, 0, huge_buffer));
    CHECK_GPU(hipMemset(d_dynMem_const, 0, huge_buffer));
    CHECK_GPU(hipMemset(d_eePos_traj, 0, huge_buffer));
    CHECK_GPU(hipMemset(d_xs, 0, huge_buffer));
    CHECK_GPU(hipMemset(d_xu, 0, huge_buffer));

    dim3 blocks(knot_points);
    dim3 threads(64); 

    size_t smem_size = get_kkt_smem_size<float>(state_size, control_size);
    std::cout << "Calculated Shared Memory: " << smem_size << " bytes." << std::endl;

    generate_kkt_submatrices<float, 0, false><<<blocks, threads, smem_size>>>(
        state_size, control_size, knot_points,
        d_G_dense, d_C_dense, d_g, d_c,
        d_dynMem_const, timestep,
        d_eePos_traj, d_xs, d_xu
    );

    CHECK_GPU(hipDeviceSynchronize());
    std::cout << "OK: T3 (generate_kkt_submatrices) executed successfully!" << std::endl;

    CHECK_GPU(hipFree(d_G_dense));
    CHECK_GPU(hipFree(d_C_dense));
    CHECK_GPU(hipFree(d_g));
    CHECK_GPU(hipFree(d_c));
    CHECK_GPU(hipFree(d_dynMem_const));
    CHECK_GPU(hipFree(d_eePos_traj));
    CHECK_GPU(hipFree(d_xs));
    CHECK_GPU(hipFree(d_xu));

    return 0;
}