#include "hip/hip_runtime.h"
#include <iostream>
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


__constant__ float d_weights[4];

__global__ void evaluate_cost_kernel(const float* states, float* costs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = states[i];
        
        costs[i] = d_weights[0] + 
                   d_weights[1] * x + 
                   d_weights[2] * x * x + 
                   d_weights[3] * x * x * x;
    }
}

int main() {
    const int N = 50000;
    std::vector<float> h_states(N, 2.0f);
    std::vector<float> h_costs(N, 0.0f);

    // parameters are set on cpu
    // f(x) = 1 + 2x + 3x^2 + 4x^3, f(2) = 49
    float h_weights[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    float *d_states, *d_costs;
    
    CHECK_GPU(hipMalloc(&d_states, N * sizeof(float)));
    CHECK_GPU(hipMalloc(&d_costs, N * sizeof(float)));

    CHECK_GPU(hipMemcpy(d_states, h_states.data(), N * sizeof(float), hipMemcpyHostToDevice));

    CHECK_GPU(hipMemcpyToSymbol(HIP_SYMBOL(d_weights), h_weights, 4 * sizeof(float)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    evaluate_cost_kernel<<<blocks, threads>>>(d_states, d_costs, N);

    CHECK_GPU(hipGetLastError());
    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(h_costs.data(), d_costs, N * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "Constant memory" << std::endl;
    std::cout << "expected number: 49" << std::endl;
    std::cout << "calculated number: " << h_costs[0] << std::endl;

    CHECK_GPU(hipFree(d_states));
    CHECK_GPU(hipFree(d_costs));
    
    return 0;
}