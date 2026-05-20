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


__global__ 
void reduction_kernel(const float* g_idata, float* g_odata, int n) {
    // shared memory inside one block
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load data from global memory to shared memory
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    
    __syncthreads();

    // every "second" thread adds two numbers together
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(g_odata, sdata[0]);
    }
}

int main(int argc, char** argv) {
    const int N = 1 << 20; // 1.048.576 elements
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // only 1s in the vector to add
    std::vector<float> h_idata(N, 1.0f);
    float h_odata = 0.0f;

    float *d_idata, *d_odata;
    CHECK_GPU(hipMalloc(&d_idata, N * sizeof(float)));
    CHECK_GPU(hipMalloc(&d_odata, sizeof(float)));

    CHECK_GPU(hipMemcpy(d_idata, h_idata.data(), N * sizeof(float), hipMemcpyHostToDevice));
    CHECK_GPU(hipMemcpy(d_odata, &h_odata, sizeof(float), hipMemcpyHostToDevice));

    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    reduction_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_idata, d_odata, N);

    CHECK_GPU(hipGetLastError());
    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(&h_odata, d_odata, sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "Parallel reduction" << std::endl;
    std::cout << "Expected number:   " << N << std::endl;
    std::cout << "Calculated sum:  " << (int)h_odata << std::endl;
    
    CHECK_GPU(hipFree(d_idata));
    CHECK_GPU(hipFree(d_odata));

    return 0;
}