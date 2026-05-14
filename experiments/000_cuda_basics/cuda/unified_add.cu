#include <iostream>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err));      \
            return 1;                                                       \
        }                                                                  \
    } while (0)

__global__
void add(int n, float *x, float *y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20;

    float *x, *y;

    CHECK_GPU(cudaMallocManaged(&x, N * sizeof(float)));
    CHECK_GPU(cudaMallocManaged(&y, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 256>>>(N, x, y);

    CHECK_GPU(cudaGetLastError());
    CHECK_GPU(cudaDeviceSynchronize());

    float maxError = 0.0f;

    for (int i = 0; i < N; i++) {
        maxError = std::fmax(maxError, std::fabs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;

    CHECK_GPU(cudaFree(x));
    CHECK_GPU(cudaFree(y));

    return 0;
}