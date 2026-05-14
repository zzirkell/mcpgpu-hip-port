#include <iostream>
#include <cmath>
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

    CHECK_GPU(hipMallocManaged(&x, N * sizeof(float)));
    CHECK_GPU(hipMallocManaged(&y, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 256>>>(N, x, y);

    CHECK_GPU(hipGetLastError());
    CHECK_GPU(hipDeviceSynchronize());

    float maxError = 0.0f;

    for (int i = 0; i < N; i++) {
        maxError = std::fmax(maxError, std::fabs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;

    CHECK_GPU(hipFree(x));
    CHECK_GPU(hipFree(y));

    return 0;
}