#include <iostream>
#include <cmath>
#include <hip/hip_runtime.h>

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
    int N = 1 << 20; // 1M elements

    // Allocate Unified Memory -- accessible from CPU and GPU
    float *x, *y;

    hipMallocManaged(&x, N * sizeof(float));
    hipMallocManaged(&y, N * sizeof(float));

    // Initialize x and y arrays on the CPU/host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Launch one block with 256 threads.
    // Each thread processes multiple array elements using the stride loop.
    add<<<1, 256>>>(N, x, y);

    // Wait until GPU finishes before CPU reads y.
    hipDeviceSynchronize();

    // Check result. All values should be 3.0f.
    float maxError = 0.0f;

    for (int i = 0; i < N; i++) {
        maxError = std::fmax(maxError, std::fabs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;

    hipFree(x);
    hipFree(y);

    return 0;
}