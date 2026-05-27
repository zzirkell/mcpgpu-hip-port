#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

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

template <typename T>
__device__
void glass_like_copy(const T* src, T* dst, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

template <typename T>
__global__
void copy_kernel(const T* src, T* dst, int n)
{
    glass_like_copy(src, dst, n);
}

int main()
{
    const int n = 257;
    const int threads = 64;
    const int blocks = 4;

    std::vector<float> h_x_float(n);
    std::vector<float> h_y_float(n, 0.0f);

    std::vector<double> h_x_double(n);
    std::vector<double> h_y_double(n, 0.0);

    for (int i = 0; i < n; i++) {
        h_x_float[i] = 1.5f * i;
        h_x_double[i] = 2.5 * i;
    }

    float* d_x_float = nullptr;
    float* d_y_float = nullptr;

    double* d_x_double = nullptr;
    double* d_y_double = nullptr;

    CHECK_GPU(cudaMalloc((void**)&d_x_float, n * sizeof(float)));
    CHECK_GPU(cudaMalloc((void**)&d_y_float, n * sizeof(float)));
    CHECK_GPU(cudaMalloc((void**)&d_x_double, n * sizeof(double)));
    CHECK_GPU(cudaMalloc((void**)&d_y_double, n * sizeof(double)));

    CHECK_GPU(cudaMemcpy(d_x_float, h_x_float.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_GPU(cudaMemcpy(d_x_double, h_x_double.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    copy_kernel<float><<<blocks, threads>>>(d_x_float, d_y_float, n);
    CHECK_GPU(cudaPeekAtLastError());

    copy_kernel<double><<<blocks, threads>>>(d_x_double, d_y_double, n);
    CHECK_GPU(cudaPeekAtLastError());

    CHECK_GPU(cudaDeviceSynchronize());

    CHECK_GPU(cudaMemcpy(h_y_float.data(), d_y_float, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_GPU(cudaMemcpy(h_y_double.data(), d_y_double, n * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_GPU(cudaFree(d_x_float));
    CHECK_GPU(cudaFree(d_y_float));
    CHECK_GPU(cudaFree(d_x_double));
    CHECK_GPU(cudaFree(d_y_double));

    double max_error_float = 0.0;
    double max_error_double = 0.0;

    for (int i = 0; i < n; i++) {
        max_error_float = std::max(
            max_error_float,
            std::fabs(static_cast<double>(h_y_float[i] - h_x_float[i]))
        );

        max_error_double = std::max(
            max_error_double,
            std::fabs(h_y_double[i] - h_x_double[i])
        );
    }

    if (max_error_float > 1e-6 || max_error_double > 1e-12) {
        std::fprintf(stderr,
                     "FAILED: copy mismatch, float error=%e, double error=%e\n",
                     max_error_float, max_error_double);
        return 1;
    }

    std::printf("OK: GLASS-like L1 copy test passed, n = %d\n", n);
    std::printf("Checked float and double device helper copy.\n");

    return 0;
}
