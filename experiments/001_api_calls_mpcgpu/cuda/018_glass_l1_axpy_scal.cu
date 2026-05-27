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
void glass_like_scal(T alpha, T* x, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        x[i] = alpha * x[i];
    }
}

template <typename T>
__device__
void glass_like_axpy(T alpha, const T* x, T* y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}

template <typename T>
__global__
void scal_kernel(T alpha, T* x, int n)
{
    glass_like_scal(alpha, x, n);
}

template <typename T>
__global__
void axpy_kernel(T alpha, const T* x, T* y, int n)
{
    glass_like_axpy(alpha, x, y, n);
}

int main()
{
    const int n = 259;
    const int threads = 64;
    const int blocks = 4;

    const float alpha_scal_float = 2.0f;
    const float alpha_axpy_float = 3.0f;

    const double alpha_scal_double = 2.0;
    const double alpha_axpy_double = 3.0;

    std::vector<float> h_x_float(n);
    std::vector<float> h_y_float(n);

    std::vector<double> h_x_double(n);
    std::vector<double> h_y_double(n);

    std::vector<float> expected_x_float(n);
    std::vector<float> expected_y_float(n);

    std::vector<double> expected_x_double(n);
    std::vector<double> expected_y_double(n);

    for (int i = 0; i < n; i++) {
        h_x_float[i] = static_cast<float>((i % 17) + 1);
        h_y_float[i] = static_cast<float>(100 + (i % 13));

        h_x_double[i] = static_cast<double>((i % 17) + 1);
        h_y_double[i] = static_cast<double>(100 + (i % 13));

        expected_x_float[i] = alpha_scal_float * h_x_float[i];
        expected_y_float[i] = alpha_axpy_float * expected_x_float[i] + h_y_float[i];

        expected_x_double[i] = alpha_scal_double * h_x_double[i];
        expected_y_double[i] = alpha_axpy_double * expected_x_double[i] + h_y_double[i];
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
    CHECK_GPU(cudaMemcpy(d_y_float, h_y_float.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_GPU(cudaMemcpy(d_x_double, h_x_double.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_GPU(cudaMemcpy(d_y_double, h_y_double.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    scal_kernel<float><<<blocks, threads>>>(alpha_scal_float, d_x_float, n);
    CHECK_GPU(cudaPeekAtLastError());

    axpy_kernel<float><<<blocks, threads>>>(alpha_axpy_float, d_x_float, d_y_float, n);
    CHECK_GPU(cudaPeekAtLastError());

    scal_kernel<double><<<blocks, threads>>>(alpha_scal_double, d_x_double, n);
    CHECK_GPU(cudaPeekAtLastError());

    axpy_kernel<double><<<blocks, threads>>>(alpha_axpy_double, d_x_double, d_y_double, n);
    CHECK_GPU(cudaPeekAtLastError());

    CHECK_GPU(cudaDeviceSynchronize());

    CHECK_GPU(cudaMemcpy(h_x_float.data(), d_x_float, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_GPU(cudaMemcpy(h_y_float.data(), d_y_float, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_GPU(cudaMemcpy(h_x_double.data(), d_x_double, n * sizeof(double), cudaMemcpyDeviceToHost));
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
            std::fabs(static_cast<double>(h_x_float[i] - expected_x_float[i]))
        );
        max_error_float = std::max(
            max_error_float,
            std::fabs(static_cast<double>(h_y_float[i] - expected_y_float[i]))
        );

        max_error_double = std::max(
            max_error_double,
            std::fabs(h_x_double[i] - expected_x_double[i])
        );
        max_error_double = std::max(
            max_error_double,
            std::fabs(h_y_double[i] - expected_y_double[i])
        );
    }

    if (max_error_float > 1e-6 || max_error_double > 1e-12) {
        std::fprintf(stderr,
                     "FAILED: scal/axpy mismatch, float error=%e, double error=%e\n",
                     max_error_float, max_error_double);
        return 1;
    }

    std::printf("OK: GLASS-like L1 scal/axpy test passed, n = %d\n", n);
    std::printf("Checked float and double device helpers.\n");

    return 0;
}
