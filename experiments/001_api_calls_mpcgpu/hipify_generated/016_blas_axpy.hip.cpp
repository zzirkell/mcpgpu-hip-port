#include "hip/hip_runtime.h"
#include <cstdio>
#include <cmath>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipblas.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                             \
        if (err != hipSuccess) {                                           \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, hipGetErrorString(err));      \
            return 1;                                                       \
        }                                                                  \
    } while (0)

#define CHECK_BLAS(call)                                                   \
    do {                                                                   \
        hipblasStatus_t status = call;                                       \
        if (status != HIPBLAS_STATUS_SUCCESS) {                             \
            std::fprintf(stderr, "BLAS error at %s:%d: status=%d\n",       \
                         __FILE__, __LINE__, (int)status);                 \
            return 1;                                                       \
        }                                                                  \
    } while (0)

int main() {
    const int n = 8;

    std::vector<float> h_x_float(n);
    std::vector<float> h_y_float(n);

    std::vector<double> h_x_double(n);
    std::vector<double> h_y_double(n);

    for (int i = 0; i < n; i++) {
        h_x_float[i] = 1.0f + i;
        h_y_float[i] = 10.0f;

        h_x_double[i] = 1.0 + i;
        h_y_double[i] = 20.0;
    }

    float *d_x_float = nullptr;
    float *d_y_float = nullptr;

    double *d_x_double = nullptr;
    double *d_y_double = nullptr;

    CHECK_GPU(hipMalloc((void**)&d_x_float, n * sizeof(float)));
    CHECK_GPU(hipMalloc((void**)&d_y_float, n * sizeof(float)));

    CHECK_GPU(hipMalloc((void**)&d_x_double, n * sizeof(double)));
    CHECK_GPU(hipMalloc((void**)&d_y_double, n * sizeof(double)));

    CHECK_GPU(hipMemcpy(d_x_float, h_x_float.data(), n * sizeof(float), hipMemcpyHostToDevice));
    CHECK_GPU(hipMemcpy(d_y_float, h_y_float.data(), n * sizeof(float), hipMemcpyHostToDevice));

    CHECK_GPU(hipMemcpy(d_x_double, h_x_double.data(), n * sizeof(double), hipMemcpyHostToDevice));
    CHECK_GPU(hipMemcpy(d_y_double, h_y_double.data(), n * sizeof(double), hipMemcpyHostToDevice));

    hipblasHandle_t handle;
    CHECK_BLAS(hipblasCreate(&handle));

    const float alpha_float = 2.0f;
    const double alpha_double = 3.0;

    CHECK_BLAS(hipblasSaxpy(handle, n, &alpha_float, d_x_float, 1, d_y_float, 1));
    CHECK_BLAS(hipblasDaxpy(handle, n, &alpha_double, d_x_double, 1, d_y_double, 1));

    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(h_y_float.data(), d_y_float, n * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_GPU(hipMemcpy(h_y_double.data(), d_y_double, n * sizeof(double), hipMemcpyDeviceToHost));

    CHECK_BLAS(hipblasDestroy(handle));

    CHECK_GPU(hipFree(d_x_float));
    CHECK_GPU(hipFree(d_y_float));
    CHECK_GPU(hipFree(d_x_double));
    CHECK_GPU(hipFree(d_y_double));

    float max_error_float = 0.0f;
    double max_error_double = 0.0;

    for (int i = 0; i < n; i++) {
        float expected_float = 10.0f + alpha_float * (1.0f + i);
        double expected_double = 20.0 + alpha_double * (1.0 + i);

        max_error_float = std::fmax(max_error_float, std::fabs(h_y_float[i] - expected_float));
        max_error_double = std::fmax(max_error_double, std::fabs(h_y_double[i] - expected_double));
    }

    if (max_error_float > 1e-5f || max_error_double > 1e-12) {
        std::fprintf(stderr,
                     "Wrong result: max_error_float=%g, max_error_double=%g\n",
                     max_error_float, max_error_double);
        return 1;
    }

    std::printf("OK: BLAS AXPY test passed, n = %d\n", n);
    std::printf("float:  y = 10 + 2*x\n");
    std::printf("double: y = 20 + 3*x\n");

    return 0;
}
