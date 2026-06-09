#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#define CHECK_GPU(call)                                                     \
    do {                                                                    \
        hipError_t err = call;                                              \
        if (err != hipSuccess) {                                            \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",                 \
                         __FILE__, __LINE__, hipGetErrorString(err));       \
            return 1;                                                        \
        }                                                                   \
    } while (0)

template <typename T>
__device__ T gemv_row_value(const T* A, const T* x, int rows, int cols, int row)
{
    T sum = static_cast<T>(0);

    for (int col = 0; col < cols; col++) {
        // Row-major layout: A[row, col]
        sum += A[row * cols + col] * x[col];
    }

    return sum;
}

template <typename T>
__global__ void gemv_kernel(const T* A, const T* x, T* y, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        y[row] = gemv_row_value(A, x, rows, cols, row);
    }
}

template <typename T>
T cpu_gemv_row_value(const std::vector<T>& A, const std::vector<T>& x,
                     int rows, int cols, int row)
{
    T sum = static_cast<T>(0);

    for (int col = 0; col < cols; col++) {
        sum += A[row * cols + col] * x[col];
    }

    return sum;
}

template <typename T>
T max_abs_error(const std::vector<T>& actual, const std::vector<T>& expected)
{
    T max_error = static_cast<T>(0);

    for (size_t i = 0; i < actual.size(); i++) {
        T err = std::abs(actual[i] - expected[i]);
        if (err > max_error) {
            max_error = err;
        }
    }

    return max_error;
}

template <typename T>
void fill_test_data(std::vector<T>& A, std::vector<T>& x, int rows, int cols)
{
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            A[row * cols + col] =
                static_cast<T>(0.01) * static_cast<T>((row + 1) + (col + 1));
        }
    }

    for (int col = 0; col < cols; col++) {
        x[col] = static_cast<T>(0.02) * static_cast<T>(col + 1);
    }
}

template <typename T>
int run_gemv_test(const char* name, T tolerance)
{
    const int rows = 19;
    const int cols = 23;

    const size_t matrix_bytes = rows * cols * sizeof(T);
    const size_t x_bytes = cols * sizeof(T);
    const size_t y_bytes = rows * sizeof(T);

    std::vector<T> h_A(rows * cols);
    std::vector<T> h_x(cols);
    std::vector<T> h_y(rows);
    std::vector<T> h_expected(rows);

    fill_test_data(h_A, h_x, rows, cols);

    for (int row = 0; row < rows; row++) {
        h_expected[row] = cpu_gemv_row_value(h_A, h_x, rows, cols, row);
    }

    T* d_A = nullptr;
    T* d_x = nullptr;
    T* d_y = nullptr;

    CHECK_GPU(hipMalloc((void**)&d_A, matrix_bytes));
    CHECK_GPU(hipMalloc((void**)&d_x, x_bytes));
    CHECK_GPU(hipMalloc((void**)&d_y, y_bytes));

    CHECK_GPU(hipMemcpy(d_A, h_A.data(), matrix_bytes, hipMemcpyHostToDevice));
    CHECK_GPU(hipMemcpy(d_x, h_x.data(), x_bytes, hipMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = (rows + threads - 1) / threads;

    gemv_kernel<T><<<blocks, threads>>>(d_A, d_x, d_y, rows, cols);

    CHECK_GPU(hipPeekAtLastError());
    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(h_y.data(), d_y, y_bytes, hipMemcpyDeviceToHost));

    CHECK_GPU(hipFree(d_A));
    CHECK_GPU(hipFree(d_x));
    CHECK_GPU(hipFree(d_y));

    T max_error = max_abs_error(h_y, h_expected);

    if (max_error > tolerance) {
        std::cerr << "FAIL: " << name << " GEMV max error = "
                  << static_cast<double>(max_error) << std::endl;
        return 1;
    }

    return 0;
}

int main()
{
    int result_float = run_gemv_test<float>("float", 1e-5f);
    if (result_float != 0) {
        return result_float;
    }

    int result_double = run_gemv_test<double>("double", 1e-12);
    if (result_double != 0) {
        return result_double;
    }

    std::cout << "OK: GLASS-like L2 GEMV test passed" << std::endl;
    std::cout << "Checked float and double matrix-vector multiplication." << std::endl;

    return 0;
}