#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>
#include <cmath>

#define CHECK_GPU(call)                                                       \
    do {                                                                      \
        hipError_t err = (call);                                              \
        if (err != hipSuccess) {                                              \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, hipGetErrorString(err));         \
            return false;                                                     \
        }                                                                     \
    } while (0)

template <typename T>
__device__ void glass_like_gemm_one(
    int rows_a,
    int cols_b,
    int inner_dim,
    const T* A,
    const T* B,
    T* C,
    T alpha,
    T beta,
    int row,
    int col)
{
    if (row >= rows_a || col >= cols_b) {
        return;
    }

    T sum = T(0);

    for (int k = 0; k < inner_dim; k++) {
        sum += A[row * inner_dim + k] * B[k * cols_b + col];
    }

    C[row * cols_b + col] = alpha * sum + beta * C[row * cols_b + col];
}

template <typename T>
__global__ void gemm_kernel(
    int rows_a,
    int cols_b,
    int inner_dim,
    const T* A,
    const T* B,
    T* C,
    T alpha,
    T beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    glass_like_gemm_one(
        rows_a,
        cols_b,
        inner_dim,
        A,
        B,
        C,
        alpha,
        beta,
        row,
        col);
}

template <typename T>
void host_reference_gemm(
    int rows_a,
    int cols_b,
    int inner_dim,
    const std::vector<T>& A,
    const std::vector<T>& B,
    const std::vector<T>& C_original,
    std::vector<T>& C_reference,
    T alpha,
    T beta)
{
    for (int row = 0; row < rows_a; row++) {
        for (int col = 0; col < cols_b; col++) {
            T sum = T(0);

            for (int k = 0; k < inner_dim; k++) {
                sum += A[row * inner_dim + k] * B[k * cols_b + col];
            }

            C_reference[row * cols_b + col] =
                alpha * sum + beta * C_original[row * cols_b + col];
        }
    }
}

template <typename T>
bool run_one_gemm_test(const char* type_name, double tolerance)
{
    const int rows_a = 5;
    const int inner_dim = 4;
    const int cols_b = 6;

    const int size_a = rows_a * inner_dim;
    const int size_b = inner_dim * cols_b;
    const int size_c = rows_a * cols_b;

    const T alpha = T(1.25);
    const T beta = T(0.5);

    std::vector<T> h_a(size_a);
    std::vector<T> h_b(size_b);
    std::vector<T> h_c(size_c);
    std::vector<T> h_c_original(size_c);
    std::vector<T> h_reference(size_c);

    for (int i = 0; i < size_a; i++) {
        h_a[i] = T(0.1) * T((i % 7) - 3);
    }

    for (int i = 0; i < size_b; i++) {
        h_b[i] = T(0.2) * T((i % 5) + 1);
    }

    for (int i = 0; i < size_c; i++) {
        h_c[i] = T(0.01) * T((i % 11) - 5);
    }

    h_c_original = h_c;

    host_reference_gemm(
        rows_a,
        cols_b,
        inner_dim,
        h_a,
        h_b,
        h_c_original,
        h_reference,
        alpha,
        beta);

    T* d_a = nullptr;
    T* d_b = nullptr;
    T* d_c = nullptr;

    CHECK_GPU(hipMalloc((void**)&d_a, size_a * sizeof(T)));
    CHECK_GPU(hipMalloc((void**)&d_b, size_b * sizeof(T)));
    CHECK_GPU(hipMalloc((void**)&d_c, size_c * sizeof(T)));

    CHECK_GPU(hipMemcpy(d_a, h_a.data(), size_a * sizeof(T), hipMemcpyHostToDevice));
    CHECK_GPU(hipMemcpy(d_b, h_b.data(), size_b * sizeof(T), hipMemcpyHostToDevice));
    CHECK_GPU(hipMemcpy(d_c, h_c.data(), size_c * sizeof(T), hipMemcpyHostToDevice));

    dim3 block(4, 4);
    dim3 grid(
        (cols_b + block.x - 1) / block.x,
        (rows_a + block.y - 1) / block.y);

    gemm_kernel<T><<<grid, block>>>(
        rows_a,
        cols_b,
        inner_dim,
        d_a,
        d_b,
        d_c,
        alpha,
        beta);

    CHECK_GPU(hipPeekAtLastError());
    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipMemcpy(h_c.data(), d_c, size_c * sizeof(T), hipMemcpyDeviceToHost));

    CHECK_GPU(hipFree(d_a));
    CHECK_GPU(hipFree(d_b));
    CHECK_GPU(hipFree(d_c));

    double max_error = 0.0;

    for (int i = 0; i < size_c; i++) {
        double error = std::fabs(double(h_c[i] - h_reference[i]));

        if (error > max_error) {
            max_error = error;
        }
    }

    if (max_error > tolerance) {
        std::fprintf(stderr,
                     "FAIL: %s GEMM max error = %.12f\n",
                     type_name,
                     max_error);
        return false;
    }

    return true;
}

int main()
{
    bool float_ok = run_one_gemm_test<float>("float", 1e-4);
    bool double_ok = run_one_gemm_test<double>("double", 1e-10);

    if (!float_ok || !double_ok) {
        return 1;
    }

    std::printf("OK: GLASS-like L3 GEMM test passed\n");
    std::printf("Checked float and double matrix-matrix multiplication.\n");

    return 0;
}
