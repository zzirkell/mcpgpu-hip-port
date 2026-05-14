#include <cstdio>
#include <cstdlib>
#include <cmath>
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

constexpr int TILE = 16;

__global__
void init_matrices(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;

    if (i < total) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }
}

__global__
void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int N = 4096;
    int repeats = 20;

    if (argc >= 2) {
        N = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        repeats = std::atoi(argv[2]);
    }

    std::printf("Matrix size: %d x %d\n", N, N);
    std::printf("Repeats: %d\n", repeats);

    size_t elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    size_t bytes = elements * sizeof(float);

    std::printf("Memory per matrix: %.2f MB\n", bytes / 1024.0 / 1024.0);
    std::printf("Total GPU memory used: %.2f MB\n", 3.0 * bytes / 1024.0 / 1024.0);

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    CHECK_GPU(hipMalloc(&A, bytes));
    CHECK_GPU(hipMalloc(&B, bytes));
    CHECK_GPU(hipMalloc(&C, bytes));

    int init_threads = 256;
    int init_blocks = static_cast<int>((elements + init_threads - 1) / init_threads);

    init_matrices<<<init_blocks, init_threads>>>(A, B, C, N);
    CHECK_GPU(hipGetLastError());
    CHECK_GPU(hipDeviceSynchronize());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Warm-up launch
    matmul_tiled<<<grid, block>>>(A, B, C, N);
    CHECK_GPU(hipGetLastError());
    CHECK_GPU(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_GPU(hipEventCreate(&start));
    CHECK_GPU(hipEventCreate(&stop));

    CHECK_GPU(hipEventRecord(start));

    for (int r = 0; r < repeats; ++r) {
        matmul_tiled<<<grid, block>>>(A, B, C, N);
    }

    CHECK_GPU(hipGetLastError());
    CHECK_GPU(hipEventRecord(stop));
    CHECK_GPU(hipEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK_GPU(hipEventElapsedTime(&milliseconds, start, stop));

    double seconds = milliseconds / 1000.0;
    double flops_per_matmul = 2.0 * static_cast<double>(N) * N * N;
    double total_gflops = flops_per_matmul * repeats / 1e9;
    double gflops_per_second = total_gflops / seconds;

    std::printf("Time: %.3f seconds\n", seconds);
    std::printf("Total work: %.2f GFLOP\n", total_gflops);
    std::printf("Performance: %.2f GFLOP/s\n", gflops_per_second);

    std::vector<float> host_C(elements);
    CHECK_GPU(hipMemcpy(host_C.data(), C, bytes, hipMemcpyDeviceToHost));

    float expected = 2.0f * N;
    float max_error = 0.0f;

    size_t step = elements / 1024;
    if (step == 0) {
        step = 1;
    }

    for (size_t i = 0; i < elements; i += step) {
        float error = std::fabs(host_C[i] - expected);
        if (error > max_error) {
            max_error = error;
        }
    }

    std::printf("Expected sample value: %.1f\n", expected);
    std::printf("Max sampled error: %.6f\n", max_error);

    CHECK_GPU(hipEventDestroy(start));
    CHECK_GPU(hipEventDestroy(stop));

    CHECK_GPU(hipFree(A));
    CHECK_GPU(hipFree(B));
    CHECK_GPU(hipFree(C));

    return 0;
}