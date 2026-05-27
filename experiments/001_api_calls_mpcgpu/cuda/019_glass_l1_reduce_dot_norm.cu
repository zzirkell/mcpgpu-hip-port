#include <cuda_runtime.h>

#include <cstdio>
#include <vector>
#include <cmath>

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
__device__ __forceinline__
T gpu_abs(T x)
{
    return x < (T)0 ? -x : x;
}

template <typename T>
__device__ __forceinline__
T gpu_max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__device__ __forceinline__
T gpu_sqrt(T x);

template <>
__device__ __forceinline__
float gpu_sqrt<float>(float x)
{
    return sqrtf(x);
}

template <>
__device__ __forceinline__
double gpu_sqrt<double>(double x)
{
    return sqrt(x);
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__
void block_reduce_three(T &dot, T &norm2_sum, T &inf_norm)
{
    __shared__ T s_dot[BLOCK_SIZE];
    __shared__ T s_norm2[BLOCK_SIZE];
    __shared__ T s_inf[BLOCK_SIZE];

    int tid = threadIdx.x;

    s_dot[tid] = dot;
    s_norm2[tid] = norm2_sum;
    s_inf[tid] = inf_norm;

    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_dot[tid] += s_dot[tid + offset];
            s_norm2[tid] += s_norm2[tid + offset];
            s_inf[tid] = gpu_max(s_inf[tid], s_inf[tid + offset]);
        }

        __syncthreads();
    }

    dot = s_dot[0];
    norm2_sum = s_norm2[0];
    inf_norm = s_inf[0];
}

template <typename T, int BLOCK_SIZE>
__global__
void l1_reduce_kernel(const T *x, const T *y, int n, T *out)
{
    int tid = threadIdx.x;

    T local_dot = (T)0;
    T local_norm2_sum = (T)0;
    T local_inf = (T)0;

    for (int i = tid; i < n; i += BLOCK_SIZE) {
        T xi = x[i];

        local_dot += xi * y[i];
        local_norm2_sum += xi * xi;
        local_inf = gpu_max(local_inf, gpu_abs(xi));
    }

    block_reduce_three<T, BLOCK_SIZE>(local_dot, local_norm2_sum, local_inf);

    if (tid == 0) {
        out[0] = local_dot;
        out[1] = gpu_sqrt(local_norm2_sum);
        out[2] = local_inf;
    }
}

template <typename T>
int run_one_type(const char *type_name)
{
    constexpr int n = 257;
    constexpr int block_size = 128;

    std::vector<T> h_x(n);
    std::vector<T> h_y(n);
    std::vector<T> h_out(3);

    for (int i = 0; i < n; i++) {
        h_x[i] = (T)(((i % 11) - 5) * 0.25);
        h_y[i] = (T)(((i % 7) + 1) * 0.5);
    }

    T *d_x = nullptr;
    T *d_y = nullptr;
    T *d_out = nullptr;

    CHECK_GPU(cudaMalloc((void **)&d_x, n * sizeof(T)));
    CHECK_GPU(cudaMalloc((void **)&d_y, n * sizeof(T)));
    CHECK_GPU(cudaMalloc((void **)&d_out, 3 * sizeof(T)));

    CHECK_GPU(cudaMemcpy(d_x, h_x.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_GPU(cudaMemcpy(d_y, h_y.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    l1_reduce_kernel<T, block_size><<<1, block_size>>>(d_x, d_y, n, d_out);

    CHECK_GPU(cudaGetLastError());
    CHECK_GPU(cudaDeviceSynchronize());

    CHECK_GPU(cudaMemcpy(h_out.data(), d_out, 3 * sizeof(T), cudaMemcpyDeviceToHost));

    CHECK_GPU(cudaFree(d_x));
    CHECK_GPU(cudaFree(d_y));
    CHECK_GPU(cudaFree(d_out));

    double expected_dot = 0.0;
    double expected_norm2_sum = 0.0;
    double expected_inf = 0.0;

    for (int i = 0; i < n; i++) {
        double xi = (double)h_x[i];
        double yi = (double)h_y[i];

        expected_dot += xi * yi;
        expected_norm2_sum += xi * xi;

        double abs_xi = xi < 0.0 ? -xi : xi;
        expected_inf = expected_inf > abs_xi ? expected_inf : abs_xi;
    }

    double expected_norm2 = std::sqrt(expected_norm2_sum);

    double got_dot = (double)h_out[0];
    double got_norm2 = (double)h_out[1];
    double got_inf = (double)h_out[2];

    double tol = sizeof(T) == sizeof(float) ? 1e-4 : 1e-10;

    if (std::fabs(got_dot - expected_dot) > tol ||
        std::fabs(got_norm2 - expected_norm2) > tol ||
        std::fabs(got_inf - expected_inf) > tol) {
        std::fprintf(stderr,
                     "FAIL: %s reduction mismatch\n"
                     "dot:   got %.12f expected %.12f\n"
                     "norm2: got %.12f expected %.12f\n"
                     "inf:   got %.12f expected %.12f\n",
                     type_name,
                     got_dot, expected_dot,
                     got_norm2, expected_norm2,
                     got_inf, expected_inf);
        return 1;
    }

    return 0;
}

int main()
{
    if (run_one_type<float>("float") != 0) {
        return 1;
    }

    if (run_one_type<double>("double") != 0) {
        return 1;
    }

    std::printf("OK: GLASS-like L1 reduction test passed\n");
    std::printf("Checked dot product, L2 norm, and infinity norm for float and double.\n");

    return 0;
}
