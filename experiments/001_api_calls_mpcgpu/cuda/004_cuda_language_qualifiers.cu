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

__host__
int host_add(int a, int b) {
    return a + b;
}

__host__ __device__
int host_device_multiply(int a, int b) {
    return a * b;
}

__device__
int device_add_one(int x) {
    return x + 1;
}

__device__ __forceinline__
int device_square(int x) {
    return x * x;
}

__global__
void qualifier_kernel(int *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        int a = device_add_one(i);
        int b = host_device_multiply(a, 2);
        int c = device_square(i);

        out[i] = b + c;
    }
}

int main() {
    const int n = 16;
    std::vector<int> h_out(n);

    int *d_out = nullptr;

    int host_result = host_add(10, 5);
    int host_device_cpu_result = host_device_multiply(3, 4);

    if (host_result != 15) {
        std::fprintf(stderr, "Wrong host result: expected 15, got %d\n", host_result);
        return 1;
    }

    if (host_device_cpu_result != 12) {
        std::fprintf(stderr, "Wrong host-device CPU result: expected 12, got %d\n",
                     host_device_cpu_result);
        return 1;
    }

    CHECK_GPU(cudaMalloc((void**)&d_out, n * sizeof(int)));

    qualifier_kernel<<<1, n>>>(d_out, n);

    CHECK_GPU(cudaPeekAtLastError());
    CHECK_GPU(cudaDeviceSynchronize());

    CHECK_GPU(cudaMemcpy(h_out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_GPU(cudaFree(d_out));

    for (int i = 0; i < n; i++) {
        int expected = 2 * (i + 1) + i * i;

        if (h_out[i] != expected) {
            std::fprintf(stderr, "Wrong value at %d: expected %d, got %d\n",
                         i, expected, h_out[i]);
            return 1;
        }
    }

    std::printf("OK: CUDA language qualifier test passed, n = %d\n", n);
    return 0;
}
