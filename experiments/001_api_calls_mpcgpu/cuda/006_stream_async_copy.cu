#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err));     \
            return 1;                                                      \
        }                                                                  \
    } while (0)

__global__ void simpleKernel(int *d_data) {
    int idx = threadIdx.x;
    d_data[idx] *= 2;
}

int main() {
    const int size = 5;
    const int bytes = size * sizeof(int);
    
    int h_in[size] = {1, 2, 3, 4, 5};
    int h_out[size] = {0};

    int *d_data;
    
    CHECK_GPU(cudaMalloc(&d_data, bytes));

    cudaStream_t stream;
    CHECK_GPU(cudaStreamCreate(&stream));

    CHECK_GPU(cudaMemcpyAsync(d_data, h_in, bytes, cudaMemcpyHostToDevice, stream));
    
    simpleKernel<<<1, size, 0, stream>>>(d_data);

    CHECK_GPU(cudaGetLastError());

    CHECK_GPU(cudaMemcpyAsync(h_out, d_data, bytes, cudaMemcpyDeviceToHost, stream));

    CHECK_GPU(cudaStreamSynchronize(stream));

    bool success = true;
    for(int i = 0; i < size; i++) {
        if (h_out[i] != h_in[i] * 2) {
            success = false;
        }
    }

    if (success) {
        std::cout << "OK: 006_stream_async_copy" << std::endl;
    } else {
        std::cout << "FAIL: 006_stream_async_copy" << std::endl;
    }

    CHECK_GPU(cudaStreamDestroy(stream));
    CHECK_GPU(cudaFree(d_data));

    return 0;
}