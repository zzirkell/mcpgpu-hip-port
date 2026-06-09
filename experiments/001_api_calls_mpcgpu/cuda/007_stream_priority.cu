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

__global__ void dummyKernel(float *data) {
    int idx = threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}

int main() {
    int leastPriority;
    int greatestPriority;
    
    CHECK_GPU(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

    cudaStream_t highPriorityStream;
    CHECK_GPU(cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamNonBlocking, greatestPriority));

    float *d_data;
    CHECK_GPU(cudaMalloc(&d_data, 256 * sizeof(float)));

    dummyKernel<<<1, 256, 0, highPriorityStream>>>(d_data);
    CHECK_GPU(cudaGetLastError());

    CHECK_GPU(cudaStreamSynchronize(highPriorityStream));

    std::cout << "OK: 007_stream_priority" << std::endl;

    CHECK_GPU(cudaFree(d_data));
    CHECK_GPU(cudaStreamDestroy(highPriorityStream));

    return 0;
}