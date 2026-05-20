#include "hip/hip_runtime.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
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


__global__
void heavy_kernel(float* data, int n, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        for (int j = 0; j < 200; ++j) {
            val = sinf(val) * cosf(val) + factor;
        }
        data[i] = val;
    }
}

int main() {
    const int N = 1 << 22; // ca. 4 million elements
    const int num_streams = 4;
    const int stream_size = N / num_streams;
    const int stream_bytes = stream_size * sizeof(float);

    float* h_data;
    // pinned memeory
    CHECK_GPU(hipHostMalloc(&h_data, N * sizeof(float))); 

    for (int i = 0; i < N; ++i) h_data[i] = static_cast<float>(i);

    float* d_data;
    CHECK_GPU(hipMalloc(&d_data, N * sizeof(float)));

    // iniate streams
    hipStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        CHECK_GPU(hipStreamCreate(&streams[i]));
    }

    hipEvent_t start, stop;
    CHECK_GPU(hipEventCreate(&start));
    CHECK_GPU(hipEventCreate(&stop));

    std::cout << "ASYNCHRONE STREAMS" << std::endl;
    
    CHECK_GPU(hipEventRecord(start)); 

    // start pipeline
    for (int i = 0; i < num_streams; ++i) {
        int offset = i * stream_size;

        // Chunk i asynchron cpy to gpu
        CHECK_GPU(hipMemcpyAsync(&d_data[offset], &h_data[offset], stream_bytes, hipMemcpyHostToDevice, streams[i]));

        // calculate chunk i
        int threads = 256;
        int blocks = (stream_size + threads - 1) / threads;
        heavy_kernel<<<blocks, threads, 0, streams[i]>>>(&d_data[offset], stream_size, static_cast<float>(i));
        
        CHECK_GPU(hipGetLastError());

        // copy chunk i back asychron
        CHECK_GPU(hipMemcpyAsync(&h_data[offset], &d_data[offset], stream_bytes, hipMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < num_streams; ++i) {
        CHECK_GPU(hipStreamSynchronize(streams[i]));
    }

    CHECK_GPU(hipEventRecord(stop)); 
    CHECK_GPU(hipEventSynchronize(stop)); 

    float milliseconds = 0;
    CHECK_GPU(hipEventElapsedTime(&milliseconds, start, stop));
    std::cout << "runtime with " << num_streams << " parallel streams: " << milliseconds << " ms" << std::endl;

    for (int i = 0; i < num_streams; ++i) {
        CHECK_GPU(hipStreamDestroy(streams[i]));
    }
    CHECK_GPU(hipEventDestroy(start)); 
    CHECK_GPU(hipEventDestroy(stop));
    
    CHECK_GPU(hipFree(d_data)); 
    CHECK_GPU(hipHostFree(h_data));
    
    return 0;
}