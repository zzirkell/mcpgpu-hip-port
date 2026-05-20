#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
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
    CHECK_GPU(cudaMallocHost(&h_data, N * sizeof(float))); 

    for (int i = 0; i < N; ++i) h_data[i] = static_cast<float>(i);

    float* d_data;
    CHECK_GPU(cudaMalloc(&d_data, N * sizeof(float)));

    // iniate streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        CHECK_GPU(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CHECK_GPU(cudaEventCreate(&start));
    CHECK_GPU(cudaEventCreate(&stop));

    std::cout << "Asynchrone streams" << std::endl;
    
    CHECK_GPU(cudaEventRecord(start)); 

    // start pipeline
    for (int i = 0; i < num_streams; ++i) {
        int offset = i * stream_size;

        // Chunk i asynchron cpy to gpu
        CHECK_GPU(cudaMemcpyAsync(&d_data[offset], &h_data[offset], stream_bytes, cudaMemcpyHostToDevice, streams[i]));

        // calculate chunk i
        int threads = 256;
        int blocks = (stream_size + threads - 1) / threads;
        heavy_kernel<<<blocks, threads, 0, streams[i]>>>(&d_data[offset], stream_size, static_cast<float>(i));
        
        CHECK_GPU(cudaGetLastError());

        // copy chunk i back asychron
        CHECK_GPU(cudaMemcpyAsync(&h_data[offset], &d_data[offset], stream_bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < num_streams; ++i) {
        CHECK_GPU(cudaStreamSynchronize(streams[i]));
    }

    CHECK_GPU(cudaEventRecord(stop)); 
    CHECK_GPU(cudaEventSynchronize(stop)); 

    float milliseconds = 0;
    CHECK_GPU(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "runtime with " << num_streams << " parallel streams: " << milliseconds << " ms" << std::endl;

    for (int i = 0; i < num_streams; ++i) {
        CHECK_GPU(cudaStreamDestroy(streams[i]));
    }
    CHECK_GPU(cudaEventDestroy(start)); 
    CHECK_GPU(cudaEventDestroy(stop));
    
    CHECK_GPU(cudaFree(d_data)); 
    CHECK_GPU(cudaFreeHost(h_data));
    
    return 0;
}