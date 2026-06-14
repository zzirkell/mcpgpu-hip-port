#include "hip/hip_runtime.h"
#include <iostream>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                            \
        if (err != hipSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, hipGetErrorString(err));     \
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
    
    CHECK_GPU(hipMalloc(&d_data, bytes));

    hipStream_t stream;
    CHECK_GPU(hipStreamCreate(&stream));

    CHECK_GPU(hipMemcpyAsync(d_data, h_in, bytes, hipMemcpyHostToDevice, stream));
    
    simpleKernel<<<1, size, 0, stream>>>(d_data);

    CHECK_GPU(hipGetLastError());

    CHECK_GPU(hipMemcpyAsync(h_out, d_data, bytes, hipMemcpyDeviceToHost, stream));

    CHECK_GPU(hipStreamSynchronize(stream));

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

    CHECK_GPU(hipStreamDestroy(stream));
    CHECK_GPU(hipFree(d_data));

    return 0;
}