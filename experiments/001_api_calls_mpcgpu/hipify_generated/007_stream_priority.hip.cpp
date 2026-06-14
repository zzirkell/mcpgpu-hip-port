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

__global__ void dummyKernel(float *data) {
    int idx = threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}

int main() {
    int leastPriority;
    int greatestPriority;
    
    CHECK_GPU(hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

    hipStream_t highPriorityStream;
    CHECK_GPU(hipStreamCreateWithPriority(&highPriorityStream, hipStreamNonBlocking, greatestPriority));

    float *d_data;
    CHECK_GPU(hipMalloc(&d_data, 256 * sizeof(float)));

    dummyKernel<<<1, 256, 0, highPriorityStream>>>(d_data);
    CHECK_GPU(hipGetLastError());

    CHECK_GPU(hipStreamSynchronize(highPriorityStream));

    std::cout << "OK: 007_stream_priority" << std::endl;

    CHECK_GPU(hipFree(d_data));
    CHECK_GPU(hipStreamDestroy(highPriorityStream));

    return 0;
}