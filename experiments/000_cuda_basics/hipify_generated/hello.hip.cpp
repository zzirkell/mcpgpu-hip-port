#include <cstdio>
#include <hip/hip_runtime.h>

__global__
void hello_from_gpu() {
    printf("Hello from GPU! block=%d thread=%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello_from_gpu<<<2, 4>>>();
    hipDeviceSynchronize();

    return 0;
}