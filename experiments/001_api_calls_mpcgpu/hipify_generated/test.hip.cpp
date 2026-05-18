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

int main() {
    int h_in = 42;
    int h_out = 0;
    int *d_value = nullptr;

    CHECK_GPU(hipMalloc((void**)&d_value, sizeof(int)));

    CHECK_GPU(hipMemcpy(d_value, &h_in, sizeof(int), hipMemcpyHostToDevice));
    CHECK_GPU(hipMemcpy(&h_out, d_value, sizeof(int), hipMemcpyDeviceToHost));

    CHECK_GPU(hipFree(d_value));

    if (h_out != h_in) {
        std::fprintf(stderr, "Wrong result: expected %d, got %d\n", h_in, h_out);
        return 1;
    }

    std::printf("OK: memory API test passed, value = %d\n", h_out);
    return 0;
}