#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "pcg/linsys_setup.cuh"

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << ": " << cudaGetErrorString(err) << std::endl;          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

template <typename T>
bool all_finite(const std::vector<T>& values)
{
    for (T v : values) {
        if (!std::isfinite(static_cast<double>(v))) {
            return false;
        }
    }
    return true;
}

template <typename T>
void print_vector(const std::string& name, const std::vector<T>& values, std::size_t max_count = 12)
{
    std::cout << name << ": ";
    const std::size_t count = std::min(values.size(), max_count);
    for (std::size_t i = 0; i < count; ++i) {
        std::cout << static_cast<double>(values[i]) << " ";
    }
    if (values.size() > count) {
        std::cout << "...";
    }
    std::cout << std::endl;
}

template <typename T>
bool run_one_linsys_test(const std::string& label)
{
    constexpr uint32_t state_size = 2;
    constexpr uint32_t control_size = 1;
    constexpr uint32_t knot_points = 3;

    constexpr uint32_t g_set_size = state_size * state_size + control_size * control_size;
    constexpr uint32_t c_set_size = state_size * state_size + state_size * control_size;

    constexpr std::size_t g_count = knot_points * g_set_size;
    constexpr std::size_t c_count = (knot_points - 1) * c_set_size;
    constexpr std::size_t grad_count = knot_points * (state_size + control_size);
    constexpr std::size_t defect_count = knot_points * state_size;
    constexpr std::size_t block_tridiag_count = knot_points * 3 * state_size * state_size;
    constexpr std::size_t gamma_count = knot_points * state_size;

    std::vector<T> h_G(g_count, T(0));
    std::vector<T> h_C(c_count, T(0));
    std::vector<T> h_g(grad_count, T(0));
    std::vector<T> h_c(defect_count, T(0));
    std::vector<T> h_S(block_tridiag_count, T(0));
    std::vector<T> h_Pinv(block_tridiag_count, T(0));
    std::vector<T> h_gamma(gamma_count, T(0));

    auto set_Q = [&](uint32_t k, T q00, T q11) {
        const std::size_t base = k * g_set_size;
        h_G[base + 0] = q00;
        h_G[base + 1] = T(0);
        h_G[base + 2] = T(0);
        h_G[base + 3] = q11;
    };

    auto set_R = [&](uint32_t k, T r00) {
        const std::size_t base = k * g_set_size + state_size * state_size;
        h_G[base] = r00;
    };

    auto set_A_B = [&](uint32_t k, T a00, T a10, T a01, T a11, T b0, T b1) {
        const std::size_t base = k * c_set_size;
        h_C[base + 0] = a00;
        h_C[base + 1] = a10;
        h_C[base + 2] = a01;
        h_C[base + 3] = a11;
        h_C[base + 4] = b0;
        h_C[base + 5] = b1;
    };

    auto set_grad = [&](uint32_t k, T q0, T q1, T r0) {
        const std::size_t base = k * (state_size + control_size);
        h_g[base + 0] = q0;
        h_g[base + 1] = q1;
        h_g[base + 2] = r0;
    };

    set_Q(0, T(2.0), T(3.0));
    set_R(0, T(1.5));

    set_Q(1, T(2.2), T(2.8));
    set_R(1, T(1.7));

    set_Q(2, T(3.0), T(2.4));
    set_R(2, T(1.9));

    set_A_B(0, T(1.0), T(0.05), T(0.10), T(1.0), T(0.20), T(-0.10));
    set_A_B(1, T(0.9), T(0.03), T(-0.02), T(1.1), T(0.15), T(0.05));

    set_grad(0, T(0.10), T(-0.20), T(0.05));
    set_grad(1, T(0.00), T(0.10), T(-0.03));
    set_grad(2, T(-0.10), T(0.20), T(0.00));

    h_c[0] = T(0.00);
    h_c[1] = T(0.00);
    h_c[2] = T(0.01);
    h_c[3] = T(-0.02);
    h_c[4] = T(-0.01);
    h_c[5] = T(0.03);

    T* d_G = nullptr;
    T* d_C = nullptr;
    T* d_g = nullptr;
    T* d_c = nullptr;
    T* d_S = nullptr;
    T* d_Pinv = nullptr;
    T* d_gamma = nullptr;

    CHECK_CUDA(cudaMalloc(&d_G, h_G.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_C, h_C.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_g, h_g.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_c, h_c.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_S, h_S.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_Pinv, h_Pinv.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_gamma, h_gamma.size() * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_G, h_G.data(), h_G.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_g, h_g.data(), h_g.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), h_c.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_S, 0, h_S.size() * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_Pinv, 0, h_Pinv.size() * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_gamma, 0, h_gamma.size() * sizeof(T)));

    form_schur_system<T>(
        state_size,
        control_size,
        knot_points,
        d_G,
        d_C,
        d_g,
        d_c,
        d_S,
        d_Pinv,
        d_gamma,
        T(0.1)
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, h_S.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_Pinv.data(), d_Pinv, h_Pinv.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gamma.data(), d_gamma, h_gamma.size() * sizeof(T), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_G));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_g));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_Pinv));
    CHECK_CUDA(cudaFree(d_gamma));

    const bool ok = all_finite(h_S) && all_finite(h_Pinv) && all_finite(h_gamma);

    std::cout << "\n" << label << std::endl;
    print_vector("S", h_S);
    print_vector("Pinv", h_Pinv);
    print_vector("gamma", h_gamma);

    if (!ok) {
        std::cerr << "FAIL: non-finite value detected" << std::endl;
        return false;
    }

    std::cout << "OK: linsys setup produced finite S, Pinv, and gamma" << std::endl;
    return true;
}

int main()
{
    int dev = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    if (!prop.cooperativeLaunch) {
        std::cout << "SKIP: device does not support cooperativeLaunch" << std::endl;
        return 0;
    }

    bool ok = true;
    ok &= run_one_linsys_test<float>("float");
    ok &= run_one_linsys_test<double>("double");

    return ok ? 0 : 1;
}
