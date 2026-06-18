#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "gpuassert.hip.hpp"
#include "pcg/linsys_setup.hip.hpp"
#include "common/dz.hip.hpp"
#include "gpu_pcg.hip.hpp"

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << ": " << hipGetErrorString(err) << std::endl;          \
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
    constexpr uint32_t state_size = 14;
    constexpr uint32_t control_size = 7;
    constexpr uint32_t knot_points = 32;

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

    for (uint32_t k = 0; k < knot_points; ++k) {
        const std::size_t gbase = k * g_set_size;

        for (uint32_t i = 0; i < state_size; ++i) {
            h_G[gbase + i * state_size + i] = T(2.0) + T(0.01) * T(k + i);
        }

        if (k < knot_points - 1) {
            for (uint32_t i = 0; i < control_size; ++i) {
                h_G[gbase + state_size * state_size + i * control_size + i] = T(1.5) + T(0.01) * T(k + i);
            }
        }

        const std::size_t grad_base = k * (state_size + control_size);
        for (uint32_t i = 0; i < state_size + control_size; ++i) {
            h_g[grad_base + i] = T(0.001) * T((k + 1) * (i + 1));
        }

        const std::size_t c_base = k * state_size;
        for (uint32_t i = 0; i < state_size; ++i) {
            h_c[c_base + i] = T(0.001) * T(k + i);
        }
    }

    for (uint32_t k = 0; k < knot_points - 1; ++k) {
        const std::size_t cbase = k * c_set_size;

        for (uint32_t i = 0; i < state_size; ++i) {
            h_C[cbase + i * state_size + i] = T(1.0);
        }

        for (uint32_t j = 0; j < control_size; ++j) {
            h_C[cbase + state_size * state_size + j * state_size + (j % state_size)] = T(0.01);
        }
    }

    T* d_G = nullptr;
    T* d_C = nullptr;
    T* d_g = nullptr;
    T* d_c = nullptr;
    T* d_S = nullptr;
    T* d_Pinv = nullptr;
    T* d_gamma = nullptr;

    CHECK_CUDA(hipMalloc(&d_G, h_G.size() * sizeof(T)));
    CHECK_CUDA(hipMalloc(&d_C, h_C.size() * sizeof(T)));
    CHECK_CUDA(hipMalloc(&d_g, h_g.size() * sizeof(T)));
    CHECK_CUDA(hipMalloc(&d_c, h_c.size() * sizeof(T)));
    CHECK_CUDA(hipMalloc(&d_S, h_S.size() * sizeof(T)));
    CHECK_CUDA(hipMalloc(&d_Pinv, h_Pinv.size() * sizeof(T)));
    CHECK_CUDA(hipMalloc(&d_gamma, h_gamma.size() * sizeof(T)));

    CHECK_CUDA(hipMemcpy(d_G, h_G.data(), h_G.size() * sizeof(T), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(d_C, h_C.data(), h_C.size() * sizeof(T), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(d_g, h_g.data(), h_g.size() * sizeof(T), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(d_c, h_c.data(), h_c.size() * sizeof(T), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemset(d_S, 0, h_S.size() * sizeof(T)));
    CHECK_CUDA(hipMemset(d_Pinv, 0, h_Pinv.size() * sizeof(T)));
    CHECK_CUDA(hipMemset(d_gamma, 0, h_gamma.size() * sizeof(T)));

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

    CHECK_CUDA(hipDeviceSynchronize());
    std::vector<T> h_lambda(gamma_count, T(0));
    std::vector<T> h_r(gamma_count, T(0));
    std::vector<T> h_p(gamma_count, T(0));
    std::vector<T> h_v_temp(knot_points, T(0));
    std::vector<T> h_eta_new_temp(knot_points, T(0));

    T* d_lambda = nullptr;
    T* d_r = nullptr;
    T* d_p = nullptr;
    T* d_v_temp = nullptr;
    T* d_eta_new_temp = nullptr;

    CHECK_CUDA(hipMalloc(&d_lambda, h_lambda.size() * sizeof(T)));
    CHECK_CUDA(hipMalloc(&d_r, h_r.size() * sizeof(T)));
    CHECK_CUDA(hipMalloc(&d_p, h_p.size() * sizeof(T)));
    CHECK_CUDA(hipMalloc(&d_v_temp, h_v_temp.size() * sizeof(T)));
    CHECK_CUDA(hipMalloc(&d_eta_new_temp, h_eta_new_temp.size() * sizeof(T)));

    CHECK_CUDA(hipMemset(d_lambda, 0, h_lambda.size() * sizeof(T)));
    CHECK_CUDA(hipMemset(d_r, 0, h_r.size() * sizeof(T)));
    CHECK_CUDA(hipMemset(d_p, 0, h_p.size() * sizeof(T)));
    CHECK_CUDA(hipMemset(d_v_temp, 0, h_v_temp.size() * sizeof(T)));
    CHECK_CUDA(hipMemset(d_eta_new_temp, 0, h_eta_new_temp.size() * sizeof(T)));

    pcg_config<T> config(T(1e-6), 173, dim3(knot_points), pcg_constants::DEFAULT_BLOCK, 0);

    uint32_t pcg_iters = solvePCG(
        state_size,
        knot_points,
        d_S,
        d_Pinv,
        d_gamma,
        d_lambda,
        d_r,
        d_p,
        d_v_temp,
        d_eta_new_temp,
        &config
    );

    CHECK_CUDA(hipDeviceSynchronize());
    CHECK_CUDA(hipMemcpy(h_lambda.data(), d_lambda, h_lambda.size() * sizeof(T), hipMemcpyDeviceToHost));

    T lambda_sum = T(0);
    T lambda_abs_sum = T(0);
    for (T v : h_lambda) {
        lambda_sum += v;
        lambda_abs_sum += std::abs(v);
    }

    std::cout << "pcg_iters: " << pcg_iters << std::endl;
    std::cout << "lambda_sum: " << static_cast<double>(lambda_sum) << std::endl;
    std::cout << "lambda_abs_sum: " << static_cast<double>(lambda_abs_sum) << std::endl;
    print_vector("lambda", h_lambda, 32);

    const std::size_t dz_count = knot_points * (state_size + control_size) - control_size;
    std::vector<T> h_dz(dz_count, T(0));
    T* d_dz = nullptr;

    CHECK_CUDA(hipMalloc(&d_dz, h_dz.size() * sizeof(T)));
    CHECK_CUDA(hipMemset(d_dz, 0, h_dz.size() * sizeof(T)));

    compute_dz(
        state_size,
        control_size,
        knot_points,
        d_G,
        d_C,
        d_g,
        d_lambda,
        d_dz
    );

    CHECK_CUDA(hipDeviceSynchronize());
    CHECK_CUDA(hipMemcpy(h_dz.data(), d_dz, h_dz.size() * sizeof(T), hipMemcpyDeviceToHost));

    T dz_sum = T(0);
    T dz_abs_sum = T(0);
    for (T v : h_dz) {
        dz_sum += v;
        dz_abs_sum += std::abs(v);
    }

    std::cout << "dz_sum: " << static_cast<double>(dz_sum) << std::endl;
    std::cout << "dz_abs_sum: " << static_cast<double>(dz_abs_sum) << std::endl;
    print_vector("dz", h_dz, 32);

    CHECK_CUDA(hipFree(d_dz));
    CHECK_CUDA(hipFree(d_lambda));
    CHECK_CUDA(hipFree(d_r));
    CHECK_CUDA(hipFree(d_p));
    CHECK_CUDA(hipFree(d_v_temp));
    CHECK_CUDA(hipFree(d_eta_new_temp));

    CHECK_CUDA(hipMemcpy(h_S.data(), d_S, h_S.size() * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_CUDA(hipMemcpy(h_Pinv.data(), d_Pinv, h_Pinv.size() * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_CUDA(hipMemcpy(h_gamma.data(), d_gamma, h_gamma.size() * sizeof(T), hipMemcpyDeviceToHost));

    CHECK_CUDA(hipFree(d_G));
    CHECK_CUDA(hipFree(d_C));
    CHECK_CUDA(hipFree(d_g));
    CHECK_CUDA(hipFree(d_c));
    CHECK_CUDA(hipFree(d_S));
    CHECK_CUDA(hipFree(d_Pinv));
    CHECK_CUDA(hipFree(d_gamma));

    const bool ok = all_finite(h_S) && all_finite(h_Pinv) && all_finite(h_gamma);

    std::cout << "\n" << label << std::endl;
    print_vector("S", h_S, h_S.size());
    print_vector("Pinv", h_Pinv, h_Pinv.size());
    print_vector("gamma", h_gamma, h_gamma.size());

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
    hipDeviceProp_t prop{};
    CHECK_CUDA(hipGetDeviceProperties(&prop, dev));

    if (!prop.cooperativeLaunch) {
        std::cout << "SKIP: device does not support cooperativeLaunch" << std::endl;
        return 0;
    }

    bool ok = true;
    ok &= run_one_linsys_test<float>("float");

    return ok ? 0 : 1;
}
