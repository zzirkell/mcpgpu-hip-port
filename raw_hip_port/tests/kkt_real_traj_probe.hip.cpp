#include <hip/hip_runtime.h>
#ifdef __HIP_PLATFORM_NVIDIA__
#include <cublas_v2.h>
#else
#include <hipblas.h>
#endif
#include <algorithm>
#include "common/merit.hip.hpp"
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
#include "settings.hip.hpp"
#include "utils/experiment.hip.hpp"
#include "common/kkt.hip.hpp"
#include "pcg/linsys_setup.hip.hpp"
#include "common/dz.hip.hpp"
#include "gpu_pcg.hip.hpp"
template <typename T>
void print_summary(const std::string& name, const std::vector<T>& values)
{
    double sum = 0.0;
    double abs_sum = 0.0;
    double max_abs = 0.0;
    int nonfinite = 0;

    for (T v : values) {
        const double x = static_cast<double>(v);
        if (!std::isfinite(x)) {
            nonfinite++;
            continue;
        }
        sum += x;
        abs_sum += std::abs(x);
        max_abs = std::max(max_abs, std::abs(x));
    }

    std::cout << name << "_count " << values.size() << "\n";
    std::cout << name << "_sum " << std::setprecision(17) << sum << "\n";
    std::cout << name << "_abs_sum " << std::setprecision(17) << abs_sum << "\n";
    std::cout << name << "_max_abs " << std::setprecision(17) << max_abs << "\n";
    std::cout << name << "_nonfinite " << nonfinite << "\n";
    std::cout << name << "_first32 ";
    for (std::size_t i = 0; i < values.size() && i < 32; ++i) {
        std::cout << std::setprecision(9) << static_cast<double>(values[i]) << " ";
    }
    std::cout << "\n";
}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& src)
{
    std::vector<T> out;
    for (const auto& row : src) {
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

int main()
{
    using T = float;

    constexpr uint32_t state_size = grid::NUM_JOINTS * 2;
    constexpr uint32_t control_size = grid::NUM_JOINTS;
    constexpr uint32_t knot_points = KNOT_POINTS;
    constexpr T timestep = static_cast<T>(0.015625);

    const uint32_t states_sq = state_size * state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;

    const uint32_t KKT_G_COUNT = (states_sq + controls_sq) * knot_points - controls_sq;
    const uint32_t KKT_C_COUNT = (states_sq + states_p_controls) * (knot_points - 1);
    const uint32_t KKT_g_COUNT = (state_size + control_size) * knot_points - control_size;
    const uint32_t KKT_c_COUNT = state_size * knot_points;

    auto ee2d = readCSVToVecVec<T>("examples/trajfiles/0_0_eepos.traj");
    auto xu2d = readCSVToVecVec<T>("examples/trajfiles/0_0_traj.csv");

    std::vector<T> h_ee = flatten(ee2d);
    std::vector<T> h_xu = flatten(xu2d);
    std::vector<T> h_xs(h_xu.begin(), h_xu.begin() + state_size);

    std::vector<T> h_G(KKT_G_COUNT, T(0));
    std::vector<T> h_C(KKT_C_COUNT, T(0));
    std::vector<T> h_g(KKT_g_COUNT, T(0));
    std::vector<T> h_c(KKT_c_COUNT, T(0));

    T *d_ee = nullptr;
    T *d_xu = nullptr;
    T *d_xs = nullptr;
    T *d_G = nullptr;
    T *d_C = nullptr;
    T *d_g = nullptr;
    T *d_c = nullptr;

    gpuErrchk(hipMalloc(&d_ee, h_ee.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_xu, h_xu.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_xs, h_xs.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_G, h_G.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_C, h_C.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_g, h_g.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_c, h_c.size() * sizeof(T)));

    gpuErrchk(hipMemcpy(d_ee, h_ee.data(), h_ee.size() * sizeof(T), hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(d_xu, h_xu.data(), h_xu.size() * sizeof(T), hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(d_xs, h_xs.data(), h_xs.size() * sizeof(T), hipMemcpyHostToDevice));
    gpuErrchk(hipMemset(d_G, 0, h_G.size() * sizeof(T)));
    gpuErrchk(hipMemset(d_C, 0, h_C.size() * sizeof(T)));
    gpuErrchk(hipMemset(d_g, 0, h_g.size() * sizeof(T)));
    gpuErrchk(hipMemset(d_c, 0, h_c.size() * sizeof(T)));

    void* d_dynmem = gato_plant::initializeDynamicsConstMem<T>();

    generate_kkt_submatrices<T><<<
        knot_points,
        KKT_THREADS,
        2 * get_kkt_smem_size<T>(state_size, control_size)
    >>>(
        state_size,
        control_size,
        knot_points,
        d_G,
        d_C,
        d_g,
        d_c,
        d_dynmem,
        timestep,
        d_ee,
        d_xs,
        d_xu
    );

    gpuErrchk(hipPeekAtLastError());
    gpuErrchk(hipDeviceSynchronize());

    gpuErrchk(hipMemcpy(h_G.data(), d_G, h_G.size() * sizeof(T), hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_C.data(), d_C, h_C.size() * sizeof(T), hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_g.data(), d_g, h_g.size() * sizeof(T), hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_c.data(), d_c, h_c.size() * sizeof(T), hipMemcpyDeviceToHost));

    print_summary("G", h_G);
    print_summary("C", h_C);
    print_summary("g", h_g);
    print_summary("c", h_c);

    const uint32_t block_tridiag_count = knot_points * 3 * state_size * state_size;
    const uint32_t gamma_count = knot_points * state_size;
    const uint32_t dz_count = knot_points * (state_size + control_size) - control_size;

    std::vector<T> h_S(block_tridiag_count, T(0));
    std::vector<T> h_Pinv(block_tridiag_count, T(0));
    std::vector<T> h_gamma(gamma_count, T(0));
    std::vector<T> h_lambda(gamma_count, T(0));
    std::vector<T> h_dz(dz_count, T(0));

    T *d_S = nullptr;
    T *d_Pinv = nullptr;
    T *d_gamma = nullptr;
    T *d_lambda = nullptr;
    T *d_r = nullptr;
    T *d_p = nullptr;
    T *d_v_temp = nullptr;
    T *d_eta_new_temp = nullptr;
    T *d_dz = nullptr;

    gpuErrchk(hipMalloc(&d_S, h_S.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_Pinv, h_Pinv.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_gamma, h_gamma.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_lambda, h_lambda.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_r, h_lambda.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_p, h_lambda.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_v_temp, knot_points * sizeof(T)));
    gpuErrchk(hipMalloc(&d_eta_new_temp, knot_points * sizeof(T)));
    gpuErrchk(hipMalloc(&d_dz, h_dz.size() * sizeof(T)));

    gpuErrchk(hipMemset(d_S, 0, h_S.size() * sizeof(T)));
    gpuErrchk(hipMemset(d_Pinv, 0, h_Pinv.size() * sizeof(T)));
    gpuErrchk(hipMemset(d_gamma, 0, h_gamma.size() * sizeof(T)));
    gpuErrchk(hipMemset(d_lambda, 0, h_lambda.size() * sizeof(T)));
    gpuErrchk(hipMemset(d_r, 0, h_lambda.size() * sizeof(T)));
    gpuErrchk(hipMemset(d_p, 0, h_lambda.size() * sizeof(T)));
    gpuErrchk(hipMemset(d_v_temp, 0, knot_points * sizeof(T)));
    gpuErrchk(hipMemset(d_eta_new_temp, 0, knot_points * sizeof(T)));
    gpuErrchk(hipMemset(d_dz, 0, h_dz.size() * sizeof(T)));

    T rho = static_cast<T>(1e-3);

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
        rho
    );

    gpuErrchk(hipDeviceSynchronize());

    pcg_config<T> config(static_cast<T>(1e-6), 173, dim3(knot_points), pcg_constants::DEFAULT_BLOCK, 0);

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

    gpuErrchk(hipDeviceSynchronize());

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

    gpuErrchk(hipDeviceSynchronize());

    gpuErrchk(hipMemcpy(h_S.data(), d_S, h_S.size() * sizeof(T), hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_Pinv.data(), d_Pinv, h_Pinv.size() * sizeof(T), hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_gamma.data(), d_gamma, h_gamma.size() * sizeof(T), hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_lambda.data(), d_lambda, h_lambda.size() * sizeof(T), hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_dz.data(), d_dz, h_dz.size() * sizeof(T), hipMemcpyDeviceToHost));

    std::cout << "pcg_iters " << pcg_iters << "\n";
    print_summary("S", h_S);
    print_summary("Pinv", h_Pinv);
    print_summary("gamma", h_gamma);
    print_summary("lambda", h_lambda);
    print_summary("dz", h_dz);
    std::vector<T> h_xu_updated(h_xu.size(), T(0));

    #ifdef __HIP_PLATFORM_NVIDIA__
    cublasHandle_t blas_handle;
    if (cublasCreate(&blas_handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS create failed\n";
        return 2;
    }
    #else
    hipblasHandle_t blas_handle;
    if (hipblasCreate(&blas_handle) != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLAS create failed\n";
        return 2;
    }
    #endif

    T alpha = static_cast<T>(-1.0);

    #ifdef __HIP_PLATFORM_NVIDIA__
    if (cublasSaxpy(blas_handle, dz_count, &alpha, d_dz, 1, d_xu, 1) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Saxpy failed\n";
        return 3;
    }
    #else
    if (hipblasSaxpy(blas_handle, dz_count, &alpha, d_dz, 1, d_xu, 1) != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLAS Saxpy failed\n";
        return 3;
    }
    #endif

    gpuErrchk(hipDeviceSynchronize());
    gpuErrchk(hipMemcpy(h_xu_updated.data(), d_xu, h_xu_updated.size() * sizeof(T), hipMemcpyDeviceToHost));

    print_summary("xu_updated", h_xu_updated);

    constexpr uint32_t num_alphas = 8;
    const T mu = static_cast<T>(10);
    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);

    std::vector<T> h_merit_initial(1, T(0));
    std::vector<T> h_merit_seq(num_alphas, T(0));
    std::vector<T> h_merit_conc(num_alphas, T(0));

    T *d_merit_initial = nullptr;
    T *d_merit_news = nullptr;
    T *d_merit_temp = nullptr;

    gpuErrchk(hipMalloc(&d_merit_initial, sizeof(T)));
    gpuErrchk(hipMalloc(&d_merit_news, num_alphas * sizeof(T)));
    gpuErrchk(hipMalloc(&d_merit_temp, num_alphas * knot_points * sizeof(T)));

    gpuErrchk(hipMemset(d_merit_initial, 0, sizeof(T)));

    compute_merit<T><<<knot_points, MERIT_THREADS, merit_smem_size>>>(
        state_size, control_size, knot_points,
        d_xu, d_ee, mu, timestep, d_dynmem, d_merit_initial
    );
    gpuErrchk(hipDeviceSynchronize());
    gpuErrchk(hipMemcpy(h_merit_initial.data(), d_merit_initial, sizeof(T), hipMemcpyDeviceToHost));
    print_summary("merit_initial", h_merit_initial);

    // Sequential line search merit.
    gpuErrchk(hipMemset(d_merit_news, 0, num_alphas * sizeof(T)));
    gpuErrchk(hipMemset(d_merit_temp, 0, num_alphas * knot_points * sizeof(T)));

    for (uint32_t p = 0; p < num_alphas; ++p) {
        void *args[] = {
            (void *)&state_size, (void *)&control_size, (void *)&knot_points,
            (void *)&d_xs, (void *)&d_xu, (void *)&d_ee,
            (void *)&mu, (void *)&timestep, (void *)&d_dynmem,
            (void *)&d_dz, (void *)&p, (void *)&d_merit_news, (void *)&d_merit_temp
        };

        gpuErrchk(hipLaunchCooperativeKernel(
            reinterpret_cast<const void*>(ls_gato_compute_merit<T>),
            dim3(knot_points), dim3(MERIT_THREADS),
            args, static_cast<unsigned int>(merit_smem_size), 0
        ));
        gpuErrchk(hipDeviceSynchronize());
    }

    gpuErrchk(hipMemcpy(h_merit_seq.data(), d_merit_news, num_alphas * sizeof(T), hipMemcpyDeviceToHost));
    print_summary("merit_seq", h_merit_seq);

    // Concurrent line search merit, like sqp.hip.hpp.
    gpuErrchk(hipMemset(d_merit_news, 0, num_alphas * sizeof(T)));
    gpuErrchk(hipMemset(d_merit_temp, 0, num_alphas * knot_points * sizeof(T)));

    hipStream_t streams[num_alphas];
    uint32_t pvals[num_alphas];
    for (uint32_t p = 0; p < num_alphas; ++p) {
        pvals[p] = p;
        gpuErrchk(hipStreamCreate(&streams[p]));

        void *args[] = {
            (void *)&state_size, (void *)&control_size, (void *)&knot_points,
            (void *)&d_xs, (void *)&d_xu, (void *)&d_ee,
            (void *)&mu, (void *)&timestep, (void *)&d_dynmem,
            (void *)&d_dz, (void *)&pvals[p], (void *)&d_merit_news, (void *)&d_merit_temp
        };

        gpuErrchk(hipLaunchCooperativeKernel(
            reinterpret_cast<const void*>(ls_gato_compute_merit<T>),
            dim3(knot_points), dim3(MERIT_THREADS),
            args, static_cast<unsigned int>(merit_smem_size), streams[p]
        ));
    }

    gpuErrchk(hipDeviceSynchronize());
    gpuErrchk(hipMemcpy(h_merit_conc.data(), d_merit_news, num_alphas * sizeof(T), hipMemcpyDeviceToHost));
    print_summary("merit_conc", h_merit_conc);

    // Concurrent line search merit, exactly like sqp.hip.hpp: pass &p.
    std::vector<T> h_merit_conc_original_p(num_alphas, T(0));

    gpuErrchk(hipMemset(d_merit_news, 0, num_alphas * sizeof(T)));
    gpuErrchk(hipMemset(d_merit_temp, 0, num_alphas * knot_points * sizeof(T)));

    hipStream_t streams_original_p[num_alphas];

    for (uint32_t p = 0; p < num_alphas; ++p) {
        gpuErrchk(hipStreamCreate(&streams_original_p[p]));

        void *args[] = {
            (void *)&state_size,
            (void *)&control_size,
            (void *)&knot_points,
            (void *)&d_xs,
            (void *)&d_xu,
            (void *)&d_ee,
            (void *)&mu,
            (void *)&timestep,
            (void *)&d_dynmem,
            (void *)&d_dz,
            (void *)&p,
            (void *)&d_merit_news,
            (void *)&d_merit_temp
        };

        gpuErrchk(hipLaunchCooperativeKernel(
            reinterpret_cast<const void*>(ls_gato_compute_merit<T>),
            dim3(knot_points),
            dim3(MERIT_THREADS),
            args,
            static_cast<unsigned int>(merit_smem_size),
            streams_original_p[p]
        ));
    }

    gpuErrchk(hipDeviceSynchronize());
    gpuErrchk(hipMemcpy(h_merit_conc_original_p.data(), d_merit_news, num_alphas * sizeof(T), hipMemcpyDeviceToHost));

    print_summary("merit_conc_original_p", h_merit_conc_original_p);

    for (uint32_t p = 0; p < num_alphas; ++p) {
        gpuErrchk(hipStreamDestroy(streams_original_p[p]));
    }
    for (uint32_t p = 0; p < num_alphas; ++p) {
        gpuErrchk(hipStreamDestroy(streams[p]));
    }

    gpuErrchk(hipFree(d_merit_initial));
    gpuErrchk(hipFree(d_merit_news));
    gpuErrchk(hipFree(d_merit_temp));

    #ifdef __HIP_PLATFORM_NVIDIA__
    cublasDestroy(blas_handle);
    #else
    hipblasDestroy(blas_handle);
    #endif

    gpuErrchk(hipFree(d_S));
    gpuErrchk(hipFree(d_Pinv));
    gpuErrchk(hipFree(d_gamma));
    gpuErrchk(hipFree(d_lambda));
    gpuErrchk(hipFree(d_r));
    gpuErrchk(hipFree(d_p));
    gpuErrchk(hipFree(d_v_temp));
    gpuErrchk(hipFree(d_eta_new_temp));
    gpuErrchk(hipFree(d_dz));

    gpuErrchk(hipFree(d_ee));
    gpuErrchk(hipFree(d_xu));
    gpuErrchk(hipFree(d_xs));
    gpuErrchk(hipFree(d_G));
    gpuErrchk(hipFree(d_C));
    gpuErrchk(hipFree(d_g));
    gpuErrchk(hipFree(d_c));

    return 0;
}
