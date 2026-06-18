#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "gpuassert.hip.hpp"
#include "settings.hip.hpp"
#include "utils/experiment.hip.hpp"
#include "pcg/sqp.hip.hpp"

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& src)
{
    std::vector<T> out;
    for (const auto& row : src) {
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

template <typename T>
void print_summary(const std::string& name, const std::vector<T>& values)
{
    double sum = 0.0;
    double abs_sum = 0.0;
    for (T v : values) {
        const double x = static_cast<double>(v);
        sum += x;
        abs_sum += std::abs(x);
    }

    std::cout << name << "_sum " << std::setprecision(17) << sum << "\n";
    std::cout << name << "_abs_sum " << std::setprecision(17) << abs_sum << "\n";
    std::cout << name << "_first32 ";
    for (std::size_t i = 0; i < values.size() && i < 32; ++i) {
        std::cout << std::setprecision(9) << static_cast<double>(values[i]) << " ";
    }
    std::cout << "\n";
}

int main()
{
    using T = float;

    constexpr uint32_t state_size = grid::NUM_JOINTS * 2;
    constexpr uint32_t control_size = grid::NUM_JOINTS;
    constexpr uint32_t knot_points = KNOT_POINTS;
    constexpr T timestep = static_cast<T>(0.015625);

    const uint32_t traj_len = (state_size + control_size) * knot_points - control_size;

    auto ee2d = readCSVToVecVec<T>("examples/trajfiles/0_0_eepos.traj");
    auto xu2d = readCSVToVecVec<T>("examples/trajfiles/0_0_traj.csv");

    std::vector<T> h_ee = flatten(ee2d);
    std::vector<T> h_xu = flatten(xu2d);
    std::vector<T> h_xs(h_xu.begin(), h_xu.begin() + state_size);
    std::vector<T> h_lambda(state_size * knot_points, T(0));

    T *d_ee = nullptr;
    T *d_xu = nullptr;
    T *d_xs = nullptr;
    T *d_lambda = nullptr;

    gpuErrchk(hipMalloc(&d_ee, h_ee.size() * sizeof(T)));
    gpuErrchk(hipMalloc(&d_xu, traj_len * sizeof(T)));
    gpuErrchk(hipMalloc(&d_xs, state_size * sizeof(T)));
    gpuErrchk(hipMalloc(&d_lambda, h_lambda.size() * sizeof(T)));

    gpuErrchk(hipMemcpy(d_ee, h_ee.data(), h_ee.size() * sizeof(T), hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(d_xu, h_xu.data(), traj_len * sizeof(T), hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(d_xs, h_xs.data(), state_size * sizeof(T), hipMemcpyHostToDevice));
    gpuErrchk(hipMemset(d_lambda, 0, h_lambda.size() * sizeof(T)));

    void* d_dynmem = gato_plant::initializeDynamicsConstMem<T>();

    pcg_config<T> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = static_cast<T>(5e-6);
    config.pcg_max_iter = PCG_MAX_ITER;

    T rho = static_cast<T>(1e-3);
    T rho_reset = static_cast<T>(1e-3);

    auto stats = sqpSolvePcg<T>(
        state_size,
        control_size,
        knot_points,
        timestep,
        d_ee,
        d_lambda,
        d_xu,
        d_dynmem,
        config,
        rho,
        rho_reset
    );

    gpuErrchk(hipMemcpy(h_xu.data(), d_xu, traj_len * sizeof(T), hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_lambda.data(), d_lambda, h_lambda.size() * sizeof(T), hipMemcpyDeviceToHost));

    std::cout << "returned_sqp_iters " << std::get<3>(stats) << "\n";
    std::cout << "returned_sqp_time_exit " << std::get<4>(stats) << "\n";
    std::cout << "final_rho " << std::setprecision(9) << rho << "\n";
    print_summary("final_xu", h_xu);
    print_summary("final_lambda", h_lambda);

    gpuErrchk(hipFree(d_ee));
    gpuErrchk(hipFree(d_xu));
    gpuErrchk(hipFree(d_xs));
    gpuErrchk(hipFree(d_lambda));

    return 0;
}
