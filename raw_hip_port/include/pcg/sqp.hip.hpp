#include "hip/hip_runtime.h"
#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#ifdef __HIP_PLATFORM_NVIDIA__
#include <cublas_v2.h>
#else
#include <hipblas.h>
#endif  
#include <math.h>
#include <cmath>
#include <random>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <tuple>
#include <time.h>
#include "linsys_setup.hip.hpp"
#include "common/kkt.hip.hpp"
#include "common/dz.hip.hpp"
#include "merit.hip.hpp"
#include "gpu_pcg.hip.hpp"
#include "settings.hip.hpp"
#include <cstdio>
#include <cstdlib>
#include <array>

template <typename T>
struct SqpWorkspace {
    static constexpr uint32_t num_alphas = 8;

    std::array<hipStream_t, num_alphas> streams{};

#ifdef __HIP_PLATFORM_NVIDIA__
    cublasHandle_t handle{};
#else
    hipblasHandle_t handle{};
#endif

    T *d_merit_initial{};
    T *d_merit_news{};
    T *d_merit_temp{};
    T *d_G_dense{};
    T *d_C_dense{};
    T *d_g{};
    T *d_c{};
    T *d_S{};
    T *d_gamma{};
    T *d_dz{};
    T *d_xs{};
    T *d_Pinv{};
    T *d_r{};
    T *d_p{};
    T *d_v_temp{};
    T *d_eta_new_temp{};

    uint32_t *d_pcg_iters{};
    bool *d_pcg_exit{};

    SqpWorkspace(
        uint32_t state_size,
        uint32_t control_size,
        uint32_t knot_points
    ) {
        const uint32_t states_sq = state_size * state_size;
        const uint32_t controls_sq = control_size * control_size;
        const uint32_t states_p_controls = state_size * control_size;
        const uint32_t states_s_controls = state_size + control_size;

        const size_t g_dense_bytes =
            ((states_sq + controls_sq) * knot_points - controls_sq)
            * sizeof(T);

        const size_t c_dense_bytes =
            (states_sq + states_p_controls) * (knot_points - 1)
            * sizeof(T);

        const size_t g_bytes =
            ((state_size + control_size) * knot_points - control_size)
            * sizeof(T);

        const size_t c_bytes =
            state_size * knot_points * sizeof(T);

        const size_t dz_bytes =
            (states_s_controls * knot_points - control_size)
            * sizeof(T);

        for (auto& stream : streams) {
            gpuErrchk(hipStreamCreate(&stream));
        }

#ifdef __HIP_PLATFORM_NVIDIA__
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "cuBLAS initialization failed\n");
            std::exit(13);
        }
#else
        if (hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "hipBLAS initialization failed\n");
            std::exit(13);
        }
#endif

        gpuErrchk(hipMalloc(&d_G_dense, g_dense_bytes));
        gpuErrchk(hipMalloc(&d_C_dense, c_dense_bytes));
        gpuErrchk(hipMalloc(&d_g, g_bytes));
        gpuErrchk(hipMalloc(&d_c, c_bytes));
        gpuErrchk(hipMalloc(
            &d_S, 3 * states_sq * knot_points * sizeof(T)));
        gpuErrchk(hipMalloc(
            &d_gamma, state_size * knot_points * sizeof(T)));
        gpuErrchk(hipMalloc(&d_dz, dz_bytes));
        gpuErrchk(hipMalloc(&d_xs, state_size * sizeof(T)));

        gpuErrchk(hipMalloc(
            &d_merit_news, num_alphas * sizeof(T)));
        gpuErrchk(hipMalloc(
            &d_merit_temp, num_alphas * knot_points * sizeof(T)));
        gpuErrchk(hipMalloc(&d_merit_initial, sizeof(T)));

        gpuErrchk(hipMalloc(
            &d_Pinv, 3 * states_sq * knot_points * sizeof(T)));
        gpuErrchk(hipMalloc(
            &d_r, state_size * knot_points * sizeof(T)));
        gpuErrchk(hipMalloc(
            &d_p, state_size * knot_points * sizeof(T)));
        gpuErrchk(hipMalloc(
            &d_v_temp, knot_points * sizeof(T)));
        gpuErrchk(hipMalloc(
            &d_eta_new_temp, knot_points * sizeof(T)));

        gpuErrchk(hipMalloc(&d_pcg_iters, sizeof(uint32_t)));
        gpuErrchk(hipMalloc(&d_pcg_exit, sizeof(bool)));
    }

    SqpWorkspace(const SqpWorkspace&) = delete;
    SqpWorkspace& operator=(const SqpWorkspace&) = delete;

    ~SqpWorkspace() {
        gpuErrchk(hipDeviceSynchronize());

        gpuErrchk(hipFree(d_merit_initial));
        gpuErrchk(hipFree(d_merit_news));
        gpuErrchk(hipFree(d_merit_temp));
        gpuErrchk(hipFree(d_G_dense));
        gpuErrchk(hipFree(d_C_dense));
        gpuErrchk(hipFree(d_g));
        gpuErrchk(hipFree(d_c));
        gpuErrchk(hipFree(d_S));
        gpuErrchk(hipFree(d_gamma));
        gpuErrchk(hipFree(d_dz));
        gpuErrchk(hipFree(d_xs));
        gpuErrchk(hipFree(d_Pinv));
        gpuErrchk(hipFree(d_r));
        gpuErrchk(hipFree(d_p));
        gpuErrchk(hipFree(d_v_temp));
        gpuErrchk(hipFree(d_eta_new_temp));
        gpuErrchk(hipFree(d_pcg_iters));
        gpuErrchk(hipFree(d_pcg_exit));

#ifdef __HIP_PLATFORM_NVIDIA__
        cublasDestroy(handle);
#else
        hipblasDestroy(handle);
#endif

        for (auto stream : streams) {
            gpuErrchk(hipStreamDestroy(stream));
        }
    }
};

template <typename T>
#if USE_SQP_WORKSPACE
auto sqpSolvePcg(
    const uint32_t state_size,
    const uint32_t control_size,
    const uint32_t knot_points,
    float timestep,
    T *d_eePos_traj,
    T *d_lambda,
    T *d_xu,
    void *d_dynMem_const,
    pcg_config<T>& config,
    T &rho,
    T rho_reset,
    SqpWorkspace<T>& workspace
)
#else
auto sqpSolvePcg(
    const uint32_t state_size,
    const uint32_t control_size,
    const uint32_t knot_points,
    float timestep,
    T *d_eePos_traj,
    T *d_lambda,
    T *d_xu,
    void *d_dynMem_const,
    pcg_config<T>& config,
    T &rho,
    T rho_reset
)
#endif
{
    
    // data storage
    std::vector<int> pcg_iter_vec;
    std::vector<bool> pcg_exit_vec;
    std::vector<double> linsys_time_vec;
    
    std::vector<double> ktt_time_vec;
    std::vector<double> shur_time_vec;
    std::vector<double> dz_time_vec;
    std::vector<double> line_search_time_vec;

    bool sqp_time_exit = 1;     // for data recording, not a flag
    

#if FINE_GRAINED_TIMING
    timespec ktt_start, ktt_end, shur_start, shur_end, dz_start, dz_end, ls_start, ls_end;
#endif // FINE_GRAINED_TIMING

    // sqp timing
    struct timespec sqp_solve_start, sqp_solve_end;
    gpuErrchk(hipDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_start);



    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t DZ_SIZE_BYTES          =   static_cast<uint32_t>((states_s_controls*knot_points-control_size)*sizeof(T));


    // line search things
    const float mu = 10.0f;
    constexpr uint32_t num_alphas = 8;
    std::array<T, num_alphas> h_merit_news{};
    void *ls_merit_kernel = (void *) ls_gato_compute_merit<T>;
    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);
    T h_merit_initial, min_merit;
    T alphafinal;
    T delta_merit_iter = 0;
    T delta_merit_total = 0;
    uint32_t line_search_step = 0;


    uint32_t sqp_iter = 0;

    T drho = static_cast<T>(1);
    T rho_factor = static_cast<T>(RHO_FACTOR);
    T rho_max = static_cast<T>(RHO_MAX);
    T rho_min = static_cast<T>(RHO_MIN);

#if USE_SQP_WORKSPACE

    auto& streams = workspace.streams;
    auto handle = workspace.handle;

    T *d_merit_initial = workspace.d_merit_initial;
    T *d_merit_news = workspace.d_merit_news;
    T *d_merit_temp = workspace.d_merit_temp;
    T *d_G_dense = workspace.d_G_dense;
    T *d_C_dense = workspace.d_C_dense;
    T *d_g = workspace.d_g;
    T *d_c = workspace.d_c;
    T *d_Ginv_dense = workspace.d_G_dense;
    T *d_S = workspace.d_S;
    T *d_gamma = workspace.d_gamma;
    T *d_dz = workspace.d_dz;
    T *d_xs = workspace.d_xs;
    T *d_Pinv = workspace.d_Pinv;
    T *d_r = workspace.d_r;
    T *d_p = workspace.d_p;
    T *d_v_temp = workspace.d_v_temp;
    T *d_eta_new_temp = workspace.d_eta_new_temp;
    uint32_t *d_pcg_iters = workspace.d_pcg_iters;
    bool *d_pcg_exit = workspace.d_pcg_exit;

    gpuErrchk(hipMemcpy(
        d_xs,
        d_xu,
        state_size * sizeof(T),
        hipMemcpyDeviceToDevice
    ));

    // Workspace buffers are reused across SQP solves.
    // Reset scratch/output buffers that previously came from fresh allocations.
    gpuErrchk(hipMemset(d_merit_initial, 0, sizeof(T)));
    gpuErrchk(hipMemset(d_merit_news, 0, num_alphas * sizeof(T)));
    gpuErrchk(hipMemset(d_merit_temp, 0, num_alphas * knot_points * sizeof(T)));

    gpuErrchk(hipMemset(d_S, 0, 3 * state_size * state_size * knot_points * sizeof(T)));
    gpuErrchk(hipMemset(d_gamma, 0, state_size * knot_points * sizeof(T)));
    gpuErrchk(hipMemset(d_dz, 0, DZ_SIZE_BYTES));

    gpuErrchk(hipMemset(d_Pinv, 0, 3 * state_size * state_size * knot_points * sizeof(T)));
    gpuErrchk(hipMemset(d_r, 0, state_size * knot_points * sizeof(T)));
    gpuErrchk(hipMemset(d_p, 0, state_size * knot_points * sizeof(T)));
    gpuErrchk(hipMemset(d_v_temp, 0, knot_points * sizeof(T)));
    gpuErrchk(hipMemset(d_eta_new_temp, 0, knot_points * sizeof(T)));

    gpuErrchk(hipMemset(d_pcg_iters, 0, sizeof(uint32_t)));
    gpuErrchk(hipMemset(d_pcg_exit, 0, sizeof(bool)));

#else

    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t KKT_G_DENSE_SIZE_BYTES = static_cast<uint32_t>(((states_sq+controls_sq)*knot_points-controls_sq)*sizeof(T));
    const uint32_t KKT_C_DENSE_SIZE_BYTES = static_cast<uint32_t>((states_sq+states_p_controls)*(knot_points-1)*sizeof(T));
    const uint32_t KKT_g_SIZE_BYTES       = static_cast<uint32_t>(((state_size+control_size)*knot_points-control_size)*sizeof(T));
    const uint32_t KKT_c_SIZE_BYTES       = static_cast<uint32_t>((state_size*knot_points)*sizeof(T));     

    hipStream_t streams[num_alphas];
    for(uint32_t str = 0; str < num_alphas; str++){
        hipStreamCreate(&streams[str]);
    }
    hipblasHandle_t handle;
    hipblasCreate(&handle);

    T *d_merit_initial, *d_merit_news, *d_merit_temp,
      *d_G_dense, *d_C_dense, *d_g, *d_c, *d_Ginv_dense,
      *d_S, *d_gamma,
      *d_dz,
      *d_xs;


    gpuErrchk(hipMalloc(&d_G_dense,  KKT_G_DENSE_SIZE_BYTES));
    gpuErrchk(hipMalloc(&d_C_dense,  KKT_C_DENSE_SIZE_BYTES));
    gpuErrchk(hipMalloc(&d_g,        KKT_g_SIZE_BYTES));
    gpuErrchk(hipMalloc(&d_c,        KKT_c_SIZE_BYTES));
    d_Ginv_dense = d_G_dense;

    gpuErrchk(hipMalloc(&d_S, 3*states_sq*knot_points*sizeof(T)));
    gpuErrchk(hipMalloc(&d_gamma, state_size*knot_points*sizeof(T)));
    
    gpuErrchk(hipMalloc(&d_dz,       DZ_SIZE_BYTES));
    gpuErrchk(hipMalloc(&d_xs,       state_size*sizeof(T)));
    gpuErrchk(hipMemcpy(d_xs, d_xu,  state_size*sizeof(T), hipMemcpyDeviceToDevice));
    gpuErrchk(hipMalloc(&d_merit_news, 8*sizeof(T)));
    gpuErrchk(hipMalloc(&d_merit_temp, 8*knot_points*sizeof(T)));

    gpuErrchk(hipMalloc(&d_merit_initial, sizeof(T)));
    gpuErrchk(hipMemset(d_merit_initial, 0, sizeof(T)));

    T *d_Pinv;
    gpuErrchk(hipMalloc(&d_Pinv, 3*states_sq*knot_points*sizeof(T)));
    
    T  *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(hipMalloc(&d_r, state_size*knot_points*sizeof(T)));
    gpuErrchk(hipMalloc(&d_p, state_size*knot_points*sizeof(T)));
    gpuErrchk(hipMalloc(&d_v_temp, knot_points*sizeof(T)));
    gpuErrchk(hipMalloc(&d_eta_new_temp, knot_points*sizeof(T)));
    
    uint32_t *d_pcg_iters;
    gpuErrchk(hipMalloc(&d_pcg_iters, sizeof(uint32_t)));
    bool *d_pcg_exit;
    gpuErrchk(hipMalloc(&d_pcg_exit, sizeof(bool)));

#endif

    void *pcg_kernel = (void *) pcg<T, STATE_SIZE, KNOT_POINTS>;
    uint32_t pcg_iters = 0;
    bool pcg_exit = false;
    
    void *pcgKernelArgs[] = {
        (void *)&d_S,
        (void *)&d_Pinv,
        (void *)&d_gamma, 
        (void *)&d_lambda,
        (void *)&d_r,
        (void *)&d_p,
        (void *)&d_v_temp,
        (void *)&d_eta_new_temp,
        (void *)&d_pcg_iters,
        (void *)&d_pcg_exit,
        (void *)&config.pcg_max_iter,
        (void *)&config.pcg_exit_tol
    };
    size_t ppcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points);


    gpuErrchk(hipPeekAtLastError());
    gpuErrchk(hipDeviceSynchronize());

#if TIME_LINSYS
    struct timespec linsys_start, linsys_end;
    double linsys_time;
#endif
#if CONST_UPDATE_FREQ
    struct timespec sqp_cur;
    auto sqpTimecheck = [&]() {
        clock_gettime(CLOCK_MONOTONIC, &sqp_cur);
        return time_delta_us_timespec(sqp_solve_start,sqp_cur) > SQP_MAX_TIME_US;
    };
#else
    auto sqpTimecheck = [&]() { return false; };
#endif


    ///TODO: atomic race conditions here aren't fixed but don't seem to be problematic
    compute_merit<T><<<knot_points, MERIT_THREADS, merit_smem_size>>>(
        state_size, control_size, knot_points,
        d_xu, 
        d_eePos_traj, 
        static_cast<T>(10), 
        timestep, 
        d_dynMem_const, 
        d_merit_initial
    );
    gpuErrchk(hipMemcpyAsync(&h_merit_initial, d_merit_initial, sizeof(T), hipMemcpyDeviceToHost));
    gpuErrchk(hipPeekAtLastError());

    //
    //      SQP LOOP
    //
    for(uint32_t sqpiter = 0; sqpiter < SQP_MAX_ITER; sqpiter++){

    #if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&ktt_start);
    #endif // FINE_GRAINED_TIMING
        
        generate_kkt_submatrices<T><<<knot_points, KKT_THREADS, 2 * get_kkt_smem_size<T>(state_size, control_size)>>>(
            state_size,
            control_size,
            knot_points,
            d_G_dense, 
            d_C_dense, 
            d_g, 
            d_c,
            d_dynMem_const,
            timestep,
            d_eePos_traj,
            d_xs,
            d_xu
        );
        gpuErrchk(hipPeekAtLastError());

    #if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&ktt_end);
        double ktt_time = time_delta_us_timespec(ktt_start,ktt_end);
        ktt_time_vec.push_back(ktt_time);
    #endif // FINE_GRAINED_TIMING


        if (sqpTimecheck()){ break; }

    #if FINE_GRAINED_TIMING
        // NOTE: probably not necessary to synchronize here, but it is done for consistency
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&shur_start);
    #endif // FINE_GRAINED_TIMING

        form_schur_system<T>(
            state_size, 
            control_size, 
            knot_points, 
            d_G_dense, 
            d_C_dense, 
            d_g, 
            d_c,
            d_S, 
            d_Pinv, 
            d_gamma,
            rho
        );
        gpuErrchk(hipPeekAtLastError());

    #if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&shur_end);
        double shur_time = time_delta_us_timespec(shur_start,shur_end);
        shur_time_vec.push_back(shur_time);
    #endif // FINE_GRAINED_TIMING

        if (sqpTimecheck()){ break; }
        

    #if TIME_LINSYS    
        gpuErrchk(hipDeviceSynchronize());
        if (sqpTimecheck()){ break; }
        clock_gettime(CLOCK_MONOTONIC,&linsys_start);
    #endif // #if TIME_LINSYS

        //gpuErrchk(hipLaunchCooperativeKernel(reinterpret_cast<const void*>(pcg_kernel), knot_points, PCG_NUM_THREADS, pcgKernelArgs, ppcg_kernel_smem_size));
        dim3 pcg_grid(knot_points);
        dim3 pcg_block(PCG_NUM_THREADS);
        hipStream_t pcg_stream = 0;

        gpuErrchk(hipLaunchCooperativeKernel(
            reinterpret_cast<const void*>(pcg_kernel),
            pcg_grid,
            pcg_block,
            pcgKernelArgs,
            static_cast<unsigned int>(ppcg_kernel_smem_size),
            pcg_stream
        ));    
        gpuErrchk(hipMemcpy(&pcg_iters, d_pcg_iters, sizeof(uint32_t), hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(&pcg_exit, d_pcg_exit, sizeof(bool), hipMemcpyDeviceToHost));
        gpuErrchk(hipPeekAtLastError());

    #if TIME_LINSYS
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&linsys_end);
        
        linsys_time = time_delta_us_timespec(linsys_start,linsys_end);
        linsys_time_vec.push_back(linsys_time);
    #endif // #if TIME_LINSYS

        pcg_iter_vec.push_back(pcg_iters);
        pcg_exit_vec.push_back(pcg_exit);

        
        if (sqpTimecheck()){ break; }

    #if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&dz_start);
    #endif // FINE_GRAINED_TIMING
        
        // recover dz
        compute_dz(
            state_size,
            control_size,
            knot_points,
            d_Ginv_dense, 
            d_C_dense, 
            d_g, 
            d_lambda, 
            d_dz
        );
        gpuErrchk(hipPeekAtLastError());

    #if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&dz_end);
        double dz_time = time_delta_us_timespec(dz_start,dz_end);
        dz_time_vec.push_back(dz_time);
    #endif // FINE_GRAINED_TIMING

        if (sqpTimecheck()){ break; }
        
    #if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC,&ls_start);
    #endif // FINE_GRAINED_TIMING


        // line search
        for(uint32_t p = 0; p < num_alphas; p++){
            void *kernelArgs[] = {
                (void *)&state_size,
                (void *)&control_size,
                (void *)&knot_points,
                (void *)&d_xs,
                (void *)&d_xu,
                (void *)&d_eePos_traj,
                (void *)&mu, 
                (void *)&timestep,
                (void *)&d_dynMem_const,
                (void *)&d_dz,
                (void *)&p,
                (void *)&d_merit_news,
                (void *)&d_merit_temp
            };
            gpuErrchk(hipLaunchCooperativeKernel(reinterpret_cast<const void*>(ls_merit_kernel), knot_points, MERIT_THREADS, kernelArgs, get_merit_smem_size<T>(state_size, knot_points), streams[p]));
        }
        if (sqpTimecheck()){ break; }
        gpuErrchk(hipPeekAtLastError());
        gpuErrchk(hipDeviceSynchronize());


        
        
        gpuErrchk(hipMemcpy(
            h_merit_news.data(),
            d_merit_news,
            h_merit_news.size() * sizeof(T),
            hipMemcpyDeviceToHost
        ));

    #if FINE_GRAINED_TIMING
        clock_gettime(CLOCK_MONOTONIC,&ls_end);
        double ls_time = time_delta_us_timespec(ls_start,ls_end);
        line_search_time_vec.push_back(ls_time);
    #endif // FINE_GRAINED_TIMING
    
        if (sqpTimecheck()){ break; }


        line_search_step = 0;
        min_merit = h_merit_initial;
        for(int i = 0; i < 8; i++){
        //     std::cout << h_merit_news[i] << (i == 7 ? "\n" : " ");
            ///TODO: reduction ratio
            if(h_merit_news[i] < min_merit){
                min_merit = h_merit_news[i];
                line_search_step = i;
            }
        }

        #if MPCGPU_TRACE_SQP
        std::cout << std::setprecision(9)
                << "TRACE_SQP iter=" << sqpiter
                << " rho=" << rho
                << " pcg_iters=" << pcg_iters
                << " pcg_exit=" << pcg_exit
                << " merit_initial=" << h_merit_initial
                << " merit_news=[";
        for (uint32_t mi = 0; mi < num_alphas; ++mi) {
            std::cout << h_merit_news[mi];
            if (mi + 1 < num_alphas) std::cout << ",";
        }
        std::cout << "] min_merit=" << min_merit
                << " line_step=" << line_search_step
                << " accepted=" << (min_merit != h_merit_initial)
                << std::endl;
        #endif


        if(min_merit == h_merit_initial){
            // line search failure
            drho = std::max<T>(drho * rho_factor, rho_factor);
            rho = std::max<T>(rho * drho, rho_min);
            sqp_iter++;
            if(rho > rho_max){
                sqp_time_exit = 0;
                rho = rho_reset;
                break; 
            }
            continue;
        }
        // std::cout << "line search accepted\n";
        alphafinal = -1.0 / (1 << line_search_step);        // alpha sign

        drho = std::min<T>(drho / rho_factor, static_cast<T>(1) / rho_factor);
        rho = std::max<T>(rho * drho, rho_min);
        

#if USE_DOUBLES
#ifdef __HIP_PLATFORM_NVIDIA__
        if (cublasDaxpy(handle, DZ_SIZE_BYTES / sizeof(T), &alphafinal, d_dz, 1, d_xu, 1) != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS Daxpy failed\n");
            exit(14);
        }
#else
        if (hipblasDaxpy(handle, DZ_SIZE_BYTES / sizeof(T), &alphafinal, d_dz, 1, d_xu, 1) != HIPBLAS_STATUS_SUCCESS) {
            printf("hipBLAS Daxpy failed\n");
            exit(14);
        }
#endif
#else
#ifdef __HIP_PLATFORM_NVIDIA__
        if (cublasSaxpy(handle, DZ_SIZE_BYTES / sizeof(T), &alphafinal, d_dz, 1, d_xu, 1) != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS Saxpy failed\n");
            exit(14);
        }
#else
        if (hipblasSaxpy(handle, DZ_SIZE_BYTES / sizeof(T), &alphafinal, d_dz, 1, d_xu, 1) != HIPBLAS_STATUS_SUCCESS) {
            printf("hipBLAS Saxpy failed\n");
            exit(14);
        }
#endif
#endif

        gpuErrchk(hipPeekAtLastError());
        // if success increment after update
        sqp_iter++;

        if (sqpTimecheck()){ break; }


        delta_merit_iter = h_merit_initial - min_merit;
        delta_merit_total += delta_merit_iter;
        

        h_merit_initial = min_merit;
    
    }
    
    gpuErrchk(hipPeekAtLastError());
    gpuErrchk(hipDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_end);

    double sqp_solve_time = time_delta_us_timespec(sqp_solve_start, sqp_solve_end);
    #if MPCGPU_TRACE_UPDATE_TIME
    std::cout << "SQP_RETURN"
            << " sqp_iter=" << sqp_iter
            << " sqp_us=" << sqp_solve_time
            << std::endl;
    #endif
    
#if !USE_SQP_WORKSPACE
    for(uint32_t str = 0; str < num_alphas; str++){
        hipStreamDestroy(streams[str]);
    }
    hipblasDestroy(handle);

    gpuErrchk(hipFree(d_G_dense));
    gpuErrchk(hipFree(d_C_dense));
    gpuErrchk(hipFree(d_g));
    gpuErrchk(hipFree(d_c));
    
    gpuErrchk(hipFree(d_S));
    gpuErrchk(hipFree(d_gamma));
    gpuErrchk(hipFree(d_dz));
    gpuErrchk(hipFree(d_xs));
    gpuErrchk(hipFree(d_merit_news));
    gpuErrchk(hipFree(d_merit_temp));
    gpuErrchk(hipFree(d_merit_initial));
    
    gpuErrchk(hipFree(d_Pinv));
    gpuErrchk(hipFree(d_r));
    gpuErrchk(hipFree(d_p));
    gpuErrchk(hipFree(d_v_temp));
    gpuErrchk(hipFree(d_eta_new_temp));
    
    gpuErrchk(hipFree(d_pcg_iters));
    gpuErrchk(hipFree(d_pcg_exit));
#endif

    return std::make_tuple(pcg_iter_vec, linsys_time_vec, sqp_solve_time, sqp_iter, sqp_time_exit, pcg_exit_vec, 
        ktt_time_vec, // new
        shur_time_vec,     
        dz_time_vec,           
        line_search_time_vec   );
}
