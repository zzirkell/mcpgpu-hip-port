#include "hip/hip_runtime.h"
#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)
#include <cublas_v2.h>

using hipblasHandle_t = cublasHandle_t;
using hipblasStatus_t = cublasStatus_t;

#define HIPBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define hipblasCreate cublasCreate
#define hipblasDestroy cublasDestroy
#define hipblasSaxpy cublasSaxpy

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
#include "qdldl.h"
#include "qdldl/linsys_setup.hip.hpp"
#include "merit.hip.hpp"
#include "settings.hip.hpp"
#include "kkt.hip.hpp"
#include "dz.hip.hpp"

#ifndef MPCGPU_LS_NORMAL_REDUCE
#define MPCGPU_LS_NORMAL_REDUCE 0
#endif

#ifndef MPCGPU_LS_NORMAL_REDUCE_DIAG
#define MPCGPU_LS_NORMAL_REDUCE_DIAG 0
#endif

#ifndef MPCGPU_LS_STEP_DIAG
#define MPCGPU_LS_STEP_DIAG 0
#endif


__host__
void qdldl_solve_schur(const QDLDL_int An,
					   QDLDL_int *h_col_ptr, QDLDL_int *h_row_ind, QDLDL_float *Ax, QDLDL_float *b, 
					   QDLDL_float *h_lambda,
					   QDLDL_int *Lp, QDLDL_int *Li, QDLDL_float *Lx, QDLDL_float *D, QDLDL_float *Dinv, QDLDL_int *Lnz, QDLDL_int *etree, QDLDL_bool *bwork, QDLDL_int *iwork, QDLDL_float *fwork){

	



    QDLDL_int i;

	const QDLDL_int *Ap = h_col_ptr;
	const QDLDL_int *Ai = h_row_ind;

    //data for L and D factors
	QDLDL_int Ln = An;


	//Data for results of A\b
	QDLDL_float *x = h_lambda;

	QDLDL_factor(An,Ap,Ai,Ax,Lp,Li,Lx,D,Dinv,Lnz,etree,bwork,iwork,fwork);

	for(i=0;i < Ln; i++) x[i] = b[i];

	QDLDL_solve(Ln,Lp,Li,Lx,Dinv,x);
}


template <typename T>
auto sqpSolveQdldl(uint32_t state_size, uint32_t control_size, uint32_t knot_points, float timestep, T *d_eePos_traj, T *d_lambda, T *d_xu, void *d_dynMem_const, T &rho, T rho_reset){
    
    // data storage
    std::vector<int> linsys_iter_vec;
    std::vector<bool> linsys_exit_vec;
    std::vector<double> linsys_time_vec;
#if FINE_GRAINED_TIMING
    std::vector<double> ktt_time_vec;
    std::vector<double> shur_time_vec;
    std::vector<double> dz_time_vec;
    std::vector<double> line_search_time_vec;
#endif
    bool sqp_time_exit = 1;     // for data recording, not a flag
    


    // sqp timing
    struct timespec sqp_solve_start, sqp_solve_end;
    gpuErrchk(hipDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &sqp_solve_start);


    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    const uint32_t KKT_G_DENSE_SIZE_BYTES = static_cast<uint32_t>(((states_sq+controls_sq)*knot_points-controls_sq)*sizeof(T));
    const uint32_t KKT_C_DENSE_SIZE_BYTES = static_cast<uint32_t>((states_sq+states_p_controls)*(knot_points-1)*sizeof(T));
    const uint32_t KKT_g_SIZE_BYTES       = static_cast<uint32_t>(((state_size+control_size)*knot_points-control_size)*sizeof(T));
    const uint32_t KKT_c_SIZE_BYTES       =   static_cast<uint32_t>((state_size*knot_points)*sizeof(T));     
    const uint32_t DZ_SIZE_BYTES          =   static_cast<uint32_t>((states_s_controls*knot_points-control_size)*sizeof(T));


    // line search things
    const float mu = 10.0f;
    const uint32_t num_alphas = 8;

#if MPCGPU_LS_STEP_DIAG
    unsigned long long ls_step_hist[8] = {0,0,0,0,0,0,0,0};
    unsigned long long ls_step_fail_count = 0;
#endif

#if MPCGPU_LS_NORMAL_REDUCE_DIAG
    hipEvent_t ls_nr_ev0, ls_nr_ev1, ls_nr_ev2;
    gpuErrchk(hipEventCreate(&ls_nr_ev0));
    gpuErrchk(hipEventCreate(&ls_nr_ev1));
    gpuErrchk(hipEventCreate(&ls_nr_ev2));

    double ls_nr_partial_total_us = 0.0;
    double ls_nr_reduce_total_us = 0.0;
    double ls_nr_sync_total_us = 0.0;
    unsigned long long ls_nr_count = 0;
#endif
    T h_merit_news[num_alphas];
    void *ls_merit_kernel = (void *) ls_gato_compute_merit<T>;
    const size_t merit_smem_size = get_merit_smem_size<T>(state_size, control_size);
    T h_merit_initial, min_merit;
    T alphafinal;
    T delta_merit_iter = 0;
    T delta_merit_total = 0;
    uint32_t line_search_step = 0;


    // streams n cublas init
    hipStream_t streams[num_alphas];
    for(uint32_t str = 0; str < num_alphas; str++){
        gpuErrchk(hipStreamCreate(&streams[str]));
    }
    gpuErrchk(hipPeekAtLastError());

    hipblasHandle_t handle;
    if (hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed\n"); exit(13); }
    gpuErrchk(hipPeekAtLastError());


    uint32_t sqp_iter = 0;



    T *d_merit_initial, *d_merit_news, *d_merit_temp,
          *d_G_dense, *d_C_dense, *d_g, *d_c, *d_Ginv_dense,
          *d_S, *d_gamma,
          *d_dz,
          *d_xs;

    
    T drho = 1.0;
    T rho_factor = RHO_FACTOR;
    T rho_max = RHO_MAX;
    T rho_min = RHO_MIN;

    


    gpuErrchk(hipMalloc(&d_G_dense,  KKT_G_DENSE_SIZE_BYTES));
    gpuErrchk(hipMalloc(&d_C_dense,  KKT_C_DENSE_SIZE_BYTES));
    gpuErrchk(hipMalloc(&d_g,        KKT_g_SIZE_BYTES));
    gpuErrchk(hipMalloc(&d_c,        KKT_c_SIZE_BYTES));
    d_Ginv_dense = d_G_dense;

    gpuErrchk(hipMalloc(&d_S, 3*states_sq*knot_points*sizeof(T)));
    gpuErrchk(hipMalloc(&d_gamma, state_size*knot_points*sizeof(T)));
    gpuErrchk(hipPeekAtLastError());

    
    gpuErrchk(hipMalloc(&d_dz,       DZ_SIZE_BYTES));
    gpuErrchk(hipMalloc(&d_xs,       state_size*sizeof(T)));
    gpuErrchk(hipMemcpy(d_xs, d_xu,  state_size*sizeof(T), hipMemcpyDeviceToDevice));
    gpuErrchk(hipMalloc(&d_merit_news, 8*sizeof(T)));
    gpuErrchk(hipMalloc(&d_merit_temp, 8*knot_points*sizeof(T)));
    // linsys iterates

    gpuErrchk(hipMalloc(&d_merit_initial, sizeof(T)));
    gpuErrchk(hipMemset(d_merit_initial, 0, sizeof(T)));
    



    const int nnz = (knot_points-1)*states_sq + knot_points*(((state_size+1)*state_size)/2);
    
    const size_t qdldl_n = static_cast<size_t>(state_size) * static_cast<size_t>(knot_points);
    const size_t qdldl_nnz = static_cast<size_t>(nnz);

    std::vector<QDLDL_float> h_lambda_vec(qdldl_n);
    std::vector<QDLDL_float> h_gamma_vec(qdldl_n);
    std::vector<QDLDL_int> h_col_ptr_vec(qdldl_n + 1);
    std::vector<QDLDL_int> h_row_ind_vec(qdldl_nnz);
    std::vector<QDLDL_float> h_val_vec(qdldl_nnz);

    QDLDL_float* h_lambda = h_lambda_vec.data();
    QDLDL_float* h_gamma = h_gamma_vec.data();
    QDLDL_int* h_col_ptr = h_col_ptr_vec.data();
    QDLDL_int* h_row_ind = h_row_ind_vec.data();
    QDLDL_float* h_val = h_val_vec.data();
    
    QDLDL_int *d_row_ind, *d_col_ptr;
    QDLDL_float *d_val, *d_lambda_double;
    gpuErrchk(hipMalloc(&d_col_ptr, (state_size*knot_points+1)*sizeof(QDLDL_int)));
    gpuErrchk(hipMalloc(&d_row_ind, nnz*sizeof(QDLDL_int)));
	gpuErrchk(hipMalloc(&d_val, nnz*sizeof(QDLDL_float)));
	gpuErrchk(hipMalloc(&d_lambda_double, (state_size*knot_points)*sizeof(QDLDL_float)));
    
    // fill col ptr and row ind, these won't change 
    prep_csr<<<knot_points, 64>>>(state_size, knot_points, d_col_ptr, d_row_ind);
    gpuErrchk(hipMemcpy(h_col_ptr, d_col_ptr, (state_size*knot_points+1)*sizeof(QDLDL_int), hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_row_ind, d_row_ind, (nnz)*sizeof(QDLDL_int), hipMemcpyDeviceToHost));

    
    const QDLDL_int An = state_size*knot_points;

    // Q things
    QDLDL_int  sumLnz;
    QDLDL_int *etree;
	QDLDL_int *Lnz;
    etree = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
	Lnz   = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
    
    QDLDL_int *Lp;
	QDLDL_float *D;
	QDLDL_float *Dinv;
    Lp    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An+1));
	D     = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
	Dinv  = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    //working data for factorisation
	QDLDL_int   *iwork;
	QDLDL_bool  *bwork;
	QDLDL_float *fwork;
    iwork = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(3*An));
	bwork = (QDLDL_bool*)malloc(sizeof(QDLDL_bool)*An);
	fwork = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    sumLnz = QDLDL_etree(An,h_col_ptr,h_row_ind,iwork,Lnz,etree);
    
    QDLDL_int *Li;
	QDLDL_float *Lx;
    Li    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*sumLnz);
	Lx    = (QDLDL_float*)malloc(sizeof(QDLDL_float)*sumLnz);

    gpuErrchk(hipPeekAtLastError());
    gpuErrchk(hipDeviceSynchronize());
#if TIME_LINSYS == 1
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

    // gpuErrchk(hipDeviceSynchronize());
    // std::cout << "initial merit " << h_merit_initial << std::endl;
    // exit(0);

    //
    //      SQP LOOP
    //
    for(uint32_t sqpiter = 0; sqpiter < SQP_MAX_ITER; sqpiter++){
        
#if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        struct timespec ktt_start, ktt_end;
        clock_gettime(CLOCK_MONOTONIC, &ktt_start);
#endif
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
        clock_gettime(CLOCK_MONOTONIC, &ktt_end);
        ktt_time_vec.push_back(time_delta_us_timespec(ktt_start, ktt_end));
#endif
        if (sqpTimecheck()){ break; }


#if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        struct timespec shur_start, shur_end;
        clock_gettime(CLOCK_MONOTONIC, &shur_start);
#endif
        form_schur_system_qdldl<T>(state_size, control_size, knot_points, d_G_dense, d_C_dense, d_g, d_c, d_val, d_gamma, rho);
        gpuErrchk(hipPeekAtLastError());
#if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &shur_end);
        shur_time_vec.push_back(time_delta_us_timespec(shur_start, shur_end));
#endif
        if (sqpTimecheck()){ break; }

    #if TIME_LINSYS == 1
        gpuErrchk(hipDeviceSynchronize());
        if (sqpTimecheck()){ break; }
        clock_gettime(CLOCK_MONOTONIC, &linsys_start);
    #endif // #if TIME_LINSYS


        gpuErrchk(hipMemcpy(h_val, d_val, (nnz)*sizeof(T), hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(h_gamma, d_gamma, (state_size*knot_points)*sizeof(T), hipMemcpyDeviceToHost));

        qdldl_solve_schur(An, h_col_ptr, h_row_ind, h_val, h_gamma, h_lambda, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);
        
        gpuErrchk(hipMemcpy(d_lambda, h_lambda, (state_size*knot_points)*sizeof(T), hipMemcpyHostToDevice));


    #if TIME_LINSYS == 1
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &linsys_end);
        
        linsys_time = time_delta_us_timespec(linsys_start, linsys_end);
        linsys_time_vec.push_back(linsys_time);
    #endif // #if TIME_LINSYS
        
        if (sqpTimecheck()){ break; }
        
        // recover dz
#if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        struct timespec dz_start, dz_end;
        clock_gettime(CLOCK_MONOTONIC, &dz_start);
#endif
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
        clock_gettime(CLOCK_MONOTONIC, &dz_end);
        dz_time_vec.push_back(time_delta_us_timespec(dz_start, dz_end));
#endif
        if (sqpTimecheck()){ break; }
        

        // line search
#if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        struct timespec line_search_start, line_search_end;
        clock_gettime(CLOCK_MONOTONIC, &line_search_start);
#endif
#if MPCGPU_LS_NORMAL_REDUCE
        {
            dim3 partial_grid(knot_points, num_alphas);
            dim3 partial_block(MERIT_THREADS);

#if MPCGPU_LS_NORMAL_REDUCE_DIAG
            gpuErrchk(hipEventRecord(ls_nr_ev0));
#endif

            ls_gato_compute_merit_partials<T><<<
                partial_grid,
                partial_block,
                get_merit_smem_size<T>(state_size, control_size)
            >>>(
                state_size,
                control_size,
                knot_points,
                d_xs,
                d_xu,
                d_eePos_traj,
                mu,
                timestep,
                d_dynMem_const,
                d_dz,
                d_merit_temp
            );
            gpuErrchk(hipPeekAtLastError());

#if MPCGPU_LS_NORMAL_REDUCE_DIAG
            gpuErrchk(hipEventRecord(ls_nr_ev1));
#endif

            ls_gato_reduce_merit_partials<T><<<
                num_alphas,
                MERIT_THREADS,
                MERIT_THREADS * sizeof(T)
            >>>(
                knot_points,
                d_merit_temp,
                d_merit_news
            );
            gpuErrchk(hipPeekAtLastError());

#if MPCGPU_LS_NORMAL_REDUCE_DIAG
            gpuErrchk(hipEventRecord(ls_nr_ev2));
#endif
        }
#else
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
#endif
        if (sqpTimecheck()){ break; }
        gpuErrchk(hipPeekAtLastError());
#if MPCGPU_LS_NORMAL_REDUCE_DIAG
        struct timespec ls_nr_sync_start, ls_nr_sync_end;
        clock_gettime(CLOCK_MONOTONIC, &ls_nr_sync_start);
#endif

        gpuErrchk(hipDeviceSynchronize());

#if MPCGPU_LS_NORMAL_REDUCE_DIAG
        clock_gettime(CLOCK_MONOTONIC, &ls_nr_sync_end);
        ls_nr_sync_total_us += time_delta_us_timespec(ls_nr_sync_start, ls_nr_sync_end);

        float ls_nr_partial_ms = 0.0f;
        float ls_nr_reduce_ms = 0.0f;
        gpuErrchk(hipEventElapsedTime(&ls_nr_partial_ms, ls_nr_ev0, ls_nr_ev1));
        gpuErrchk(hipEventElapsedTime(&ls_nr_reduce_ms, ls_nr_ev1, ls_nr_ev2));

        ls_nr_partial_total_us += static_cast<double>(ls_nr_partial_ms) * 1000.0;
        ls_nr_reduce_total_us += static_cast<double>(ls_nr_reduce_ms) * 1000.0;
        ls_nr_count++;
#endif
        
        
        gpuErrchk(hipMemcpy(h_merit_news, d_merit_news, 8*sizeof(T), hipMemcpyDeviceToHost));
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


        if(min_merit == h_merit_initial){
#if MPCGPU_LS_STEP_DIAG
            ls_step_fail_count++;
#endif
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
#if MPCGPU_LS_STEP_DIAG
        if (line_search_step < 8) {
            ls_step_hist[line_search_step]++;
        }
#endif

        alphafinal = static_cast<T>(-1) / static_cast<T>(1u << line_search_step);        // alpha sign

        drho = std::min<T>(drho / rho_factor, static_cast<T>(1) / rho_factor);
        rho = std::max<T>(rho * drho, rho_min);
        

#if USE_DOUBLES
        hipblasDaxpy(
            handle, 
            DZ_SIZE_BYTES / sizeof(T),
            &alphafinal,
            d_dz, 1,
            d_xu, 1
        );
#else
        hipblasSaxpy(
            handle, 
            DZ_SIZE_BYTES / sizeof(T),
            &alphafinal,
            d_dz, 1,
            d_xu, 1
        );
#endif

        gpuErrchk(hipPeekAtLastError());
#if FINE_GRAINED_TIMING
        gpuErrchk(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &line_search_end);
        line_search_time_vec.push_back(time_delta_us_timespec(line_search_start, line_search_end));
#endif
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

    hipblasDestroy(handle);

    for(uint32_t st=0; st < num_alphas; st++){
        gpuErrchk(hipStreamDestroy(streams[st]));
    }




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
    gpuErrchk(hipFree(d_col_ptr));
    gpuErrchk(hipFree(d_row_ind));
    gpuErrchk(hipFree(d_val));
    gpuErrchk(hipFree(d_lambda_double));
	free(etree);
	free(Lnz);
    free(Lp);
	free(D);
	free(Dinv);
	free(iwork);
	free(bwork);
	free(fwork);
	free(Li);
	free(Lx);

    double sqp_solve_time = time_delta_us_timespec(sqp_solve_start, sqp_solve_end);

#if FINE_GRAINED_TIMING

#if MPCGPU_LS_NORMAL_REDUCE_DIAG
    if (ls_nr_count > 0) {
        printf("LS_NR_KERNEL_DIAG solver=QDLDL K=%u count=%llu "
               "partial_mean_us=%.3f reduce_mean_us=%.3f sync_mean_us=%.3f\n",
               knot_points,
               ls_nr_count,
               ls_nr_partial_total_us / (double)ls_nr_count,
               ls_nr_reduce_total_us / (double)ls_nr_count,
               ls_nr_sync_total_us / (double)ls_nr_count);
    }

    gpuErrchk(hipEventDestroy(ls_nr_ev0));
    gpuErrchk(hipEventDestroy(ls_nr_ev1));
    gpuErrchk(hipEventDestroy(ls_nr_ev2));
#endif


#if MPCGPU_LS_STEP_DIAG
    unsigned long long ls_step_accept_count = 0;
    for (int i = 0; i < 8; i++) {
        ls_step_accept_count += ls_step_hist[i];
    }

    printf("LS_STEP_DIAG solver=QDLDL K=%u accepts=%llu fails=%llu "
           "step0=%llu step1=%llu step2=%llu step3=%llu "
           "step4=%llu step5=%llu step6=%llu step7=%llu\n",
           knot_points,
           ls_step_accept_count,
           ls_step_fail_count,
           ls_step_hist[0],
           ls_step_hist[1],
           ls_step_hist[2],
           ls_step_hist[3],
           ls_step_hist[4],
           ls_step_hist[5],
           ls_step_hist[6],
           ls_step_hist[7]);
#endif

    return std::make_tuple(
        linsys_iter_vec,
        linsys_time_vec,
        sqp_solve_time,
        sqp_iter,
        sqp_time_exit,
        linsys_exit_vec,
        ktt_time_vec,
        shur_time_vec,
        dz_time_vec,
        line_search_time_vec
    );
#else
    return std::make_tuple(
        linsys_iter_vec,
        linsys_time_vec,
        sqp_solve_time,
        sqp_iter,
        sqp_time_exit,
        linsys_exit_vec
    );
#endif
}
