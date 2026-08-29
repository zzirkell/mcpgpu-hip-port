#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include "gpuassert.hip.hpp"

#ifndef MPCGPU_PCG_K32_FAST_REDUCE
#define MPCGPU_PCG_K32_FAST_REDUCE 1
#endif

#ifndef MPCGPU_PCG_K32_BLOCK_THREADS
#define MPCGPU_PCG_K32_BLOCK_THREADS 448
#endif

#ifndef MPCGPU_PCG_K32_PAD16
#define MPCGPU_PCG_K32_PAD16 0
#endif

template <typename T>
__device__ inline T pcg_k32_abs_dev(T x)
{
    return x < T(0) ? -x : x;
}

template <typename T, uint32_t state_size, uint32_t knot_points>
__device__ inline T pcg_k32_bdmv_global(
    const T *A,
    const T *x,
    uint32_t knot,
    uint32_t row)
{
    constexpr uint32_t states_sq = state_size * state_size;
    const T *Ak = A + knot * 3 * states_sq;

    T sum = T(0);

    if (knot > 0) {
        const T *Aprev = Ak;
        const T *xprev = x + (knot - 1) * state_size;
        for (uint32_t col = 0; col < state_size; ++col) {
            sum += Aprev[col * state_size + row] * xprev[col];
        }
    }

    {
        const T *Adiag = Ak + states_sq;
        const T *xself = x + knot * state_size;
        for (uint32_t col = 0; col < state_size; ++col) {
            sum += Adiag[col * state_size + row] * xself[col];
        }
    }

    if (knot + 1 < knot_points) {
        const T *Anext = Ak + 2 * states_sq;
        const T *xnext = x + (knot + 1) * state_size;
        for (uint32_t col = 0; col < state_size; ++col) {
            sum += Anext[col * state_size + row] * xnext[col];
        }
    }

    return sum;
}

template <typename T>
__device__ inline T pcg_k32_wave_reduce_sum(T v)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset);
    }
    return v;
}

template <typename T>
__device__ inline T pcg_k32_block_reduce_sum_slow(T local, T *s_red)
{
    const uint32_t tid = threadIdx.x;
    s_red[tid] = local;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_red[tid] += s_red[tid + stride];
        }
        __syncthreads();
    }

    return s_red[0];
}

template <typename T>
__device__ inline T pcg_k32_block_reduce_sum_fast(T local, T *s_red)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % warpSize;
    const uint32_t wave = tid / warpSize;
    const uint32_t num_waves = (blockDim.x + warpSize - 1) / warpSize;

    T wave_sum = pcg_k32_wave_reduce_sum<T>(local);

    if (lane == 0) {
        s_red[wave] = wave_sum;
    }
    __syncthreads();

    T block_sum = T(0);
    if (wave == 0) {
        block_sum = lane < num_waves ? s_red[lane] : T(0);
        block_sum = pcg_k32_wave_reduce_sum<T>(block_sum);

        if (lane == 0) {
            s_red[0] = block_sum;
        }
    }
    __syncthreads();

    return s_red[0];
}

template <typename T>
__device__ inline T pcg_k32_block_reduce_sum(T local, T *s_red)
{
#if MPCGPU_PCG_K32_FAST_REDUCE
    return pcg_k32_block_reduce_sum_fast<T>(local, s_red);
#else
    return pcg_k32_block_reduce_sum_slow<T>(local, s_red);
#endif
}

template <typename T, uint32_t state_size, uint32_t knot_points>
__global__ void pcg_k32_singleblock_kernel(
    T *d_S,
    T *d_Pinv,
    T *d_gamma,
    T *d_lambda,
    T *d_r,
    T *d_p,
    T *d_v_temp,
    T *d_eta_new_temp,
    uint32_t *d_iters,
    bool *d_max_iter_exit,
    uint32_t max_iter,
    T exit_tol)
{
    constexpr uint32_t N = state_size * knot_points;
    const uint32_t tid = threadIdx.x;

#if MPCGPU_PCG_K32_PAD16
    const uint32_t knot_for_tid = tid >> 4;
    const uint32_t row_for_tid = tid & 15u;
    const uint32_t idx_for_tid = knot_for_tid * state_size + row_for_tid;
    const bool active_tid = knot_for_tid < knot_points && row_for_tid < state_size;
#else
    const uint32_t idx_for_tid = tid;
    const uint32_t knot_for_tid = tid / state_size;
    const uint32_t row_for_tid = tid % state_size;
    const bool active_tid = tid < N;
#endif

    extern __shared__ unsigned char raw_smem[];
    T *s_lambda = reinterpret_cast<T *>(raw_smem);
    T *s_r      = s_lambda + N;
    T *s_rt     = s_r + N;
    T *s_p      = s_rt + N;
    T *s_sp     = s_p + N;
    T *s_red    = s_sp + N;

    if (active_tid) {
        s_lambda[idx_for_tid] = d_lambda[idx_for_tid];
    }
    __syncthreads();

    if (active_tid) {
        const T Slambda = pcg_k32_bdmv_global<T, state_size, knot_points>(
            d_S, s_lambda, knot_for_tid, row_for_tid);

        s_r[idx_for_tid] = d_gamma[idx_for_tid] - Slambda;
    }
    __syncthreads();

    if (active_tid) {
        s_rt[idx_for_tid] = pcg_k32_bdmv_global<T, state_size, knot_points>(
            d_Pinv, s_r, knot_for_tid, row_for_tid);

        s_p[idx_for_tid] = s_rt[idx_for_tid];
    }
    __syncthreads();

    T local_eta = T(0);
    if (active_tid) {
        local_eta = s_r[idx_for_tid] * s_rt[idx_for_tid];
    }

    T eta = pcg_k32_block_reduce_sum<T>(local_eta, s_red);

    uint32_t iter = 0;
    bool max_iter_exit = true;

    for (iter = 0; iter < max_iter; ++iter) {
        if (active_tid) {
            s_sp[idx_for_tid] = pcg_k32_bdmv_global<T, state_size, knot_points>(
                d_S, s_p, knot_for_tid, row_for_tid);
        }
        __syncthreads();

        T local_denom = T(0);
        if (active_tid) {
            local_denom = s_p[idx_for_tid] * s_sp[idx_for_tid];
        }

        T denom = pcg_k32_block_reduce_sum<T>(local_denom, s_red);

        if (pcg_k32_abs_dev(denom) < T(1e-20)) {
            break;
        }

        const T alpha = eta / denom;

        if (active_tid) {
            s_lambda[idx_for_tid] += alpha * s_p[idx_for_tid];
            s_r[idx_for_tid]      -= alpha * s_sp[idx_for_tid];
        }
        __syncthreads();

        if (active_tid) {
            s_rt[idx_for_tid] = pcg_k32_bdmv_global<T, state_size, knot_points>(
                d_Pinv, s_r, knot_for_tid, row_for_tid);
        }
        __syncthreads();

        T local_eta_new = T(0);
        if (active_tid) {
            local_eta_new = s_r[idx_for_tid] * s_rt[idx_for_tid];
        }

        T eta_new = pcg_k32_block_reduce_sum<T>(local_eta_new, s_red);

        if (pcg_k32_abs_dev(eta_new) < exit_tol) {
            ++iter;
            max_iter_exit = false;
            eta = eta_new;
            break;
        }

        const T beta = eta_new / eta;
        eta = eta_new;

        if (active_tid) {
            s_p[idx_for_tid] = s_rt[idx_for_tid] + beta * s_p[idx_for_tid];
        }
        __syncthreads();
    }

    if (active_tid) {
        d_lambda[idx_for_tid] = s_lambda[idx_for_tid];
        d_r[idx_for_tid] = s_r[idx_for_tid];
        d_p[idx_for_tid] = s_p[idx_for_tid];
    }

    if (tid == 0) {
        d_iters[0] = iter;
        d_max_iter_exit[0] = max_iter_exit;
    }
}

template <typename T>
inline void launchPCGK32SingleBlock(
    uint32_t state_size,
    uint32_t knot_points,
    T *d_S,
    T *d_Pinv,
    T *d_gamma,
    T *d_lambda,
    T *d_r,
    T *d_p,
    T *d_v_temp,
    T *d_eta_new_temp,
    uint32_t *d_pcg_iters,
    bool *d_pcg_exit,
    uint32_t max_iter,
    T exit_tol)
{
    if (state_size != 14 || knot_points != 32) {
        fprintf(stderr, "[PCG_K32_SINGLEBLOCK] only supports state_size=14, knot_points=32\n");
        std::abort();
    }

    constexpr uint32_t state_size_c = 14;
    constexpr uint32_t knot_points_c = 32;
    constexpr uint32_t N = state_size_c * knot_points_c;

#if MPCGPU_PCG_K32_PAD16
    constexpr uint32_t block_threads = 512;
#else
    constexpr uint32_t block_threads = MPCGPU_PCG_K32_BLOCK_THREADS;
#endif

    const size_t smem = sizeof(T) * (5 * N + block_threads);

    hipLaunchKernelGGL(
        (pcg_k32_singleblock_kernel<T, state_size_c, knot_points_c>),
        dim3(1),
        dim3(block_threads),
        smem,
        0,
        d_S,
        d_Pinv,
        d_gamma,
        d_lambda,
        d_r,
        d_p,
        d_v_temp,
        d_eta_new_temp,
        d_pcg_iters,
        d_pcg_exit,
        max_iter,
        exit_tol);

    gpuErrchk(hipPeekAtLastError());
}
