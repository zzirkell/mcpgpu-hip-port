#include "hip/hip_runtime.h"
#include <iostream>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                            \
        if (err != hipSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, hipGetErrorString(err));     \
            std::cout << "FAIL: 022_kkt_dz_common_kernels" << std::endl;   \
            return 1;                                                      \
        }                                                                  \
    } while (0)


namespace cgrps = cooperative_groups;

namespace glass {
    template<typename T> __device__ void copy(size_t n, const T* src, T* dst) {}
    template<typename T> __device__ void copy(size_t n, T val, const T* src, T* dst) {}
}

namespace gato_plant {
    template<typename T> __device__ void trackingCostGradientAndHessian_lastblock(
        uint32_t state_size, uint32_t control_size, T* s_xux, T* s_eePos_traj, T* s_Qk, T* s_qk, T* s_Rk, T* s_rk, T* s_Qkp1, T* s_qkp1, T* s_extra_temp, void* d_dynMem_const) {}
    
    template<typename T> __device__ void trackingCostGradientAndHessian(
        uint32_t state_size, uint32_t control_size, T* s_xux, T* s_eePos_traj, T* s_Qk, T* s_qk, T* s_Rk, T* s_rk, T* s_extra_temp, void* d_dynMem_const) {}
}

template<typename T, unsigned I, bool A, bool B> __device__ void integratorAndGradient(
    uint32_t state_size, uint32_t control_size, T* s_xux, T* s_Ak, T* s_Bk, T* s_integrator_error, T* s_extra_temp, void* d_dynMem_const, T timestep, cgrps::thread_block block) {}

template<typename T> __device__ void gato_ATx(T* out, T* A, T* x, int rows, int cols) {}
template<typename T> __device__ void gato_vec_dif(T* out, T* a, T* b, int len) {}
template<typename T> __device__ void mat_vec_prod(int rows, int cols, T* A, T* x, T* out) {}
template<typename T> __device__ void gato_vec_sum(T* out, T* a, T* b, int len) {}

#define DZ_THREADS 256

// ==========================================
// 2. DEIN ORIGINAL-CODE
// ==========================================

template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void generate_kkt_submatrices(uint32_t state_size, 
                              uint32_t control_size, 
                              uint32_t knot_points,
                              T *d_G_dense, 
                              T *d_C_dense, 
                              T *d_g, 
                              T *d_c,
                              void *d_dynMem_const, 
                              T timestep,
                              T *d_eePos_traj, 
                              T *d_xs, 
                              T *d_xu)
{

    const cgrps::thread_block block = cgrps::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t num_blocks = gridDim.x;

    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    

    extern __shared__ T s_temp[];

    T *s_xux = s_temp;
    T *s_eePos_traj = s_xux + 2*state_size + control_size;
    T *s_Qk = s_eePos_traj + 6;
    T *s_Rk = s_Qk + states_sq;
    T *s_qk = s_Rk + controls_sq;
    T *s_rk = s_qk + state_size;
    T *s_end = s_rk + control_size;

    
    for(unsigned k = block_id; k < knot_points-1; k += num_blocks){

        glass::copy<T>(2*state_size + control_size, &d_xu[k*states_s_controls], s_xux);
        glass::copy<T>(2 * 6, &d_eePos_traj[k*6], s_eePos_traj);
        
        __syncthreads();    

        if(k==knot_points-2){          // last block

            T *s_Ak = s_end;
            T *s_Bk = s_Ak + states_sq;
            T *s_Qkp1 = s_Bk + states_p_controls;
            T *s_qkp1 = s_Qkp1 + states_sq;
            T *s_integrator_error = s_qkp1 + state_size;
            T *s_extra_temp = s_integrator_error + state_size;
            
            integratorAndGradient<T, INTEGRATOR_TYPE, ANGLE_WRAP, true>(
                state_size, control_size,
                s_xux,
                s_Ak,
                s_Bk,
                s_integrator_error,
                s_extra_temp,
                d_dynMem_const,
                timestep,
                block
            );
            __syncthreads();
            
            gato_plant::trackingCostGradientAndHessian_lastblock<T>(
                state_size,
                control_size,
                s_xux,
                s_eePos_traj,
                s_Qk,
                s_qk,
                s_Rk,
                s_rk,
                s_Qkp1,
                s_qkp1,
                s_extra_temp,
                d_dynMem_const
            );
            __syncthreads();

            for(int i = thread_id; i < state_size; i+=num_threads){
                d_c[i] = d_xu[i] - d_xs[i];
            }
            glass::copy<T>(states_sq, s_Qk, &d_G_dense[(states_sq+controls_sq)*k]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense[(states_sq+controls_sq)*k+states_sq]);
            glass::copy<T>(states_sq, s_Qkp1, &d_G_dense[(states_sq+controls_sq)*(k+1)]);
            glass::copy<T>(state_size, s_qk, &d_g[states_s_controls*k]);
            glass::copy<T>(control_size, s_rk, &d_g[states_s_controls*k+state_size]);
            glass::copy<T>(state_size, s_qkp1, &d_g[states_s_controls*(k+1)]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense[(states_sq+states_p_controls)*k]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense[(states_sq+states_p_controls)*k+states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c[state_size*(k+1)]);

        }
        else{                               // not last knot

            T *s_Ak = s_end;
            T *s_Bk = s_Ak + states_sq;
            T *s_integrator_error = s_Bk + states_p_controls;
            T *s_extra_temp = s_integrator_error + state_size;

            integratorAndGradient<T, 
                                  INTEGRATOR_TYPE, 
                                  ANGLE_WRAP, 
                                  true>
                                 (state_size, control_size,
                                  s_xux,
                                  s_Ak,
                                  s_Bk,
                                  s_integrator_error,
                                  s_extra_temp,
                                  d_dynMem_const,
                                  timestep,
                                  block);
            __syncthreads();
           
            gato_plant::trackingCostGradientAndHessian<T>(state_size,
                                                  control_size,
                                                  s_xux,
                                                  s_eePos_traj,
                                                  s_Qk,
                                                  s_qk,
                                                  s_Rk,
                                                  s_rk,
                                                  s_extra_temp,
                                                  d_dynMem_const);
            __syncthreads();
 
            glass::copy<T>(states_sq, s_Qk, &d_G_dense[(states_sq+controls_sq)*k]);
            glass::copy<T>(controls_sq, s_Rk, &d_G_dense[(states_sq+controls_sq)*k+states_sq]);
            glass::copy<T>(state_size, s_qk, &d_g[states_s_controls*k]);
            glass::copy<T>(control_size, s_rk, &d_g[states_s_controls*k+state_size]);
            glass::copy<T>(states_sq, static_cast<T>(-1), s_Ak, &d_C_dense[(states_sq+states_p_controls)*k]);
            glass::copy<T>(states_p_controls, static_cast<T>(-1), s_Bk, &d_C_dense[(states_sq+states_p_controls)*k+states_sq]);
            glass::copy<T>(state_size, s_integrator_error, &d_c[state_size*(k+1)]);
        }
    }
}

template <typename T>
__global__
void compute_dz_kernel(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_G_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz){

    extern __shared__ T s_mem[]; 
    
    const uint32_t states_sq = state_size*state_size;
    const uint32_t states_p_controls = state_size * control_size;
    const uint32_t controls_sq = control_size * control_size;
    const uint32_t states_s_controls = state_size + control_size;
    unsigned set;

    for(int blockrow = blockIdx.x; blockrow < 2*knot_points-1; blockrow+=gridDim.x){

        set = blockrow/2;
        
        if(blockrow%2){ // control row
            // shared mem config
            //    Rkinv |   BkT
            //      C^2  |  S*C

            T *s_Rk_i = s_mem;
            T *s_BkT = s_Rk_i + controls_sq;
            T *s_scratch = s_BkT + states_p_controls;

            // load Rkinv from G
            glass::copy<T>(controls_sq, d_G_dense+set*(states_sq+controls_sq)+states_sq, s_Rk_i);

            // load Bk from C
            glass::copy<T>(states_p_controls, d_C_dense+set*(states_sq+states_p_controls)+states_sq, s_BkT);

            __syncthreads();

            // // compute BkT*lkp1
            gato_ATx<T>(s_scratch,
                    s_BkT,
                    d_lambda+(set+1)*state_size,
                    state_size,
                    control_size);
            __syncthreads();

            // subtract from rk
            gato_vec_dif(s_scratch,
                        d_g_val+set*(states_s_controls)+state_size,
                        s_scratch,
                        control_size);
            __syncthreads();

            // multiply Rk_i*scratch in scratch + C
            mat_vec_prod<T>( control_size, control_size,s_Rk_i,
                                                            s_scratch,
                                                            s_scratch+control_size);
            __syncthreads();
            
            // store in d_dz
            glass::copy<T>(control_size, s_scratch+control_size, d_dz+set*(states_s_controls)+state_size);

        }
        else{   // state row

            T *s_Qk_i = s_mem;
            T *s_AkT = s_Qk_i + states_sq;
            T *s_scratch = s_AkT + states_sq;
            
            // shared mem config
            //    Qkinv |  AkT | scratch
            //      S^2     S^2

            /// TODO: error check
            // load Qkinv from G
            glass::copy<T>(states_sq, d_G_dense+set*(states_sq+controls_sq), s_Qk_i);

                        ///TODO: linsys solver hasn't been checked with this change
            if(set != knot_points-1){
                // load Ak from C
                glass::copy<T>(states_sq, d_C_dense+set*(states_sq+states_p_controls), s_AkT);
                __syncthreads();
                            
                // // compute AkT*lkp1 in scratch
                gato_ATx(s_scratch,
                        s_AkT,
                        d_lambda+(set+1)*state_size,
                        state_size,
                        state_size);
                __syncthreads();
            }
            else{
                for(int i = threadIdx.x; i < state_size; i+=blockDim.x){
                    s_scratch[i] = 0;
                }
            }
            

            // add lk to scratch
            gato_vec_sum<T>(s_scratch,     // out
                        d_lambda+set*state_size,
                        s_scratch,
                        state_size);
            __syncthreads();

            // subtract from qk in scratch
            gato_vec_dif<T>(s_scratch,
                        d_g_val+set*(states_s_controls),
                        s_scratch,
                        state_size);
            __syncthreads();
            
            
            // multiply Qk_i(scratch) in Akt
            mat_vec_prod<T>( state_size, state_size,s_Qk_i,
                                                        s_scratch,
                                                        s_AkT);
            __syncthreads();

            // store in dz
            glass::copy<T>(state_size, s_AkT, d_dz+set*(states_s_controls));
        }
    }
}


template <typename T>
void compute_dz(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_G_dense, T *d_C_dense, T *d_g_val, T *d_lambda, T *d_dz){
    
    compute_dz_kernel<<<knot_points, DZ_THREADS, sizeof(T)*(2*state_size*state_size+state_size)>>>(
        state_size, 
        control_size, 
        knot_points, 
        d_G_dense, 
        d_C_dense, 
        d_g_val, 
        d_lambda, 
        d_dz
    );
}


int main() {
    uint32_t state_size = 12;
    uint32_t control_size = 6;
    uint32_t knot_points = 2;
    
    float *d_dummy;
    CHECK_GPU(hipMalloc(&d_dummy, 1024 * sizeof(float)));

    size_t smem_size = 1024 * sizeof(float); 
    
    generate_kkt_submatrices<float, 0, false><<<1, 32, smem_size>>>(
        state_size, control_size, knot_points, d_dummy, d_dummy, d_dummy, d_dummy, nullptr, 0.01f, d_dummy, d_dummy, d_dummy
    );
    CHECK_GPU(hipGetLastError());


    compute_dz_kernel<float><<<1, 32, smem_size>>>(
        state_size, control_size, knot_points, d_dummy, d_dummy, d_dummy, d_dummy, d_dummy
    );
    CHECK_GPU(hipGetLastError());

    CHECK_GPU(hipDeviceSynchronize());
    CHECK_GPU(hipFree(d_dummy));

    std::cout << "OK: 022_kkt_dz_common_kernels" << std::endl;
    return 0;
}