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
            std::cout << "FAIL: 024_merit_tracking_kernels" << std::endl;  \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

namespace cgrps = cooperative_groups;


namespace grid {
    template <typename T>
    struct robotModel {}; // Dummy-Struktur für den Pointer-Cast
}

namespace glass {
    template <typename T>
    __device__ void reduce(uint32_t size, T *arr) {}
}

namespace gato_plant {
    template <typename T>
    __device__ T trackingcost(uint32_t state_size, uint32_t control_size, uint32_t knot_points, 
                              T *s_xux_k, T *s_eePos_k_traj, T *s_temp, grid::robotModel<T> *d_robotModel) {
        return 0.0f;
    }

    constexpr size_t forwardDynamics_TempMemSize_Shared() {
        return 1024;
    }
}

template <typename T>
__device__ T integratorError(uint32_t state_size, T *s_xux_k, T *s_xux_kp1, T *s_temp, 
                             grid::robotModel<T> *d_robotModel, T dt, cgrps::thread_block block) {
    return 0.0f;
}

template <typename T>
__global__
void compute_tracking_error_kernel(T *d_tracking_error, uint32_t state_size, T *d_xu_goal, T *d_xs){
    
    T err;
    
    for(int ind = threadIdx.x; ind < state_size/2; ind += blockDim.x){
        err = abs(d_xs[ind] - d_xu_goal[ind]);
        atomicAdd(d_tracking_error, err);
    }
}


template <typename T>
size_t get_merit_smem_size(uint32_t state_size, uint32_t control_size)
{
    return sizeof(T) * (6 + (2 * state_size + control_size ) + 
                        ((int) 1.5 * state_size) + gato_plant::forwardDynamics_TempMemSize_Shared());
}

template <typename T>
__global__
void ls_gato_compute_merit(uint32_t state_size,
                           uint32_t control_size,
                           uint32_t knot_points,
                           T *d_xs,
                           T *d_xu, 
                           T *d_eePos_traj, 
                           T mu, 
                           T dt, 
                           void *d_dynMem_const, 
                           T *d_dz,
                           uint32_t alpha_multiplier, 
                           T *d_merits_out, 
                           T *d_merit_temp)
{

    grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *)d_dynMem_const;
    const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t num_blocks = gridDim.x;

    const uint32_t states_s_controls = state_size + control_size;

    extern __shared__ T s_xux_k[];

    T Jk, ck, pointmerit;

    T alpha = -1.0 / (1 << alpha_multiplier);   // alpha sign
    T *s_eePos_k_traj = s_xux_k + 2*state_size+control_size;
    T *s_temp = s_eePos_k_traj + 6;

    for(unsigned knot = block_id; knot < knot_points; knot += num_blocks){

        for(int i = thread_id; i < state_size+(knot < knot_points-1)*(states_s_controls); i+=num_threads){
            s_xux_k[i] = d_xu[knot*states_s_controls+i] + alpha * d_dz[knot*states_s_controls+i];  
            if (i < 6){
                s_eePos_k_traj[i] = d_eePos_traj[knot*6+i];                            
            }
        }
        block.sync();
        
        Jk = gato_plant::trackingcost<T>(state_size, control_size, knot_points, s_xux_k, s_eePos_k_traj, s_temp, d_robotModel);
        
        block.sync();
        if(knot < knot_points-1){
            ck = integratorError<T>(state_size, s_xux_k, &s_xux_k[states_s_controls], s_temp, d_robotModel, dt, block);
        }
        else{
            for(int i = threadIdx.x; i < state_size; i++){
                s_temp[i] = abs((d_xu[i] + alpha *d_dz[i]) - d_xs[i]);
            }
            block.sync();
            glass::reduce<T>(state_size, s_temp);
            block.sync();
            ck = s_temp[0];
        }
        block.sync();

        if(thread_id == 0){
            pointmerit = Jk + mu*ck;
            d_merit_temp[alpha_multiplier*knot_points+knot] = pointmerit;
        }
    }
    
    cooperative_groups::this_grid().sync();
    
    if(block_id == 0){
        glass::reduce<T>(knot_points, &d_merit_temp[alpha_multiplier*knot_points]);
    
        if(thread_id == 0){
            d_merits_out[alpha_multiplier] = d_merit_temp[alpha_multiplier*knot_points];
        }
    }
}

template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void compute_merit(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xu, T *d_eePos_traj, T mu, T dt, void *d_dynMem_const, T *d_merit_out)
{
    grid::robotModel<T> *d_robotModel = (grid::robotModel<T> *)d_dynMem_const;
    const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;

    const uint32_t states_s_controls = state_size + control_size;
    extern __shared__ T s_xux_k[];

    T Jk, ck, pointmerit;
    T *s_eePos_k_traj = s_xux_k + 2 * state_size + control_size;
    T *s_temp = s_eePos_k_traj + 6;

    for(unsigned knot = block_id; knot < knot_points; knot += gridDim.x){

        for(int i = thread_id; i < state_size+(knot < knot_points-1)*(states_s_controls); i+=num_threads){
            s_xux_k[i] = d_xu[knot*states_s_controls+i];  
            if (i < 6){
                s_eePos_k_traj[i] = d_eePos_traj[knot*6+i];                            
            }
        }

        block.sync();
        Jk = gato_plant::trackingcost<T>(state_size, control_size, knot_points, s_xux_k, s_eePos_k_traj, s_temp, d_robotModel);


        block.sync();
        if(knot < knot_points-1){
            ck = integratorError<T>(state_size, s_xux_k, &s_xux_k[states_s_controls], s_temp, d_robotModel, dt, block);
        }
        else{
            ck = 0;
        }
        block.sync();

        if(thread_id == 0){
            pointmerit = Jk + mu*ck;
            atomicAdd(d_merit_out, pointmerit);
        }
    }
}


int main() {
    uint32_t state_size = 12;
    uint32_t control_size = 6;
    uint32_t knot_points = 0;
    float dt = 0.01f;
    float mu = 1.0f;
    
    float *d_data;
    CHECK_GPU(hipMalloc(&d_data, 1024 * sizeof(float)));

    size_t smem_size = get_merit_smem_size<float>(state_size, control_size);

    // ls_gato_compute_merit (COOPERATIVE LAUNCH)
    
    uint32_t alpha_mult = 1;
    void* d_dyn_mem = nullptr;

    void* kernel_args[] = {
        (void*)&state_size,
        (void*)&control_size,
        (void*)&knot_points,
        (void*)&d_data,
        (void*)&d_data,
        (void*)&d_data,
        (void*)&mu,
        (void*)&dt,
        (void*)&d_dyn_mem,
        (void*)&d_data,
        (void*)&alpha_mult,
        (void*)&d_data,
        (void*)&d_data
    };

    const unsigned int shared_memory_bytes = smem_size; 
    hipStream_t stream = 0; 

    CHECK_GPU(hipLaunchCooperativeKernel( 
        (void*)ls_gato_compute_merit<float>, 
        dim3(1),               // blocks
        dim3(32),              // threads_per_block
        kernel_args, 
        shared_memory_bytes, 
        stream 
    ));
    CHECK_GPU(hipGetLastError());

    compute_merit<float><<<1, 32, smem_size>>>(
        state_size, control_size, knot_points, d_data, d_data, mu, dt, nullptr, d_data
    );
    CHECK_GPU(hipGetLastError());

    compute_tracking_error_kernel<float><<<1, 32>>>(d_data, state_size, d_data, d_data);
    CHECK_GPU(hipGetLastError());

    CHECK_GPU(hipDeviceSynchronize());
    CHECK_GPU(hipFree(d_data));

    std::cout << "OK: 024_merit_tracking_kernels" << std::endl;
    return 0;
}