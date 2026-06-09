#include <iostream>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err));     \
            std::cout << "FAIL: 023_integrator_kernels" << std::endl;      \
            exit(1);                                                       \
        }                                                                  \
    } while (0)


#define gpuErrchk CHECK_GPU 
namespace cgrps = cooperative_groups;



namespace glass {
    template <typename T>
    __device__ void reduce(uint32_t size, T *arr) {
        // Dummy: Macht nichts im Test
    }
}

namespace gato_plant {
    template <typename T>
    __device__ void forwardDynamicsAndGradient(T *dqdd, T *qdd, T *q, T *qd, T *u, T *extra_temp, void *d_dynMem_const) {
        // Dummy
    }

    template <typename T>
    __device__ void forwardDynamics(T *qdd, T *q, T *qd, T *u, T *extra_temp, void *d_dynMem_const, cgrps::thread_block block) {
        // Dummy
    }

    constexpr size_t forwardDynamicsAndGradient_TempMemSize_Shared() {
        return 1024; 
    }
}


template<typename T>
__host__ __device__ 
T angleWrap(T input){
    const T pi = static_cast<T>(3.14159);
    if(input > pi){input = -(input - pi);}
    if(input < -pi){input = -(input + pi);}
    return input;
}


template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator_error(uint32_t state_size, T *s_err, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt, cgrps::thread_block block, bool absval = false){
    T new_qkp1; T new_qdkp1;
    for (unsigned ind = threadIdx.x; ind < state_size/2; ind += blockDim.x){
        // euler xk = xk + dt *dxk
        if (INTEGRATOR_TYPE == 0){
            new_qkp1 = s_q[ind] + dt*s_qd[ind];
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
        }
        // semi-inplicit euler
        else if (INTEGRATOR_TYPE == 1){
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
            new_qkp1 = s_q[ind] + dt*new_qdkp1;
        }
        else {printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}

        // wrap angles if needed
        if(ANGLE_WRAP){ printf("ANGLE_WRAP!\n");
            new_qkp1 = angleWrap(new_qkp1);
        }

        // then computre error
        if(absval){
            s_err[ind] = abs(s_qkp1[ind] - new_qkp1);
            s_err[ind + state_size/2] = abs(s_qdkp1[ind] - new_qdkp1);    
        }
        else{
            s_err[ind] = s_qkp1[ind] - new_qkp1;
            s_err[ind + state_size/2] = s_qdkp1[ind] - new_qdkp1;
        }
    }
}

template <typename T, unsigned INTEGRATOR_TYPE = 0>
__device__
void exec_integrator_gradient(uint32_t state_size, uint32_t control_size, T *s_Ak, T *s_Bk, T *s_dqdd, T dt, cgrps::thread_block block){
        
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_dim = blockDim.x;

    if (INTEGRATOR_TYPE == 0){
        for (unsigned ind = thread_id; ind < state_size*(state_size + control_size); ind += block_dim){
            int c = ind / state_size; int r = ind % state_size;
            T *dst = (c < state_size)? &s_Ak[ind] : &s_Bk[ind - state_size*state_size]; // dst
            T val = (r == c) * static_cast<T>(1); // first term (non-branching)
            val += (r < state_size/2 && r == c - state_size/2) * dt; // first dxd term (non-branching)
            if(r >= state_size/2) { val += dt * s_dqdd[c*state_size/2 + r - state_size/2]; }
            *dst = val;
        }
    }
    else if (INTEGRATOR_TYPE == 1){
        for (unsigned ind = thread_id; ind < state_size*state_size; ind += block_dim){
            int c = ind / state_size; int r = ind % state_size; int rdqdd = r % (state_size/2);
            T dtVal = static_cast<T>((r == rdqdd)*dt + (r != rdqdd));
            s_Ak[ind] = static_cast<T>((r == c) + dt*(r == c - state_size/2)) +
                        dt * s_dqdd[c*state_size/2 + rdqdd] * dtVal;
            if(c < control_size){
                s_Bk[ind] = dt * s_dqdd[state_size*state_size/2 + c*state_size/2 + rdqdd] * dtVal;
            }
        }
    }
    else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}
}


template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator(uint32_t state_size, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt, cgrps::thread_block block){

    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_dim = blockDim.x;

    for (unsigned ind = thread_id; ind < state_size/2; ind += block_dim){
        if (INTEGRATOR_TYPE == 0){
            s_qkp1[ind] = s_q[ind] + dt*s_qd[ind];
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
        }
        else if (INTEGRATOR_TYPE == 1){
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
            s_qkp1[ind] = s_q[ind] + dt*s_qdkp1[ind];
        }
        else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}

        if(ANGLE_WRAP){
            s_qkp1[ind] = angleWrap(s_qkp1[ind]);
        }
    }
}

template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false, bool COMPUTE_INTEGRATOR_ERROR = false>
__device__ __forceinline__
void integratorAndGradient(uint32_t state_size, uint32_t control_size, T *s_xux, T *s_Ak, T *s_Bk, T *s_xnew_err, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){
    
    T *s_qdd = s_temp; 	
    T *s_dqdd = s_qdd + state_size/2;	
    T *s_extra_temp = s_dqdd + state_size/2*(state_size+control_size);
    T *s_q = s_xux; 	
    T *s_qd = s_q + state_size/2; 		
    T *s_u = s_qd + state_size/2;
    gato_plant::forwardDynamicsAndGradient<T>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const);
    block.sync();
    
    if (COMPUTE_INTEGRATOR_ERROR){
        exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xux[state_size+control_size], &s_xux[state_size+control_size+state_size/2], s_q, s_qd, s_qdd, dt, block);
    }
    else{
        exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xnew_err, &s_xnew_err[state_size/2], s_q, s_qd, s_qdd, dt, block);
    }
    
    exec_integrator_gradient<T,INTEGRATOR_TYPE>(state_size, control_size, s_Ak, s_Bk, s_dqdd, dt, block);
}

template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
T integratorError(uint32_t state_size, T *s_xuk, T *s_xkp1, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){

    T *s_q = s_xuk; 					
    T *s_qd = s_q + state_size/2; 				
    T *s_u = s_qd + state_size/2;
    T *s_qkp1 = s_xkp1; 				
    T *s_qdkp1 = s_qkp1 + state_size/2;
    T *s_qdd = s_temp; 					
    T *s_err = s_qdd + state_size/2;
    T *s_extra_temp = s_err + state_size/2;
    gato_plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, block);
    block.sync();
    exec_integrator_error<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_err, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, block, true);
    block.sync();
    glass::reduce<T>(state_size, s_err);
    block.sync();
    return s_err[0];
}

template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void integrator(uint32_t state_size, T *s_xkp1, T *s_xuk, T *s_temp, void *d_dynMem_const, T dt, cgrps::thread_block block){
    T *s_q = s_xuk; 					T *s_qd = s_q + state_size/2; 				T *s_u = s_qd + state_size/2;
    T *s_qkp1 = s_xkp1; 				T *s_qdkp1 = s_qkp1 + state_size/2;
    T *s_qdd = s_temp; 					T *s_extra_temp = s_qdd + state_size/2;
    gato_plant::forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const, block);
    block.sync();
    exec_integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_qkp1, s_qdkp1, s_q, s_qd, s_qdd, dt, block);
}

template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void integrator_kernel(uint32_t state_size, uint32_t control_size, T *d_xkp1, T *d_xuk, void *d_dynMem_const, T dt){
    extern __shared__ T s_smem[];
    T *s_xkp1 = s_smem;
    T *s_xuk = s_xkp1 + state_size; 
    T *s_temp = s_xuk + state_size + control_size;
    cgrps::thread_block block = cgrps::this_thread_block();	  
    
    for (unsigned ind = threadIdx.x; ind < state_size + control_size; ind += blockDim.x){
        s_xuk[ind] = d_xuk[ind];
    }

    block.sync();
    integrator<T,INTEGRATOR_TYPE,ANGLE_WRAP>(state_size, s_xkp1, s_xuk, s_temp, d_dynMem_const, dt, block);
    block.sync();

    for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x){
        d_xkp1[ind] = s_xkp1[ind];
    }
}

template <typename T>
__global__
void simple_integrator_kernel(uint32_t state_size, uint32_t control_size, T *d_x, T *d_u, void *d_dynMem_const, T dt){

    extern __shared__ T s_mem[];
    T *s_xkp1 = s_mem;
    T *s_xuk = s_xkp1 + state_size; 
    T *s_temp = s_xuk + state_size + control_size;
    cgrps::thread_block block = cgrps::this_thread_block();	  
    
    for (unsigned ind = threadIdx.x; ind < state_size + control_size; ind += blockDim.x){
        if(ind < state_size){
            s_xuk[ind] = d_x[ind];
        }
        else{
            s_xuk[ind] = d_u[ind-state_size];
        }
    }

    block.sync();
    integrator<T,0,0>(state_size, s_xkp1, s_xuk, s_temp, d_dynMem_const, dt, block);
    block.sync();

    for (unsigned ind = threadIdx.x; ind < state_size; ind += blockDim.x){
        d_x[ind] = s_xkp1[ind];
    }
}


int main() {
    uint32_t state_size = 12;
    uint32_t control_size = 6;
    float dt = 0.01f;
    
    float *d_data;
    CHECK_GPU(cudaMalloc(&d_data, 1024 * sizeof(float)));

    size_t smem_size = sizeof(float)*(2*state_size + control_size + state_size/2 + gato_plant::forwardDynamicsAndGradient_TempMemSize_Shared());

    integrator_kernel<float, 0, false><<<1, 32, smem_size>>>(
        state_size, control_size, d_data, d_data, nullptr, dt
    );
    CHECK_GPU(cudaGetLastError());

    simple_integrator_kernel<float><<<1, 32, smem_size>>>(
        state_size, control_size, d_data, d_data, nullptr, dt
    );
    CHECK_GPU(cudaGetLastError());

    CHECK_GPU(cudaDeviceSynchronize());
    CHECK_GPU(cudaFree(d_data));

    std::cout << "OK: 023_integrator_kernels" << std::endl;
    return 0;
}