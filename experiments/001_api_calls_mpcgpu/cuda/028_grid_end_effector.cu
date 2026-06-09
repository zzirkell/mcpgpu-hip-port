#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err));     \
            std::cout << "FAIL: 028_grid_end_effector" << std::endl;       \
            exit(1);                                                       \
        }                                                                  \
    } while (0)


template <typename T>
struct robotModel {};

// Helper for end_effector_positions_kernel
template <typename T>
__device__ void load_update_XmatsHom_helpers(T *s_XmatsHom, T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {}

template <typename T>
__device__ void end_effector_positions_inner(T *s_eePos, T *s_q, T *s_XmatsHom, T *s_temp) {}

// Helper for end_effector_positions_gradient_kernel (Overloaded)
template <typename T>
__device__ void load_update_XmatsHom_helpers(T *s_XmatsHom, T *s_dXmatsHom, T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {}

template <typename T>
__device__ void end_effector_positions_gradient_inner(T *s_deePos, T *s_q, T *s_XmatsHom, T *s_dXmatsHom, T *s_temp) {}


// ==========================================
// 2. ORIGINAL-CODE 
// ==========================================

    /**
     * Compute the End Effector Position
     *
     * @param d_eePos is the vector of end effector positions
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_positions_kernel(T *d_eePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_eePos[6];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_temp = &s_XHomTemp[112];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q, d_robotModel, s_temp);
            end_effector_positions_inner<T>(s_eePos, s_q, s_XmatsHom, s_temp);
            __syncthreads();
            // save down to global
            T *d_eePos_k = &d_eePos[k*6];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 6; ind += blockDim.x*blockDim.y){
                d_eePos_k[ind] = s_eePos[ind];
            }
            __syncthreads();
        }
    }    

    /**
     * Computes the Gradient of the End Effector Position with respect to joint position
     *
     * @param d_deePos is the vector of end effector positions gradients
     * @param d_q is the vector of joint positions
     * @param stride_q is the stide between each q
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void end_effector_positions_gradient_kernel(T *d_deePos, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {
        __shared__ T s_q[7];
        __shared__ T s_deePos[42];
        extern __shared__ T s_XHomTemp[]; T *s_XmatsHom = s_XHomTemp; T *s_dXmatsHom = &s_XHomTemp[112]; T *s_temp = &s_dXmatsHom[112];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_q, d_robotModel, s_temp);
            end_effector_positions_gradient_inner<T>(s_deePos, s_q, s_XmatsHom, s_dXmatsHom, s_temp);
            __syncthreads();
            // save down to global
            T *d_deePos_k = &d_deePos[k*42];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 42; ind += blockDim.x*blockDim.y){
                d_deePos_k[ind] = s_deePos[ind];
            }
            __syncthreads();
        }
    }



int main() {
    float *d_data;
    CHECK_GPU(cudaMalloc(&d_data, 1024 * sizeof(float)));

    robotModel<float> *d_robot_model = nullptr;
    
    int num_timesteps = 0;
    int stride = 7;

    size_t smem_size = 1024 * sizeof(float);

    dim3 blocks(1);
    dim3 threads(32);

    // 1. End Effector Position Kernel
    end_effector_positions_kernel<float><<<blocks, threads, smem_size>>>(
        d_data, d_data, stride, d_robot_model, num_timesteps
    );
    CHECK_GPU(cudaGetLastError());

    // 2. End Effector Position Gradient Kernel
    end_effector_positions_gradient_kernel<float><<<blocks, threads, smem_size>>>(
        d_data, d_data, stride, d_robot_model, num_timesteps
    );
    CHECK_GPU(cudaGetLastError());

    CHECK_GPU(cudaDeviceSynchronize());
    CHECK_GPU(cudaFree(d_data));

    std::cout << "OK: 028_grid_end_effector" << std::endl;
    return 0;
}