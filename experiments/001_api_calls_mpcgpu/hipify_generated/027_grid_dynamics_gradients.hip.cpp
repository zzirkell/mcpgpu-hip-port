#include "hip/hip_runtime.h"
#include <iostream>
#include <cstdio>
#include <hip/hip_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                            \
        if (err != hipSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, hipGetErrorString(err));     \
            std::cout << "FAIL: 027_grid_dynamics_gradients" << std::endl; \
            exit(1);                                                       \
        }                                                                  \
    } while (0)



template <typename T>
struct robotModel {};

template <typename T>
__device__ void load_update_XImats_helpers(T *s_XImats, T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {}

template <typename T>
__device__ void inverse_dynamics_inner_vaf(T *s_vaf, T *s_q, T *s_qd, T *s_qdd, T *s_XImats, T *s_temp, T gravity) {}

template <typename T>
__device__ void inverse_dynamics_gradient_inner(T *s_dc_du, T *s_q, T *s_qd, T *s_vaf, T *s_XImats, T *s_temp, T gravity) {}




    /**
     * Computes the gradient of inverse dynamics
     *
     * @param d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_gradient_kernel(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_qdd[7]; 
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_dc_du_k = &d_dc_du[k*98];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_dc_du_k[ind] = s_dc_du[ind];
            }
            __syncthreads();
        }
    }    

    /**
     * Computes the gradient of forward dynamics
     *
     * @param d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = 98
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param d_qdd is the vector of joint accelerations
     * @param d_Minv is the mass matrix
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_gradient_kernel(T *d_df_du, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_dc_du[98];
        __shared__ T s_vaf[126];
        __shared__ T s_qdd[7];
        __shared__ T s_Minv[49];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            const T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_qdd[ind] = d_qdd_k[ind];
            }
            const T *d_Minv_k = &d_Minv[k*49];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                s_Minv[ind] = d_Minv_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner_vaf<T>(s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            inverse_dynamics_gradient_inner<T>(s_dc_du, s_q, s_qd, s_vaf, s_XImats, s_temp, gravity);
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                int row = ind % 7; int dc_col_offset = ind - row;
                // account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix
                T val = static_cast<T>(0);
                for(int col = 0; col < 7; col++) {
                    int index = (row <= col) * (col * 7 + row) + (row > col) * (row * 7 + col);
                    val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                }
                s_temp[ind] = -val;
            }
            // save down to global
            T *d_df_du_k = &d_df_du[k*98];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 98; ind += blockDim.x*blockDim.y){
                d_df_du_k[ind] = s_temp[ind];
            }
            __syncthreads();
        }
    }



int main() {
    float *d_data;
    CHECK_GPU(hipMalloc(&d_data, 1024 * sizeof(float)));

    robotModel<float> *d_robot_model = nullptr;
    
    int num_timesteps = 0;
    int stride = 14;
    float gravity = 9.81f;

    size_t smem_size = 2048 * sizeof(float);

    dim3 blocks(1);
    dim3 threads(32);

    // 1. Inverse Dynamics Gradient
    inverse_dynamics_gradient_kernel<float><<<blocks, threads, smem_size>>>(
        d_data, d_data, stride, d_data, d_robot_model, gravity, num_timesteps
    );
    CHECK_GPU(hipGetLastError());

    // 2. Forward Dynamics Gradient
    forward_dynamics_gradient_kernel<float><<<blocks, threads, smem_size>>>(
        d_data, d_data, stride, d_data, d_data, d_robot_model, gravity, num_timesteps
    );
    CHECK_GPU(hipGetLastError());

    CHECK_GPU(hipDeviceSynchronize());
    CHECK_GPU(hipFree(d_data));

    std::cout << "OK: 027_grid_dynamics_gradients" << std::endl;
    return 0;
}