#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_GPU(call)                                                    \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "GPU error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err));     \
            std::cout << "FAIL: 026_grid_dynamics_core" << std::endl;      \
            exit(1);                                                       \
        }                                                                  \
    } while (0)


template <typename T>
struct robotModel {};

template <typename T>
__device__ void load_update_XImats_helpers(T *s_XImats, T *s_q, const robotModel<T> *d_robotModel, T *s_temp) {}

template <typename T>
__device__ void inverse_dynamics_inner(T *s_c, T *s_vaf, T *s_q, T *s_qd, T *s_qdd, T *s_XImats, T *s_temp, T gravity) {}

// Overload für qdd = 0
template <typename T>
__device__ void inverse_dynamics_inner(T *s_c, T *s_vaf, T *s_q, T *s_qd, T *s_XImats, T *s_temp, T gravity) {}

template <typename T>
__device__ void direct_minv_inner(T *s_Minv, T *s_q, T *s_XImats, T *s_temp) {}

template <typename T>
__device__ void forward_dynamics_inner(T *s_qdd, T *s_q, T *s_qd, T *s_u, T *s_XImats, T *s_temp, T gravity) {}


// ==========================================
// 2. ORIGINAL-CODE 
// ==========================================

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param d_qdd is the vector of joint accelerations
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_qdd[7]; 
        __shared__ T s_c[7];
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
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_c_k = &d_c[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_c_k[ind] = s_c[ind];
            }
            __syncthreads();
        }
    }    

    /**
     * Compute the RNEA (Recursive Newton-Euler Algorithm)
     *
     * Notes:
     * optimized for qdd = 0
     *
     * @param d_c is the vector of output torques
     * @param d_q_dq is the vector of joint positions and velocities
     * @param stride_q_qd is the stide between each q, qd
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant,num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd[2*7]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[7];
        __shared__ T s_c[7];
        __shared__ T s_vaf[126];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_k = &d_q_qd[k*stride_q_qd];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 14; ind += blockDim.x*blockDim.y){
                s_q_qd[ind] = d_q_qd_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            inverse_dynamics_inner<T>(s_c, s_vaf, s_q, s_qd, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_c_k = &d_c[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_c_k[ind] = s_c[ind];
            }
            __syncthreads();
        }
    }    

    /**
     * Compute the inverse of the mass matrix
     *
     * Notes:
     * Outputs a SYMMETRIC_UPPER triangular matrix for Minv
     *
     * @param d_Minv is a pointer to memory for the final result
     * @param d_q is the vector of joint positions
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void direct_minv_kernel(T *d_Minv, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){
        __shared__ T s_q[7];
        __shared__ T s_Minv[49];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_k = &d_q[k*stride_q];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                s_q[ind] = d_q_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);
            __syncthreads();
            // save down to global
            T *d_Minv_k = &d_Minv[k*49];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
                d_Minv_k[ind] = s_Minv[ind];
            }
            __syncthreads();
        }
    }    

    /**
     * Computes forward dynamics
     *
     * @param d_qdd is a pointer to memory for the final result
     * @param d_q_qd_u is the vector of joint positions, velocities, and input torques
     * @param d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)
     * @param gravity is the gravity constant
     * @param num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)
     */
    template <typename T>
    __global__
    void forward_dynamics_kernel(T *d_qdd, const T *d_q_qd_u, const int stride_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {
        __shared__ T s_q_qd_u[21]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[7]; T *s_u = &s_q_qd_u[14];
        __shared__ T s_qdd[7];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
        for(int k = blockIdx.x + blockIdx.y*gridDim.x; k < NUM_TIMESTEPS; k += gridDim.x*gridDim.y){
            // load to shared mem
            const T *d_q_qd_u_k = &d_q_qd_u[k*stride_q_qd_u];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 21; ind += blockDim.x*blockDim.y){
                s_q_qd_u[ind] = d_q_qd_u_k[ind];
            }
            __syncthreads();
            // compute
            load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
            forward_dynamics_inner<T>(s_qdd, s_q, s_qd, s_u, s_XImats, s_temp, gravity);
            __syncthreads();
            // save down to global
            T *d_qdd_k = &d_qdd[k*7];
            for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 7; ind += blockDim.x*blockDim.y){
                d_qdd_k[ind] = s_qdd[ind];
            }
            __syncthreads();
        }
    }




int main() {
    float *d_data;
    CHECK_GPU(cudaMalloc(&d_data, 1024 * sizeof(float)));

    robotModel<float> *d_robot_model = nullptr;
    
    int num_timesteps = 0;
    int stride = 14;
    float gravity = 9.81f;

    // Speichergröße muss > 504 sein, wegen &s_XITemp[504]
    size_t smem_size = 2048 * sizeof(float);

    dim3 blocks(1);
    dim3 threads(32);

    // 1. Inverse Dynamics (with qdd)
    inverse_dynamics_kernel<float><<<blocks, threads, smem_size>>>(
        d_data, d_data, stride, d_data, d_robot_model, gravity, num_timesteps
    );
    CHECK_GPU(cudaGetLastError());

    // 2. Inverse Dynamics (without qdd)
    inverse_dynamics_kernel<float><<<blocks, threads, smem_size>>>(
        d_data, d_data, stride, d_robot_model, gravity, num_timesteps
    );
    CHECK_GPU(cudaGetLastError());

    // 3. Direct Minv
    direct_minv_kernel<float><<<blocks, threads, smem_size>>>(
        d_data, d_data, stride, d_robot_model, num_timesteps
    );
    CHECK_GPU(cudaGetLastError());

    // 4. Forward Dynamics
    forward_dynamics_kernel<float><<<blocks, threads, smem_size>>>(
        d_data, d_data, stride, d_robot_model, gravity, num_timesteps
    );
    CHECK_GPU(cudaGetLastError());

    CHECK_GPU(cudaDeviceSynchronize());
    CHECK_GPU(cudaFree(d_data));

    std::cout << "OK: 026_grid_dynamics_core" << std::endl;
    return 0;
}