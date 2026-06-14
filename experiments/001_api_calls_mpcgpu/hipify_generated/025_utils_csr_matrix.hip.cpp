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
            std::cout << "FAIL: 025_utils_csr_matrix" << std::endl;        \
            exit(1);                                                       \
        }                                                                  \
    } while (0)


typedef float QDLDL_float;
typedef int QDLDL_int;



// fills in the values of the lower triangle of a symmetric block tridiagonal matrix
template <typename T>
__device__
void store_block_csr_lowertri(uint32_t bdim, uint32_t mdim, T *d_src, QDLDL_float *d_val, bool col1, uint32_t bd_block_row, int32_t multiplier=1){
    
    const int brow_val_ct = bdim*bdim + ((bdim+1)*bdim)/2;
    int row, col, csr_row_offset, full_csr_offset;
    int write_len;
    int cur_triangle_offset;

    for(row = threadIdx.x; row < bdim; row += blockDim.x){

        cur_triangle_offset = ((row+1)*row)/2;
        csr_row_offset = (bd_block_row>0)*((bdim+1)*bdim)/2 +                   // add triangle if not first block row
                         (bd_block_row>0) * (bd_block_row-1)*brow_val_ct +      // add previous full block rows if not first block row
                         (bd_block_row>0)*row*bdim +                            // 
                         cur_triangle_offset;                                   // triangle offset

        write_len = (bd_block_row>0)*((!col1)*(bdim)+(col1)*(row+1)) + (col1)*(bd_block_row==0)*(row+1);
        
        for(col = 0; col<write_len; col++){
            full_csr_offset = csr_row_offset + (bd_block_row>0)*(col1)*bdim + col;
            d_val[full_csr_offset] = static_cast<QDLDL_float>(d_src[row + col*bdim]) * multiplier;
        }
    }
}


// fills in the column pointers and row indices for the CSR representation of the lower triangle of a symmetric block tridiagonal matrix
__global__
void prep_csr(uint32_t state_size, uint32_t knot_points, QDLDL_int *d_col_ptr, QDLDL_int *d_row_ind){
    
    for (uint32_t blockrow = blockIdx.x; blockrow < knot_points; blockrow+=gridDim.x)
    {
        const int brow_val_ct = state_size*state_size + ((state_size+1)*state_size)/2;
        int row, col, csr_row_offset, full_csr_offset, bd_row_len;
        int cur_triangle_offset;

        for(row = threadIdx.x; row < state_size; row += blockDim.x){

            if(blockrow==0 && row==0){
                d_col_ptr[0] = 0;
            }
            
            cur_triangle_offset = ((row+1)*row)/2;
            csr_row_offset = (blockrow>0)*((state_size+1)*state_size)/2 +                   // add triangle if not first block row
                            (blockrow>0) * (blockrow-1)*brow_val_ct +      // add previous full block rows if not first block row
                            (blockrow>0)*row*state_size +                            // 
                            cur_triangle_offset;                                   // triangle offset

            bd_row_len = (blockrow>0)*state_size + row+1;
            d_col_ptr[blockrow*state_size + row+1] = csr_row_offset+bd_row_len;
            
            for(col = 0; col < bd_row_len; col++){
                full_csr_offset = csr_row_offset + col;
                d_row_ind[full_csr_offset] = (blockrow>0)*(blockrow-1)*state_size + col;
            }
        }
    }
}


template <typename T>
__global__
void test_store_block_wrapper(uint32_t bdim, uint32_t mdim, T *d_src, QDLDL_float *d_val, bool col1, uint32_t bd_block_row) {
    store_block_csr_lowertri<T>(bdim, mdim, d_src, d_val, col1, bd_block_row, 1);
}


int main() {
    uint32_t state_size = 12;
    uint32_t knot_points = 2;
    
    QDLDL_int *d_col_ptr;
    QDLDL_int *d_row_ind;
    QDLDL_float *d_val;
    float *d_src;

    CHECK_GPU(hipMalloc(&d_col_ptr, 1024 * sizeof(QDLDL_int)));
    CHECK_GPU(hipMalloc(&d_row_ind, 1024 * sizeof(QDLDL_int)));
    CHECK_GPU(hipMalloc(&d_val, 1024 * sizeof(QDLDL_float)));
    CHECK_GPU(hipMalloc(&d_src, 1024 * sizeof(float)));


    prep_csr<<<1, 32>>>(state_size, knot_points, d_col_ptr, d_row_ind);
    CHECK_GPU(hipGetLastError());


    test_store_block_wrapper<float><<<1, 32>>>(state_size, state_size, d_src, d_val, true, 0);
    CHECK_GPU(hipGetLastError());


    float h_matrix[4][4] = {{1.0f}};
    float *d_matrix;
    size_t pitch;

    CHECK_GPU(hipMallocPitch(&d_matrix, &pitch, 4 * sizeof(float), 4));
    CHECK_GPU(hipMemcpy2D(d_matrix, pitch, h_matrix, 4 * sizeof(float), 4 * sizeof(float), 4, hipMemcpyHostToDevice));

    CHECK_GPU(hipDeviceSynchronize());

    CHECK_GPU(hipFree(d_col_ptr));
    CHECK_GPU(hipFree(d_row_ind));
    CHECK_GPU(hipFree(d_val));
    CHECK_GPU(hipFree(d_src));
    CHECK_GPU(hipFree(d_matrix));

    std::cout << "OK: 025_utils_csr_matrix" << std::endl;
    return 0;
}