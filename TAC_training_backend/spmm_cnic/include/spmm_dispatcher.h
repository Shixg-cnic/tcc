#pragma once

#include "class.h"
#include "config.h"
#include "ptx_tf32.h"
#include "spmm_kernel.h"
#include "utils.h"

#include <cuda_runtime.h>


// ======================================= BCSR =======================================

__host__
float mixed_tf32_spmm_bcsr(
    const BCSR<MAT_VAL_TYPE>& bcsr, 
    const int32_t gpu_node_capacity, 
    const int32_t feature_dim,
    int32_t *id_map,
    float* cpu_feature_cache_ptr,
    int32_t* cache_search_map,
    float** gpu_feature_cache_ptr,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);

__host__
float tf32_spmm_bcsr(
    const BCSR<MAT_VAL_TYPE>& bcsr, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);

// ======================================= BCSC =======================================

__host__
float mixed_tf32_spmm_bcsc(
    const BCSC<MAT_VAL_TYPE>& bcsc, 
    const int32_t gpu_node_capacity, 
    const int32_t feature_dim,
    int32_t *id_map,
    float* cpu_feature_cache_ptr,
    int32_t* cache_search_map,
    float** gpu_feature_cache_ptr,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);

__host__
float tf32_spmm_bcsc(
    const BCSC<MAT_VAL_TYPE>& bcsc, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
);