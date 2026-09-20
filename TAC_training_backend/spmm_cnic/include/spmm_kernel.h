#pragma once

#include "ptx_tf32.h"

// ======================================= BCSR =======================================

__global__
void tf32_computeX128TransposePipe2_BCSR(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2B,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    int32_t* id_map,
    float* cpu_float_features,
    int32_t* cache_search_map,
    float** gpu_float_feature,
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    int32_t gpu_node_capacity,
    const vint feature_dim
);

__global__
void tf32_computeX128TransposePipe2_BCSR(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2B,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim
);

// ======================================= BCSC =======================================

__global__
void tf32_computeX128TransposePipe2_BCSC(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2C,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    int32_t* id_map,
    float* cpu_float_features,
    int32_t* cache_search_map,
    float** gpu_float_feature,
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    int32_t gpu_node_capacity,
    const vint feature_dim
);

__global__
void tf32_computeX128TransposePipe2_BCSC(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2C,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim
);

// ======================================= mixed =======================================

// first layer
__global__
void tf32_computeX128TransposePipe2(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2X,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    int32_t* id_map,
    float* cpu_float_features,
    int32_t* cache_search_map,
    float** gpu_float_feature,
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    int32_t gpu_node_capacity,
    const vint feature_dim,
    int flag
);

// second layer
__global__
void tf32_computeX128TransposePipe2(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2X,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim,
    int flag
);