#pragma once
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

#include "config.h"
#include "class.h"
#include "utils.h"
#include "spmm_dispatcher.h"

#define CHECK_CUDA_ERROR(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)

#define CHECK_LAST_CUDA_ERROR() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)

void launch_spmm_kernel(
    const BCSR<MAT_VAL_TYPE> &bcsr,
    const BCSC<MAT_VAL_TYPE> &bcsc,
    int32_t* id_map,
    float* cpu_feature_cache_ptr,
    int32_t* cache_search_map,
    float** gpu_feature_cache_ptr,
    MAT_VAL_TYPE* &d_MatC,
    int64_t gpu_node_capacity,
    int32_t feature_dim,
    const int32_t device_id,
    cudaStream_t stream_bcsr,
    cudaStream_t stream_bcsc
);

void launch_spmm_kernel(
    const BCSR<MAT_VAL_TYPE> &bcsr,
    const BCSC<MAT_VAL_TYPE> &bcsc,
    MAT_VAL_TYPE* d_MatB,
    int32_t* id_map,
    float* cpu_feature_cache_ptr,
    int32_t* cache_search_map,
    float** gpu_feature_cache_ptr,
    MAT_VAL_TYPE* &d_MatC,
    int64_t gpu_node_capacity,
    int32_t feature_dim,
    const int32_t device_id,
    cudaStream_t stream_bcsr,
    cudaStream_t stream_bcsc
);

void launch_spmm_kernel(
    const BCSR<MAT_VAL_TYPE> &bcsr,
    const BCSC<MAT_VAL_TYPE> &bcsc,
    MAT_VAL_TYPE* d_MatB,
    MAT_VAL_TYPE* &d_MatC,
    int32_t feature_dim,
    const int32_t device_id,
    cudaStream_t stream_bcsr,
    cudaStream_t stream_bcsc
);