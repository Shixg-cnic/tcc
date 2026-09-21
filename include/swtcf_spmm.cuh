#pragma once

#include "swtcf_convert.cuh"
#include <cuda_runtime.h>
#include <cstdint>

constexpr IndexType SWTCF_FEATURE_DIM = 128;
constexpr IndexType SWTCF_DEFAULT_TC_THRESHOLD = 12;

void launchSWTCFSpMM(
    const DeviceSWTCFMatrix& matrix,
    const ValueType* matrixX,
    ValueType* matrixY,
    IndexType tcThreshold = SWTCF_DEFAULT_TC_THRESHOLD,
    cudaStream_t stream = 0
);

void launchSWTCFSpMMQuantized(
    const DeviceSWTCFMatrix& matrix,
    const std::int8_t* quantizedX,
    const ValueType* scales,
    const IndexType* srcGlobal,
    ValueType* matrixY,
    IndexType tcThreshold = SWTCF_DEFAULT_TC_THRESHOLD,
    cudaStream_t stream = 0
);