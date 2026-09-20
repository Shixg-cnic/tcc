#pragma once

#include "swtcf_convert.cuh"
#include <cuda_runtime.h>

namespace tcc {

    constexpr IndexType SWTCF_FEATURE_DIM = 128;
    constexpr IndexType SWTCF_DEFAULT_TC_THRESHOLD = 12;
    
    void launchSWTCFSpMM(
        const DeviceSWTCFMatrix& matrix,
        const ValueType* matrixX,
        ValueType* matrixY,
        IndexType tcThreshold = SWTCF_DEFAULT_TC_THRESHOLD,
        cudaStream_t stream = 0
    );

    
}   //tcc