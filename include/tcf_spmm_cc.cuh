#pragma once

#include "format.hpp"
#include "cuda_runtime.h"

namespace tcc {

void launchTcfSpMMCC(
    const OffsetType* colWindowOffset,
    const OffsetType* tileOffset,
    const IndexType* tileRowIndices,
    const BitmapType* tileLocalBit,
    const ValueType* values,
    const ValueType* matrixX,
    ValueType* matrixY,
    IndexType rows,
    IndexType cols,
    IndexType numColWindows,
    IndexType featureDim,
    cudaStream_t stream = nullptr
);

}