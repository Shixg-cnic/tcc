#include "swtcf_spmm.cuh"
#include "mma_tf32.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tcc {
namespace {

constexpr int WARPS_PER_BLOCK = 8;
constexpr int WARPS_PER_GROUP = 4;
constexpr int WARP_SIZE = 32;
constexpr int GROUPS_PER_BLOCK = 2;

constexpr int FEATURES_PER_WARP = 32;
constexpr int FEATURES_PER_MMA = 16;

constexpr int SUPER_WINDOW_SIZE = 16;
constexpr int WINDOWS_PER_ITERATION = 2;
constexpr int NUM_ITERATIONS = SUPER_WINDOW_SIZE / WINDOWS_PER_ITERATION;

constexpr int SHARED_X_PER_GROUP = SWTCF_COL_WINDOW_WIDTH * SWTCF_FEATURE_DIM;
constexpr int SHARED_X_ELEMENTS = GROUPS_PER_BLOCK * SHARED_X_PER_GROUP;

constexpr int SHARED_ACC_PER_GROUP = SWTCF_REPEAT_CACHE_SIZE * SWTCF_FEATURE_DIM;
constexpr int SHARED_ACC_ELEMENTS = GROUPS_PER_BLOCK * SHARED_ACC_PER_GROUP;

constexpr IndexType INVALID_ROW_SLOT = std::numeric_limits<IndexType>::max();

__device__ __forceinline__ void accumulateOutput(
    OffsetType superRowBegin,
    IndexType rowSlot,
    IndexType feature,
    ValueType value,
    int warpGroup,
    const IndexType* __restrict__ superWindowRows,
    const std::uint8_t* __restrict__ superWindowRowCacheSlot,
    ValueType* __restrict__ groupAccumulator,
    ValueType* __restrict__ matrixY
) {
    if(rowSlot == INVALID_ROW_SLOT || value == 0.0f) return;

    const OffsetType superRowIndex = superRowBegin + rowSlot;
    const std::uint8_t cacheSlot = superWindowRowCacheSlot[superRowIndex];

    if(cacheSlot != SWTCF_INVALID_CACHE_SLOT) {
        const int index = warpGroup * SHARED_ACC_PER_GROUP + static_cast<int>(cacheSlot) * SWTCF_FEATURE_DIM + feature;
        groupAccumulator[index] += value;
    } else {
        const IndexType globalRow = superWindowRows[superRowIndex];
        atomicAdd(&matrixY[static_cast<std::size_t>(globalRow) * SWTCF_FEATURE_DIM + feature], value);
    }
}

__global__ void swtcfSpmmKernel(
    const OffsetType* __restrict__ superWindowRowOffset,
    const IndexType* __restrict__ superWindowRows,
    const std::uint8_t* __restrict__ superWindowRowCacheSlot,
    const IndexType* __restrict__ superWindowCachedRowSlot,
    const std::uint8_t* __restrict__ superWindowCachedCount,
    const OffsetType* __restrict__ colWindowOffset,
    const IndexType* __restrict__ tileRowSlot,
    const BitmapType* __restrict__ tileLocalBit,
    const ValueType* __restrict__ matrixX,
    ValueType* __restrict__ matrixY,
    IndexType cols,
    IndexType numColWindows,
    IndexType tcThreshold
) {
    const IndexType superWindow = static_cast<IndexType>(blockIdx.x);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int warpGroup = warp / WARPS_PER_GROUP;
    const int warpInGroup = warp % WARPS_PER_GROUP;

    const int groupThread = warpInGroup * WARP_SIZE + lane;
    const int blockThread = warp * WARP_SIZE + lane;

    const int mmaGroup = lane >> 2;
    const int threadInGroup = lane & 3;

    __shared__ ValueType sharedX[SHARED_X_ELEMENTS];
    __shared__ ValueType groupAccumulator[SHARED_ACC_ELEMENTS];

    for(int index = blockThread; index < SHARED_ACC_ELEMENTS; index += WARPS_PER_BLOCK * WARP_SIZE) groupAccumulator[index] = 0.0f;

    __syncthreads();

    const OffsetType superRowBegin = superWindowRowOffset[superWindow];

    for(int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
        const IndexType localWindow = static_cast<IndexType>(iteration * WINDOWS_PER_ITERATION + warpGroup);
        const IndexType colWindow = superWindow * SUPER_WINDOW_SIZE + localWindow;
        const bool validWindow = colWindow < numColWindows;

        ValueType* groupX = &sharedX[warpGroup * SHARED_X_PER_GROUP];

        for(int index = groupThread; index < SHARED_X_PER_GROUP; index += WARPS_PER_GROUP * WARP_SIZE) {
            const IndexType localColumn = static_cast<IndexType>(index / SWTCF_FEATURE_DIM);
            const IndexType feature = static_cast<IndexType>(index % SWTCF_FEATURE_DIM);
            const IndexType globalColumn = colWindow * SWTCF_COL_WINDOW_WIDTH + localColumn;

            if(validWindow && globalColumn < cols) {
                groupX[index] = matrixX[static_cast<std::size_t>(globalColumn) * SWTCF_FEATURE_DIM + feature];
            } else {
                groupX[index] = 0.0f;
            }
        }

        __syncthreads();

        if(validWindow) {
            const OffsetType tileBegin = colWindowOffset[colWindow];
            const OffsetType tileEnd = colWindowOffset[colWindow + 1];

            const IndexType featureBase = static_cast<IndexType>(warpInGroup * FEATURES_PER_WARP);
            const IndexType column0 = static_cast<IndexType>(threadInGroup);
            const IndexType column1 = static_cast<IndexType>(threadInGroup + 4);

            float mmaA0[4];
            float mmaA1[4];

            mmaA0[0] = groupX[column0 * SWTCF_FEATURE_DIM + featureBase + mmaGroup];
            mmaA0[1] = groupX[column0 * SWTCF_FEATURE_DIM + featureBase + mmaGroup + 8];
            mmaA0[2] = groupX[column1 * SWTCF_FEATURE_DIM + featureBase + mmaGroup];
            mmaA0[3] = groupX[column1 * SWTCF_FEATURE_DIM + featureBase + mmaGroup + 8];

            mmaA1[0] = groupX[column0 * SWTCF_FEATURE_DIM + featureBase + FEATURES_PER_MMA + mmaGroup];
            mmaA1[1] = groupX[column0 * SWTCF_FEATURE_DIM + featureBase + FEATURES_PER_MMA + mmaGroup + 8];
            mmaA1[2] = groupX[column1 * SWTCF_FEATURE_DIM + featureBase + FEATURES_PER_MMA + mmaGroup];
            mmaA1[3] = groupX[column1 * SWTCF_FEATURE_DIM + featureBase + FEATURES_PER_MMA + mmaGroup + 8];

            for(OffsetType tile = tileBegin; tile < tileEnd; ++tile) {
                const BitmapType bitmap = tileLocalBit[tile];
                const IndexType tileNnz = static_cast<IndexType>(__popcll(static_cast<unsigned long long>(bitmap)));

                if(tileNnz >= tcThreshold) {
                    float mmaB[2];
                    float mmaC0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    float mmaC1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                    const int bitPosition0 = mmaGroup * SWTCF_COL_WINDOW_WIDTH + threadInGroup;
                    const int bitPosition1 = mmaGroup * SWTCF_COL_WINDOW_WIDTH + threadInGroup + 4;

                    mmaB[0] = (bitmap & (BitmapType{1} << bitPosition0)) ? 1.0f : 0.0f;
                    mmaB[1] = (bitmap & (BitmapType{1} << bitPosition1)) ? 1.0f : 0.0f;

                    mmaTf32M16N8K8(mmaA0, mmaB, mmaC0);
                    mmaTf32M16N8K8(mmaA1, mmaB, mmaC1);

                    const IndexType localRow0 = static_cast<IndexType>(threadInGroup * 2);
                    const IndexType localRow1 = localRow0 + 1;

                    const IndexType rowSlot0 = tileRowSlot[static_cast<std::size_t>(tile) * SWTCF_TILE_ROWS + localRow0];
                    const IndexType rowSlot1 = tileRowSlot[static_cast<std::size_t>(tile) * SWTCF_TILE_ROWS + localRow1];

                    const IndexType feature0 = featureBase + mmaGroup;
                    const IndexType feature1 = featureBase + mmaGroup + 8;
                    const IndexType feature2 = featureBase + mmaGroup + 16;
                    const IndexType feature3 = featureBase + mmaGroup + 24;

                    accumulateOutput(superRowBegin, rowSlot0, feature0, mmaC0[0], warpGroup, superWindowRows, superWindowRowCacheSlot, groupAccumulator, matrixY);
                    accumulateOutput(superRowBegin, rowSlot0, feature1, mmaC0[2], warpGroup, superWindowRows, superWindowRowCacheSlot, groupAccumulator, matrixY);
                    accumulateOutput(superRowBegin, rowSlot0, feature2, mmaC1[0], warpGroup, superWindowRows, superWindowRowCacheSlot, groupAccumulator, matrixY);
                    accumulateOutput(superRowBegin, rowSlot0, feature3, mmaC1[2], warpGroup, superWindowRows, superWindowRowCacheSlot, groupAccumulator, matrixY);

                    accumulateOutput(superRowBegin, rowSlot1, feature0, mmaC0[1], warpGroup, superWindowRows, superWindowRowCacheSlot, groupAccumulator, matrixY);
                    accumulateOutput(superRowBegin, rowSlot1, feature1, mmaC0[3], warpGroup, superWindowRows, superWindowRowCacheSlot, groupAccumulator, matrixY);
                    accumulateOutput(superRowBegin, rowSlot1, feature2, mmaC1[1], warpGroup, superWindowRows, superWindowRowCacheSlot, groupAccumulator, matrixY);
                    accumulateOutput(superRowBegin, rowSlot1, feature3, mmaC1[3], warpGroup, superWindowRows, superWindowRowCacheSlot, groupAccumulator, matrixY);
                } else {
                    const IndexType feature = static_cast<IndexType>(groupThread);

                    for(IndexType localRow = 0; localRow < SWTCF_TILE_ROWS; ++localRow) {
                        const std::uint32_t rowBits = static_cast<std::uint32_t>((bitmap >> (localRow * SWTCF_COL_WINDOW_WIDTH)) & 0xFFULL);
                        if(rowBits == 0) continue;

                        ValueType sum = 0.0f;

                        #pragma unroll
                        for(IndexType localColumn = 0; localColumn < SWTCF_COL_WINDOW_WIDTH; ++localColumn) {
                            if(rowBits & (1u << localColumn)) sum += groupX[localColumn * SWTCF_FEATURE_DIM + feature];
                        }

                        const IndexType rowSlot = tileRowSlot[static_cast<std::size_t>(tile) * SWTCF_TILE_ROWS + localRow];

                        accumulateOutput(superRowBegin, rowSlot, feature, sum, warpGroup, superWindowRows, superWindowRowCacheSlot, groupAccumulator, matrixY);
                    }
                }
            }
        }

        __syncthreads();
    }

    __syncthreads();

    const IndexType cachedCount = static_cast<IndexType>(superWindowCachedCount[superWindow]);
    const int flushElements = static_cast<int>(cachedCount * SWTCF_FEATURE_DIM);

    for(int index = blockThread; index < flushElements; index += WARPS_PER_BLOCK * WARP_SIZE) {
        const IndexType cacheSlot = static_cast<IndexType>(index / SWTCF_FEATURE_DIM);
        const IndexType feature = static_cast<IndexType>(index % SWTCF_FEATURE_DIM);

        const IndexType rowSlot = superWindowCachedRowSlot[static_cast<std::size_t>(superWindow) * SWTCF_REPEAT_CACHE_SIZE + cacheSlot];
        const IndexType globalRow = superWindowRows[superRowBegin + rowSlot];

        const ValueType value0 = groupAccumulator[0 * SHARED_ACC_PER_GROUP + cacheSlot * SWTCF_FEATURE_DIM + feature];
        const ValueType value1 = groupAccumulator[1 * SHARED_ACC_PER_GROUP + cacheSlot * SWTCF_FEATURE_DIM + feature];
        const ValueType value = value0 + value1;

        if(value != 0.0f) atomicAdd(&matrixY[static_cast<std::size_t>(globalRow) * SWTCF_FEATURE_DIM + feature], value);
    }
}

}  // namespace

void launchSWTCFSpMM(
    const DeviceSWTCFMatrix& matrix,
    const ValueType* matrixX,
    ValueType* matrixY,
    IndexType tcThreshold,
    cudaStream_t stream
) {
    if(matrix.superWindowSize != SUPER_WINDOW_SIZE) throw std::runtime_error("SWTCF SpMM V1 requires superWindowSize=16");

    const dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    const dim3 grid(matrix.numSuperWindows);

    swtcfSpmmKernel<<<grid, block, 0, stream>>>(
        matrix.superWindowRowOffset,
        matrix.superWindowRows,
        matrix.superWindowRowCacheSlot,
        matrix.superWindowCachedRowSlot,
        matrix.superWindowCachedCount,
        matrix.colWindowOffset,
        matrix.tileRowSlot,
        matrix.tileLocalBit,
        matrixX,
        matrixY,
        matrix.cols,
        matrix.numColWindows,
        tcThreshold
    );
}

}  // namespace tcc