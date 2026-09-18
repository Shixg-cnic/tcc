#include "swtcf_spmm.cuh"
#include "mma_tf32.cuh"
#include "cuda_runtime.h"
#include <stdexcept>

namespace tcc {
namespace {

constexpr int WARPS_PER_BLOCK = 8;
constexpr int WARPS_PER_GROUP = 4;
constexpr int WARP_SIZE = 32;

constexpr int GROUPS_PER_BLOCK = 2;

constexpr int FEATURES_PER_WARP = 32;
constexpr int FEATURES_PER_MMA = 16;
constexpr int WINDOWS_PER_ITERATION = 2;

// 2 groups * 8 source rows * 128 features
constexpr int SHARED_X_ELEMENTS = GROUPS_PER_BLOCK * SWTCF_COL_WINDOW_WIDTH * SWTCF_FEATURE_DIM;
//  32 repeated rows * 128 features
constexpr int SHARED_ACC_ELEMENTS = SWTCF_REPEAT_CACHE_SIZE * SWTCF_FEATURE_DIM;

__device__ __forceinline__ void accumulateOutput(
    OffsetType superRowBegin,
    IndexType rowSlot,
    IndexType feature,
    ValueType value,
    
    const IndexType* __restrict__ superWindowRows,
    const std::uint8_t* __restrict__ superWindowRowCacheSlot,
    
    ValueType* __restrict__ sharedAccumulator,
    ValueType* __restrict__ matrixY
) {
    if(rowSlot == std::numeric_limits<IndexType>::max()) { return; }
    if(value == 0.0f) { return; }
    const OffsetType superRowIndex = superRowBegin + rowSlot;
    const std::uint8_t cacheSlot = superWindowRowCacheSlot[superRowIndex];
    if(cacheSlot != SWTCF_INVALID_CACHE_SLOT){
        atomicAdd(&sharedAccumulator[static_cast<int>(cacheSlot) * SWTCF_FEATURE_DIM + feature], value);
    }else{
        const IndexType globalRow = superWindowRows[superRowIndex];
        atomicAdd(&matrixY[static_cast<std::size_t>(globalRow) * SWTCF_FEATURE_DIM +feature], value);   
    }
    
}

__global__ void swtcfSpmmKernel(
    const OffsetType* __restrict__ superWindowRowOffset,
    const IndexType* __restrict__ superWindowRows,
    const std::uint8_t* __restrict__ superWindowRowCacheSlot,
    const OffsetType* __restrict__ colWindowOffset,
    const IndexType* __restrict__ tileLocalBit,
    const ValueType* __restrict__ matrixX,
    ValueType* __restrict__ matrixY,
    IndexType cols,
    IndexType numColWindows,
    IndexType tcThreshold
) {
    // --------------------------------------------------
    // CTA:
    //
    // warp 0~3 -> group 0
    // warp 4~7 -> group 1
    //
    // One CTA owns one 16-window super-window.
    // --------------------------------------------------
    const IndexType superWindow = blockIdx.x;
    const int lane = threadIdx.x;
    const int warp threadIdx.y;
    const int warpGroup = warp / WARPS_PER_GROUP;
    const int warpInGroup = warp % WARPS_PER_GROUP;
    const int gruopThread = warpInGroup * WARP_SIZE;
    const int blockThread = warp * WARP_SIZE + lane;
    const int mmaGroup = lane >> 2;
    const int threadGroup = lane & 3;
    

        // --------------------------------------------------
    // Shared memory.
    //
    // sharedX:
    //
    // [group][local source column][feature]
    //
    // repeatAccumulator:
    //
    // [cacheSlot][feature]
    // --------------------------------------------------

    __shared__ ValueType sharedX[SHARED_X_ELEMENTS];
    __shared__ ValueType repeatAccumulator[SHARED_ACC_ELEMENTS];

    for(int index = blockThread; index < SHARED_ACC_ELEMENTS; index += WARPS_PER_BLOCK * WARP_SIZE){
        repeatAccumulator[index] = 0.0f;
    }
    __synsthreads();
    
    const OffsetType superRowBegin = superWindowRowOffset[superWindow];
    for(IndexType iteration = 0; iteration < SWTCF_MAX_SUPER_WINDOW_SIZE / 2; iteration++){
        const IndexType localWindow = iteration * WINDOWS_PER_ITERATION + static_cast<IndexType>(warpGroup);
        const bool validLocalWindow = localWindow < 16;
        const IndexType colWindow = superWindow * 16 + localWindow;
        const bool validWindow = validLocalWindow && colWindow < numColWindows;

        ValueType* groupX = &sharedX[warpGroup * SWTCF_COL_WINDOW_WIDTH * SWTCF_FEATURE_DIM];
        for(int index = gruopThread; index < SWTCF_COL_WINDOW_WIDTH * SWTCF_FEATURE_DIM; index += WARPS_PER_BLOCK * WARP_SIZE) {
            const IndexType localColumn = static_cast<IndexType>(index / SWTCF_FEATURE_DIM);
            const IndexType feature = static_cast<IndexType>(index % SWTCF_FEATURE_DIM);
            const IndexType globalColumn = colWindow * SWTCF_COL_WINDOW_WIDTH + localColumn;
            
            if(validWindow && globalColumn < cols) {
                groupX[index] = matrixX[static_cast<std::size_t>(globalColumn) * SWTCF_FEATURE_DIM +feature];
            }else{
                groupX[index] = 0.0f;
            }
        }
        __synsthreads();
        
        if(validWindow) {
            const OffsetType tileBegin = colWindowOffset[colWindow];
            const OffsetType tileEnd = colWindowOffset[colWindow + 1];
            const IndexType featureBase = static_cast<IndexType>(warpInGroup * FEATURES_PER_WARP);
            float mmaA0[4];
            float mmaA1[4];
            const IndexType column0 = static_cast<IndexType>(threadInGroup);
            const IndexType column1 = static_cast<IndexType>(threadInGroup + 4);

            mmaA0[0] = groupX[column0 * SWTCF_FEATURE_DIM + featureBase + mmaGroup];
            mmaA0[1] = groupX[column0 * SWTCF_FEATURE_DIM + featureBase + mmaGroup + 8];
            mmaA0[2] = groupX[column1 * SWTCF_FEATURE_DIM + featureBase + mmaGroup];
            mmaA0[3] = groupX[column1 * SWTCF_FEATURE_DIM + featureBase + mmaGroup + 8];

            mmaA1[0] = groupX[column0 * SWTCF_FEATURE_DIM + featureBase + FEATURES_PER_MMA + mmaGroup];
            mmaA1[1] = groupX[column0 * SWTCF_FEATURE_DIM + featureBase + FEATURES_PER_MMA + mmaGroup + 8];
            mmaA1[2] = groupX[column1 * SWTCF_FEATURE_DIM + featureBase + FEATURES_PER_MMA + mmaGroup];
            mmaA1[3] = groupX[column1 * SWTCF_FEATURE_DIM + featureBase + FEATURES_PER_MMA + mmaGroup + 8];

            for(OffsetType tile = tileBegin; tile < tileEnd; tile++) {
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
                    const IndexType rowSlot1 = tileRowSlot[static_cast<std::size_t>(tile) * SWTCF_TILE_ROWS + localROw1];

                    const IndexType feature0 = featureBase + mmaGroup;
                    const IndexType feature1 = featureBase + mmaGroup + 8;
                    const IndexType feature2 = featureBase + mmaGroup + 16;
                    const IndexType feature3 = featureBase + mmaGroup + 24;

                    accumulateOutput(
                        superRowBegin,
                        rowSlot0,
                        feature0,
                        mmaC0[0],
                        superWindowRows,
                        superWindowRowCacheSlot,
                        repeatAccumulator,
                        matrixY
                    );
                     accumulateOutput(
                        superRowBegin,
                        rowSlot0,
                        feature1,
                        mmaC0[2],
                        superWindowRows,
                        superWindowRowCacheSlot,
                        repeatAccumulator,
                        matrixY
                    );

                    accumulateOutput(
                        superRowBegin,
                        rowSlot0,
                        feature2,
                        mmaC1[0],
                        superWindowRows,
                        superWindowRowCacheSlot,
                        repeatAccumulator,
                        matrixY
                    );

                    accumulateOutput(
                        superRowBegin,
                        rowSlot0,
                        feature3,
                        mmaC1[2],
                        superWindowRows,
                        superWindowRowCacheSlot,
                        repeatAccumulator,
                        matrixY
                    );

                    accumulateOutput(
                        superRowBegin,
                        rowSlot1,
                        feature0,
                        mmaC0[1],
                        superWindowRows,
                        superWindowRowCacheSlot,
                        repeatAccumulator,
                        matrixY
                    );

                    accumulateOutput(
                        superRowBegin,
                        rowSlot1,
                        feature1,
                        mmaC0[3],
                        superWindowRows,
                        superWindowRowCacheSlot,
                        repeatAccumulator,
                        matrixY
                    );

                    accumulateOutput(
                        superRowBegin,
                        rowSlot1,
                        feature2,
                        mmaC1[1],
                        superWindowRows,
                        superWindowRowCacheSlot,
                        repeatAccumulator,
                        matrixY
                    );

                    accumulateOutput(
                        superRowBegin,
                        rowSlot1,
                        feature3,
                        mmaC1[3],
                        superWindowRows,
                        superWindowRowCacheSlot,
                        repeatAccumulator,
                        matrixY
                    );
                }else {
                    const IndexType feature = static_cast<IndexType>(gruopThread);
                    for(IndexType localRow = 0; localRow < SWTCF_TILE_ROWS; localRow++) {
                        const std::uint32_t rowBits = static_cast<std::uint32_t>((bitmap >> (localRow * SWTCF_COL_WINDOW_WIDTH)) & 0XFFULL);
                        if(rowBits == 0) { continue; }
                        ValueType sum = 0.0f;
                        #pragma unroll
                        for(IndexType localColumn = 0; localColumn < SWTCF_COL_WINDOW_WIDTH; localColumn++) {
                            if(rowBits & (1u << localColumn)) {
                                sum += groupX[localColumn * SWTCF_FEATURE_DIM + feature];
                            }
                        }
                        const IndexType rowSlot = tileRowSlot[static_cast<std::size_t>(tile) * SWTCF_TILE_ROWS + localRow];
                        accumulateOutput(
                            superRowBegin,
                            rowSlot,
                            feature,
                            sum,
                            superWindowRows,
                            superWindowRowCacheSlot,
                            repeatAccumulator,
                            matrixY
                        );
                    }
                }
                }
            }
            __synsthreads();
        }
        // --------------------------------------------------
        // Flush repeated rows.
        //
        // Cached partial sums have now absorbed all
        // contributions from this super-window.
        //
        // Only one global atomic per:
        //
        //     cached row x feature
        //
        // is required.
        // --------------------------------------------------
        const IndexType cacheCount = static_cast<IndexType>(superWindowCachedCount[superWindow]);
        const int flushElements = static_cast<int>(cacheCount * SWTCF_FEATURE_DIM);
        for(int index = blockThread; index < flushElements; index +=WARPS_PER_BLOCK * WARP_SIZE) {
            const IndexType cacheSlot = static_cast<IndexType>(index / SWTCF_FEATURE_DIM);
            const IndexType feature = static_cast<IndexType>(index % SWTCF_FEATURE_DIM);
            const IndexType rowSlot = superWindowCachedRowSlot[static_cast<std::size_t>(superWindow) * SWTCF_REPEAT_CACHE_SIZE + cacheSlot];
            const IndexType globalRow = superWindowRows[superRowBegin + rowSlot];
            const IndexType value = repeatAccumulator[cacheSlot * SWTCF_FEATURE_DIM + feature];
            if(value != 0.0f) {
                atomicAdd(&matrixY[static_cast<std::size_t>(globalRow) * SWTCF_FEATURE_DIM + feature], value);
            }
        }

    } 

}


} //namespace

void launchSWTCFSpMM(
    const DeviceSWTCFMatrix& matrix,
    const ValueType* matrixX,
    ValueType* matrixY,
    IndexType tcThreshold,
    cudaStream_t stream
) {
    if(
        matrix.superWindowSize != 16
    ) {
        throw std::runtime_error(
            "SWTCF SpMM V0 requires superWindowSize=16"
        );
    }

    const dim3 block(
        WARP_SIZE,
        WARPS_PER_BLOCK
    );

    const dim3 grid(
        matrix.numSuperWindows
    );

    swtcfSpmmKernel<<<
        grid,
        block,
        0,
        stream
    >>>(
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

}   //tcc