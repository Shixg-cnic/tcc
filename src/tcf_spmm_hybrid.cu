#include "tcf_spmm_hybrid.cuh"
#include "mma_tf32.cuh"

namespace tcc{
namespace{

constexpr int TILE_ROWS = 8;
constexpr int COL_WINDOW_WIDTH = 8;
constexpr int FEATURES_PER_MMA = 16;
constexpr int FEATURES_PER_WARP = 32;

__global__ void tcfSpmmHybridKernel(
    const OffsetType* colWindowOffset,
    const OffsetType* tileOffset,
    const IndexType* tileRowIndices,
    const BitmapType* tileLocalBit,
    const ValueType* values,
    const ValueType* matrixX,
    ValueType* matrixY,
    IndexType rows,
    IndexType cols,
    IndexType featureDim
) {
    const IndexType colWindow = blockIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int group = lane >> 2;
    const int threadInGroup = lane & 3;

    const IndexType columnBase = colWindow * COL_WINDOW_WIDTH;
    const IndexType featureBase = warp * FEATURES_PER_WARP;

    const IndexType feature = featureBase + lane;

    float mmaA0[4];
    float mmaA1[4];
    const IndexType column0 = columnBase + threadInGroup;
    const IndexType column1 = columnBase + threadInGroup + 4;
    mmaA0[0] = column0 < cols ? matrixX[column0 * featureDim + featureBase + group] : 0.0f;
    mmaA0[1] = column0 < cols ? matrixX[column0 * featureDim + featureBase + group + 8] : 0.0f;
    mmaA0[2] = column1 < cols ? matrixX[column1 * featureDim + featureBase + group] : 0.0f;
    mmaA0[3] = column1 < cols ? matrixX[column1 * featureDim + featureBase + group + 8] : 0.0f;
    mmaA1[0] = column0 < cols ? matrixX[column0 * featureDim + featureBase + FEATURES_PER_MMA + group] : 0.0f;
    mmaA1[1] = column0 < cols ? matrixX[column0 * featureDim + featureBase + FEATURES_PER_MMA + group + 8] : 0.0f;
    mmaA1[2] = column1 < cols ? matrixX[column1 * featureDim + featureBase + FEATURES_PER_MMA + group] : 0.0f;
    mmaA1[3] = column1 < cols ? matrixX[column1 * featureDim + featureBase + FEATURES_PER_MMA + group + 8] : 0.0f;

    __shared__ ValueType sparseTile[TILE_ROWS * COL_WINDOW_WIDTH];
    const int thread = threadIdx.y * blockDim.x + threadIdx.x;
    const OffsetType tileBegin = colWindowOffset[colWindow];
    const OffsetType tileEnd = colWindowOffset[colWindow + 1];
    
    for(OffsetType tile = tileBegin; tile < tileEnd; tile++) {
        const BitmapType bitmap = tileLocalBit[tile];
        const OffsetType valueBegin = tileOffset[tile];
        const IndexType tileNnz = static_cast<IndexType>(__popcll(static_cast<unsigned long long>(bitmap)));
        if(tileNnz >= HYBRID_TC_THRESHOLD) {
            if(thread < TILE_ROWS * COL_WINDOW_WIDTH) {
                const BitmapType currentBit = BitmapType{1} << thread;
                if(bitmap & currentBit) {
                    const BitmapType lowerBits = bitmap & (currentBit - 1);
                    const OffsetType valueIndex = valueBegin + static_cast<OffsetType>(__popcll(static_cast<unsigned long long>(lowerBits)));
                    sparseTile[thread] = values[valueIndex];
                } else {
                    sparseTile[thread] = 0.0f;
                }
            }
            __syncthreads();
            float mmaB[2];
            float mmaC0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float mmaC1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mmaB[0] = sparseTile[group * COL_WINDOW_WIDTH + threadInGroup];
            mmaB[1] = sparseTile[group * COL_WINDOW_WIDTH + threadInGroup + 4];
            mmaTf32M16N8K8(mmaA0,mmaB,mmaC0);
            mmaTf32M16N8K8(mmaA1,mmaB,mmaC1);

            const IndexType localRow0 = threadInGroup * 2;
            const IndexType localRow1 = threadInGroup * 2 + 1;
            const IndexType globalRow0 = tileRowIndices[tile * TILE_ROWS + localRow0];
            const IndexType globalRow1 = tileRowIndices[tile * TILE_ROWS + localRow1];
            const IndexType feature0 = featureBase + group;
            const IndexType feature1 = featureBase + group + 8;
            const IndexType feature2 = featureBase + group + 16;
            const IndexType feature3 = featureBase + group + 24;

            if(globalRow0 < rows) {
                atomicAdd(&matrixY[globalRow0 * featureDim + feature0], mmaC0[0]);
                atomicAdd(&matrixY[globalRow0 * featureDim + feature1], mmaC0[2]);
                atomicAdd(&matrixY[globalRow0 * featureDim + feature2], mmaC1[0]);
                atomicAdd(&matrixY[globalRow0 * featureDim + feature3], mmaC1[2]);
            }
            if(globalRow1 < rows){
                atomicAdd(&matrixY[globalRow1 * featureDim + feature0], mmaC0[1]);
                atomicAdd(&matrixY[globalRow1 * featureDim + feature1], mmaC0[3]);
                atomicAdd(&matrixY[globalRow1 * featureDim + feature2], mmaC1[1]);
                atomicAdd(&matrixY[globalRow1 * featureDim + feature3], mmaC1[3]);
            }
            __syncthreads();
        }else{
            if(feature < featureDim) {
                for(IndexType localRow = 0; localRow < TILE_ROWS; localRow++){
                    const IndexType globalRow = tileRowIndices[tile * TILE_ROWS + localRow];
                    if(globalRow >= rows) { continue; }
                    ValueType sum = 0.0f;
                    for(IndexType localCol = 0; localCol < COL_WINDOW_WIDTH; localCol++) {
                        const IndexType globalCol = columnBase + localCol;
                        if(globalCol >= cols) { continue; }
                        const IndexType bitPosition = localRow * COL_WINDOW_WIDTH + localCol;
                        const BitmapType currentBit = BitmapType{1}<< bitPosition;
                        if(bitmap & currentBit) {
                            const BitmapType lowerBits = bitmap & (currentBit - 1);
                            const OffsetType valueIndex = valueBegin + static_cast<OffsetType>(__popcll(static_cast<unsigned long long>(lowerBits)));
                            const ValueType a = values[valueIndex];
                            const ValueType x =matrixX[globalCol * featureDim + feature];
                            sum += a * x;
                        }
                    }
                    atomicAdd(&matrixY[globalRow * featureDim + feature], sum);
                }
            }
        }
    }
}
} //namespace
void launchTcfSpMMHybrid(
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
    cudaStream_t stream
) {
    const dim3 block(32,4);
    const dim3 grid(numColWindows);
    tcfSpmmHybridKernel<<<grid, block, 0, stream>>>(
        colWindowOffset,
        tileOffset,
        tileRowIndices,
        tileLocalBit,
        values,
        matrixX,
        matrixY,
        rows,
        cols,
        featureDim
    );
}

} // tcc