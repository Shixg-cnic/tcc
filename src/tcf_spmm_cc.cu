#include "tcf_spmm_cc.cuh"

namespace tcc{

namespace{
    
constexpr int TILE_ROWS = 8;
constexpr int COL_WINDOW_WIDTH = 8;
constexpr int FEATURES_PER_WARP = 32;

__global__ void tcfSpmmCCKernel(
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
){
    const IndexType colWindow = blockIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    
    const IndexType columnBase = colWindow * COL_WINDOW_WIDTH;
    const IndexType feature = warp * FEATURES_PER_WARP + lane;
    const OffsetType tileBegin = colWindowOffset[colWindow];
    const OffsetType tileEnd = colWindowOffset[colWindow + 1];
    
    if(feature >= featureDim) { return;}
    
    ValueType xFragment[COL_WINDOW_WIDTH];
    #pragma unroll
    for(IndexType localCol = 0; localCol < COL_WINDOW_WIDTH; localCol++) {
        const IndexType globalCol = columnBase + localCol;
        if(globalCol < cols) {
            xFragment[localCol] = matrixX[globalCol * featureDim + feature];
        }else{
            xFragment[localCol] = 0.0f;
        }
    }

    for(OffsetType tile = tileBegin; tile < tileEnd; tile++){
        const BitmapType bitmap = tileLocalBit[tile];
        const OffsetType valueBegin = tileOffset[tile];
        #pragma unroll
        for(IndexType localRow = 0; localRow < TILE_ROWS; localRow++){
            const IndexType globalRow = tileRowIndices[tile * TILE_ROWS + localRow];
            if(globalRow >= rows){ continue; }
            ValueType sum = 0.0f;

            #pragma unroll
            for(IndexType localCol = 0; localCol < COL_WINDOW_WIDTH; localCol++) {
                const IndexType bitPosition = localRow * COL_WINDOW_WIDTH + localCol;
                const BitmapType currentBit = BitmapType{1} << bitPosition;
                if(bitmap & currentBit) {
                    const BitmapType lowerBits = bitmap & (currentBit - 1);
                    const OffsetType valueIndex = valueBegin + static_cast<OffsetType>(__popcll(static_cast<unsigned long long>(lowerBits)));
                    const ValueType a = values[valueIndex];
                    sum += a * xFragment[localCol];
                }
            }
            atomicAdd(&matrixY[globalRow * featureDim +feature], sum);
        }
    }

    // base cc implement
    // for(OffsetType tile = tileBegin; tile < tileEnd; tile++){
    //     const BitmapType bitmap = tileLocalBit[tile];
    //     const OffsetType valueBegin = tileOffset[tile];
    //     for(IndexType localRow = 0; localRow < TILE_ROWS; localRow++){
    //         const IndexType globalRow = tileRowIndices[tile * TILE_ROWS + localRow];
    //         if(globalRow >= rows){ continue; }
    //         ValueType sum = 0.0f;
    //         for(IndexType localCol = 0; localCol < COL_WINDOW_WIDTH; localCol++){
    //             const IndexType globalCol = columnBase + localCol;
    //             if(globalCol >= cols) { continue; }
    //             const IndexType bitPosition = localRow * COL_WINDOW_WIDTH + localCol;
    //             const BitmapType currentBit = BitmapType{1} << bitPosition;
    //             if(bitmap & currentBit) {
    //                 const BitmapType lowerBits = bitmap & (currentBit -1);
    //                 const OffsetType valueIndex = valueBegin + static_cast<OffsetType>(__popcll(static_cast<unsigned long long>(lowerBits)));
    //                 const ValueType a = values[valueIndex];
    //                 const ValueType x = matrixX[globalCol * featureDim + feature];
    //                 sum += a * x;
                    
    //             }
    //         }
    //         atomicAdd(&matrixY[globalRow * featureDim + feature], sum);
    //     }
    // }
}

} //namespace

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
    cudaStream_t stream
) {
    const int warpsPerBlock = (featureDim + FEATURES_PER_WARP - 1) / FEATURES_PER_WARP;
    const dim3 block(32, warpsPerBlock);
    const dim3 grid(numColWindows);
    tcfSpmmCCKernel<<<grid,block,0,stream>>>(
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

}// namespace tcc