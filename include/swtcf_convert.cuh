#pragma once

#include "types.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace tcc {

constexpr IndexType SWTCF_COL_WINDOW_WIDTH = 8;
constexpr IndexType SWTCF_TILE_ROWS = 8;

constexpr IndexType SWTCF_REPEAT_CACHE_SIZE = 32;
constexpr IndexType SWTCF_MAX_SUPER_WINDOW_SIZE = 32;

constexpr std::uint8_t SWTCF_INVALID_CACHE_SLOT = 0xFFu;


struct DeviceSWTCFMatrix {
    IndexType rows = 0;
    IndexType cols = 0;
    IndexType nnz = 0;

    IndexType colWindowWidth = SWTCF_COL_WINDOW_WIDTH;
    IndexType tileRows = SWTCF_TILE_ROWS;
    IndexType superWindowSize = 0;

    IndexType numColWindows = 0;
    IndexType numSuperWindows = 0;

    // --------------------------------------------------
    // SW row metadata
    // --------------------------------------------------

    // [numSuperWindows + 1]
    OffsetType* superWindowRowOffset = nullptr;

    // capacity = nnz
    //
    // superWindowRows[
    //     superWindowRowOffset[sw] + rowSlot
    // ]
    //
    // gives global dst row.
    IndexType* superWindowRows = nullptr;

    // --------------------------------------------------
    // repeated-row cache metadata
    // --------------------------------------------------

    // One byte for every element of superWindowRows.
    //
    // 0xFF:
    //     this row is not cached.
    //
    // 0..31:
    //     use repeatAccumulator[cacheSlot].
    std::uint8_t* superWindowRowCacheSlot = nullptr;

    // Fixed 32 entries for every super-window.
    //
    // superWindowCachedRowSlot[
    //     sw * 32 + cacheSlot
    // ]
    //
    // gives rowSlot inside this super-window.
    IndexType* superWindowCachedRowSlot = nullptr;

    // [numSuperWindows]
    //
    // Number of valid entries in
    // superWindowCachedRowSlot for each SW.
    std::uint8_t* superWindowCachedCount = nullptr;

    // --------------------------------------------------
    // ColWindow / tile metadata
    // --------------------------------------------------

    // [numColWindows + 1]
    OffsetType* colWindowOffset = nullptr;

    // capacity = nnz * 8
    IndexType* tileRowSlot = nullptr;

    // capacity = nnz
    BitmapType* tileLocalBit = nullptr;
};


struct SWTCFConvertWorkspace {
    IndexType edgeCapacity = 0;

    IndexType numColWindows = 0;
    IndexType numSuperWindows = 0;

    IndexType hashCapacity = 0;

    // --------------------------------------------------
    // Edge sorting
    //
    // key:
    //     (colWindow << 32) | dst
    //
    // value:
    //     local column 0..7
    // --------------------------------------------------

    std::uint64_t* edgeKeysIn = nullptr;
    std::uint64_t* edgeKeysOut = nullptr;

    IndexType* edgeLocalColIn = nullptr;
    IndexType* edgeLocalColOut = nullptr;

    // --------------------------------------------------
    // One group = one (colWindow, dst)
    // --------------------------------------------------

    std::uint64_t* uniqueGroupKeys = nullptr;

    OffsetType* groupCounts = nullptr;
    OffsetType* groupEdgeOffset = nullptr;

    IndexType* deviceNumGroups = nullptr;

    // --------------------------------------------------
    // Degree bucket reorder
    // --------------------------------------------------

    OffsetType* windowDegreeCount = nullptr;
    OffsetType* windowDegreeBase = nullptr;
    OffsetType* windowDegreeCursor = nullptr;

    OffsetType* windowActiveRows = nullptr;
    OffsetType* windowTileCount = nullptr;

    // packed:
    //
    // (tile << 3) | localRow
    IndexType* groupLocation = nullptr;

    // group -> super-window row slot
    IndexType* groupSlot = nullptr; 

    // --------------------------------------------------
    // Hash table:
    //
    // key = (superWindow << 32) | dst
    // --------------------------------------------------

    std::uint64_t* superHashKeys = nullptr;

    // slot inside superWindowRows
    IndexType* superHashSlot = nullptr;

    // Number of different colWindows in this SW
    // containing this dst row.
    IndexType* superHashCount = nullptr;

    OffsetType* superWindowUniqueCount = nullptr;

    // Temporary occurrence count aligned with
    // superWindowRows.
    IndexType* superWindowRowOccurrence = nullptr;

    // device-side error flag
    IndexType* deviceError = nullptr;

    // CUB temporary storage
    void* cubTemp = nullptr;
    std::size_t cubTempBytes = 0;
};


void allocateDeviceSWTCF(
    DeviceSWTCFMatrix& matrix,
    IndexType rows,
    IndexType cols,
    IndexType nnz,
    IndexType superWindowSize
);


void freeDeviceSWTCF(
    DeviceSWTCFMatrix& matrix
);


void allocateSWTCFConvertWorkspace(
    SWTCFConvertWorkspace& workspace,
    IndexType edgeCapacity,
    IndexType numSrc,
    IndexType superWindowSize
);


void freeSWTCFConvertWorkspace(
    SWTCFConvertWorkspace& workspace
);


void convertCOOToSWTCF(
    const IndexType* deviceDst,
    const IndexType* deviceSrc,
    IndexType numEdges,
    DeviceSWTCFMatrix& output,
    SWTCFConvertWorkspace& workspace,
    cudaStream_t stream = 0
);

}  // namespace tcc