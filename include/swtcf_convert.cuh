#pragma once
#include "types.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

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

    OffsetType* superWindowRowOffset = nullptr;
    IndexType* superWindowRows = nullptr;

    std::uint8_t* superWindowRowCacheSlot = nullptr;
    IndexType* superWindowCachedRowSlot = nullptr;
    std::uint8_t* superWindowCachedCount = nullptr;

    OffsetType* colWindowOffset = nullptr;
    IndexType* tileRowSlot = nullptr;
    BitmapType* tileLocalBit = nullptr;
};

struct SWTCFConvertWorkspace {
    IndexType edgeCapacity = 0;
    IndexType numColWindows = 0;
    IndexType numSuperWindows = 0;
    IndexType hashCapacity = 0;

    std::uint64_t* edgeKeysIn = nullptr;
    std::uint64_t* edgeKeysOut = nullptr;
    IndexType* edgeLocalColIn = nullptr;
    IndexType* edgeLocalColOut = nullptr;

    std::uint64_t* uniqueGroupKeys = nullptr;
    OffsetType* groupCounts = nullptr;
    OffsetType* groupEdgeOffset = nullptr;
    IndexType* deviceNumGroups = nullptr;

    OffsetType* windowDegreeCount = nullptr;
    OffsetType* windowDegreeBase = nullptr;
    OffsetType* windowDegreeCursor = nullptr;
    OffsetType* windowActiveRows = nullptr;
    OffsetType* windowTileCount = nullptr;

    IndexType* groupLocation = nullptr;
    IndexType* groupSlot = nullptr;

    std::uint64_t* superHashKeys = nullptr;
    IndexType* superHashSlot = nullptr;
    IndexType* superHashCount = nullptr;
    OffsetType* superWindowUniqueCount = nullptr;
    IndexType* superWindowRowOccurrence = nullptr;

    IndexType* deviceError = nullptr;

    void* cubTemp = nullptr;
    std::size_t cubTempBytes = 0;
};

void allocateDeviceSWTCF(DeviceSWTCFMatrix& matrix, IndexType rows, IndexType cols, IndexType nnz, IndexType superWindowSize);
void freeDeviceSWTCF(DeviceSWTCFMatrix& matrix);
void allocateSWTCFConvertWorkspace(SWTCFConvertWorkspace& workspace, IndexType edgeCapacity, IndexType numSrc, IndexType superWindowSize);
void freeSWTCFConvertWorkspace(SWTCFConvertWorkspace& workspace);
void convertCOOToSWTCF(const IndexType* deviceDst, const IndexType* deviceSrc, IndexType numEdges, DeviceSWTCFMatrix& output, SWTCFConvertWorkspace& workspace, cudaStream_t stream = 0);