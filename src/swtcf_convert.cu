#include "swtcf_convert.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>


namespace tcc {
namespace {

constexpr int THREADS = 256;

constexpr std::uint64_t EMPTY_HASH_KEY =
    std::numeric_limits<std::uint64_t>::max();

constexpr IndexType INVALID_ROW_SLOT =
    std::numeric_limits<IndexType>::max();


#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        const cudaError_t error__ = (call);                                 \
        if(error__ != cudaSuccess) {                                        \
            throw std::runtime_error(                                       \
                std::string("CUDA error: ") +                               \
                cudaGetErrorString(error__)                                 \
            );                                                              \
        }                                                                   \
    } while(0)


inline int blockCount(
    std::uint64_t count
) {
    return static_cast<int>(
        (count + THREADS - 1) / THREADS
    );
}


inline IndexType nextPowerOfTwo(
    std::uint64_t value
) {
    if(value <= 1) {
        return 1;
    }

    --value;

    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;

    ++value;

    if(
        value >
        static_cast<std::uint64_t>(
            std::numeric_limits<IndexType>::max()
        )
    ) {
        throw std::runtime_error(
            "hash capacity exceeds IndexType range"
        );
    }

    return static_cast<IndexType>(
        value
    );
}


__device__ __forceinline__
IndexType hashPosition(
    std::uint64_t key,
    IndexType mask
) {
    key ^= key >> 33;

    key *=
        0xff51afd7ed558ccdULL;

    key ^= key >> 33;

    key *=
        0xc4ceb9fe1a85ec53ULL;

    key ^= key >> 33;

    return
        static_cast<IndexType>(key)
        &
        mask;
}


__global__ void buildEdgeKeysKernel(
    const IndexType* __restrict__ dst,
    const IndexType* __restrict__ src,
    IndexType numEdges,
    std::uint64_t* __restrict__ keys,
    IndexType* __restrict__ localColumns
) {
    const IndexType edge =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(edge >= numEdges) {
        return;
    }

    const IndexType source =
        src[edge];

    const IndexType destination =
        dst[edge];

    const IndexType window =
        source >> 3;

    const IndexType localColumn =
        source & 7u;

    keys[edge] =
        (
            static_cast<std::uint64_t>(
                window
            ) << 32
        )
        |
        static_cast<std::uint64_t>(
            destination
        );

    localColumns[edge] =
        localColumn;
}


__global__ void countWindowDegreesKernel(
    const std::uint64_t* __restrict__ groupKeys,
    const OffsetType* __restrict__ groupCounts,
    const IndexType* __restrict__ deviceNumGroups,
    IndexType groupCapacity,
    OffsetType* __restrict__ windowDegreeCount,
    IndexType* __restrict__ deviceError
) {
    __shared__ IndexType numGroups;

    if(threadIdx.x == 0) {
        numGroups =
            *deviceNumGroups;
    }

    __syncthreads();

    const IndexType group =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(
        group >= groupCapacity ||
        group >= numGroups
    ) {
        return;
    }

    const std::uint64_t key =
        groupKeys[group];

    const IndexType window =
        static_cast<IndexType>(
            key >> 32
        );

    const OffsetType degree =
        groupCounts[group];

    if(
        degree == 0 ||
        degree > SWTCF_COL_WINDOW_WIDTH
    ) {
        atomicExch(
            deviceError,
            IndexType{1}
        );

        return;
    }

    const IndexType degreeIndex =
        static_cast<IndexType>(
            degree - 1
        );

    atomicAdd(
        &windowDegreeCount[
            static_cast<std::size_t>(
                window
            ) * 8
            +
            degreeIndex
        ],
        OffsetType{1}
    );
}


__global__ void buildWindowMetadataKernel(
    const OffsetType* __restrict__ windowDegreeCount,
    OffsetType* __restrict__ windowDegreeBase,
    OffsetType* __restrict__ windowActiveRows,
    OffsetType* __restrict__ windowTileCount,
    IndexType numColWindows
) {
    const IndexType window =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(window >= numColWindows) {
        return;
    }

    OffsetType running = 0;

    // degree 8 first, then 7 ... 1
    for(
        int degreeIndex = 7;
        degreeIndex >= 0;
        --degreeIndex
    ) {
        const std::size_t index =
            static_cast<std::size_t>(
                window
            ) * 8
            +
            static_cast<std::size_t>(
                degreeIndex
            );

        windowDegreeBase[index] =
            running;

        running +=
            windowDegreeCount[index];
    }

    windowActiveRows[window] =
        running;

    windowTileCount[window] =
        (
            running +
            SWTCF_TILE_ROWS - 1
        )
        /
        SWTCF_TILE_ROWS;
}


__global__ void setOffsetTailKernel(
    OffsetType* __restrict__ offsets,
    const OffsetType* __restrict__ counts,
    IndexType count
) {
    if(
        blockIdx.x != 0 ||
        threadIdx.x != 0
    ) {
        return;
    }

    if(count == 0) {
        offsets[0] = 0;
        return;
    }

    offsets[count] =
        offsets[count - 1]
        +
        counts[count - 1];
}


__global__ void assignGroupLocationsKernel(
    const std::uint64_t* __restrict__ groupKeys,
    const OffsetType* __restrict__ groupCounts,
    const IndexType* __restrict__ deviceNumGroups,
    IndexType groupCapacity,
    const OffsetType* __restrict__ windowDegreeBase,
    OffsetType* __restrict__ windowDegreeCursor,
    const OffsetType* __restrict__ colWindowOffset,
    IndexType* __restrict__ groupLocation
) {
    __shared__ IndexType numGroups;

    if(threadIdx.x == 0) {
        numGroups =
            *deviceNumGroups;
    }

    __syncthreads();

    const IndexType group =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(
        group >= groupCapacity ||
        group >= numGroups
    ) {
        return;
    }

    const std::uint64_t key =
        groupKeys[group];

    const IndexType window =
        static_cast<IndexType>(
            key >> 32
        );

    const OffsetType degree =
        groupCounts[group];

    const IndexType degreeIndex =
        static_cast<IndexType>(
            degree - 1
        );

    const std::size_t bucket =
        static_cast<std::size_t>(
            window
        ) * 8
        +
        degreeIndex;

    const OffsetType rankInBucket =
        atomicAdd(
            &windowDegreeCursor[bucket],
            OffsetType{1}
        );

    const OffsetType reorderedRow =
        windowDegreeBase[bucket]
        +
        rankInBucket;

    const OffsetType tile =
        colWindowOffset[window]
        +
        reorderedRow / SWTCF_TILE_ROWS;

    const IndexType localRow =
        static_cast<IndexType>(
            reorderedRow
            &
            (SWTCF_TILE_ROWS - 1)
        );

    groupLocation[group] =
        (
            static_cast<IndexType>(
                tile
            ) << 3
        )
        |
        localRow;
}


__global__ void buildTileBitmapsKernel(
    const IndexType* __restrict__ sortedLocalColumns,
    const OffsetType* __restrict__ groupCounts,
    const OffsetType* __restrict__ groupEdgeOffset,
    const IndexType* __restrict__ groupLocation,
    const IndexType* __restrict__ deviceNumGroups,
    IndexType groupCapacity,
    BitmapType* __restrict__ tileLocalBit
) {
    __shared__ IndexType numGroups;

    if(threadIdx.x == 0) {
        numGroups =
            *deviceNumGroups;
    }

    __syncthreads();

    const IndexType group =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(
        group >= groupCapacity ||
        group >= numGroups
    ) {
        return;
    }

    const OffsetType edgeBegin =
        groupEdgeOffset[group];

    const OffsetType degree =
        groupCounts[group];

    unsigned int rowBits = 0;

    for(
        OffsetType offset = 0;
        offset < degree;
        ++offset
    ) {
        const IndexType localColumn =
            sortedLocalColumns[
                edgeBegin + offset
            ];

        rowBits |=
            1u << localColumn;
    }

    const IndexType location =
        groupLocation[group];

    const IndexType tile =
        location >> 3;

    const IndexType localRow =
        location & 7u;

    const unsigned long long bits =
        static_cast<unsigned long long>(
            rowBits
        )
        <<
        (
            localRow * 8
        );

    atomicOr(
        reinterpret_cast<
            unsigned long long*
        >(
            &tileLocalBit[tile]
        ),
        bits
    );
}


__global__ void insertSuperWindowKeysKernel(
    const std::uint64_t* __restrict__ groupKeys,
    const IndexType* __restrict__ deviceNumGroups,
    IndexType groupCapacity,
    IndexType superWindowSize,
    std::uint64_t* __restrict__ hashKeys,
    IndexType* __restrict__ hashCount,
    IndexType hashCapacity,
    IndexType* __restrict__ deviceError
) {
    __shared__ IndexType numGroups;

    if(threadIdx.x == 0) {
        numGroups =
            *deviceNumGroups;
    }

    __syncthreads();

    const IndexType group =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(
        group >= groupCapacity ||
        group >= numGroups
    ) {
        return;
    }

    const std::uint64_t groupKey =
        groupKeys[group];

    const IndexType window =
        static_cast<IndexType>(
            groupKey >> 32
        );

    const IndexType dst =
        static_cast<IndexType>(
            groupKey
        );

    const IndexType superWindow =
        window /
        superWindowSize;

    const std::uint64_t key =
        (
            static_cast<std::uint64_t>(
                superWindow
            ) << 32
        )
        |
        static_cast<std::uint64_t>(
            dst
        );

    const IndexType mask =
        hashCapacity - 1;

    IndexType position =
        hashPosition(
            key,
            mask
        );

    for(
        IndexType probe = 0;
        probe < hashCapacity;
        ++probe
    ) {
        const unsigned long long old =
            atomicCAS(
                reinterpret_cast<
                    unsigned long long*
                >(
                    &hashKeys[position]
                ),
                static_cast<
                    unsigned long long
                >(
                    EMPTY_HASH_KEY
                ),
                static_cast<
                    unsigned long long
                >(
                    key
                )
            );

        if(
            old ==
                static_cast<
                    unsigned long long
                >(
                    EMPTY_HASH_KEY
                )
            ||
            old ==
                static_cast<
                    unsigned long long
                >(
                    key
                )
        ) {
            // One group means this dst appears in one
            // colWindow. Therefore this count is exactly
            // the cross-window occurrence count.
            atomicAdd(
                &hashCount[position],
                IndexType{1}
            );

            return;
        }

        position =
            (position + 1)
            &
            mask;
    }

    atomicExch(
        deviceError,
        IndexType{2}
    );
}


__global__ void assignSuperWindowSlotsKernel(
    const std::uint64_t* __restrict__ hashKeys,
    IndexType* __restrict__ hashSlot,
    IndexType hashCapacity,
    OffsetType* __restrict__ superWindowUniqueCount
) {
    const IndexType position =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(position >= hashCapacity) {
        return;
    }

    const std::uint64_t key =
        hashKeys[position];

    if(key == EMPTY_HASH_KEY) {
        return;
    }

    const IndexType superWindow =
        static_cast<IndexType>(
            key >> 32
        );

    const IndexType slot =
        atomicAdd(
            &superWindowUniqueCount[
                superWindow
            ],
            OffsetType{1}
        );

    hashSlot[position] =
        slot;
}


__global__ void fillSuperWindowRowsKernel(
    const std::uint64_t* __restrict__ hashKeys,
    const IndexType* __restrict__ hashSlot,
    const IndexType* __restrict__ hashCount,
    IndexType hashCapacity,
    const OffsetType* __restrict__ superWindowRowOffset,
    IndexType* __restrict__ superWindowRows,
    IndexType* __restrict__ superWindowRowOccurrence
) {
    const IndexType position =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(position >= hashCapacity) {
        return;
    }

    const std::uint64_t key =
        hashKeys[position];

    if(key == EMPTY_HASH_KEY) {
        return;
    }

    const IndexType superWindow =
        static_cast<IndexType>(
            key >> 32
        );

    const IndexType dst =
        static_cast<IndexType>(
            key
        );

    const IndexType slot =
        hashSlot[position];

    const OffsetType outputIndex =
        superWindowRowOffset[
            superWindow
        ]
        +
        slot;

    superWindowRows[
        outputIndex
    ] =
        dst;

    superWindowRowOccurrence[
        outputIndex
    ] =
        hashCount[position];
}


// ------------------------------------------------------
// One CTA handles one super-window.
//
// occurrence count is in [1, superWindowSize].
//
// Instead of sorting rows, use tiny occurrence buckets:
//
// count=S rows first
// count=S-1 rows next
// ...
// count=2 rows
//
// Only first 32 repeated rows get cache slots.
// ------------------------------------------------------

__global__ void buildRepeatCacheMetadataKernel(
    const OffsetType* __restrict__ superWindowRowOffset,
    const IndexType* __restrict__ superWindowRowOccurrence,
    IndexType numSuperWindows,
    std::uint8_t* __restrict__ superWindowRowCacheSlot,
    IndexType* __restrict__ superWindowCachedRowSlot,
    std::uint8_t* __restrict__ superWindowCachedCount,
    IndexType* __restrict__ deviceError
) {
    const IndexType superWindow =
        blockIdx.x;

    if(superWindow >= numSuperWindows) {
        return;
    }

    __shared__ OffsetType
        bucketCount[
            SWTCF_MAX_SUPER_WINDOW_SIZE + 1
        ];

    __shared__ OffsetType
        bucketBase[
            SWTCF_MAX_SUPER_WINDOW_SIZE + 1
        ];

    __shared__ OffsetType
        bucketCursor[
            SWTCF_MAX_SUPER_WINDOW_SIZE + 1
        ];

    for(
        IndexType i = threadIdx.x;
        i <= SWTCF_MAX_SUPER_WINDOW_SIZE;
        i += blockDim.x
    ) {
        bucketCount[i] = 0;
        bucketBase[i] = 0;
        bucketCursor[i] = 0;
    }

    __syncthreads();

    const OffsetType rowBegin =
        superWindowRowOffset[
            superWindow
        ];

    const OffsetType rowEnd =
        superWindowRowOffset[
            superWindow + 1
        ];

    // Count rows for every occurrence bucket.
    for(
        OffsetType index =
            rowBegin +
            threadIdx.x;
        index < rowEnd;
        index += blockDim.x
    ) {
        const IndexType occurrence =
            superWindowRowOccurrence[
                index
            ];

        if(
            occurrence >
            SWTCF_MAX_SUPER_WINDOW_SIZE
        ) {
            atomicExch(
                deviceError,
                IndexType{4}
            );

            continue;
        }

        if(occurrence > 1) {
            atomicAdd(
                &bucketCount[
                    occurrence
                ],
                OffsetType{1}
            );
        }
    }

    __syncthreads();

    // Build descending bucket bases.
    if(threadIdx.x == 0) {
        OffsetType running = 0;

        for(
            int occurrence =
                static_cast<int>(
                    SWTCF_MAX_SUPER_WINDOW_SIZE
                );
            occurrence >= 2;
            --occurrence
        ) {
            bucketBase[
                occurrence
            ] =
                running;

            running +=
                bucketCount[
                    occurrence
                ];
        }

        const OffsetType cached =
            running <
                SWTCF_REPEAT_CACHE_SIZE
            ?
                running
            :
                SWTCF_REPEAT_CACHE_SIZE;

        superWindowCachedCount[
            superWindow
        ] =
            static_cast<std::uint8_t>(
                cached
            );
    }

    __syncthreads();

    // Assign cache slots.
    for(
        OffsetType index =
            rowBegin +
            threadIdx.x;
        index < rowEnd;
        index += blockDim.x
    ) {
        const IndexType occurrence =
            superWindowRowOccurrence[
                index
            ];

        if(
            occurrence <= 1 ||
            occurrence >
                SWTCF_MAX_SUPER_WINDOW_SIZE
        ) {
            continue;
        }

        const OffsetType rank =
            atomicAdd(
                &bucketCursor[
                    occurrence
                ],
                OffsetType{1}
            );

        const OffsetType cacheSlot =
            bucketBase[
                occurrence
            ]
            +
            rank;

        if(
            cacheSlot <
            SWTCF_REPEAT_CACHE_SIZE
        ) {
            const IndexType rowSlot =
                static_cast<IndexType>(
                    index - rowBegin
                );

            superWindowRowCacheSlot[
                index
            ] =
                static_cast<
                    std::uint8_t
                >(
                    cacheSlot
                );

            superWindowCachedRowSlot[
                static_cast<std::size_t>(
                    superWindow
                ) *
                SWTCF_REPEAT_CACHE_SIZE
                +
                cacheSlot
            ] =
                rowSlot;
        }
    }
}


__global__ void lookupGroupSlotsKernel(
    const std::uint64_t* __restrict__ groupKeys,
    const IndexType* __restrict__ deviceNumGroups,
    IndexType groupCapacity,
    IndexType superWindowSize,
    const std::uint64_t* __restrict__ hashKeys,
    const IndexType* __restrict__ hashSlot,
    IndexType hashCapacity,
    IndexType* __restrict__ groupSlot,
    IndexType* __restrict__ deviceError
) {
    __shared__ IndexType numGroups;

    if(threadIdx.x == 0) {
        numGroups =
            *deviceNumGroups;
    }

    __syncthreads();

    const IndexType group =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(
        group >= groupCapacity ||
        group >= numGroups
    ) {
        return;
    }

    const std::uint64_t groupKey =
        groupKeys[group];

    const IndexType window =
        static_cast<IndexType>(
            groupKey >> 32
        );

    const IndexType dst =
        static_cast<IndexType>(
            groupKey
        );

    const IndexType superWindow =
        window /
        superWindowSize;

    const std::uint64_t key =
        (
            static_cast<std::uint64_t>(
                superWindow
            ) << 32
        )
        |
        static_cast<std::uint64_t>(
            dst
        );

    const IndexType mask =
        hashCapacity - 1;

    IndexType position =
        hashPosition(
            key,
            mask
        );

    for(
        IndexType probe = 0;
        probe < hashCapacity;
        ++probe
    ) {
        const std::uint64_t existing =
            hashKeys[position];

        if(existing == key) {
            groupSlot[group] =
                hashSlot[position];

            return;
        }

        if(existing == EMPTY_HASH_KEY) {
            atomicExch(
                deviceError,
                IndexType{3}
            );

            return;
        }

        position =
            (position + 1)
            &
            mask;
    }

    atomicExch(
        deviceError,
        IndexType{3}
    );
}


__global__ void fillValidTileRowSlotsKernel(
    const IndexType* __restrict__ groupLocation,
    const IndexType* __restrict__ groupSlot,
    const IndexType* __restrict__ deviceNumGroups,
    IndexType groupCapacity,
    IndexType* __restrict__ tileRowSlot
) {
    __shared__ IndexType numGroups;

    if(threadIdx.x == 0) {
        numGroups =
            *deviceNumGroups;
    }

    __syncthreads();

    const IndexType group =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(
        group >= groupCapacity ||
        group >= numGroups
    ) {
        return;
    }

    const IndexType location =
        groupLocation[group];

    const IndexType tile =
        location >> 3;

    const IndexType localRow =
        location & 7u;

    tileRowSlot[
        static_cast<std::size_t>(
            tile
        ) *
        SWTCF_TILE_ROWS
        +
        localRow
    ] =
        groupSlot[group];
}


__global__ void fillPaddingTileRowSlotsKernel(
    const OffsetType* __restrict__ windowActiveRows,
    const OffsetType* __restrict__ colWindowOffset,
    IndexType numColWindows,
    IndexType* __restrict__ tileRowSlot
) {
    const IndexType window =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(window >= numColWindows) {
        return;
    }

    const OffsetType activeRows =
        windowActiveRows[
            window
        ];

    if(activeRows == 0) {
        return;
    }

    const IndexType remainder =
        static_cast<IndexType>(
            activeRows
            &
            (SWTCF_TILE_ROWS - 1)
        );

    if(remainder == 0) {
        return;
    }

    const OffsetType lastTile =
        colWindowOffset[
            window + 1
        ] - 1;

    for(
        IndexType localRow =
            remainder;
        localRow <
            SWTCF_TILE_ROWS;
        ++localRow
    ) {
        tileRowSlot[
            static_cast<std::size_t>(
                lastTile
            ) *
            SWTCF_TILE_ROWS
            +
            localRow
        ] =
            INVALID_ROW_SLOT;
    }
}


}  // namespace


void allocateDeviceSWTCF(
    DeviceSWTCFMatrix& matrix,
    IndexType rows,
    IndexType cols,
    IndexType nnz,
    IndexType superWindowSize
) {
    if(
        cols == 0 ||
        nnz == 0 ||
        superWindowSize == 0 ||
        superWindowSize >
            SWTCF_MAX_SUPER_WINDOW_SIZE
    ) {
        throw std::runtime_error(
            "invalid SWTCF allocation size"
        );
    }

    matrix.rows =
        rows;

    matrix.cols =
        cols;

    matrix.nnz =
        nnz;

    matrix.colWindowWidth =
        SWTCF_COL_WINDOW_WIDTH;

    matrix.tileRows =
        SWTCF_TILE_ROWS;

    matrix.superWindowSize =
        superWindowSize;

    matrix.numColWindows =
        (
            cols +
            SWTCF_COL_WINDOW_WIDTH - 1
        )
        /
        SWTCF_COL_WINDOW_WIDTH;

    matrix.numSuperWindows =
        (
            matrix.numColWindows +
            superWindowSize - 1
        )
        /
        superWindowSize;

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &matrix.superWindowRowOffset
            ),
            static_cast<std::size_t>(
                matrix.numSuperWindows + 1
            ) *
            sizeof(OffsetType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &matrix.superWindowRows
            ),
            static_cast<std::size_t>(
                nnz
            ) *
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &matrix.superWindowRowCacheSlot
            ),
            static_cast<std::size_t>(
                nnz
            ) *
            sizeof(std::uint8_t)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &matrix.superWindowCachedRowSlot
            ),
            static_cast<std::size_t>(
                matrix.numSuperWindows
            ) *
            SWTCF_REPEAT_CACHE_SIZE *
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &matrix.superWindowCachedCount
            ),
            static_cast<std::size_t>(
                matrix.numSuperWindows
            ) *
            sizeof(std::uint8_t)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &matrix.colWindowOffset
            ),
            static_cast<std::size_t>(
                matrix.numColWindows + 1
            ) *
            sizeof(OffsetType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &matrix.tileRowSlot
            ),
            static_cast<std::size_t>(
                nnz
            ) *
            SWTCF_TILE_ROWS *
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &matrix.tileLocalBit
            ),
            static_cast<std::size_t>(
                nnz
            ) *
            sizeof(BitmapType)
        )
    );
}


void freeDeviceSWTCF(
    DeviceSWTCFMatrix& matrix
) {
    cudaFree(
        matrix.superWindowRowOffset
    );

    cudaFree(
        matrix.superWindowRows
    );

    cudaFree(
        matrix.superWindowRowCacheSlot
    );

    cudaFree(
        matrix.superWindowCachedRowSlot
    );

    cudaFree(
        matrix.superWindowCachedCount
    );

    cudaFree(
        matrix.colWindowOffset
    );

    cudaFree(
        matrix.tileRowSlot
    );

    cudaFree(
        matrix.tileLocalBit
    );

    matrix =
        DeviceSWTCFMatrix{};
}


void allocateSWTCFConvertWorkspace(
    SWTCFConvertWorkspace& workspace,
    IndexType edgeCapacity,
    IndexType numSrc,
    IndexType superWindowSize
) {
    if(
        edgeCapacity == 0 ||
        numSrc == 0 ||
        superWindowSize == 0 ||
        superWindowSize >
            SWTCF_MAX_SUPER_WINDOW_SIZE
    ) {
        throw std::runtime_error(
            "invalid converter workspace size"
        );
    }

    if(
        edgeCapacity >
        static_cast<IndexType>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::runtime_error(
            "CUB item count exceeds int range"
        );
    }

    workspace.edgeCapacity =
        edgeCapacity;

    workspace.numColWindows =
        (
            numSrc +
            SWTCF_COL_WINDOW_WIDTH - 1
        )
        /
        SWTCF_COL_WINDOW_WIDTH;

    workspace.numSuperWindows =
        (
            workspace.numColWindows +
            superWindowSize - 1
        )
        /
        superWindowSize;

    workspace.hashCapacity =
        nextPowerOfTwo(
            static_cast<std::uint64_t>(
                edgeCapacity
            ) *
            2ULL
        );

    const std::size_t edgeCount =
        edgeCapacity;

    const std::size_t degreeCount =
        static_cast<std::size_t>(
            workspace.numColWindows
        ) *
        8;

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.edgeKeysIn
            ),
            edgeCount *
            sizeof(std::uint64_t)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.edgeKeysOut
            ),
            edgeCount *
            sizeof(std::uint64_t)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.edgeLocalColIn
            ),
            edgeCount *
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.edgeLocalColOut
            ),
            edgeCount *
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.uniqueGroupKeys
            ),
            edgeCount *
            sizeof(std::uint64_t)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.groupCounts
            ),
            edgeCount *
            sizeof(OffsetType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.groupEdgeOffset
            ),
            edgeCount *
            sizeof(OffsetType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.deviceNumGroups
            ),
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.windowDegreeCount
            ),
            degreeCount *
            sizeof(OffsetType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.windowDegreeBase
            ),
            degreeCount *
            sizeof(OffsetType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.windowDegreeCursor
            ),
            degreeCount *
            sizeof(OffsetType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.windowActiveRows
            ),
            static_cast<std::size_t>(
                workspace.numColWindows
            ) *
            sizeof(OffsetType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.windowTileCount
            ),
            static_cast<std::size_t>(
                workspace.numColWindows
            ) *
            sizeof(OffsetType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.groupLocation
            ),
            edgeCount *
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.groupSlot
            ),
            edgeCount *
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.superHashKeys
            ),
            static_cast<std::size_t>(
                workspace.hashCapacity
            ) *
            sizeof(std::uint64_t)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.superHashSlot
            ),
            static_cast<std::size_t>(
                workspace.hashCapacity
            ) *
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.superHashCount
            ),
            static_cast<std::size_t>(
                workspace.hashCapacity
            ) *
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.superWindowUniqueCount
            ),
            static_cast<std::size_t>(
                workspace.numSuperWindows
            ) *
            sizeof(OffsetType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.superWindowRowOccurrence
            ),
            edgeCount *
            sizeof(IndexType)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &workspace.deviceError
            ),
            sizeof(IndexType)
        )
    );

    std::size_t sortBytes =
        0;

    std::size_t rleBytes =
        0;

    std::size_t scanBytes =
        0;

    CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sortBytes,
            workspace.edgeKeysIn,
            workspace.edgeKeysOut,
            workspace.edgeLocalColIn,
            workspace.edgeLocalColOut,
            static_cast<int>(
                edgeCapacity
            )
        )
    );

    CUDA_CHECK(
        cub::DeviceRunLengthEncode::Encode(
            nullptr,
            rleBytes,
            workspace.edgeKeysOut,
            workspace.uniqueGroupKeys,
            workspace.groupCounts,
            workspace.deviceNumGroups,
            static_cast<int>(
                edgeCapacity
            )
        )
    );

    CUDA_CHECK(
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            scanBytes,
            workspace.groupCounts,
            workspace.groupEdgeOffset,
            static_cast<int>(
                edgeCapacity
            )
        )
    );

    workspace.cubTempBytes =
        std::max(
            sortBytes,
            std::max(
                rleBytes,
                scanBytes
            )
        );

    CUDA_CHECK(
        cudaMalloc(
            &workspace.cubTemp,
            workspace.cubTempBytes
        )
    );
}


void freeSWTCFConvertWorkspace(
    SWTCFConvertWorkspace& workspace
) {
    cudaFree(
        workspace.edgeKeysIn
    );

    cudaFree(
        workspace.edgeKeysOut
    );

    cudaFree(
        workspace.edgeLocalColIn
    );

    cudaFree(
        workspace.edgeLocalColOut
    );

    cudaFree(
        workspace.uniqueGroupKeys
    );

    cudaFree(
        workspace.groupCounts
    );

    cudaFree(
        workspace.groupEdgeOffset
    );

    cudaFree(
        workspace.deviceNumGroups
    );

    cudaFree(
        workspace.windowDegreeCount
    );

    cudaFree(
        workspace.windowDegreeBase
    );

    cudaFree(
        workspace.windowDegreeCursor
    );

    cudaFree(
        workspace.windowActiveRows
    );

    cudaFree(
        workspace.windowTileCount
    );

    cudaFree(
        workspace.groupLocation
    );

    cudaFree(
        workspace.groupSlot
    );

    cudaFree(
        workspace.superHashKeys
    );

    cudaFree(
        workspace.superHashSlot
    );

    cudaFree(
        workspace.superHashCount
    );

    cudaFree(
        workspace.superWindowUniqueCount
    );

    cudaFree(
        workspace.superWindowRowOccurrence
    );

    cudaFree(
        workspace.deviceError
    );

    cudaFree(
        workspace.cubTemp
    );

    workspace =
        SWTCFConvertWorkspace{};
}


void convertCOOToSWTCF(
    const IndexType* deviceDst,
    const IndexType* deviceSrc,
    IndexType numEdges,
    DeviceSWTCFMatrix& output,
    SWTCFConvertWorkspace& workspace,
    cudaStream_t stream
) {
    if(
        numEdges == 0 ||
        numEdges >
            workspace.edgeCapacity ||
        output.colWindowWidth !=
            SWTCF_COL_WINDOW_WIDTH ||
        output.tileRows !=
            SWTCF_TILE_ROWS ||
        output.superWindowSize == 0 ||
        output.superWindowSize >
            SWTCF_MAX_SUPER_WINDOW_SIZE
    ) {
        throw std::runtime_error(
            "invalid COO -> SWTCF conversion arguments"
        );
    }

    const IndexType numColWindows =
        output.numColWindows;

    const IndexType numSuperWindows =
        output.numSuperWindows;

    // --------------------------------------------------
    // Reset per-conversion state.
    // --------------------------------------------------

    CUDA_CHECK(
        cudaMemsetAsync(
            workspace.groupCounts,
            0,
            static_cast<std::size_t>(
                workspace.edgeCapacity
            ) *
            sizeof(OffsetType),
            stream
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            workspace.windowDegreeCount,
            0,
            static_cast<std::size_t>(
                numColWindows
            ) *
            8 *
            sizeof(OffsetType),
            stream
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            workspace.windowDegreeCursor,
            0,
            static_cast<std::size_t>(
                numColWindows
            ) *
            8 *
            sizeof(OffsetType),
            stream
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            output.tileLocalBit,
            0,
            static_cast<std::size_t>(
                workspace.edgeCapacity
            ) *
            sizeof(BitmapType),
            stream
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            workspace.superHashKeys,
            0xFF,
            static_cast<std::size_t>(
                workspace.hashCapacity
            ) *
            sizeof(std::uint64_t),
            stream
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            workspace.superHashCount,
            0,
            static_cast<std::size_t>(
                workspace.hashCapacity
            ) *
            sizeof(IndexType),
            stream
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            workspace.superWindowUniqueCount,
            0,
            static_cast<std::size_t>(
                numSuperWindows
            ) *
            sizeof(OffsetType),
            stream
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            output.superWindowRowCacheSlot,
            0xFF,
            static_cast<std::size_t>(
                output.nnz
            ) *
            sizeof(std::uint8_t),
            stream
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            output.superWindowCachedRowSlot,
            0xFF,
            static_cast<std::size_t>(
                numSuperWindows
            ) *
            SWTCF_REPEAT_CACHE_SIZE *
            sizeof(IndexType),
            stream
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            output.superWindowCachedCount,
            0,
            static_cast<std::size_t>(
                numSuperWindows
            ) *
            sizeof(std::uint8_t),
            stream
        )
    );

    CUDA_CHECK(
        cudaMemsetAsync(
            workspace.deviceError,
            0,
            sizeof(IndexType),
            stream
        )
    );

    // --------------------------------------------------
    // 1. COO -> (colWindow,dst)
    // --------------------------------------------------

    buildEdgeKeysKernel<<<
        blockCount(
            numEdges
        ),
        THREADS,
        0,
        stream
    >>>(
        deviceDst,
        deviceSrc,
        numEdges,
        workspace.edgeKeysIn,
        workspace.edgeLocalColIn
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // --------------------------------------------------
    // 2. Global sort by (colWindow,dst)
    // --------------------------------------------------

    std::size_t tempBytes =
        workspace.cubTempBytes;

    CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            workspace.cubTemp,
            tempBytes,
            workspace.edgeKeysIn,
            workspace.edgeKeysOut,
            workspace.edgeLocalColIn,
            workspace.edgeLocalColOut,
            static_cast<int>(
                numEdges
            ),
            0,
            64,
            stream
        )
    );

    // --------------------------------------------------
    // 3. RLE:
    //
    // one run = one (colWindow,dst)
    // --------------------------------------------------

    tempBytes =
        workspace.cubTempBytes;

    CUDA_CHECK(
        cub::DeviceRunLengthEncode::Encode(
            workspace.cubTemp,
            tempBytes,
            workspace.edgeKeysOut,
            workspace.uniqueGroupKeys,
            workspace.groupCounts,
            workspace.deviceNumGroups,
            static_cast<int>(
                numEdges
            ),
            stream
        )
    );

    // Tail of groupCounts remains zero.
    tempBytes =
        workspace.cubTempBytes;

    CUDA_CHECK(
        cub::DeviceScan::ExclusiveSum(
            workspace.cubTemp,
            tempBytes,
            workspace.groupCounts,
            workspace.groupEdgeOffset,
            static_cast<int>(
                workspace.edgeCapacity
            ),
            stream
        )
    );

    // --------------------------------------------------
    // 4. Degree histogram per colWindow
    // --------------------------------------------------

    countWindowDegreesKernel<<<
        blockCount(
            workspace.edgeCapacity
        ),
        THREADS,
        0,
        stream
    >>>(
        workspace.uniqueGroupKeys,
        workspace.groupCounts,
        workspace.deviceNumGroups,
        workspace.edgeCapacity,
        workspace.windowDegreeCount,
        workspace.deviceError
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    buildWindowMetadataKernel<<<
        blockCount(
            numColWindows
        ),
        THREADS,
        0,
        stream
    >>>(
        workspace.windowDegreeCount,
        workspace.windowDegreeBase,
        workspace.windowActiveRows,
        workspace.windowTileCount,
        numColWindows
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // --------------------------------------------------
    // 5. colWindow -> tile offset
    // --------------------------------------------------

    tempBytes =
        workspace.cubTempBytes;

    CUDA_CHECK(
        cub::DeviceScan::ExclusiveSum(
            workspace.cubTemp,
            tempBytes,
            workspace.windowTileCount,
            output.colWindowOffset,
            static_cast<int>(
                numColWindows
            ),
            stream
        )
    );

    setOffsetTailKernel<<<
        1,
        1,
        0,
        stream
    >>>(
        output.colWindowOffset,
        workspace.windowTileCount,
        numColWindows
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // --------------------------------------------------
    // 6. Degree-descending row reorder
    // --------------------------------------------------

    assignGroupLocationsKernel<<<
        blockCount(
            workspace.edgeCapacity
        ),
        THREADS,
        0,
        stream
    >>>(
        workspace.uniqueGroupKeys,
        workspace.groupCounts,
        workspace.deviceNumGroups,
        workspace.edgeCapacity,
        workspace.windowDegreeBase,
        workspace.windowDegreeCursor,
        output.colWindowOffset,
        workspace.groupLocation
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // --------------------------------------------------
    // 7. Build tile bitmaps
    // --------------------------------------------------

    buildTileBitmapsKernel<<<
        blockCount(
            workspace.edgeCapacity
        ),
        THREADS,
        0,
        stream
    >>>(
        workspace.edgeLocalColOut,
        workspace.groupCounts,
        workspace.groupEdgeOffset,
        workspace.groupLocation,
        workspace.deviceNumGroups,
        workspace.edgeCapacity,
        output.tileLocalBit
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // --------------------------------------------------
    // 8. Build unique (superWindow,dst) table
    //
    // At the same time count how many colWindows contain
    // this dst inside the SW.
    // --------------------------------------------------

    insertSuperWindowKeysKernel<<<
        blockCount(
            workspace.edgeCapacity
        ),
        THREADS,
        0,
        stream
    >>>(
        workspace.uniqueGroupKeys,
        workspace.deviceNumGroups,
        workspace.edgeCapacity,
        output.superWindowSize,
        workspace.superHashKeys,
        workspace.superHashCount,
        workspace.hashCapacity,
        workspace.deviceError
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    assignSuperWindowSlotsKernel<<<
        blockCount(
            workspace.hashCapacity
        ),
        THREADS,
        0,
        stream
    >>>(
        workspace.superHashKeys,
        workspace.superHashSlot,
        workspace.hashCapacity,
        workspace.superWindowUniqueCount
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // --------------------------------------------------
    // 9. Build superWindowRowOffset
    // --------------------------------------------------

    tempBytes =
        workspace.cubTempBytes;

    CUDA_CHECK(
        cub::DeviceScan::ExclusiveSum(
            workspace.cubTemp,
            tempBytes,
            workspace.superWindowUniqueCount,
            output.superWindowRowOffset,
            static_cast<int>(
                numSuperWindows
            ),
            stream
        )
    );

    setOffsetTailKernel<<<
        1,
        1,
        0,
        stream
    >>>(
        output.superWindowRowOffset,
        workspace.superWindowUniqueCount,
        numSuperWindows
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // --------------------------------------------------
    // 10. Fill:
    //
    // superWindowRows
    // occurrence count
    // --------------------------------------------------

    fillSuperWindowRowsKernel<<<
        blockCount(
            workspace.hashCapacity
        ),
        THREADS,
        0,
        stream
    >>>(
        workspace.superHashKeys,
        workspace.superHashSlot,
        workspace.superHashCount,
        workspace.hashCapacity,
        output.superWindowRowOffset,
        output.superWindowRows,
        workspace.superWindowRowOccurrence
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // --------------------------------------------------
    // 11. Build repeated-row cache metadata.
    //
    // One CTA per super-window.
    //
    // Highest occurrence rows receive cache slots first.
    // --------------------------------------------------

    buildRepeatCacheMetadataKernel<<<
        numSuperWindows,
        THREADS,
        0,
        stream
    >>>(
        output.superWindowRowOffset,
        workspace.superWindowRowOccurrence,
        numSuperWindows,
        output.superWindowRowCacheSlot,
        output.superWindowCachedRowSlot,
        output.superWindowCachedCount,
        workspace.deviceError
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // --------------------------------------------------
    // 12. group -> SW-local row slot
    // --------------------------------------------------

    lookupGroupSlotsKernel<<<
        blockCount(
            workspace.edgeCapacity
        ),
        THREADS,
        0,
        stream
    >>>(
        workspace.uniqueGroupKeys,
        workspace.deviceNumGroups,
        workspace.edgeCapacity,
        output.superWindowSize,
        workspace.superHashKeys,
        workspace.superHashSlot,
        workspace.hashCapacity,
        workspace.groupSlot,
        workspace.deviceError
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    // --------------------------------------------------
    // 13. tile row -> SW-local row slot
    // --------------------------------------------------

    fillValidTileRowSlotsKernel<<<
        blockCount(
            workspace.edgeCapacity
        ),
        THREADS,
        0,
        stream
    >>>(
        workspace.groupLocation,
        workspace.groupSlot,
        workspace.deviceNumGroups,
        workspace.edgeCapacity,
        output.tileRowSlot
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    fillPaddingTileRowSlotsKernel<<<
        blockCount(
            numColWindows
        ),
        THREADS,
        0,
        stream
    >>>(
        workspace.windowActiveRows,
        output.colWindowOffset,
        numColWindows,
        output.tileRowSlot
    );

    CUDA_CHECK(
        cudaGetLastError()
    );
}

}  // namespace tcc