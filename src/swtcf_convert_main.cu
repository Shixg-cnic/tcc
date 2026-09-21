#include "swtcf_convert.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> readBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file) throw std::runtime_error("cannot open: " + path);

    const std::streamsize bytes = file.tellg();
    if(bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid binary size: " + path);
    }

    std::vector<T> data(static_cast<std::size_t>(bytes / sizeof(T)));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

std::uint32_t popcount8(std::uint32_t value) {
    return static_cast<std::uint32_t>(__builtin_popcount(value & 0xFFu));
}

bool validateSWTCF(
    const std::vector<IndexType>& dst,
    const std::vector<IndexType>& src,
    IndexType numSrc,
    IndexType superWindowSize,
    const std::vector<OffsetType>& colWindowOffset,
    const std::vector<OffsetType>& superWindowRowOffset,
    const std::vector<IndexType>& superWindowRows,
    const std::vector<IndexType>& tileRowSlot,
    const std::vector<BitmapType>& tileLocalBit
) {
    std::vector<std::uint64_t> expected;
    std::vector<std::uint64_t> reconstructed;

    expected.reserve(dst.size());
    reconstructed.reserve(dst.size());

    for(std::size_t edge = 0; edge < dst.size(); ++edge) {
        expected.push_back((static_cast<std::uint64_t>(dst[edge]) << 32) | static_cast<std::uint64_t>(src[edge]));
    }

    const IndexType numColWindows = static_cast<IndexType>(colWindowOffset.size() - 1);

    for(IndexType window = 0; window < numColWindows; ++window) {
        const IndexType superWindow = window / superWindowSize;
        const OffsetType superBegin = superWindowRowOffset[superWindow];
        const OffsetType superEnd = superWindowRowOffset[superWindow + 1];

        std::uint32_t previousDegree = 9;

        for(OffsetType tile = colWindowOffset[window]; tile < colWindowOffset[window + 1]; ++tile) {
            const BitmapType bitmap = tileLocalBit[tile];

            for(IndexType localRow = 0; localRow < SWTCF_TILE_ROWS; ++localRow) {
                const std::uint32_t rowBits = static_cast<std::uint32_t>((bitmap >> (localRow * 8)) & 0xFFULL);
                const IndexType slot = tileRowSlot[static_cast<std::size_t>(tile) * SWTCF_TILE_ROWS + localRow];

                if(rowBits == 0) {
                    if(slot != std::numeric_limits<IndexType>::max()) {
                        std::cerr << "padding slot is not INVALID\n";
                        return false;
                    }

                    continue;
                }

                const std::uint32_t degree = popcount8(rowBits);

                if(degree > previousDegree) {
                    std::cerr << "row reorder is not degree descending at window " << window << '\n';
                    return false;
                }

                previousDegree = degree;

                const OffsetType rowCount = superEnd - superBegin;

                if(slot >= rowCount) {
                    std::cerr << "invalid row slot\n";
                    return false;
                }

                const IndexType globalDst = superWindowRows[superBegin + slot];

                for(IndexType localCol = 0; localCol < 8; ++localCol) {
                    if((rowBits & (1u << localCol)) == 0) continue;

                    const IndexType globalSrc = window * 8 + localCol;

                    if(globalSrc >= numSrc) {
                        std::cerr << "src exceeds numSrc\n";
                        return false;
                    }

                    reconstructed.push_back((static_cast<std::uint64_t>(globalDst) << 32) | static_cast<std::uint64_t>(globalSrc));
                }
            }
        }
    }

    std::sort(expected.begin(), expected.end());
    std::sort(reconstructed.begin(), reconstructed.end());

    if(expected.size() != reconstructed.size()) {
        std::cerr << "edge count mismatch: expected=" << expected.size() << " reconstructed=" << reconstructed.size() << '\n';
        return false;
    }

    if(expected != reconstructed) {
        std::cerr << "COO reconstruction mismatch\n";
        return false;
    }

    return true;
}

bool validateRepeatCacheMetadata(
    IndexType superWindowSize,
    const std::vector<OffsetType>& colWindowOffset,
    const std::vector<OffsetType>& superWindowRowOffset,
    const std::vector<IndexType>& tileRowSlot,
    const std::vector<BitmapType>& tileLocalBit,
    const std::vector<std::uint8_t>& superWindowRowCacheSlot,
    const std::vector<IndexType>& superWindowCachedRowSlot,
    const std::vector<std::uint8_t>& superWindowCachedCount
) {
    const IndexType numColWindows = static_cast<IndexType>(colWindowOffset.size() - 1);
    const IndexType numSuperWindows = static_cast<IndexType>(superWindowRowOffset.size() - 1);

    std::uint64_t totalRowOccurrences = 0;
    std::uint64_t totalRepeatedRows = 0;
    std::uint64_t totalCachedRows = 0;
    std::uint64_t totalPossibleSaving = 0;
    std::uint64_t totalCapturedSaving = 0;

    for(IndexType sw = 0; sw < numSuperWindows; ++sw) {
        const OffsetType rowBegin = superWindowRowOffset[sw];
        const OffsetType rowEnd = superWindowRowOffset[sw + 1];
        const IndexType rowCount = static_cast<IndexType>(rowEnd - rowBegin);

        std::vector<IndexType> occurrence(rowCount, 0);

        const IndexType windowBegin = sw * superWindowSize;
        const IndexType windowEnd = std::min<IndexType>(numColWindows, windowBegin + superWindowSize);

        for(IndexType window = windowBegin; window < windowEnd; ++window) {
            for(OffsetType tile = colWindowOffset[window]; tile < colWindowOffset[window + 1]; ++tile) {
                const BitmapType bitmap = tileLocalBit[tile];

                for(IndexType localRow = 0; localRow < SWTCF_TILE_ROWS; ++localRow) {
                    const std::uint32_t rowBits = static_cast<std::uint32_t>((bitmap >> (localRow * 8)) & 0xFFULL);
                    if(rowBits == 0) continue;

                    const IndexType rowSlot = tileRowSlot[static_cast<std::size_t>(tile) * SWTCF_TILE_ROWS + localRow];

                    if(rowSlot >= rowCount) {
                        std::cerr << "cache validation: rowSlot out of range\n";
                        return false;
                    }

                    ++occurrence[rowSlot];
                }
            }
        }

        IndexType repeatedRows = 0;

        for(IndexType rowSlot = 0; rowSlot < rowCount; ++rowSlot) {
            const IndexType count = occurrence[rowSlot];

            if(count == 0) {
                std::cerr << "cache validation: superWindowRows contains unused row\n";
                return false;
            }

            totalRowOccurrences += count;

            if(count > 1) {
                ++repeatedRows;
                ++totalRepeatedRows;
                totalPossibleSaving += count - 1;
            }
        }

        const IndexType expectedCached = std::min<IndexType>(repeatedRows, SWTCF_REPEAT_CACHE_SIZE);
        const IndexType cachedCount = superWindowCachedCount[sw];

        if(cachedCount != expectedCached) {
            std::cerr
                << "cache validation: cachedCount mismatch at SW "
                << sw
                << ", expected "
                << expectedCached
                << ", got "
                << cachedCount
                << '\n';
            return false;
        }

        std::vector<bool> rowIsCached(rowCount, false);
        std::vector<bool> cacheSlotSeen(SWTCF_REPEAT_CACHE_SIZE, false);

        IndexType minCachedOccurrence = std::numeric_limits<IndexType>::max();
        IndexType maxUncachedOccurrence = 0;

        for(IndexType cacheSlot = 0; cacheSlot < SWTCF_REPEAT_CACHE_SIZE; ++cacheSlot) {
            const IndexType rowSlot = superWindowCachedRowSlot[static_cast<std::size_t>(sw) * SWTCF_REPEAT_CACHE_SIZE + cacheSlot];

            if(cacheSlot < cachedCount) {
                if(rowSlot >= rowCount) {
                    std::cerr << "cache validation: cached rowSlot invalid at SW " << sw << '\n';
                    return false;
                }

                if(occurrence[rowSlot] <= 1) {
                    std::cerr << "cache validation: non-repeated row was cached\n";
                    return false;
                }

                if(rowIsCached[rowSlot]) {
                    std::cerr << "cache validation: row cached twice\n";
                    return false;
                }

                if(cacheSlotSeen[cacheSlot]) {
                    std::cerr << "cache validation: cache slot duplicated\n";
                    return false;
                }

                rowIsCached[rowSlot] = true;
                cacheSlotSeen[cacheSlot] = true;

                const std::uint8_t reverse = superWindowRowCacheSlot[rowBegin + rowSlot];

                if(reverse != static_cast<std::uint8_t>(cacheSlot)) {
                    std::cerr << "cache validation: forward/reverse mapping mismatch\n";
                    return false;
                }

                const IndexType count = occurrence[rowSlot];
                minCachedOccurrence = std::min(minCachedOccurrence, count);

                totalCapturedSaving += count - 1;
                ++totalCachedRows;
            } else {
                if(rowSlot != std::numeric_limits<IndexType>::max()) {
                    std::cerr << "cache validation: unused cache entry is not INVALID\n";
                    return false;
                }
            }
        }

        for(IndexType rowSlot = 0; rowSlot < rowCount; ++rowSlot) {
            const std::uint8_t cacheSlot = superWindowRowCacheSlot[rowBegin + rowSlot];

            if(rowIsCached[rowSlot]) {
                if(cacheSlot >= SWTCF_REPEAT_CACHE_SIZE) {
                    std::cerr << "cache validation: cached row has invalid cacheSlot\n";
                    return false;
                }
            } else {
                if(cacheSlot != SWTCF_INVALID_CACHE_SLOT) {
                    std::cerr << "cache validation: uncached row has cacheSlot\n";
                    return false;
                }

                if(occurrence[rowSlot] > 1) {
                    maxUncachedOccurrence = std::max(maxUncachedOccurrence, occurrence[rowSlot]);
                }
            }
        }

        if(repeatedRows > SWTCF_REPEAT_CACHE_SIZE && minCachedOccurrence < maxUncachedOccurrence) {
            std::cerr << "cache validation: top-H priority violated at SW " << sw << '\n';
            return false;
        }
    }

    const double cacheCoverage = totalPossibleSaving > 0
        ? 100.0 * static_cast<double>(totalCapturedSaving) / static_cast<double>(totalPossibleSaving)
        : 100.0;

    const double baselineReduction = totalRowOccurrences > 0
        ? 100.0 * static_cast<double>(totalCapturedSaving) / static_cast<double>(totalRowOccurrences)
        : 0.0;

    std::cout << "repeated rows: " << totalRepeatedRows << '\n';
    std::cout << "cached repeated rows: " << totalCachedRows << '\n';
    std::cout << "possible saved row writes: " << totalPossibleSaving << '\n';
    std::cout << "captured saved row writes: " << totalCapturedSaving << '\n';
    std::cout << "saving coverage: " << cacheCoverage << "%\n";
    std::cout << "baseline row-write reduction: " << baselineReduction << "%\n";

    return true;
}

}

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cerr
            << "usage:\n"
            << argv[0]
            << " <dst_local.bin>"
            << " <src_local.bin>"
            << " [super_window_size=16]"
            << " [repeat=20]\n";
        return 1;
    }

    const std::string dstPath = argv[1];
    const std::string srcPath = argv[2];

    const IndexType superWindowSize = argc > 3 ? static_cast<IndexType>(std::stoul(argv[3])) : 16;
    const int repeatCount = argc > 4 ? std::stoi(argv[4]) : 20;

    const auto dst = readBinary<IndexType>(dstPath);
    const auto src = readBinary<IndexType>(srcPath);

    if(dst.empty() || dst.size() != src.size()) {
        throw std::runtime_error("invalid COO input");
    }

    const IndexType numEdges = static_cast<IndexType>(dst.size());
    const IndexType numDst = *std::max_element(dst.begin(), dst.end()) + 1;
    const IndexType numSrc = *std::max_element(src.begin(), src.end()) + 1;

    std::cout << "rows / dst: " << numDst << '\n';
    std::cout << "cols / src: " << numSrc << '\n';
    std::cout << "edges: " << numEdges << '\n';
    std::cout << "super window size: " << superWindowSize << '\n';
    std::cout << "repeat cache size: " << SWTCF_REPEAT_CACHE_SIZE << '\n';

    IndexType* deviceDst = nullptr;
    IndexType* deviceSrc = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&deviceDst), dst.size() * sizeof(IndexType));
    cudaMalloc(reinterpret_cast<void**>(&deviceSrc), src.size() * sizeof(IndexType));

    cudaMemcpy(deviceDst, dst.data(), dst.size() * sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSrc, src.data(), src.size() * sizeof(IndexType), cudaMemcpyHostToDevice);

    DeviceSWTCFMatrix swtcf;
    SWTCFConvertWorkspace workspace;

    allocateDeviceSWTCF(
        swtcf,
        numDst,
        numSrc,
        numEdges,
        superWindowSize
    );

    allocateSWTCFConvertWorkspace(
        workspace,
        numEdges,
        numSrc,
        superWindowSize
    );

    constexpr int warmupCount = 5;

    for(int iteration = 0; iteration < warmupCount; ++iteration) {
        convertCOOToSWTCF(
            deviceDst,
            deviceSrc,
            numEdges,
            swtcf,
            workspace
        );
    }

    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int iteration = 0; iteration < repeatCount; ++iteration) {
        convertCOOToSWTCF(
            deviceDst,
            deviceSrc,
            numEdges,
            swtcf,
            workspace
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalMilliseconds = 0.0f;
    cudaEventElapsedTime(&totalMilliseconds, start, stop);

    const float averageMilliseconds = totalMilliseconds / repeatCount;

    std::cout << "conversion time: " << averageMilliseconds << " ms\n";
    std::cout << "throughput: " << static_cast<double>(numEdges) / averageMilliseconds / 1.0e6 << " GEdges/s\n";

    IndexType deviceError = 0;
    IndexType numGroups = 0;

    cudaMemcpy(&deviceError, workspace.deviceError, sizeof(IndexType), cudaMemcpyDeviceToHost);
    cudaMemcpy(&numGroups, workspace.deviceNumGroups, sizeof(IndexType), cudaMemcpyDeviceToHost);

    if(deviceError != 0) {
        std::cerr << "converter device error: " << deviceError << '\n';
        return 2;
    }

    std::vector<OffsetType> colWindowOffset(swtcf.numColWindows + 1);
    std::vector<OffsetType> superWindowRowOffset(swtcf.numSuperWindows + 1);

    cudaMemcpy(
        colWindowOffset.data(),
        swtcf.colWindowOffset,
        colWindowOffset.size() * sizeof(OffsetType),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        superWindowRowOffset.data(),
        swtcf.superWindowRowOffset,
        superWindowRowOffset.size() * sizeof(OffsetType),
        cudaMemcpyDeviceToHost
    );

    const OffsetType numTiles = colWindowOffset.back();
    const OffsetType numSuperRows = superWindowRowOffset.back();

    std::vector<BitmapType> tileLocalBit(numTiles);
    std::vector<IndexType> tileRowSlot(static_cast<std::size_t>(numTiles) * SWTCF_TILE_ROWS);
    std::vector<IndexType> superWindowRows(numSuperRows);
    std::vector<std::uint8_t> superWindowRowCacheSlot(numSuperRows);
    std::vector<IndexType> superWindowCachedRowSlot(static_cast<std::size_t>(swtcf.numSuperWindows) * SWTCF_REPEAT_CACHE_SIZE);
    std::vector<std::uint8_t> superWindowCachedCount(swtcf.numSuperWindows);

    cudaMemcpy(
        tileLocalBit.data(),
        swtcf.tileLocalBit,
        tileLocalBit.size() * sizeof(BitmapType),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        tileRowSlot.data(),
        swtcf.tileRowSlot,
        tileRowSlot.size() * sizeof(IndexType),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        superWindowRows.data(),
        swtcf.superWindowRows,
        superWindowRows.size() * sizeof(IndexType),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        superWindowRowCacheSlot.data(),
        swtcf.superWindowRowCacheSlot,
        superWindowRowCacheSlot.size() * sizeof(std::uint8_t),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        superWindowCachedRowSlot.data(),
        swtcf.superWindowCachedRowSlot,
        superWindowCachedRowSlot.size() * sizeof(IndexType),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        superWindowCachedCount.data(),
        swtcf.superWindowCachedCount,
        superWindowCachedCount.size() * sizeof(std::uint8_t),
        cudaMemcpyDeviceToHost
    );

    std::cout << "active window rows / groups: " << numGroups << '\n';
    std::cout << "col windows: " << swtcf.numColWindows << '\n';
    std::cout << "super windows: " << swtcf.numSuperWindows << '\n';
    std::cout << "tiles: " << numTiles << '\n';
    std::cout << "super-window unique rows: " << numSuperRows << '\n';

    const bool swtcfValid = validateSWTCF(
        dst,
        src,
        numSrc,
        superWindowSize,
        colWindowOffset,
        superWindowRowOffset,
        superWindowRows,
        tileRowSlot,
        tileLocalBit
    );

    std::cout << "SWTCF validation: " << (swtcfValid ? "PASS" : "FAIL") << '\n';

    bool cacheValid = false;

    if(swtcfValid) {
        cacheValid = validateRepeatCacheMetadata(
            superWindowSize,
            colWindowOffset,
            superWindowRowOffset,
            tileRowSlot,
            tileLocalBit,
            superWindowRowCacheSlot,
            superWindowCachedRowSlot,
            superWindowCachedCount
        );
    }

    std::cout << "repeat-cache validation: " << (cacheValid ? "PASS" : "FAIL") << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    freeSWTCFConvertWorkspace(workspace);
    freeDeviceSWTCF(swtcf);

    cudaFree(deviceDst);
    cudaFree(deviceSrc);

    return swtcfValid && cacheValid ? 0 : 3;
}