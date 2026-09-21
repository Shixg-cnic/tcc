#include "swtcf_convert.cuh"
#include "swtcf_spmm.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
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
        throw std::runtime_error("invalid file size: " + path);
    }

    std::vector<T> data(static_cast<std::size_t>(bytes / sizeof(T)));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

__host__ __device__ float sourceA(IndexType source) {
    const std::uint32_t value = (source * 17u + 13u) % 97u;
    return static_cast<float>(value) * 0.001f;
}

__host__ __device__ float sourceB(IndexType source) {
    const std::uint32_t value = (source * 29u + 7u) % 89u;
    return static_cast<float>(value) * 0.00001f;
}

__host__ __device__ float featureValue(IndexType source, IndexType feature) {
    return sourceA(source) + sourceB(source) * static_cast<float>(feature);
}

__global__ void initializeXKernel(ValueType* matrixX, IndexType numSrc) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(numSrc) * SWTCF_FEATURE_DIM;

    if(index >= total) return;

    const IndexType source = static_cast<IndexType>(index / SWTCF_FEATURE_DIM);
    const IndexType feature = static_cast<IndexType>(index % SWTCF_FEATURE_DIM);
    matrixX[index] = featureValue(source, feature);
}

bool validateOutput(
    const std::vector<IndexType>& dst,
    const std::vector<IndexType>& src,
    IndexType numDst,
    const std::vector<ValueType>& output,
    IndexType tcThreshold
) {
    std::vector<double> sumA(numDst, 0.0);
    std::vector<double> sumB(numDst, 0.0);

    for(std::size_t edge = 0; edge < dst.size(); ++edge) {
        const IndexType row = dst[edge];
        const IndexType column = src[edge];
        sumA[row] += static_cast<double>(sourceA(column));
        sumB[row] += static_cast<double>(sourceB(column));
    }

    const double atol = tcThreshold > 64 ? 2.0e-4 : 1.0e-2;
    const double rtol = tcThreshold > 64 ? 1.0e-4 : 1.0e-2;

    double maxAbsError = 0.0;
    double maxRelativeError = 0.0;
    long double sumAbsError = 0.0;
    std::uint64_t failed = 0;

    const std::size_t total = static_cast<std::size_t>(numDst) * SWTCF_FEATURE_DIM;

    for(IndexType row = 0; row < numDst; ++row) {
        for(IndexType feature = 0; feature < SWTCF_FEATURE_DIM; ++feature) {
            const double reference = sumA[row] + sumB[row] * static_cast<double>(feature);
            const double actual = static_cast<double>(output[static_cast<std::size_t>(row) * SWTCF_FEATURE_DIM + feature]);

            const double absError = std::abs(actual - reference);
            const double denominator = std::max(std::abs(reference), 1.0e-8);
            const double relativeError = absError / denominator;

            maxAbsError = std::max(maxAbsError, absError);
            maxRelativeError = std::max(maxRelativeError, relativeError);
            sumAbsError += absError;

            const double tolerance = atol + rtol * std::abs(reference);
            if(absError > tolerance) ++failed;
        }
    }

    const double meanAbsError = static_cast<double>(sumAbsError / static_cast<long double>(total));

    std::cout << "max abs error: " << maxAbsError << '\n';
    std::cout << "mean abs error: " << meanAbsError << '\n';
    std::cout << "max relative error: " << maxRelativeError << '\n';
    std::cout << "failed elements: " << failed << " / " << total << '\n';
    std::cout << "tolerance: atol=" << atol << " rtol=" << rtol << '\n';

    return failed == 0;
}

}

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cerr
            << "usage:\n"
            << argv[0]
            << " <dst_local.bin>"
            << " <src_local.bin>"
            << " [tc_threshold=12]"
            << " [repeat=20]\n";
        return 1;
    }

    const std::string dstPath = argv[1];
    const std::string srcPath = argv[2];

    const IndexType tcThreshold = argc > 3 ? static_cast<IndexType>(std::stoul(argv[3])) : SWTCF_DEFAULT_TC_THRESHOLD;
    const int repeatCount = argc > 4 ? std::stoi(argv[4]) : 20;
    constexpr IndexType superWindowSize = 16;

    const auto dst = readBinary<IndexType>(dstPath);
    const auto src = readBinary<IndexType>(srcPath);

    if(dst.empty() || dst.size() != src.size()) throw std::runtime_error("invalid COO");

    const IndexType numEdges = static_cast<IndexType>(dst.size());
    const IndexType numDst = *std::max_element(dst.begin(), dst.end()) + 1;
    const IndexType numSrc = *std::max_element(src.begin(), src.end()) + 1;

    std::cout << "rows / dst: " << numDst << '\n';
    std::cout << "cols / src: " << numSrc << '\n';
    std::cout << "edges: " << numEdges << '\n';
    std::cout << "super window size: " << superWindowSize << '\n';
    std::cout << "repeat cache size: " << SWTCF_REPEAT_CACHE_SIZE << '\n';
    std::cout << "TC threshold: " << tcThreshold << '\n';

    IndexType* deviceDst = nullptr;
    IndexType* deviceSrc = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&deviceDst), dst.size() * sizeof(IndexType));
    cudaMalloc(reinterpret_cast<void**>(&deviceSrc), src.size() * sizeof(IndexType));

    cudaMemcpy(deviceDst, dst.data(), dst.size() * sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSrc, src.data(), src.size() * sizeof(IndexType), cudaMemcpyHostToDevice);

    DeviceSWTCFMatrix swtcf;
    SWTCFConvertWorkspace workspace;

    allocateDeviceSWTCF(swtcf, numDst, numSrc, numEdges, superWindowSize);
    allocateSWTCFConvertWorkspace(workspace, numEdges, numSrc, superWindowSize);

    convertCOOToSWTCF(
        deviceDst,
        deviceSrc,
        numEdges,
        swtcf,
        workspace
    );

    cudaDeviceSynchronize();

    IndexType deviceError = 0;
    cudaMemcpy(&deviceError, workspace.deviceError, sizeof(IndexType), cudaMemcpyDeviceToHost);

    if(deviceError != 0) {
        std::cerr << "converter error: " << deviceError << '\n';
        return 2;
    }

    const std::size_t xElements = static_cast<std::size_t>(numSrc) * SWTCF_FEATURE_DIM;
    const std::size_t yElements = static_cast<std::size_t>(numDst) * SWTCF_FEATURE_DIM;

    ValueType* deviceX = nullptr;
    ValueType* deviceY = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&deviceX), xElements * sizeof(ValueType));
    cudaMalloc(reinterpret_cast<void**>(&deviceY), yElements * sizeof(ValueType));

    constexpr int threads = 256;
    const int xBlocks = static_cast<int>((xElements + threads - 1) / threads);

    initializeXKernel<<<xBlocks, threads>>>(
        deviceX,
        numSrc
    );

    cudaDeviceSynchronize();

    constexpr int warmupCount = 5;

    cudaMemset(deviceY, 0, yElements * sizeof(ValueType));

    for(int iteration = 0; iteration < warmupCount; ++iteration) {
        launchSWTCFSpMM(
            swtcf,
            deviceX,
            deviceY,
            tcThreshold
        );
    }

    cudaDeviceSynchronize();

    cudaMemset(deviceY, 0, yElements * sizeof(ValueType));

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int iteration = 0; iteration < repeatCount; ++iteration) {
        launchSWTCFSpMM(
            swtcf,
            deviceX,
            deviceY,
            tcThreshold
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalMilliseconds = 0.0f;
    cudaEventElapsedTime(&totalMilliseconds, start, stop);

    const float averageMilliseconds = totalMilliseconds / repeatCount;
    const double gflops = 2.0 * static_cast<double>(numEdges) * SWTCF_FEATURE_DIM / (averageMilliseconds * 1.0e6);

    std::cout << "kernel time: " << averageMilliseconds << " ms\n";
    std::cout << "effective performance: " << gflops << " GFLOPS\n";

    cudaMemset(deviceY, 0, yElements * sizeof(ValueType));

    launchSWTCFSpMM(
        swtcf,
        deviceX,
        deviceY,
        tcThreshold
    );

    cudaDeviceSynchronize();

    std::vector<ValueType> output(yElements);
    cudaMemcpy(output.data(), deviceY, yElements * sizeof(ValueType), cudaMemcpyDeviceToHost);

    const bool valid = validateOutput(dst, src, numDst, output, tcThreshold);

    std::cout << "SpMM validation: " << (valid ? "PASS" : "FAIL") << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(deviceX);
    cudaFree(deviceY);

    freeSWTCFConvertWorkspace(workspace);
    freeDeviceSWTCF(swtcf);

    cudaFree(deviceDst);
    cudaFree(deviceSrc);

    return valid ? 0 : 3;
}