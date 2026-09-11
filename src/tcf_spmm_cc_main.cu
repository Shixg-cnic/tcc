#include "format.hpp"
#include "tcf_spmm_cc.cuh"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


template <typename T>
std::vector<T> readBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamsize bytes = file.tellg();
    std::vector<T> data(bytes / sizeof(T));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

template <typename T>
void writeBinary(const std::string& path, const std::vector<T>& data) {
    std::ofstream file(path, std::ios::binary);
    file.write(
        reinterpret_cast<const char*>(data.data()),
        data.size() * sizeof(T)
    );
}

int main(int argc, char** argv) {
    const std::string datasetDir =
        argc > 1
            ? argv[1]
            : "/cnic/work/shixg/GNN/pro_dataset/arxiv";

    const std::string outputPath =
        argc > 2
            ? argv[2]
            : "/cnic/work/shixg/GNN/tc_cc_kernel/TccSpmm/test/data/arxiv_tcf_cc_output.bin";

    constexpr tcc::IndexType featureDim = 128;

    const auto rawRowPtr =
        readBinary<std::uint64_t>(datasetDir + "/arxiv_indptr.bin");

    const auto rawColIdx =
        readBinary<std::uint64_t>(datasetDir + "/arxiv_indices.bin");

    const auto matrixX =
        readBinary<tcc::ValueType>(datasetDir + "/arxiv_features.bin");

    const tcc::IndexType rows =
        static_cast<tcc::IndexType>(rawRowPtr.size() - 1);

    const tcc::IndexType cols = rows;

    const tcc::IndexType nnz =
        static_cast<tcc::IndexType>(rawColIdx.size());

    tcc::CSRMatrix csr(rows, cols, nnz);
    csr.row_ptr.resize(rawRowPtr.size());
    csr.colIdx.resize(rawColIdx.size());
    csr.values.assign(nnz, 1.0f);

    for(std::size_t index = 0; index < rawRowPtr.size(); ++index) {
        csr.row_ptr[index] =
            static_cast<tcc::OffsetType>(rawRowPtr[index]);
    }

    for(std::size_t index = 0; index < rawColIdx.size(); ++index) {
        csr.colIdx[index] =
            static_cast<tcc::IndexType>(rawColIdx[index]);
    }

    tcc::TCFMatrix tcf(rows, cols, nnz, 8, 8);
    tcf.convertFromCSR(csr);

    const tcc::IndexType numColWindows =
        static_cast<tcc::IndexType>(
            tcf.colWindowOffset.size() - 1
        );

    std::cout << "rows: " << rows << '\n';
    std::cout << "cols: " << cols << '\n';
    std::cout << "nnz: " << nnz << '\n';
    std::cout << "feature dim: " << featureDim << '\n';
    std::cout << "col windows: " << numColWindows << '\n';
    std::cout << "tiles: " << tcf.tileLocalBit.size() << '\n';

    const std::size_t outputElements = static_cast<std::size_t>(rows) * featureDim;

    std::vector<tcc::ValueType> matrixY(outputElements, 0.0f);

    tcc::OffsetType* deviceColWindowOffset = nullptr;
    tcc::OffsetType* deviceTileOffset = nullptr;
    tcc::IndexType* deviceTileRowIndices = nullptr;
    tcc::BitmapType* deviceTileLocalBit = nullptr;
    tcc::ValueType* deviceValues = nullptr;
    tcc::ValueType* deviceX = nullptr;
    tcc::ValueType* deviceY = nullptr;

    cudaMalloc(&deviceColWindowOffset, tcf.colWindowOffset.size() * sizeof(tcc::OffsetType));
    cudaMalloc(&deviceTileOffset, tcf.tileOffset.size() * sizeof(tcc::OffsetType));
    cudaMalloc(&deviceTileRowIndices, tcf.tileRowIndices.size() * sizeof(tcc::IndexType));
    cudaMalloc(&deviceTileLocalBit, tcf.tileLocalBit.size() * sizeof(tcc::BitmapType));
    cudaMalloc(&deviceValues, tcf.values.size() * sizeof(tcc::ValueType));
    cudaMalloc(&deviceX, matrixX.size() * sizeof(tcc::ValueType));
    cudaMalloc(&deviceY, matrixY.size() * sizeof(tcc::ValueType));

    cudaMemcpy(deviceColWindowOffset, tcf.colWindowOffset.data(), tcf.colWindowOffset.size() * sizeof(tcc::OffsetType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceTileOffset, tcf.tileOffset.data(), tcf.tileOffset.size() * sizeof(tcc::OffsetType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceTileRowIndices, tcf.tileRowIndices.data(), tcf.tileRowIndices.size() * sizeof(tcc::IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceTileLocalBit, tcf.tileLocalBit.data(), tcf.tileLocalBit.size() * sizeof(tcc::BitmapType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, tcf.values.data(), tcf.values.size() * sizeof(tcc::ValueType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceX, matrixX.data(), matrixX.size() * sizeof(tcc::ValueType), cudaMemcpyHostToDevice);
    constexpr int warmupCount = 10;
    constexpr int repeatCount = 100;

    cudaMemset(deviceY, 0, matrixY.size() * sizeof(tcc::ValueType));
    for(int repeat = 0; repeat < warmupCount; ++repeat) {
        tcc::launchTcfSpMMCC(deviceColWindowOffset, deviceTileOffset, deviceTileRowIndices, deviceTileLocalBit, deviceValues, deviceX, deviceY, rows, cols, numColWindows, featureDim);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int repeat = 0; repeat < repeatCount; ++repeat) {
        tcc::launchTcfSpMMCC(deviceColWindowOffset, deviceTileOffset, deviceTileRowIndices, deviceTileLocalBit, deviceValues, deviceX, deviceY, rows, cols, numColWindows, featureDim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalMilliseconds = 0.0f;
    cudaEventElapsedTime(&totalMilliseconds, start, stop);

    const float averageMilliseconds = totalMilliseconds / repeatCount;
    const double gflops = 2.0 * nnz * featureDim / (averageMilliseconds * 1.0e6);

    std::cout << "kernel time: " << averageMilliseconds << " ms\n";
    std::cout << "effective performance: " << gflops << " GFLOPS\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemset(deviceY, 0, matrixY.size() * sizeof(tcc::ValueType));
    tcc::launchTcfSpMMCC(deviceColWindowOffset, deviceTileOffset, deviceTileRowIndices, deviceTileLocalBit, deviceValues, deviceX, deviceY, rows, cols, numColWindows, featureDim);
    cudaDeviceSynchronize();

    cudaMemcpy(matrixY.data(), deviceY, matrixY.size() * sizeof(tcc::ValueType), cudaMemcpyDeviceToHost);
    writeBinary(outputPath, matrixY);

    cudaFree(deviceColWindowOffset);
    cudaFree(deviceTileOffset);
    cudaFree(deviceTileRowIndices);
    cudaFree(deviceTileLocalBit);
    cudaFree(deviceValues);
    cudaFree(deviceX);
    cudaFree(deviceY);

    std::cout << "output: " << outputPath << '\n';
    return 0;
}
