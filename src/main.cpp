#include "format.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> readBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamsize bytes = file.tellg();

    std::vector<T> data(bytes / sizeof(T));

    file.seekg(0, std::ios::beg);
    file.read(
        reinterpret_cast<char*>(data.data()),
        bytes
    );

    return data;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string datasetDir =
        argc > 1
            ? argv[1]
            : "/cnic/work/shixg/GNN/pro_dataset/arxiv";

    const auto rawRowPtr = readBinary<std::uint64_t>(
        datasetDir + "/arxiv_indptr.bin"
    );

    const auto rawColIdx = readBinary<std::uint64_t>(
        datasetDir + "/arxiv_indices.bin"
    );

    const tcc::IndexType rows = rawRowPtr.size() - 1;
    const tcc::IndexType cols = rows;
    const tcc::IndexType nnz = rawColIdx.size();

    tcc::CSRMatrix csr(rows, cols, nnz);
    csr.row_ptr.resize(rawRowPtr.size());

    for (std::size_t index = 0; index < rawRowPtr.size(); ++index) {
        csr.row_ptr[index] =
            static_cast<tcc::OffsetType>(rawRowPtr[index]);
    }

    csr.colIdx.resize(rawColIdx.size());

    for (std::size_t index = 0; index < rawColIdx.size(); ++index) {
        csr.colIdx[index] =
            static_cast<tcc::IndexType>(rawColIdx[index]);
    }

    csr.values.assign(nnz, 1.0f);

    tcc::TCFMatrix tcf(rows, cols, nnz, 8, 8);
    tcf.convertFromCSR(csr);

    std::cout << "CSR rows: " << csr.rows << '\n';
    std::cout << "CSR cols: " << csr.cols << '\n';
    std::cout << "CSR nnz:  " << csr.nnz << '\n';
    std::cout << "TCF column windows: "
              << tcf.colWindowOffset.size() - 1 << '\n';
    std::cout << "TCF tiles: "
              << tcf.tileLocalBit.size() << '\n';
    std::cout << "TCF stored values: "
              << tcf.values.size() << '\n';

    return 0;
}
