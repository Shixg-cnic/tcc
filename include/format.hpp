#pragma once

#include <cstdint>
#include <vector>
#include <limits>
namespace tcc {

using IndexType = std::uint32_t;
using OffsetType = std::uint32_t;
using ValueType = float;
using BitmapType = std::uint64_t;

class SparseMatrix {
public:
    IndexType rows = 0;
    IndexType cols = 0;
    IndexType nnz = 0;

    SparseMatrix();
    SparseMatrix(IndexType rows_, IndexType cols_, IndexType nnz_);
    virtual ~SparseMatrix();
};

class CSRMatrix : public SparseMatrix {
public:
    std::vector<OffsetType> row_ptr;
    std::vector<IndexType> colIdx;
    std::vector<ValueType> values;

    CSRMatrix();
    CSRMatrix(IndexType rows_, IndexType cols_, IndexType nnz_);
};

class TCFMatrix : public SparseMatrix {
public:
    IndexType colWindowWidth = 0;
    IndexType tileRows = 0;

    std::vector<OffsetType> colWindowOffset;
    std::vector<OffsetType> tileOffset;
    std::vector<IndexType> tileRowIndices;
    std::vector<BitmapType> tileLocalBit;
    std::vector<ValueType> values;

    TCFMatrix();
    TCFMatrix(
        IndexType rows_,
        IndexType cols_,
        IndexType nnz_,
        IndexType colWindowWidth_,
        IndexType tileRows_
    );

    void convertFromCSR(const CSRMatrix& csr);
};

class SWTCFMatrix : public SparseMatrix {
public:
    static constexpr IndexType INVALID_ROW_SLOT = std::numeric_limits<IndexType>::max();
    IndexType colWindowWidth = 0;
    IndexType tileRows = 0;
    IndexType superWindowSize = 0;
    
    std::vector<OffsetType> superWindowRowOffset;
    std::vector<IndexType>  superWindowRows;
    std::vector<OffsetType> colWindowOffset;
    // std::vector<OffsetType> tileOffset;
    std::vector<IndexType>  tileRowSlot;
    std::vector<BitmapType> tileLocalBit;
    
    SWTCFMatrix();
    SWTCFMatrix(
        IndexType rows_,
        IndexType cols_,
        IndexType nnz_,
        IndexType colWindowWidth_,
        IndexType tileRows_,
        IndexType superWindowSize_
    );
};


}  // namespace tcc
