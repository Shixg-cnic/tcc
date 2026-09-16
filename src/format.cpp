#include "format.hpp"

namespace tcc {

SparseMatrix::SparseMatrix() = default;

SparseMatrix::SparseMatrix(
    IndexType rows_,
    IndexType cols_,
    IndexType nnz_
)
    : rows(rows_), cols(cols_), nnz(nnz_) {}

SparseMatrix::~SparseMatrix() = default;

CSRMatrix::CSRMatrix() = default;

CSRMatrix::CSRMatrix(
    IndexType rows_,
    IndexType cols_,
    IndexType nnz_
)
    : SparseMatrix(rows_, cols_, nnz_) {}

TCFMatrix::TCFMatrix() = default;

TCFMatrix::TCFMatrix(
    IndexType rows_,
    IndexType cols_,
    IndexType nnz_,
    IndexType colWindowWidth_,
    IndexType tileRows_
)
    : SparseMatrix(rows_, cols_, nnz_),
      colWindowWidth(colWindowWidth_),
      tileRows(tileRows_) {}

void TCFMatrix::convertFromCSR(const CSRMatrix& csr) {
    struct WindowEntry {
        IndexType row;
        IndexType localCol;
        ValueType value;
    };

    rows = csr.rows;
    cols = csr.cols;
    nnz = csr.nnz;

    colWindowOffset.clear();
    tileOffset.clear();
    tileRowIndices.clear();
    tileLocalBit.clear();
    values.clear();

    const IndexType numColWindows =
        (cols + colWindowWidth - 1) / colWindowWidth;

    std::vector<OffsetType> windowNnz(numColWindows, 0);

    // 统计 col window 的 nnz 
    for (IndexType column : csr.colIdx) {
        ++windowNnz[column / colWindowWidth];
    }

    // 每个 col window 的 nnz(row,local col,value)
    std::vector<std::vector<WindowEntry>> windows(numColWindows);
    for (IndexType window = 0; window < numColWindows; ++window) {
        windows[window].reserve(windowNnz[window]);
    }

    for (IndexType row = 0; row < rows; ++row) {
        for (OffsetType edge = csr.row_ptr[row]; edge < csr.row_ptr[row + 1]; ++edge) {
            const IndexType column = csr.colIdx[edge];
            const IndexType window = column / colWindowWidth;

            windows[window].push_back({
                row,
                column % colWindowWidth,
                csr.values[edge]
            });
        }
    }

    colWindowOffset.push_back(0);
    tileOffset.push_back(0);

    for (const auto& windowEntries : windows) {
        OffsetType entry = 0;
        OffsetType windowTileCount = 0;

        while (entry < windowEntries.size()) {
            BitmapType bitmap = 0;
            std::vector<ValueType> tileValues(
                tileRows * colWindowWidth,
                ValueType{0}
            );

            for (IndexType localRow = 0; localRow < tileRows; ++localRow) {
                if (entry >= windowEntries.size()) {
                    tileRowIndices.push_back(rows);
                    continue;
                }

                const IndexType globalRow = windowEntries[entry].row;
                tileRowIndices.push_back(globalRow);

                while (entry < windowEntries.size() &&
                       windowEntries[entry].row == globalRow) {
                    const IndexType bitPosition =
                        localRow * colWindowWidth +
                        windowEntries[entry].localCol;

                    bitmap |= BitmapType{1} << bitPosition;
                    tileValues[bitPosition] = windowEntries[entry].value;
                    ++entry;
                }
            }

            OffsetType tileNnz = 0;

            for (IndexType bitPosition = 0; bitPosition < tileRows * colWindowWidth; ++bitPosition) {
                if (bitmap & (BitmapType{1} << bitPosition)) {
                    values.push_back(tileValues[bitPosition]);
                    ++tileNnz;
                }
            }

            tileLocalBit.push_back(bitmap);
            tileOffset.push_back(tileOffset.back() + tileNnz);
            ++windowTileCount;
        }

        colWindowOffset.push_back(
            colWindowOffset.back() + windowTileCount
        );
    }
}

SWTCFMatrix::SWTCFMatrix() = default;
SWTCFMatrix::SWTCFMatrix(
    IndexType rows_,
    IndexType cols_,
    IndexType nnz_,
    IndexType colWindowWidth_,
    IndexType tileRows_,
    IndexType superWindowSize_
)
    :SparseMatrix(rows_, cols_, nnz_),
     colWindowWidth(colWindowWidth_),
     tileRows(tileRows_),
     superWindowSize(superWindowSize_) {}

}  // namespace tcc
