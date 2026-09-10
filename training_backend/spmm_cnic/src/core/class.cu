#include "../include/class.h"

#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }


// ======================== BCSR ========================
template <class dataType>
BCSR<dataType>::BCSR(
    vint num_rows,
    vint num_cols,
    vint nnz,
    vint num_TCBlocks,
    vint num_rowWindows,
    vint *rowWindowOffset,
    vint *tcOffset,
    vint *sparseA2B,
    TCLOCAL_TYPE *tcLocalBit)
    : num_rows(num_rows),
      num_cols(num_cols),
      nnz(nnz),
      num_TCBlocks(num_TCBlocks),
      num_rowWindows(num_rowWindows),
      rowWindowOffset(rowWindowOffset),
      tcOffset(tcOffset),
      sparseA2B(sparseA2B),
      tcLocalBit(tcLocalBit)
      {}


template <class dataType>
std::string BCSR<dataType>::toString() const {
    std::ostringstream oss;

    // 输出基本信息
    oss << "GBCSR Structure:\n";
    oss << "num_rows: " << num_rows << "\n";
    oss << "num_cols: " << num_cols << "\n";
    oss << "nnz: " << nnz << "\n";
    oss << "num_TCBlocks: " << num_TCBlocks << "\n";
    oss << "num_rowWindows: " << num_rowWindows << "\n";



    // 输出 rowWindowOffset
    oss << "rowWindowOffset: [";
    thrust::device_vector<vint> d_rowWindowOffset(rowWindowOffset, rowWindowOffset + (num_rowWindows + 1));
    thrust::host_vector<vint> h_rowWindowOffset = d_rowWindowOffset;
    for (size_t i = 0; i < h_rowWindowOffset.size(); ++i) {
        oss << h_rowWindowOffset[i];
        if (i != h_rowWindowOffset.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 tcOffset
    oss << "tcOffset: [";
    thrust::device_vector<vint> d_tcOffset(tcOffset, tcOffset + (num_TCBlocks + 1));
    thrust::host_vector<vint> h_tcOffset = d_tcOffset;
    for (size_t i = 0; i < h_tcOffset.size(); ++i) {
        oss << h_tcOffset[i];
        if (i != h_tcOffset.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 sparseA2B
    oss << "sparseA2B: [";
    thrust::device_vector<vint> d_sparseA2B(sparseA2B, sparseA2B + num_TCBlocks * COL_WINDOW);
    thrust::host_vector<vint> h_sparseA2B = d_sparseA2B;
    for (size_t i = 0; i < h_sparseA2B.size(); ++i) {
        oss << h_sparseA2B[i];
        if (i != h_sparseA2B.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 tcLocalBit
    oss << "tcLocalBit: [";
    thrust::device_vector<TCLOCAL_TYPE> d_tcLocalBit(tcLocalBit, tcLocalBit + num_TCBlocks);
    thrust::host_vector<TCLOCAL_TYPE> h_tcLocalBit = d_tcLocalBit;
    for (size_t i = 0; i < h_tcLocalBit.size(); ++i) {
        oss << h_tcLocalBit[i];
        if (i != h_tcLocalBit.size() - 1) oss << ", ";
    }
    oss << "]\n";

    return oss.str();
}

// ======================== BCSC ========================
template <class dataType>
BCSC<dataType>::BCSC(
    vint num_rows,
    vint num_cols,
    vint nnz,
    vint num_TCBlocks,
    vint num_colWindows,
    vint *colWindowOffset,
    vint *tcOffset,
    vint *sparseA2C,
    TCLOCAL_TYPE *tcLocalBit)
    : num_rows(num_rows),
      num_cols(num_cols),
      nnz(nnz),
      num_TCBlocks(num_TCBlocks),
      num_colWindows(num_colWindows),
      colWindowOffset(colWindowOffset),
      tcOffset(tcOffset),
      sparseA2C(sparseA2C),
      tcLocalBit(tcLocalBit)
      {}
    
template <class dataType>
std::string BCSC<dataType>::toString() const {
    std::ostringstream oss;

    // 输出基本信息
    oss << "GBCSC Structure:\n";
    oss << "num_rows: " << num_rows << "\n";
    oss << "num_cols: " << num_cols << "\n";
    oss << "nnz: " << nnz << "\n";
    oss << "num_TCBlocks: " << num_TCBlocks << "\n";
    oss << "num_colWindows: " << num_colWindows << "\n";

    // 输出 rowWindowOffset
    oss << "colWindowOffset: [";
    thrust::device_vector<vint> d_colWindowOffset(colWindowOffset, colWindowOffset + (num_colWindows + 1));
    thrust::host_vector<vint> h_colWindowOffset = d_colWindowOffset;
    for (size_t i = 0; i < h_colWindowOffset.size(); ++i) {
        oss << h_colWindowOffset[i];
        if (i != h_colWindowOffset.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 tcOffset
    oss << "tcOffset: [";
    thrust::device_vector<vint> d_tcOffset(tcOffset, tcOffset + (num_TCBlocks + 1));
    thrust::host_vector<vint> h_tcOffset = d_tcOffset;
    for (size_t i = 0; i < h_tcOffset.size(); ++i) {
        oss << h_tcOffset[i];
        if (i != h_tcOffset.size() - 1) oss << ", ";
    }
    oss << "]\n";

    std::printf("ROW_WINDOW: %d\n", ROW_WINDOW);

    // 输出 sparseA2B
    oss << "sparseA2C: [";
    thrust::device_vector<vint> d_sparseA2C(sparseA2C, sparseA2C + num_TCBlocks * ROW_WINDOW);
    thrust::host_vector<vint> h_sparseA2C = d_sparseA2C;
    for (size_t i = 0; i < h_sparseA2C.size(); ++i) {
        oss << h_sparseA2C[i];
        if (i != h_sparseA2C.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 tcLocalBit
    oss << "tcLocalBit: [";
    thrust::device_vector<TCLOCAL_TYPE> d_tcLocalBit(tcLocalBit, tcLocalBit + num_TCBlocks);
    thrust::host_vector<TCLOCAL_TYPE> h_tcLocalBit = d_tcLocalBit;
    for (size_t i = 0; i < h_tcLocalBit.size(); ++i) {
        oss << h_tcLocalBit[i];
        if (i != h_tcLocalBit.size() - 1) oss << ", ";
    }
    oss << "]\n";

    return oss.str();
}

template class BCSR<float>;
//template class BCSR<double>;

template class BCSC<float>;
//template class BCSC<double>;