#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <bitset>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include "config.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>

template <typename dataType>
class BCSR {
public:
    vint num_rows, num_cols, nnz;
    vint num_TCBlocks;      // TC 块数量
    vint num_rowWindows;    // ROW_WINDOW 数量

    vint            *rowWindowOffset;
    vint            *tcOffset;
    vint            *sparseA2B;
    TCLOCAL_TYPE    *tcLocalBit;

    explicit BCSR(
        vint num_rows, vint num_cols, vint nnz, vint num_TCBlocks, vint num_rowWindows,
        vint *rowWindowOffset, vint *tcOffset, vint *sparseA2B, TCLOCAL_TYPE *tcLocalBit
    );

    std::string toString() const;
};

template <class dataType>
class BCSC {
public:
    vint num_rows, num_cols, nnz;
    vint num_TCBlocks;  // TC 块数量
    vint num_colWindows;    // col_window 数量

    vint            *colWindowOffset;
    vint            *tcOffset;
    vint            *sparseA2C;   // 应该从 CSR 的列号转变为行号了，将非连续的行号进行连续化
    TCLOCAL_TYPE    *tcLocalBit;

    explicit BCSC(
        vint num_rows, vint num_cols, vint nnz, vint num_TCBlocks, vint num_rowWindows,
        vint *colWindowOffset, vint *tcOffset, vint *sparseA2C, TCLOCAL_TYPE *tcLocalBit
    );

    std::string toString() const;
};