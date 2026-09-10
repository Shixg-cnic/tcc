#include <torch/extension.h>
#include <thrust/device_vector.h>

#include "spmm_launcher.h"

#include <fstream>

std::string base_path = "/home/gaohy/Legions/TAC/result/products_batch_800/";

void print_denseC(MAT_VAL_TYPE* denseC, const vint rows, const vint rows_ori, const vint cols) {
    for(vint i = 0; i < rows; ++i) {
        printf("row %d: { ", i);
        for(vint j = 0; j < cols; ++j) {
            // if(denseC[i * FEATURE_DIM + j] > 0.0 || denseC[i * FEATURE_DIM + j] < 0.0) 
            // printf("%.2lf ", denseC[i * cols + j]);
            printf("%f ", denseC[i * cols + j]);
        }
        printf(" }\n");
    }
    // for(vint i = 0; i < rows_ori - rows; ++i) {
    //     for(vint j = 0; j < cols; ++j) {
    //         printf("0.00 ");
    //     }
    //     printf("\n");
    // }
    printf("\n");
}

// 第一层 SpMM 计算
void launch_spmm_kernel(
    const BCSR<MAT_VAL_TYPE> &bcsr,
    const BCSC<MAT_VAL_TYPE> &bcsc,
    int32_t* id_map,
    float* cpu_feature_cache_ptr,
    int32_t* cache_search_map,
    float** gpu_feature_cache_ptr,
    MAT_VAL_TYPE* &d_MatC,
    int64_t gpu_node_capacity,
    int32_t feature_dim,
    const int32_t device_id,
    cudaStream_t stream_bcsr,
    cudaStream_t stream_bcsc
) {
    //printf("[Info] first layer launch_spmm_kernel\n");
    vint numRows       = bcsr.num_rows;
    vint numCols       = bcsr.num_cols;
    vint numNnz        = bcsr.nnz;
    vint numTCBlocks     = bcsr.num_TCBlocks;     // TC 块数量
    vint numRowWindows    = bcsr.num_rowWindows;   // row_window 的数量

    // 开辟 denseC 空间
    //vint denseC_size = std::max(rowWndSize * ROW_WINDOW, bcsc.rowIndices.back() + 8) * feature_dim;
    vint denseC_size = (numRowWindows + 1) * ROW_WINDOW * feature_dim; // 这里多开一个 ROW_WINDOW 是为了应对 BCSC 的行补全情况
    //cudaMalloc(&d_MatC, sizeof(MAT_VAL_TYPE) * denseC_size);
    cudaMemset(d_MatC, 0, denseC_size * sizeof(MAT_VAL_TYPE));

    // BCSR & BCSC 计算
    float bcsr_time = mixed_tf32_spmm_bcsr(bcsr, gpu_node_capacity, feature_dim, id_map, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr, d_MatC, stream_bcsr);
    float bcsc_time = mixed_tf32_spmm_bcsc(bcsc, gpu_node_capacity, feature_dim, id_map, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr, d_MatC, stream_bcsc);

    //#define output_denseC_size
    #ifdef output_denseC_size
    std::string graph_attri_path = base_path + "shape/" + std::to_string(device_id) + "_1_layer.csv";
    std::ofstream graph_attri_ofs(graph_attri_path, std::ios::out | std::ios::app); 

    if (!graph_attri_ofs.is_open()) {
        std::cerr << "Failed to open output file: " << graph_attri_path << "\n";
        return;
    }

    //graph_attri_ofs << "numRows" << ", " << "numCols" << ", " << "numNnz" << ", " << "numTCBlocks" << ", " << "numRowWindows" << ", " << "denseC_size" << "\n";
    graph_attri_ofs << numRows << ", " << numCols << ", " << numNnz << ", " << numTCBlocks << ", " << numRowWindows << ", " << denseC_size << "\n";

    #endif
    #undef output_denseC_size

    // 结果写入文件
    #ifdef output_result_to_file
    // 矩阵 shape
    std::string graph_attri_path = base_path + "/shape/" + std::to_string(device_id) + ".csv";
    std::ofstream graph_attri_ofs(graph_attri_path, std::ios::out | std::ios::app);

    if (!graph_attri_ofs.is_open()) {
        std::cerr << "Failed to open output file: " << graph_attri_path << "\n";
        return;
    }

    graph_attri_ofs << numRows << ", " << numCols << ", " << numNnz << ", " << numTCBlocks << ", " << numRowWindows << "\n";

    // 时间
    std::string file_path = base_path + "/compute-memory/" + std::to_string(device_id) + ".csv";
    std::ofstream ofs(file_path, std::ios::out | std::ios::app);

    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << file_path << "\n";
        return;
    }

    ofs << bcsr_time << ", " << bcsc_time << "\n";
    #endif

    //#define debug_check_result_MatC
    #ifdef debug_check_result_MatC
    MAT_VAL_TYPE *h_DenseMatC = (MAT_VAL_TYPE*) malloc(denseC_size * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h_DenseMatC, d_MatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyDeviceToHost);
    print_denseC(h_DenseMatC, numRows, denseC_size / feature_dim, feature_dim);
    #endif
    
    // 同步操作后移
    //CHECK_LAST_CUDA_ERROR();
    //cudaDeviceSynchronize();
}

// 第一层 SpMM 计算
void launch_spmm_kernel(
    const BCSR<MAT_VAL_TYPE> &bcsr,
    const BCSC<MAT_VAL_TYPE> &bcsc,
    MAT_VAL_TYPE* d_MatB,
    int32_t* id_map,
    float* cpu_feature_cache_ptr,
    int32_t* cache_search_map,
    float** gpu_feature_cache_ptr,
    MAT_VAL_TYPE* &d_MatC,
    int64_t gpu_node_capacity,
    int32_t feature_dim,
    const int32_t device_id,
    cudaStream_t stream_bcsr,
    cudaStream_t stream_bcsc
) {
    //printf("[Info] first layer launch_spmm_kernel\n");
    vint numRows       = bcsr.num_rows;
    vint numCols       = bcsr.num_cols;
    vint numNnz        = bcsr.nnz;
    vint numTCBlocks     = bcsr.num_TCBlocks;     // TC 块数量
    vint numRowWindows    = bcsr.num_rowWindows;   // row_window 的数量

    // 开辟 denseC 空间
    //vint denseC_size = std::max(rowWndSize * ROW_WINDOW, bcsc.rowIndices.back() + 8) * feature_dim;
    vint denseC_size = (numRowWindows + 1) * ROW_WINDOW * feature_dim; // 这里多开一个 ROW_WINDOW 是为了应对 BCSC 的行补全情况
    //cudaMalloc(&d_MatC, sizeof(MAT_VAL_TYPE) * denseC_size);
    cudaMemset(d_MatC, 0, denseC_size * sizeof(MAT_VAL_TYPE));

    // BCSR & BCSC 计算
    //float bcsr_time = mixed_tf32_spmm_bcsr(bcsr, gpu_node_capacity, feature_dim, id_map, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr, d_MatC, stream_bcsr);
    //float bcsc_time = mixed_tf32_spmm_bcsc(bcsc, gpu_node_capacity, feature_dim, id_map, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr, d_MatC, stream_bcsc);
    float bcsr_time = tf32_spmm_bcsr(bcsr, feature_dim, d_MatB, d_MatC, stream_bcsr);
    float bcsc_time = tf32_spmm_bcsc(bcsc, feature_dim, d_MatB, d_MatC, stream_bcsc);

    //#define output_denseC_size
    #ifdef output_denseC_size
    std::string graph_attri_path = base_path + "shape/" + std::to_string(device_id) + "_1_layer.csv";
    std::ofstream graph_attri_ofs(graph_attri_path, std::ios::out | std::ios::app); 

    if (!graph_attri_ofs.is_open()) {
        std::cerr << "Failed to open output file: " << graph_attri_path << "\n";
        return;
    }

    //graph_attri_ofs << "numRows" << ", " << "numCols" << ", " << "numNnz" << ", " << "numTCBlocks" << ", " << "numRowWindows" << ", " << "denseC_size" << "\n";
    graph_attri_ofs << numRows << ", " << numCols << ", " << numNnz << ", " << numTCBlocks << ", " << numRowWindows << ", " << denseC_size << "\n";

    #endif
    #undef output_denseC_size

    // 结果写入文件
    #ifdef output_result_to_file
    // 矩阵 shape
    std::string graph_attri_path = base_path + "/shape/" + std::to_string(device_id) + ".csv";
    std::ofstream graph_attri_ofs(graph_attri_path, std::ios::out | std::ios::app);

    if (!graph_attri_ofs.is_open()) {
        std::cerr << "Failed to open output file: " << graph_attri_path << "\n";
        return;
    }

    graph_attri_ofs << numRows << ", " << numCols << ", " << numNnz << ", " << numTCBlocks << ", " << numRowWindows << "\n";

    // 时间
    std::string file_path = base_path + "/compute-memory/" + std::to_string(device_id) + ".csv";
    std::ofstream ofs(file_path, std::ios::out | std::ios::app);

    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << file_path << "\n";
        return;
    }

    ofs << bcsr_time << ", " << bcsc_time << "\n";
    #endif

    //#define debug_check_result_MatC
    #ifdef debug_check_result_MatC
    MAT_VAL_TYPE *h_DenseMatC = (MAT_VAL_TYPE*) malloc(denseC_size * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h_DenseMatC, d_MatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyDeviceToHost);
    print_denseC(h_DenseMatC, numRows, denseC_size / feature_dim, feature_dim);
    #endif
    
    // 同步操作后移
    //CHECK_LAST_CUDA_ERROR();
    //cudaDeviceSynchronize();
}

// 非第一层 SpMM 计算
void launch_spmm_kernel(
    const BCSR<MAT_VAL_TYPE> &bcsr,
    const BCSC<MAT_VAL_TYPE> &bcsc,
    MAT_VAL_TYPE* d_MatB,
    MAT_VAL_TYPE* &d_MatC,
    int32_t feature_dim,
    const int32_t device_id,
    cudaStream_t stream_bcsr,
    cudaStream_t stream_bcsc
) {
    //printf("[Info] second layer launch_spmm_kernel\n");
    vint numRows       = bcsr.num_rows;
    vint numCols       = bcsr.num_cols;
    vint numNnz        = bcsr.nnz;
    vint numTCBlocks     = bcsr.num_TCBlocks;     // TC 块数量
    vint numRowWindows    = bcsr.num_rowWindows;   // row_window 的数量

    // 开辟 denseC 空间
    //vint denseC_size = std::max(rowWndSize * ROW_WINDOW, bcsc.rowIndices.back() + 8) * feature_dim;
    vint denseC_size = (numRowWindows + 1) * ROW_WINDOW * feature_dim;
    //cudaMalloc(&d_MatC, sizeof(MAT_VAL_TYPE) * denseC_size);
    cudaMemset(d_MatC, 0, denseC_size * sizeof(MAT_VAL_TYPE));


    // BCSR & BCSC 计算
    float bcsr_time = tf32_spmm_bcsr(bcsr, feature_dim, d_MatB, d_MatC, stream_bcsr);
    float bcsc_time = tf32_spmm_bcsc(bcsc, feature_dim, d_MatB, d_MatC, stream_bcsc);

    //#define output_denseC_size
    #ifdef output_denseC_size
    std::string graph_attri_path = base_path + "shape/" + std::to_string(device_id) + "_2_layer.csv";
    std::ofstream graph_attri_ofs(graph_attri_path, std::ios::out | std::ios::app); 

    if (!graph_attri_ofs.is_open()) {
        std::cerr << "Failed to open output file: " << graph_attri_path << "\n";
        return;
    }

    //graph_attri_ofs << "numRows" << ", " << "numCols" << ", " << "numNnz" << ", " << "numTCBlocks" << ", " << "numRowWindows" << ", " << "denseC_size" << "\n";
    graph_attri_ofs << numRows << ", " << numCols << ", " << numNnz << ", " << numTCBlocks << ", " << numRowWindows << ", " << denseC_size << "\n";

    #endif
    #undef output_denseC_size

    // 结果写入文件
    #ifdef output_result_to_file
    // 时间
    std::string file_path = base_path + "/compute/" + std::to_string(device_id) + ".csv";
    std::ofstream ofs(file_path, std::ios::out | std::ios::app);
    
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << file_path << "\n";
        return;
    }
    
    ofs << bcsr_time << ", " << bcsc_time << "\n";
    #endif

    //#define debug_check_result_MatC
    #ifdef debug_check_result_MatC
    MAT_VAL_TYPE *h_DenseMatC = (MAT_VAL_TYPE*) malloc(denseC_size * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(h_DenseMatC, d_MatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyDeviceToHost);
    print_denseC(h_DenseMatC, numRows, denseC_size / feature_dim, feature_dim);
    #endif
    
    // 同步操作后移
    //CHECK_LAST_CUDA_ERROR();
    //cudaDeviceSynchronize();
}
