#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>

#include "spmm_dispatcher.h"

#define origin

#ifdef origin
// ======================================= gnn first layer =======================================

// gcn first layer bcsr compute
__host__
float mixed_tf32_spmm_bcsr(
    const BCSR<MAT_VAL_TYPE>& bcsr, 
    const int32_t gpu_node_capacity, 
    const int32_t feature_dim,
    int32_t *id_map,
    float* cpu_feature_cache_ptr,
    int32_t* cache_search_map,
    float** gpu_feature_cache_ptr,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
) {
    //printf("\n[Info] mixed memory access BCSR SpMM\n");
    //std::cout << std::endl;

    //bool wait_for_debugger = true;
    //while(wait_for_debugger);

    vint numRows       = bcsr.num_rows;
    vint numCols       = bcsr.num_cols;
    vint numNnz        = bcsr.nnz;
    vint numTCBlocks     =   bcsr.num_TCBlocks;     // TC 块数量
    vint numRowWindows    =   bcsr.num_rowWindows;   // row_window 的数量

    // ================================================== GPU 数据加载 ==================================================
    // SparseA 数据加载
    vint           *d_rowWindowOffset = bcsr.rowWindowOffset;
    MAT_PTR_TYPE   *d_tc_offset = bcsr.tcOffset;
    MAT_PTR_TYPE   *d_sparseA2B = bcsr.sparseA2B;
    TCLOCAL_TYPE   *d_tcLocalBit = bcsr.tcLocalBit;

    // ================================================== 负载均衡判断 ==================================================
    bool load_balance = false;
    
    // ================================================== 计算 ==================================================
    //GpuTimer timer;

    vint threshold =  load_balance ? 128 : 512;

    int off = (feature_dim <= threshold) ? 1 : 2;
    int warpsPerBlk = feature_dim / (COL_WINDOW_R << off);

    dim3 grid_size(numRowWindows, 1, 1); // 每一个 block 对应一个 row_window，满足了对于 A 矩阵的访问需求
    dim3 block_size(WARP_SIZE, warpsPerBlk, 1);

    if(feature_dim <= threshold) {        
        //timer.Start();

        tf32_computeX128TransposePipe2_BCSR<<<grid_size, block_size, 0, (stream_hdl)>>>(
            d_rowWindowOffset, d_tc_offset, d_sparseA2B, d_tcLocalBit,
            id_map, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
            d_DenseMatC,
            numRows, numCols, gpu_node_capacity, feature_dim);

        //timer.Stop();
        // 同步操作后移
        //cudaDeviceSynchronize();
    } else {
    } 
    
    // ================================================== 计算结果输出 ==================================================
    //float elapsed_time = timer.Elapsed() / EXE_TIME;
    float elapsed_time = 0.0f;
    //printf("Elapsed time: %8.4lf ms\n", elapsed_time);

    //CHECK_LAST_CUDA_ERROR();
    
    #ifdef debug_output
    float spmm_flop = float(coo->nnz) * float(feature_dim) * 2.0;
    float throughput_ = (float(spmm_flop * 1000.00)) / (elapsed_time * 1000. * 1000. * 1000.);
    std::cout << feature_dim << "," << elapsed_time << "," << throughput_ << "\n";
    #endif

    return elapsed_time;
}

// gcn first layer bcsc compute
__host__
float mixed_tf32_spmm_bcsc(
    const BCSC<MAT_VAL_TYPE>& bcsc, 
    const int32_t gpu_node_capacity, 
    const int32_t feature_dim,
    int32_t *id_map,
    float* cpu_feature_cache_ptr,
    int32_t* cache_search_map,
    float** gpu_feature_cache_ptr,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
) {
    //printf("\n[Info] mixed memory access BCSC SpMM\n");

    vint numRows       = bcsc.num_rows;
    vint numCols       = bcsc.num_cols;
    vint numNnz        = bcsc.nnz;
    vint numTCBlocks     =   bcsc.num_TCBlocks;     // TC 块数量
    vint numColWindows    =   bcsc.num_colWindows;   // row_window 的数量

    // ================================================== GPU 数据加载 ==================================================
    // SparseA 数据加载
    vint           *d_colWindowOffset = bcsc.colWindowOffset;
    MAT_PTR_TYPE   *d_tc_offset = bcsc.tcOffset;
    MAT_PTR_TYPE   *d_sparseA2C = bcsc.sparseA2C;
    TCLOCAL_TYPE   *d_tcLocalBit = bcsc.tcLocalBit;

    // ================================================== 负载均衡判断 ==================================================
    bool load_balance = false;
    
    // ================================================== 计算 ==================================================
    //GpuTimer timer;

    vint threshold =  load_balance ? 128 : 512;

    int off = (feature_dim <= threshold) ? 1 : 2;
    int warpsPerBlk = feature_dim / (COL_WINDOW_R << off);

    dim3 grid_size(numColWindows, 1, 1); // 每一个 block 对应一个 row_window，满足了对于 A 矩阵的访问需求
    dim3 block_size(WARP_SIZE, warpsPerBlk, 1);

    if(feature_dim <= threshold) {        
        //timer.Start();

        tf32_computeX128TransposePipe2_BCSC<<<grid_size, block_size, 0, (stream_hdl)>>>(
            d_colWindowOffset, d_tc_offset, d_sparseA2C, d_tcLocalBit, 
            id_map, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
            d_DenseMatC,
            numRows, numCols, gpu_node_capacity, feature_dim);

        //timer.Stop();
        // 同步操作后移
        //cudaDeviceSynchronize();
    } else {
    } 
    
    // ================================================== 计算结果输出 ==================================================
    //float elapsed_time = timer.Elapsed() / EXE_TIME;
    float elapsed_time = 0.0f;
    //printf("Elapsed time: %8.4lf ms\n", elapsed_time);

    //CHECK_LAST_CUDA_ERROR();
    
    #ifdef debug_output
    float spmm_flop = float(coo->nnz) * float(feature_dim) * 2.0;
    float throughput_ = (float(spmm_flop * 1000.00)) / (elapsed_time * 1000. * 1000. * 1000.);
    std::cout << feature_dim << "," << elapsed_time << "," << throughput_ << "\n";
    #endif

    return elapsed_time;
}

// ======================================= gnn second layer =======================================

// gcn second layer bcsr compute
__host__
float tf32_spmm_bcsr(
    const BCSR<MAT_VAL_TYPE>& bcsr, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
) {
    //printf("\n[Info] normal BCSR SpMM\n");

    vint numRows       = bcsr.num_rows;
    vint numCols       = bcsr.num_cols;
    vint numNnz        = bcsr.nnz;
    vint numTCBlocks     =   bcsr.num_TCBlocks;     // TC 块数量
    vint numRowWindows    =   bcsr.num_rowWindows;   // row_window 的数量

    // ================================================== GPU 数据加载 ==================================================
    // SparseA 数据加载
    vint           *d_rowWindowOffset = bcsr.rowWindowOffset;
    MAT_PTR_TYPE   *d_tc_offset = bcsr.tcOffset;
    MAT_PTR_TYPE   *d_sparseA2B = bcsr.sparseA2B;
    TCLOCAL_TYPE   *d_tcLocalBit = bcsr.tcLocalBit;

    // ================================================== 负载均衡判断 ==================================================
    bool load_balance = false;
    
    // ================================================== 计算 ==================================================
    //GpuTimer timer;

    vint threshold =  load_balance ? 128 : 512;

    int off = (feature_dim <= threshold) ? 1 : 2;
    int warpsPerBlk = feature_dim / (COL_WINDOW_R << off);

    dim3 grid_size(numRowWindows, 1, 1); // 每一个 block 对应一个 row_window，满足了对于 A 矩阵的访问需求
    dim3 block_size(WARP_SIZE, warpsPerBlk, 1);

    if(feature_dim <= threshold) {        
        //timer.Start();

        tf32_computeX128TransposePipe2_BCSR<<<grid_size, block_size, 0, (stream_hdl)>>>(
            d_rowWindowOffset, d_tc_offset, d_sparseA2B, d_tcLocalBit, 
            d_DenseMatB,
            d_DenseMatC,
            numRows, numCols, feature_dim);

        //timer.Stop();
        // 同步操作后移
        //cudaDeviceSynchronize();
    } else {
    } 
    
    // ================================================== 计算结果输出 ==================================================
    //float elapsed_time = timer.Elapsed() / EXE_TIME;
    float elapsed_time = 0.0f;
    //printf("Elapsed time: %8.4lf ms\n", elapsed_time);

    //CHECK_LAST_CUDA_ERROR();
    
    #ifdef debug_output
    float spmm_flop = float(coo->nnz) * float(feature_dim) * 2.0;
    float throughput_ = (float(spmm_flop * 1000.00)) / (elapsed_time * 1000. * 1000. * 1000.);
    std::cout << feature_dim << "," << elapsed_time << "," << throughput_ << "\n";
    #endif

    return elapsed_time;
}

// gcn second layer bcsc compute
__host__
float tf32_spmm_bcsc(
    const BCSC<MAT_VAL_TYPE>& bcsc, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
) {
    //printf("\n[Info] normal BCSC SpMM\n");

    vint numRows       = bcsc.num_rows;
    vint numCols       = bcsc.num_cols;
    vint numNnz        = bcsc.nnz;
    vint numTCBlocks     =   bcsc.num_TCBlocks;     // TC 块数量
    vint numColWindows    =   bcsc.num_colWindows;   // row_window 的数量

    // ================================================== GPU 数据加载 ==================================================
    // SparseA 数据加载
    vint           *d_colWindowOffset = bcsc.colWindowOffset;
    MAT_PTR_TYPE   *d_tc_offset = bcsc.tcOffset;
    MAT_PTR_TYPE   *d_sparseA2C = bcsc.sparseA2C;
    TCLOCAL_TYPE   *d_tcLocalBit = bcsc.tcLocalBit;

    // ================================================== 负载均衡判断 ==================================================
    bool load_balance = false;
    
    // ================================================== 计算 ==================================================
    //GpuTimer timer;

    vint threshold =  load_balance ? 128 : 512;

    int off = (feature_dim <= threshold) ? 1 : 2;
    int warpsPerBlk = feature_dim / (COL_WINDOW_R << off);

    dim3 grid_size(numColWindows, 1, 1); // 每一个 block 对应一个 row_window，满足了对于 A 矩阵的访问需求
    dim3 block_size(WARP_SIZE, warpsPerBlk, 1);

    if(feature_dim <= threshold) {        
        //timer.Start();

        tf32_computeX128TransposePipe2_BCSC<<<grid_size, block_size, 0, (stream_hdl)>>>(
            d_colWindowOffset, d_tc_offset, d_sparseA2C, d_tcLocalBit, 
            d_DenseMatB,
            d_DenseMatC,
            numRows, numCols, feature_dim);

        //timer.Stop();
        // 同步操作后移
        //cudaDeviceSynchronize();
    } else {
    } 
    
    // ================================================== 计算结果输出 ==================================================
    //float elapsed_time = timer.Elapsed() / EXE_TIME;
    float elapsed_time = 0.0f;
    //printf("Elapsed time: %8.4lf ms\n", elapsed_time);

    //CHECK_LAST_CUDA_ERROR();
    
    #ifdef debug_output
    float spmm_flop = float(coo->nnz) * float(feature_dim) * 2.0;
    float throughput_ = (float(spmm_flop * 1000.00)) / (elapsed_time * 1000. * 1000. * 1000.);
    std::cout << feature_dim << "," << elapsed_time << "," << throughput_ << "\n";
    #endif

    return elapsed_time;
}
#endif

#ifdef unified
// ======================================= gnn first layer =======================================

// gcn first layer bcsr compute
__host__
float mixed_tf32_spmm_bcsr(
    const BCSR<MAT_VAL_TYPE>& bcsr, 
    const int32_t gpu_node_capacity, 
    const int32_t feature_dim,
    int32_t *id_map,
    float* cpu_feature_cache_ptr,
    int32_t* cache_search_map,
    float** gpu_feature_cache_ptr,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
) {
    //printf("\n[Info] mixed memory access BCSR SpMM\n");
    //std::cout << std::endl;

    //bool wait_for_debugger = true;
    //while(wait_for_debugger);

    vint numRows       = bcsr.num_rows;
    vint numCols       = bcsr.num_cols;
    vint numNnz        = bcsr.nnz;
    vint numTCBlocks     =   bcsr.num_TCBlocks;     // TC 块数量
    vint numRowWindows    =   bcsr.num_rowWindows;   // row_window 的数量

    // ================================================== GPU 数据加载 ==================================================
    // SparseA 数据加载
    vint           *d_rowWindowOffset = bcsr.rowWindowOffset;
    MAT_PTR_TYPE   *d_tc_offset = bcsr.tcOffset;
    MAT_PTR_TYPE   *d_sparseA2B = bcsr.sparseA2B;
    TCLOCAL_TYPE   *d_tcLocalBit = bcsr.tcLocalBit;

    // ================================================== 负载均衡判断 ==================================================
    bool load_balance = false;
    
    // ================================================== 计算 ==================================================
    //GpuTimer timer;

    vint threshold =  load_balance ? 128 : 512;

    int off = (feature_dim <= threshold) ? 1 : 2;
    int warpsPerBlk = feature_dim / (COL_WINDOW_R << off);

    dim3 grid_size(numRowWindows, 1, 1); // 每一个 block 对应一个 row_window，满足了对于 A 矩阵的访问需求
    dim3 block_size(WARP_SIZE, warpsPerBlk, 1);

    if(feature_dim <= threshold) {        
        //timer.Start();

        tf32_computeX128TransposePipe2<<<grid_size, block_size, 0, (stream_hdl)>>>(
            d_rowWindowOffset, d_tc_offset, d_sparseA2B, d_tcLocalBit,
            id_map, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
            d_DenseMatC,
            numRows, numCols, gpu_node_capacity, feature_dim, 0);

        //timer.Stop();
        // 同步操作后移
        //cudaDeviceSynchronize();
    } else {
    } 
    
    // ================================================== 计算结果输出 ==================================================
    //float elapsed_time = timer.Elapsed() / EXE_TIME;
    float elapsed_time = 0.0f;
    //printf("Elapsed time: %8.4lf ms\n", elapsed_time);

    //CHECK_LAST_CUDA_ERROR();
    
    #ifdef debug_output
    float spmm_flop = float(coo->nnz) * float(feature_dim) * 2.0;
    float throughput_ = (float(spmm_flop * 1000.00)) / (elapsed_time * 1000. * 1000. * 1000.);
    std::cout << feature_dim << "," << elapsed_time << "," << throughput_ << "\n";
    #endif

    return elapsed_time;
}

// gcn first layer bcsc compute
__host__
float mixed_tf32_spmm_bcsc(
    const BCSC<MAT_VAL_TYPE>& bcsc, 
    const int32_t gpu_node_capacity, 
    const int32_t feature_dim,
    int32_t *id_map,
    float* cpu_feature_cache_ptr,
    int32_t* cache_search_map,
    float** gpu_feature_cache_ptr,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
) {
    //printf("\n[Info] mixed memory access BCSC SpMM\n");

    vint numRows       = bcsc.num_rows;
    vint numCols       = bcsc.num_cols;
    vint numNnz        = bcsc.nnz;
    vint numTCBlocks     =   bcsc.num_TCBlocks;     // TC 块数量
    vint numColWindows    =   bcsc.num_colWindows;   // row_window 的数量

    // ================================================== GPU 数据加载 ==================================================
    // SparseA 数据加载
    vint           *d_colWindowOffset = bcsc.colWindowOffset;
    MAT_PTR_TYPE   *d_tc_offset = bcsc.tcOffset;
    MAT_PTR_TYPE   *d_sparseA2C = bcsc.sparseA2C;
    TCLOCAL_TYPE   *d_tcLocalBit = bcsc.tcLocalBit;

    // ================================================== 负载均衡判断 ==================================================
    bool load_balance = false;
    
    // ================================================== 计算 ==================================================
    //GpuTimer timer;

    vint threshold =  load_balance ? 128 : 512;

    int off = (feature_dim <= threshold) ? 1 : 2;
    int warpsPerBlk = feature_dim / (COL_WINDOW_R << off);

    dim3 grid_size(numColWindows, 1, 1); // 每一个 block 对应一个 row_window，满足了对于 A 矩阵的访问需求
    dim3 block_size(WARP_SIZE, warpsPerBlk, 1);

    if(feature_dim <= threshold) {        
        //timer.Start();

        tf32_computeX128TransposePipe2<<<grid_size, block_size, 0, (stream_hdl)>>>(
            d_colWindowOffset, d_tc_offset, d_sparseA2C, d_tcLocalBit, 
            id_map, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
            d_DenseMatC,
            numRows, numCols, gpu_node_capacity, feature_dim, 1);

        //timer.Stop();
        // 同步操作后移
        //cudaDeviceSynchronize();
    } else {
    } 
    
    // ================================================== 计算结果输出 ==================================================
    //float elapsed_time = timer.Elapsed() / EXE_TIME;
    float elapsed_time = 0.0f;
    //printf("Elapsed time: %8.4lf ms\n", elapsed_time);

    //CHECK_LAST_CUDA_ERROR();
    
    #ifdef debug_output
    float spmm_flop = float(coo->nnz) * float(feature_dim) * 2.0;
    float throughput_ = (float(spmm_flop * 1000.00)) / (elapsed_time * 1000. * 1000. * 1000.);
    std::cout << feature_dim << "," << elapsed_time << "," << throughput_ << "\n";
    #endif

    return elapsed_time;
}

// ======================================= gnn second layer =======================================

// gcn second layer bcsr compute
__host__
float tf32_spmm_bcsr(
    const BCSR<MAT_VAL_TYPE>& bcsr, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
) {
    //printf("\n[Info] normal BCSR SpMM\n");

    vint numRows       = bcsr.num_rows;
    vint numCols       = bcsr.num_cols;
    vint numNnz        = bcsr.nnz;
    vint numTCBlocks     =   bcsr.num_TCBlocks;     // TC 块数量
    vint numRowWindows    =   bcsr.num_rowWindows;   // row_window 的数量

    // ================================================== GPU 数据加载 ==================================================
    // SparseA 数据加载
    vint           *d_rowWindowOffset = bcsr.rowWindowOffset;
    MAT_PTR_TYPE   *d_tc_offset = bcsr.tcOffset;
    MAT_PTR_TYPE   *d_sparseA2B = bcsr.sparseA2B;
    TCLOCAL_TYPE   *d_tcLocalBit = bcsr.tcLocalBit;

    // ================================================== 负载均衡判断 ==================================================
    bool load_balance = false;
    
    // ================================================== 计算 ==================================================
    //GpuTimer timer;

    vint threshold =  load_balance ? 128 : 512;

    int off = (feature_dim <= threshold) ? 1 : 2;
    int warpsPerBlk = feature_dim / (COL_WINDOW_R << off);

    dim3 grid_size(numRowWindows, 1, 1); // 每一个 block 对应一个 row_window，满足了对于 A 矩阵的访问需求
    dim3 block_size(WARP_SIZE, warpsPerBlk, 1);

    if(feature_dim <= threshold) {        
        //timer.Start();

        tf32_computeX128TransposePipe2<<<grid_size, block_size, 0, (stream_hdl)>>>(
            d_rowWindowOffset, d_tc_offset, d_sparseA2B, d_tcLocalBit, 
            d_DenseMatB,
            d_DenseMatC,
            numRows, numCols, feature_dim, 0);

        //timer.Stop();
        // 同步操作后移
        //cudaDeviceSynchronize();
    } else {
    } 
    
    // ================================================== 计算结果输出 ==================================================
    //float elapsed_time = timer.Elapsed() / EXE_TIME;
    float elapsed_time = 0.0f;
    //printf("Elapsed time: %8.4lf ms\n", elapsed_time);

    //CHECK_LAST_CUDA_ERROR();
    
    #ifdef debug_output
    float spmm_flop = float(coo->nnz) * float(feature_dim) * 2.0;
    float throughput_ = (float(spmm_flop * 1000.00)) / (elapsed_time * 1000. * 1000. * 1000.);
    std::cout << feature_dim << "," << elapsed_time << "," << throughput_ << "\n";
    #endif

    return elapsed_time;
}

// gcn second layer bcsc compute
__host__
float tf32_spmm_bcsc(
    const BCSC<MAT_VAL_TYPE>& bcsc, 
    const int32_t feature_dim,
    MAT_VAL_TYPE* d_DenseMatB,
    MAT_VAL_TYPE* &d_DenseMatC,
    cudaStream_t stream_hdl
) {
    //printf("\n[Info] normal BCSC SpMM\n");

    vint numRows       = bcsc.num_rows;
    vint numCols       = bcsc.num_cols;
    vint numNnz        = bcsc.nnz;
    vint numTCBlocks     =   bcsc.num_TCBlocks;     // TC 块数量
    vint numColWindows    =   bcsc.num_colWindows;   // row_window 的数量

    // ================================================== GPU 数据加载 ==================================================
    // SparseA 数据加载
    vint           *d_colWindowOffset = bcsc.colWindowOffset;
    MAT_PTR_TYPE   *d_tc_offset = bcsc.tcOffset;
    MAT_PTR_TYPE   *d_sparseA2C = bcsc.sparseA2C;
    TCLOCAL_TYPE   *d_tcLocalBit = bcsc.tcLocalBit;

    // ================================================== 负载均衡判断 ==================================================
    bool load_balance = false;
    
    // ================================================== 计算 ==================================================
    //GpuTimer timer;

    vint threshold =  load_balance ? 128 : 512;

    int off = (feature_dim <= threshold) ? 1 : 2;
    int warpsPerBlk = feature_dim / (COL_WINDOW_R << off);

    dim3 grid_size(numColWindows, 1, 1); // 每一个 block 对应一个 row_window，满足了对于 A 矩阵的访问需求
    dim3 block_size(WARP_SIZE, warpsPerBlk, 1);

    if(feature_dim <= threshold) {        
        //timer.Start();

        tf32_computeX128TransposePipe2<<<grid_size, block_size, 0, (stream_hdl)>>>(
            d_colWindowOffset, d_tc_offset, d_sparseA2C, d_tcLocalBit, 
            d_DenseMatB,
            d_DenseMatC,
            numRows, numCols, feature_dim, 1);

        //timer.Stop();
        // 同步操作后移
        //cudaDeviceSynchronize();
    } else {
    } 
    
    // ================================================== 计算结果输出 ==================================================
    //float elapsed_time = timer.Elapsed() / EXE_TIME;
    float elapsed_time = 0.0f;
    //printf("Elapsed time: %8.4lf ms\n", elapsed_time);

    //CHECK_LAST_CUDA_ERROR();
    
    #ifdef debug_output
    float spmm_flop = float(coo->nnz) * float(feature_dim) * 2.0;
    float throughput_ = (float(spmm_flop * 1000.00)) / (elapsed_time * 1000. * 1000. * 1000.);
    std::cout << feature_dim << "," << elapsed_time << "," << throughput_ << "\n";
    #endif

    return elapsed_time;
}
#endif