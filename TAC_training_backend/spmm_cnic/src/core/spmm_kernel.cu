#include "spmm_kernel.h"
#include "ptx_tf32.h"

// ======================================= gnn first layer =======================================

// gcn first layer hot data(bcsr) SpMM
__global__
void tf32_computeX128TransposePipe2_BCSR(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2B,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    int32_t* id_map,
    float* cpu_float_features,
    int32_t* cache_search_map,
    float** gpu_float_feature,
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    int32_t gpu_node_capacity,
    const vint feature_dim
) {
    //if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
    //    printf("[Info] mixed memory access bcsr SpMM compute operator\n");
    //}

    // ====================================== 定义所需的寄存器、共享内存、线程和块相关变量 ======================================
    using ARegisters = MAT_VAL_TYPE[2];     // 8 * 8
    using BRegisters = MAT_VAL_TYPE[4];     // 算 2 个 m16n8k8，共用一个 A 16 * 8
    using CRegisters = MAT_VAL_TYPE[2][4];  // 16 * 8
    
    // 当前 MMA 所需数据
    ARegisters fragA;
    BRegisters fragB00;
    BRegisters fragB01;
    CRegisters fragC = {0.0};

    // 下一次 MMA 预取数据
    BRegisters fragB10;
    BRegisters fragB11;

    vint bid                  =   blockIdx.x;
    vint offY                 =   (blockIdx.y << 7);
    const vint laneid         =   31 & threadIdx.x; // threadIdx.x % warpSize(32)，但是目前 blockDim.x == 32，实际上 threadIdx.x 并不会大于 32，因此 landid == threadIdx.x
    const vint warpSize       =   32;
    const vint tid            =   threadIdx.y * warpSize + laneid; // 全局线程 ID
    const vint local_warpID   =   threadIdx.y;

    // ====================================== 确定每个线程组在矩阵块中的角色 ======================================
    vint groupID         =   laneid >> 2;   // laneid / 4
    vint tID_in_group    =   3 & laneid;    // laneid % 4

    // sparseA 原始数据是 row-major 的，这里按照 row-major 的方式进行索引
    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;      // 0, 1, 2, 3
    vint colA1           =   tID_in_group + 4;  // 4, 5, 6, 7

    // 对 denseB 原始数据访问行短列长（因为行是由 sparse A TC 块中非零元所在列确定的)
    vint colB02          =   groupID + (local_warpID << 5); // local_warpID << 5 <=> local_warpID * 32 <=> threadIdx.y * 32
    vint colB13          =   groupID + (local_warpID << 5) + 8;
    // 这个访问 B 的行号和访问 A 的列号是相同的
    vint row01           =   tID_in_group;      // 0, 1, 2, 3
    vint row23           =   tID_in_group + 4;  // 4, 5, 6, 7
    
    constexpr const int inst_k  = 8;
    constexpr const int inst_n  = 8;

    const vint mat_len = 64;
    const vint idx_len = 8;
    vint  local_idx    = 0;

    // ====================================== 初始化共享内存地址，读取块范围 ======================================
    // 均开设两倍空间，用于数据预取
    __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
    __shared__ vint         d_sharedSparseA2B[2 * idx_len];
    
    // 当 CUDA 设备函数接收到一个 generic 指针 时，它可能是 global、shared、local 中的任意一种地址
    // 异步拷贝需要使用共享内存地址空间的指针，泛型地址无法自动识别为共享地址空间，使用 __cvta_generic_to_shared 进行显式转换
    vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
    vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
    
    MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];     // 当前 thread 所处 block 对应的 TC 块的起始索引
    MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid + 1];   // 当前 thread 所处 block 对应的 TC 块的结束索引

    // ====================================== 第一次 MMA 所需 A 矩阵数据(sparseA 和 sparseA2B)加载(g->s) ======================================
    // 一个 block 中的前 64 个 thread 预取稀疏矩阵 A（一个 row_window 对应一个 block，所以 block 逐一处理所属 row_window 中的 TC 块)，这里就是解压缩过程
    if(tid < mat_len) {  
        TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx]; // 当前 TC 块的 bitmap
        vint start_dataIdx         = d_data2Idx[start_blk_idx];   // 当前 TC 块中非零元起始偏移

        // 每个 thread 对应 8*8 TC 块的一个元素，判断当前位置是否为非零元
        if(present_local & (1ULL << tid))
            local_idx = __popcll(present_local << (63 - tid));  // 计算包含当前位置的之前总共的非零元的个数，用于 data 索引

        // prefetch 1 tc_block
        if(local_idx == 0) {
            d_sharedSparseA[tid] = 0.0;
        } else {
            // FIXME 不确定是否有问题，统一修改为直接赋值
            d_sharedSparseA[tid] = 1.0f;
            //d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
        }
    }

    // 读取 2 个 TC 块的 sparseA2B 数据
    if(tid < inst_k) {
        d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid); // offset = start_blk_idx * 8 + tid，因为每个 block 在 sparseA2B 中都存在 8 个数据，所以用块数 * 每块列数即为当前块包含列的初始位置

        // 如果当前 row_window 包含的 TC 块数量 >= 2，那么预取下一个 TC 块的索引
        if(start_blk_idx + 1 < end_blk_idx) {
            d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
        }

    }
    __syncthreads();

    // ====================================== 第一次 MMA 所需 denseB transpose mapping 数据加载(g->r) ======================================
    vint dense_rowIdx01 = d_sharedSparseA2B[row01];
    vint dense_rowIdx23 = d_sharedSparseA2B[row23];

    if(dense_rowIdx01 >= numCols) { // 处理 row_window 最后一个 TC 块补列的情况
        fragB00[0] = 0.0; fragB00[1] = 0.0; 
        fragB01[0] = 0.0; fragB01[1] = 0.0;
    } else {
        if (dense_rowIdx01 < 0) {
            printf("[Error] first layer bcsr dense_rowIdx01 == 0\n");
            return;
        }
        // 缓存情况判定
        int32_t gidx = (cache_search_map[dense_rowIdx01]);   // 当前 thread 对应节点缓存所在 clique 中的编号

        if(gidx < 0) {/*cache miss*/
            // 列实际时映射为真实列号(只有在 cache miss 需要访问 cpu 数据时，才需要映射为实际的图上节点编号)
            dense_rowIdx01 = id_map[dense_rowIdx01];

            if (dense_rowIdx01 >= 0) {
                const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx01) * feature_dim)];

                fragB00[0] = load_fp32_from_global(d_MatB + colB02);
                fragB00[1] = load_fp32_from_global(d_MatB + colB13);
                fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
            }
		}else { /*cache hit, find global position*/
		    int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
		    int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
            const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置

            fragB00[0] = load_fp32_from_global(d_MatB + colB02);
            fragB00[1] = load_fp32_from_global(d_MatB + colB13);
            fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
            fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
		}
    }

    if(dense_rowIdx23 >= numCols) {
        fragB00[2] = 0.0; fragB00[3] = 0.0; 
        fragB01[2] = 0.0; fragB01[3] = 0.0;
    } else {
        if (dense_rowIdx23 < 0) {
            printf("[Error] first layer bcsr dense_rowIdx23 == 0\n");
            return;
        }

        int32_t gidx = (cache_search_map[dense_rowIdx23]);   // 当前 thread 对应节点缓存所在 clique 中的编号

		if(gidx < 0) {/*cache miss*/
            // 列实际时映射为真实列号
            dense_rowIdx23 = id_map[dense_rowIdx23];

            if (dense_rowIdx23 >= 0) {
                const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx23) * feature_dim)];

                fragB00[2] = load_fp32_from_global(d_MatB + colB02);
                fragB00[3] = load_fp32_from_global(d_MatB + colB13);
                fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
            }
		}else { /*cache hit, find global position*/
		    int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
		    int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
            const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置

            fragB00[2] = load_fp32_from_global(d_MatB + colB02);
            fragB00[3] = load_fp32_from_global(d_MatB + colB13);
            fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
            fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
		}
    } 

    __syncthreads();

    // ====================================== 遍历所有块，进行稀疏矩阵 A 和稠密矩阵 B 的乘法 ======================================
    for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
        // select which buffer to read，block 内所有 thread 计算出的结果都是一样的
        // 标识 d_sharedSparseA 的起始地址（双 buffer，一个存储区域逻辑上分为两部分，sel_shm 和 sel_shm_next 分别指向这两部分的起始偏移)
        vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 当前 TC 块对应的 sharedSparseA 的起始地址
        vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;      // 下一个 TC 块对应的 sharedSparseA 的起始地址
        // 标识 d_sharedSparseA2B 的起始地址
        vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;   // 当前 TC 块(当前是相对于 tc_block 而言的）对应的 sharedSparseA2B 的起始地址 
        vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

        // 1.数据预取
        // 1.1 下一次 MMA 所需 denseB transpose mapping 数据预取(g->r)
        vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
        vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];

        if(sel_shm_next) {
            if(dense_rowIdx101 > numCols) {
                fragB10[0] = 0.0; fragB10[1] = 0.0; 
                fragB11[0] = 0.0; fragB11[1] = 0.0;
            } else {
                if (dense_rowIdx101 < 0) {
                    printf("[Error] first layer bcsr dense_rowIdx101 == 0\n");
                    return;
                }

                int32_t gidx = (cache_search_map[dense_rowIdx101]);   // 当前 thread 对应节点缓存所在 clique 中的编号

                if(gidx < 0) {/*cache miss*/
                    // 列号映射
                    dense_rowIdx101 = id_map[dense_rowIdx101];

                    if (dense_rowIdx101 >= 0) {
                        const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx101) * feature_dim)];
        
                        fragB10[0] = load_fp32_from_global(d_MatB + colB02);
                        fragB10[1] = load_fp32_from_global(d_MatB + colB13);
                        fragB11[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                        fragB11[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                    }
                }else { /*cache hit, find global position*/
                    int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
                    int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
                    // TODO 确定 &gpu_float_feature[didx][0] 是否会对访存造成影响（考虑 Legion 设计为二级指针数据的意图）
                    const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置
        
                    fragB10[0] = load_fp32_from_global(d_MatB + colB02);
                    fragB10[1] = load_fp32_from_global(d_MatB + colB13);
                    fragB11[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                    fragB11[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                }
            }
            if(dense_rowIdx123 > numCols) {
                fragB10[2] = 0.0; fragB10[3] = 0.0; 
                fragB11[2] = 0.0; fragB11[3] = 0.0;
            } else {
                if (dense_rowIdx123 < 0) {
                    printf("[Error] first layer bcsr dense_rowIdx123 == 0\n");
                    return;
                }

                int32_t gidx = (cache_search_map[dense_rowIdx123]);   // 当前 thread 对应节点缓存所在 clique 中的编号

                if(gidx < 0) {/*cache miss*/
                    // 列号映射
                    dense_rowIdx123 = id_map[dense_rowIdx123];

                    if (dense_rowIdx123 >= 0) {
                        const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx123) * feature_dim)];
        
                        fragB10[2] = load_fp32_from_global(d_MatB + colB02);
                        fragB10[3] = load_fp32_from_global(d_MatB + colB13);
                        fragB11[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                        fragB11[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                    }
                }else { /*cache hit, find global position*/
                    int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
                    int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
                    // TODO 确定 &gpu_float_feature[didx][0] 是否会对访存造成影响（考虑 Legion 设计为二级指针数据的意图）
                    const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置
        
                    fragB10[2] = load_fp32_from_global(d_MatB + colB02);
                    fragB10[3] = load_fp32_from_global(d_MatB + colB13);
                    fragB11[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                    fragB11[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                }
            }
        } else {
            if(dense_rowIdx101 > numCols) {
                fragB00[0] = 0.0; fragB00[1] = 0.0; 
                fragB01[0] = 0.0; fragB01[1] = 0.0;
            } else {
                if (dense_rowIdx101 < 0) {
                    printf("[Error] first layer bcsr dense_rowIdx101 == 0\n");
                    return;
                }

                int32_t gidx = (cache_search_map[dense_rowIdx101]);   // 当前 thread 对应节点缓存所在 clique 中的编号

                if(gidx < 0) {/*cache miss*/
                    // 列号映射
                    dense_rowIdx101 = id_map[dense_rowIdx101];

                    if (dense_rowIdx101 >= 0) {
                        const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx101) * feature_dim)];
        
                        fragB00[0] = load_fp32_from_global(d_MatB + colB02);
                        fragB00[1] = load_fp32_from_global(d_MatB + colB13);
                        fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                        fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                    }
                }else { /*cache hit, find global position*/
                    int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
                    int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号

                    const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置
        
                    fragB00[0] = load_fp32_from_global(d_MatB + colB02);
                    fragB00[1] = load_fp32_from_global(d_MatB + colB13);
                    fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                    fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                }
            }
            if(dense_rowIdx123 > numCols) {
                fragB00[2] = 0.0; fragB00[3] = 0.0; 
                fragB01[2] = 0.0; fragB01[3] = 0.0;
            } else {
                if (dense_rowIdx123 < 0) {
                    printf("[Error] first layer bcsr dense_rowIdx123 == 0\n");
                    return;
                }

                int32_t gidx = (cache_search_map[dense_rowIdx123]);   // 当前 thread 对应节点缓存所在 clique 中的编号

                if(gidx < 0) {/*cache miss*/
                    // 列号映射
                    dense_rowIdx123 = id_map[dense_rowIdx123];

                    if (dense_rowIdx123 >= 0) {
                        const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx123) * feature_dim)];
        
                        fragB00[2] = load_fp32_from_global(d_MatB + colB02);
                        fragB00[3] = load_fp32_from_global(d_MatB + colB13);
                        fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                        fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                    }
                }else { /*cache hit, find global position*/
                    int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
                    int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号

                    const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置
        
                    fragB00[2] = load_fp32_from_global(d_MatB + colB02);
                    fragB00[3] = load_fp32_from_global(d_MatB + colB13);
                    fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                    fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                }
            }
        }   // end if(sel_shm_next)
    
        // 1.2 sparseA 和 sparseA2B 数据预取(g->s)
        local_idx = 0;
        if(tid < mat_len) {  
            TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
            vint         start_dataIdx = d_data2Idx[tc_block];
            if(present_local & (1ULL << tid))
                local_idx = __popcll(present_local << (63 - tid));

            if(local_idx == 0) {
                d_sharedSparseA[sel_shm_next + tid] = 0.0;
            } else {
                // FIXME 原始的访存可能会导致 illegal memory access，尚不清楚原因（不确定是否是这里造成的，算子间统一修改）
                d_sharedSparseA[sel_shm_next + tid] = 1.0f;
                //async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
            }
        }
        if(tid < inst_k) {
            if(tc_block + 1 < end_blk_idx)
                // 在 CUDA 设备端，shared memory 指针并不像 CPU 指针那样自动进行按类型步长计算, siPtr + 1 只是简单的 uint32_t 数值加法，而不会自动按 sizeof(int) 调整
                //async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
                // 两个关键点：d_sharedSparseA2B 的写入位置和 d_sparseA2B 的读取位置(在 tc_block == start_blk_idx + 1 时，截止到这里 d_sharedSparseA2B 的两片空间已经都用完了，需要预取的是 start_blk_idx + 2 的块数据)
                async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
        }

        // 2. mma 计算
        // 2.1 本次计算 A 部分 TC 块加载（B 部分已经直接加载到寄存器了）
        fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

        // 2.2 tensor core 计算
        if(sel_shm_next) {
            tf32_m16n8k8(fragB00, fragA, fragC[0]);
            tf32_m16n8k8(fragB01, fragA, fragC[1]);
            //fp64_m16n8k8(fragB00, fragA, fragC[0]);
            //fp64_m16n8k8(fragB01, fragA, fragC[1]);
        } else {
            tf32_m16n8k8(fragB10, fragA, fragC[0]);
            tf32_m16n8k8(fragB11, fragA, fragC[1]);
            //fp64_m16n8k8(fragB10, fragA, fragC[0]);
            //fp64_m16n8k8(fragB11, fragA, fragC[1]);
        }

        wait_group();
		__syncthreads();
    }   // end for(tc_block)

    // ====================================== 最后一块计算(无需进行数据预取) ======================================
    if (end_blk_idx - start_blk_idx > 0) {
        vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
        fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

        // 两个 buffer，选择其一进行计算
        if(!smem_sel) {
            tf32_m16n8k8(fragB00, fragA, fragC[0]);
            tf32_m16n8k8(fragB01, fragA, fragC[1]);
            //fp64_m16n8k8(fragB00, fragA, fragC[0]);
            //fp64_m16n8k8(fragB01, fragA, fragC[1]);
        } else {
            tf32_m16n8k8(fragB10, fragA, fragC[0]);
            tf32_m16n8k8(fragB11, fragA, fragC[1]);
            //fp64_m16n8k8(fragB10, fragA, fragC[0]);
            //fp64_m16n8k8(fragB11, fragA, fragC[1]);
        }

        // ====================================== 将结果矩阵 C 从寄存器写回到全局内存 ======================================
        vint colC  =  0;
        vint rowC  =  0;
        vint outOff = (bid << 3) * feature_dim + (local_warpID << 5) + offY; // blockIdx.x * (8 * 128) + threadIdx.y * 32（目前 offY 为 0）

        #pragma unroll
        for(vint i = 0; i < 4; ++i) {
            rowC = (tID_in_group << 1) + (i & 0x1); // tID_in_group * 2: base; i % 2: offset

            if(i < 2) colC = groupID;
            else colC = groupID + 8;

            // 这里并没有判断对行的写入是否可能越界，因为在创建 matC 的空间时已经创建了冗余空间，列上只可能写不到，不会越界
            atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
            atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
            //store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
            //store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
        }
    }
}

// gcn first layer cold data(bcsc) SpMM
__global__
void tf32_computeX128TransposePipe2_BCSC(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2C,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    int32_t* id_map,
    float* cpu_float_features,
    int32_t* cache_search_map,
    float** gpu_float_feature,
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    int32_t gpu_node_capacity,
    const vint feature_dim
) {
    // ====================================== 定义所需的寄存器、共享内存、线程和块相关变量 ======================================
    using ARegisters = MAT_VAL_TYPE[2];     // 8 * 8
    using BRegisters = MAT_VAL_TYPE[4];     // 算 2 个 m16n8k8，共用一个 A 16 * 8
    using CRegisters = MAT_VAL_TYPE[2][4];  // 16 * 8
    
    // 当前 MMA 所需数据
    ARegisters fragA;
    BRegisters fragB00;
    BRegisters fragB01;
    CRegisters fragC = {0.0};

    vint bid                  =   blockIdx.x;
    vint offY                 =   (blockIdx.y << 7);
    const vint laneid         =   31 & threadIdx.x; // threadIdx.x % warpSize(32)，但是目前 blockDim.x == 32，实际上 threadIdx.x 并不会大于 32，因此 landid == threadIdx.x
    const vint warpSize       =   32;
    const vint tid            =   threadIdx.y * warpSize + laneid; // 全局线程 ID
    const vint local_warpID   =   threadIdx.y;

    // ====================================== 确定每个线程组在矩阵块中的角色 ======================================
    vint groupID         =   laneid >> 2;   // laneid / 4
    vint tID_in_group    =   3 & laneid;    // laneid % 4

    // sparseA 原始数据是 row-major 的，这里按照 row-major 的方式进行索引
    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;      // 0, 1, 2, 3
    vint colA1           =   tID_in_group + 4;  // 4, 5, 6, 7

    // 对 denseB 原始数据访问行短列长（因为行是由 sparse A TC 块中非零元所在列确定的)
    vint rowB01          = tID_in_group;
    vint rowB23          = tID_in_group + 4;
    vint colB02          =   groupID + (local_warpID << 5); // local_warpID << 5 <=> local_warpID * 32 <=> threadIdx.y * 32
    vint colB13          =   colB02 + 8;

    // denseC 中局部偏移
    vint rowC02 = (tID_in_group << 1);
    vint rowC13 = rowC02 + 1;
    vint colC01 = (local_warpID << 5) + groupID;    // block 内考虑各个 warp 的全局偏移列号
    vint colC23 = colC01 + 8;
    
    constexpr const int inst_k  = 8;
    constexpr const int inst_n  = 8;

    const vint mat_len = 64;
    const vint idx_len = 8;
    vint  local_idx    = 0;

    // ====================================== 初始化共享内存地址，读取块范围 ======================================
    // 均开设两倍空间，用于数据预取
    __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
    __shared__ vint         d_sharedSparseA2C[2 * idx_len];
    
    // 异步拷贝需要使用共享内存地址空间的指针，泛型地址无法自动识别为共享地址空间，使用 __cvta_generic_to_shared 进行显式转换
    vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
    vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2C);
    
    MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];     // 当前 thread 所处 block 对应的 TC 块的起始索引
    MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];   // 当前 thread 所处 block 对应的 TC 块的结束索引

    #ifdef debug_block_id
    if (bid == 1 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("start_blk_idx: %d, end_blk_idx: %d\n", start_blk_idx, end_blk_idx);
    }
    #endif

    // ====================================== denseB transpose mapping 数据加载(g->r)(block 涉及的所有 A tc block 对应的 B 数据都是相同的) ======================================
    // block 内线程在 denseB 中负责的第 1 个行编号
    vint dense_rowIdx01 = bid * inst_k + rowB01; // 若未使用 shared memory，则需要确定当前 block 的初始列号 start_blk_idx * inst_k

    // 不同与 BCSR 存在列补全，BCSC 在列上不存在补全，所以理论上不会出现 dense_rowIdx01 >= numNodes 的情况，但是加上条件结果也不会收到影响
    if(dense_rowIdx01 >= numCols) { // 处理 row_window 最后一个 TC 块补列的情况
        fragB00[0] = 0.0; fragB00[1] = 0.0; 
        fragB01[0] = 0.0; fragB01[1] = 0.0;
    } else {
        if (dense_rowIdx01 < 0) {
            printf("[Error] first layer bcsc dense_rowIdx01 == 0\n");
            return;
        }

        int32_t gidx = (cache_search_map[dense_rowIdx01]);   // 当前 thread 对应节点缓存所在 clique 中的编号

        if(gidx < 0) {/*cache miss*/
            // 列号映射
            dense_rowIdx01 = id_map[dense_rowIdx01];

            if (dense_rowIdx01 >= 0) {
                const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx01) * feature_dim)];
  
                fragB00[0] = load_fp32_from_global(d_MatB + colB02);
                fragB00[1] = load_fp32_from_global(d_MatB + colB13);
                fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
            }
        } else { /*cache hit, find global position*/
            int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
            int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
            const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置
  
            fragB00[0] = load_fp32_from_global(d_MatB + colB02);
            fragB00[1] = load_fp32_from_global(d_MatB + colB13);
            fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
            fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
        }
    }

    // block 内线程在 denseB 中负责的第 2 个行编号
    vint dense_rowIdx23 = bid * inst_k + rowB23;

    if(dense_rowIdx23 >= numCols) {
        fragB00[2] = 0.0; fragB00[3] = 0.0; 
        fragB01[2] = 0.0; fragB01[3] = 0.0;
    } else {
        if (dense_rowIdx23 < 0) {
            printf("[Error] first layer bcsc dense_rowIdx23 == 0\n");
            return;
        }

        int32_t gidx = (cache_search_map[dense_rowIdx23]);   // 当前 thread 对应节点缓存所在 clique 中的编号

		if(gidx < 0) {/*cache miss*/
            // 列号映射
            dense_rowIdx23 = id_map[dense_rowIdx23];

            if (dense_rowIdx23 >= 0) {
                const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx23) * feature_dim)];

                fragB00[2] = load_fp32_from_global(d_MatB + colB02);
                fragB00[3] = load_fp32_from_global(d_MatB + colB13);
                fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
            }
		}else { /*cache hit, find global position*/
		    int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
		    int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
            const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置

            fragB00[2] = load_fp32_from_global(d_MatB + colB02);
            fragB00[3] = load_fp32_from_global(d_MatB + colB13);
            fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
            fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
		}
    } 

    // ====================================== shared memory 数据加载 ======================================
    // 1.1 第一次 MMA 所需 A 矩阵数据(sparseA 和 sparseA2B)加载(g->s)
    // 一个 block 中的前 64 个 thread 预取稀疏矩阵 A（一个 row_window 对应一个 block，所以 block 逐一处理所属 row_window 中的 TC 块)，这里就是解压缩过程
    if(tid < mat_len) {  
        TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx]; // 当前 TC 块的 bitmap
        vint start_dataIdx         = d_data2Idx[start_blk_idx];   // 当前 TC 块中非零元起始偏移

        // 每个 thread 对应 8*8 TC 块的一个元素，判断当前位置是否为非零元
        if(present_local & (1ULL << tid))
            local_idx = __popcll(present_local << (63 - tid));  // 计算包含当前位置的之前总共的非零元的个数，用于 data 索引

        // prefetch 1 tc_block
        if(local_idx == 0) {
            d_sharedSparseA[tid] = 0.0;
        } else {
            // FIXME 不确定是否有问题，统一修改为直接赋值
            d_sharedSparseA[tid] = 1.0f;
            //d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
        }
    }

    // 1.2 读取 1 个 TC 块的 sparseA2C 数据
    if(tid < inst_k) {
        d_sharedSparseA2C[tid] = load_int_from_global(d_sparseA2C + start_blk_idx * inst_k + tid); // offset = start_blk_idx * 8 + tid，因为每个 block 在 sparseA2B 中都存在 8 个数据，所以用块数 * 每块列数即为当前块包含列的初始位置
    }
    __syncthreads();

    #ifdef debug_sharedSparseA_sharedSparseA2C_load
    if (bid == 1) {
        if (tid < mat_len) {
            printf("thread %d: d_sharedSparseA[%d] = %f\n", tid, tid, d_sharedSparseA[tid]);
        }
        if (tid < inst_k) {
            printf("thread %d: d_sharedSparseA2C[%d] = %u\n", tid, tid, d_sharedSparseA2C[tid]);
        }
    }
    #endif

    // ====================================== 遍历所有块，进行稀疏矩阵 A 和稠密矩阵 B 的乘法 ======================================
    for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
        // select which buffer to read，block 内所有 thread 计算出的结果都是一样的
        // 标识 d_sharedSparseA 的起始地址（双 buffer，一个存储区域逻辑上分为两部分，sel_shm 和 sel_shm_next 分别指向这两部分的起始偏移)
        vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 当前 TC 块对应的 sharedSparseA 的起始地址
        vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;      // 下一个 TC 块对应的 sharedSparseA 的起始地址
        // 标识 d_sharedSparseA2B 的起始地址
        vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;   // 当前 TC 块对应的 sharedSparseA2B 的起始地址 
        vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

        // 1.数据预取
        // 1.1 sparseA 和 sparseA2B 数据预取(g->s)
        local_idx = 0;
        if(tid < mat_len) {  
            TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
            vint         start_dataIdx = d_data2Idx[tc_block];
            if(present_local & (1ULL << tid))
                local_idx = __popcll(present_local << (63 - tid));
            if(local_idx == 0) {
                d_sharedSparseA[sel_shm_next + tid] = 0.0;
            } else {
                // FIXME 原始的访存可能会导致 illegal memory access，尚不清楚原因（不确定是否是这里造成的，算子间统一修改）
                d_sharedSparseA[sel_shm_next + tid] = 1.0f;
                //async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
            }
        }
        if(tid < inst_k) {
            async_copy_idx(siPtr + ((sel_idx_shm_next + tid) << 2), d_sparseA2C + (tc_block) * inst_k + tid);
        }

        // 2. mma 计算
        // 2.1 本次计算 A 部分 TC 块加载（B 部分已经直接加载到寄存器了）
        fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

        // 2.2 tensor core 计算
        tf32_m16n8k8(fragB00, fragA, fragC[0]);
        tf32_m16n8k8(fragB01, fragA, fragC[1]);
        //fp64_m16n8k8(fragB00, fragA, fragC[0]);
        //fp64_m16n8k8(fragB01, fragA, fragC[1]);

        // 因为不同 block 在 denseC 中可能会写入相同的位置，没有办法让一个 block 持续修改 denseC 中的一段区域
        // block 每次计算得到结果后，都先采用原子操作写入到 denseC 的 global memory 中，同时需要将 fragC 清零（因为结果已经进行了累加）
        // 在 row_window 计算中，一个 block 只要处理的不是自己所负责的最后一个 tc block，结果就可以一直放在 fragment 中进行累加，不需要写回 global memory
        // 目前在 col_window 下，block 的写入位置一直在变化，结果无法持续保存在 fragment 中，只能计算完一个 tc block 就原子加到 global memory 一次

        // mma.sync 指令执行完成后，warp 内所有线程的 fragC 都已经被正确更新，且不存在 warp 间共享 fragC，因此无需__syncthreads()显式 block 内线程同步
        // 每个线程在 matC 内负责的行编号：rowC02 和 rowC13，分别对应的 C 的行编号是 d_sharedSparseA2C[sel_idx_shm + rowC02] 和 d_sharedSparseA2C[sel_idx_shm + rowC13]
        #ifdef debug_matC0_write_addr
        if (bid == 0 && threadIdx.y == 0) {
            printf("thread %d: global_rowC0:%d, global_rowC0:%d, addr: {%d, %d, %d, %d}\n", threadIdx.x, global_rowC0, global_rowC1, outOff0 + colC01, outOff0 + colC23, outOff1 + colC01, outOff1 + colC23);
        } 
        #endif

        vint outRow0 = d_sharedSparseA2C[sel_idx_shm + rowC02]; // 待写回 denseC 行号 * feature_dim
        if (outRow0 < numRows) {
            vint outOff0 = outRow0 * feature_dim; // 待写回 denseC 行号 * feature_dim

            atomicAdd(d_MatC + outOff0 + colC01, fragC[0][0]);
            atomicAdd(d_MatC + outOff0 + colC23, fragC[0][2]);

            atomicAdd(d_MatC + outOff0 + colC01 + COL_WINDOW_R, fragC[1][0]);
            atomicAdd(d_MatC + outOff0 + colC23 + COL_WINDOW_R, fragC[1][2]);
        }

        #ifdef debug_matC0_result
        if (bid == 0 && threadIdx.y == 0) {
            printf("thread %d: fragC: {%f, %f, %f, %f}\n", threadIdx.x, fragC[0][0], fragC[0][1], fragC[0][2], fragC[0][3]);
        }
        #endif

        vint outRow1 = d_sharedSparseA2C[sel_idx_shm + rowC13]; // 待写回 denseC 行号 * feature_dim        
        if (outRow1 < numRows) { 
            vint outOff1 = outRow1 * feature_dim; // 待写回 denseC 行号 * feature_dim

            atomicAdd(d_MatC + outOff1 + colC01, fragC[0][1]);
            atomicAdd(d_MatC + outOff1 + colC23, fragC[0][3]);

            atomicAdd(d_MatC + outOff1 + colC01 + COL_WINDOW_R, fragC[1][1]);
            atomicAdd(d_MatC + outOff1 + colC23 + COL_WINDOW_R, fragC[1][3]);
        }

        fragC[0][0] = fragC[0][1] = fragC[0][2] = fragC[0][3] =   0.0; 
        fragC[1][0] = fragC[1][1] = fragC[1][2] = fragC[1][3] =   0.0; 

        wait_group();
		__syncthreads();
    }   // end for(tc_block)

    // ====================================== 最后一块计算(无需进行数据预取) ======================================
    vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
    fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
    fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

    #ifdef debug_last_block_fragA
    if (bid == 1) {
        printf("warp: %d, thread: %d, fragA: {%f, %f}\n", threadIdx.y, threadIdx.x, fragA[0], fragA[1]);
    }
    #endif

    #ifdef debug_last_block_fragB
    if (bid == 1) {
        printf("warp: %d, thread: %d, fragB0: {%f, %f, %f, %f}\n", threadIdx.y, threadIdx.x, fragB01[0], fragB01[1], fragB01[2], fragB01[3]);
    }
    #endif

    if (end_blk_idx - start_blk_idx > 0) {
        // 两个 buffer，选择其一进行计算
        tf32_m16n8k8(fragB00, fragA, fragC[0]);
        tf32_m16n8k8(fragB01, fragA, fragC[1]);
        //fp64_m16n8k8(fragB00, fragA, fragC[0]);
        //fp64_m16n8k8(fragB01, fragA, fragC[1]);

        //vint sel_idx_shm_next = ((end_blk_idx - 1 - start_blk_idx) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址
        vint sel_idx_shm = ((end_blk_idx - start_blk_idx + 1) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

        vint outRow0 = d_sharedSparseA2C[sel_idx_shm + rowC02]; // 待写回 denseC 行号 * feature_dim
        // 因为 A 在行上存在补全，所以可能待写回位置超出了 denseC 的范围，这里需要进行判断
        if (outRow0 < numRows) {
            vint outOff0 = outRow0 * feature_dim; // 待写回 denseC 行号 * feature_dim

            atomicAdd(d_MatC + outOff0 + colC01, fragC[0][0]);
            atomicAdd(d_MatC + outOff0 + colC23, fragC[0][2]);

            atomicAdd(d_MatC + outOff0 + colC01 + COL_WINDOW_R, fragC[1][0]);
            atomicAdd(d_MatC + outOff0 + colC23 + COL_WINDOW_R, fragC[1][2]);
        }

        vint outRow1 = d_sharedSparseA2C[sel_idx_shm + rowC13]; // 待写回 denseC 行号 * feature_dim        
        if (outRow1 < numRows) {
            vint outOff1 = outRow1 * feature_dim; // 待写回 denseC 行号 * feature_dim

            atomicAdd(d_MatC + outOff1 + colC01, fragC[0][1]);
            atomicAdd(d_MatC + outOff1 + colC23, fragC[0][3]);

            atomicAdd(d_MatC + outOff1 + colC01 + COL_WINDOW_R, fragC[1][1]);
            atomicAdd(d_MatC + outOff1 + colC23 + COL_WINDOW_R, fragC[1][3]);
        }
    }

    #ifdef debug_last_block_addr
    if (bid == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 8; i++) {
            printf("d_sharedSparseA2c[%d]: %u\n", i, d_sharedSparseA2C[sel_idx_shm + i]);
        }
    }
    if (bid == 0 && threadIdx.y == 0 && threadIdx.x < 8) {
        printf("thread %d: outRow0:{%u}, outRow1:{%u}\n", threadIdx.x, outRow0, outRow1);
    }
    #endif

    #ifdef debug_last_block_result
    if (bid == 1) {
        printf("warp: %d, thread: %d, fragC: {%f, %f, %f, %f}\n", threadIdx.y, threadIdx.x, fragC[0][0], fragC[0][1], fragC[0][2], fragC[0][3]);
    }
    #endif

    // ====================================== 将结果矩阵 C 从寄存器写回到全局内存 ======================================
    // 因每次计算完成后都已将结果累加回 global memory，此处无需写回
}

__global__
void tf32_computeX128TransposePipe2(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2X,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    int32_t* id_map,
    float* cpu_float_features,
    int32_t* cache_search_map,
    float** gpu_float_feature,
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    int32_t gpu_node_capacity,
    const vint feature_dim,
    int flag
) {

    switch (flag) {
        case 0: {
            const vint* d_sparseA2B = d_sparseA2X;
            // ====================================== 定义所需的寄存器、共享内存、线程和块相关变量 ======================================
            using ARegisters = MAT_VAL_TYPE[2];     // 8 * 8
            using BRegisters = MAT_VAL_TYPE[4];     // 算 2 个 m16n8k8，共用一个 A 16 * 8
            using CRegisters = MAT_VAL_TYPE[2][4];  // 16 * 8
    
            // 当前 MMA 所需数据
            ARegisters fragA;
            BRegisters fragB00;
            BRegisters fragB01;
            CRegisters fragC = {0.0};

            // 下一次 MMA 预取数据
            BRegisters fragB10;
            BRegisters fragB11;

            vint bid                  =   blockIdx.x;
            vint offY                 =   (blockIdx.y << 7);
            const vint laneid         =   31 & threadIdx.x; // threadIdx.x % warpSize(32)，但是目前 blockDim.x == 32，实际上 threadIdx.x 并不会大于 32，因此 landid == threadIdx.x
            const vint warpSize       =   32;
            const vint tid            =   threadIdx.y * warpSize + laneid; // 全局线程 ID
            const vint local_warpID   =   threadIdx.y;

            // ====================================== 确定每个线程组在矩阵块中的角色 ======================================
            vint groupID         =   laneid >> 2;   // laneid / 4
            vint tID_in_group    =   3 & laneid;    // laneid % 4

            // sparseA 原始数据是 row-major 的，这里按照 row-major 的方式进行索引
            vint rowA            =   groupID;
            vint colA0           =   tID_in_group;      // 0, 1, 2, 3
            vint colA1           =   tID_in_group + 4;  // 4, 5, 6, 7

            // 对 denseB 原始数据访问行短列长（因为行是由 sparse A TC 块中非零元所在列确定的)
            vint colB02          =   groupID + (local_warpID << 5); // local_warpID << 5 <=> local_warpID * 32 <=> threadIdx.y * 32
            vint colB13          =   groupID + (local_warpID << 5) + 8;
            // 这个访问 B 的行号和访问 A 的列号是相同的
            vint row01           =   tID_in_group;      // 0, 1, 2, 3
            vint row23           =   tID_in_group + 4;  // 4, 5, 6, 7
    
            constexpr const int inst_k  = 8;
            constexpr const int inst_n  = 8;

            const vint mat_len = 64;
            const vint idx_len = 8;
            vint  local_idx    = 0;

            // ====================================== 初始化共享内存地址，读取块范围 ======================================
            // 均开设两倍空间，用于数据预取
            __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
            __shared__ vint         d_sharedSparseA2B[2 * idx_len];
    
            // 当 CUDA 设备函数接收到一个 generic 指针 时，它可能是 global、shared、local 中的任意一种地址
            // 异步拷贝需要使用共享内存地址空间的指针，泛型地址无法自动识别为共享地址空间，使用 __cvta_generic_to_shared 进行显式转换
            vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
            vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
    
            MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];     // 当前 thread 所处 block 对应的 TC 块的起始索引
            MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid + 1];   // 当前 thread 所处 block 对应的 TC 块的结束索引

            // ====================================== 第一次 MMA 所需 A 矩阵数据(sparseA 和 sparseA2B)加载(g->s) ======================================
            // 一个 block 中的前 64 个 thread 预取稀疏矩阵 A（一个 row_window 对应一个 block，所以 block 逐一处理所属 row_window 中的 TC 块)，这里就是解压缩过程
            if(tid < mat_len) {  
                TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx]; // 当前 TC 块的 bitmap
                vint start_dataIdx         = d_data2Idx[start_blk_idx];   // 当前 TC 块中非零元起始偏移

                // 每个 thread 对应 8*8 TC 块的一个元素，判断当前位置是否为非零元
                if(present_local & (1ULL << tid))
                    local_idx = __popcll(present_local << (63 - tid));  // 计算包含当前位置的之前总共的非零元的个数，用于 data 索引

                // prefetch 1 tc_block
                if(local_idx == 0) {
                    d_sharedSparseA[tid] = 0.0;
                } else {
                    // FIXME 不确定是否有问题，统一修改为直接赋值
                    d_sharedSparseA[tid] = 1.0f;
                    //d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
                }
            }

            // 读取 2 个 TC 块的 sparseA2B 数据
            if(tid < inst_k) {
                d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid); // offset = start_blk_idx * 8 + tid，因为每个 block 在 sparseA2B 中都存在 8 个数据，所以用块数 * 每块列数即为当前块包含列的初始位置

                // 如果当前 row_window 包含的 TC 块数量 >= 2，那么预取下一个 TC 块的索引
                if(start_blk_idx + 1 < end_blk_idx) {
                    d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
                }

            }
            __syncthreads();

            // ====================================== 第一次 MMA 所需 denseB transpose mapping 数据加载(g->r) ======================================
            vint dense_rowIdx01 = d_sharedSparseA2B[row01];
            vint dense_rowIdx23 = d_sharedSparseA2B[row23];

            if(dense_rowIdx01 >= numCols) { // 处理 row_window 最后一个 TC 块补列的情况
                fragB00[0] = 0.0; fragB00[1] = 0.0; 
                fragB01[0] = 0.0; fragB01[1] = 0.0;
            } else {
                if (dense_rowIdx01 < 0) {
                    printf("[Error] first layer bcsr dense_rowIdx01 == 0\n");
                    return;
                }
                // 缓存情况判定
                int32_t gidx = (cache_search_map[dense_rowIdx01]);   // 当前 thread 对应节点缓存所在 clique 中的编号

                if(gidx < 0) {/*cache miss*/
                    // 列实际时映射为真实列号(只有在 cache miss 需要访问 cpu 数据时，才需要映射为实际的图上节点编号)
                    dense_rowIdx01 = id_map[dense_rowIdx01];

                    if (dense_rowIdx01 >= 0) {
                        const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx01) * feature_dim)];

                        fragB00[0] = load_fp32_from_global(d_MatB + colB02);
                        fragB00[1] = load_fp32_from_global(d_MatB + colB13);
                        fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                        fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                    }
		        }else { /*cache hit, find global position*/
		            int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
		            int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
                    const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置

                    fragB00[0] = load_fp32_from_global(d_MatB + colB02);
                    fragB00[1] = load_fp32_from_global(d_MatB + colB13);
                    fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                    fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
		        }
            }

            if(dense_rowIdx23 >= numCols) {
                fragB00[2] = 0.0; fragB00[3] = 0.0; 
                fragB01[2] = 0.0; fragB01[3] = 0.0;
            } else {
                if (dense_rowIdx23 < 0) {
                    printf("[Error] first layer bcsr dense_rowIdx23 == 0\n");
                    return;
                }

                int32_t gidx = (cache_search_map[dense_rowIdx23]);   // 当前 thread 对应节点缓存所在 clique 中的编号

		        if(gidx < 0) {/*cache miss*/
                    // 列实际时映射为真实列号
                    dense_rowIdx23 = id_map[dense_rowIdx23];

                    if (dense_rowIdx23 >= 0) {
                        const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx23) * feature_dim)];

                        fragB00[2] = load_fp32_from_global(d_MatB + colB02);
                        fragB00[3] = load_fp32_from_global(d_MatB + colB13);
                        fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                        fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                    }
		        }else { /*cache hit, find global position*/
		            int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
		            int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
                    const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置

                    fragB00[2] = load_fp32_from_global(d_MatB + colB02);
                    fragB00[3] = load_fp32_from_global(d_MatB + colB13);
                    fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                    fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
		        }
            } 

            __syncthreads();

            // ====================================== 遍历所有块，进行稀疏矩阵 A 和稠密矩阵 B 的乘法 ======================================
            for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
                // select which buffer to read，block 内所有 thread 计算出的结果都是一样的
                // 标识 d_sharedSparseA 的起始地址（双 buffer，一个存储区域逻辑上分为两部分，sel_shm 和 sel_shm_next 分别指向这两部分的起始偏移)
                vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 当前 TC 块对应的 sharedSparseA 的起始地址
                vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;      // 下一个 TC 块对应的 sharedSparseA 的起始地址
                // 标识 d_sharedSparseA2B 的起始地址
                vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;   // 当前 TC 块(当前是相对于 tc_block 而言的）对应的 sharedSparseA2B 的起始地址 
                vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

                // 1.数据预取
                // 1.1 下一次 MMA 所需 denseB transpose mapping 数据预取(g->r)
                vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
                vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];

                if(sel_shm_next) {
                    if(dense_rowIdx101 > numCols) {
                        fragB10[0] = 0.0; fragB10[1] = 0.0; 
                        fragB11[0] = 0.0; fragB11[1] = 0.0;
                    } else {
                        if (dense_rowIdx101 < 0) {
                            printf("[Error] first layer bcsr dense_rowIdx101 == 0\n");
                            return;
                        }

                        int32_t gidx = (cache_search_map[dense_rowIdx101]);   // 当前 thread 对应节点缓存所在 clique 中的编号

                        if(gidx < 0) {/*cache miss*/
                            // 列号映射
                            dense_rowIdx101 = id_map[dense_rowIdx101];

                            if (dense_rowIdx101 >= 0) {
                                const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx101) * feature_dim)];
        
                                fragB10[0] = load_fp32_from_global(d_MatB + colB02);
                                fragB10[1] = load_fp32_from_global(d_MatB + colB13);
                                fragB11[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                                fragB11[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                            }
                        }else { /*cache hit, find global position*/
                            int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
                            int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
                            // TODO 确定 &gpu_float_feature[didx][0] 是否会对访存造成影响（考虑 Legion 设计为二级指针数据的意图）
                            const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置
        
                            fragB10[0] = load_fp32_from_global(d_MatB + colB02);
                            fragB10[1] = load_fp32_from_global(d_MatB + colB13);
                            fragB11[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                            fragB11[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                        }
                    }
                    if(dense_rowIdx123 > numCols) {
                        fragB10[2] = 0.0; fragB10[3] = 0.0; 
                        fragB11[2] = 0.0; fragB11[3] = 0.0;
                    } else {
                        if (dense_rowIdx123 < 0) {
                            printf("[Error] first layer bcsr dense_rowIdx123 == 0\n");
                            return;
                        }

                        int32_t gidx = (cache_search_map[dense_rowIdx123]);   // 当前 thread 对应节点缓存所在 clique 中的编号

                        if(gidx < 0) {/*cache miss*/
                            // 列号映射
                            dense_rowIdx123 = id_map[dense_rowIdx123];

                            if (dense_rowIdx123 >= 0) {
                                const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx123) * feature_dim)];
        
                                fragB10[2] = load_fp32_from_global(d_MatB + colB02);
                                fragB10[3] = load_fp32_from_global(d_MatB + colB13);
                                fragB11[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                                fragB11[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                            }
                        }else { /*cache hit, find global position*/
                            int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
                            int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
                            // TODO 确定 &gpu_float_feature[didx][0] 是否会对访存造成影响（考虑 Legion 设计为二级指针数据的意图）
                            const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置
        
                            fragB10[2] = load_fp32_from_global(d_MatB + colB02);
                            fragB10[3] = load_fp32_from_global(d_MatB + colB13);
                            fragB11[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                            fragB11[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                        }
                    }
                } else {
                    if(dense_rowIdx101 > numCols) {
                        fragB00[0] = 0.0; fragB00[1] = 0.0; 
                        fragB01[0] = 0.0; fragB01[1] = 0.0;
                    } else {
                        if (dense_rowIdx101 < 0) {
                            printf("[Error] first layer bcsr dense_rowIdx101 == 0\n");
                            return;
                        }

                        int32_t gidx = (cache_search_map[dense_rowIdx101]);   // 当前 thread 对应节点缓存所在 clique 中的编号

                        if(gidx < 0) {/*cache miss*/
                            // 列号映射
                            dense_rowIdx101 = id_map[dense_rowIdx101];

                            if (dense_rowIdx101 >= 0) {
                                const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx101) * feature_dim)];
        
                                fragB00[0] = load_fp32_from_global(d_MatB + colB02);
                                fragB00[1] = load_fp32_from_global(d_MatB + colB13);
                                fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                                fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                            }
                        }else { /*cache hit, find global position*/
                            int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
                            int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号

                            const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置
        
                            fragB00[0] = load_fp32_from_global(d_MatB + colB02);
                            fragB00[1] = load_fp32_from_global(d_MatB + colB13);
                            fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                            fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                        }
                    }
                    if(dense_rowIdx123 > numCols) {
                        fragB00[2] = 0.0; fragB00[3] = 0.0; 
                        fragB01[2] = 0.0; fragB01[3] = 0.0;
                    } else {
                        if (dense_rowIdx123 < 0) {
                            printf("[Error] first layer bcsr dense_rowIdx123 == 0\n");
                            return;
                        }

                        int32_t gidx = (cache_search_map[dense_rowIdx123]);   // 当前 thread 对应节点缓存所在 clique 中的编号

                        if(gidx < 0) {/*cache miss*/
                            // 列号映射
                            dense_rowIdx123 = id_map[dense_rowIdx123];

                            if (dense_rowIdx123 >= 0) {
                                const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx123) * feature_dim)];
        
                                fragB00[2] = load_fp32_from_global(d_MatB + colB02);
                                fragB00[3] = load_fp32_from_global(d_MatB + colB13);
                                fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                                fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                            }
                        }else { /*cache hit, find global position*/
                            int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
                            int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号

                            const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置
        
                            fragB00[2] = load_fp32_from_global(d_MatB + colB02);
                            fragB00[3] = load_fp32_from_global(d_MatB + colB13);
                            fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                            fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                        }
                    }
                }   // end if(sel_shm_next)
    
                // 1.2 sparseA 和 sparseA2B 数据预取(g->s)
                local_idx = 0;
                if(tid < mat_len) {  
                    TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
                    vint         start_dataIdx = d_data2Idx[tc_block];
                    if(present_local & (1ULL << tid))
                        local_idx = __popcll(present_local << (63 - tid));

                    if(local_idx == 0) {
                        d_sharedSparseA[sel_shm_next + tid] = 0.0;
                    } else {
                        // FIXME 原始的访存可能会导致 illegal memory access，尚不清楚原因（不确定是否是这里造成的，算子间统一修改）
                        d_sharedSparseA[sel_shm_next + tid] = 1.0f;
                        //async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
                    }
                }
                if(tid < inst_k) {
                    if(tc_block + 1 < end_blk_idx)
                        // 在 CUDA 设备端，shared memory 指针并不像 CPU 指针那样自动进行按类型步长计算, siPtr + 1 只是简单的 uint32_t 数值加法，而不会自动按 sizeof(int) 调整
                        //async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
                        // 两个关键点：d_sharedSparseA2B 的写入位置和 d_sparseA2B 的读取位置(在 tc_block == start_blk_idx + 1 时，截止到这里 d_sharedSparseA2B 的两片空间已经都用完了，需要预取的是 start_blk_idx + 2 的块数据)
                        async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
                }

                // 2. mma 计算
                // 2.1 本次计算 A 部分 TC 块加载（B 部分已经直接加载到寄存器了）
                fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
                fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

                // 2.2 tensor core 计算
                if(sel_shm_next) {
                    tf32_m16n8k8(fragB00, fragA, fragC[0]);
                    tf32_m16n8k8(fragB01, fragA, fragC[1]);
                    //fp64_m16n8k8(fragB00, fragA, fragC[0]);
                    //fp64_m16n8k8(fragB01, fragA, fragC[1]);
                } else {
                    tf32_m16n8k8(fragB10, fragA, fragC[0]);
                    tf32_m16n8k8(fragB11, fragA, fragC[1]);
                    //fp64_m16n8k8(fragB10, fragA, fragC[0]);
                    //fp64_m16n8k8(fragB11, fragA, fragC[1]);
                }

                wait_group();
		        __syncthreads();
            }   // end for(tc_block)

            // ====================================== 最后一块计算(无需进行数据预取) ======================================
            if (end_blk_idx - start_blk_idx > 0) {
                vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
                fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
                fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

                // 两个 buffer，选择其一进行计算
                if(!smem_sel) {
                    tf32_m16n8k8(fragB00, fragA, fragC[0]);
                    tf32_m16n8k8(fragB01, fragA, fragC[1]);
                    //fp64_m16n8k8(fragB00, fragA, fragC[0]);
                    //fp64_m16n8k8(fragB01, fragA, fragC[1]);
                } else {
                    tf32_m16n8k8(fragB10, fragA, fragC[0]);
                    tf32_m16n8k8(fragB11, fragA, fragC[1]);
                    //fp64_m16n8k8(fragB10, fragA, fragC[0]);
                    //fp64_m16n8k8(fragB11, fragA, fragC[1]);
                }

                // ====================================== 将结果矩阵 C 从寄存器写回到全局内存 ======================================
                vint colC  =  0;
                vint rowC  =  0;
                vint outOff = (bid << 3) * feature_dim + (local_warpID << 5) + offY; // blockIdx.x * (8 * 128) + threadIdx.y * 32（目前 offY 为 0）

                #pragma unroll
                for(vint i = 0; i < 4; ++i) {
                    rowC = (tID_in_group << 1) + (i & 0x1); // tID_in_group * 2: base; i % 2: offset

                    if(i < 2) colC = groupID;
                    else colC = groupID + 8;

                    // 这里并没有判断对行的写入是否可能越界，因为在创建 matC 的空间时已经创建了冗余空间，列上只可能写不到，不会越界
                    atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
                    atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
                    //store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
                    //store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
                }
            }

            break;
        }
        case 1: {
            const vint* d_sparseA2C = d_sparseA2X;
            // ====================================== 定义所需的寄存器、共享内存、线程和块相关变量 ======================================
            using ARegisters = MAT_VAL_TYPE[2];     // 8 * 8
            using BRegisters = MAT_VAL_TYPE[4];     // 算 2 个 m16n8k8，共用一个 A 16 * 8
            using CRegisters = MAT_VAL_TYPE[2][4];  // 16 * 8
    
            // 当前 MMA 所需数据
            ARegisters fragA;
            BRegisters fragB00;
            BRegisters fragB01;
            CRegisters fragC = {0.0};

            vint bid                  =   blockIdx.x;
            vint offY                 =   (blockIdx.y << 7);
            const vint laneid         =   31 & threadIdx.x; // threadIdx.x % warpSize(32)，但是目前 blockDim.x == 32，实际上 threadIdx.x 并不会大于 32，因此 landid == threadIdx.x
            const vint warpSize       =   32;
            const vint tid            =   threadIdx.y * warpSize + laneid; // 全局线程 ID
            const vint local_warpID   =   threadIdx.y;

            // ====================================== 确定每个线程组在矩阵块中的角色 ======================================
            vint groupID         =   laneid >> 2;   // laneid / 4
            vint tID_in_group    =   3 & laneid;    // laneid % 4

            // sparseA 原始数据是 row-major 的，这里按照 row-major 的方式进行索引
            vint rowA            =   groupID;
            vint colA0           =   tID_in_group;      // 0, 1, 2, 3
            vint colA1           =   tID_in_group + 4;  // 4, 5, 6, 7

            // 对 denseB 原始数据访问行短列长（因为行是由 sparse A TC 块中非零元所在列确定的)
            vint rowB01          = tID_in_group;
            vint rowB23          = tID_in_group + 4;
            vint colB02          =   groupID + (local_warpID << 5); // local_warpID << 5 <=> local_warpID * 32 <=> threadIdx.y * 32
            vint colB13          =   colB02 + 8;

            // denseC 中局部偏移
            vint rowC02 = (tID_in_group << 1);
            vint rowC13 = rowC02 + 1;
            vint colC01 = (local_warpID << 5) + groupID;    // block 内考虑各个 warp 的全局偏移列号
            vint colC23 = colC01 + 8;
    
            constexpr const int inst_k  = 8;
            constexpr const int inst_n  = 8;

            const vint mat_len = 64;
            const vint idx_len = 8;
            vint  local_idx    = 0;

            // ====================================== 初始化共享内存地址，读取块范围 ======================================
            // 均开设两倍空间，用于数据预取
            __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
            __shared__ vint         d_sharedSparseA2C[2 * idx_len];
    
            // 异步拷贝需要使用共享内存地址空间的指针，泛型地址无法自动识别为共享地址空间，使用 __cvta_generic_to_shared 进行显式转换
            vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
            vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2C);
    
            MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];     // 当前 thread 所处 block 对应的 TC 块的起始索引
            MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];   // 当前 thread 所处 block 对应的 TC 块的结束索引

            #ifdef debug_block_id
            if (bid == 1 && threadIdx.x == 0 && threadIdx.y == 0) {
                printf("start_blk_idx: %d, end_blk_idx: %d\n", start_blk_idx, end_blk_idx);
            }
            #endif

            // ====================================== denseB transpose mapping 数据加载(g->r)(block 涉及的所有 A tc block 对应的 B 数据都是相同的) ======================================
            // block 内线程在 denseB 中负责的第 1 个行编号
            vint dense_rowIdx01 = bid * inst_k + rowB01; // 若未使用 shared memory，则需要确定当前 block 的初始列号 start_blk_idx * inst_k

            // 不同与 BCSR 存在列补全，BCSC 在列上不存在补全，所以理论上不会出现 dense_rowIdx01 >= numNodes 的情况，但是加上条件结果也不会收到影响
            if(dense_rowIdx01 >= numCols) { // 处理 row_window 最后一个 TC 块补列的情况
                fragB00[0] = 0.0; fragB00[1] = 0.0; 
                fragB01[0] = 0.0; fragB01[1] = 0.0;
            } else {
                if (dense_rowIdx01 < 0) {
                    printf("[Error] first layer bcsc dense_rowIdx01 == 0\n");
                    return;
                }

                int32_t gidx = (cache_search_map[dense_rowIdx01]);   // 当前 thread 对应节点缓存所在 clique 中的编号

                if(gidx < 0) {/*cache miss*/
                    // 列号映射
                    dense_rowIdx01 = id_map[dense_rowIdx01];

                    if (dense_rowIdx01 >= 0) {
                        const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx01) * feature_dim)];
  
                        fragB00[0] = load_fp32_from_global(d_MatB + colB02);
                        fragB00[1] = load_fp32_from_global(d_MatB + colB13);
                        fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                        fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                    }
                } else { /*cache hit, find global position*/
                    int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
                    int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
                    const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置
  
                    fragB00[0] = load_fp32_from_global(d_MatB + colB02);
                    fragB00[1] = load_fp32_from_global(d_MatB + colB13);
                    fragB01[0] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R);
                    fragB01[1] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                }
            }

            // block 内线程在 denseB 中负责的第 2 个行编号
            vint dense_rowIdx23 = bid * inst_k + rowB23;

            if(dense_rowIdx23 >= numCols) {
                fragB00[2] = 0.0; fragB00[3] = 0.0; 
                fragB01[2] = 0.0; fragB01[3] = 0.0;
            } else {
                if (dense_rowIdx23 < 0) {
                    printf("[Error] first layer bcsc dense_rowIdx23 == 0\n");
                    return;
                }

                int32_t gidx = (cache_search_map[dense_rowIdx23]);   // 当前 thread 对应节点缓存所在 clique 中的编号

		        if(gidx < 0) {/*cache miss*/
                    // 列号映射
                    dense_rowIdx23 = id_map[dense_rowIdx23];

                    if (dense_rowIdx23 >= 0) {
                        const MAT_VAL_TYPE* __restrict__ d_MatB = &cpu_float_features[int64_t(int64_t(dense_rowIdx23) * feature_dim)];

                        fragB00[2] = load_fp32_from_global(d_MatB + colB02);
                        fragB00[3] = load_fp32_from_global(d_MatB + colB13);
                        fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                        fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
                    }
		        }else { /*cache hit, find global position*/
		            int32_t didx = gidx / gpu_node_capacity;   // 当前 thread 对应节点缓存在哪个 gpu 上（clique 中局部编号）
		            int32_t fidx = gidx % gpu_node_capacity;   // 当前 thread 对应节点在所在 gpu 上的局部编号
                    const MAT_VAL_TYPE* __restrict__ d_MatB = &gpu_float_feature[didx][int64_t(int64_t(fidx) * feature_dim)];   // 一个 gpu feature cache 起始位置 + 当前节点在 feature cache 中的偏移位置

                    fragB00[2] = load_fp32_from_global(d_MatB + colB02);
                    fragB00[3] = load_fp32_from_global(d_MatB + colB13);
                    fragB01[2] = load_fp32_from_global(d_MatB + colB02 + COL_WINDOW_R); // +16 应该是因为是 8 * 16 的块，所以需要跳过 16 个数据
                    fragB01[3] = load_fp32_from_global(d_MatB + colB13 + COL_WINDOW_R);
		        }
            } 

            // ====================================== shared memory 数据加载 ======================================
            // 1.1 第一次 MMA 所需 A 矩阵数据(sparseA 和 sparseA2B)加载(g->s)
            // 一个 block 中的前 64 个 thread 预取稀疏矩阵 A（一个 row_window 对应一个 block，所以 block 逐一处理所属 row_window 中的 TC 块)，这里就是解压缩过程
            if(tid < mat_len) {  
                TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx]; // 当前 TC 块的 bitmap
                vint start_dataIdx         = d_data2Idx[start_blk_idx];   // 当前 TC 块中非零元起始偏移

                // 每个 thread 对应 8*8 TC 块的一个元素，判断当前位置是否为非零元
                if(present_local & (1ULL << tid))
                    local_idx = __popcll(present_local << (63 - tid));  // 计算包含当前位置的之前总共的非零元的个数，用于 data 索引

                // prefetch 1 tc_block
                if(local_idx == 0) {
                    d_sharedSparseA[tid] = 0.0;
                } else {
                    // FIXME 不确定是否有问题，统一修改为直接赋值
                    d_sharedSparseA[tid] = 1.0f;
                    //d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
                }
            }

            // 1.2 读取 1 个 TC 块的 sparseA2C 数据
            if(tid < inst_k) {
                d_sharedSparseA2C[tid] = load_int_from_global(d_sparseA2C + start_blk_idx * inst_k + tid); // offset = start_blk_idx * 8 + tid，因为每个 block 在 sparseA2B 中都存在 8 个数据，所以用块数 * 每块列数即为当前块包含列的初始位置
            }
            __syncthreads();

            #ifdef debug_sharedSparseA_sharedSparseA2C_load
            if (bid == 1) {
                if (tid < mat_len) {
                    printf("thread %d: d_sharedSparseA[%d] = %f\n", tid, tid, d_sharedSparseA[tid]);
                }
                if (tid < inst_k) {
                    printf("thread %d: d_sharedSparseA2C[%d] = %u\n", tid, tid, d_sharedSparseA2C[tid]);
                }
            }
            #endif

            // ====================================== 遍历所有块，进行稀疏矩阵 A 和稠密矩阵 B 的乘法 ======================================
            for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
                // select which buffer to read，block 内所有 thread 计算出的结果都是一样的
                // 标识 d_sharedSparseA 的起始地址（双 buffer，一个存储区域逻辑上分为两部分，sel_shm 和 sel_shm_next 分别指向这两部分的起始偏移)
                vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 当前 TC 块对应的 sharedSparseA 的起始地址
                vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;      // 下一个 TC 块对应的 sharedSparseA 的起始地址
                // 标识 d_sharedSparseA2B 的起始地址
                vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;   // 当前 TC 块对应的 sharedSparseA2B 的起始地址 
                vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

                // 1.数据预取
                // 1.1 sparseA 和 sparseA2B 数据预取(g->s)
                local_idx = 0;
                if(tid < mat_len) {  
                    TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
                    vint         start_dataIdx = d_data2Idx[tc_block];
                    if(present_local & (1ULL << tid))
                        local_idx = __popcll(present_local << (63 - tid));
                    if(local_idx == 0) {
                        d_sharedSparseA[sel_shm_next + tid] = 0.0;
                    } else {
                        // FIXME 原始的访存可能会导致 illegal memory access，尚不清楚原因（不确定是否是这里造成的，算子间统一修改）
                        d_sharedSparseA[sel_shm_next + tid] = 1.0f;
                        //async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
                    }
                }
                if(tid < inst_k) {
                    async_copy_idx(siPtr + ((sel_idx_shm_next + tid) << 2), d_sparseA2C + (tc_block) * inst_k + tid);
                }

                // 2. mma 计算
                // 2.1 本次计算 A 部分 TC 块加载（B 部分已经直接加载到寄存器了）
                fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
                fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

                // 2.2 tensor core 计算
                tf32_m16n8k8(fragB00, fragA, fragC[0]);
                tf32_m16n8k8(fragB01, fragA, fragC[1]);
                //fp64_m16n8k8(fragB00, fragA, fragC[0]);
                //fp64_m16n8k8(fragB01, fragA, fragC[1]);

                // 因为不同 block 在 denseC 中可能会写入相同的位置，没有办法让一个 block 持续修改 denseC 中的一段区域
                // block 每次计算得到结果后，都先采用原子操作写入到 denseC 的 global memory 中，同时需要将 fragC 清零（因为结果已经进行了累加）
                // 在 row_window 计算中，一个 block 只要处理的不是自己所负责的最后一个 tc block，结果就可以一直放在 fragment 中进行累加，不需要写回 global memory
                // 目前在 col_window 下，block 的写入位置一直在变化，结果无法持续保存在 fragment 中，只能计算完一个 tc block 就原子加到 global memory 一次

                // mma.sync 指令执行完成后，warp 内所有线程的 fragC 都已经被正确更新，且不存在 warp 间共享 fragC，因此无需__syncthreads()显式 block 内线程同步
                // 每个线程在 matC 内负责的行编号：rowC02 和 rowC13，分别对应的 C 的行编号是 d_sharedSparseA2C[sel_idx_shm + rowC02] 和 d_sharedSparseA2C[sel_idx_shm + rowC13]
                #ifdef debug_matC0_write_addr
                if (bid == 0 && threadIdx.y == 0) {
                    printf("thread %d: global_rowC0:%d, global_rowC0:%d, addr: {%d, %d, %d, %d}\n", threadIdx.x, global_rowC0, global_rowC1, outOff0 + colC01, outOff0 + colC23, outOff1 + colC01, outOff1 + colC23);
                } 
                #endif

                vint outRow0 = d_sharedSparseA2C[sel_idx_shm + rowC02]; // 待写回 denseC 行号 * feature_dim
                if (outRow0 < numRows) {
                    vint outOff0 = outRow0 * feature_dim; // 待写回 denseC 行号 * feature_dim

                    atomicAdd(d_MatC + outOff0 + colC01, fragC[0][0]);
                    atomicAdd(d_MatC + outOff0 + colC23, fragC[0][2]);

                    atomicAdd(d_MatC + outOff0 + colC01 + COL_WINDOW_R, fragC[1][0]);
                    atomicAdd(d_MatC + outOff0 + colC23 + COL_WINDOW_R, fragC[1][2]);
                }

                #ifdef debug_matC0_result
                if (bid == 0 && threadIdx.y == 0) {
                    printf("thread %d: fragC: {%f, %f, %f, %f}\n", threadIdx.x, fragC[0][0], fragC[0][1], fragC[0][2], fragC[0][3]);
                }
                #endif

                vint outRow1 = d_sharedSparseA2C[sel_idx_shm + rowC13]; // 待写回 denseC 行号 * feature_dim        
                if (outRow1 < numRows) { 
                    vint outOff1 = outRow1 * feature_dim; // 待写回 denseC 行号 * feature_dim

                    atomicAdd(d_MatC + outOff1 + colC01, fragC[0][1]);
                    atomicAdd(d_MatC + outOff1 + colC23, fragC[0][3]);

                    atomicAdd(d_MatC + outOff1 + colC01 + COL_WINDOW_R, fragC[1][1]);
                    atomicAdd(d_MatC + outOff1 + colC23 + COL_WINDOW_R, fragC[1][3]);
                }

                fragC[0][0] = fragC[0][1] = fragC[0][2] = fragC[0][3] =   0.0; 
                fragC[1][0] = fragC[1][1] = fragC[1][2] = fragC[1][3] =   0.0; 

                wait_group();
		        __syncthreads();
            }   // end for(tc_block)

            // ====================================== 最后一块计算(无需进行数据预取) ======================================
            vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
            fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
            fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

            #ifdef debug_last_block_fragA
            if (bid == 1) {
                printf("warp: %d, thread: %d, fragA: {%f, %f}\n", threadIdx.y, threadIdx.x, fragA[0], fragA[1]);
            }
            #endif

            #ifdef debug_last_block_fragB
            if (bid == 1) {
                printf("warp: %d, thread: %d, fragB0: {%f, %f, %f, %f}\n", threadIdx.y, threadIdx.x, fragB01[0], fragB01[1], fragB01[2], fragB01[3]);
            }
            #endif

            if (end_blk_idx - start_blk_idx > 0) {
                // 两个 buffer，选择其一进行计算
                tf32_m16n8k8(fragB00, fragA, fragC[0]);
                tf32_m16n8k8(fragB01, fragA, fragC[1]);
                //fp64_m16n8k8(fragB00, fragA, fragC[0]);
                //fp64_m16n8k8(fragB01, fragA, fragC[1]);

                //vint sel_idx_shm_next = ((end_blk_idx - 1 - start_blk_idx) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址
                vint sel_idx_shm = ((end_blk_idx - start_blk_idx + 1) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

                vint outRow0 = d_sharedSparseA2C[sel_idx_shm + rowC02]; // 待写回 denseC 行号 * feature_dim
                // 因为 A 在行上存在补全，所以可能待写回位置超出了 denseC 的范围，这里需要进行判断
                if (outRow0 < numRows) {
                    vint outOff0 = outRow0 * feature_dim; // 待写回 denseC 行号 * feature_dim

                    atomicAdd(d_MatC + outOff0 + colC01, fragC[0][0]);
                    atomicAdd(d_MatC + outOff0 + colC23, fragC[0][2]);

                    atomicAdd(d_MatC + outOff0 + colC01 + COL_WINDOW_R, fragC[1][0]);
                    atomicAdd(d_MatC + outOff0 + colC23 + COL_WINDOW_R, fragC[1][2]);
                }

                vint outRow1 = d_sharedSparseA2C[sel_idx_shm + rowC13]; // 待写回 denseC 行号 * feature_dim        
                if (outRow1 < numRows) {
                    vint outOff1 = outRow1 * feature_dim; // 待写回 denseC 行号 * feature_dim

                    atomicAdd(d_MatC + outOff1 + colC01, fragC[0][1]);
                    atomicAdd(d_MatC + outOff1 + colC23, fragC[0][3]);

                    atomicAdd(d_MatC + outOff1 + colC01 + COL_WINDOW_R, fragC[1][1]);
                    atomicAdd(d_MatC + outOff1 + colC23 + COL_WINDOW_R, fragC[1][3]);
                }
            }
            break;
        }
        default: 
            break;
    }
}

// ======================================= gnn second layer =======================================

// gcn second layer hot data(bcsr) SpMM
__global__
void tf32_computeX128TransposePipe2_BCSR(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2B,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim
) {
    // ====================================== 定义所需的寄存器、共享内存、线程和块相关变量 ======================================
    using ARegisters = MAT_VAL_TYPE[2];     // 8 * 8
    using BRegisters = MAT_VAL_TYPE[4];     // 算 2 个 m16n8k8，共用一个 A 16 * 8
    using CRegisters = MAT_VAL_TYPE[2][4];  // 16 * 8
    
    // 当前 MMA 所需数据
    ARegisters fragA;
    BRegisters fragB00;
    BRegisters fragB01;
    CRegisters fragC = {0.0};

    // 下一次 MMA 预取数据
    BRegisters fragB10;
    BRegisters fragB11;
    

    vint bid                  =   blockIdx.x;
    vint offY                 =   (blockIdx.y << 7);
    const vint laneid         =   31 & threadIdx.x; // threadIdx.x % warpSize(32)，但是目前 blockDim.x == 32，实际上 threadIdx.x 并不会大于 32，因此 landid == threadIdx.x
    const vint warpSize       =   32;
    const vint tid            =   threadIdx.y * warpSize + laneid; // 全局线程 ID
    const vint local_warpID   =   threadIdx.y;

    // ====================================== 确定每个线程组在矩阵块中的角色 ======================================
    vint groupID         =   laneid >> 2;   // laneid / 4
    vint tID_in_group    =   3 & laneid;    // laneid % 4

    // sparseA 原始数据是 row-major 的，这里按照 row-major 的方式进行索引
    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;      // 0, 1, 2, 3
    vint colA1           =   tID_in_group + 4;  // 4, 5, 6, 7

    // 对 denseB 原始数据访问行短列长（因为行是由 sparse A TC 块中非零元所在列确定的)
    vint colB02          =   groupID + (local_warpID << 5); // local_warpID << 5 <=> local_warpID * 32 <=> threadIdx.y * 32
    vint colB13          =   groupID + (local_warpID << 5) + 8;
    vint row01           =   tID_in_group;      // 0, 1, 2, 3
    vint row23           =   tID_in_group + 4;  // 4, 5, 6, 7
    
    constexpr const int inst_k  = 8;
    constexpr const int inst_n  = 8;

    const vint mat_len = 64;
    const vint idx_len = 8;
    vint  local_idx    = 0;

    // ====================================== 初始化共享内存地址，读取块范围 ======================================
    // 均开设两倍空间，用于数据预取
    __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
    __shared__ vint         d_sharedSparseA2B[2 * idx_len];
    // MAT_VAL_TYPE            d_denseB[inst_m * inst_n];
    
    // 异步拷贝需要使用共享内存地址空间的指针，泛型地址无法自动识别为共享地址空间，使用 __cvta_generic_to_shared 进行显式转换
    vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
    vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
    
    MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];     // 当前 thread 所处 block 对应的 TC 块的起始索引
    MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];   // 当前 thread 所处 block 对应的 TC 块的结束索引

    // ====================================== 第一次 MMA 所需 A 矩阵数据(sparseA 和 sparseA2B)加载(g->s) ======================================
    // 一个 block 中的前 64 个 thread 预取稀疏矩阵 A（一个 row_window 对应一个 block，所以 block 逐一处理所属 row_window 中的 TC 块)，这里就是解压缩过程
    if(tid < mat_len) {  
        TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx]; // 当前 TC 块的 bitmap
        vint start_dataIdx         = d_data2Idx[start_blk_idx];   // 当前 TC 块中非零元起始偏移

        // 每个 thread 对应 8*8 TC 块的一个元素，判断当前位置是否为非零元
        if(present_local & (1ULL << tid))
            local_idx = __popcll(present_local << (63 - tid));  // 计算包含当前位置的之前总共的非零元的个数，用于 data 索引

        // prefetch 1 tc_block
        if(local_idx == 0) {
            d_sharedSparseA[tid] = 0.0;
        } else {
            // FIXME 不确定是否有问题，统一修改为直接赋值
            d_sharedSparseA[tid] = 1.0f;
            //d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
        }
    }

    // 读取 2 个 TC 块的 sparseA2B 数据
    if(tid < inst_k) {
        d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid); // offset = start_blk_idx * 8 + tid，因为每个 block 在 sparseA2B 中都存在 8 个数据，所以用块数 * 每块列数即为当前块包含列的初始位置

        // 如果当前 row_window 包含的 TC 块数量 >= 2，那么预取下一个 TC 块的索引
        if(start_blk_idx + 1 < end_blk_idx) {
            d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
        }
    }
    __syncthreads();

    // ====================================== 第一次 MMA 所需 denseB transpose mapping 数据加载(g->r) ======================================
    vint dense_rowIdx01 = d_sharedSparseA2B[row01];
    vint dense_rowIdx23 = d_sharedSparseA2B[row23];

    if(dense_rowIdx01 >= numCols) { // 处理 row_window 最后一个 TC 块补列的情况
        fragB00[0] = 0.0; fragB00[1] = 0.0; 
        fragB01[0] = 0.0; fragB01[1] = 0.0;
    } else {
        // 计算当前 thread 负责的 denseB 中的两个数据的索引
        vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;

        fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
        fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
    }

    if(dense_rowIdx23 >= numCols) {
        fragB00[2] = 0.0; fragB00[3] = 0.0; 
        fragB01[2] = 0.0; fragB01[3] = 0.0;
    } else {
        vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
        fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
        fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
    } 

    __syncthreads();

    // ====================================== 遍历所有块，进行稀疏矩阵 A 和稠密矩阵 B 的乘法 ======================================
    for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
        // select which buffer to read，block 内所有 thread 计算出的结果都是一样的
        // 标识 d_sharedSparseA 的起始地址（双 buffer，一个存储区域逻辑上分为两部分，sel_shm 和 sel_shm_next 分别指向这两部分的起始偏移)
        vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 当前 TC 块对应的 sharedSparseA 的起始地址
        vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;      // 下一个 TC 块对应的 sharedSparseA 的起始地址
        // 标识 d_sharedSparseA2B 的起始地址
        vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;   // 当前 TC 块(当前是相对于 tc_block 而言的）对应的 sharedSparseA2B 的起始地址 
        vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

        // 1.数据预取
        // 1.1 下一次 MMA 所需 denseB transpose mapping 数据预取(g->r)
        vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
        vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];

        if(sel_shm_next) {
            if(dense_rowIdx101 > numCols) {
                fragB10[0] = 0.0; fragB10[1] = 0.0; 
                fragB11[0] = 0.0; fragB11[1] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                fragB10[0] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB10[1] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB11[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB11[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
            if(dense_rowIdx123 > numCols) {
                fragB10[2] = 0.0; fragB10[3] = 0.0; 
                fragB11[2] = 0.0; fragB11[3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                fragB10[2] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB10[3] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB11[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB11[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
        } else {
            if(dense_rowIdx101 > numCols) {
                fragB00[0] = 0.0; fragB00[1] = 0.0; 
                fragB01[0] = 0.0; fragB01[1] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
            if(dense_rowIdx123 > numCols) {
                fragB00[2] = 0.0; fragB00[3] = 0.0; 
                fragB01[2] = 0.0; fragB01[3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
        }   // end if(sel_shm_next)

        // 1.2 sparseA 和 sparseA2B 数据预取(g->s)
        local_idx = 0;
        if(tid < mat_len) {  
            TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
            vint         start_dataIdx = d_data2Idx[tc_block];

            if(present_local & (1ULL << tid))
                local_idx = __popcll(present_local << (63 - tid));

            if(local_idx == 0) {
                d_sharedSparseA[sel_shm_next + tid] = 0.0;
            } else {
                // FIXME 原始的访存可能会导致 illegal memory access，尚不清楚原因（不确定是否是这里造成的，算子间统一修改）
                d_sharedSparseA[sel_shm_next + tid] = 1.0f;
                //async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
            }
        }
        if(tid < inst_k) {
            if(tc_block + 1 < end_blk_idx)
                // 在 CUDA 设备端，shared memory 指针并不像 CPU 指针那样自动进行按类型步长计算, siPtr + 1 只是简单的 uint32_t 数值加法，而不会自动按 sizeof(int) 调整
                //async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
                // 两个关键点：d_sharedSparseA2B 的写入位置和 d_sparseA2B 的读取位置(在 tc_block == start_blk_idx + 1 时，截止到这里 d_sharedSparseA2B 的两片空间已经都用完了，需要预取的是 start_blk_idx + 2 的块数据)
                async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
        }

        // 2. mma 计算
        // 2.1 本次计算 A 部分 TC 块加载（B 部分已经直接加载到寄存器了）
        fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

        // 2.2 tensor core 计算
        if(sel_shm_next) {
            tf32_m16n8k8(fragB00, fragA, fragC[0]);
            tf32_m16n8k8(fragB01, fragA, fragC[1]);
            //fp64_m16n8k8(fragB00, fragA, fragC[0]);
            //fp64_m16n8k8(fragB01, fragA, fragC[1]);
        } else {
            tf32_m16n8k8(fragB10, fragA, fragC[0]);
            tf32_m16n8k8(fragB11, fragA, fragC[1]);
            //fp64_m16n8k8(fragB10, fragA, fragC[0]);
            //fp64_m16n8k8(fragB11, fragA, fragC[1]);
        }

        wait_group();
		__syncthreads();
    }   // end for(tc_block)


    // ====================================== 最后一块计算(无需进行数据预取) ======================================
    if (end_blk_idx - start_blk_idx > 0) {
        vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
        fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

        // 两个 buffer，选择其一进行计算
        if(!smem_sel) {
            tf32_m16n8k8(fragB00, fragA, fragC[0]);
            tf32_m16n8k8(fragB01, fragA, fragC[1]);
            //fp64_m16n8k8(fragB00, fragA, fragC[0]);
            //fp64_m16n8k8(fragB01, fragA, fragC[1]);
        } else {
            tf32_m16n8k8(fragB10, fragA, fragC[0]);
            tf32_m16n8k8(fragB11, fragA, fragC[1]);
            //fp64_m16n8k8(fragB10, fragA, fragC[0]);
            //fp64_m16n8k8(fragB11, fragA, fragC[1]);
        }

        // ====================================== 将结果矩阵 C 从寄存器写回到全局内存 ======================================
        vint colC  =  0;
        vint rowC  =  0;
        vint outOff = (bid << 3) * feature_dim + (local_warpID << 5) + offY; // blockIdx.x * (8 * 128) + blockIdx.y * 32（目前 offY 为 0）

        #pragma unroll
        for(vint i = 0; i < 4; ++i) {
            rowC = (tID_in_group << 1) + (i & 0x1); // tID_in_group * 2: base; i % 2: offset

            if(i < 2) colC = groupID;
            else colC = groupID + 8;

            atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
            atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
            //store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
            //store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
        }
    }
}

// gcn second layer cold data(bcsc) SpMM
__global__
void tf32_computeX128TransposePipe2_BCSC(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2C,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim
) {
    //if (blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
    //    printf("gcn second layer bcsc SpMM\n");
    //}

    // ====================================== 定义所需的寄存器、共享内存、线程和块相关变量 ======================================
    using ARegisters = MAT_VAL_TYPE[2];     // 8 * 8
    using BRegisters = MAT_VAL_TYPE[4];     // 算 2 个 m16n8k8，共用一个 A 16 * 8
    using CRegisters = MAT_VAL_TYPE[2][4];  // 16 * 8
    
    // 当前 MMA 所需数据
    ARegisters fragA;
    BRegisters fragB00;
    BRegisters fragB01;
    CRegisters fragC = {0.0};

    vint bid                  =   blockIdx.x;
    vint offY                 =   (blockIdx.y << 7);
    const vint laneid         =   31 & threadIdx.x; // threadIdx.x % warpSize(32)，但是目前 blockDim.x == 32，实际上 threadIdx.x 并不会大于 32，因此 landid == threadIdx.x
    const vint warpSize       =   32;
    const vint tid            =   threadIdx.y * warpSize + laneid; // 全局线程 ID
    const vint local_warpID   =   threadIdx.y;

    // ====================================== 确定每个线程组在矩阵块中的角色 ======================================
    vint groupID         =   laneid >> 2;   // laneid / 4
    vint tID_in_group    =   3 & laneid;    // laneid % 4

    // sparseA 原始数据是 row-major 的，这里按照 row-major 的方式进行索引
    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;      // 0, 1, 2, 3
    vint colA1           =   tID_in_group + 4;  // 4, 5, 6, 7

    // 对 denseB 原始数据访问行短列长（因为行是由 sparse A TC 块中非零元所在列确定的)
    vint rowB01          = tID_in_group;
    vint rowB23          = tID_in_group + 4;
    vint colB02          =   groupID + (local_warpID << 5); // local_warpID << 5 <=> local_warpID * 32 <=> threadIdx.y * 32
    vint colB13          =   colB02 + 8;

    // denseC 中局部偏移
    vint rowC02 = (tID_in_group << 1);
    vint rowC13 = rowC02 + 1;
    vint colC01 = (local_warpID << 5) + groupID;    // block 内考虑各个 warp 的全局偏移列号
    vint colC23 = colC01 + 8;
    
    constexpr const int inst_k  = 8;
    constexpr const int inst_n  = 8;

    const vint mat_len = 64;
    const vint idx_len = 8;
    vint  local_idx    = 0;

    // ====================================== 初始化共享内存地址，读取块范围 ======================================
    // 均开设两倍空间，用于数据预取
    __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
    __shared__ vint         d_sharedSparseA2C[2 * idx_len];
    
    // 异步拷贝需要使用共享内存地址空间的指针，泛型地址无法自动识别为共享地址空间，使用 __cvta_generic_to_shared 进行显式转换
    vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
    vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2C);
    
    MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];     // 当前 thread 所处 block 对应的 TC 块的起始索引
    MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];   // 当前 thread 所处 block 对应的 TC 块的结束索引

    #ifdef debug_block_id
    if (bid == 1 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("start_blk_idx: %d, end_blk_idx: %d\n", start_blk_idx, end_blk_idx);
    }
    #endif

    // ====================================== denseB transpose mapping 数据加载(g->r)(block 涉及的所有 A tc block 对应的 B 数据都是相同的) ======================================
    // block 内线程在 denseB 中负责的第 1 个行编号
    vint dense_rowIdx01 = bid * inst_k + rowB01; // 若未使用 shared memory，则需要确定当前 block 的初始列号 start_blk_idx * inst_k

    // 不同与 BCSR 存在列补全，BCSC 在列上不存在补全，所以理论上不会出现 dense_rowIdx01 >= numNodes 的情况，但是加上条件结果也不会收到影响
    if(dense_rowIdx01 >= numCols) { // 处理 row_window 最后一个 TC 块补列的情况
        fragB00[0] = 0.0; fragB00[1] = 0.0; 
        fragB01[0] = 0.0; fragB01[1] = 0.0;
    } else {
        // 所负责行在 denseB 中的偏移
        vint block_denseB_baseAddr = dense_rowIdx01 * feature_dim;

        // 所在行初始位置 + 列偏移计算数据位置(列偏移是考虑 block 内全部 warp 后的 block 内全局偏移)
        vint sourceIdx0 = block_denseB_baseAddr + colB02;
        vint sourceIdx1 = block_denseB_baseAddr + colB13;

        #ifdef debug_denseB_load
        if (bid == 0 && threadIdx.y == 1) {
            printf("thread %d: sourceIdx0 = %d, sourceIdx1 = %d\n", threadIdx.x, sourceIdx0, sourceIdx1);
        }
        #endif

        fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
        fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
    }

    // block 内线程在 denseB 中负责的第 2 个行编号
    vint dense_rowIdx23 = bid * inst_k + rowB23;

    if(dense_rowIdx23 >= numCols) {
        fragB00[2] = 0.0; fragB00[3] = 0.0; 
        fragB01[2] = 0.0; fragB01[3] = 0.0;
    } else {
        vint block_denseB_baseAddr = dense_rowIdx23 * feature_dim;

        vint sourceIdx2 = block_denseB_baseAddr + colB02;
        vint sourceIdx3 = block_denseB_baseAddr + colB13;

        fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx2);
        fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx3);
        fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx2 + COL_WINDOW_R);
        fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx3 + COL_WINDOW_R);
    } 

    // ====================================== shared memory 数据加载 ======================================
    // 1.1 第一次 MMA 所需 A 矩阵数据(sparseA 和 sparseA2B)加载(g->s)
    // 一个 block 中的前 64 个 thread 预取稀疏矩阵 A（一个 row_window 对应一个 block，所以 block 逐一处理所属 row_window 中的 TC 块)，这里就是解压缩过程
    if(tid < mat_len) {  
        TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx]; // 当前 TC 块的 bitmap
        vint start_dataIdx         = d_data2Idx[start_blk_idx];   // 当前 TC 块中非零元起始偏移

        // 每个 thread 对应 8*8 TC 块的一个元素，判断当前位置是否为非零元
        if(present_local & (1ULL << tid))
            local_idx = __popcll(present_local << (63 - tid));  // 计算包含当前位置的之前总共的非零元的个数，用于 data 索引

        // prefetch 1 tc_block
        if(local_idx == 0) {
            d_sharedSparseA[tid] = 0.0;
        } else {
            // FIXME 不确定是否有问题，统一修改为直接赋值
            d_sharedSparseA[tid] = 1.0f;
            //d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
        }
    }

    // 1.2 读取 1 个 TC 块的 sparseA2C 数据
    if(tid < inst_k) {
        d_sharedSparseA2C[tid] = load_int_from_global(d_sparseA2C + start_blk_idx * inst_k + tid); // offset = start_blk_idx * 8 + tid，因为每个 block 在 sparseA2B 中都存在 8 个数据，所以用块数 * 每块列数即为当前块包含列的初始位置
    }
    __syncthreads();

    #ifdef debug_sharedSparseA_sharedSparseA2C_load
    if (bid == 1) {
        if (tid < mat_len) {
            printf("thread %d: d_sharedSparseA[%d] = %f\n", tid, tid, d_sharedSparseA[tid]);
        }
        if (tid < inst_k) {
            printf("thread %d: d_sharedSparseA2C[%d] = %u\n", tid, tid, d_sharedSparseA2C[tid]);
        }
    }
    #endif

    //if (threadIdx.y == 0 && threadIdx.x == 0) {
    //    if (end_blk_idx - start_blk_idx > 1) {
    //        printf("rowWindow-%d, start_blk_idx-%d, end_bld_idx-%d, size-%d\n", blockIdx.x, start_blk_idx, end_blk_idx, end_blk_idx - start_blk_idx);
    //    }
    //}
    // ====================================== 遍历所有块，进行稀疏矩阵 A 和稠密矩阵 B 的乘法 ======================================
    for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
        // select which buffer to read，block 内所有 thread 计算出的结果都是一样的
        // 标识 d_sharedSparseA 的起始地址（双 buffer，一个存储区域逻辑上分为两部分，sel_shm 和 sel_shm_next 分别指向这两部分的起始偏移)
        vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 当前 TC 块对应的 sharedSparseA 的起始地址
        vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;      // 下一个 TC 块对应的 sharedSparseA 的起始地址
        // 标识 d_sharedSparseA2B 的起始地址
        vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;   // 当前 TC 块对应的 sharedSparseA2B 的起始地址 
        vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

        // continue; // 加在这里运行正常

        // 1.数据预取
        // 1.1 sparseA 和 sparseA2B 数据预取(g->s)
        local_idx = 0;
        if(tid < mat_len) {  
            TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
            vint         start_dataIdx = d_data2Idx[tc_block];

            if(present_local & (1ULL << tid))
                local_idx = __popcll(present_local << (63 - tid));

            if(local_idx == 0) {
                d_sharedSparseA[sel_shm_next + tid] = 0.0;
            } else {
                // FIXME 原始的访存可能会导致 illegal memory access，尚不清楚原因（不确定是否是这里造成的，但是注释之后可正常执行）
                d_sharedSparseA[sel_shm_next + tid] = 1.0f;
                //async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
            }
        }
        //continue;
        if(tid < inst_k) {
            async_copy_idx(siPtr + ((sel_idx_shm_next + tid) << 2), d_sparseA2C + (tc_block) * inst_k + tid);
        }

        // 2. mma 计算
        // 2.1 本次计算 A 部分 TC 块加载（B 部分已经直接加载到寄存器了）
        fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

        // 2.2 tensor core 计算
        tf32_m16n8k8(fragB00, fragA, fragC[0]);
        tf32_m16n8k8(fragB01, fragA, fragC[1]);
        //fp64_m16n8k8(fragB00, fragA, fragC[0]);
        //fp64_m16n8k8(fragB01, fragA, fragC[1]);

        // 因为不同 block 在 denseC 中可能会写入相同的位置，没有办法让一个 block 持续修改 denseC 中的一段区域
        // block 每次计算得到结果后，都先采用原子操作写入到 denseC 的 global memory 中，同时需要将 fragC 清零（因为结果已经进行了累加）
        // 在 row_window 计算中，一个 block 只要处理的不是自己所负责的最后一个 tc block，结果就可以一直放在 fragment 中进行累加，不需要写回 global memory
        // 目前在 col_window 下，block 的写入位置一直在变化，结果无法持续保存在 fragment 中，只能计算完一个 tc block 就原子加到 global memory 一次

        // mma.sync 指令执行完成后，warp 内所有线程的 fragC 都已经被正确更新，且不存在 warp 间共享 fragC，因此无需__syncthreads()显式 block 内线程同步
        // 每个线程在 matC 内负责的行编号：rowC02 和 rowC13，分别对应的 C 的行编号是 d_sharedSparseA2C[sel_idx_shm + rowC02] 和 d_sharedSparseA2C[sel_idx_shm + rowC13]
        vint outRow0 = d_sharedSparseA2C[sel_idx_shm + rowC02]; // 待写回 denseC 行号 * feature_dim
        if (outRow0 < numRows) {
            vint outOff0 = outRow0 * feature_dim; // 待写回 denseC 行号 * feature_dim

            atomicAdd(d_MatC + outOff0 + colC01, fragC[0][0]);
            atomicAdd(d_MatC + outOff0 + colC23, fragC[0][2]);
            
            atomicAdd(d_MatC + outOff0 + colC01 + COL_WINDOW_R, fragC[1][0]);
            atomicAdd(d_MatC + outOff0 + colC23 + COL_WINDOW_R, fragC[1][2]);
        }

        vint outRow1 = d_sharedSparseA2C[sel_idx_shm + rowC13]; // 待写回 denseC 行号 * feature_dim        
        if (outRow1 < numRows) {
            vint outOff1 = outRow1 * feature_dim; // 待写回 denseC 行号 * feature_dim

            atomicAdd(d_MatC + outOff1 + colC01, fragC[0][1]);
            atomicAdd(d_MatC + outOff1 + colC23, fragC[0][3]);

            atomicAdd(d_MatC + outOff1 + colC01 + COL_WINDOW_R, fragC[1][1]);
            atomicAdd(d_MatC + outOff1 + colC23 + COL_WINDOW_R, fragC[1][3]);
        }

        fragC[0][0] = fragC[0][1] = fragC[0][2] = fragC[0][3] =   0.0; 
        fragC[1][0] = fragC[1][1] = fragC[1][2] = fragC[1][3] =   0.0; 

        wait_group();
		__syncthreads();
    }   // end for(tc_block)

    //return;
    // ====================================== 最后一块计算(无需进行数据预取) ======================================
    vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
    fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
    fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

    #ifdef debug_last_block_fragA
    if (bid == 1) {
        printf("warp: %d, thread: %d, fragA: {%f, %f}\n", threadIdx.y, threadIdx.x, fragA[0], fragA[1]);
    }
    #endif

    #ifdef debug_last_block_fragB
    if (bid == 1) {
        printf("warp: %d, thread: %d, fragB0: {%f, %f, %f, %f}\n", threadIdx.y, threadIdx.x, fragB01[0], fragB01[1], fragB01[2], fragB01[3]);
    }
    #endif

    if (end_blk_idx - start_blk_idx > 0) {
        // 两个 buffer，选择其一进行计算
        tf32_m16n8k8(fragB00, fragA, fragC[0]);
        tf32_m16n8k8(fragB01, fragA, fragC[1]);
        //fp64_m16n8k8(fragB00, fragA, fragC[0]);
        //fp64_m16n8k8(fragB01, fragA, fragC[1]);

        //vint sel_idx_shm_next = ((end_blk_idx - 1 - start_blk_idx) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址
        vint sel_idx_shm = ((end_blk_idx - start_blk_idx + 1) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

        vint outRow0 = d_sharedSparseA2C[sel_idx_shm + rowC02]; // 待写回 denseC 行号 * feature_dim
        // 因为 A 在行上存在补全，所以可能待写回位置超出了 denseC 的范围，这里需要进行判断
        if (outRow0 < numRows) {
            vint outOff0 = outRow0 * feature_dim; // 待写回 denseC 行号 * feature_dim

            atomicAdd(d_MatC + outOff0 + colC01, fragC[0][0]);
            atomicAdd(d_MatC + outOff0 + colC23, fragC[0][2]);

            atomicAdd(d_MatC + outOff0 + colC01 + COL_WINDOW_R, fragC[1][0]);
            atomicAdd(d_MatC + outOff0 + colC23 + COL_WINDOW_R, fragC[1][2]);
        }

        vint outRow1 = d_sharedSparseA2C[sel_idx_shm + rowC13]; // 待写回 denseC 行号 * feature_dim        
        if (outRow1 < numRows) {
            vint outOff1 = outRow1 * feature_dim; // 待写回 denseC 行号 * feature_dim

            atomicAdd(d_MatC + outOff1 + colC01, fragC[0][1]);
            atomicAdd(d_MatC + outOff1 + colC23, fragC[0][3]);

            atomicAdd(d_MatC + outOff1 + colC01 + COL_WINDOW_R, fragC[1][1]);
            atomicAdd(d_MatC + outOff1 + colC23 + COL_WINDOW_R, fragC[1][3]);
        }
    }

    #ifdef debug_last_block_addr
    if (bid == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 8; i++) {
            printf("d_sharedSparseA2c[%d]: %u\n", i, d_sharedSparseA2C[sel_idx_shm + i]);
        }
    }
    if (bid == 0 && threadIdx.y == 0 && threadIdx.x < 8) {
        printf("thread %d: outRow0:{%u}, outRow1:{%u}\n", threadIdx.x, outRow0, outRow1);
    }
    #endif

    #ifdef debug_last_block_result
    if (bid == 1) {
        printf("warp: %d, thread: %d, fragC: {%f, %f, %f, %f}\n", threadIdx.y, threadIdx.x, fragC[0][0], fragC[0][1], fragC[0][2], fragC[0][3]);
    }
    #endif

    // ====================================== 将结果矩阵 C 从寄存器写回到全局内存 ======================================
    // 因每次计算完成后都已将结果累加回 global memory，此处无需写回
}

__global__
void tf32_computeX128TransposePipe2(
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const vint*         __restrict__    d_sparseA2X,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numRows,
    const vint numCols,
    const vint feature_dim,
    int flag
) {
    switch (flag) {
        case 0: {
            const vint* d_sparseA2B = d_sparseA2X;
            // ====================================== 定义所需的寄存器、共享内存、线程和块相关变量 ======================================
            using ARegisters = MAT_VAL_TYPE[2];     // 8 * 8
            using BRegisters = MAT_VAL_TYPE[4];     // 算 2 个 m16n8k8，共用一个 A 16 * 8
            using CRegisters = MAT_VAL_TYPE[2][4];  // 16 * 8
    
            // 当前 MMA 所需数据
            ARegisters fragA;
            BRegisters fragB00;
            BRegisters fragB01;
            CRegisters fragC = {0.0};

            // 下一次 MMA 预取数据
            BRegisters fragB10;
            BRegisters fragB11;
    

            vint bid                  =   blockIdx.x;
            vint offY                 =   (blockIdx.y << 7);
            const vint laneid         =   31 & threadIdx.x; // threadIdx.x % warpSize(32)，但是目前 blockDim.x == 32，实际上 threadIdx.x 并不会大于 32，因此 landid == threadIdx.x
            const vint warpSize       =   32;
            const vint tid            =   threadIdx.y * warpSize + laneid; // 全局线程 ID
            const vint local_warpID   =   threadIdx.y;

            // ====================================== 确定每个线程组在矩阵块中的角色 ======================================
            vint groupID         =   laneid >> 2;   // laneid / 4
            vint tID_in_group    =   3 & laneid;    // laneid % 4

            // sparseA 原始数据是 row-major 的，这里按照 row-major 的方式进行索引
            vint rowA            =   groupID;
            vint colA0           =   tID_in_group;      // 0, 1, 2, 3
            vint colA1           =   tID_in_group + 4;  // 4, 5, 6, 7

            // 对 denseB 原始数据访问行短列长（因为行是由 sparse A TC 块中非零元所在列确定的)
            vint colB02          =   groupID + (local_warpID << 5); // local_warpID << 5 <=> local_warpID * 32 <=> threadIdx.y * 32
            vint colB13          =   groupID + (local_warpID << 5) + 8;
            vint row01           =   tID_in_group;      // 0, 1, 2, 3
            vint row23           =   tID_in_group + 4;  // 4, 5, 6, 7
    
            constexpr const int inst_k  = 8;
            constexpr const int inst_n  = 8;

            const vint mat_len = 64;
            const vint idx_len = 8;
            vint  local_idx    = 0;

            // ====================================== 初始化共享内存地址，读取块范围 ======================================
            // 均开设两倍空间，用于数据预取
            __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
            __shared__ vint         d_sharedSparseA2B[2 * idx_len];
            // MAT_VAL_TYPE            d_denseB[inst_m * inst_n];
    
            // 异步拷贝需要使用共享内存地址空间的指针，泛型地址无法自动识别为共享地址空间，使用 __cvta_generic_to_shared 进行显式转换
            vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
            vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
    
            MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];     // 当前 thread 所处 block 对应的 TC 块的起始索引
            MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];   // 当前 thread 所处 block 对应的 TC 块的结束索引

            // ====================================== 第一次 MMA 所需 A 矩阵数据(sparseA 和 sparseA2B)加载(g->s) ======================================
            // 一个 block 中的前 64 个 thread 预取稀疏矩阵 A（一个 row_window 对应一个 block，所以 block 逐一处理所属 row_window 中的 TC 块)，这里就是解压缩过程
            if(tid < mat_len) {  
                TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx]; // 当前 TC 块的 bitmap
                vint start_dataIdx         = d_data2Idx[start_blk_idx];   // 当前 TC 块中非零元起始偏移

                // 每个 thread 对应 8*8 TC 块的一个元素，判断当前位置是否为非零元
                if(present_local & (1ULL << tid))
                    local_idx = __popcll(present_local << (63 - tid));  // 计算包含当前位置的之前总共的非零元的个数，用于 data 索引

                // prefetch 1 tc_block
                if(local_idx == 0) {
                    d_sharedSparseA[tid] = 0.0;
                } else {
                    // FIXME 不确定是否有问题，统一修改为直接赋值
                    d_sharedSparseA[tid] = 1.0f;
                    //d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
                }
            }

            // 读取 2 个 TC 块的 sparseA2B 数据
            if(tid < inst_k) {
                d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid); // offset = start_blk_idx * 8 + tid，因为每个 block 在 sparseA2B 中都存在 8 个数据，所以用块数 * 每块列数即为当前块包含列的初始位置

                // 如果当前 row_window 包含的 TC 块数量 >= 2，那么预取下一个 TC 块的索引
                if(start_blk_idx + 1 < end_blk_idx) {
                    d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
                }
            }
            __syncthreads();

            // ====================================== 第一次 MMA 所需 denseB transpose mapping 数据加载(g->r) ======================================
            vint dense_rowIdx01 = d_sharedSparseA2B[row01];
            vint dense_rowIdx23 = d_sharedSparseA2B[row23];

            if(dense_rowIdx01 >= numCols) { // 处理 row_window 最后一个 TC 块补列的情况
                fragB00[0] = 0.0; fragB00[1] = 0.0; 
                fragB01[0] = 0.0; fragB01[1] = 0.0;
            } else {
                // 计算当前 thread 负责的 denseB 中的两个数据的索引
                vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;

                fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }

            if(dense_rowIdx23 >= numCols) {
                fragB00[2] = 0.0; fragB00[3] = 0.0; 
                fragB01[2] = 0.0; fragB01[3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
                fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            } 

            __syncthreads();

            // ====================================== 遍历所有块，进行稀疏矩阵 A 和稠密矩阵 B 的乘法 ======================================
            for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
                // select which buffer to read，block 内所有 thread 计算出的结果都是一样的
                // 标识 d_sharedSparseA 的起始地址（双 buffer，一个存储区域逻辑上分为两部分，sel_shm 和 sel_shm_next 分别指向这两部分的起始偏移)
                vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 当前 TC 块对应的 sharedSparseA 的起始地址
                vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;      // 下一个 TC 块对应的 sharedSparseA 的起始地址
                // 标识 d_sharedSparseA2B 的起始地址
                vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;   // 当前 TC 块(当前是相对于 tc_block 而言的）对应的 sharedSparseA2B 的起始地址 
                vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

                // 1.数据预取
                // 1.1 下一次 MMA 所需 denseB transpose mapping 数据预取(g->r)
                vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
                vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];

                if(sel_shm_next) {
                    if(dense_rowIdx101 > numCols) {
                        fragB10[0] = 0.0; fragB10[1] = 0.0; 
                        fragB11[0] = 0.0; fragB11[1] = 0.0;
                    } else {
                        vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                        vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                        fragB10[0] = load_fp32_from_global(d_MatB + sourceIdx0);
                        fragB10[1] = load_fp32_from_global(d_MatB + sourceIdx1);
                        fragB11[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                        fragB11[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
                    }
                    if(dense_rowIdx123 > numCols) {
                        fragB10[2] = 0.0; fragB10[3] = 0.0; 
                        fragB11[2] = 0.0; fragB11[3] = 0.0;
                    } else {
                        vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                        vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                        fragB10[2] = load_fp32_from_global(d_MatB + sourceIdx0);
                        fragB10[3] = load_fp32_from_global(d_MatB + sourceIdx1);
                        fragB11[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                        fragB11[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
                    }
                } else {
                    if(dense_rowIdx101 > numCols) {
                        fragB00[0] = 0.0; fragB00[1] = 0.0; 
                        fragB01[0] = 0.0; fragB01[1] = 0.0;
                    } else {
                        vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                        vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                        fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
                        fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
                        fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                        fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
                    }
                    if(dense_rowIdx123 > numCols) {
                        fragB00[2] = 0.0; fragB00[3] = 0.0; 
                        fragB01[2] = 0.0; fragB01[3] = 0.0;
                    } else {
                        vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                        vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                        fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
                        fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
                        fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                        fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
                    }
                }   // end if(sel_shm_next)

                // 1.2 sparseA 和 sparseA2B 数据预取(g->s)
                local_idx = 0;
                if(tid < mat_len) {  
                    TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
                    vint         start_dataIdx = d_data2Idx[tc_block];

                    if(present_local & (1ULL << tid))
                        local_idx = __popcll(present_local << (63 - tid));

                    if(local_idx == 0) {
                        d_sharedSparseA[sel_shm_next + tid] = 0.0;
                    } else {
                        // FIXME 原始的访存可能会导致 illegal memory access，尚不清楚原因（不确定是否是这里造成的，算子间统一修改）
                        d_sharedSparseA[sel_shm_next + tid] = 1.0f;
                        //async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
                    }
                }
                if(tid < inst_k) {
                    if(tc_block + 1 < end_blk_idx)
                        // 在 CUDA 设备端，shared memory 指针并不像 CPU 指针那样自动进行按类型步长计算, siPtr + 1 只是简单的 uint32_t 数值加法，而不会自动按 sizeof(int) 调整
                        //async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
                        // 两个关键点：d_sharedSparseA2B 的写入位置和 d_sparseA2B 的读取位置(在 tc_block == start_blk_idx + 1 时，截止到这里 d_sharedSparseA2B 的两片空间已经都用完了，需要预取的是 start_blk_idx + 2 的块数据)
                        async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
                }

                // 2. mma 计算
                // 2.1 本次计算 A 部分 TC 块加载（B 部分已经直接加载到寄存器了）
                fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
                fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

                // 2.2 tensor core 计算
                if(sel_shm_next) {
                    tf32_m16n8k8(fragB00, fragA, fragC[0]);
                    tf32_m16n8k8(fragB01, fragA, fragC[1]);
                    //fp64_m16n8k8(fragB00, fragA, fragC[0]);
                    //fp64_m16n8k8(fragB01, fragA, fragC[1]);
                } else {
                    tf32_m16n8k8(fragB10, fragA, fragC[0]);
                    tf32_m16n8k8(fragB11, fragA, fragC[1]);
                    //fp64_m16n8k8(fragB10, fragA, fragC[0]);
                    //fp64_m16n8k8(fragB11, fragA, fragC[1]);
                }

                wait_group();
		        __syncthreads();
            }   // end for(tc_block)


            // ====================================== 最后一块计算(无需进行数据预取) ======================================
            if (end_blk_idx - start_blk_idx > 0) {
                vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
                fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
                fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

                // 两个 buffer，选择其一进行计算
                if(!smem_sel) {
                    tf32_m16n8k8(fragB00, fragA, fragC[0]);
                    tf32_m16n8k8(fragB01, fragA, fragC[1]);
                    //fp64_m16n8k8(fragB00, fragA, fragC[0]);
                    //fp64_m16n8k8(fragB01, fragA, fragC[1]);
                } else {
                    tf32_m16n8k8(fragB10, fragA, fragC[0]);
                    tf32_m16n8k8(fragB11, fragA, fragC[1]);
                    //fp64_m16n8k8(fragB10, fragA, fragC[0]);
                    //fp64_m16n8k8(fragB11, fragA, fragC[1]);
                }

                // ====================================== 将结果矩阵 C 从寄存器写回到全局内存 ======================================
                vint colC  =  0;
                vint rowC  =  0;
                vint outOff = (bid << 3) * feature_dim + (local_warpID << 5) + offY; // blockIdx.x * (8 * 128) + blockIdx.y * 32（目前 offY 为 0）

                #pragma unroll
                for(vint i = 0; i < 4; ++i) {
                    rowC = (tID_in_group << 1) + (i & 0x1); // tID_in_group * 2: base; i % 2: offset

                    if(i < 2) colC = groupID;
                    else colC = groupID + 8;

                    atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
                    atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
                    //store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
                    //store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
                }
            }
            break;
        }

        case 1: {
            const vint* d_sparseA2C = d_sparseA2X;
            // ====================================== 定义所需的寄存器、共享内存、线程和块相关变量 ======================================
            using ARegisters = MAT_VAL_TYPE[2];     // 8 * 8
            using BRegisters = MAT_VAL_TYPE[4];     // 算 2 个 m16n8k8，共用一个 A 16 * 8
            using CRegisters = MAT_VAL_TYPE[2][4];  // 16 * 8
    
            // 当前 MMA 所需数据
            ARegisters fragA;
            BRegisters fragB00;
            BRegisters fragB01;
            CRegisters fragC = {0.0};

            vint bid                  =   blockIdx.x;
            vint offY                 =   (blockIdx.y << 7);
            const vint laneid         =   31 & threadIdx.x; // threadIdx.x % warpSize(32)，但是目前 blockDim.x == 32，实际上 threadIdx.x 并不会大于 32，因此 landid == threadIdx.x
            const vint warpSize       =   32;
            const vint tid            =   threadIdx.y * warpSize + laneid; // 全局线程 ID
            const vint local_warpID   =   threadIdx.y;

            // ====================================== 确定每个线程组在矩阵块中的角色 ======================================
            vint groupID         =   laneid >> 2;   // laneid / 4
            vint tID_in_group    =   3 & laneid;    // laneid % 4

            // sparseA 原始数据是 row-major 的，这里按照 row-major 的方式进行索引
            vint rowA            =   groupID;
            vint colA0           =   tID_in_group;      // 0, 1, 2, 3
            vint colA1           =   tID_in_group + 4;  // 4, 5, 6, 7

            // 对 denseB 原始数据访问行短列长（因为行是由 sparse A TC 块中非零元所在列确定的)
            vint rowB01          = tID_in_group;
            vint rowB23          = tID_in_group + 4;
            vint colB02          =   groupID + (local_warpID << 5); // local_warpID << 5 <=> local_warpID * 32 <=> threadIdx.y * 32
            vint colB13          =   colB02 + 8;

            // denseC 中局部偏移
            vint rowC02 = (tID_in_group << 1);
            vint rowC13 = rowC02 + 1;
            vint colC01 = (local_warpID << 5) + groupID;    // block 内考虑各个 warp 的全局偏移列号
            vint colC23 = colC01 + 8;
    
            constexpr const int inst_k  = 8;
            constexpr const int inst_n  = 8;

            const vint mat_len = 64;
            const vint idx_len = 8;
            vint  local_idx    = 0;

            // ====================================== 初始化共享内存地址，读取块范围 ======================================
            // 均开设两倍空间，用于数据预取
            __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
            __shared__ vint         d_sharedSparseA2C[2 * idx_len];
    
            // 异步拷贝需要使用共享内存地址空间的指针，泛型地址无法自动识别为共享地址空间，使用 __cvta_generic_to_shared 进行显式转换
            vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
            vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2C);
    
            MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];     // 当前 thread 所处 block 对应的 TC 块的起始索引
            MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];   // 当前 thread 所处 block 对应的 TC 块的结束索引

            #ifdef debug_block_id
            if (bid == 1 && threadIdx.x == 0 && threadIdx.y == 0) {
                printf("start_blk_idx: %d, end_blk_idx: %d\n", start_blk_idx, end_blk_idx);
            }
            #endif

            // ====================================== denseB transpose mapping 数据加载(g->r)(block 涉及的所有 A tc block 对应的 B 数据都是相同的) ======================================
            // block 内线程在 denseB 中负责的第 1 个行编号
            vint dense_rowIdx01 = bid * inst_k + rowB01; // 若未使用 shared memory，则需要确定当前 block 的初始列号 start_blk_idx * inst_k

            // 不同与 BCSR 存在列补全，BCSC 在列上不存在补全，所以理论上不会出现 dense_rowIdx01 >= numNodes 的情况，但是加上条件结果也不会收到影响
            if(dense_rowIdx01 >= numCols) { // 处理 row_window 最后一个 TC 块补列的情况
                fragB00[0] = 0.0; fragB00[1] = 0.0; 
                fragB01[0] = 0.0; fragB01[1] = 0.0;
            } else {
                // 所负责行在 denseB 中的偏移
                vint block_denseB_baseAddr = dense_rowIdx01 * feature_dim;

                // 所在行初始位置 + 列偏移计算数据位置(列偏移是考虑 block 内全部 warp 后的 block 内全局偏移)
                vint sourceIdx0 = block_denseB_baseAddr + colB02;
                vint sourceIdx1 = block_denseB_baseAddr + colB13;

                #ifdef debug_denseB_load
                if (bid == 0 && threadIdx.y == 1) {
                    printf("thread %d: sourceIdx0 = %d, sourceIdx1 = %d\n", threadIdx.x, sourceIdx0, sourceIdx1);
                }
                #endif

                fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }

            // block 内线程在 denseB 中负责的第 2 个行编号
            vint dense_rowIdx23 = bid * inst_k + rowB23;

            if(dense_rowIdx23 >= numCols) {
                fragB00[2] = 0.0; fragB00[3] = 0.0; 
                fragB01[2] = 0.0; fragB01[3] = 0.0;
            } else {
                vint block_denseB_baseAddr = dense_rowIdx23 * feature_dim;

                vint sourceIdx2 = block_denseB_baseAddr + colB02;
                vint sourceIdx3 = block_denseB_baseAddr + colB13;

                fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx2);
                fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx3);
                fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx2 + COL_WINDOW_R);
                fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx3 + COL_WINDOW_R);
            } 

            // ====================================== shared memory 数据加载 ======================================
            // 1.1 第一次 MMA 所需 A 矩阵数据(sparseA 和 sparseA2B)加载(g->s)
            // 一个 block 中的前 64 个 thread 预取稀疏矩阵 A（一个 row_window 对应一个 block，所以 block 逐一处理所属 row_window 中的 TC 块)，这里就是解压缩过程
            if(tid < mat_len) {  
                TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx]; // 当前 TC 块的 bitmap
                vint start_dataIdx         = d_data2Idx[start_blk_idx];   // 当前 TC 块中非零元起始偏移

                // 每个 thread 对应 8*8 TC 块的一个元素，判断当前位置是否为非零元
                if(present_local & (1ULL << tid))
                    local_idx = __popcll(present_local << (63 - tid));  // 计算包含当前位置的之前总共的非零元的个数，用于 data 索引

                // prefetch 1 tc_block
                if(local_idx == 0) {
                    d_sharedSparseA[tid] = 0.0;
                } else {
                    // FIXME 不确定是否有问题，统一修改为直接赋值
                    d_sharedSparseA[tid] = 1.0f;
                    //d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
                }
            }

            // 1.2 读取 1 个 TC 块的 sparseA2C 数据
            if(tid < inst_k) {
                d_sharedSparseA2C[tid] = load_int_from_global(d_sparseA2C + start_blk_idx * inst_k + tid); // offset = start_blk_idx * 8 + tid，因为每个 block 在 sparseA2B 中都存在 8 个数据，所以用块数 * 每块列数即为当前块包含列的初始位置
            }
            __syncthreads();

            #ifdef debug_sharedSparseA_sharedSparseA2C_load
            if (bid == 1) {
                if (tid < mat_len) {
                    printf("thread %d: d_sharedSparseA[%d] = %f\n", tid, tid, d_sharedSparseA[tid]);
                }
                if (tid < inst_k) {
                    printf("thread %d: d_sharedSparseA2C[%d] = %u\n", tid, tid, d_sharedSparseA2C[tid]);
                }
            }
            #endif

            //if (threadIdx.y == 0 && threadIdx.x == 0) {
            //    if (end_blk_idx - start_blk_idx > 1) {
            //        printf("rowWindow-%d, start_blk_idx-%d, end_bld_idx-%d, size-%d\n", blockIdx.x, start_blk_idx, end_blk_idx, end_blk_idx - start_blk_idx);
            //    }
            //}
            // ====================================== 遍历所有块，进行稀疏矩阵 A 和稠密矩阵 B 的乘法 ======================================
            for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
                // select which buffer to read，block 内所有 thread 计算出的结果都是一样的
                // 标识 d_sharedSparseA 的起始地址（双 buffer，一个存储区域逻辑上分为两部分，sel_shm 和 sel_shm_next 分别指向这两部分的起始偏移)
                vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 当前 TC 块对应的 sharedSparseA 的起始地址
                vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;      // 下一个 TC 块对应的 sharedSparseA 的起始地址
                // 标识 d_sharedSparseA2B 的起始地址
                vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;   // 当前 TC 块对应的 sharedSparseA2B 的起始地址 
                vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

                // continue; // 加在这里运行正常

                // 1.数据预取
                // 1.1 sparseA 和 sparseA2B 数据预取(g->s)
                local_idx = 0;
                if(tid < mat_len) {  
                    TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
                    vint         start_dataIdx = d_data2Idx[tc_block];

                    if(present_local & (1ULL << tid))
                        local_idx = __popcll(present_local << (63 - tid));

                    if(local_idx == 0) {
                        d_sharedSparseA[sel_shm_next + tid] = 0.0;
                    } else {
                        // FIXME 原始的访存可能会导致 illegal memory access，尚不清楚原因（不确定是否是这里造成的，但是注释之后可正常执行）
                        d_sharedSparseA[sel_shm_next + tid] = 1.0f;
                        //async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
                    }
                }
                //continue;
                if(tid < inst_k) {
                    async_copy_idx(siPtr + ((sel_idx_shm_next + tid) << 2), d_sparseA2C + (tc_block) * inst_k + tid);
                }

                // 2. mma 计算
                // 2.1 本次计算 A 部分 TC 块加载（B 部分已经直接加载到寄存器了）
                fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
                fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

                // 2.2 tensor core 计算
                tf32_m16n8k8(fragB00, fragA, fragC[0]);
                tf32_m16n8k8(fragB01, fragA, fragC[1]);
                //fp64_m16n8k8(fragB00, fragA, fragC[0]);
                //fp64_m16n8k8(fragB01, fragA, fragC[1]);

                // 因为不同 block 在 denseC 中可能会写入相同的位置，没有办法让一个 block 持续修改 denseC 中的一段区域
                // block 每次计算得到结果后，都先采用原子操作写入到 denseC 的 global memory 中，同时需要将 fragC 清零（因为结果已经进行了累加）
                // 在 row_window 计算中，一个 block 只要处理的不是自己所负责的最后一个 tc block，结果就可以一直放在 fragment 中进行累加，不需要写回 global memory
                // 目前在 col_window 下，block 的写入位置一直在变化，结果无法持续保存在 fragment 中，只能计算完一个 tc block 就原子加到 global memory 一次

                // mma.sync 指令执行完成后，warp 内所有线程的 fragC 都已经被正确更新，且不存在 warp 间共享 fragC，因此无需__syncthreads()显式 block 内线程同步
                // 每个线程在 matC 内负责的行编号：rowC02 和 rowC13，分别对应的 C 的行编号是 d_sharedSparseA2C[sel_idx_shm + rowC02] 和 d_sharedSparseA2C[sel_idx_shm + rowC13]
                vint outRow0 = d_sharedSparseA2C[sel_idx_shm + rowC02]; // 待写回 denseC 行号 * feature_dim
                if (outRow0 < numRows) {
                    vint outOff0 = outRow0 * feature_dim; // 待写回 denseC 行号 * feature_dim

                    atomicAdd(d_MatC + outOff0 + colC01, fragC[0][0]);
                    atomicAdd(d_MatC + outOff0 + colC23, fragC[0][2]);
            
                    atomicAdd(d_MatC + outOff0 + colC01 + COL_WINDOW_R, fragC[1][0]);
                    atomicAdd(d_MatC + outOff0 + colC23 + COL_WINDOW_R, fragC[1][2]);
                }

                vint outRow1 = d_sharedSparseA2C[sel_idx_shm + rowC13]; // 待写回 denseC 行号 * feature_dim        
                if (outRow1 < numRows) {
                    vint outOff1 = outRow1 * feature_dim; // 待写回 denseC 行号 * feature_dim

                    atomicAdd(d_MatC + outOff1 + colC01, fragC[0][1]);
                    atomicAdd(d_MatC + outOff1 + colC23, fragC[0][3]);

                    atomicAdd(d_MatC + outOff1 + colC01 + COL_WINDOW_R, fragC[1][1]);
                    atomicAdd(d_MatC + outOff1 + colC23 + COL_WINDOW_R, fragC[1][3]);
                }

                fragC[0][0] = fragC[0][1] = fragC[0][2] = fragC[0][3] =   0.0; 
                fragC[1][0] = fragC[1][1] = fragC[1][2] = fragC[1][3] =   0.0; 

                wait_group();
		        __syncthreads();
            }   // end for(tc_block)

            //return;
            // ====================================== 最后一块计算(无需进行数据预取) ======================================
            vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
            fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
            fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

            #ifdef debug_last_block_fragA
            if (bid == 1) {
                printf("warp: %d, thread: %d, fragA: {%f, %f}\n", threadIdx.y, threadIdx.x, fragA[0], fragA[1]);
            }
            #endif

            #ifdef debug_last_block_fragB
            if (bid == 1) {
                printf("warp: %d, thread: %d, fragB0: {%f, %f, %f, %f}\n", threadIdx.y, threadIdx.x, fragB01[0], fragB01[1], fragB01[2], fragB01[3]);
            }
            #endif

            if (end_blk_idx - start_blk_idx > 0) {
                // 两个 buffer，选择其一进行计算
                tf32_m16n8k8(fragB00, fragA, fragC[0]);
                tf32_m16n8k8(fragB01, fragA, fragC[1]);
                //fp64_m16n8k8(fragB00, fragA, fragC[0]);
                //fp64_m16n8k8(fragB01, fragA, fragC[1]);

                //vint sel_idx_shm_next = ((end_blk_idx - 1 - start_blk_idx) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址
                vint sel_idx_shm = ((end_blk_idx - start_blk_idx + 1) & 1) << 3;      // 下一个 TC 块对应的 sharedSparseA2B 的起始地址

                vint outRow0 = d_sharedSparseA2C[sel_idx_shm + rowC02]; // 待写回 denseC 行号 * feature_dim
                // 因为 A 在行上存在补全，所以可能待写回位置超出了 denseC 的范围，这里需要进行判断
                if (outRow0 < numRows) {
                    vint outOff0 = outRow0 * feature_dim; // 待写回 denseC 行号 * feature_dim

                    atomicAdd(d_MatC + outOff0 + colC01, fragC[0][0]);
                    atomicAdd(d_MatC + outOff0 + colC23, fragC[0][2]);

                    atomicAdd(d_MatC + outOff0 + colC01 + COL_WINDOW_R, fragC[1][0]);
                    atomicAdd(d_MatC + outOff0 + colC23 + COL_WINDOW_R, fragC[1][2]);
                }

                vint outRow1 = d_sharedSparseA2C[sel_idx_shm + rowC13]; // 待写回 denseC 行号 * feature_dim        
                if (outRow1 < numRows) {
                    vint outOff1 = outRow1 * feature_dim; // 待写回 denseC 行号 * feature_dim

                    atomicAdd(d_MatC + outOff1 + colC01, fragC[0][1]);
                    atomicAdd(d_MatC + outOff1 + colC23, fragC[0][3]);

                    atomicAdd(d_MatC + outOff1 + colC01 + COL_WINDOW_R, fragC[1][1]);
                    atomicAdd(d_MatC + outOff1 + colC23 + COL_WINDOW_R, fragC[1][3]);
                }
            }
            break;
        }

        default:
            break;
    }
}