#!/usr/bin/env python3
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import tempfile
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as Func

import ipc_service
import spmm_cnic

import time
import numpy as np
import torchmetrics
torch.set_printoptions(threshold=np.inf)

from collections import namedtuple

import ctypes
import csv

CooData = namedtuple('CooData', [
    'f_row_indices', 'f_col_indices',
    's_row_indices', 's_col_indices'
])

SparseMatrixSize = namedtuple('SparseMatrixSize', [
    'f_bcsr_num_rows', 'f_bcsr_num_cols', 'f_bcsr_nnz', 'f_bcsr_num_TCBlocks', 'f_bcsr_num_rowWindows',
    'f_bcsc_num_rows', 'f_bcsc_num_cols', 'f_bcsc_nnz', 'f_bcsc_num_TCBlocks', 'f_bcsc_num_colWindows',
    's_bcsr_num_rows', 's_bcsr_num_cols', 's_bcsr_nnz', 's_bcsr_num_TCBlocks', 's_bcsr_num_rowWindows',
    's_bcsc_num_rows', 's_bcsc_num_cols', 's_bcsc_nnz', 's_bcsc_num_TCBlocks', 's_bcsc_num_colWindows'
])

SparseMatrixData = namedtuple('SparseMatrixData', [
    'f_bcsr_rowWindowOffset', 'f_bcsr_tcOffset', 'f_bcsr_sparseA2B', 'f_bcsr_tcLocalBit', 'f_bcsr_data', 
    'f_bcsc_colWindowOffset', 'f_bcsc_tcOffset', 'f_bcsc_sparseA2C', 'f_bcsc_tcLocalBit', 'f_bcsc_data',
    's_bcsr_rowWindowOffset', 's_bcsr_tcOffset', 's_bcsr_sparseA2B', 's_bcsr_tcLocalBit', 's_bcsr_data', 
    's_bcsc_colWindowOffset', 's_bcsc_tcOffset', 's_bcsc_sparseA2C', 's_bcsc_tcLocalBit', 's_bcsc_data'
])

class CNICSPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        first_layer,
        sparse_matrix_size,
        sparse_matrix_data, 
        coo_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        x, weights, weight_vertical_axis_dim, device, device_id):

        if first_layer == True:
            # bcsr
            bcsr_num_rows        = sparse_matrix_size.s_bcsr_num_rows
            bcsr_num_cols        = sparse_matrix_size.s_bcsr_num_cols
            bcsr_nnz             = sparse_matrix_size.s_bcsr_nnz
            bcsr_num_TCBlocks    = sparse_matrix_size.s_bcsr_num_TCBlocks
            bcsr_num_rowWindows  = sparse_matrix_size.s_bcsr_num_rowWindows
            bcsr_rowWindowOffset = sparse_matrix_data.s_bcsr_rowWindowOffset
            bcsr_tcOffset        = sparse_matrix_data.s_bcsr_tcOffset
            bcsr_sparseA2B       = sparse_matrix_data.s_bcsr_sparseA2B
            bcsr_tcLocalBit      = sparse_matrix_data.s_bcsr_tcLocalBit
            bcsr_data            = sparse_matrix_data.s_bcsr_data
            # bcsc
            bcsc_num_rows        = sparse_matrix_size.s_bcsc_num_rows
            bcsc_num_cols        = sparse_matrix_size.s_bcsc_num_cols
            bcsc_nnz             = sparse_matrix_size.s_bcsc_nnz
            bcsc_num_TCBlocks    = sparse_matrix_size.s_bcsc_num_TCBlocks
            bcsc_num_colWindows  = sparse_matrix_size.s_bcsc_num_colWindows
            bcsc_colWindowOffset = sparse_matrix_data.s_bcsc_colWindowOffset
            bcsc_tcOffset        = sparse_matrix_data.s_bcsc_tcOffset
            bcsc_sparseA2C       = sparse_matrix_data.s_bcsc_sparseA2C
            bcsc_tcLocalBit      = sparse_matrix_data.s_bcsc_tcLocalBit
            bcsc_data            = sparse_matrix_data.s_bcsc_data
            #coo
            row_indices          = coo_data.s_row_indices
            col_indices          = coo_data.s_col_indices
        else:
            # bcsr
            bcsr_num_rows        = sparse_matrix_size.f_bcsr_num_rows
            bcsr_num_cols        = sparse_matrix_size.f_bcsr_num_cols
            bcsr_nnz             = sparse_matrix_size.f_bcsr_nnz
            bcsr_num_TCBlocks    = sparse_matrix_size.f_bcsr_num_TCBlocks
            bcsr_num_rowWindows  = sparse_matrix_size.f_bcsr_num_rowWindows
            bcsr_rowWindowOffset = sparse_matrix_data.f_bcsr_rowWindowOffset
            bcsr_tcOffset        = sparse_matrix_data.f_bcsr_tcOffset
            bcsr_sparseA2B       = sparse_matrix_data.f_bcsr_sparseA2B
            bcsr_tcLocalBit      = sparse_matrix_data.f_bcsr_tcLocalBit
            bcsr_data            = sparse_matrix_data.f_bcsr_data
            # bcsc
            bcsc_num_rows        = sparse_matrix_size.f_bcsc_num_rows
            bcsc_num_cols        = sparse_matrix_size.f_bcsc_num_cols
            bcsc_nnz             = sparse_matrix_size.f_bcsc_nnz
            bcsc_num_TCBlocks    = sparse_matrix_size.f_bcsc_num_TCBlocks
            bcsc_num_colWindows  = sparse_matrix_size.f_bcsc_num_colWindows
            bcsc_colWindowOffset = sparse_matrix_data.f_bcsc_colWindowOffset
            bcsc_tcOffset        = sparse_matrix_data.f_bcsc_tcOffset
            bcsc_sparseA2C       = sparse_matrix_data.f_bcsc_sparseA2C
            bcsc_tcLocalBit      = sparse_matrix_data.f_bcsc_tcLocalBit
            bcsc_data            = sparse_matrix_data.f_bcsc_data
            #coo
            row_indices          = coo_data.f_row_indices
            col_indices          = coo_data.f_col_indices
                            
        # 构建节点编号映射表（第一层采用实际节点编号，其他层采用顺序编号，用于 SpMM 过程中 MatB 的访存）
        if first_layer == True:
            X_out = spmm_cnic.forward_tensorcore_mixed(
                bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
                bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data,
                bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
                bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data,
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr)
        else:
            x = x.flatten().contiguous()
            X_out = spmm_cnic.forward_tensorcore(
                bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
                bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data,
                bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
                bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data,
                weight_vertical_axis_dim, x)

        # 1D -> 2D matrix
        X_out = X_out.reshape(bcsr_num_rows, weight_vertical_axis_dim)
        X_gat = torch.mm(X_out, weights)
        
        indices = torch.stack([row_indices, col_indices]) # (2, N) 的张量
        values = torch.ones(indices.shape[1]).to(device)  # 构造出来
        A_topo = torch.sparse_coo_tensor(indices, values, size=(bcsr_num_rows, bcsr_num_cols))
        # 把参数保存到 context 中，以便在反向传播时使用
        ctx.save_for_backward(X_out, A_topo, weights)

        # 结果输出至文件
        if device_id == 0: 
            logs_dir = "/work/gaohy/Legion_fuse/logs"
            latest_folder = sorted(
                [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))],
                reverse=True
            )[0] if os.listdir(logs_dir) else None

            if latest_folder:
                save_path = os.path.join(logs_dir, latest_folder)

                # AX result
                AX_file_name = "AX_1_layer.csv" if first_layer else "AX_2_layer.csv"
                AX_file_path = os.path.join(save_path, AX_file_name)

                np.savetxt(AX_file_path, X_out.cpu().numpy(), fmt="%8.2f", delimiter=",")  # 保存为 CSV
                print(f"spmm result 已保存至: {AX_file_path}")

                # AXW result
                AXW_file_name = "AXW_1_layer.csv" if first_layer else "AXW_2_layer.csv"
                AXW_file_path = os.path.join(save_path, AXW_file_name)

                np.savetxt(AXW_file_path, X_gat.cpu().numpy(), fmt="%8.2f", delimiter=",")  # 保存为 CSV
                print(f"spmm result 已保存至: {AXW_file_path}")

                # coo
                row_indices_file_name = "row_1_layer.csv" if first_layer else "row_2_layer.csv"
                col_indices_file_name = "col_1_layer.csv" if first_layer else "col_2_layer.csv"
                row_file_path = os.path.join(save_path, row_indices_file_name)
                col_file_path = os.path.join(save_path, col_indices_file_name)

                np.savetxt(row_file_path, row_indices.cpu().numpy(), fmt="%6d", delimiter=",")  # 保存为 CSV
                np.savetxt(col_file_path, col_indices.cpu().numpy(), fmt="%6d", delimiter=",")  # 保存为 CSV

                print(f"coo-row 已保存至: {row_file_path}")
                print(f"coo-col 已保存至: {col_file_path}")

                # combine coo
                if first_layer == True:
                    combine_coo_file_name = "coo.csv"
                    coo_file_path = os.path.join(save_path, combine_coo_file_name)

                    combined_data = np.column_stack((row_indices.cpu().numpy(), col_indices.cpu().numpy()))

                    np.savetxt(coo_file_path, combined_data, fmt="%d", delimiter=",")  # 保存为 CSV

                    print(f"coo 已保存至: {coo_file_path}")
                
                # weights
                weights_file_name = "weights_1_layer.csv" if first_layer else "weights_2_layer.csv"
                weights_file_path = os.path.join(save_path, weights_file_name)

                np.savetxt(weights_file_path, weights.cpu().numpy(), fmt="%6f", delimiter=",")  # 保存为 CSV

                print(f"coo-row 已保存至: {weights_file_path}")

            else:
                print("logs 目录为空，无法保存文件")

        return X_gat

    @staticmethod
    def backward(ctx, d_output):
        # 1. 获取 context 保存的中间变量
        X_out, A_topo, weights = ctx.saved_tensors # 返回的是一个 tuple，所以要用逗号解包

        # 2. 先反向传播过 dense matmul
        grad_weights = torch.mm(X_out.T, d_output) # 注意矩阵顺序

        grad_sp = torch.sparse.mm(A_topo.T, d_output)

        grad_X = grad_sp @ weights.T

        # 4. 返回梯度(返回值必须和 forward 的参数数量/位置一一对应, 非张量（标量、int、bool）或者不需要梯度的参数 → 返回 None)
        
        return (
            None,
            None, None, None,
            None, None, None,
            None, None, None, None,
            grad_X,
            grad_weights,  # weights 的梯度
            None, None, None
        )


class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, first_layer=False):
        super(GCNConv, self).__init__()
        self.first_layer = first_layer
        #print("input_dim: {}, output_dim: {}".format(input_dim, output_dim), flush=True)
        #self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.weights = torch.nn.Parameter(torch.empty(input_dim, output_dim))
        torch.nn.init.xavier_uniform_(self.weights)
        self.weight_vertical_axis_dim = input_dim

    def forward(
        self,
        sparse_matrix_size,
        sparse_matrix_data, 
        coo_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device, device_id, x = None):

        return CNICSPMMFunction.apply(
            self.first_layer,
            sparse_matrix_size,
            sparse_matrix_data, 
            coo_data,
            cpu_float_feature_len, gpu_node_capacity, feature_dim,
            ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
            x, self.weights, self.weight_vertical_axis_dim, device, device_id)

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, activation, dropout):
        super(GCN, self).__init__()
        self.num_classes = num_classes

        # 第一层
        # 调用具体 layer 的 init 方法
        self.conv1 = GCNConv(num_features, hidden_dim, first_layer=True)
        
        # 中间层
        """
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            #print("add hidden layer\n", flush=True)
            self.hidden_layers.append(GCNConv(hidden_dim, hidden_dim))
        """
        
        # 最后一层
        self.conv_final = GCNConv(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = activation

    def forward(
        self,
        sparse_matrix_size,
        sparse_matrix_data, 
        coo_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device,
        device_id
    ):
        # 第一层
        x = torch.empty(0)
        x = self.relu(
            self.conv1(
                sparse_matrix_size,
                sparse_matrix_data, 
                coo_data,
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr, device, device_id, x)
        )
        x = self.dropout(x)
        
        # 中间层
        """
        for conv in self.hidden_layers:
            #print("[Info] GCN middle layer forward begin", flush=True)
            x = conv(
                sparse_matrix_size,
                sparse_matrix_data, 
                coo_data,
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                x)
            x = self.relu(x)
            x = self.dropout(x)
        """
        
        # 最后一层
        x = self.conv_final(
                sparse_matrix_size,
                sparse_matrix_data, 
                coo_data,
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                device, device_id, x)

        # softmax 需要 2d matrix
        #x = x.reshape(sparse_matrix_size.f_bcsr_num_rows, self.num_classes)

        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    # initialize the process group
    if torch.cuda.is_available():
      dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
      dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_one_step(model, optimizer, metric, loss_fcn, device, feat_len, iter, device_id):
    # ======================================== 原始采样数据读取 ========================================
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()

    '''debug'''
    if device_id == 0:
        logs_dir = "/work/liangzq/Qugion"
        latest_folder = "logging"

        if latest_folder:
            save_path = os.path.join(logs_dir, latest_folder)

            # file_name = "Qugion_first_layer_feature.csv"
            # file_path = os.path.join(save_path, file_name)
            # np.savetxt(file_path, features.cpu().numpy(), delimiter=",", fmt="%.2f")  # 保存为 CSV
            # print(f"文件已保存至: {file_path}")

            # file_name = "blk1_agg_src.csv" 
            # file_path = os.path.join(save_path, file_name)
            # np.savetxt(file_path, block1_agg_src.cpu().numpy(), fmt="%d", delimiter=",")  # 保存为 CSV
            # print(f"block1_agg_src 已保存至: {file_path}")

            # file_name = "blk1_agg_dst.csv" 
            # file_path = os.path.join(save_path, file_name)
            # np.savetxt(file_path, block1_agg_dst.cpu().numpy(), fmt="%d", delimiter=",")  # 保存为 CSV
            # print(f"block1_agg_dst 已保存至: {file_path}")

            # file_name = "blk2_agg_src.csv" 
            # file_path = os.path.join(save_path, file_name)
            # np.savetxt(file_path, block2_agg_src.cpu().numpy(), fmt="%d", delimiter=",")  # 保存为 CSV
            # print(f"block2_agg_src 已保存至: {file_path}")

            # file_name = "blk2_agg_dst.csv" 
            # file_path = os.path.join(save_path, file_name)
            # np.savetxt(file_path, block2_agg_dst.cpu().numpy(), fmt="%d", delimiter=",")  # 保存为 CSV
            # print(f"block2_agg_dst 已保存至: {file_path}")

            block1_agg_src_np = block1_agg_src.cpu().numpy()
            block1_agg_dst_np = block1_agg_dst.cpu().numpy()
            combined_data_1 = np.column_stack((block1_agg_dst_np, block1_agg_src_np))
            unique_data_1 = np.unique(combined_data_1, axis=0) 
            # sorted_indices_1 = np.lexsort((unique_data_1[:, 1], unique_data_1[:, 0]))
            file_name1 = "fuse_block1_agg_combined_unique.csv"
            file_path1 = os.path.join(save_path, file_name1)
            np.savetxt(file_path1, unique_data_1, fmt="%d", delimiter=",")
            print(f"去重后的合并数据已保存至: {file_path1}")

            block2_agg_src_np = block2_agg_src.cpu().numpy()
            block2_agg_dst_np = block2_agg_dst.cpu().numpy()
            combined_data_2 = np.column_stack((block2_agg_dst_np, block2_agg_src_np))
            unique_data_2 = np.unique(combined_data_2, axis=0) 
            # sorted_indices = np.lexsort((unique_data_2[:, 1], unique_data_2[:, 0]))
            file_name2 = "fuse_block2_agg_combined_unique.csv"
            file_path2 = os.path.join(save_path, file_name2)
            np.savetxt(file_path2, unique_data_2, fmt="%d", delimiter=",")
            print(f"去重后的合并数据已保存至: {file_path2}")
            
        else:
            print("logs 目录为空，无法保存文件")
    
    #while(1):
    #    pass
    '''debug-end'''

    # 注意 Legion 为了适应 dgl 的需求而进行的边反向存储
    coo_data = CooData(
        f_row_indices = block2_agg_dst,   
        f_col_indices = block2_agg_src,   
        s_row_indices = block1_agg_dst,   
        s_col_indices = block1_agg_src
    ) 

    # ======================================== 混合数据结构读取 ========================================
    sparse_matrix_size = SparseMatrixSize(*tuple(ipc_service.get_sparseMatrix_size()))
    sparse_matrix_data = SparseMatrixData(*tuple(ipc_service.get_sparseMatrix_data()))

    # ======================================== feature 信息和数据读取 ========================================
    cache_search_map = ipc_service.get_memoryaccess_data()[0] # 这个函数目前只返回了一个 tensor，但是也放到了一个 vector 中，所以类型是 list 而非 tensor
    cpu_float_feature_len, gpu_node_capacity, feature_dim = ipc_service.get_feature_attribute()
    cpu_feature_cache_ptr, gpu_feature_cache_ptr = ipc_service.get_feature_cache_ptr()
    
    # ======================================== 计算 ========================================
    batch_pred = model(
        sparse_matrix_size,
        sparse_matrix_data,
        coo_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device,
        device_id,
    )

    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    loss = loss_fcn(batch_pred, long_labels)

    optimizer.zero_grad() # 清空梯度(.grad)，避免累积
    loss.backward() # 反向传播
    optimizer.step() # 权重更新(基于 optimizer 设置的优化器对参数进行更新)

    metric.update(batch_pred, long_labels)

    # 结果输出至文件
    if device_id == 0: 
        logs_dir = "/work/gaohy/Legion_fuse/logs"
        latest_folder = sorted(
            [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))],
            reverse=True
        )[0] if os.listdir(logs_dir) else None

        if latest_folder:
            save_path = os.path.join(logs_dir, latest_folder)

            # sparseMatrixSize
            file_name = "sparse_matrix_size.csv" 
            file_path = os.path.join(save_path, file_name) 
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)

                # 写入列名（字段名称）
                writer.writerow(sparse_matrix_size._fields)

                # 写入对应的值
                writer.writerow(sparse_matrix_size)

            print(f"sparseMatrixSize 已保存至: {file_path}")

            # ids
            file_name = "ids.csv" 
            file_path = os.path.join(save_path, file_name)
            np.savetxt(file_path, ids.cpu().numpy(), fmt="%d", delimiter=",")  # 保存为 CSV
            print(f"ids 已保存至: {file_path}")

            # cache_search_map
            file_name = "cache_search_map.csv" 
            file_path = os.path.join(save_path, file_name)
            np.savetxt(file_path, cache_search_map.cpu().numpy(), fmt="%d", delimiter=",")  # 保存为 CSV
            print(f"cache_search_map 已保存至: {file_path}")

            # feature（Legion 采样时处理）
            file_name = "feature_1_layer.csv" 
            file_path = os.path.join(save_path, file_name)
            np.savetxt(file_path, features.cpu().numpy(), fmt="%6.2f", delimiter=",")  # 保存为 CSV
            print(f"feature 已保存至: {file_path}")

            # feature（模拟算子计算时的处理方式获取到的数据）
            #simulate_feature = ipc_service.get_simulate_feature()

            #file_name = "feature_1_layer.csv" 
            #file_path = os.path.join(save_path, file_name)
            #np.savetxt(file_path, features.cpu().numpy(), fmt="%6.2f", delimiter=",")  # 保存为 CSV
            #print(f"feature 已保存至: {file_path}")


        else:
            print("logs 目录为空，无法保存文件")

    # 一个 batch 的计算完成后，需要将 AX 的结果 free，避免显存占用持续增大
    spmm_cnic.freeMatC()

    torch.cuda.synchronize()
    ipc_service.synchronize()

    return loss

def valid_one_step(model, metric, device, feat_len):
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    # 注意 Legion 为了适应 dgl 的需求而进行的边反向存储
    coo_data = CooData(
        f_row_indices = block2_agg_dst,   
        f_col_indices = block2_agg_src,   
        s_row_indices = block1_agg_dst,   
        s_col_indices = block1_agg_src
    ) 

    # ======================================== 混合数据结构读取 ========================================
    sparse_matrix_size = SparseMatrixSize(*tuple(ipc_service.get_sparseMatrix_size()))
    sparse_matrix_data = SparseMatrixData(*tuple(ipc_service.get_sparseMatrix_data()))

    # ======================================== feature 信息和数据读取 ========================================
    cache_search_map = ipc_service.get_memoryaccess_data()[0] # 这个函数目前只返回了一个 tensor，但是也放到了一个 vector 中，所以类型是 list 而非 tensor
    cpu_float_feature_len, gpu_node_capacity, feature_dim = ipc_service.get_feature_attribute()
    cpu_feature_cache_ptr, gpu_feature_cache_ptr = ipc_service.get_feature_cache_ptr()

    # ======================================== 计算 ========================================
    batch_pred = model(
        sparse_matrix_size,
        sparse_matrix_data,
        coo_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device
    )

    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    metric.update(batch_pred, long_labels)

    spmm_cnic.freeMatC()

    ipc_service.synchronize()

    return 0

def test_one_step(model, metric, device, feat_len):
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    # 注意 Legion 为了适应 dgl 的需求而进行的边反向存储
    coo_data = CooData(
        f_row_indices = block2_agg_dst,   
        f_col_indices = block2_agg_src,   
        s_row_indices = block1_agg_dst,   
        s_col_indices = block1_agg_src
    ) 

    # ======================================== 混合数据结构读取 ========================================
    sparse_matrix_size = SparseMatrixSize(*tuple(ipc_service.get_sparseMatrix_size()))
    sparse_matrix_data = SparseMatrixData(*tuple(ipc_service.get_sparseMatrix_data()))

    # ======================================== feature 信息和数据读取 ========================================
    cache_search_map = ipc_service.get_memoryaccess_data()[0] # 这个函数目前只返回了一个 tensor，但是也放到了一个 vector 中，所以类型是 list 而非 tensor
    cpu_float_feature_len, gpu_node_capacity, feature_dim = ipc_service.get_feature_attribute()
    cpu_feature_cache_ptr, gpu_feature_cache_ptr = ipc_service.get_feature_cache_ptr()

    # ======================================== 计算 ========================================
    batch_pred = model(
        sparse_matrix_size,
        sparse_matrix_data,
        coo_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device
    )

    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    metric.update(batch_pred, long_labels)

    spmm_cnic.freeMatC()

    ipc_service.synchronize()

    return 0

def worker_process(rank, world_size, args):
    print(f"Running GNN Training on CUDA {rank}.")
    device_id = rank
    setup(rank, world_size)
    cuda_device = torch.device("cuda:{}".format(device_id))
    torch.cuda.set_device(cuda_device)

    ipc_service.initialize(world_size)
    train_steps, valid_steps, test_steps = ipc_service.get_steps()
    print("[Info]: train_steps: {}, valid_steps: {}, test_steps: {}".format(train_steps, valid_steps, test_steps))

    feat_len = args.features_num

    # 调用 GCN class 的 init 方法
    model = GCN (
        num_features=args.features_num,
        hidden_dim=args.hidden_dim,
        num_classes=args.class_num,
        num_layers=args.hops_num,
        activation=Func.relu,
        dropout=args.drop_rate
    ).to(cuda_device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[device_id])
    
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epoch_num = args.epoch
    for epoch in range(epoch_num):
        # =========================== train ===========================
        print("device-{}, epoch-{} train".format(rank, epoch), flush=True)
        model.train()
        
        # 创建用于计算分类任务准确率的度量指标, 跟踪模型的准确率
        train_metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
        train_metric = train_metric.to(cuda_device)

        epoch_time = 0
        epoch_loss = 0.0

        train_steps = 1
        start = time.time()
        for iter in range(train_steps):
            train_loss = train_one_step(model, optimizer, train_metric, loss_fcn, cuda_device, feat_len, iter, device_id)
            epoch_loss += train_loss.item()
            if device_id == 0:
                print("epoch-{}, iter-{}, loss: {}".format(epoch, iter, train_loss), flush=True)
        epoch_time += time.time() - start

        avg_loss = epoch_loss / train_steps
        acc_train = train_metric.compute().item()
        if device_id == 0:
            print('>>Epoch {} Train Loss :{} Train Accuracy:{}'.format(epoch, avg_loss, acc_train))

        train_metric.reset()
    """    
        # =========================== valid ===========================
        print("device-{}, epoch-{} valid".format(rank, epoch), flush=True)

        model.eval()

        # 创建用于计算分类任务准确率的度量指标, 跟踪模型的准确率
        valid_metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
        valid_metric = valid_metric.to(cuda_device)
        
        with torch.no_grad():
            for iter in range(valid_steps):
                valid_one_step(model, valid_metric, cuda_device, feat_len)
            acc_val = valid_metric.compute().item()
        
        if device_id == 0:
            print("Epoch:{}, Cost:{} s, Val Acc: {}".format(epoch, epoch_time, acc_val))
        
        valid_metric.reset()

    # =========================== test ===========================
    print("device-{}, epoch-{} test".format(rank, epoch), flush=True)

    model.eval()

    # 创建用于计算分类任务准确率的度量指标, 跟踪模型的准确率
    test_metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
    test_metric = test_metric.to(cuda_device)

    with torch.no_grad():
        for iter in range(test_steps):
            test_one_step(model, test_metric, cuda_device, feat_len)
        acc_test = test_metric.compute().item()

    if device_id == 0:
        print("Accuracy on test data: {}".format(acc_test))

    test_metric.reset()
"""

    ipc_service.finalize()
    cleanup()

def run_distribute(dist_fn, world_size, args):
    mp.spawn(dist_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    cur_path = sys.path[0]
    argparser = argparse.ArgumentParser("Train GNN.")
    argparser.add_argument('--class_num', type=int, default=2)
    argparser.add_argument('--features_num', type=int, default=128)
    argparser.add_argument('--hidden_dim', type=int, default=256)
    argparser.add_argument('--hops_num', type=int, default=2)
    argparser.add_argument('--nbrs_num', type=list, default=[25, 10])
    argparser.add_argument('--drop_rate', type=float, default=0.5)
    argparser.add_argument('--learning_rate', type=float, default=0.003)
    argparser.add_argument('--epoch', type=int, default=2)
    argparser.add_argument('--gpu_number', type=int, default=2)
    args = argparser.parse_args()

    world_size = args.gpu_number

    run_distribute(worker_process, world_size, args)