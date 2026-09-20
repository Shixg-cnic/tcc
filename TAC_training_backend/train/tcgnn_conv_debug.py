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

DEBUG = True

from collections import namedtuple

import faulthandler
faulthandler.enable()

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


# 1.SPMMFunction（低层封装）: 将 PyTorch 张量转换成适合 CUDA 计算的格式，并在前向传播中执行 SPMM，在反向传播中计算梯度
# 第一层计算，feature 需要同时访问 cpu + gpu，因此 feature 访存时需要获取实际的节点编号，ids 需要作为函数参数
# 其他层计算，feature 全部位于 gpu，因此不需要获取实际的节点编号
class CNICSPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        first_layer,
        sparse_matrix_size,
        sparse_matrix_data, 
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        x, weights, weight_vertical_axis_dim, device_id):

        #print("function forward begin")

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
                            
        # 构建节点编号映射表（第一层采用实际节点编号，其他层采用顺序编号，用于 SpMM 过程中 MatB 的访存）
        if first_layer == True:
            if device_id == 0:
                print("[Info] 1' layer, numRows: {}, numCols: {}".format(bcsr_num_rows, bcsr_num_cols), flush=True)

            X_out = spmm_cnic.forward_tensorcore_mixed(
                bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
                bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data,
                bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
                bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data,
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr)
        else:
            if device_id == 0:
                print("[Info] 2' layer, numRows: {}, numCols: {}".format(bcsr_num_rows, bcsr_num_cols), flush=True)

            X_out = spmm_cnic.forward_tensorcore(
                bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
                bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data,
                bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
                bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data,
                weight_vertical_axis_dim, x)

        # 1D -> 2D matrix
        X_out = X_out.reshape(bcsr_num_rows, weight_vertical_axis_dim)

        #"""
        # 把参数保存到 context 中，以便在反向传播时使用
        ctx.save_for_backward(X_out)
        #"""
        
        X_out = torch.mm(X_out, weights)
        
        """
        print("\n[Info] spmm result: \n", flush=True)
        X_out_flat = X_out[0].cpu().flatten()
        for i in range(0, X_out_flat.size(0), 128):
            row = X_out_flat[i:i+128]
            print("row-{}".format(i / 128), row.tolist(), flush=True)
        """

        """
        print("\nX_out shape: ", X_out.shape, flush=True)
        """

        return X_out

    @staticmethod
    def backward(ctx, d_output):
        #print("backward begin")

        #print("ctx size: ", len(ctx.saved_tensors))
        # 1. 获取 context 保存的中间变量
        X_mid_2d, = ctx.saved_tensors # 返回的是一个 tuple，所以要用逗号解包

        #print("d_output shape:", d_output.shape)
        #print("X_mid_2d shape:", X_mid_2d.shape)

        # 2. 先反向传播过 dense matmul
        grad_weights = torch.mm(X_mid_2d.T, d_output) # 注意矩阵顺序

        # 4. 返回梯度(返回值必须和 forward 的参数数量/位置一一对应, 非张量（标量、int、bool）或者不需要梯度的参数 → 返回 None)
        return (
            None,
            None, None,
            None, None, None,
            None, None, None, None,
            None,
            grad_weights,  # weights 的梯度
            None, None
        )

# 2.GCNLayer（中间层）: 核心计算是 A @ X，即图的稀疏矩阵 A 与节点特征 X 的乘法,使用自定义 CUDA SPMM
class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, first_layer=False):
        super(GCNConv, self).__init__()
        self.first_layer = first_layer
        #print("input_dim: {}, output_dim: {}".format(input_dim, output_dim), flush=True)
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.weight_vertical_axis_dim = input_dim

    def forward(
        self,
        sparse_matrix_size,
        sparse_matrix_data, 
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id, x = None):

        #print("GCNConv forward begin(input_dim:{})".format(self.weight_vertical_axis_dim))

        return CNICSPMMFunction.apply(
            self.first_layer,
            sparse_matrix_size,
            sparse_matrix_data, 
            cpu_float_feature_len, gpu_node_capacity, feature_dim,
            ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
            x, self.weights, self.weight_vertical_axis_dim, device_id)


# 3.GCN（高层封装）: 表示完整的 GCN 网络，由多个 GCNLayer 组成，并包含 Dropout、ReLU 等额外操作
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, activation, dropout):
        super(GCN, self).__init__()
        self.num_classes = num_classes

        # 第一层
        # 调用具体 layer 的 init 方法
        self.conv1 = GCNConv(num_features, hidden_dim, first_layer=True)
        
        # 中间层
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            #print("add hidden layer\n", flush=True)
            self.hidden_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # 最后一层
        self.conv_final = GCNConv(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = activation

    def forward(
        self,
        sparse_matrix_size,
        sparse_matrix_data, 
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id
    ):
        #"""
        #print("[Info] GCN first layer forward begin", flush=True)
        # 第一层
        # 这里就是调用具体 layer 的 forward 方法了
        x = self.relu(
            self.conv1(
                sparse_matrix_size,
                sparse_matrix_data, 
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                device_id)
        )
        x = self.dropout(x)

        #print("[Info] first layer compute result shape: ", x.size(), flush=True)
        #print("[Info] first layer compute result: ", x, flush=True)
        
        # 返回的是 2d matrix，但是 C++ 接口需要 1d matrix
        x = x.flatten()
        # 中间层
        for conv in self.hidden_layers:
            #print("[Info] GCN middle layer forward begin", flush=True)
            x = conv(
                sparse_matrix_size,
                sparse_matrix_data, 
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                x)
            x = self.relu(x)
            x = self.dropout(x)
        
        # 返回的是 2d matrix，但是 C++ 接口需要 1d matrix
        x = x.flatten()
        # 最后一层
        #print("[Info] GCN last layer forward begin", flush=True)
        x = self.conv_final(
                sparse_matrix_size,
                sparse_matrix_data, 
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                device_id, x)

        #print("[Info] second layer compute result: ", x.shape, flush=True)
        
        # softmax 需要 2d matrix
        x = x.reshape(sparse_matrix_size.f_bcsr_num_rows, self.num_classes)
        #"""

        # 此处无需手动进行 softmax，因为 CrossEntropyLoss 会自动完成
        #return F.log_softmax(x, dim=1)
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    # initialize the process group
    if torch.cuda.is_available():
      dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
      dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_one_step(model, optimizer, loss_fcn, device, feat_len, iter, device_id):
    # ======================================== 原始采样数据读取 ========================================
    # features 和 labels 所属节点是同 ids 中的节点的顺序相对应的
    # block1 和 block2 分别对应 1 阶和 2 阶邻居的目标节点和源节点(不过此处获取到的 src-dst 实际分别是原图中的目标节点和源节点，翻转是便于进行消息传递和汇聚)
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()

    """
    print("sampled src number", block1_dst_num, flush=True)
    print("sampled dst number", block1_src_num, flush=True)
    print("\nid", ids, flush=True)
    print("\nsampled src", block1_agg_dst, flush=True)
    print("\nsampled dst", block1_agg_src, flush=True)
    sort_block1_agg_src, _ = torch.sort(block1_agg_src)
    sort_block1_agg_dst, _ = torch.sort(block1_agg_dst)
    print("\nsorted sampled src ", sort_block1_agg_dst, flush=True)
    print("\nsorted sampled dst ", sort_block1_agg_src, flush=True)

    print("\nfeature length: {}, features ".format(features.shape), features, flush=True)
    print("\nlabels length: {}".format(labels.shape), labels, flush=True)
    """

    # ======================================== 混合数据结构读取 ========================================
    sparse_matrix_size = SparseMatrixSize(*tuple(ipc_service.get_sparseMatrix_size()))
    sparse_matrix_data = SparseMatrixData(*tuple(ipc_service.get_sparseMatrix_data()))

    #print("\n[Info] first bcsr data:", flush=True)
    #print("num_rows: {}, num_cols: {}, nnz: {}, num_TCBlocks: {}, num_rowWindows: {}".format(sparse_matrix_size.f_bcsr_num_rows, sparse_matrix_size.f_bcsr_num_cols, sparse_matrix_size.f_bcsr_nnz, sparse_matrix_size.f_bcsr_num_TCBlocks, sparse_matrix_size.f_bcsr_num_rowWindows), flush=True)
    """
    print("rowWindowOffset: ", sparse_matrix_data.f_bcsr_rowWindowOffset, flush=True)
    print("tcOffset: ", sparse_matrix_data.f_bcsr_tcOffset, flush=True)
    print("sparseA2B: ", sparse_matrix_data.f_bcsr_sparseA2B, flush=True)
    print("tcLocalBit: ", sparse_matrix_data.f_bcsr_tcLocalBit, flush=True)
    print("data: ", sparse_matrix_data.f_bcsr_data, flush=True)
    """

    #print("\n[Info] first bcsc data:", flush=True)
    #print("num_rows: {}, num_cols: {}, nnz: {}, num_TCBlocks: {}, num_colWindows: {}".format(sparse_matrix_size.f_bcsc_num_rows, sparse_matrix_size.f_bcsc_num_cols, sparse_matrix_size.f_bcsc_nnz, sparse_matrix_size.f_bcsc_num_TCBlocks, sparse_matrix_size.f_bcsc_num_colWindows), flush=True)
    """
    print("colWindowOffset: ", sparse_matrix_data.f_bcsc_colWindowOffset, flush=True)
    print("tcOffset: ", sparse_matrix_data.f_bcsc_tcOffset, flush=True)
    print("sparseA2C: ", sparse_matrix_data.f_bcsc_sparseA2C, flush=True)
    print("tcLocalBit: ", sparse_matrix_data.f_bcsc_tcLocalBit, flush=True)
    print("data: ", sparse_matrix_data.f_bcsc_data, flush=True)
    """

    #print("\n[Info] second bcsr data:", flush=True)
    #print("num_rows: {}, num_cols: {}, nnz: {}, num_TCBlocks: {}, num_rowWindows: {}".format(sparse_matrix_size.s_bcsr_num_rows, sparse_matrix_size.s_bcsr_num_cols, sparse_matrix_size.s_bcsr_nnz, sparse_matrix_size.s_bcsr_num_TCBlocks, sparse_matrix_size.s_bcsr_num_rowWindows), flush=True)
    """
    print("rowWindowOffset: ", sparse_matrix_data.s_bcsr_rowWindowOffset, flush=True)
    print("tcOffset: ", sparse_matrix_data.s_bcsr_tcOffset, flush=True)
    print("sparseA2B: ", sparse_matrix_data.s_bcsr_sparseA2B, flush=True)
    print("tcLocalBit: ", sparse_matrix_data.s_bcsr_tcLocalBit, flush=True)
    print("data: ", sparse_matrix_data.s_bcsr_data, flush=True)
    """

    #print("\n[Info] second bcsc data:", flush=True)
    #print("num_rows: {}, num_cols: {}, nnz: {}, num_TCBlocks: {}, num_colWindows: {}".format(sparse_matrix_size.s_bcsc_num_rows, sparse_matrix_size.s_bcsc_num_cols, sparse_matrix_size.s_bcsc_nnz, sparse_matrix_size.s_bcsc_num_TCBlocks, sparse_matrix_size.s_bcsc_num_colWindows), flush=True)
    """
    print("colWindowOffset: ", sparse_matrix_data.s_bcsc_colWindowOffset, flush=True)
    print("tcOffset: ", sparse_matrix_data.s_bcsc_tcOffset, flush=True)
    print("sparseA2C: ", sparse_matrix_data.s_bcsc_sparseA2C, flush=True)
    print("tcLocalBit: ", sparse_matrix_data.s_bcsc_tcLocalBit, flush=True)
    print("data: ", sparse_matrix_data.s_bcsc_data, flush=True)
    #"""

    # ======================================== feature 信息和数据读取 ========================================
    cache_search_map = ipc_service.get_memoryaccess_data()[0] # 这个函数目前只返回了一个 tensor，但是也放到了一个 vector 中，所以类型是 list 而非 tensor
    cpu_float_feature_len, gpu_node_capacity, feature_dim = ipc_service.get_feature_attribute()
    cpu_feature_cache_ptr, gpu_feature_cache_ptr = ipc_service.get_feature_cache_ptr()
    
    """
    print("cache_search_map: ", cache_search_map, flush=True)
    print("cpu_float_feature_len: {}, gpu_node_capacity: {}, feature_dim: {}".format(cpu_float_feature_len, gpu_node_capacity, feature_dim))
    #spmm_cnic.read_cpu_feature_cache(cpu_feature_cache_ptr, cpu_float_feature_len)
    """
    
    # ======================================== 计算 ========================================
    #"""
    batch_pred = model(
        sparse_matrix_size,
        sparse_matrix_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id
    )
    #"""

    #print("result: ", batch_pred, flush=True)

    #"""
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    #=================================== debug-模拟 label
    #batch_size = batch_pred.shape[0]
    #num_classes = batch_pred.shape[1]
    #long_labels = torch.randint(0, num_classes, (batch_size,), device=batch_pred.device)
    #=================================== debug-模拟 label

    #print("batch_pred shape: {}, labels shape: {}".format(batch_pred.shape, long_labels.shape), flush=True)
    # nn.CrossEntropyLoss() 损失计算函数会完成两个步骤：1.先对 batch_pred 做 softmax，得到每个类别的概率分布 2.用目标 long_labels 计算 负对数似然损失
    loss = loss_fcn(batch_pred, long_labels)

    optimizer.zero_grad() # 清空梯度(.grad)，避免累积
    loss.backward() # 反向传播
    optimizer.step() # 权重更新(基于 optimizer 设置的优化器对参数进行更新)
    #"""

    # 一个 batch 的计算完成后，需要将 AX 的结果 free，避免显存占用持续增大
    spmm_cnic.freeMatC()

    torch.cuda.synchronize()
    ipc_service.synchronize()

    return loss
    #return 0

def valid_one_step(model, metric, device, feat_len):
    # 原本由采样阶段负责的 features 数据，现在在算子执行过程中获取，因此此处删除了 features 数据，labels 数据量较小，可以采用原始方案
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()

    # ======================================== 混合数据结构读取 ========================================
    bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows, bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows = ipc_service.get_sparseMatrix_size()
    bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data, bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data = ipc_service.get_sparseMatrix_data()

    # ======================================== feature 信息和数据读取 ========================================
    cache_search_map = ipc_service.get_memoryaccess_data()[0] # 这个函数目前只返回了一个 tensor，但是也放到了一个 vector 中，所以类型是 list 而非 tensor
    cpu_float_feature_len, gpu_node_capacity, feature_dim = ipc_service.get_feature_attribute()
    cpu_feature_cache_ptr, gpu_feature_cache_ptr = ipc_service.get_feature_cache_ptr()

    # ======================================== 计算 ========================================
    batch_pred = model(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data,
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr
    )

    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    #=================================== debug-模拟 label
    #batch_size = batch_pred.shape[0]
    #num_classes = batch_pred.shape[1]
    #long_labels = torch.randint(0, num_classes, (batch_size,), device=batch_pred.device)
    #=================================== debug-模拟 label

    batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)

    ipc_service.synchronize()
    return acc

def test_one_step(model, metric, device, feat_len):
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()

    # ======================================== 混合数据结构读取 ========================================
    bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows, bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows = ipc_service.get_sparseMatrix_size()
    bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data, bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data = ipc_service.get_sparseMatrix_data()

    # ======================================== feature 信息和数据读取 ========================================
    cache_search_map = ipc_service.get_memoryaccess_data()[0] # 这个函数目前只返回了一个 tensor，但是也放到了一个 vector 中，所以类型是 list 而非 tensor
    cpu_float_feature_len, gpu_node_capacity, feature_dim = ipc_service.get_feature_attribute()
    cpu_feature_cache_ptr, gpu_feature_cache_ptr = ipc_service.get_feature_cache_ptr()

    # ======================================== 计算 ========================================
    batch_pred = model(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data,
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr
    )

    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    #=================================== debug-模拟 label
    #batch_size = batch_pred.shape[0]
    #num_classes = batch_pred.shape[1]
    #long_labels = torch.randint(0, num_classes, (batch_size,), device=batch_pred.device)
    #=================================== debug-模拟 label

    batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)

    ipc_service.synchronize()
    return acc

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

    # 创建用于计算分类任务准确率的度量指标, 在训练和验证时跟踪模型的准确率
    """
    train_metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
    train_metric = train_metric.to(cuda_device)
    valid_metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
    valid_metric = valid_metric.to(cuda_device)
    """

    if DEBUG == True:
        train_steps = 1 # 3
    
    epoch_num = args.epoch
    for epoch in range(epoch_num):
        # =========================== train ===========================
        model.train()
        print("device-{}, epoch-{} train".format(rank, epoch), flush=True)
        epoch_time = 0
        start = time.time()
        for iter in range(train_steps):
            train_loss = train_one_step(model, optimizer, loss_fcn, cuda_device, feat_len, iter, device_id)
            if device_id == 0:
                print("epoch-{}, iter-{}, loss: {}".format(epoch, iter, train_loss), flush=True)
        epoch_time += time.time() - start
        #train_metric.reset()
        
        # =========================== valid ===========================
    """
        model.eval()
        metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
        metric = metric.to(device_id)
        model.metric = metric

        print("device-{}, epoch-{} valid".format(rank, epoch), flush=True)
        with torch.no_grad():
            for iter in range(valid_steps):
                valid_one_step(model, metric, cuda_device, feat_len)
            acc_val = metric.compute()
        
        if device_id == 0:
            print("Epoch:{}, Cost:{} s, Val Acc: {}".format(epoch, epoch_time, acc_val))

    # =========================== test ===========================
    model.eval()
    metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
    metric = metric.to(device_id)
    model.metric = metric

    print("device-{}, epoch-{} test".format(rank, epoch), flush=True)
    with torch.no_grad():
        for iter in range(test_steps):
            test_one_step(model, metric, cuda_device, feat_len)
        acc = metric.compute()

    if device_id == 0:
        print("Accuracy on test data: {}".format(acc))
    metric.reset()
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