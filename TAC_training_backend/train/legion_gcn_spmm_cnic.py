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

debug = False

"""
# 1.SPMMFunction（低层封装）: 将 PyTorch 张量转换成适合 CUDA 计算的格式，并在前向传播中执行 SPMM，在反向传播中计算梯度
# 第一层计算，feature 需要同时访问 cpu + gpu，因此 feature 访存时需要获取实际的节点编号，ids 需要作为函数参数
# 其他层计算，feature 全部位于 gpu，因此不需要获取实际的节点编号
class CNICSPMMFunction_TensorCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index,
                blockPartition, edgeToColumn, edgeToRow,
                hybrid_type, row_nzr, col_nzr, sample_ids, first_layer):
        # 把参数保存到 context 中，以便在反向传播时使用
        ctx.save_for_backward(X, weights, row_pointers, column_index,
                            blockPartition, edgeToColumn, edgeToRow,
                            hybrid_type, row_nzr, col_nzr)
                            
        # 构建节点编号映射表（第一层采用实际节点编号，其他层采用顺序编号，用于 SpMM 过程中 MatB 的访存）
        if first_layer == True:
            id_map = sample_ids;
        else:
            device = sample_ids.device  # 获取 sample_ids 所在的设备
            id_map = torch.arange(0, sample_ids.size(0), device=device, dtype=torch.uint32)
        
        # TensorCore实现
        X_out = spmm_cnic.forward_tensorcore(X, id_map, weights,
                                             row_pointers, column_index,
                                             blockPartition, edgeToColumn, edgeToRow,
                                             hybrid_type, row_nzr, col_nzr)[0]
        return X_out

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, blockPartition, \
        edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        
        # 使用融合版本进行反向传播
        d_input, d_weights = spmm_cnic.backward_tensorcore_fused(
            d_output, X, weights, row_pointers, column_index,
            blockPartition, edgeToColumn, edgeToRow,
            hybrid_type, row_nzr, col_nzr)
        
        return d_input, d_weights, None, None, None, None, None, None, None, None

# 2.GCNLayer（中间层）: 核心计算是 A @ X，即图的稀疏矩阵 A 与节点特征 X 的乘法,使用自定义 CUDA SPMM
class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, first_layer=False):
        super(GCNConv, self).__init__()
        self.first_layer = first_layer
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, X, row_pointers, column_index, 
                blockPartition, edgeToColumn, edgeToRow,
                hybrid_type, row_nzr, col_nzr):
        # 第一层，SpMM 算子中 feature 涉及到 cpu + gpu 访问
        # 其他层，SpMM 算子中 feature 全部位于 gpu
        return CNICSPMMFunction_TensorCore.apply(
            X, self.weights, row_pointers, column_index,
            blockPartition, edgeToColumn, edgeToRow,
            hybrid_type, row_nzr, col_nzr, self.first_layer)


# 3.GCN（高层封装）: 表示完整的 GCN 网络，由多个 GCNLayer 组成，并包含 Dropout、ReLU 等额外操作
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, activation, dropout):
        super(GCN, self).__init__()

        # 第一层
        # 调用具体 layer 的 init 方法
        self.conv1 = GCNConv(num_features, hidden_dim, fixed=1, first_layer=True)
        
        # 中间层
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GCNConv(hidden_dim, hidden_dim, fixed=0))
        
        # 最后一层
        self.conv_final = GCNConv(hidden_dim, num_classes, fixed=2)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = activation

    def forward(self, sample_ids, row_indices, col_indices, value, nnz):
        # coo -> bcsr & bcsc 格式转换，并将 bcsr & bcsc 打包为大的数据结构
        sparseMatrix = spmm_cnic.format_transform(sample_ids, row_indices, col_indices, value, nnz); 

        # 第一层
        # 这里就是调用具体 layer 的 forward 方法了
        x = self.relu(self.conv1(sparseMatrix))
        x = self.dropout(x)
        
        # 中间层
        for conv in self.hidden_layers:
            x = conv(sparseMatrix, x, row_pointers, column_index, blockPartition,
                    edgeToColumn, edgeToRow, hybrid_type, 
                    row_nzr, col_nzr, output)
            x = self.relu(x)
            x = self.dropout(x)
        
        # 最后一层
        x = self.conv_final(sparseMatrix, x, row_pointers, column_index, blockPartition,
                           edgeToColumn, edgeToRow, hybrid_type, 
                           row_nzr, col_nzr, output)
        
        return F.log_softmax(x, dim=1)
"""

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '32355'
    # initialize the process group
    if torch.cuda.is_available():
      dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
      dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

#def train_one_step(model, optimizer, loss_fcn, device, feat_len, iter, device_id):
def train_one_step(feat_len):
    # features 和 labels 所属节点是同 ids 中的节点的顺序相对应的
    # block1 和 block2 分别对应 1 阶和 2 阶邻居的目标节点和源节点(不过此处获取到的 src-dst 实际分别是原图中的目标节点和源节点，翻转是便于进行消息传递和汇聚)
    #"""
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    #"""

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
    """

    # 混合数据结构读取
    #"""
    bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows, bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows = ipc_service.get_sparseMatrix_size()
    bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data, bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data = ipc_service.get_sparseMatrix_data()
    #"""

    """
    print("\n[Info] bcsr data:", flush=True)
    print("num_rows: {}, num_cols: {}, nnz: {}, num_TCBlocks: {}, num_rowWindows: {}".format(bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows), flush=True)
    print("rowWindowOffset: ", bcsr_rowWindowOffset, flush=True)
    print("tcOffset: ", bcsr_tcOffset, flush=True)
    print("sparseA2B: ", bcsr_sparseA2B, flush=True)
    print("tcLocalBit: ", bcsr_tcLocalBit, flush=True)
    print("data: ", bcsr_data, flush=True)

    print("\n\n[Info] bcsc data:", flush=True)
    print("num_rows: {}, num_cols: {}, nnz: {}, num_TCBlocks: {}, num_colWindows: {}".format(bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows), flush=True)
    print("colWindowOffset: ", bcsc_colWindowOffset, flush=True)
    print("tcOffset: ", bcsc_tcOffset, flush=True)
    print("sparseA2C: ", bcsc_sparseA2C, flush=True)
    print("tcLocalBit: ", bcsc_tcLocalBit, flush=True)
    print("data: ", bcsc_data, flush=True)
    """

    # feature 信息和数据读取
    #"""
    cache_search_map = ipc_service.get_memoryaccess_data()[0] # 这个函数目前只返回了一个 tensor，但是也放到了一个 vector 中，所以类型是 list 而非 tensor
    cpu_float_feature_len, gpu_node_capacity, feature_dim = ipc_service.get_feature_attribute()
    cpu_feature_cache_ptr, gpu_feature_cache_ptr = ipc_service.get_feature_cache_ptr()
    #"""
    

    """
    print("cache_search_map: ", cache_search_map, flush=True)
    print("cpu_float_feature_len: {}, gpu_node_capacity: {}, feature_dim: {}".format(cpu_float_feature_len, gpu_node_capacity, feature_dim))
    #spmm_cnic.read_cpu_feature_cache(cpu_feature_cache_ptr, cpu_float_feature_len)
    """

    # 向 C++ api 传递数据完成 BCSR 和 BCSC 结构重组
    #"""
    X_out = spmm_cnic.forward_tensorcore_mixed(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data,
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr
    )
    #"""

    """
    print("\n[Info] spmm result: \n", flush=True)
    X_out_flat = X_out[0].cpu().flatten()
    for i in range(0, X_out_flat.size(0), 128):
        row = X_out_flat[i:i+128]
        print("row-{}".format(i / 128), row.tolist(), flush=True)
    """
    
    # 采样进程为了适应 dgl 节点信息聚合过程对于边的要求，在存储时将节点 src->dst 的顺序进行了对调，因此这里需要对调顺序
    """
    batch_pred = model(ids, block1_agg_dst, block1_agg_src, labels)

    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    loss = loss_fcn(batch_pred, long_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    """

    torch.cuda.synchronize()
    ipc_service.synchronize()

    #return loss
    return 0

#def valid_one_step(model, metric, device, feat_len):
def valid_one_step():
    """
    # 原本由采样阶段负责的 features 数据，现在在算子执行过程中获取，因此此处删除了 features 数据，labels 数据量较小，可以采用原始方案
    ids, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()

    # 采样进程为了适应 dgl 节点信息聚合过程对于边的要求，在存储时将节点 src->dst 的顺序进行了对调，因此这里需要对调顺序
    batch_pred = model(ids, block1_agg_dst, block1_agg_src, labels)

    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)
    """
    ipc_service.synchronize()
    #return acc
    return 0

#def test_one_step(model, metric, device, feat_len):
def test_one_step():
    """ 
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()

    # TODO blocks 修改为 model forward 所需的参数
    blocks = []
    blocks.append(create_dgl_block(block1_agg_src, block1_agg_dst, block1_src_num, block1_dst_num))
    blocks.append(create_dgl_block(block2_agg_src, block2_agg_dst, block2_src_num, block2_dst_num))

    batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)
    """
    ipc_service.synchronize()
    #return acc
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

    """
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

    model.train()
    """

    epoch_num = args.epoch

    if debug == True:
        train_steps = 1

    for epoch in range(epoch_num):
        forward = 0

        # =========================== train ===========================
        print("device-{}, epoch-{} train".format(rank, epoch), flush=True)
        start = time.time()
        epoch_time = 0
        for iter in range(train_steps):
            #train_loss = train_one_step(model, optimizer, loss_fcn, cuda_device, feat_len, iter, device_id)    
            train_loss = train_one_step(feat_len)    
            # if device_id == 0:
            #     print('Iter {} Train Loss :{} '.format(iter, train_loss))
        epoch_time += time.time() - start
        
        # =========================== valid ===========================
        """
        model.eval()
        metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
        metric = metric.to(device_id)
        model.metric = metric
        """

        print("device-{}, epoch-{} valid".format(rank, epoch), flush=True)
        with torch.no_grad():
            for iter in range(valid_steps):
                #valid_one_step(model, metric, cuda_device, feat_len)
                valid_one_step()
        """
            acc_val = metric.compute()
        
        if device_id == 0:
            print("Epoch:{}, Cost:{} s, Val Acc: {}".format(epoch, epoch_time, acc_val))
        """

    # =========================== test ===========================
    """
    model.eval()
    metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
    metric = metric.to(device_id)
    model.metric = metric
    """

    print("device-{}, epoch-{} test".format(rank, epoch), flush=True)
    with torch.no_grad():
        for iter in range(test_steps):
            #test_one_step(model, metric, cuda_device, feat_len)
            test_one_step()
    """
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
