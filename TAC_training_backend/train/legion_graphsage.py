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

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as Func

import ipc_service
import dgl
from dgl.nn.pytorch import SAGEConv
from dgl.heterograph import DGLBlock
import time
import numpy as np
import torchmetrics
torch.set_printoptions(threshold=np.inf)

debug = False

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    if torch.cuda.is_available():
      dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
      dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        for _ in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

def create_dgl_block(src, dst, num_src_nodes, num_dst_nodes):
    gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, src, dst, 'coo', row_sorted=True)
    g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])

    return g

def train_one_step(model, optimizer, loss_fcn, device, feat_len, iter, device_id):
    
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()

    """
    print("sampled src number", block1_dst_num)
    print("sampled dst number", block1_src_num)
    print("\nid", ids)
    print("\nsampled src", block1_agg_dst)
    print("\nsampled dst", block1_agg_src)
    sort_block1_agg_src, _ = torch.sort(block1_agg_src)
    sort_block1_agg_dst, _ = torch.sort(block1_agg_dst)
    print("\nsorted sampled src ", sort_block1_agg_dst)
    print("\nsorted sampled dst ", sort_block1_agg_src)
    """

    """
    # 也可以在 C++ 层面就把数据打包，这里只暴露出 sparseMatrix，但是需要额外维护这个对象的生命周期。采用这里暴露指针，在实际计算之前再组装结构体（都是指针，不存在转换开销)
    bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows, bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows = ipc_service.get_sparseMatrix_size()
    bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit, bcsr_data, bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, bcsc_data = ipc_service.get_sparseMatrix_data()
    
    print("\n[Info] bcsr data:")
    print("num_rows: {}, num_cols: {}, nnz: {}, num_TCBlocks: {}, num_rowWindows: {}".format(bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows))
    print("rowWindowOffset: ", bcsr_rowWindowOffset)
    print("tcOffset: ", bcsr_tcOffset)
    print("sparseA2B: ", bcsr_sparseA2B)
    print("tcLocalBit: ", bcsr_tcLocalBit)
    print("data: ", bcsr_data)

    print("\n\n[Info] bcsc data:")
    print("num_rows: {}, num_cols: {}, nnz: {}, num_TCBlocks: {}, num_colWindows: {}".format(bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows))
    print("colWindowOffset: ", bcsc_colWindowOffset)
    print("tcOffset: ", bcsc_tcOffset)
    print("sparseA2C: ", bcsc_sparseA2C)
    print("tcLocalBit: ", bcsc_tcLocalBit)
    print("data: ", bcsc_data)
    """

    # 测试把这些数据传递给 C++ 函数，能否正常组装成 BCSR 和 BCSC 类，从而方面后续进行 SpMM 算子的调用

    blocks = []
    blocks.append(create_dgl_block(block1_agg_src, block1_agg_dst, block1_src_num, block1_dst_num))
    blocks.append(create_dgl_block(block2_agg_src, block2_agg_dst, block2_src_num, block2_dst_num))
    # print(features[:100])
    # print(ids[:100])
    batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    loss = loss_fcn(batch_pred, long_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize()
    ipc_service.synchronize()
    #return 0
    return loss

def valid_one_step(model, metric, device, feat_len):
    
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    blocks = []
    blocks.append(create_dgl_block(block1_agg_src, block1_agg_dst, block1_src_num, block1_dst_num))
    blocks.append(create_dgl_block(block2_agg_src, block2_agg_dst, block2_src_num, block2_dst_num))
    batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)
    ipc_service.synchronize()
    return acc

def test_one_step(model, metric, device, feat_len):
    
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    blocks = []
    blocks.append(create_dgl_block(block1_agg_src, block1_agg_dst, block1_src_num, block1_dst_num))
    blocks.append(create_dgl_block(block2_agg_src, block2_agg_dst, block2_src_num, block2_dst_num))
    batch_pred = model(blocks, features)
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    batch_pred = torch.softmax(batch_pred, dim=1).to(device)
    acc = metric(batch_pred, long_labels)
    ipc_service.synchronize()
    return acc

def worker_process(rank, world_size, args):
    print(f"Running GNN Training on CUDA {rank}.", flush=True)
    device_id = rank
    setup(rank, world_size)
    cuda_device = torch.device("cuda:{}".format(device_id))
    torch.cuda.set_device(cuda_device)
    
    # initialize 过程会调用 sem_post，由此采样进程会填充两个 buffer 的采样数据
    ipc_service.initialize(world_size)

    train_steps, valid_steps, test_steps = ipc_service.get_steps()
    print("[Info]: train_steps: {}, valid_steps: {}, test_steps: {}".format(train_steps, valid_steps, test_steps))

    feat_len = args.features_num

    model = SAGE(in_feats=args.features_num,
                        n_hidden=args.hidden_dim,
                        n_classes=args.class_num,
                        n_layers=args.hops_num,
                        activation=Func.relu,
                        dropout=args.drop_rate).to(cuda_device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[device_id])
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    epoch_num = args.epoch

    if debug == True:
        train_steps = 1

    for epoch in range(epoch_num):
        forward = 0
        start = time.time()
        epoch_time = 0
        for iter in range(train_steps):
            train_loss = train_one_step(model, optimizer, loss_fcn, cuda_device, feat_len, iter, device_id)    
            if device_id == 0:
                print('Iter {} Train Loss :{} '.format(iter, train_loss), flush=True)
        epoch_time += time.time() - start
        
        model.eval()
        metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
        metric = metric.to(device_id)
        model.metric = metric
        
        with torch.no_grad():
            for iter in range(valid_steps):
                valid_one_step(model, metric, cuda_device, feat_len)
            acc_val = metric.compute()
        if device_id == 0:
            print("Epoch:{}, Cost:{} s, Val Acc: {}".format(epoch, epoch_time, acc_val))

    
    model.eval()
    metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
    metric = metric.to(device_id)
    model.metric = metric
    with torch.no_grad():
        for iter in range(test_steps):
            test_one_step(model, metric, cuda_device, feat_len)
        acc = metric.compute()
    if device_id == 0:
        print("Accuracy on test data: {}".format(acc))
    metric.reset()

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
