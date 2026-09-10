#!/usr/bin/env bash

export LD_LIBRARY_PATH=/home/user/anaconda3/envs/gaohy_spmm_pytorch/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

# import spmm_cnic 时 nvjitlink 错误解决方法：https://github.com/vllm-project/vllm/issues/10300
export LD_LIBRARY_PATH=/home/user/anaconda3/envs/gaohy_spmm_pytorch/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

