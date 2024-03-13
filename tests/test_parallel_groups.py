import time
import torch
import json
import sys
import torch.distributed as dist
import os
import torch.nn as nn
# os.environ['RANK'] = '0'
rank = int(os.environ['RANK'])
# local_rank = int(os.environ['LOCAL_RANK'])
# world_size = int(os.environ['WORLD_SIZE'])

# dist.init_process_group(world_size=world_size, rank=rank,
#                         init_method="env://", backend="nccl")
import torch.nn.functional as F
# torch.cuda.set_device(local_rank)
import copy
world_size = 16
pipline_model_parallel_size = 2
tensor_model_parallel_size = 2
context_parallel_size = 2
num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
num_pipline_model_parallel_groups = world_size // pipline_model_parallel_size
data_parallel_size = world_size // (tensor_model_parallel_size*context_parallel_size*pipline_model_parallel_size)
def say(str):
    if rank == 0:
        print(str)
all_data_parallel_group_ranks_with_cp = []
for i in range(pipline_model_parallel_size):
    start_rank = i * num_pipline_model_parallel_groups
    end_rank = (i+1) * num_pipline_model_parallel_groups
    # 且同一个context，同一个tenser块，因此有
    for j in range(context_parallel_size*tensor_model_parallel_size):
        ranks = range(
            start_rank + j, end_rank, context_parallel_size*tensor_model_parallel_size
        )
        say(f"dp ranks: {list(ranks)}")
        # print(rank in ranks)
    say("")
    for j in range(tensor_model_parallel_size):
        ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
        all_data_parallel_group_ranks_with_cp.append(ranks_with_cp)
        say(f"dp ranks with cp: {list(ranks_with_cp)}")
        # print(rank in ranks)
for i in range(data_parallel_size * context_parallel_size):
    ranks = [
        data_parallel_group_ranks_with_cp[i]
        for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
    ]
    say(f"mp ranks: {ranks}")
for i in range(num_tensor_model_parallel_groups):
    ranks = range(i * tensor_model_parallel_size, (i+1) * tensor_model_parallel_size)
    say(f"tensor mp ranks: {list(ranks)}")

