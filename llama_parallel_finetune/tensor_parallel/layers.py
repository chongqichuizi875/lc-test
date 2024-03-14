import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Sequence
from llama_parallel_finetune.tensor_parallel.mappings import(
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from llama_parallel_finetune.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
SplitDim = {
    'COLUMN': 0,
    'ROW': -1
}

class VocabUtility:
    """ Split the vocabulary into `world_size` chunks and return the first
        and last index of the vocabulary belonging to the `rank`
        partition: Note that indices in [fist, last)

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        per_partition_vocab_size = global_vocab_size // world_size
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )

class VocabParallelEmbedding(nn.Module):
    def __init__(self, emb: nn.Embedding):
        super(VocabParallelEmbedding, self).__init__()
        self.num_embeddings = emb.num_embeddings
        self.embedding_dim = emb.embedding_dim
        self.padding_idx = emb.padding_idx
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # 根据当前进程在TP组中的序号，确定其所需维护的WE部分，沿着vocab维度对WE进行切割
        # 例如，进程id=0, 维护词表序号[0,5)范围内的数据；进程id=1，维护[5,10)
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings,
            get_tensor_model_parallel_rank(),
            self.tensor_model_parallel_size,
        )
        # 计算当前进程维护的词表大小
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )
        idx = dist.get_rank() % self.tensor_model_parallel_size
        self.weight = nn.Parameter(torch.split(emb.weight, self.num_embeddings_per_partition, dim=SplitDim['COLUMN'])[idx])
        # print(f"vocab size: {self.num_embeddings}, emb dim: {self.embedding_dim}, after split: {self.weight.shape}")

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1: # 如果使用TP
            # 如果在当前进程维护的WE上，找不到对应的单词，那么对应位置就赋0
            # 例如当前的数据的tokenid是：[2,7,1,5]，当前维护的词表是[0,1,2](start_index=0, end_index = 3)，
            # 则mask之后的数据为[2,0,1,0]
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        # 输入X，过当前进程维护的部分WE的结果
        output_parallel = F.embedding(
            masked_input, # tensor containing indices into the embedding matrix
            self.weight, # 切割好的word embedding的权重
            self.padding_idx,
            # self.max_norm,
            # self.norm_type,
            # self.scale_grad_by_freq,
            # self.sparse,
        )
        # 当前词表不维护的部分，都设为0
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0 #
        
        # 将TP组各GPU上的结果做AllReduce
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        skip_bias_add = False
    ):
        super(RowParallelLinear, self).__init__()
        self.tp = get_tensor_model_parallel_world_size()
        # Keep input parameters
        self.input_size = linear.in_features
        self.output_size = linear.out_features
        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = self.input_size // self.tp
        idx = dist.get_rank() % self.tp
        self.skip_bias_add = skip_bias_add
        # Y = XA^T + b
        # X (b, s, in_features)
        # A (out_features, in_features) nn.LInear的初始化是反的nn.Linear(in, out) -> shape (out, in)，所以dim=-1
        self.weight = nn.Parameter(torch.split(linear.weight, self.input_size_per_partition, dim=SplitDim['ROW'])[idx])
        setattr(self.weight, 'allreduce', True)
        if linear.bias:
            self.bias = nn.Parameter(linear.bias)
            setattr(self.bias, 'allreduce', True)
        else:
            self.bias = None
        # setattr(self.bias, 'sequence_parallel', False)

    def forward(self, input_):
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output
        
class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        gather_output = True, # 是否对输出allgather
        skip_bias_add = False,
    ):
        super(ColumnParallelLinear, self).__init__()
        self.tp = get_tensor_model_parallel_world_size()
        self.input_size = linear.in_features
        self.output_size = linear.out_features
        self.gather_output = gather_output
        self.output_size_per_partition = self.output_size // self.tp
        idx = dist.get_rank() % self.tp
            
        self.weight = nn.Parameter(torch.split(linear.weight, self.output_size_per_partition, dim=SplitDim['COLUMN'])[idx])
        self.bias = nn.Parameter(torch.split(linear.bias, self.output_size_per_partition, dim=SplitDim['COLUMN'])[idx]) if linear.bias else None
        self.skip_bias_add = skip_bias_add

    def forward(self, input_):
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        # print(f"output parallel shape: {output_parallel.shape}")
        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        # print(f"output shape: {output.shape}")
        output_bias = self.bias if self.skip_bias_add else None
        return output
