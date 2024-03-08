
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from tensor_parallel.mappings import(
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)

class RowParallelLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        tp: int,
        skip_bias_add = False
    ):
        super(RowParallelLinear, self).__init__()
        self.tp = tp
        # Keep input parameters
        self.input_size = linear.in_features
        self.output_size = linear.out_features
        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = self.input_size // tp
        idx = dist.get_rank() % tp
        self.skip_bias_add = skip_bias_add
        # Y = XA^T + b
        # X (b, s, in_features)
        # A (out_features, in_features) nn.LInear的初始化是反的nn.Linear(in, out) -> shape (out, in)，所以dim=-1
        self.weight = nn.Parameter(torch.split(linear.weight, self.input_size_per_partition, dim=-1)[idx])
        # setattr(self.weight, 'allreduce', True)
        self.bias = nn.Parameter(linear.bias) if linear.bias else None
        # setattr(self.bias, 'allreduce', True)
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
        tp: int,
        gather_output = True, # 是否对输出allgather
        skip_bias_add = False,
    ):
        super(ColumnParallelLinear, self).__init__()
        self.tp = tp
        self.input_size = linear.in_features
        self.output_size = linear.out_features
        self.gather_output = gather_output
        self.output_size_per_partition = self.output_size // tp
        idx = dist.get_rank() % tp
        self.weight = nn.Parameter(torch.split(linear.weight, self.output_size_per_partition, dim=0)[idx])
        self.bias = nn.Parameter(torch.split(linear.bias, self.output_size_per_partition, dim=0)[idx]) if linear.bias else None
        self.skip_bias_add = skip_bias_add

    def forward(self, input_):
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output