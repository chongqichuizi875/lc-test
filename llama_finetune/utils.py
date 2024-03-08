import torch
from functools import reduce
import operator

def gpu_usage(local_rank):
    return torch.cuda.memory_allocated(local_rank) / (1024**2) 

class GlobalMemoryBuffer:
    def __init__(self) -> None:
      self.buffer = None
    
    def get_tensor(self, tensor_shape, dtype, name):
        # 连乘得到内存大小
        required_len = reduce(operator.mul, tensor_shape, 1)
        if self.buffer.get((name, dtype), None) is None or self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )
        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)
 