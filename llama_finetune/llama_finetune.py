import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler, custom_fwd, custom_bwd
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoModel, AutoConfig, LlamaConfig, LlamaModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset
from typing import List
import torch.nn.functional as F
import json
import sys
from datasets import load_dataset
from tqdm import tqdm
import os
import copy
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaDecoderLayer
sys.path.append('.')
global TP
global DP
TP = 4
DP = 1
def gpu_usage(local_rank):
   return torch.cuda.memory_allocated(local_rank) / (1024**2) 


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if TP == 1:
        return input_

    # All-reduce.
    dist.all_reduce(input_, group=_TENSOR_MODEL_PARALLEL_GROUP)

    return input_

def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def _split_along_last_dim(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = TP
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=_TENSOR_MODEL_PARALLEL_GROUP)
    output = input_list[rank].contiguous()  

    return output

def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = TP
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = dist.get_rank(group=_TENSOR_MODEL_PARALLEL_GROUP)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    dist.all_gather(tensor_list, input_, group=_TENSOR_MODEL_PARALLEL_GROUP)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output



class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)

    

# 定义一个简单的数据集
class SimpleDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], 
                                  return_tensors='pt', 
                                  padding='max_length', 
                                  truncation=True, 
                                  max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze(0)  # Remove the batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask

def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)

def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)

def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)
        
def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)

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


class ParallelLlamaSelfAttention(nn.Module):
    def __init__(self, attn: LlamaAttention, tp, world_size):
        super(ParallelLlamaSelfAttention, self).__init__()
        self.tp = tp
        self.world_size = world_size
        idx = dist.get_rank() % self.tp
        # attn.q_proj = nn.Linear(attn.q_proj.in_features // self.tp, attn.q_proj.out_features)
        # attn.k_proj = nn.Linear(attn.k_proj.in_features // self.tp, attn.k_proj.out_features)
        # attn.v_proj = nn.Linear(attn.v_proj.in_features // self.tp, attn.v_proj.out_features)
        # attn.o_proj = nn.Linear(attn.o_proj.in_features, attn.o_proj.out_features // self.tp)
        attn.q_proj.weight = nn.Parameter(torch.split(attn.q_proj.weight, attn.q_proj.in_features // self.tp, -1)[idx])
        attn.k_proj.weight = nn.Parameter(torch.split(attn.k_proj.weight, attn.k_proj.in_features // self.tp, -1)[idx])
        attn.v_proj.weight = nn.Parameter(torch.split(attn.v_proj.weight, attn.v_proj.in_features // self.tp, -1)[idx])
        attn.o_proj.weight = nn.Parameter(torch.split(attn.o_proj.weight, attn.o_proj.out_features // self.tp, 0)[idx])
        self.model = attn
    def get_model(self):
        return self.model

    # def forward(self):
    #     print("pre forward")
    #     self.model.forward()




class ParallelLlama(nn.Module):
    def __init__(self, model: LlamaModel, tp, world_size):
        super(ParallelLlama, self).__init__()
        self.tp = tp
        self.world_size = world_size
        self.module_hook = dict()
        self.name_hook = dict()
        self.global_index = 0
        self.reduced_model = self._create_reduced_model(model)
        # local_rank = dist.get_rank()
        # ori_memory = gpu_usage(local_rank)
        # model.to(f"cuda:{local_rank}")
        # model_consumption = gpu_usage(local_rank) - ori_memory
        # self.reduced_model = self._create_reduced_model(model)
        # self.reduced_model.to(f"cuda:{local_rank}")
        # reduced_consumption = gpu_usage(local_rank) - ori_memory - model_consumption
        # print(f"model: {model_consumption}, reduced_model: {reduced_consumption}")
        
        
        
    def get_model(self):
        return self.reduced_model
        
    def _create_hook_for_modules(self, model):
        if self.global_index == 0:
            return
        for name, module in model.named_modules():
            self.name_hook[self.global_index] = name
            self.module_hook[self.global_index] = module
            self.global_index += 1
    
    def _create_reduced_model(self, original_model):
        reduced_model = copy.deepcopy(original_model)
        self._create_hook_for_modules(reduced_model)

        for name, module in reduced_model.named_modules():
            if name.endswith("self_attn"):
                setattr(module, 'q_proj', ColumnParallelLinear(module.q_proj, self.tp))
                setattr(module, 'k_proj', ColumnParallelLinear(module.k_proj, self.tp))
                setattr(module, 'v_proj', ColumnParallelLinear(module.v_proj, self.tp))
                setattr(module, 'o_proj', RowParallelLinear(module.o_proj, self.tp))
            # if module.__class__.__name__.endswith("LlamaDecoderLayer"):
            #     setattr(module, 'self_attn', ParallelLlamaSelfAttention(module.self_attn, self.tp, self.world_size).get_model())
                # setattr(module, 'mlp', ParallelLlamaMlp(module.mlp, self.tp, self.world_size))
            # if name.endswith("self_attn"):  # 假设嵌套模块的名称以 ".model" 结尾
            #     self._reduce_attn_layer(module)
            #     print(module.__class__.__name__.endswith("LlamaAttention"))
        
        
        return reduced_model
    


class ParallTrainer():
    def __init__(self, 
                 model, 
                 tokenizer, 
                 gradient_checkpointing=False, 
                 mixed_precision=False, 
                 tp=4,
                 dp=2):
        self.model = model
        self.tokenizer = tokenizer
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.tp = tp
        self.dp = dp
        self.initialize_parallism()

    def initialize_parallism(self):
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device_count = torch.cuda.device_count()
        if rank == 0:
            print("> initializing torch distributed env", flush=True)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(world_size=world_size, rank=rank,
                                init_method="env://", backend="nccl")
        global _TENSOR_MODEL_PARALLEL_GROUP
        num_tensor_model_parallel_groups = world_size // self.tp
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(
                i * self.tp, (i + 1) * self.tp
            )
            group = dist.new_group(ranks) # 设置TP组
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group
        if self.dp > 1:
            global _DATA_PARALLEL_GROUP
            num_data_parallel_groups = world_size // self.dp
            for i in range(num_data_parallel_groups):
                ranks = [i*self.dp, i*self.dp+1]
                print(ranks)
                group = dist.new_group(ranks) # 设置DP组
                if ranks in ranks:
                    _DATA_PARALLEL_GROUP = group
        
        if self.gradient_checkpointing:
           self.model.gradient_checkpointing_enable()

        if self.dp > 1:
            i = torch.cuda.current_device()
        ori_memory = gpu_usage(local_rank)
        test_model = copy.deepcopy(self.model)
        test_model.to(f"cuda:{local_rank}")
        model_consumption = gpu_usage(local_rank) - ori_memory
        self.model = ParallelLlama(self.model, tp=self.tp, world_size=world_size).get_model()
        self.model.to(f"cuda:{local_rank}")
        reduced_consumption = gpu_usage(local_rank) - ori_memory - model_consumption
        print(f"model: {model_consumption}, reduced_model: {reduced_consumption}")
        # for name, module in self.model.named_modules():
        #     if module.__class__.__name__.endswith("LlamaAttention"):
        #         for name, submodule in module.named_modules():
        #             print(name)
        #         break
        self.dp_group_rank = local_rank // self.tp
        self.tp_group_rank = local_rank % self.dp
        
        
        

    def train(self):
        
        if self.dp > 1:
            model = DDP(self.model)
        else:
            model = self.model
        # 定义优化器和学习率调度器
        optimizer = AdamW(model.parameters(), lr=5e-5)
        scaler = GradScaler(enabled=self.mixed_precision)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

        # 准备数据
        texts = ["This is a sample text for training."] * int(1e4)
        dataset = SimpleDataset(texts, self.tokenizer, max_length=1024)
        # dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size() // self.tp, rank=dist.get_rank(group=_DATA_PARALLEL_GROUP)) if self.dp > 1 else None
        dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
        # 训练循环
        model.train()
        total_epoches = 100
        for epoch in range(total_epoches):
            if self.dp > 1:
                sampler.set_epoch(epoch)
            with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epoches}", unit="batch", leave=False) as tepoch:
                for batch in dataloader:
                    optimizer.zero_grad()
                    input_ids, attention_mask = batch
                    input_ids = input_ids.to(f"cuda:{dist.get_rank()}")
                    attention_mask = attention_mask.to(f"cuda:{dist.get_rank()}")
                    with autocast(enabled=self.mixed_precision):
                        outputs = model(input_ids, attention_mask, labels=input_ids)
                        loss = outputs.loss

                    if self.mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
                    tepoch.set_postfix(loss="%.2f" % loss)
                    tepoch.update(1)




def main():
    # 初始化分布式训练环境
    # dist.init_process_group(backend='nccl')
    # local_rank = dist.get_rank()
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    # word_size = torch.cuda.device_count() 

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("/home/zhongyuting/model/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    with open("llama_finetune/small_llama.json", 'r') as f:
        small_llama_config = json.load(f)
    configureation = LlamaConfig(**small_llama_config)
    model = AutoModelForCausalLM.from_config(configureation)
    model.config.use_cache = False
    # model.config.pretraining_tp = torch.cuda.device_count()

    # 微调模型
    trainer = ParallTrainer(model, tokenizer, gradient_checkpointing=True, mixed_precision=True, dp=DP, tp=TP)
    trainer.train()

if __name__ == "__main__":
    main()
