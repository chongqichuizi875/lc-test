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
from tp_modelling_llama import ColumnParallelLinear, RowParallelLinear
import json
import sys
import copy
sys.path.append('.')
from datasets import load_dataset
from tp_modelling_llama import ParallelTrainer
from parallel_state import get_data_parallel_group
from custom_dataset import SimpleDataset
import os
from utils import set_seed
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(local_rank)
dist.init_process_group(world_size=world_size, rank=rank,
                                init_method="env://", backend="nccl")

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("/home/zhongyuting/model/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
with open("small_llama.json", 'r') as f:
    small_llama_config = json.load(f)
configureation = LlamaConfig(**small_llama_config)
model = AutoModelForCausalLM.from_config(configureation)
model.config.use_cache = False
# model.config.pretraining_tp = torch.cuda.device_count()
tp = 4

# 微调模型
trainer = ParallelTrainer(model, tokenizer, gradient_checkpointing=True, mixed_precision=True, tp=tp)
parallel_model = trainer.model
model.to(f"cuda:{local_rank}")
model.train()
model = DDP(model,
            process_group=get_data_parallel_group())


# 准备数据
texts = ["This is a sample text for training."] * int(1e4)
dataset = SimpleDataset(texts, tokenizer, max_length=1024)
# dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size() // tp, rank=dist.get_rank(group=get_data_parallel_group()))
dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
for batch in dataloader:
    input_ids, attention_mask = batch
    input_ids = input_ids.to(f"cuda:{dist.get_rank()}")
    attention_mask = attention_mask.to(f"cuda:{dist.get_rank()}")
    outputs = model(input_ids, attention_mask, labels=input_ids)
    parallel_outputs = parallel_model(input_ids, attention_mask, labels=input_ids)
    break
if local_rank == 0:
    print(outputs)
    print(parallel_outputs)

# for name, module in model.named_modules():
#     if name.endswith("mlp"):
#         ori_mlp = copy.deepcopy(module)
#         print(name)
#         setattr(module, 'gate_proj', ColumnParallelLinear(module.gate_proj))
#         # setattr(module, 'gate_proj', RowParallelLinear(module.gate_proj))
#         # setattr(module, 'up_proj', RowParallelLinear(module.up_proj))
#         # setattr(module, 'down_proj', ColumnParallelLinear(module.down_proj))
        
#         print(f"ori mlp {dist.get_rank()}: {ori_mlp.gate_proj.weight.shape}, {ori_mlp.gate_proj.weight}")
#         split_weight = nn.Parameter(torch.split(ori_mlp.gate_proj.weight, ori_mlp.gate_proj.out_features // tp, dim=0)[dist.get_rank()])
#         print(f"ori split {dist.get_rank()}: {split_weight.shape}{split_weight}")
#         print(f"mlp {dist.get_rank()}: {module.gate_proj.weight.shape}, {module.gate_proj.weight}")
#         break

    


