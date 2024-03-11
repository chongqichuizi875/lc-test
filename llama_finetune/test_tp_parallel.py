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
sys.path.append('.')
from datasets import load_dataset
from tp_modelling_llama import ParallelTrainer
from parallel_state import get_data_parallel_group
from custom_dataset import SimpleDataset
import os
local_rank = int(os.environ['LOCAL_RANK'])


tokenizer = AutoTokenizer.from_pretrained("/home/zhongyuting/model/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
with open("small_llama.json", 'r') as f:
    small_llama_config = json.load(f)
configureation = LlamaConfig(**small_llama_config)
model = AutoModelForCausalLM.from_config(configureation)
model.config.use_cache = False
# model.config.pretraining_tp = torch.cuda.device_count()
tp = 1

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

