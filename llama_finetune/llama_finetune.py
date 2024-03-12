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
import os


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if rank == 0:
        print("> initializing torch distributed env", flush=True)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(world_size=world_size, rank=rank,
                                init_method="env://", backend="nccl")
    tokenizer = AutoTokenizer.from_pretrained("/home/zhongyuting/model/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    with open("small_llama.json", 'r') as f:
        small_llama_config = json.load(f)
    configureation = LlamaConfig(**small_llama_config)
    model = AutoModelForCausalLM.from_config(configureation)
    model.config.use_cache = False
    # model.config.pretraining_tp = torch.cuda.device_count()

    # 微调模型
    trainer = ParallelTrainer(model, tokenizer, gradient_checkpointing=True, mixed_precision=True, tp=8)
    trainer.train()

if __name__ == "__main__":
    main()
