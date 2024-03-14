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
from dataset.custom_dataset import SimpleDataset
import json
import sys
from datasets import load_dataset
from llama_parallel_finetune.trainer import ParallelTrainer
import os
import argparse
from llama_parallel_finetune.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-tp', '--tensor_parallel', type=int, default=1)
    parser.add_argument('-pp', '--pipeline_parallel', type=int, default=1)
    parser.add_argument('-gc', '--gradient_checkpointing', type=int, default=1)
    parser.add_argument('-mix', '--mixed_precision', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', type=str, default='adam')
    args = parser.parse_args()

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if rank == 0:
        print("> initializing torch distributed env", flush=True)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(world_size=world_size, rank=rank,
                                init_method="env://", backend="nccl")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("/home/zhongyuting/model/Llama-2-7b-hf", model_max_length=1024)
    tokenizer.pad_token = tokenizer.eos_token
    with open("configs/small_llama.json", 'r') as f:
        small_llama_config = json.load(f)
    configureation = LlamaConfig(**small_llama_config)
    model = AutoModelForCausalLM.from_config(configureation)
    model.config.use_cache = False
    # model.config.pretraining_tp = torch.cuda.device_count()
    texts = ["This is a sample text for training."] * int(1e4)
    simple_dataset = SimpleDataset(texts, tokenizer, max_length=1024)

    dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    # 微调模型
    trainer = ParallelTrainer(
        args,
        model, 
        tokenizer, 
        gradient_checkpointing=bool(args.gradient_checkpointing), 
        mixed_precision=bool(args.mixed_precision), 
        tp=args.tensor_parallel,
        dataset=tokenized_datasets)
    trainer.train()

if __name__ == "__main__":
    main()
