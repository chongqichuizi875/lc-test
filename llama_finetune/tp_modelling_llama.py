import torch.distributed as dist
import os
import torch
from utils import gpu_usage
import copy
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler, custom_fwd, custom_bwd
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoModel, AutoConfig, LlamaConfig, LlamaModel
from custom_dataset import SimpleDataset
import torch.nn as nn
import torch.nn.functional as F
from parallel_state import initialize_model_parallel, get_data_parallel_group
from tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear

class ParallelTrainer():
    def __init__(self, 
                 model, 
                 tokenizer, 
                 gradient_checkpointing=False, 
                 mixed_precision=False, 
                 tp=4,
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.tp = tp
        self.initialize_parallism()

    def initialize_parallism(self):
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if rank == 0:
            print("> initializing torch distributed env", flush=True)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(world_size=world_size, rank=rank,
                                init_method="env://", backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=self.tp)
        
        if self.gradient_checkpointing:
           self.model.gradient_checkpointing_enable()

        ori_memory = gpu_usage(local_rank)
        test_model = copy.deepcopy(self.model)
        test_model.to(f"cuda:{local_rank}")
        model_consumption = gpu_usage(local_rank) - ori_memory
        self.model = ParallelLlama(self.model, tp=self.tp, world_size=world_size).get_model()
        self.model.to(f"cuda:{local_rank}")
        reduced_consumption = gpu_usage(local_rank) - ori_memory - model_consumption
        print(f"model: {model_consumption}, reduced_model: {reduced_consumption}")
        
        
        

    def train(self):
        
        
        model = DDP(self.model,
                    process_group=get_data_parallel_group())
        
        # 定义优化器和学习率调度器
        optimizer = AdamW(model.parameters(), lr=5e-5)
        scaler = GradScaler(enabled=self.mixed_precision)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

        # 准备数据
        texts = ["This is a sample text for training."] * int(1e4)
        dataset = SimpleDataset(texts, self.tokenizer, max_length=1024)
        # dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size() // self.tp, rank=dist.get_rank(group=get_data_parallel_group()))
        dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
        # 训练循环
        model.train()
        total_epoches = 100
        for epoch in range(total_epoches):
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


class ParallelLlama(nn.Module):
    def __init__(self, model: LlamaModel, tp, world_size):
        super(ParallelLlama, self).__init__()
        self.tp = tp
        self.world_size = world_size
        self.module_hook = dict()
        self.name_hook = dict()
        self.global_index = 0
        self.reduced_model = self._create_reduced_model(model)        
        
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
            if name.endswith("mlp"):
                # setattr(module, 'gate_proj', RowParallelLinear(module.gate_proj))
                # setattr(module, 'up_proj', RowParallelLinear(module.up_proj))
                setattr(module, 'down_proj', ColumnParallelLinear(module.down_proj))
            # if name.endswith("self_attn"):
                # setattr(module, 'q_proj', ColumnParallelLinear(module.q_proj))
                # setattr(module, 'k_proj', ColumnParallelLinear(module.k_proj))
                # setattr(module, 'v_proj', ColumnParallelLinear(module.v_proj))
                # setattr(module, 'o_proj', RowParallelLinear(module.o_proj))

        return reduced_model
    
