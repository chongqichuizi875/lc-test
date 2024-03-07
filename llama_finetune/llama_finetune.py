import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoModel, AutoConfig, LlamaConfig, LlamaModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset
from typing import List
import json
import sys
from datasets import load_dataset
from tqdm import tqdm
import os
import copy
from transformers.models.llama.modeling_llama import LlamaAttention
sys.path.append('.')
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
    
class ParallelLlama(nn.Module):
    def __init__(self, model: LlamaModel, tp):
        super(ParallelLlama, self).__init__()
        self.reduced_model = self._create_reduced_model(model, tp)
        local_rank = dist.get_rank()
        ori_memory = torch.cuda.memory_allocated(local_rank) / (1024**2) 
        model.to(f"cuda:{local_rank}")
        model_consumption = torch.cuda.memory_allocated(local_rank) / (1024**2) - ori_memory
        self.reduced_model.to(f"cuda:{local_rank}")
        reduced_consumption = torch.cuda.memory_allocated(local_rank) / (1024**2) - ori_memory - model_consumption
        print(f"model: {model_consumption}, reduced_model: {reduced_consumption}")
        # self.reduced_model.load_state_dict(model.state_dict())
        # model.model.embed_tokens = ParallelLlamaEmb(model.model.embed_tokens)
        # for layer in self.partial_model.model.layers:
        #     layer.self_attn = ParallelLlamaSelfAttn(layer.self_attn)
            # layer.mlp = ParallelLlamaMlp(layer.mlp)
    
    def _create_reduced_model(self, original_model, tp):
        reduced_model = copy.deepcopy(original_model)
        
        # 遍历原始模型的所有层，并复制权重
        for model_name, model in original_model.named_children():
            if hasattr(model, 'model'):
                for layer_name, layer in model.named_children():
                    if hasattr(layer, 'self_attn'):
                        # 如果层中包含注意力层，则复制除了注意力层以外的所有子模块的权重
                        for name, module in layer.named_children():
                            if name != 'self_attn':
                                setattr(getattr(reduced_model.model, layer_name), name, copy.deepcopy(module))
                        
                        # 复制注意力层的部分权重
                        # setattr(getattr(reduced_model.model, layer_name), 'self_attn', self._reduce_attn_layer(layer.self_attn, tp))
                        reduced_model.model.layer = self._reduce_attn_layer(layer.self_attn, tp)
                    else:
                        # 如果层中不包含注意力层，则直接复制整个层的权重
                        setattr(reduced_model.model, layer_name, copy.deepcopy(layer))
            else:
                setattr(reduced_model, model_name, copy.deepcopy(model))
        
        return reduced_model
    
    def _reduce_attn_layer(self, attn, tp, dim=-1):
        reduced_attn_layer = copy.deepcopy(attn)
        idx = dist.get_rank() // tp
        reduced_attn_layer.q_proj = nn.Linear(attn.q_proj.weight.size()[1]//tp, attn.q_proj.weight.size()[0])
        reduced_attn_layer.k_proj = nn.Linear(attn.k_proj.weight.size()[1]//tp, attn.k_proj.weight.size()[0])
        reduced_attn_layer.v_proj = nn.Linear(attn.v_proj.weight.size()[1]//tp, attn.v_proj.weight.size()[0])
        with torch.no_grad():
            reduced_attn_layer.q_proj.weight.copy_(torch.chunk(attn.q_proj.weight, tp, dim)[idx])
            reduced_attn_layer.q_proj.weight.copy_(torch.chunk(attn.q_proj.weight, tp, dim)[idx]) 
            reduced_attn_layer.q_proj.weight.copy_(torch.chunk(attn.q_proj.weight, tp, dim)[idx])
        return reduced_attn_layer 

        



    
class ParallelLlamaSelfAttn(LlamaAttention):
    def __init__(self, self_attn: LlamaAttention):
        self.q_proj = torch.chunk(self_attn.q_proj, dim=-1)
        

    
    

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

        global _DATA_PARALLEL_GROUP
        num_data_parallel_groups = world_size // self.dp
        for i in range(num_data_parallel_groups):
            ranks = [i*self.dp, i*self.dp+1]
            group = dist.new_group(ranks) # 设置DP组
            if ranks in ranks:
                _DATA_PARALLEL_GROUP = group
        
        if self.gradient_checkpointing:
           self.model.gradient_checkpointing_enable()

        if self.dp > 1:
            i = torch.cuda.current_device()
        
        model = ParallelLlama(self.model, tp=self.tp)
        
        
        

    def train(self):
        # 设置模型
        self.model.to(device)
        
        if self.data_parallel:
            model = DDP(model)

        # 定义优化器和学习率调度器
        optimizer = AdamW(model.parameters(), lr=5e-5)
        scaler = GradScaler(enabled=self.mixed_precision)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)

        # 准备数据
        texts = ["This is a sample text for training."] * int(1e4)
        dataset = SimpleDataset(texts, self.tokenizer, max_length=1024)
        # dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
        sampler = DistributedSampler(dataset) if self.data_parallel else None
        dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

        # 训练循环
        model.train()
        total_epoches = 100
        for epoch in range(total_epoches):
            if self.data_parallel:
                sampler.set_epoch(epoch)
            with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epoches}", unit="batch", leave=False) as tepoch:
                for batch in dataloader:
                    optimizer.zero_grad()
                    input_ids, attention_mask = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
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
    trainer = ParallTrainer(model, tokenizer, gradient_checkpointing=True, mixed_precision=True, dp=2, tp=4)

if __name__ == "__main__":
    main()
