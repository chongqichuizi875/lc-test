import torch.distributed as dist
import os
import torch
from llama_parallel_finetune.utils import gpu_usage
import copy
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler, custom_fwd, custom_bwd
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoModel, AutoConfig, LlamaConfig, LlamaModel
import torch.nn as nn
import torch.nn.functional as F
from llama_parallel_finetune.parallel_state import initialize_model_parallel, get_data_parallel_group, get_tensor_model_parallel_group
from llama_parallel_finetune.tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear, VocabParallelEmbedding
from llama_parallel_finetune.models.llama_tp_modelling import ParallelLlama
class ParallelTrainer():
    def __init__(self, 
                 model, 
                 tokenizer, 
                 gradient_checkpointing=False, 
                 mixed_precision=False, 
                 tp=4,
                 optimizer=None,
                 scheduler=None,
                 dataset=None,
                 batch_size=16,
                 epoches=100,
                 lr=5e-5,
                 ):
        self.epoches = epoches
        self.lr = lr
        self.model = model
        self.tokenizer = tokenizer
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.tp = tp

        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        initialize_model_parallel(tensor_model_parallel_size=self.tp)    

        ori_params = list(self.model.parameters())
        ori_model_memory = sum([param.nelement() * param.element_size() for param in ori_params])
        self.model = ParallelLlama(self.model, tp=self.tp, world_size=world_size).get_model()
        params = list(self.model.parameters())
        model_memory = sum([param.nelement() * param.element_size() for param in params])
        if rank == 0:
            print(f"Original size: {ori_model_memory / 1024**2} MB, TP size: {model_memory / 1024**2} MB")

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(f"cuda:{local_rank}")

        self.model = DDP(self.model,
                    process_group=get_data_parallel_group())
        
        self.dataloader, self.sampler = self.setup_dataloader(dataset, batch_size)
        
        self.optimizer, self.scheduler = self.setup_optimizer(self.model, optimizer, scheduler)

        if self.mixed_precision:
            self.scaler = GradScaler(enabled=self.mixed_precision)

        
           
        # if rank == 0:
        #     for name, module in test_model.named_modules():
        #         params = list(module.parameters())
        #         module_memory = sum([param.nelement() * param.element_size() for param in params])
        #         print(f'Module name: {name}, Memory: {module_memory / 1024**2} MB')

    def setup_dataloader(self, dataset, batch_size):
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size() // self.tp, rank=dist.get_rank(group=get_data_parallel_group()))
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        return dataloader, sampler
    
    def setup_optimizer(self, model, optimizer, scheduler):
        if not optimizer or optimizer == "default":
            optimizer = AdamW(model.parameters(), lr=self.lr)
            num_training_steps = len(self.dataloader) * self.epoches
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
            return optimizer, scheduler
            

    def train(self):    
        self.model.train()
        total_epoches = 100
        for epoch in range(total_epoches):
            self.sampler.set_epoch(epoch)
            with tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{total_epoches}", unit="batch", leave=False) as tepoch:
                for batch in self.dataloader:
                    self.optimizer.zero_grad()
                    input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
                    input_ids = input_ids.to(f"cuda:{dist.get_rank()}")
                    attention_mask = attention_mask.to(f"cuda:{dist.get_rank()}")
                    with autocast(enabled=self.mixed_precision):
                        outputs = self.model(input_ids, attention_mask, labels=input_ids)
                        loss = outputs.loss

                    if self.mixed_precision:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    self.scheduler.step()
                    tepoch.set_postfix(loss="%.2f" % loss)
                    tepoch.update(1)



    
