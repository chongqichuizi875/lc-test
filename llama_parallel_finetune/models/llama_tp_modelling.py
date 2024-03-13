from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoModel, AutoConfig, LlamaConfig, LlamaModel
import copy
from llama_parallel_finetune.tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear, VocabParallelEmbedding
import torch.nn as nn
from llama_parallel_finetune.parallel_state import get_tensor_model_parallel_group

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
        # self._create_hook_for_modules(reduced_model)
        setattr(reduced_model, 'lm_head', ColumnParallelLinear(reduced_model.lm_head))
        for name, module in reduced_model.named_modules():
            if name.endswith('model'):
                setattr(module, 'embed_tokens', VocabParallelEmbedding(module.embed_tokens))
            if name.endswith("mlp"):
                setattr(module, 'gate_proj', RowParallelLinear(module.gate_proj))
                setattr(module, 'up_proj', RowParallelLinear(module.up_proj))
                setattr(module, 'down_proj', ColumnParallelLinear(module.down_proj))
            if name.endswith("self_attn"):
                setattr(module, 'q_proj', ColumnParallelLinear(module.q_proj))
                setattr(module, 'k_proj', ColumnParallelLinear(module.k_proj))
                setattr(module, 'v_proj', ColumnParallelLinear(module.v_proj))
                setattr(module, 'o_proj', RowParallelLinear(module.o_proj))

        return reduced_model