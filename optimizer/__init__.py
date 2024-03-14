import torch
import torch.distributed as dist
from llama_parallel_finetune.parallel_state import (
    get_model_parallel_group,
    get_data_parallel_group,

)

def get_param_groups(model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult):
    """Create parameter groups for optimizer.

    Creates parameter groups based on weight decay condition (regularized vs
    non regularized), learning rate scale condition (lr vs lr_mult * lr),
    and whether it is expert parameters. scale_lr_cond is used during finetuning
    where head of the network requires a scaled version of the base learning rate.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        no_weight_decay_cond (func): function to determine whether a parameter
            should not perform weight decay.
        scale_lr_cond (func): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
    """
    # map (wd_mult, lr_mult, is_expert_parallel) to params
    params_map = {
        (1.0, 1.0, False): [],
        (1.0, 1.0, True): [],
        (1.0, lr_mult, False): [],
        (1.0, lr_mult, True): [],
        (0.0, 1.0, False): [],
        (0.0, 1.0, True): [],
        (0.0, lr_mult, False): [],
        (0.0, lr_mult, True): [],
    }

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_mult, lr_mult = 1.0, 1.0
            elif not no_wd and scale_lr:
                wd_mult, lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:
                wd_mult, lr_mult = 0.0, 1.0
            else:
                wd_mult, lr_mult = 0.0, lr_mult

            params_map[(wd_mult, lr_mult, is_expert_parallel)].append(param)

    param_groups = []
    for (wd_mult, lr_mult, is_expert_parallel), params in params_map.items():
        if len(params) == 0:
            continue
        param_groups.append(
            {
                'params': params,
                'wd_mult': wd_mult,
                'lr_mult': lr_mult,
                'is_expert_parallel': is_expert_parallel,
            }
        )

    return param_groups

def get_optimizer_based_on_param_groups(
    config,
    param_groups,
    per_model_grad_buffers=None,
    data_parallel_group=None,
    data_parallel_group_gloo=None,
    data_parallel_group_idx=None,
):
    """Get optimizer based on parameter groups.

    For distributed optimizer, we need the parameter gradients to be stored in a
    contiguous grad_buffer.

    Args:
        param_groups (list): list of parameter groups.
        per_model_grad_buffers (list, optional): list of gradient buffers for
            distributed optimizer. Defaults to None.
        data_parallel_group (ProcessGroup, optional): data parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (ProcessGroup, optional): data parallel
            group-gloo for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data parallel
            group index for distributed optimizer. Defaults to None.
    """
    if config.optimizer == 'adam':
        optimizer = Adam(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )

        def init_state_fn(opt):
            for group in opt.param_groups:
                for p in group['params']:
                    if len(opt.state[p]) == 0:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)


def get_optimizer(config, model_chunks, no_weight_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
    # Collect param groups.
    param_groups = get_param_groups(model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult)

    # Collect grad buffers for distributed optimizer.
    per_model_grad_buffers = {}
    per_model_ep_grad_buffers = {}
    for model_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, 'grad_buffers'):
            per_model_grad_buffers[model_idx] = model_chunk.grad_buffers
            per_model_ep_grad_buffers[model_idx] = model_chunk.expert_parallel_grad_buffers

    # Split param groups into dense and moe.
    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))
    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))

    # Create optimizers.
    model_parallel_rank = dist.get_rank(get_model_parallel_group())
    optimizers = [
        get_optimizer_based_on_param_groups(
            config,
            param_groups=dense_param_groups,
            per_model_grad_buffers=per_model_grad_buffers,
            data_parallel_group=get_data_parallel_group(with_context_parallel=True),
            data_parallel_group_idx=model_parallel_rank,
        )
    ]
    