import torch
import torch.distributed as dist
from typing import Optional
from llama_parallel_finetune.utils import GlobalMemoryBuffer

# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_TENSOR_MODEL_PARALLEL_RANK = None

# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# Embedding group.
_EMBEDDING_GROUP = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# Position embedding group.
_POSITION_EMBEDDING_GROUP = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP

def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()

def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def get_tensor_model_parallel_group():
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, "_TENSOR_MODEL_PARALLEL_GROUP not set yet"
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return dist.get_world_size(group=get_tensor_model_parallel_group())

def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _TENSOR_MODEL_PARALLEL_RANK
    if _TENSOR_MODEL_PARALLEL_RANK is not None:
        return _TENSOR_MODEL_PARALLEL_RANK
    return dist.get_rank(group=get_tensor_model_parallel_group())

def get_data_parallel_group():
    assert _DATA_PARALLEL_GROUP is not None, "_DATA_PARALLEL_GROUP not set yet"
    return _DATA_PARALLEL_GROUP

def get_data_parallel_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(
            group=get_data_parallel_group()
        )
    else:
        return 0
    
def get_data_parallel_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(
            group=get_data_parallel_group()
        )
    else:
        return 0


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    context_parallel_size: int = 1,
):
    
    assert dist.is_initialized(), "Please Initialize torch ddp first"
    world_size: int = dist.get_world_size()
    assert world_size % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size) == 0, \
    f"world size {world_size} not divisible by tensor_model_parallel_size {tensor_model_parallel_size} \
     * pipline_model_parallel_size {pipeline_model_parallel_size} * context_parallel_size {context_parallel_size} \
    "
    data_parallel_size: int = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    rank = dist.get_rank()

    # build data parallel groups
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group already initialized"
    all_data_parallel_group_ranks_with_cp = []
    # 一个data parallel group中需要通信的ranks，都包含相同的其他并行部分，即一个data parallel group组中的rank必须在其他parallel groups的同一个inner rank
    #  0 1 | 2 3
    #  4 5 | 6 7
    #  8 9 | A B
    #  C D | E F
    # 如果pp size=4，即每一行是同一个层, tp size=2, 即一列隔一列是同一块tensor, 所以distinct的组就是2*4=8组，每组内2个gpu上tenser一样，需要data通信
    # data paralle组中，必须是同一个layer层
    # different DATA in a group
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i+1) * num_pipeline_model_parallel_groups
        # 且同一个context，同一个tenser块，相同context，即把context parallel看作tensor parallel
        for j in range(context_parallel_size * tensor_model_parallel_size):
            ranks = range(
                start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
            )
            group = dist.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GLOBAL_RANKS = ranks
        # 允许不同context，因此num_groups变少了，但是一个ranks group中要通信的ranks变多了
        # 因此ranks_with_cp组之间，相同idx的rank具有相同layer和tensor，但是不一定有相同的context
        # different DATA or CONTEXT in a group
        for j in range(tensor_model_parallel_size):
            ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
            group_with_cp = dist.new_group(ranks_with_cp)
            if rank in ranks_with_cp:
                _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp


    # build model parallel groups
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group already initialized"
    # different TENSOR or LAYER in a group
    for i in range(data_parallel_size * context_parallel_size):
        ranks = [
            data_parallel_group_ranks_with_cp[i] 
            for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
        ]
        group = dist.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group
    
    # build tensor model parallel groups
    # different TENSOR in a group
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, "tensor model parallel group already initialized"
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i+1) * tensor_model_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group


    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized' 
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = dist.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # setup embedding group(to exchange gradient between first and last stages)
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            # 把分割encoder和decoder的rank存进去
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank],
                        ranks[-1],
                    ]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = dist.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = dist.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = rank
    
    # Build the tensor + data parallel groups
    # TODO
    # Build the tensor + expert parallel groups
    # TODO
            
    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()
    

