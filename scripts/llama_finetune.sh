export PYTHONPATH="../:$PYTHONPATH"
device_count=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1 + 1}')
env OMP_NUM_THREADS=2 torchrun --nproc_per_node $device_count --master-port=${2:-12345} llama_parallel_finetune/llama_finetune.py --tensor_parallel ${1:-$device_count}

