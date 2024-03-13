export PYTHONPATH="../:$PYTHONPATH"
# # test GPU distribution bandwidth
# env OMP_NUM_THREADS=2 torchrun --nproc_per_node 4 --master-port=12345 tests/test_dist.py

# # test CPU-GPU data transfer bandwidth
# env OMP_NUM_THREADS=2 torchrun --nproc_per_node 4 --master-port=12345 tests/test_move.py

# # test tp, pp, dp groups
# env OMP_NUM_THREADS=2 torchrun --nproc_per_node 4 --master-port=12345 tests/test_parallel_groups.py

# test tensor parallel forward and backward results
env OMP_NUM_THREADS=2 torchrun --nproc_per_node 4 --master-port=12345  tests/test_tp_forward_backward_results.py --tensor_parallel ${1:-$CUDA_VISIBLE_DEVICES}

