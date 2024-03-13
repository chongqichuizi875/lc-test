# Tensor Parallel和Data Parallel混合实现
入口 llama_parallel_finetune/llama_finetune.py
测试脚本 scripts/test.sh

切分tensor: lm_head, vocabulary_embedding, self_attn, mlp

多卡TP+DP与单卡forward和backward的logits在rtol=1e-3, atol=1e-4下torch.allclose()
loss在默认rtol和atol下allclose

在公共数据集上收敛性已验证
