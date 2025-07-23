#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_ALLOW_TF32=1
export CUDA_ALLOW_TF32=1
export NCCL_DEBUG=INFO

mkdir -p checkpoints
mkdir -p runs/experiment_2b

echo "Clearing PyTorch cache..."
python -c "import torch; torch.cuda.empty_cache()"

echo "Starting training on single A100..."
python pretrain_ddp.py --batch_size 16 --val
