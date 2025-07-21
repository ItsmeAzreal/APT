#!/bin/bash
# launch_training.sh - Optimized launch script for 2x RTX 5090

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable TF32 for better performance on RTX 5090
export TORCH_ALLOW_TF32=1
export CUDA_ALLOW_TF32=1

# Set NCCL options for better multi-GPU communication
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1

# Create necessary directories
mkdir -p checkpoints
mkdir -p runs/experiment_2b

# Clear cache before starting
echo "Clearing PyTorch cache..."
python -c "import torch; torch.cuda.empty_cache()"

# Launch training with torchrun
echo "Starting distributed training on 2x RTX 5090..."
torchrun --nproc_per_node=2 \
         --master_port=12355 \
         --master_addr=localhost \
         pretrain_ddp.py \
         --world_size=2

# Optional: Monitor GPU usage in another terminal with:
# watch -n 1 nvidia-smi