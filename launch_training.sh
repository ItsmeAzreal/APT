#!/bin/bash
# launch_training.sh - Improved launch script for A100 training with monitoring

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LLM Pre-training Launch Script ===${NC}"
echo -e "${BLUE}Target: 3B tokens with curriculum learning${NC}"
echo ""

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo -e "${YELLOW}Detected ${NUM_GPUS} GPU(s)${NC}"

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Environment setup
export CUDA_VISIBLE_DEVICES=0  # Use first GPU for single GPU, or 0,1 for multi
export OMP_NUM_THREADS=8
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_ALLOW_TF32=1
export CUDNN_ALLOW_TF32=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=ERROR  # Reduce verbosity
# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,roundup_power2_divisions:16

# Performance optimizations  
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1

# PyTorch 2.0 compilation
export TORCH_COMPILE_MODE=max-autotune
export TORCHDYNAMO_DISABLE_GUARD_VIOLATIONS=1

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo "Please create .env file with HF_TOKEN=your_token_here"
    exit 1
fi

# Load environment variables
source .env

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}ERROR: HF_TOKEN not set in .env file!${NC}"
    exit 1
fi

# Create necessary directories
mkdir -p checkpoints
mkdir -p runs/experiment_3b
mkdir -p logs

# Clear PyTorch cache
echo -e "${YELLOW}Clearing PyTorch cache...${NC}"
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"

# Check for existing checkpoints
RESUME_CKPT=""
if [ -d "checkpoints" ]; then
    LATEST_CKPT=$(ls -t checkpoints/ckpt_step*.pt 2>/