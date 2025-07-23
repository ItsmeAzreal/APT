#!/bin/bash
# launch_training.sh - Fixed launch script for A100 training

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LLM Pre-training Launch Script ===${NC}"
echo -e "${BLUE}Target: 3B tokens with curriculum learning${NC}"
echo ""

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo -e "${YELLOW}Detected ${NUM_GPUS} GPU(s)${NC}"

# Environment setup
export CUDA_VISIBLE_DEVICES=0  # Use first GPU for single GPU, or 0,1 for multi
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_ALLOW_TF32=1
export CUDNN_ALLOW_TF32=1
export TOKENIZERS_PARALLELISM=false

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
    LATEST_CKPT=$(ls -t checkpoints/ckpt_step*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo -e "${GREEN}Found checkpoint: $LATEST_CKPT${NC}"
        read -p "Resume from this checkpoint? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            RESUME_CKPT="--resume $LATEST_CKPT"
        fi
    fi
fi

# Set batch size based on GPU memory
BATCH_SIZE=64  # Conservative for A100 40GB

# Determine training command based on GPU count
if [ $NUM_GPUS -eq 1 ]; then
    echo -e "${BLUE}Starting single-GPU training on A100...${NC}"
    CMD="python pretrain.py --batch_size $BATCH_SIZE $RESUME_CKPT"
else
    echo -e "${BLUE}Starting multi-GPU training on $NUM_GPUS GPUs...${NC}"
    CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=12355 pretrain.py --batch_size $BATCH_SIZE $RESUME_CKPT"
fi

# Log file with timestamp
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# Function to monitor GPU
monitor_gpu() {
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits >> logs/gpu_monitor.log
        sleep 60
    done
}

# Start GPU monitoring in background
monitor_gpu &
MONITOR_PID=$!

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    kill $MONITOR_PID 2>/dev/null
    echo -e "${GREEN}Training stopped${NC}"
}

trap cleanup EXIT

# Run training with error handling
echo -e "${GREEN}Starting training...${NC}"
echo "Command: $CMD"
echo "Logging to: $LOG_FILE"
echo ""

# Execute training
$CMD 2>&1 | tee $LOG_FILE

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n${GREEN}✅ Training completed successfully!${NC}"
else
    echo -e "\n${RED}❌ Training failed with error code ${PIPESTATUS[0]}${NC}"
    echo -e "${YELLOW}Check log file: $LOG_FILE${NC}"
    exit 1
fi