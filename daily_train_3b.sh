#!/bin/bash
# daily_train_3b.sh - Daily training script for 3B tokens on RunPod

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get current day from argument or training progress
if [ -f "training_progress.json" ]; then
    CURRENT_TOKENS=$(python -c "import json; print(json.load(open('training_progress.json'))['total_tokens'])")
    DAY=$((($CURRENT_TOKENS / 500000000) + 1))
else
    DAY=${1:-1}
fi

echo -e "${GREEN}=== 3B Token Training - Day $DAY/6 ===${NC}"
echo -e "Target: 500M tokens today ($(((DAY-1)*500))M → $((DAY*500))M total)"
echo ""

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_ALLOW_TF32=1
export CUDA_ALLOW_TF32=1

# Find latest checkpoint
LATEST_CKPT=""
if [ -d "checkpoints" ]; then
    LATEST_CKPT=$(ls -t checkpoints/ckpt_step*.pt 2>/dev/null | head -1)
fi

# Training command
if [ -z "$LATEST_CKPT" ]; then
    echo -e "${YELLOW}Starting fresh training...${NC}"
    CMD="torchrun --nproc_per_node=2 --master_port=12355 pretrain_ddp.py"
else
    echo -e "${YELLOW}Resuming from checkpoint: $LATEST_CKPT${NC}"
    CMD="torchrun --nproc_per_node=2 --master_port=12355 pretrain_ddp.py --resume $LATEST_CKPT"
fi

# Start time tracking
START_TIME=$(date +%s)

# Run training
echo -e "${GREEN}Starting training...${NC}"
$CMD

# End time tracking
END_TIME=$(date +%s)
DURATION=$(( ($END_TIME - $START_TIME) / 60 ))

# Generate progress report
echo -e "\n${GREEN}Generating progress report...${NC}"
python3 << EOF
import json
import glob
import torch
import os
from datetime import datetime

# Find all checkpoints
ckpts = sorted(glob.glob('checkpoints/ckpt_step*.pt'))
if not ckpts:
    print("❌ No checkpoints found!")
    exit(1)

# Load latest checkpoint
latest_ckpt = torch.load(ckpts[-1], map_location='cpu')
tokens_seen = latest_ckpt['tokens_seen']
steps = latest_ckpt['step']

# Calculate progress
progress_pct = (tokens_seen / 3_000_000_000) * 100
tokens_today = tokens_seen % 500_000_000 if tokens_seen % 500_000_000 != 0 else 500_000_000

# Update progress file
progress = {
    'day': $DAY,
    'total_tokens': tokens_seen,
    'total_steps': steps,
    'progress_percent': progress_pct,
    'latest_checkpoint': ckpts[-1],
    'timestamp': datetime.now().isoformat(),
    'duration_minutes': $DURATION
}

with open('training_progress.json', 'w') as f:
    json.dump(progress, f, indent=2)

# Print summary
print(f"✅ Day $DAY Training Complete!")
print(f"📊 Total tokens: {tokens_seen:,} ({progress_pct:.1f}% of 3B)")
print(f"📈 Tokens today: {tokens_today:,}")
print(f"⏱️  Duration: {$DURATION} minutes")
print(f"💾 Latest checkpoint: {os.path.basename(ckpts[-1])}")
print(f"" )
print(f"📋 Next steps:")
if progress_pct >= 100:
    print(f"   🎉 TRAINING COMPLETE! Download final model.")
else:
    print(f"   1. Download checkpoint: {os.path.basename(ckpts[-1])}")
    print(f"   2. Resume tomorrow for Day {$DAY + 1}")
EOF

# Create checkpoint archive for download
echo -e "\n${GREEN}Creating checkpoint archive...${NC}"
ARCHIVE_NAME="checkpoints_day${DAY}_$(date +%Y%m%d_%H%M%S).tar.gz"

# Only include the latest checkpoint and progress file
LATEST_CKPT_NAME=$(basename $LATEST_CKPT)
tar -czf $ARCHIVE_NAME \
    checkpoints/$LATEST_CKPT_NAME \
    training_progress.json \
    runs/experiment_3b/ 2>/dev/null || true

echo -e "${GREEN}✅ Archive created: $ARCHIVE_NAME${NC}"
echo -e "${YELLOW}📥 Download this file before shutting down RunPod!${NC}"

# Show total cost estimate
HOURS=$(echo "scale=2; $DURATION / 60" | bc)
COST=$(echo "scale=2; $HOURS * 1.88" | bc)
echo -e "\n${GREEN}💰 Today's training cost: ~\$$COST USD${NC}"