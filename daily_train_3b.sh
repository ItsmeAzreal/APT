#!/bin/bash
# daily_train_3b.sh - Fixed daily training script for 3B tokens on RunPod

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get current day from training progress
if [ -f "checkpoints/training_progress.json" ]; then
    CURRENT_TOKENS=$(python3 -c "import json; print(json.load(open('checkpoints/training_progress.json'))['tokens_seen'])")
    DAY=$((($CURRENT_TOKENS / 500000000) + 1))
else
    DAY=${1:-1}
fi

echo -e "${GREEN}=== 3B Token Training - Day $DAY/6 ===${NC}"
echo -e "${CYAN}Target: 500M tokens today ($(((DAY-1)*500))M → $((DAY*500))M total)${NC}"
echo ""

# Environment setup
export CUDA_VISIBLE_DEVICES=0  # Single A100
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_ALLOW_TF32=1
export CUDNN_ALLOW_TF32=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO  # For debugging distributed training

# Load environment variables
if [ -f ".env" ]; then
    source .env
else
    echo -e "${RED}ERROR: .env file not found!${NC}"
    exit 1
fi

# Find latest checkpoint
LATEST_CKPT=""
RESUME_ARG=""
if [ -d "checkpoints" ]; then
    LATEST_CKPT=$(ls -t checkpoints/ckpt_step*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_ARG="--resume $LATEST_CKPT"
        echo -e "${YELLOW}Resuming from checkpoint: $LATEST_CKPT${NC}"
    fi
fi

# Clear cache before starting
echo -e "${YELLOW}Clearing GPU cache...${NC}"
python3 -c "import torch; torch.cuda.empty_cache()"

# Training command - using the fixed pretrain.py
CMD="python pretrain.py --batch_size 64 $RESUME_ARG"

# Start time tracking
START_TIME=$(date +%s)
START_DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo -e "${GREEN}Starting training at $START_DATE...${NC}"
echo -e "${BLUE}Command: $CMD${NC}"
echo ""

# Create log file
LOG_DIR="logs/day_${DAY}"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

# Function to monitor training
monitor_training() {
    while true; do
        if [ -f "checkpoints/training_progress.json" ]; then
            python3 << EOF
import json
import os
from datetime import datetime

with open('checkpoints/training_progress.json', 'r') as f:
    progress = json.load(f)

tokens = progress['tokens_seen']
day_progress = ((tokens % 500_000_000) / 500_000_000 * 100) if tokens % 500_000_000 != 0 else 100
total_progress = (tokens / 3_000_000_000) * 100

print(f"\n{'='*60}")
print(f"📊 TRAINING MONITOR - {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*60}")
print(f"Day {int(tokens // 500_000_000) + 1}/6 Progress: {day_progress:.1f}%")
print(f"Total Progress: {total_progress:.1f}% ({tokens:,} / 3,000,000,000)")
print(f"Current Step: {progress.get('step', 'Unknown')}")
print(f"Loss: {progress.get('loss', 'N/A'):.4f}")
print(f"Speed: {progress.get('tokens_per_sec', 0):.0f} tok/s")
print(f"{'='*60}")
EOF
        fi
        sleep 300  # Update every 5 minutes
    done
}

# Start monitoring in background
monitor_training &
MONITOR_PID=$!

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Stopping monitor...${NC}"
    kill $MONITOR_PID 2>/dev/null
}
trap cleanup EXIT

# Run training
$CMD 2>&1 | tee $LOG_FILE
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# End time tracking
END_TIME=$(date +%s)
DURATION=$(( ($END_TIME - $START_TIME) / 60 ))

# Generate daily report
echo -e "\n${GREEN}Generating daily progress report...${NC}"

python3 << EOF
import json
import glob
import os
from datetime import datetime

# Check if training was successful
if $TRAINING_EXIT_CODE != 0:
    print("❌ Training failed with exit code $TRAINING_EXIT_CODE")
    exit(1)

# Load progress
try:
    with open('checkpoints/training_progress.json', 'r') as f:
        progress = json.load(f)
except FileNotFoundError:
    print("❌ No progress file found!")
    exit(1)

# Find all checkpoints from today
all_ckpts = sorted(glob.glob('checkpoints/ckpt_step*.pt'))
today_ckpts = []
for ckpt in all_ckpts:
    # Check modification time
    mtime = os.path.getmtime(ckpt)
    if (datetime.now().timestamp() - mtime) < 86400:  # Within 24 hours
        today_ckpts.append(ckpt)

# Calculate statistics
tokens_seen = progress['tokens_seen']
current_day = (tokens_seen // 500_000_000) + 1
progress_pct = (tokens_seen / 3_000_000_000) * 100
tokens_today = tokens_seen - ((current_day - 1) * 500_000_000)

# Create report
report = {
    'day': current_day,
    'date': datetime.now().isoformat(),
    'total_tokens': tokens_seen,
    'tokens_today': tokens_today,
    'progress_percent': progress_pct,
    'duration_minutes': $DURATION,
    'checkpoints_created': len(today_ckpts),
    'latest_checkpoint': all_ckpts[-1] if all_ckpts else None,
    'status': 'completed' if tokens_today >= 500_000_000 else 'in_progress'
}

# Save report
report_file = f'logs/day_{current_day}/daily_report.json'
with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)

# Print summary
print(f"\n{'='*60}")
print(f"✅ DAY {current_day} TRAINING REPORT")
print(f"{'='*60}")
print(f"📊 Tokens processed today: {tokens_today:,}")
print(f"📈 Total tokens: {tokens_seen:,} ({progress_pct:.1f}% of 3B)")
print(f"⏱️  Duration: {$DURATION} minutes ({$DURATION/60:.1f} hours)")
print(f"💾 Checkpoints created: {len(today_ckpts)}")
print(f"📁 Latest checkpoint: {os.path.basename(all_ckpts[-1]) if all_ckpts else 'None'}")
print(f"{'='*60}")

if progress_pct >= 100:
    print(f"\n🎉 TRAINING COMPLETE! All 3B tokens processed!")
    print(f"📥 Download final model: {all_ckpts[-1]}")
else:
    next_day = current_day + 1
    remaining_tokens = 3_000_000_000 - tokens_seen
    print(f"\n📋 Next steps:")
    print(f"   1. Download checkpoint: {os.path.basename(all_ckpts[-1])}")
    print(f"   2. Resume tomorrow for Day {next_day}")
    print(f"   3. Remaining tokens: {remaining_tokens:,}")

# Save completion marker
if tokens_today >= 500_000_000:
    with open(f'logs/day_{current_day}/completed.txt', 'w') as f:
        f.write(f"Day {current_day} completed at {datetime.now()}")
EOF

# Create checkpoint archive for download
echo -e "\n${GREEN}Creating checkpoint archive...${NC}"

# Get the latest checkpoint name
if [ -n "$LATEST_CKPT" ]; then
    ARCHIVE_NAME="checkpoint_day${DAY}_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    # Create archive with checkpoint and progress files
    tar -czf $ARCHIVE_NAME \
        $(ls -t checkpoints/ckpt_step*.pt | head -1) \
        checkpoints/training_progress.json \
        logs/day_${DAY}/daily_report.json \
        2>/dev/null || true
    
    echo -e "${GREEN}✅ Archive created: $ARCHIVE_NAME${NC}"
    echo -e "${YELLOW}📥 Download this file before shutting down RunPod!${NC}"
    
    # Show file size
    SIZE=$(ls -lh $ARCHIVE_NAME | awk '{print $5}')
    echo -e "${BLUE}Archive size: $SIZE${NC}"
fi

# Cost estimate (RunPod A100 pricing)
HOURS=$(echo "scale=2; $DURATION / 60" | bc)
COST=$(echo "scale=2; $HOURS * 1.89" | bc)  # ~$1.89/hour for A100
echo -e "\n${CYAN}💰 Estimated cost for today: ~\$$COST USD${NC}"

# Final status
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}✅ Daily training completed successfully!${NC}"
else
    echo -e "\n${RED}❌ Training failed! Check logs at: $LOG_FILE${NC}"
    exit 1
fi