# src/config.py - Optimized for 3B tokens with 500M daily increments

import os

# === RANDOM SEED FOR REPRODUCIBILITY ===
SEED = 42

# === TRAINING TARGETS ===
TOTAL_TOKENS_TARGET = 3_000_000_000  # 3B tokens total
DAILY_TOKEN_TARGET = 500_000_000     # 500M tokens per day
TRAINING_DAYS = 6                     # 3B / 500M = 6 days

# === TRAINING BATCH AND SEQUENCE SETTINGS ===
BATCH_SIZE = 192                        # Per GPU batch size
BLOCK_SIZE = 2048                     # Sequence length in tokens
ACCUMULATION_STEPS = 1                # Gradient accumulation steps

# === CALCULATED TRAINING STEPS ===
TOKENS_PER_STEP = BATCH_SIZE * BLOCK_SIZE * 2 * ACCUMULATION_STEPS  # 131,072
DAILY_TRAINING_STEPS = DAILY_TOKEN_TARGET // TOKENS_PER_STEP        # 3,815 steps
TOTAL_TRAINING_STEPS = TOTAL_TOKENS_TARGET // TOKENS_PER_STEP       # 22,888 steps

# === CURRICULUM SPLIT (60% easy, 40% hard) ===
EASY_TOKEN_TARGET = int(TOTAL_TOKENS_TARGET * 0.6)  # 1.8B tokens
HARD_TOKEN_TARGET = int(TOTAL_TOKENS_TARGET * 0.4)  # 1.2B tokens

# === LEARNING RATE SCHEDULE ===
LEARNING_RATE = 3e-4                                       # Conservative start
WARMUP_STEPS = int(0.03 * TOTAL_TRAINING_STEPS)         # ~687 steps
MIN_LEARNING_RATE = 3e-5                                 # For cosine decay

# === CHECKPOINTING (every 100M tokens) ===
CHECKPOINT_INTERVAL = 100_000_000 // TOKENS_PER_STEP     # 763 steps
CHECKPOINTS_PER_DAY = DAILY_TOKEN_TARGET // 100_000_000  # 5 checkpoints

# === LOGGING AND VALIDATION ===
LOG_INTERVAL = 100               # Log every 100 steps
VAL_INTERVAL = 500               # Validate every 500 steps
VAL_STEPS = 100                  # Validation batches

# === DATA LOADER ===
NUM_WORKERS = 1                  # For streaming dataset
PREFETCH_FACTOR = 2              # Prefetch batches

# === MODEL HYPERPARAMETERS ===
MODEL_DIM = 640
NUM_LAYERS = 24
NUM_HEADS = 10
NUM_KV_HEADS = 4                 # Grouped query attention
HIDDEN_DIM = 2560
VOCAB_SIZE = 32000               # Will be overridden by tokenizer

# === TRAINING STABILITY ===
LOSS_SPIKE_FACTOR = 4.0          # Loss spike detection
GRADIENT_CLIP_VAL = 1.0          # Gradient clipping
USE_BF16 = True                  # BFloat16 for RTX 5090

# === CHECKPOINT MANAGEMENT ===
MAX_CHECKPOINTS = 3              # Keep only last 3 per day
CHECKPOINT_DIR = "checkpoints"
RUNS_DIR = "runs/experiment_3b"
PROGRESS_FILE = "training_progress.json"

# === RESUME SETTINGS ===
RESUME_PATH = ""                 # Will be set by command line
AUTO_FIND_RESUME = True          # Automatically find latest checkpoint

# === CREATE DIRECTORIES ===
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# === DAILY SCHEDULE ===
print("=" * 60)
print("🚀 3B TOKEN TRAINING CONFIGURATION")
print("=" * 60)
print(f"Total tokens target: {TOTAL_TOKENS_TARGET:,} ({TOTAL_TOKENS_TARGET/1e9:.1f}B)")
print(f"Daily token target: {DAILY_TOKEN_TARGET:,} ({DAILY_TOKEN_TARGET/1e9:.1f}B)")
print(f"Training days: {TRAINING_DAYS}")
print(f"Total steps: {TOTAL_TRAINING_STEPS:,}")
print(f"Steps per day: {DAILY_TRAINING_STEPS:,}")
print(f"Checkpoints per day: {CHECKPOINTS_PER_DAY}")
print(f"Time per 100M tokens: ~{(100_000_000/TOKENS_PER_STEP) * ACCUMULATION_STEPS / 60:.1f} minutes")
print(f"Estimated time per day: ~{DAILY_TRAINING_STEPS * ACCUMULATION_STEPS / 60:.1f} minutes")
print("-" * 60)
print("DAILY SCHEDULE:")
for day in range(1, TRAINING_DAYS + 1):
    start_tokens = (day - 1) * DAILY_TOKEN_TARGET
    end_tokens = day * DAILY_TOKEN_TARGET
    print(f"Day {day}: {start_tokens/1e9:.1f}B → {end_tokens/1e9:.1f}B tokens")
print("=" * 60)