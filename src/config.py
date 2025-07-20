# src/config.py
import os

# === RANDOM SEED FOR REPRODUCIBILITY ===
SEED = 42

# === TRAINING BATCH AND SEQUENCE SETTINGS ===
BATCH_SIZE = 16                   # Number of samples per GPU per batch
BLOCK_SIZE = 2048                # Sequence length in tokens

# === TRAINING STEPS (Set for your total desired tokens) ===
TOTAL_TRAINING_STEPS = 40_000    # You can adjust for your 2B token target

# === LEARNING RATE SCHEDULE ===
LEARNING_RATE = 3e-4
WARMUP_STEPS = int(0.03 * TOTAL_TRAINING_STEPS)   # 3% of total steps for warmup
# HOLD_STEPS and COOLDOWN_STEPS are not used unless you use a custom scheduler

# === GRADIENT ACCUMULATION (for effective larger batch size) ===
ACCUMULATION_STEPS = 1           # Gradients accumulated before optimizer step

# === LOGGING AND CHECKPOINTING ===
LOG_INTERVAL = 500               # Print/log train loss every N steps
VAL_INTERVAL = 2000              # Run validation every N steps (~32M tokens)
CHECKPOINT_INTERVAL = 3052        # Save checkpoint every 100M tokens (~6104 steps)
CHECKPOINT_DIR = "checkpoints"   # Directory to store checkpoints
RUNS_DIR = "runs/your_experiment" # TensorBoard/logs directory

# === DATA LOADER ===
NUM_WORKERS = 0                  # DataLoader workers (0 for streaming)
VAL_STEPS = 100                  # Validation steps (batches) per val run

# === MODEL HYPERPARAMETERS ===
MODEL_DIM = 640
NUM_LAYERS = 24
NUM_HEADS = 10
NUM_KV_HEADS = 4
HIDDEN_DIM = 2560
LOSS_SPIKE_FACTOR = 4.0          # Stop if loss spikes by this factor (for safety)

# === RESUME CHECKPOINT (leave blank for fresh run) ===
RESUME_PATH = ""                 # Path to checkpoint to resume, or "" for new

# === CREATE DIRECTORIES IF NEEDED ===
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# ---- END OF CONFIG ----
