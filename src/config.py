# src/config.py - Fixed configuration with validation and proper settings

import os
import torch

# === RANDOM SEED FOR REPRODUCIBILITY ===
SEED = 42

# === HARDWARE DETECTION ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

# === TRAINING TARGETS ===
TOTAL_TOKENS_TARGET = 3_000_000_000  # 3B tokens total
DAILY_TOKEN_TARGET = 500_000_000     # 500M tokens per day
TRAINING_DAYS = 6                     # 3B / 500M = 6 days

# === TRAINING BATCH AND SEQUENCE SETTINGS ===
# Adjusted for A100 40GB memory constraints
BATCH_SIZE = 32                       # Per GPU batch size
BLOCK_SIZE = 1024                     # Sequence length in tokens
ACCUMULATION_STEPS = 2                # Gradient accumulation steps
VAL_BATCH_SIZE = 32                   # Validation batch size

# === CALCULATED TRAINING STEPS ===
TOKENS_PER_STEP = BATCH_SIZE * BLOCK_SIZE * NUM_GPUS * ACCUMULATION_STEPS
DAILY_TRAINING_STEPS = DAILY_TOKEN_TARGET // TOKENS_PER_STEP
TOTAL_TRAINING_STEPS = TOTAL_TOKENS_TARGET // TOKENS_PER_STEP

# === CURRICULUM SPLIT (60% easy, 40% hard) ===
EASY_TOKEN_TARGET = int(TOTAL_TOKENS_TARGET * 0.5)  # 1.8B tokens
MEDIUM_TOKEN_TARGET = int(TOTAL_TOKENS_TARGET * 0.25)  # 1.2B tokens
HARD_TOKEN_TARGET = TOTAL_TOKENS_TARGET - EASY_TOKEN_TARGET - MEDIUM_TOKEN_TARGET  # 750,000,000 tokens

# === LEARNING RATE SCHEDULE ===
LEARNING_RATE = 3e-4                              # Base learning rate
LR = LEARNING_RATE                                # Alias for compatibility
WARMUP_STEPS = int(0.03 * TOTAL_TRAINING_STEPS)  # 3% warmup
MIN_LEARNING_RATE = 3e-5                          # For cosine decay
WEIGHT_DECAY = 0.1                            # AdamW weight decay
EMBEDDING_DROPOUT = 0.1  # Dropout rate for embedding layer
DROPOUT = 0.1           # General dropout rate for attention/FFN
ACTIVATION = "swish" # Activation function type (options: "relu", "gelu", "silu")
USE_ROPE_SCALING = True         # Enable RoPE scaling
ROPE_SCALING_FACTOR = 2.0       # Scaling factor for position embeddings

# === CHECKPOINTING (every 100M tokens) ===
CHECKPOINT_INTERVAL = 100_000_000 // TOKENS_PER_STEP
CHECKPOINTS_PER_DAY = DAILY_TOKEN_TARGET // 100_000_000

# === LOGGING AND VALIDATION ===
LOG_INTERVAL = 50                # Log every 50 steps
VAL_INTERVAL = 500               # Validate every 500 steps
VAL_STEPS = 50                   # Validation batches

# === DATA LOADER ===
NUM_WORKERS = 2                  # DataLoader workers
PREFETCH_FACTOR = 2              # Prefetch batches

# === MODEL HYPERPARAMETERS ===
MODEL_DIM = 640
NUM_LAYERS = 24
NUM_HEADS = 10
NUM_KV_HEADS = 2                 # Grouped query attention
HIDDEN_DIM = 2560
VOCAB_SIZE = 32000               # LLaMA tokenizer vocab size

# === TRAINING STABILITY ===
LOSS_SPIKE_FACTOR = 4.0          # Loss spike detection
GRADIENT_CLIP_VAL = 1.0          # Gradient clipping
USE_BF16 = True                  # BFloat16 for A100
USE_FLASH_ATTN = True            # Use flash attention if available

# === MEMORY OPTIMIZATION ===
GRADIENT_CHECKPOINTING = True    # Enable for layers > 8
MAX_SPLIT_SIZE_MB = 512          # For CUDA memory allocation

# === CHECKPOINT MANAGEMENT ===
MAX_CHECKPOINTS = 5              # Keep last 5 checkpoints
CHECKPOINT_DIR = "checkpoints"
RUNS_DIR = "runs/experiment_3b"
PROGRESS_FILE = "training_progress.json"
GRADIENT_CHECKPOINTING = True    # Enable gradient checkpointing
CHECKPOINT_LAYERS = list(range(0, NUM_LAYERS, 2))  # Every 2nd layer
CHECKPOINT_LAYERS = [            # List of layer indices to checkpoint
    2, 5, 8, 11, 14, 17, 20, 23 # Example: checkpoint every 3rd layer in a 24-layer model
]  


# === RESUME SETTINGS ===
RESUME_PATH = ""                 # Will be set by command line
AUTO_FIND_RESUME = True          # Automatically find latest checkpoint

# === PATHS ===
HF_CACHE_DIR = os.getenv("HF_HOME", "/workspace/hf_cache")
TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"  # Using Llama-2 tokenizer

# === CREATE DIRECTORIES ===
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === ENVIRONMENT SETUP ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{MAX_SPLIT_SIZE_MB}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# === CONFIG VALIDATION ===
def validate_config():
    """Validate configuration consistency"""
    errors = []
    
    # Basic validation
    if BATCH_SIZE <= 0:
        errors.append("Batch size must be positive")
    if BLOCK_SIZE <= 0:
        errors.append("Block size must be positive")
    if LEARNING_RATE <= 0:
        errors.append("Learning rate must be positive")
    if TOTAL_TOKENS_TARGET < DAILY_TOKEN_TARGET:
        errors.append("Total tokens must be >= daily target")
    
    # Memory estimation (rough)
    param_memory_gb = (MODEL_DIM * MODEL_DIM * NUM_LAYERS * 4) / 1e9
    activation_memory_gb = (BATCH_SIZE * BLOCK_SIZE * MODEL_DIM * NUM_LAYERS * 4) / 1e9
    total_memory_gb = param_memory_gb + activation_memory_gb
    
    if total_memory_gb > 35:  # A100 has 40GB, leave some headroom
        errors.append(f"Estimated memory usage ({total_memory_gb:.1f}GB) may exceed A100 capacity")
    
    # Step calculation validation
    if TOTAL_TRAINING_STEPS <= 0:
        errors.append(f"Invalid step calculation: {TOTAL_TRAINING_STEPS}")
    
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Configuration validation failed")
    
    return True

# === DAILY SCHEDULE DISPLAY ===
def display_config():
    """Display training configuration"""
    print("=" * 60)
    print("🚀 3B TOKEN TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Device: {DEVICE} ({NUM_GPUS} GPU(s) detected)")
    print(f"Total tokens target: {TOTAL_TOKENS_TARGET:,} ({TOTAL_TOKENS_TARGET/1e9:.1f}B)")
    print(f"Daily token target: {DAILY_TOKEN_TARGET:,} ({DAILY_TOKEN_TARGET/1e9:.1f}B)")
    print(f"Training days: {TRAINING_DAYS}")
    print(f"Total steps: {TOTAL_TRAINING_STEPS:,}")
    print(f"Steps per day: {DAILY_TRAINING_STEPS:,}")
    print(f"Tokens per step: {TOKENS_PER_STEP:,}")
    print(f"Batch size per GPU: {BATCH_SIZE}")
    print(f"Gradient accumulation: {ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS * NUM_GPUS}")
    print(f"Checkpoints per day: {CHECKPOINTS_PER_DAY}")
    print(f"Time per 100M tokens: ~{(100_000_000/TOKENS_PER_STEP) / 60:.1f} minutes")
    print(f"Estimated time per day: ~{DAILY_TRAINING_STEPS / 60:.1f} minutes")
    
    # Model size estimation
    param_count = (MODEL_DIM * VOCAB_SIZE + 
                   MODEL_DIM * MODEL_DIM * NUM_LAYERS * 4 + 
                   MODEL_DIM * HIDDEN_DIM * NUM_LAYERS * 2)
    print(f"Estimated parameters: ~{param_count/1e6:.0f}M")
    
    print("-" * 60)
    print("DAILY SCHEDULE:")
    for day in range(1, TRAINING_DAYS + 1):
        start_tokens = (day - 1) * DAILY_TOKEN_TARGET
        end_tokens = day * DAILY_TOKEN_TARGET
        print(f"Day {day}: {start_tokens/1e9:.1f}B → {end_tokens/1e9:.1f}B tokens")
    print("=" * 60)

# Run validation and display
if __name__ == "__main__":
    validate_config()
    display_config()