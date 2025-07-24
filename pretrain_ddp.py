# pretrain_ddp.py - Fixed unified training script with proper GPU utilization

"""
Fixed Pretraining Script for A100
- Ensures GPU utilization by proper data movement
- Supports both single GPU and multi-GPU training
- 3B token curriculum learning with 500M daily targets
- Robust error handling and checkpointing
- Memory-efficient for A100 40GB
"""

import os
import sys
import math
import json
import glob
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import LlamaTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

# Import project modules
from src.config import *
from src.model import AnameeModel, count_parameters, model_size_mb
from src.dataset import Curriculum2BTokenDataset, build_val_dataset
from src.validate import validate

# ========================= DDP Setup Functions =========================

def setup_ddp(rank: int, world_size: int, port: str = "12355"):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Set memory fraction to avoid OOM
    torch.cuda.set_per_process_memory_fraction(0.95, device=rank)
    
    logger.info(f"[Rank {rank}] DDP initialized successfully")

def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if current process is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0

# ========================= GPU Verification =========================

def verify_gpu_setup(model: nn.Module, device: torch.device):
    """Verify that model and data are on GPU"""
    logger.info("=" * 60)
    logger.info("GPU SETUP VERIFICATION")
    logger.info("=" * 60)
    
    # Check CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name()}")
        
        # Check memory
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        logger.info(f"GPU memory allocated: {allocated:.2f} GB")
        logger.info(f"GPU memory reserved: {reserved:.2f} GB")
    
    # Check model device
    model_device = next(model.parameters()).device
    logger.info(f"Model is on: {model_device}")
    
    # Test forward pass
    logger.info("Testing GPU forward pass...")
    test_input = torch.randint(0, 32000, (2, 128), device=device)
    logger.info(f"Test input device: {test_input.device}")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            test_output = model(test_input)
    
    logger.info("✓ GPU forward pass successful")
    logger.info("=" * 60)

# ========================= Progress Tracking =========================

class ProgressTracker:
    """Track and display training progress"""
    def __init__(self, total_tokens: int, daily_tokens: int, checkpoint_dir: str):
        self.total_tokens = total_tokens
        self.daily_tokens = daily_tokens
        self.checkpoint_dir = checkpoint_dir
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
    def get_progress_stats(self, tokens_seen: int, step: int, loss: float) -> Dict[str, Any]:
        """Calculate comprehensive progress statistics"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate progress
        total_progress = (tokens_seen / self.total_tokens) * 100
        current_day = (tokens_seen // self.daily_tokens) + 1
        daily_progress = ((tokens_seen % self.daily_tokens) / self.daily_tokens * 100) if tokens_seen % self.daily_tokens != 0 else 100
        
        # Calculate speed
        tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
        
        # Calculate ETA
        remaining_tokens = self.total_tokens - tokens_seen
        eta_seconds = remaining_tokens / tokens_per_sec if tokens_per_sec > 0 else 0
        
        return {
            'step': step,
            'tokens_seen': tokens_seen,
            'total_progress': total_progress,
            'current_day': current_day,
            'daily_progress': daily_progress,
            'loss': loss,
            'tokens_per_sec': tokens_per_sec,
            'elapsed_hours': elapsed / 3600,
            'eta_hours': eta_seconds / 3600,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_progress(self, stats: Dict[str, Any]):
        """Save progress to JSON file"""
        progress_file = os.path.join(self.checkpoint_dir, "training_progress.json")
        with open(progress_file, 'w') as f:
            json.dump(stats, f, indent=2)

# ========================= Training Loop =========================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    config: Dict[str, Any],
    rank: int = 0,
    world_size: int = 1
):
    """Main training loop with comprehensive monitoring and GPU verification"""
    
    # Verify GPU setup at start
    if is_main_process():
        verify_gpu_setup(model, device)
    
    # Initialize tracking
    progress_tracker = ProgressTracker(
        config['total_tokens'],
        config['daily_tokens'],
        config['checkpoint_dir']
    )
    
    writer = SummaryWriter(config['runs_dir']) if is_main_process() else None
    
    # Training state
    step = config['start_step']
    tokens_seen = config['start_tokens']
    best_val_loss = float('inf')
    running_loss = 0.0
    grad_norms = []
    
    # GPU monitoring
    gpu_check_interval = 100  # Check every 100 steps
    last_gpu_check = 0
    
    # Progress bars (main process only)
    if is_main_process():
        pbar_total = tqdm(
            total=config['total_steps'],
            initial=step,
            desc="Total Progress",
            position=0
        )
        pbar_daily = tqdm(
            total=config['daily_steps'],
            initial=step % config['daily_steps'],
            desc=f"Day {(tokens_seen // config['daily_tokens']) + 1}",
            position=1
        )
    
    model.train()
    
    # Training loop
    while step < config['total_steps']:
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            # Convert to tensors if they're numpy arrays
            if isinstance(input_ids, np.ndarray):
                input_ids = torch.from_numpy(input_ids).long()
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long()
            
            # Move to device and verify
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # GPU check
            if step - last_gpu_check >= gpu_check_interval and is_main_process():
                logger.info(f"[Step {step}] Input device: {input_ids.device}, Model device: {next(model.parameters()).device}")
                gpu_util = torch.cuda.utilization(device)
                logger.info(f"[Step {step}] GPU utilization: {gpu_util}%")
                last_gpu_check = step
            
            # Update token count
            batch_tokens = input_ids.numel()
            tokens_seen += batch_tokens * world_size  # Account for all GPUs
            
            try:
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if config['use_bf16'] else torch.float16):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss / config['accumulation_steps']
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % config['accumulation_steps'] == 0:
                    # Unscale and clip gradients
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config['gradient_clip']
                    )
                    grad_norms.append(grad_norm.item())
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                
                # Track loss
                running_loss += loss.item() * config['accumulation_steps']
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"[Rank {rank}] OOM at step {step}, clearing cache")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise e
            
            # Logging
            if (step + 1) % config['log_interval'] == 0 and is_main_process():
                avg_loss = running_loss / config['log_interval']
                avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
                
                # Get progress stats
                stats = progress_tracker.get_progress_stats(tokens_seen, step + 1, avg_loss)
                stats['grad_norm'] = avg_grad_norm
                stats['learning_rate'] = scheduler.get_last_lr()[0]
                
                # GPU memory stats
                if torch.cuda.is_available():
                    stats['gpu_memory_allocated'] = torch.cuda.memory_allocated(device) / 1e9
                    stats['gpu_memory_reserved'] = torch.cuda.memory_reserved(device) / 1e9
                    stats['gpu_utilization'] = torch.cuda.utilization(device)
                
                # Log to console
                logger.info(
                    f"Step {step + 1} | Day {stats['current_day']} | "
                    f"Progress: {stats['total_progress']:.1f}% | "
                    f"Loss: {avg_loss:.4f} | LR: {stats['learning_rate']:.2e} | "
                    f"Speed: {stats['tokens_per_sec']:.0f} tok/s | "
                    f"GPU: {stats.get('gpu_utilization', 0)}%"
                )
                
                # Log to TensorBoard
                if writer:
                    writer.add_scalar('train/loss', avg_loss, step + 1)
                    writer.add_scalar('train/learning_rate', stats['learning_rate'], step + 1)
                    writer.add_scalar('train/grad_norm', avg_grad_norm, step + 1)
                    writer.add_scalar('train/tokens_per_second', stats['tokens_per_sec'], step + 1)
                    writer.add_scalar('progress/total_percent', stats['total_progress'], step + 1)
                    writer.add_scalar('progress/day', stats['current_day'], step + 1)
                    writer.add_scalar('gpu/memory_allocated_gb', stats.get('gpu_memory_allocated', 0), step + 1)
                    writer.add_scalar('gpu/utilization_percent', stats.get('gpu_utilization', 0), step + 1)
                
                # Save progress
                progress_tracker.save_progress(stats)
                
                # Reset tracking
                running_loss = 0.0
                grad_norms = []
            
            # Validation
            if val_loader and (step + 1) % config['val_interval'] == 0 and is_main_process():
                logger.info("Running validation...")
                val_loss = validate(model, val_loader, device, config['val_steps'])
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save best model
                    best_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
                    torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), best_path)
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                
                if writer:
                    writer.add_scalar('val/loss', val_loss, step + 1)
                    writer.add_scalar('val/best_loss', best_val_loss, step + 1)
                
                model.train()
            
            # Checkpointing
            if (step + 1) % config['checkpoint_interval'] == 0 and is_main_process():
                checkpoint_path = os.path.join(
                    config['checkpoint_dir'],
                    f"ckpt_step{step + 1}_tokens{tokens_seen}.pt"
                )
                
                checkpoint = {
                    'step': step + 1,
                    'tokens_seen': tokens_seen,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # Rotate checkpoints
                all_ckpts = sorted(glob.glob(os.path.join(config['checkpoint_dir'], "ckpt_step*.pt")))
                if len(all_ckpts) > config['max_checkpoints']:
                    for old_ckpt in all_ckpts[:-config['max_checkpoints']]:
                        os.remove(old_ckpt)
                        logger.info(f"Removed old checkpoint: {os.path.basename(old_ckpt)}")
            
            # Update progress bars
            if is_main_process():
                pbar_total.update(1)
                pbar_daily.update(1)
                
                # Check for day transition
                new_day = (tokens_seen // config['daily_tokens']) + 1
                current_day = ((tokens_seen - batch_tokens * world_size) // config['daily_tokens']) + 1
                if new_day > current_day:
                    pbar_daily.close()
                    pbar_daily = tqdm(
                        total=config['daily_steps'],
                        initial=0,
                        desc=f"Day {new_day}",
                        position=1
                    )
            
            step += 1
            if step >= config['total_steps']:
                break
    
    # Cleanup
    if is_main_process():
        pbar_total.close()
        pbar_daily.close()
        if writer:
            writer.close()
    
    logger.info(f"Training completed! Total tokens: {tokens_seen:,}")

# ========================= Main Function =========================

def main():
    parser = argparse.ArgumentParser(description="LLM Pretraining - 3B Tokens")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--rank", type=int, default=0, help="Rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--port", type=str, default="12355", help="Port for distributed training")
    args = parser.parse_args()
    
    # Set device and distributed training
    device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu")
    
    if args.world_size > 1:
        setup_ddp(args.rank, args.world_size, args.port)
    
    # Set random seeds
    torch.manual_seed(SEED + args.rank)
    torch.cuda.manual_seed_all(SEED + args.rank)
    
    # Verify CUDA setup
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! Training will be extremely slow on CPU.")
        sys.exit(1)
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = LlamaTokenizer.from_pretrained(
        TOKENIZER_NAME,
        token=os.getenv("HF_TOKEN"),
        cache_dir=HF_CACHE_DIR
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = AnameeModel(
        vocab_size=tokenizer.vocab_size,
        dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        max_seq_len=BLOCK_SIZE,
        dropout=0.0,  # No dropout during pretraining
        use_checkpoint=GRADIENT_CHECKPOINTING,
        checkpoint_start_layer=8
    ).to(device)
    
    if is_main_process():
        param_count = count_parameters(model)
        model_size = model_size_mb(model)
        logger.info(f"Model initialized: {param_count:,} parameters ({model_size:.1f} MB)")
    
    # Wrap in DDP if distributed
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.rank], find_unused_parameters=False)
    
    # Initialize dataset
    logger.info("Initializing dataset...")
    train_dataset = Curriculum2BTokenDataset(
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        easy_token_target=EASY_TOKEN_TARGET,
        hard_token_target=HARD_TOKEN_TARGET,
        seed=SEED,
        shard_id=args.rank,
        num_shards=args.world_size,
        cache_dir=HF_CACHE_DIR
    )
    
    # Custom collate function to ensure GPU tensors
    def collate_fn(batch):
        # batch is a list of (x, y) numpy arrays
        xs, ys = zip(*batch)
        # Stack into tensors
        input_ids = torch.stack([torch.from_numpy(x).long() for x in xs])
        labels = torch.stack([torch.from_numpy(y).long() for y in ys])
        return input_ids, labels
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn
    )
    
    # Validation dataset (main process only)
    val_loader = None
    if is_main_process():
        val_dataset = build_val_dataset(
            tokenizer, 
            BLOCK_SIZE, 
            VAL_STEPS,
            SEED,
            HF_CACHE_DIR
        )
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=VAL_BATCH_SIZE,
                shuffle=False,
                pin_memory=True
            )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
        fused=True  # Use fused optimizer on A100
    )
    
    # Initialize scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_TRAINING_STEPS
    )
    
    # Initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Resume from checkpoint if specified
    start_step = 0
    start_tokens = 0
    
    if args.resume or AUTO_FIND_RESUME:
        # Auto-find latest checkpoint
        if not args.resume and os.path.exists(CHECKPOINT_DIR):
            ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_step*.pt")))
            if ckpts:
                args.resume = ckpts[-1]
                logger.info(f"Auto-detected latest checkpoint: {args.resume}")
        
        # Load checkpoint
        if args.resume and os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model state
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer and scheduler
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Resume training state
            start_step = checkpoint['step']
            start_tokens = checkpoint['tokens_seen']
            
            logger.info(f"Resumed from step {start_step} ({start_tokens:,} tokens)")
    
    # Prepare config for training
    train_config = {
        'total_tokens': TOTAL_TOKENS_TARGET,
        'daily_tokens': DAILY_TOKEN_TARGET,
        'total_steps': TOTAL_TRAINING_STEPS,
        'daily_steps': DAILY_TRAINING_STEPS,
        'checkpoint_dir': CHECKPOINT_DIR,
        'runs_dir': RUNS_DIR,
        'start_step': start_step,
        'start_tokens': start_tokens,
        'accumulation_steps': ACCUMULATION_STEPS,
        'log_interval': LOG_INTERVAL,
        'val_interval': VAL_INTERVAL,
        'val_steps': VAL_STEPS,
        'checkpoint_interval': CHECKPOINT_INTERVAL,
        'max_checkpoints': MAX_CHECKPOINTS,
        'gradient_clip': GRADIENT_CLIP_VAL,
        'use_bf16': USE_BF16,
    }
    
    # Start training
    try:
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=train_config,
            rank=args.rank,
            world_size=args.world_size
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        if args.world_size > 1:
            cleanup_ddp()
    
    # Save final model (main process only)
    if is_main_process():
        final_path = os.path.join(CHECKPOINT_DIR, "final_model_3b.pt")
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(model_state, final_path)
        logger.info(f"Final model saved at {final_path}")

if __name__ == "__main__":
    # Add numpy import at the top
    import numpy as np
    main()