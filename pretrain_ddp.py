# pretrain_ddp.py - Complete version with daily progress tracking for 3B tokens

"""
Distributed Streaming Pretraining Script (PyTorch DDP)
- Optimized for 2x RTX 5090 GPUs
- 3B token training with 500M daily targets
- Enhanced progress tracking and monitoring
- Memory-efficient with gradient checkpointing
"""

import os
import math
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import LlamaTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import glob
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="APT/.env")

os.environ["HF_HOME"] = os.getenv("HF_HOME", "/default/path/hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/default/path/hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE", "/default/path/hf_cache")

# Project imports
from src.config import *
from src.model import AnameeModel, count_parameters
from src.dataset import Curriculum2BTokenDataset, build_val_dataset
from src.validate import validate

# ------------------------- DDP Setup Functions -------------------------
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # Set memory fraction to avoid OOM
    torch.cuda.set_per_process_memory_fraction(0.95, device=rank)

def cleanup_ddp():
    dist.destroy_process_group()

# ------------------------- Enhanced Training Loop with Daily Progress -------------------------
def train_loop_ddp(
    model, train_loader, optimizer, scheduler, device, scaler, start_step, total_steps,
    accumulation_steps, log_interval, writer, loss_spike_factor, rank,
    val_loader=None, val_interval=2000, vocab_size=32000
):
    running_loss = 0
    tokens_seen = 0 if start_step == 0 else start_step * TOKENS_PER_STEP
    lowest_loss = None
    step = start_step
    grad_norm = 0.0
    
    # Progress tracking variables
    current_day = (tokens_seen // DAILY_TOKEN_TARGET) + 1 if tokens_seen > 0 else 1
    daily_steps_done = step % DAILY_TRAINING_STEPS if step > 0 else 0
    best_val_loss = float('inf')
    val_history = []
    
    # Initialize timing
    start_time = time.time()
    last_log_time = start_time
    last_tokens = tokens_seen
    
    # Use config for dtype
    autocast_dtype = torch.bfloat16 if USE_BF16 else torch.float16

    model.train()
    
    # Dual progress bars for total and daily progress
    pbar_total = tqdm(
        total=total_steps, initial=step, position=0, leave=True,
        desc=f"Total 3B Progress", colour='green', dynamic_ncols=True
    )
    
    pbar_daily = tqdm(
        total=DAILY_TRAINING_STEPS, initial=daily_steps_done,
        position=1, leave=True, desc=f"Day {current_day}/6 (500M tokens)",
        colour='blue', dynamic_ncols=True
    )

    while step < total_steps:
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            bsz, seqlen = x.shape
            tokens_seen += bsz * seqlen
            
            # Check if we've moved to a new day
            new_day = (tokens_seen // DAILY_TOKEN_TARGET) + 1
            if new_day > current_day:
                current_day = new_day
                pbar_daily.close()
                pbar_daily = tqdm(
                    total=DAILY_TRAINING_STEPS, initial=0,
                    position=1, leave=True, desc=f"Day {current_day}/6 (500M tokens)",
                    colour='blue', dynamic_ncols=True
                )
                if rank == 0:
                    print(f"\n{'='*80}")
                    print(f"🎉 Starting Day {current_day} of 6!")
                    print(f"{'='*80}\n")

            try:
                if device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=autocast_dtype):
                        logits = model(x)
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)), y.view(-1)
                        )
                    loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
                    loss = loss / accumulation_steps
                    loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[Rank {rank}] OOM at step {step}, clearing cache and skipping batch")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise e

            # Loss monitoring
            if not math.isfinite(loss.item()):
                print(f"[Rank {rank}] Loss NaN/inf at step {step}! Stopping.")
                return
            
            if lowest_loss is None or loss.item() < lowest_loss:
                lowest_loss = loss.item()
            if loss.item() > loss_spike_factor * lowest_loss:
                print(f"[Rank {rank}] Loss spike at step {step}: {loss.item():.4f} > {loss_spike_factor * lowest_loss:.4f}")
                if loss.item() > 10 * lowest_loss:  # Only stop for extreme spikes
                    return

            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                if device.type == "cuda":
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VAL)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VAL)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * accumulation_steps

            # Enhanced logging with daily progress
            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                current_time = time.time()
                
                # Calculate metrics
                time_elapsed = current_time - last_log_time
                tokens_processed = tokens_seen - last_tokens
                tokens_per_sec = tokens_processed / time_elapsed if time_elapsed > 0 else 0
                
                # Progress calculations
                daily_progress = ((tokens_seen % DAILY_TOKEN_TARGET) / DAILY_TOKEN_TARGET * 100) if tokens_seen % DAILY_TOKEN_TARGET != 0 else 100
                total_progress = tokens_seen / TOTAL_TOKENS_TARGET * 100
                
                # ETA calculations
                if tokens_per_sec > 0:
                    remaining_today = DAILY_TOKEN_TARGET - (tokens_seen % DAILY_TOKEN_TARGET) if tokens_seen % DAILY_TOKEN_TARGET != 0 else 0
                    remaining_total = TOTAL_TOKENS_TARGET - tokens_seen
                    eta_today = remaining_today / tokens_per_sec / 60  # minutes
                    eta_total = remaining_total / tokens_per_sec / 3600  # hours
                else:
                    eta_today = eta_total = 0
                
                # Calculate perplexity
                perplexity = math.exp(min(avg_loss, 20))  # Cap at 20 to avoid overflow
                
                if rank == 0:
                    # GPU memory stats
                    allocated_gb = torch.cuda.memory_allocated(device) / 1e9
                    reserved_gb = torch.cuda.memory_reserved(device) / 1e9
                    
                    # Enhanced display format
                    print(f"\n{'='*80}")
                    print(f"📊 TRAINING STATUS - Step {step+1:,}")
                    print(f"{'='*80}")
                    print(f"📅 Day {current_day}/6 | Daily Progress: {daily_progress:.1f}%")
                    print(f"🎯 Total Progress: {tokens_seen:,} / {TOTAL_TOKENS_TARGET:,} ({total_progress:.1f}%)")
                    print(f"{'—'*80}")
                    print(f"📉 Train Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
                    if val_history:
                        print(f"📈 Val Loss: {val_history[-1][1]:.4f} | Best Val: {best_val_loss:.4f}")
                    print(f"{'—'*80}")
                    print(f"⚡ Speed: {tokens_per_sec:,.0f} tok/s | LR: {scheduler.get_last_lr()[0]:.2e}")
                    print(f"💾 GPU Memory: {allocated_gb:.1f}/{reserved_gb:.1f}GB ({allocated_gb/reserved_gb*100:.0f}%)")
                    print(f"⏱️  ETA - Today: {eta_today:.0f}m | Total: {eta_total:.1f}h")
                    print(f"{'='*80}\n")
                    
                    if writer is not None:
                        writer.add_scalar('train/loss', avg_loss, step + 1)
                        writer.add_scalar('train/perplexity', perplexity, step + 1)
                        writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], step + 1)
                        writer.add_scalar('train/tokens_per_second', tokens_per_sec, step + 1)
                        writer.add_scalar('train/grad_norm', grad_norm, step + 1)
                        writer.add_scalar('system/gpu_memory_allocated_gb', allocated_gb, step + 1)
                        writer.add_scalar('system/gpu_memory_reserved_gb', reserved_gb, step + 1)
                        writer.add_scalar('progress/daily_percent', daily_progress, step + 1)
                        writer.add_scalar('progress/total_percent', total_progress, step + 1)
                        writer.add_scalar('progress/current_day', current_day, step + 1)
                
                running_loss = 0
                last_log_time = current_time
                last_tokens = tokens_seen

            # Validation
            if val_loader is not None and (step + 1) % val_interval == 0 and rank == 0:
                print(f"\n🔍 Running validation...")
                val_loss = validate(model, val_loader, device, val_steps=VAL_STEPS)
                val_perplexity = math.exp(min(val_loss, 20))
                val_history.append((step + 1, val_loss))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"✨ New best validation loss: {val_loss:.4f}")
                
                print(f"📊 Val Loss: {val_loss:.4f} | Val PPL: {val_perplexity:.2f}\n")
                
                if writer is not None:
                    writer.add_scalar('val/loss', val_loss, step + 1)
                    writer.add_scalar('val/perplexity', val_perplexity, step + 1)
                    writer.add_scalar('val/best_loss', best_val_loss, step + 1)
                
                model.train()  # Back to training mode

            # Enhanced checkpointing with rotation and daily summaries
            if ((step + 1) % CHECKPOINT_INTERVAL == 0 or (step + 1) == total_steps) and rank == 0:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                
                # Calculate actual tokens per checkpoint
                tokens_per_ckpt = tokens_seen - getattr(train_loop_ddp, 'last_ckpt_tokens', 0)
                train_loop_ddp.last_ckpt_tokens = tokens_seen
                
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_step{step+1}_tokens{tokens_seen}.pt")
                
                # Save comprehensive checkpoint with daily summary
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "step": step + 1,
                    "tokens_seen": tokens_seen,
                    "config": {
                        "model_dim": MODEL_DIM,
                        "num_layers": NUM_LAYERS,
                        "num_heads": NUM_HEADS,
                        "num_kv_heads": NUM_KV_HEADS,
                        "hidden_dim": HIDDEN_DIM,
                        "vocab_size": vocab_size,
                        "block_size": BLOCK_SIZE,
                    },
                    "training_config": {
                        "batch_size": BATCH_SIZE,
                        "accumulation_steps": ACCUMULATION_STEPS,
                        "learning_rate": LEARNING_RATE,
                        "total_steps": TOTAL_TRAINING_STEPS,
                    },
                    "daily_summary": {
                        "current_day": current_day,
                        "daily_progress": daily_progress,
                        "total_progress": total_progress,
                        "best_val_loss": best_val_loss if val_history else None,
                    }
                }, ckpt_path)
                
                # Checkpoint rotation - keep only last MAX_CHECKPOINTS
                all_ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_step*.pt")))
                if len(all_ckpts) > MAX_CHECKPOINTS:
                    for old_ckpt in all_ckpts[:-MAX_CHECKPOINTS]:
                        os.remove(old_ckpt)
                        print(f"🗑️  Removed old checkpoint: {os.path.basename(old_ckpt)}")
                
                print(f"✅ Checkpoint saved: step {step+1:,} ({tokens_seen:,} tokens, {tokens_per_ckpt:,} since last)")
                
                # Update progress file
                progress_data = {
                    'day': current_day,
                    'total_tokens': tokens_seen,
                    'total_steps': step + 1,
                    'progress_percent': total_progress,
                    'latest_checkpoint': ckpt_path,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'best_val_loss': best_val_loss if val_history else None
                }
                with open(PROGRESS_FILE, 'w') as f:
                    json.dump(progress_data, f, indent=2)
            
            step += 1
            pbar_total.update(1)
            pbar_daily.update(1)
            
            if step >= total_steps:
                break
    
    pbar_total.close()
    pbar_daily.close()
    
    # Final statistics
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"🎉 TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"📊 Final Statistics:")
        print(f"  • Total tokens: {tokens_seen:,}")
        print(f"  • Total time: {total_time/3600:.2f} hours")
        print(f"  • Average speed: {tokens_seen/total_time:.0f} tok/s")
        print(f"  • Final loss: {avg_loss:.4f}")
        if val_history:
            print(f"  • Best val loss: {best_val_loss:.4f}")
        print(f"{'='*80}")

# ------------------------- Main Function -------------------------
def main(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    print(f"[Rank {rank}] Initializing on {torch.cuda.get_device_name(rank)}")
    print(f"[Rank {rank}] Memory: {torch.cuda.get_device_properties(rank).total_memory / 1e9:.1f}GB")
    
    # Initialize tokenizer
    tokenizer = LlamaTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    vocab_size = tokenizer.vocab_size
    
    # Create dataset with proper sharding for 3B tokens
    train_dataset = Curriculum2BTokenDataset(
        tokenizer, 
        BLOCK_SIZE, 
        EASY_TOKEN_TARGET,  # 1.8B easy tokens (from config)
        HARD_TOKEN_TARGET,  # 1.2B hard tokens (from config)
        shard_id=rank, 
        num_shards=world_size
    )
    
    # No DistributedSampler needed for IterableDataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
    )
    
    # Validation dataset (only on rank 0)
    val_loader = None
    if rank == 0:
        val_dataset = build_val_dataset(tokenizer, BLOCK_SIZE, val_size=VAL_STEPS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = AnameeModel(
        vocab_size, MODEL_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE
    ).to(device)
    
    if rank == 0:
        param_count = count_parameters(model)
        print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    # Wrap in DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
        fused=True  
    )
    
    # Cosine scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_TRAINING_STEPS
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # TensorBoard writer (only rank 0)
    writer = SummaryWriter(RUNS_DIR) if rank == 0 else None
    
    # Resume from checkpoint if specified
    start_step = 0
    tokens_seen = 0
    if args.resume or (AUTO_FIND_RESUME and os.path.exists(CHECKPOINT_DIR)):
        # Auto-find latest checkpoint if not specified
        if not args.resume:
            ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_step*.pt")))
            if ckpts:
                args.resume = ckpts[-1]
                print(f"[Rank {rank}] Auto-detected latest checkpoint: {args.resume}")
        
        if args.resume and os.path.isfile(args.resume):
            print(f"[Rank {rank}] Loading checkpoint from {args.resume}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            ckpt = torch.load(args.resume, map_location=map_location)
            
            model.module.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])
            
            start_step = ckpt.get("step", 0)
            tokens_seen = ckpt.get("tokens_seen", 0)
            print(f"[Rank {rank}] Resumed from step {start_step} ({tokens_seen:,} tokens)")
    
    # Start training
    try:
        train_loop_ddp(
            model, train_loader, optimizer, scheduler, device, scaler,
            start_step, TOTAL_TRAINING_STEPS,
            ACCUMULATION_STEPS, LOG_INTERVAL, writer, LOSS_SPIKE_FACTOR, rank,
            val_loader=val_loader, val_interval=VAL_INTERVAL, vocab_size=vocab_size
        )
    except KeyboardInterrupt:
        print(f"\n[Rank {rank}] Training interrupted by user")
    except Exception as e:
        print(f"\n[Rank {rank}] Training failed with error: {e}")
        raise
    
    # Save final model (rank 0 only)
    if rank == 0:
        final_path = os.path.join(CHECKPOINT_DIR, "final_model_3b.pt")
        torch.save(model.module.state_dict(), final_path)
        print(f"Final model saved at {final_path}")
        
        if writer is not None:
            writer.close()
    
    cleanup_ddp()

# ------------------------- Entry Point -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed LLM Pretraining - 3B Tokens")
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 2)),
                        help="Number of distributed processes")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    world_size = args.world_size
    rank = int(os.environ.get("RANK", 0))
    
    # Set random seeds for reproducibility
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed_all(SEED + rank)
    
    main(rank, world_size, args)