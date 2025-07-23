# pretrain_ddp.py
# Clean, well-commented, single-GPU PyTorch training script for A100

"""
Streaming Pretraining Script (Single GPU, PyTorch)
- For NVIDIA A100 (or any single CUDA GPU)
- Calls your Curriculum2BTokenDataset for curriculum learning
- Efficient: uses mixed-precision (bf16/fp16) on A100
- Tracks daily and total progress, loss, validation, and checkpoints
"""

import os
import math
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import LlamaTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import time
import json
from dotenv import load_dotenv

# --- Load environment variables (for HF cache, etc.) ---
load_dotenv(dotenv_path="APT/.env")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/default/path/hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE", "/default/path/hf_cache")

# --- Import project modules (config, model, datasets, validation) ---
from src.config import *  # BATCH_SIZE, LR, USE_BF16, etc.
from src.model import AnameeModel, count_parameters
from src.dataset import Curriculum2BTokenDataset, build_val_dataset
from src.validate import validate

TOKENIZER_PATH = "meta-llama/Llama-2-7b-hf" 

def train_loop(
    model, 
    train_loader, 
    optimizer, 
    scheduler, 
    device, 
    scaler, 
    start_step, 
    total_steps,
    accumulation_steps, 
    log_interval, 
    writer, 
    loss_spike_factor,
    val_loader=None, 
    val_interval=2000, 
    vocab_size=32000
):
    """
    Main training loop: runs forward, backward, optimizer steps,
    logs progress, saves best checkpoints, and handles curriculum logic.
    """
    running_loss = 0
    tokens_seen = 0 if start_step == 0 else start_step * TOKENS_PER_STEP
    best_val_loss = float('inf')
    step = start_step

    # Progress tracking: days and tokens
    current_day = (tokens_seen // DAILY_TOKEN_TARGET) + 1 if tokens_seen > 0 else 1
    daily_steps_done = step % DAILY_TRAINING_STEPS if step > 0 else 0

    # Timers and progress bars
    start_time = time.time()
    autocast_dtype = torch.bfloat16 if USE_BF16 else torch.float16
    model.train()
    pbar_total = tqdm(total=total_steps, initial=step, position=0, leave=True, desc="Total Progress")
    pbar_day = tqdm(total=DAILY_TRAINING_STEPS, initial=daily_steps_done, position=1, leave=True, desc=f"Day {current_day}")

    optimizer.zero_grad(set_to_none=True)
    for batch in train_loader:
        # --- Forward pass with AMP (mixed precision) ---
        with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=True):
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps  # Gradient accumulation

        # --- Backward pass and gradient accumulation ---
        scaler.scale(loss).backward()
        running_loss += loss.item()
        tokens_seen += TOKENS_PER_STEP

        # --- Optimizer step ---
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # --- Logging ---
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            print(f"[Step {step+1}] Loss: {avg_loss:.4f} | Tokens: {tokens_seen}")
            writer.add_scalar("Loss/train", avg_loss, step+1)
            running_loss = 0

        # --- Validation and checkpointing ---
        if val_loader and (step + 1) % val_interval == 0:
            val_loss = validate(model, val_loader, device)
            writer.add_scalar("Loss/val", val_loss, step+1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"checkpoints/best_model.pt")

        # --- Update progress bars and day tracking ---
        pbar_total.update(1)
        pbar_day.update(1)
        step += 1
        if (step % DAILY_TRAINING_STEPS) == 0:
            pbar_day.close()
            current_day += 1
            pbar_day = tqdm(total=DAILY_TRAINING_STEPS, initial=0, position=1, leave=True, desc=f"Day {current_day}")

        # --- End training if done ---
        if step >= total_steps:
            break

    pbar_total.close()
    pbar_day.close()

TRAINING_STEPS = 3814 
def main():
    # --- Parse command line arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--accumulation_steps", type=int, default=ACCUMULATION_STEPS)
    parser.add_argument("--log_interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--val_interval", type=int, default=VAL_INTERVAL)
    parser.add_argument("--val", action="store_true")
    args, unknown = parser.parse_known_args()


    # --- Set CUDA device (your A100) ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Using device: {device}")

    # --- Load tokenizer, model, and print parameter count ---
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AnameeModel().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # --- Build training set using your Curriculum2BTokenDataset ---
    train_dataset = Curriculum2BTokenDataset(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size or BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # --- Optionally load validation set ---
    val_loader = None
    if args.val:
        val_dataset = build_val_dataset(tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    # --- Set up optimizer, LR scheduler, AMP scaler, TensorBoard writer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=args.total_steps,
    )
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir="runs/experiment_2b")

    # --- Optionally resume from checkpoint ---
    start_step = args.start_step
    if os.path.exists("checkpoints/latest.pt"):
        print("Loading checkpoint...")
        checkpoint = torch.load("checkpoints/latest.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = checkpoint["step"]

    # --- Main training loop ---
    train_loop(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        start_step=start_step,
        total_steps=args.total_steps,
        accumulation_steps=args.accumulation_steps,
        log_interval=args.log_interval,
        writer=writer,
        loss_spike_factor=LOSS_SPIKE_FACTOR,
        val_loader=val_loader,
        val_interval=args.val_interval,
        vocab_size=tokenizer.vocab_size
    )

    # --- Save checkpoint at end ---
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": args.total_steps
    }, "checkpoints/latest.pt")
    print("Training complete! Final model saved.")

if __name__ == "__main__":
    main()
