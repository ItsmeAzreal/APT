# pretrain_ddp.py
"""
Distributed Streaming Pretraining Script (PyTorch DDP)
- Ready for multi-GPU (2x RTX 5090 or more)
- Streaming + sharded dataset support
- AMP (mixed precision)
- TensorBoard logging
- Clean structure, extensible for validation or sampling

USAGE:
torchrun --nproc_per_node=2 pretrain_ddp.py
"""

import os
import math
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import LlamaTokenizer

# Import your local modules (adjust if needed)
from src.config import *
from src.model import AnameeModel, count_parameters
from src.dataset import Curriculum2BTokenDataset
from src.validate import validate   # Optional, not used below
from src.sample import sample, decode_text  # Optional, not used below

# ------------------------- DDP Setup Functions -------------------------

def setup_ddp(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# ------------------------- Training Loop -------------------------

def train_loop_ddp(model, train_loader, optimizer, scheduler, device, scaler, start_step, total_steps, 
                   accumulation_steps, log_interval, writer, loss_spike_factor, rank):
    running_loss = 0
    tokens_seen = 0
    lowest_loss = None
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model.train()
    step = start_step
    while step < total_steps:
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            bsz, seqlen = x.shape
            tokens_seen += bsz * seqlen

            try:
                if device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16 if use_bf16 else torch.float16):
                        logits = model(x)
                        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / accumulation_steps
                    loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[Rank {rank}] OOM at step {step}, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            if not math.isfinite(loss.item()):
                print(f"[Rank {rank}] Loss NaN/inf at step {step}!")
                break

            if lowest_loss is None or loss.item() < lowest_loss:
                lowest_loss = loss.item()
            if loss.item() > loss_spike_factor * lowest_loss:
                print(f"[Rank {rank}] Loss spike detected at step {step}! Stopping training.")
                break

            if (step + 1) % accumulation_steps == 0:
                if device.type == "cuda":
                    scaler.unscale_(optimizer)
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                # Only rank 0 writes logs
                if writer is not None and rank == 0:
                    writer.add_scalar('grad_norm', grad, step + 1)

            running_loss += loss.item() * accumulation_steps

            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                if rank == 0:
                    print(f"Step {step+1} | Loss: {avg_loss:.4f} | Tokens: {tokens_seen}")
                    if writer is not None:
                        writer.add_scalar('train_loss', avg_loss, step + 1)
                running_loss = 0

            step += 1
            if step >= total_steps:
                break

# ------------------------- Main Function -------------------------

def main(rank, world_size, args):
    # Distributed setup
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Tokenizer and Dataset
    tokenizer = LlamaTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    vocab_size = tokenizer.vocab_size

    # Example of sharding: let each rank have its own shard (pass rank/world_size)
    train_dataset = Curriculum2BTokenDataset(
        tokenizer, BLOCK_SIZE, int(2_000_000_000 * 0.6), int(2_000_000_000 * 0.4),
        shard_id=rank, num_shards=world_size
    )

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
    )

    # Model and Optimizer
    model = AnameeModel(
        vocab_size, MODEL_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE
    ).to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_TRAINING_STEPS)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Only rank 0 logs
    writer = SummaryWriter(RUNS_DIR) if rank == 0 else None

    # Training loop
    train_loop_ddp(
        model, train_loader, optimizer, scheduler, device, scaler, 0, TOTAL_TRAINING_STEPS,
        ACCUMULATION_STEPS, LOG_INTERVAL, writer, LOSS_SPIKE_FACTOR, rank
    )

    # Save only on rank 0
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pt"))
        print("Model saved at", os.path.join(CHECKPOINT_DIR, "final_model.pt"))

    cleanup_ddp()

# ------------------------- CLI & Distributed Launch -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 2)),
                        help="Number of GPUs / distributed processes")
    args = parser.parse_args()

    world_size = args.world_size

    # For torchrun, this will be called in parallel for each rank
    # Get rank from env var (set automatically by torchrun)
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0

    main(rank, world_size, args)
