# src/dataset.py - Fixed streaming dataset with proper sharding

from datasets import load_dataset
import torch
from typing import Iterator, Tuple
import time

# === Curriculum Filtering Functions ===

def easy_filter(ex):
    """
    Filter for 'easy' phase of curriculum:
    - English only
    - int_score >= 3 (quality)
    - text shorter than 400 characters
    """
    return (
        ex.get('language', '') == 'en'
        and ex.get('int_score', 0) >= 3
        and len(ex.get('text', '')) < 400
    )

def hard_filter(ex):
    """
    Filter for 'hard' phase of curriculum:
    - English only
    - int_score >= 3 (quality)
    - text longer than 800 characters
    """
    return (
        ex.get('language', '') == 'en'
        and ex.get('int_score', 0) >= 3
        and len(ex.get('text', '')) > 800
    )

# === Streaming Loader with Proper Sharding ===

def filtered_stream(stage, seed, shard_id, num_shards, buffer_size=50_000):
    """
    Loads the streaming HuggingFaceFW/fineweb-edu dataset with proper sharding.
    Each GPU only processes its assigned shard of data.
    """
    print(f"[Rank {shard_id}] Loading HuggingFaceFW/fineweb-edu ({stage} curriculum stage)...")
    
    # Load the dataset with trust_remote_code for custom datasets
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
        trust_remote_code=True  # Add this to ensure proper loading
    )
    
    # Apply shuffling first
    stream = stream.shuffle(seed=seed + shard_id, buffer_size=buffer_size)
    
    # Apply stage filter
    if stage == 'easy':
        stream = stream.filter(easy_filter)
    else:
        stream = stream.filter(hard_filter)
    
    # FIXED: Shard the data without using with_indices
    # Instead of using filter with indices, we'll implement sharding manually
    # in the iteration loop
    return stream, shard_id, num_shards

class Curriculum2BTokenDataset(torch.utils.data.IterableDataset):
    """
    Streaming IterableDataset for curriculum-based LLM pretraining.
    - Streams 'easy' then 'hard' phase in order
    - Properly shards data across multiple GPUs
    - Stops when token targets are reached
    """
    def __init__(self, tokenizer, block_size, easy_token_target, hard_token_target, 
                 seed=42, shard_id=0, num_shards=1):
        self.tokenizer = tokenizer
        self.block_size = block_size
        # Divide targets by number of shards since each GPU processes a portion
        self.easy_token_target = easy_token_target // num_shards
        self.hard_token_target = hard_token_target // num_shards
        self.seed = seed
        self.shard_id = shard_id
        self.num_shards = num_shards
        
        print(f"[Rank {shard_id}] Dataset initialized:")
        print(f"  - Easy tokens target: {self.easy_token_target:,}")
        print(f"  - Hard tokens target: {self.hard_token_target:,}")
        print(f"  - Total tokens target: {self.easy_token_target + self.hard_token_target:,}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        stages = [
            ('easy', self.easy_token_target),
            ('hard', self.hard_token_target)
        ]
        buffer = []
        total_tokens_yielded = 0
        examples_processed = 0
        start_time = time.time()

        for stage, token_target in stages:
            # Get the filtered stream along with sharding info
            curr_stream, shard_id, num_shards = filtered_stream(
                stage, self.seed, self.shard_id, self.num_shards
            )
            curr_tokens = 0
            stage_examples = 0
            example_idx = 0  # Manual index counter for sharding
            
            print(f"\n[Rank {self.shard_id}] Starting '{stage.upper()}' phase for {token_target:,} tokens...")
            
            for example in curr_stream:
                # Manual sharding: only process examples for this shard
                if example_idx % num_shards != shard_id:
                    example_idx += 1
                    continue
                example_idx += 1
                
                # Tokenize on the fly
                try:
                    # Check if example has the expected structure
                    if not isinstance(example, dict) or 'text' not in example:
                        continue
                        
                    text = example['text']
                    if not isinstance(text, str) or not text.strip():
                        continue
                        
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                except Exception as e:
                    print(f"[Rank {self.shard_id}] Tokenization error: {e}")
                    continue
                
                if not tokens:
                    continue
                
                buffer.extend(tokens)
                curr_tokens += len(tokens)
                total_tokens_yielded += len(tokens)
                examples_processed += 1
                stage_examples += 1

                # Yield block_size training chunks as (x, y) pairs
                while len(buffer) >= self.block_size + 1:
                    x_chunk = buffer[:self.block_size]
                    y_chunk = buffer[1:self.block_size + 1]
                    x = torch.tensor(x_chunk, dtype=torch.long)
                    y = torch.tensor(y_chunk, dtype=torch.long)
                    yield x, y
                    buffer = buffer[self.block_size:]
                
                # Progress reporting every 10M tokens
                if curr_tokens > 0 and curr_tokens % 10_000_000 < len(tokens):
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens_yielded / elapsed if elapsed > 0 else 0
                    print(f"[Rank {self.shard_id}] {stage}: {curr_tokens:,}/{token_target:,} tokens "
                          f"({stage_examples:,} examples, {tokens_per_sec:.0f} tok/s)")
                
                # End phase when token target met
                if curr_tokens >= token_target:
                    print(f"[Rank {self.shard_id}] {stage.upper()} phase completed: "
                          f"{curr_tokens:,} tokens from {stage_examples:,} examples")
                    break
            
            # Clear buffer between stages
            buffer = []
            
            # End if total tokens reached
            if total_tokens_yielded >= (self.easy_token_target + self.hard_token_target):
                break
        
        print(f"[Rank {self.shard_id}] Dataset iteration complete. "
              f"Total: {total_tokens_yielded:,} tokens from {examples_processed:,} examples")

# === Validation Dataset Construction ===

def build_val_dataset(tokenizer, block_size, val_size=100, seed=42):
    """
    Build a small, fixed-size validation dataset.
    Returns a list of (x, y) tensor pairs.
    """
    print(f"[VAL DATASET] Building validation set with {val_size} samples...")
    
    try:
        stream = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        # Only keep English and quality examples
        def val_filter(ex):
            return (
                isinstance(ex, dict) and
                ex.get('language', '') == 'en' 
                and ex.get('int_score', 0) >= 3
                and 400 < len(ex.get('text', '')) < 2000  # Medium length for validation
            )
        
        stream = stream.filter(val_filter)
        stream = stream.shuffle(seed=seed, buffer_size=10_000)
        
        buffer = []
        val_samples = 0
        xs, ys = []
        
        for example in stream:
            try:
                if not isinstance(example, dict) or 'text' not in example:
                    continue
                    
                text = example['text']
                if not isinstance(text, str) or not text.strip():
                    continue
                    
                tokens = tokenizer.encode(text, add_special_tokens=False)
            except Exception as e:
                continue
                
            if not tokens:
                continue
                
            buffer.extend(tokens)
            
            # Extract block_size chunks
            while len(buffer) >= block_size + 1 and val_samples < val_size:
                x_chunk = buffer[:block_size]
                y_chunk = buffer[1:block_size + 1]
                xs.append(torch.tensor(x_chunk, dtype=torch.long))
                ys.append(torch.tensor(y_chunk, dtype=torch.long))
                buffer = buffer[block_size:]
                val_samples += 1
                
                if val_samples >= val_size:
                    print(f"[VAL DATASET] Built {val_samples} validation samples.")
                    return list(zip(xs, ys))
        
        print(f"[VAL DATASET] Built {val_samples} validation samples (stream ended early).")
        return list(zip(xs, ys))
    
    except Exception as e:
        print(f"[VAL DATASET] Error building validation set: {e}")
        # Return empty dataset as fallback
        return []

# === End of dataset.py ===