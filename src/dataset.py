# src/dataset.py - Fixed streaming dataset with proper authentication and error handling

import os
import torch
from datasets import load_dataset
from typing import Iterator, Tuple, Optional, Dict, Any
import time
from torch.utils.data import IterableDataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Curriculum Filtering Functions ===

def easy_filter(example: Dict[str, Any]) -> bool:
    """
    Filter for 'easy' phase of curriculum:
    - English only
    - int_score >= 3 (quality)
    - text shorter than 400 characters
    """
    try:
        return (
            example.get('language', '') == 'en'
            and example.get('int_score', 0) >= 3
            and 0 < len(example.get('text', '')) < 400
        )
    except:
        return False

def hard_filter(example: Dict[str, Any]) -> bool:
    """
    Filter for 'hard' phase of curriculum:
    - English only
    - int_score >= 3 (quality)
    - text longer than 800 characters
    """
    try:
        return (
            example.get('language', '') == 'en'
            and example.get('int_score', 0) >= 3
            and len(example.get('text', '')) > 800
        )
    except:
        return False

# === Streaming Loader with Authentication ===

def create_streaming_dataset(
    stage: str, 
    seed: int, 
    shard_id: int, 
    num_shards: int,
    cache_dir: Optional[str] = None
):
    """
    Creates a properly authenticated and sharded streaming dataset.
    """
    logger.info(f"[Rank {shard_id}] Loading FineWeb-Edu dataset ({stage} stage)...")
    
    # Set cache directory
    if cache_dir:
        os.environ["HF_DATASETS_CACHE"] = cache_dir
    
    try:
        # Load dataset with authentication token from environment
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
            trust_remote_code=True,  # Required for some datasets
            token=os.getenv("HF_TOKEN"),  # Use token from environment
        )
        
        # Apply shuffling with different seed per shard
        dataset = dataset.shuffle(seed=seed + shard_id, buffer_size=10_000)
        
        # Apply stage-specific filter
        if stage == 'easy':
            dataset = dataset.filter(easy_filter)
        elif stage == 'hard':
            dataset = dataset.filter(hard_filter)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        return dataset
    
    except Exception as e:
        logger.error(f"[Rank {shard_id}] Failed to load dataset: {e}")
        raise

class Curriculum2BTokenDataset(IterableDataset):
    """
    Streaming dataset for curriculum-based LLM pretraining.
    - Properly handles sharding across GPUs
    - Includes error recovery and retry logic
    - Tracks progress accurately
    """
    def __init__(
        self, 
        tokenizer,
        block_size: int = 2048,
        easy_token_target: int = 1_800_000_000,
        hard_token_target: int = 1_200_000_000,
        seed: int = 42,
        shard_id: int = 0,
        num_shards: int = 1,
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.easy_token_target = easy_token_target // num_shards
        self.hard_token_target = hard_token_target // num_shards
        self.seed = seed
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Add special tokens if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"[Rank {shard_id}] Curriculum2BTokenDataset initialized:")
        logger.info(f"  - Easy tokens per shard: {self.easy_token_target:,}")
        logger.info(f"  - Hard tokens per shard: {self.hard_token_target:,}")
        logger.info(f"  - Total tokens per shard: {self.easy_token_target + self.hard_token_target:,}")
        logger.info(f"  - Block size: {block_size}")
        logger.info(f"  - Cache directory: {self.cache_dir}")

    def _process_example(self, example: Dict[str, Any]) -> Optional[list]:
        """Process a single example with error handling"""
        try:
            if not isinstance(example, dict) or 'text' not in example:
                return None
                
            text = example.get('text', '')
            if not isinstance(text, str) or not text.strip():
                return None
            
            # Tokenize with truncation to avoid memory issues
            tokens = self.tokenizer.encode(
                text, 
                add_special_tokens=False,
                truncation=True,
                max_length=self.block_size * 10  # Allow some overhead
            )
            
            return tokens if tokens else None
            
        except Exception as e:
            logger.warning(f"[Rank {self.shard_id}] Tokenization error: {e}")
            return None

    def _create_dataset_with_retry(self, stage: str):
        """Create dataset with retry logic for handling transient failures"""
        for attempt in range(self.max_retries):
            try:
                return create_streaming_dataset(
                    stage, self.seed, self.shard_id, self.num_shards, self.cache_dir
                )
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"[Rank {self.shard_id}] Dataset creation failed (attempt {attempt + 1}), "
                        f"retrying in {self.retry_delay}s: {e}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"[Rank {self.shard_id}] Failed to create dataset after {self.max_retries} attempts")
                    raise

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Main iteration with proper sharding and error recovery"""
        stages = [
            ('easy', self.easy_token_target),
            ('hard', self.hard_token_target)
        ]
        
        buffer = []
        total_tokens_yielded = 0
        examples_processed = 0
        start_time = time.time()
        last_progress_time = start_time

        for stage, token_target in stages:
            if token_target == 0:
                continue
                
            dataset = self._create_dataset_with_retry(stage)
            
            curr_tokens = 0
            stage_examples = 0
            example_idx = 0
            
            logger.info(f"\n[Rank {self.shard_id}] Starting '{stage.upper()}' phase for {token_target:,} tokens...")
            
            for example in dataset:
                # Manual sharding: only process examples for this shard
                if example_idx % self.num_shards != self.shard_id:
                    example_idx += 1
                    continue
                example_idx += 1
                
                # Process example
                tokens = self._process_example(example)
                if not tokens:
                    continue
                
                buffer.extend(tokens)
                curr_tokens += len(tokens)
                total_tokens_yielded += len(tokens)
                examples_processed += 1
                stage_examples += 1

                # Yield complete blocks
                while len(buffer) >= self.block_size + 1:
                    # Create input and target sequences
                    chunk = buffer[:self.block_size + 1]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y
                    
                    # Slide window by half block for better coverage
                    buffer = buffer[self.block_size // 2:]
                
                # Progress reporting
                current_time = time.time()
                if current_time - last_progress_time > 30:  # Every 30 seconds
                    elapsed = current_time - start_time
                    tokens_per_sec = total_tokens_yielded / elapsed if elapsed > 0 else 0
                    progress_pct = (curr_tokens / token_target * 100) if token_target > 0 else 0
                    
                    logger.info(
                        f"[Rank {self.shard_id}] {stage}: {curr_tokens:,}/{token_target:,} tokens "
                        f"({progress_pct:.1f}%, {stage_examples:,} examples, {tokens_per_sec:.0f} tok/s)"
                    )
                    last_progress_time = current_time
                
                # Check if phase target reached
                if curr_tokens >= token_target:
                    logger.info(
                        f"[Rank {self.shard_id}] {stage.upper()} phase completed: "
                        f"{curr_tokens:,} tokens from {stage_examples:,} examples"
                    )
                    break
            
            # Clear buffer between stages
            buffer = []
            
            # Check if total target reached
            if total_tokens_yielded >= (self.easy_token_target + self.hard_token_target):
                break
        
        # Final statistics
        elapsed_total = time.time() - start_time
        logger.info(
            f"[Rank {self.shard_id}] Dataset iteration complete. "
            f"Total: {total_tokens_yielded:,} tokens from {examples_processed:,} examples "
            f"in {elapsed_total/60:.1f} minutes ({total_tokens_yielded/elapsed_total:.0f} tok/s)"
        )

def build_val_dataset(
    tokenizer, 
    block_size: int = 2048,
    val_size: int = 100,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> list:
    """
    Build a fixed validation dataset with proper error handling.
    Returns a list of (input_ids, labels) tensor pairs.
    """
    logger.info(f"Building validation dataset with {val_size} samples...")
    
    if cache_dir:
        os.environ["HF_DATASETS_CACHE"] = cache_dir
    
    try:
        # Load dataset
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        
        # Filter for medium-quality English texts
        def val_filter(example):
            try:
                text = example.get('text', '')
                return (
                    isinstance(example, dict) and
                    example.get('language', '') == 'en' and
                    example.get('int_score', 0) >= 3 and
                    400 < len(text) < 2000
                )
            except:
                return False
        
        dataset = dataset.filter(val_filter)
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
        
        # Collect validation samples
        buffer = []
        val_samples = []
        examples_processed = 0
        
        for example in dataset:
            try:
                if not isinstance(example, dict) or 'text' not in example:
                    continue
                    
                text = example['text']
                if not isinstance(text, str) or not text.strip():
                    continue
                
                # Tokenize
                tokens = tokenizer.encode(
                    text, 
                    add_special_tokens=False,
                    truncation=True,
                    max_length=block_size * 2
                )
                
                if not tokens:
                    continue
                
                buffer.extend(tokens)
                examples_processed += 1
                
                # Extract validation samples
                while len(buffer) >= block_size + 1 and len(val_samples) < val_size:
                    chunk = buffer[:block_size + 1]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    val_samples.append((x, y))
                    buffer = buffer[block_size // 2:]  # Slide by half
                
                if len(val_samples) >= val_size:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing validation example: {e}")
                continue
        
        logger.info(f"Built {len(val_samples)} validation samples from {examples_processed} examples")
        return val_samples
    
    except Exception as e:
        logger.error(f"Failed to build validation dataset: {e}")
        # Return empty list as fallback
        return []