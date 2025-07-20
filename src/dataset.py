from datasets import load_dataset
import torch

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

# === Streaming Loader with Curriculum ===

def filtered_stream(stage, seed, rank, buffer_size=100_000):
    """
    Loads the streaming HuggingFaceFW/fineweb-edu dataset,
    applies shuffling (buffered) and stage-specific filtering.
    """
    print(f"Loading HuggingFaceFW/fineweb-edu ({stage} curriculum stage)...")
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )
    # Buffered shuffle for better randomness (increase if RAM allows)
    stream = stream.shuffle(seed=seed + rank, buffer_size=buffer_size)
    if stage == 'easy':
        return filter(easy_filter, stream)
    else:
        return filter(hard_filter, stream)

class Curriculum2BTokenDataset(torch.utils.data.IterableDataset):
    """
    Streaming IterableDataset for curriculum-based LLM pretraining.
    - Streams 'easy' then 'hard' phase in order.
    - Stops when 2B tokens reached (or user-specified limits).
    """
    def __init__(self, tokenizer, block_size, easy_token_target, hard_token_target, seed=42, rank=0):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.easy_token_target = easy_token_target
        self.hard_token_target = hard_token_target
        self.seed = seed
        self.rank = rank

    def __iter__(self):
        stages = [
            ('easy', self.easy_token_target),
            ('hard', self.hard_token_target)
        ]
        buffer = []
        total_tokens_yielded = 0

        for stage, token_target in stages:
            curr_stream = filtered_stream(stage, self.seed, self.rank)
            curr_tokens = 0
            print(f"\n[CURRICULUM] Starting '{stage.upper()}' phase for {token_target:,} tokens...")
            for example in curr_stream:
                # Tokenize on the fly; can be slow but RAM-efficient.
                tokens = self.tokenizer.encode(example['text'], add_special_tokens=False)
                if not tokens:
                    continue
                buffer.extend(tokens)
                curr_tokens += len(tokens)
                total_tokens_yielded += len(tokens)

                # Yield block_size training chunks as (x, y) pairs
                while len(buffer) >= self.block_size + 1:
                    x_chunk = buffer[:self.block_size]
                    y_chunk = buffer[1:self.block_size + 1]
                    x = torch.tensor(x_chunk, dtype=torch.long)
                    y = torch.tensor(y_chunk, dtype=torch.long)
                    yield x, y
                    buffer = buffer[self.block_size:]
                    # End phase when token target met
                    if curr_tokens >= token_target:
                        print(f"[CURRICULUM] {stage.upper()} phase reached {curr_tokens:,} tokens.")
                        break
            buffer = []
            # End stream if total curriculum tokens reached
            if total_tokens_yielded >= (self.easy_token_target + self.hard_token_target):
                print("[CURRICULUM] Total 2B tokens reached. Stopping dataset stream.")
                break

# === Validation Dataset Construction ===

def build_val_dataset(tokenizer, block_size, val_size=1000, seed=42, buffer_size=100_000):
    """
    Build a small, fixed-size validation dataset by streaming the first N English, quality-filtered samples.
    - Returns a list of (x, y) tensor pairs.
    """
    print(f"[VAL DATASET] Building validation set with {val_size} samples...")
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )
    # Only keep English and int_score >= 3 examples
    stream = filter(lambda ex: ex.get('language', '') == 'en' and ex.get('int_score', 0) >= 3, stream)
    buffer = []
    val_samples = 0
    xs, ys = [], []
    for example in stream:
        tokens = tokenizer.encode(example['text'], add_special_tokens=False)
        buffer.extend(tokens)
        # Yield block_size chunks
        while len(buffer) >= block_size + 1:
            x_chunk = buffer[:block_size]
            y_chunk = buffer[1:block_size + 1]
            xs.append(torch.tensor(x_chunk, dtype=torch.long))
            ys.append(torch.tensor(y_chunk, dtype=torch.long))
            buffer = buffer[block_size:]
            val_samples += 1
            if val_samples >= val_size:
                print(f"[VAL DATASET] Built {val_samples} validation samples.")
                return list(zip(xs, ys))
    print(f"[VAL DATASET] Built {val_samples} validation samples (stream ended).")
    return list(zip(xs, ys))

# === End of dataset.py ===
