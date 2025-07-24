# src/validate.py - Fixed validation function with proper autocast and batch handling

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

@torch.no_grad()
def validate(
    model: nn.Module, 
    val_loader, 
    device: torch.device, 
    val_steps: Optional[int] = None,
    use_bf16: bool = True
) -> float:
    """
    Run validation and return average loss.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Device to run validation on
        val_steps: Number of validation steps (None = full validation)
        use_bf16: Whether to use bfloat16 precision
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    steps = 0
    
    # Determine precision
    autocast_dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
    
    # Create progress bar
    pbar = tqdm(
        val_loader, 
        total=val_steps if val_steps else len(val_loader),
        desc="Validation",
        leave=False,
        disable=not logger.isEnabledFor(logging.INFO)
    )
    
    try:
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                input_ids, labels = batch
            else:
                logger.error(f"Unexpected batch format: {type(batch)}")
                continue
            
            # Convert numpy arrays to tensors if needed
            if isinstance(input_ids, np.ndarray):
                input_ids = torch.from_numpy(input_ids).long()
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long()
            
            # Move to device
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            if device.type == 'cuda':
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    outputs = model(input_ids, labels=labels)
                    
                    # Handle different output formats
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    else:
                        # Compute loss manually if not provided
                        logits = outputs
                        loss = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100
                        )
            else:
                # CPU doesn't use autocast
                outputs = model(input_ids, labels=labels)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    logits = outputs
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
            
            # Accumulate loss
            batch_tokens = labels.ne(-100).sum().item()  # Count non-padding tokens
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            steps += 1
            
            # Update progress bar
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Break if reached desired steps
            if val_steps and steps >= val_steps:
                break
                
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise
    finally:
        model.train()
        pbar.close()
    
    # Calculate average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    logger.info(f"Validation complete: {steps} steps, avg loss: {avg_loss:.4f}")
    
    return avg_loss