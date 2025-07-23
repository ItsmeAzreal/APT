# src/model.py - Fixed model with proper initialization and flash attention support

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Try to import flash attention
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Flash Attention not available, using PyTorch native attention")

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale

def apply_rope(q: torch.Tensor, k: torch.Tensor, base: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to query and key tensors"""
    B, nh, T, hd = q.size()
    device = q.device
    
    # Create position indices
    pos = torch.arange(T, device=device, dtype=q.dtype)
    
    # Create frequency bands
    freq_seq = torch.arange(0, hd, 2, device=device, dtype=q.dtype)
    inv_freq = 1.0 / (base ** (freq_seq / hd))
    
    # Create sinusoidal embeddings
    sinusoid_inp = torch.outer(pos, inv_freq)
    sin = sinusoid_inp.sin()[None, None, :, :]
    cos = sinusoid_inp.cos()[None, None, :, :]
    
    # Apply rotation
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    
    q_rope = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rope = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    
    return q_rope, k_rope

class GroupedQueryAttention(nn.Module):
    """Multi-head attention with grouped query attention support"""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # GQA: separate projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = dropout
        self.use_flash = FLASH_AVAILABLE and dropout == 0.0  # Flash attn doesn't support dropout during inference
    
    def forward(self, x: torch.Tensor, use_cache: bool = False, past_kv: Optional[Tuple] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = apply_rope(q, k)
        
        # Expand K, V for GQA if needed
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Apply attention
        if self.use_flash and self.training:
            # Flash attention expects (B, T, nh, hd) format
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func(q, k, v, dropout_p=self.dropout, causal=True)
            out = out.view(B, T, C)
        else:
            # PyTorch native attention
            out = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
            out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)

class FeedForward(nn.Module):
    """SwiGLU feed-forward network"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj, x_gate = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(x_proj) * x_gate
        x = self.fc2(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization"""
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        hidden_dim: int, 
        num_kv_heads: int, 
        dropout: float = 0.0,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, dropout)
        self.ln2 = RMSNorm(dim)
        self.ff = FeedForward(dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = residual + x
        
        # Feedforward block
        residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = residual + x
        
        return x

class AnameeModel(nn.Module):
    """Main language model with proper initialization"""
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 640,
        num_heads: int = 10,
        hidden_dim: int = 2560,
        num_layers: int = 24,
        num_kv_heads: int = 4,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        use_checkpoint: bool = True,
        checkpoint_start_layer: int = 8
    ):
        super().__init__()
        
        # Store config
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim, 
                num_heads, 
                hidden_dim, 
                num_kv_heads, 
                dropout,
                use_checkpoint=(use_checkpoint and i >= checkpoint_start_layer)
            )
            for i in range(num_layers)
        ])
        
        # Output
        self.ln_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass with optional loss calculation"""
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        
        # Transformer blocks
        for layer in self.layers:
            if layer.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        
        # Output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Return object with loss attribute for compatibility
            class Output:
                def __init__(self, loss, logits):
                    self.loss = loss
                    self.logits = logits
            
            return Output(loss, logits)
        
        return logits

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024