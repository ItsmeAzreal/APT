# src/model.py - Lightning-120M Ultra-Efficient Architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

# Try to import flash attention
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

@dataclass
class LightningConfig:
    """Configuration for Lightning model"""
    vocab_size: int = 32768
    dim: int = 768
    num_heads: int = 12
    num_kv_heads: int = 3
    hidden_dim: int = 2688
    num_layers: int = 18
    max_seq_len: int = 4096
    rope_scaling_factor: float = 2.0
    activation: str = "swish"
    dropout: float = 0.0
    embedding_dropout: float = 0.1
    use_checkpoint: bool = True
    checkpoint_layers: list = None

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster than LayerNorm"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings with scaling support"""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000, scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position indices
        positions = torch.arange(max_seq_len).float()
        if scaling_factor != 1.0:
            positions = positions / scaling_factor
        freqs = torch.outer(positions, self.inv_freq)
        
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def apply_rotary_embeddings(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors"""
    # Reshape for rotary embeddings
    q_rot = q.reshape(*q.shape[:-1], -1, 2)
    k_rot = k.reshape(*k.shape[:-1], -1, 2)
    
    # Apply rotation
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_out = torch.stack([
        q_rot[..., 0] * cos - q_rot[..., 1] * sin,
        q_rot[..., 0] * sin + q_rot[..., 1] * cos
    ], dim=-1).flatten(-2)
    
    k_out = torch.stack([
        k_rot[..., 0] * cos - k_rot[..., 1] * sin,
        k_rot[..., 0] * sin + k_rot[..., 1] * cos
    ], dim=-1).flatten(-2)
    
    return q_out, k_out

class LightningAttention(nn.Module):
    """Ultra-efficient attention with aggressive GQA compression"""
    def __init__(self, config: LightningConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        
        # Rotary embeddings
        self.rotary = RotaryPositionEmbedding(
            self.head_dim, 
            config.max_seq_len, 
            scaling_factor=config.rope_scaling_factor
        )
        
        self.dropout = config.dropout
        self.use_flash = FLASH_AVAILABLE and config.dropout == 0.0
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_embeddings(q, k, cos, sin)
        
        # Repeat K,V for grouped query attention
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Attention computation
        if self.use_flash and self.training:
            # Use Flash Attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func(q, k, v, dropout_p=self.dropout, causal=True)
            out = out.view(B, T, C)
        else:
            # Standard attention with memory optimization
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=self.scale
            )
            out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(out)

class LightningFFN(nn.Module):
    """Optimized FFN with Swish activation"""
    def __init__(self, config: LightningConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = config.activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swish":
            gate = F.silu(self.gate_proj(x))
        else:
            gate = F.gelu(self.gate_proj(x))
        
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return self.dropout(x)

class LightningBlock(nn.Module):
    """Transformer block optimized for efficiency"""
    def __init__(self, config: LightningConfig, layer_idx: int):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.attention = LightningAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = LightningFFN(config)
        self.layer_idx = layer_idx
        
        # Gradient checkpointing for specific layers
        self.use_checkpoint = (
            config.use_checkpoint and 
            config.checkpoint_layers and 
            layer_idx in config.checkpoint_layers
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention block with residual
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, mask)
        x = residual + x
        
        # FFN block with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

class LightningModel(nn.Module):
    """Lightning-120M: Ultra-efficient language model"""
    def __init__(self, config: LightningConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings with special initialization
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            LightningBlock(config, i) for i in range(config.num_layers)
        ])
        
        # Output
        self.ln_f = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled initialization for stability
        for pn, p in self.named_parameters():
            if 'o_proj.weight' in pn or 'down_proj.weight' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))
    
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
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional loss calculation"""
        B, T = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = self.embedding_dropout(x)
        
        # Transformer blocks
        for i, layer in enumerate(self.layers):
            if layer.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, mask)
            else:
                x = layer(x, mask)
        
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

def create_lightning_model(
    vocab_size: int = 32768,
    dim: int = 768,
    num_heads: int = 12,
    num_kv_heads: int = 3,
    hidden_dim: int = 2688,
    num_layers: int = 18,
    max_seq_len: int = 4096,
    **kwargs
) -> LightningModel:
    """Create Lightning model with specified configuration"""
    config = LightningConfig(
        vocab_size=vocab_size,
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        checkpoint_layers=list(range(6, 18, 3)),
        **kwargs
    )
    return LightningModel(config)

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024

# Alias for compatibility
AnameeModel = create_lightning_model