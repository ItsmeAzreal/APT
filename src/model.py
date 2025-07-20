import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale

def apply_rope(q, k):
    B, nh, T, hd = q.size()
    freq_seq = torch.arange(0, hd, 2, device=q.device, dtype=q.dtype)
    inv_freq = 1.0 / (10000 ** (freq_seq / hd))
    pos = torch.arange(T, device=q.device, dtype=q.dtype)
    sinusoid_inp = torch.outer(pos, inv_freq)
    sin = sinusoid_inp.sin()[None, None, :, :]
    cos = sinusoid_inp.cos()[None, None, :, :]
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rope = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rope = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rope, k_rope

class FlashSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        q, k = apply_rope(q, k)
        # Replace flash_attn_func fallback if unavailable
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, is_causal=True
        )
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(0.05)
    def forward(self, x):
        x_proj, x_gate = self.fc1(x).chunk(2, dim=-1)
        return self.dropout(self.fc2(F.silu(x_proj) * x_gate))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, num_kv_heads, use_checkpoint=True):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = FlashSelfAttention(dim, num_heads, num_kv_heads)
        self.ln2 = RMSNorm(dim)
        self.ff = FeedForward(dim, hidden_dim)
        self.resid_dropout = nn.Dropout(0.05)
        self.use_checkpoint = use_checkpoint
    def forward(self, x):
        def block_fn(x_):
            x_ = x_ + self.resid_dropout(self.attn(self.ln1(x_)))
            x_ = x_ + self.resid_dropout(self.ff(self.ln2(x_)))
            return x_
        return block_fn(x)

class AnameeEmbedding(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(0.05)
    def forward(self, x):
        return self.dropout(self.token_embedding(x))

class AnameeModel(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, hidden_dim, num_layers, num_kv_heads, block_size, device=None):
        super().__init__()
        self.embed = AnameeEmbedding(dim, vocab_size)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim, num_kv_heads, use_checkpoint=(i >= 8))
            for i in range(num_layers)
        ])
        self.ln_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, device=device)
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
