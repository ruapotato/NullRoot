"""
Transformer language model for bash terminal simulation.
Trained from scratch — no external dependencies beyond PyTorch.

Phase 2 architecture: pure autoregressive transformer with state patches.
Each command is processed with the full state string prepended. The model
outputs the response + a state patch. The patch is applied to produce
the state for the next command. No memory modules — just tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BashTransformerConfig:
    vocab_size: int = 96  # 19 commands + 10 control + 26 letters + 10 digits + 17 punct + 2 logical + 9 special + 3 chrome
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_intermediate: int = 1536
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 131072, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len <= self._cache_len and self._cos_cache is not None:
            return
        self._cache_len = max(seq_len, 4096)
        t = torch.arange(self._cache_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        seq_len = x.shape[-2] if position_ids is None else int(position_ids.max()) + 1
        self._build_cache(seq_len, x.device, x.dtype)
        if position_ids is not None:
            cos = self._cos_cache[position_ids]
            sin = self._sin_cache[position_ids]
        else:
            cos = self._cos_cache[:seq_len]
            sin = self._sin_cache[:seq_len]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, config: BashTransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, config: BashTransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_intermediate, bias=False)
        self.down_proj = nn.Linear(config.ffn_intermediate, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: BashTransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), cos, sin)
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# BashTransformer — pure autoregressive
# ---------------------------------------------------------------------------

class BashTransformer(nn.Module):
    def __init__(self, config: BashTransformerConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_positions=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._gradient_checkpointing = config.gradient_checkpointing

        self._init_weights()

        n_params = self.num_parameters()
        n_params_non_emb = self.num_parameters(exclude_embeddings=True)
        print(f"BashTransformer initialized: {n_params:,} params ({n_params_non_emb:,} non-embedding)")

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self._gradient_checkpointing = False

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        params = set()
        for name, p in self.named_parameters():
            if exclude_embeddings and "embed_tokens" in name:
                continue
            params.add((p.data_ptr(), p.numel()))
        return sum(n for _, n in params)

    @classmethod
    def from_config(cls) -> "BashTransformer":
        return cls(BashTransformerConfig())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        cos, sin = self.rotary_emb(hidden_states)

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer, hidden_states, cos, sin,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, cos, sin)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        result: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            if loss_weights is not None:
                shift_weights = loss_weights[:, 1:].contiguous()
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                weighted = per_token_loss * shift_weights.view(-1)
                mask = shift_labels.view(-1) != -100
                loss = weighted[mask].sum() / shift_weights.view(-1)[mask].sum().clamp(min=1)
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            result["loss"] = loss

        return result


# ---------------------------------------------------------------------------
# Main — sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("BashTransformer — Architecture Verification (State Patch)")
    print("=" * 60)

    config = BashTransformerConfig()
    print(f"\nConfig:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    print(f"\nCreating model...")
    model = BashTransformer.from_config()

    total = model.num_parameters()
    print(f"\n  Total parameters:         {total:>12,}")
    print(f"  Non-embedding parameters: {model.num_parameters(exclude_embeddings=True):>12,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dummy = torch.randint(0, config.vocab_size, (2, 128), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 128), device=device)

    with torch.no_grad():
        out = model(dummy, labels=labels)
    print(f"\n  Forward pass: logits {out['logits'].shape}, loss {out['loss'].item():.4f}")
    print(f"  Expected loss (random): {math.log(config.vocab_size):.4f}")
    print(f"\nAll checks passed.")
