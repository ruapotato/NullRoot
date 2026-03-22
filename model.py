"""
Transformer language model for bash terminal simulation.
Trained from scratch — no external dependencies beyond PyTorch.

Phase 2 architecture: each command processed in isolation. Filesystem state
lives in an explicit learned memory bank (256 slots x 512 dim). Each
transformer layer reads from and writes to memory via cross-attention heads.
No token history crosses command boundaries.
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
    vocab_size: int = 63
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_intermediate: int = 1536
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False
    # Memory bank
    num_memory_slots: int = 256
    mem_dim: int = 512  # matches hidden_dim

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
    """Precomputes and caches RoPE sin/cos tables."""

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
# Attention (self-attention over current command tokens only)
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
# Memory Bank
# ---------------------------------------------------------------------------

class MemoryBank(nn.Module):
    """Explicit learned memory bank for filesystem state.

    num_slots=256 slots of mem_dim=512. Learned initial state via nn.Parameter.
    Memory persists across commands within a session, detached between commands
    for truncated BPTT.
    """

    def __init__(self, config: BashTransformerConfig):
        super().__init__()
        self.num_slots = config.num_memory_slots
        self.mem_dim = config.mem_dim
        # Learned initial state — not zeros
        self.initial_state = nn.Parameter(
            torch.randn(config.num_memory_slots, config.mem_dim) * 0.02
        )

    def reset(self, batch_size: int) -> torch.Tensor:
        """Returns initial memory expanded to (batch, num_slots, mem_dim)."""
        return self.initial_state.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# Memory Head (one per transformer layer)
# ---------------------------------------------------------------------------

class MemoryHead(nn.Module):
    """Read from and write to memory via cross-attention.

    Read: x (command tokens) attends to memory slots — injects state info.
    Write: memory slots attend to x — updates state from command.
    Gated write prevents catastrophic overwrites.
    """

    def __init__(self, config: BashTransformerConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        # Read: x attends to memory
        self.read_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.read_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            batch_first=True,
            bias=False,
        )

        # Write: memory attends to x
        self.write_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.write_attn = nn.MultiheadAttention(
            embed_dim=config.mem_dim,
            num_heads=config.num_heads,
            batch_first=True,
            bias=False,
        )
        # Gated write: memory = memory + sigmoid(gate) * update
        self.write_gate = nn.Linear(config.mem_dim, config.mem_dim)

    def forward(
        self,
        x: torch.Tensor,          # (batch, seq, hidden)
        memory: torch.Tensor,      # (batch, num_slots, mem_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (updated_x, updated_memory)."""
        # Read from memory: x attends to memory (residual)
        x_normed = self.read_norm(x)
        read_out, _ = self.read_attn(
            query=x_normed, key=memory, value=memory,
            need_weights=False,
        )
        x = x + read_out

        # Write to memory: memory attends to x (gated residual)
        x_normed = self.write_norm(x)
        write_out, _ = self.write_attn(
            query=memory, key=x_normed, value=x_normed,
            need_weights=False,
        )
        gate = torch.sigmoid(self.write_gate(write_out))
        memory = memory + gate * write_out

        return x, memory


# ---------------------------------------------------------------------------
# Transformer Block (with memory)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: BashTransformerConfig):
        super().__init__()
        # Memory head (read → ... → write)
        self.memory_head = MemoryHead(config)
        # Self attention over current command only
        self.attn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attn = Attention(config)
        # FFN
        self.ffn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Read from memory
        hidden_states, memory = self.memory_head(hidden_states, memory)
        # 2. Self-attention over current command tokens
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), cos, sin)
        # 3. FFN
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states, memory


# ---------------------------------------------------------------------------
# BashTransformer — full model with memory
# ---------------------------------------------------------------------------

class BashTransformer(nn.Module):
    def __init__(self, config: BashTransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_positions=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        # Memory bank
        self.memory_bank = MemoryBank(config)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        # LM head
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
        config = BashTransformerConfig()
        return cls(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) token IDs for a single command
            memory: (batch, num_slots, mem_dim) or None (uses initial state)
            labels: (batch, seq_len) target IDs, -100 for ignored positions
            loss_weights: (batch, seq_len) per-token loss weights

        Returns:
            dict with 'logits', 'memory', and optionally 'loss'
        """
        bsz, seq_len = input_ids.shape

        if memory is None:
            memory = self.memory_bank.reset(bsz).to(
                dtype=input_ids.dtype if input_ids.is_floating_point()
                else self.embed_tokens.weight.dtype,
                device=input_ids.device,
            )

        hidden_states = self.embed_tokens(input_ids)

        cos, sin = self.rotary_emb(hidden_states)

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                hidden_states, memory = checkpoint(
                    layer, hidden_states, cos, sin, memory,
                    use_reentrant=False,
                )
            else:
                hidden_states, memory = layer(hidden_states, cos, sin, memory)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        result: dict[str, torch.Tensor] = {"logits": logits, "memory": memory}

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
    import sys

    print("=" * 60)
    print("BashTransformer — Architecture Verification (Phase 2)")
    print("=" * 60)

    config = BashTransformerConfig()
    print(f"\nConfig:")
    print(f"  vocab_size:          {config.vocab_size}")
    print(f"  hidden_dim:          {config.hidden_dim}")
    print(f"  num_layers:          {config.num_layers}")
    print(f"  num_heads:           {config.num_heads}")
    print(f"  head_dim:            {config.head_dim}")
    print(f"  ffn_intermediate:    {config.ffn_intermediate}")
    print(f"  num_memory_slots:    {config.num_memory_slots}")
    print(f"  mem_dim:             {config.mem_dim}")

    print(f"\nCreating model...")
    model = BashTransformer.from_config()

    total = model.num_parameters()
    print(f"\nArchitecture summary:")
    print(f"  Total parameters:         {total:>12,}")
    print(f"  Non-embedding parameters: {model.num_parameters(exclude_embeddings=True):>12,}")

    # Forward pass test — single command
    print(f"\n{'=' * 60}")
    print("Forward pass test (single command, seq_len=64)")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    dummy_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)
    dummy_labels = torch.randint(0, config.vocab_size, (2, 64), device=device)

    with torch.no_grad():
        out = model(dummy_ids, labels=dummy_labels)

    print(f"  logits shape: {out['logits'].shape}")
    print(f"  memory shape: {out['memory'].shape}")
    print(f"  loss:         {out['loss'].item():.4f}")
    expected_loss = math.log(config.vocab_size)
    print(f"  expected loss (random): {expected_loss:.4f}")

    # Multi-command session test — memory carries forward
    print(f"\n{'=' * 60}")
    print("Session test (3 commands, memory carried forward)")
    print("=" * 60)
    memory = None
    for i in range(3):
        cmd_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
        with torch.no_grad():
            out = model(cmd_ids, memory=memory)
        memory = out["memory"]
        print(f"  Command {i+1}: logits {out['logits'].shape}, memory {memory.shape}")

    # Gradient checkpointing test
    print(f"\n{'=' * 60}")
    print("Gradient checkpointing + backward test")
    print("=" * 60)
    model.enable_gradient_checkpointing()
    model.train()
    memory = model.memory_bank.reset(1).to(device)
    cmd1 = torch.randint(0, config.vocab_size, (1, 32), device=device)
    lbl1 = torch.randint(0, config.vocab_size, (1, 32), device=device)
    out1 = model(cmd1, memory=memory, labels=lbl1)
    out1["loss"].backward()
    print(f"  Forward + backward OK, loss={out1['loss'].item():.4f}")

    # Detach and second command
    memory2 = out1["memory"].detach()
    cmd2 = torch.randint(0, config.vocab_size, (1, 32), device=device)
    lbl2 = torch.randint(0, config.vocab_size, (1, 32), device=device)
    model.zero_grad()
    out2 = model(cmd2, memory=memory2, labels=lbl2)
    out2["loss"].backward()
    print(f"  Second command OK, loss={out2['loss'].item():.4f}")

    print(f"\nAll checks passed.")
