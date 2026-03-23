"""
Transformer language model for bash terminal simulation.
Trained from scratch — no external dependencies beyond PyTorch.

Phase 2 architecture: standard transformer for within-command processing +
Gated DeltaNet state layer for cross-command memory. The state matrix S
is a key-value associative memory updated via the delta rule — it can store,
update, and erase distinct entries without destroying other information.

No token history crosses command boundaries — only the state matrix persists.
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
    # DeltaNet state
    state_dim: int = 256  # state matrix is (state_heads, head_dim, head_dim)
    state_heads: int = 8  # parallel state heads

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def state_head_dim(self) -> int:
        return self.state_dim // self.state_heads


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
# Attention (standard causal self-attention for within-command processing)
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
# Gated DeltaNet State Layer — cross-command memory
# ---------------------------------------------------------------------------

class GatedDeltaState(nn.Module):
    """Gated DeltaNet state layer for cross-command memory.

    Maintains a state matrix S (per head) that acts as key-value associative
    memory. Updated via the gated delta rule which supports:
    - Targeted writes: store specific key-value pairs without overwriting others
    - Selective forgetting: decay gate can erase specific entries
    - Associative reads: query the state to retrieve stored values

    State shape: (batch, num_heads, head_dim, head_dim)

    Used between commands: the transformer processes command tokens normally,
    then this layer updates S from the command and reads from S to inject
    context into the next command.
    """

    def __init__(self, config: BashTransformerConfig):
        super().__init__()
        self.num_heads = config.state_heads
        self.head_dim = config.state_head_dim
        self.state_dim = config.state_dim

        # Project from hidden_dim to state space
        self.W_k = nn.Linear(config.hidden_dim, config.state_dim, bias=False)
        self.W_v = nn.Linear(config.hidden_dim, config.state_dim, bias=False)
        self.W_q = nn.Linear(config.hidden_dim, config.state_dim, bias=False)

        # Gates
        self.W_beta = nn.Linear(config.hidden_dim, config.state_dim)  # update strength
        self.W_alpha = nn.Linear(config.hidden_dim, config.state_dim)  # decay
        # Bias alpha toward retaining (init positive = slow decay)
        nn.init.constant_(self.W_alpha.bias, 2.0)

        # Project state readout back to hidden_dim
        self.W_out = nn.Linear(config.state_dim, config.hidden_dim, bias=False)
        self.norm = RMSNorm(config.state_dim)
        self.gate_proj = nn.Linear(config.hidden_dim, config.state_dim, bias=False)

    def reset_state(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
        """Return zero-initialized state: (batch, num_heads, head_dim, head_dim)."""
        return torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,   # (batch, seq_len, hidden_dim)
        state: torch.Tensor,           # (batch, num_heads, head_dim, head_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence of tokens, updating state and producing readouts.

        Returns (output, new_state):
        - output: (batch, seq_len, hidden_dim) — state-informed context
        - new_state: (batch, num_heads, head_dim, head_dim) — updated state
        """
        bsz, seq_len, _ = hidden_states.shape
        h = self.num_heads
        d = self.head_dim

        # Project to state space: (batch, seq_len, num_heads, head_dim)
        k = self.W_k(hidden_states).view(bsz, seq_len, h, d)
        v = self.W_v(hidden_states).view(bsz, seq_len, h, d)
        q = self.W_q(hidden_states).view(bsz, seq_len, h, d)

        # Gates
        beta = torch.sigmoid(self.W_beta(hidden_states)).view(bsz, seq_len, h, d)
        alpha = torch.sigmoid(self.W_alpha(hidden_states)).view(bsz, seq_len, h, d)

        # Normalize keys for stable associative memory
        k = F.normalize(k, dim=-1)

        # Output gate for mixing state readout with input
        gate = self.gate_proj(hidden_states)  # (batch, seq_len, state_dim)

        # Process tokens recurrently through state
        outputs = []
        S = state  # (batch, num_heads, head_dim, head_dim)

        for t in range(seq_len):
            k_t = k[:, t]      # (batch, h, d)
            v_t = v[:, t]      # (batch, h, d)
            q_t = q[:, t]      # (batch, h, d)
            beta_t = beta[:, t]  # (batch, h, d)
            alpha_t = alpha[:, t]  # (batch, h, d)

            # Decay state: S = alpha * S
            S = S * alpha_t.unsqueeze(-1)  # broadcast over last dim

            # Delta update: S = S + beta * (v - S^T k) k^T
            # S^T k = prediction for this key
            pred = torch.einsum("bhij,bhj->bhi", S, k_t)  # (batch, h, d)
            error = v_t - pred  # (batch, h, d)
            # Outer product update
            update = torch.einsum("bhi,bhj->bhij", beta_t * error, k_t)
            S = S + update

            # Read from state: o = S q
            readout = torch.einsum("bhij,bhj->bhi", S, q_t)  # (batch, h, d)
            readout = readout.reshape(bsz, self.state_dim)  # (batch, state_dim)
            outputs.append(readout)

        # Stack outputs: (batch, seq_len, state_dim)
        context = torch.stack(outputs, dim=1)

        # Normalize, gate, project back to hidden_dim
        context = self.norm(context) * F.silu(gate)
        output = self.W_out(context)

        return output, S


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
# BashTransformer — transformer + DeltaNet state
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

        # DeltaNet state layers — one at input (read) and one at output (write)
        self.state_read = GatedDeltaState(config)
        self.state_write = GatedDeltaState(config)

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
        return cls(BashTransformerConfig())

    def reset_memory(self, batch_size: int, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
        """Return zero-initialized state: (batch, num_heads, head_dim, head_dim)."""
        return self.state_read.reset_state(batch_size, device, dtype)

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
            memory: (batch, num_heads, head_dim, head_dim) state matrix, or None
            labels: (batch, seq_len) target IDs, -100 for ignored positions
            loss_weights: (batch, seq_len) per-token loss weights

        Returns:
            dict with 'logits', 'memory', and optionally 'loss'
        """
        bsz, seq_len = input_ids.shape

        if memory is None:
            memory = self.reset_memory(bsz, input_ids.device,
                                       self.embed_tokens.weight.dtype)

        hidden_states = self.embed_tokens(input_ids)

        # Read from state: inject context from previous commands
        state_context, _ = self.state_read(hidden_states, memory)
        hidden_states = hidden_states + state_context

        # Get RoPE cos/sin
        cos, sin = self.rotary_emb(hidden_states)

        # Run through transformer layers (standard causal self-attention)
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer, hidden_states, cos, sin,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, cos, sin)

        # Write to state: update memory from this command's output
        _, new_memory = self.state_write(hidden_states, memory)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        result: dict[str, torch.Tensor] = {"logits": logits, "memory": new_memory}

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
    print("BashTransformer — Architecture Verification (DeltaNet)")
    print("=" * 60)

    config = BashTransformerConfig()
    print(f"\nConfig:")
    print(f"  vocab_size:          {config.vocab_size}")
    print(f"  hidden_dim:          {config.hidden_dim}")
    print(f"  num_layers:          {config.num_layers}")
    print(f"  num_heads:           {config.num_heads}")
    print(f"  head_dim:            {config.head_dim}")
    print(f"  ffn_intermediate:    {config.ffn_intermediate}")
    print(f"  state_dim:           {config.state_dim}")
    print(f"  state_heads:         {config.state_heads}")
    print(f"  state_head_dim:      {config.state_head_dim}")

    print(f"\nCreating model...")
    model = BashTransformer.from_config()

    total = model.num_parameters()
    print(f"\nArchitecture summary:")
    print(f"  Total parameters:         {total:>12,}")
    print(f"  Non-embedding parameters: {model.num_parameters(exclude_embeddings=True):>12,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Single command forward pass
    print(f"\n{'=' * 60}")
    print("Forward pass test (single command, seq_len=32)")
    print("=" * 60)
    dummy_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    dummy_labels = torch.randint(0, config.vocab_size, (2, 32), device=device)

    with torch.no_grad():
        out = model(dummy_ids, labels=dummy_labels)

    print(f"  logits shape: {out['logits'].shape}")
    print(f"  memory shape: {out['memory'].shape}")
    print(f"  loss:         {out['loss'].item():.4f}")

    # Multi-command session — state carries forward
    print(f"\n{'=' * 60}")
    print("Session test (5 commands, state carried forward)")
    print("=" * 60)
    memory = None
    for i in range(5):
        cmd_ids = torch.randint(0, config.vocab_size, (1, 20), device=device)
        with torch.no_grad():
            out = model(cmd_ids, memory=memory)
        memory = out["memory"]
        mem_norm = memory.norm().item()
        print(f"  Command {i+1}: logits {out['logits'].shape}, state norm {mem_norm:.2f}")

    # Full BPTT test
    print(f"\n{'=' * 60}")
    print("Full BPTT test (3 commands, single backward)")
    print("=" * 60)
    model.enable_gradient_checkpointing()
    model.train()
    memory = model.reset_memory(1, torch.device(device), model.embed_tokens.weight.dtype)
    total_loss = None
    for i in range(3):
        cmd = torch.randint(0, config.vocab_size, (1, 20), device=device)
        lbl = torch.randint(0, config.vocab_size, (1, 20), device=device)
        out = model(cmd, memory=memory, labels=lbl)
        memory = out["memory"]  # no detach
        n = (lbl != -100).sum().item()
        if total_loss is None:
            total_loss = out["loss"] * n
        else:
            total_loss = total_loss + out["loss"] * n
    total_loss.backward()
    print(f"  Full BPTT backward OK, loss={total_loss.item():.4f}")

    print(f"\n  State matrix: {config.state_heads} heads × {config.state_head_dim}×{config.state_head_dim}")
    print(f"  = {config.state_heads * config.state_head_dim * config.state_head_dim:,} floats of associative memory")

    print(f"\nAll checks passed.")
