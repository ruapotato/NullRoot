"""
Push recall ceiling: train on 12 words until perfect, then 24, then 36, etc.
Original 6L-512H-8h model. No step limit — train until it works.
"""

import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from tokenizer import BashTokenizer
from generator import _gen_syllable_name_rng


# ---------------------------------------------------------------------------
# Minimal autoregressive transformer
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).to(dtype)

class Attention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
    def forward(self, x):
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).contiguous().view(b, s, -1))

class FFN(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, hidden, heads, ffn_mult=3):
        super().__init__()
        self.n1 = RMSNorm(hidden)
        self.attn = Attention(hidden, heads)
        self.n2 = RMSNorm(hidden)
        self.ffn = FFN(hidden, hidden * ffn_mult)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab, hidden, layers, heads, ffn_mult=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads, ffn_mult) for _ in range(layers)])
        self.norm = RMSNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.head.weight = self.embed.weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, labels=None):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result


# ---------------------------------------------------------------------------
# Data + test
# ---------------------------------------------------------------------------

def make_recall_sample(num_words, tok, seed):
    rng = random.Random(seed)
    words = [_gen_syllable_name_rng(rng, 1) for _ in range(num_words)]
    parts = []
    for i in range(num_words):
        history = " ".join(words[:i + 1])
        parts.append(f"<prompt>{words[i]}<eoi><err>{history}<eor>")
    transcript = "".join(parts)
    ids = tok.encode(transcript)
    labels = list(ids)
    in_prompt = False
    for j, tid in enumerate(ids):
        if tid == tok.prompt_id:
            labels[j] = -100
            in_prompt = True
        elif in_prompt:
            labels[j] = -100
            if tid == tok.eoi_id:
                in_prompt = False
    return ids, labels


@torch.no_grad()
def test_recall(model, num_words, device, tok, num_trials=5):
    """Returns True if ALL trials pass perfectly."""
    for trial in range(num_trials):
        rng = random.Random(77777 + trial)
        words = [_gen_syllable_name_rng(rng, 1) for _ in range(num_words)]
        context_ids = []

        for i in range(num_words):
            cmd_ids = tok.encode(f"<prompt>{words[i]}<eoi>")
            context_ids.extend(cmd_ids)

            history = " ".join(words[:i + 1])
            expected = f"<err>{history}<eor>"
            expected_ids = tok.encode(expected)

            gen_ids = []
            ids = list(context_ids)
            for _ in range(len(expected_ids) + 20):
                input_t = torch.tensor([ids], dtype=torch.long, device=device)
                out = model(input_t)
                next_id = out["logits"][0, -1].argmax().item()
                ids.append(next_id)
                gen_ids.append(next_id)
                if next_id == tok.eor_id or next_id == tok.eos_id:
                    break

            if tok.decode(gen_ids) != expected:
                return False

            context_ids.extend(expected_ids)

    return True


# ---------------------------------------------------------------------------
# Main — push until it breaks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda")
    tok = BashTokenizer()

    model = MiniTransformer(tok.vocab_size, 512, 6, 8, 3).to(device).to(torch.bfloat16)
    print(f"Model: 6L 512H 8heads, {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Compiling...")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                                   weight_decay=0.1, fused=True)

    level = 12
    seed = 42
    total_steps = 0
    t0 = time.time()
    check_every = 1000

    while True:
        model.train()
        running_loss = 0
        steps_this_level = 0

        print(f"\n--- Level {level} words ---")

        while True:
            seed += 1
            rng = random.Random(seed)
            # Train on sequences from 1 up to current level
            num_words = rng.randint(max(1, level - 4), level)

            ids, labels = make_recall_sample(num_words, tok, seed)
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            labels_t = torch.tensor([labels], dtype=torch.long, device=device)

            with autocast("cuda", dtype=torch.bfloat16):
                out = model(ids_t, labels=labels_t)

            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += out["loss"].item()
            total_steps += 1
            steps_this_level += 1

            if steps_this_level % check_every == 0:
                avg = running_loss / check_every
                elapsed = time.time() - t0
                print(f"  step {total_steps:>7d} (lvl {steps_this_level:>6d}) | loss {avg:.4f} | {elapsed:.0f}s")
                running_loss = 0

                # Test
                model.eval()
                passed = test_recall(model, level, device, tok)
                model.train()

                if passed:
                    elapsed = time.time() - t0
                    print(f"  >>> PASSED level {level} at step {total_steps} ({elapsed:.0f}s)")
                    level *= 2
                    break
