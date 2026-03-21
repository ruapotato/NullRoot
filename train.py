"""
Training loop for BashTransformer.

Single-GPU (RTX 3090 24GB) training with:
- Live data generation (no disk)
- bf16 mixed precision
- Gradient checkpointing
- Cosine LR schedule with warmup
- Periodic validation
- Checkpointing
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from tokenizer import BashTokenizer
from model import BashTransformer, BashTransformerConfig
from dataset import BashSessionDataset, BashValidationDataset


def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: BashTransformer, val_dataset: BashValidationDataset,
             device: torch.device) -> dict:
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(len(val_dataset)):
        token_ids, labels = val_dataset[i]
        token_ids = token_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            out = model(token_ids, labels=labels)

        n_tokens = (labels != -100).sum().item()
        total_loss += out["loss"].item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow

    model.train()
    return {"val_loss": avg_loss, "val_ppl": perplexity, "val_tokens": total_tokens}


def save_checkpoint(model: BashTransformer, optimizer: torch.optim.Optimizer,
                    step: int, loss: float, path: str):
    """Save model checkpoint."""
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": model.config,
    }, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Model ---
    config = BashTransformerConfig(gradient_checkpointing=True)
    model = BashTransformer(config).to(device)
    model.train()

    # bf16 for all parameters
    model = model.to(torch.bfloat16)

    # torch.compile for speed
    print("Compiling model with torch.compile...")
    model = torch.compile(model)
    print("Compilation done (will finish on first forward pass)")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {param_count:,} parameters")
    print(f"Context window: {args.seq_len:,} tokens")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum} steps")
    effective_batch = args.batch_size * args.grad_accum
    print(f"Effective batch size: {effective_batch}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )

    # --- Data ---
    print(f"\nStarting live data generation ({args.data_workers} threads)...")
    train_ds = BashSessionDataset(
        seq_len=args.seq_len,
        buffer_size=args.buffer_size,
        workers=args.data_workers,
        min_ops=300,
        target_ops=800,
        error_rate=0.05,
        base_seed=args.seed,
    )
    train_iter = iter(train_ds)

    val_ds = None
    val_path = os.path.join(os.path.dirname(__file__) or ".", "data", "validation.jsonl")
    if os.path.exists(val_path):
        val_ds = BashValidationDataset(val_path, seq_len=args.seq_len)
        print(f"Validation: {len(val_ds)} samples from {val_path}")
    else:
        print("No validation data found, skipping eval")

    # --- Checkpoint dir ---
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Log file ---
    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "a")

    def log(entry: dict):
        entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    # --- Resume from checkpoint ---
    start_step = 0
    if args.resume:
        ckpt_path = args.resume
        if os.path.exists(ckpt_path):
            print(f"\nResuming from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt["step"]
            print(f"Resumed at step {start_step}")

    # --- Training loop ---
    print(f"\n{'='*70}")
    print(f"  Training starts at step {start_step}")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Warmup: {args.warmup_steps:,} steps")
    print(f"  LR: {args.max_lr} -> {args.min_lr}")
    print(f"  Eval every: {args.eval_every:,} steps")
    print(f"  Save every: {args.save_every:,} steps")
    print(f"{'='*70}\n")

    running_loss = 0.0
    running_trained_tokens = 0
    total_tokens_seen = 0
    t0 = time.time()
    step_t0 = time.time()

    for step in range(start_step, args.max_steps):
        # Set learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps,
                    args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_trained = 0

        for micro_step in range(args.grad_accum):
            token_ids, labels = next(train_iter)
            token_ids = token_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                out = model(token_ids, labels=labels)
                loss = out["loss"] / args.grad_accum

            loss.backward()

            n_trained = (labels != -100).sum().item()
            accum_loss += out["loss"].item()
            accum_trained += n_trained

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        # Stats
        avg_loss = accum_loss / args.grad_accum
        running_loss += avg_loss
        running_trained_tokens += accum_trained
        total_tokens_seen += args.seq_len * args.grad_accum

        # Log every N steps
        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - step_t0
            steps_done = args.log_every
            steps_per_sec = steps_done / elapsed
            tokens_per_sec = (args.seq_len * args.grad_accum * steps_done) / elapsed
            trained_per_sec = running_trained_tokens / elapsed
            avg_running = running_loss / steps_done
            ppl = math.exp(min(avg_running, 20))
            total_elapsed = time.time() - t0
            eta = (args.max_steps - step - 1) / max(steps_per_sec, 0.001)

            # GPU memory
            mem_used = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
            mem_peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0

            entry = {
                "step": step + 1,
                "loss": round(avg_running, 4),
                "ppl": round(ppl, 2),
                "lr": round(lr, 8),
                "grad_norm": round(grad_norm.item(), 4),
                "steps_per_sec": round(steps_per_sec, 3),
                "tokens_per_sec": round(tokens_per_sec),
                "trained_tokens_per_sec": round(trained_per_sec),
                "trained_tokens": running_trained_tokens,
                "total_tokens": total_tokens_seen,
                "elapsed_sec": round(total_elapsed, 1),
                "eta_sec": round(eta, 0),
                "gpu_mem_gb": round(mem_used, 2),
                "gpu_peak_gb": round(mem_peak, 2),
            }
            log(entry)

            eta_min = eta / 60
            print(f"  step {step+1:>6d}/{args.max_steps} | loss {avg_running:.4f} | ppl {ppl:>8.2f} | "
                  f"lr {lr:.2e} | gnorm {grad_norm.item():.2f} | "
                  f"{steps_per_sec:.2f} steps/s | {tokens_per_sec:,.0f} tok/s | "
                  f"gpu {mem_used:.1f}/{mem_peak:.1f}GB | "
                  f"eta {eta_min:.0f}m")

            running_loss = 0.0
            running_trained_tokens = 0
            step_t0 = time.time()

        # Validation
        if val_ds and (step + 1) % args.eval_every == 0:
            print(f"\n  Evaluating at step {step+1}...")
            val_metrics = evaluate(model, val_ds, device)
            log({"step": step + 1, "type": "eval", **val_metrics})
            print(f"  val_loss={val_metrics['val_loss']:.4f}  "
                  f"val_ppl={val_metrics['val_ppl']:.2f}\n")

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_path = str(ckpt_dir / f"step_{step+1:06d}.pt")
            save_checkpoint(model, optimizer, step + 1, avg_loss, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = str(ckpt_dir / "final.pt")
    save_checkpoint(model, optimizer, args.max_steps, avg_loss, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    # Final eval
    if val_ds:
        print("Running final evaluation...")
        val_metrics = evaluate(model, val_ds, device)
        log({"step": args.max_steps, "type": "final_eval", **val_metrics})
        print(f"  val_loss={val_metrics['val_loss']:.4f}  "
              f"val_ppl={val_metrics['val_ppl']:.2f}")

    elapsed_total = time.time() - t0
    print(f"\nTotal time: {elapsed_total/3600:.1f} hours")
    print(f"Total tokens: {total_tokens_seen:,}")

    train_ds.stop()
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BashTransformer")
    parser.add_argument("--seq-len", type=int, default=65536,
                        help="Context window size")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Micro batch size (per step)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=200,
                        help="LR warmup steps")
    parser.add_argument("--max-lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5,
                        help="Minimum learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--eval-every", type=int, default=250,
                        help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log metrics every N steps")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--data-workers", type=int, default=4,
                        help="Data generation threads")
    parser.add_argument("--buffer-size", type=int, default=16,
                        help="Data buffer size (prefetched samples)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args)
