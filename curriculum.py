"""
Curriculum training for BashTransformer.

Stages introduce commands progressively. The model must master each stage's
validation set before advancing. Once a command is introduced, it stays
in the training mix forever.

Stages:
  1. mkdir, cd, ls           — directory creation, navigation, listing
  2. + pwd                   — path tracking
  3. + touch                 — file creation
  4. + echo >                — file writing
  5. + cat                   — file reading
  6. + echo >>               — appending
  7. + rm                    — file removal
  8. + errors                — intentional invalid commands
"""

import os
import sys
import time
import math
import json
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.amp import autocast

from tokenizer import BashTokenizer
from model import BashTransformer, BashTransformerConfig
from dataset import BashSessionDataset, BashValidationDataset, _build_labels
from generator import SessionGenerator


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES = [
    {
        "name": "Stage 1: mkdir + cd + ls",
        "commands": {"mkdir", "cd_child", "cd_up", "cd_abs", "ls"},
        "error_rate": 0.0,
    },
    {
        "name": "Stage 2: + pwd",
        "commands": {"mkdir", "cd_child", "cd_up", "cd_abs", "ls", "pwd"},
        "error_rate": 0.0,
    },
    {
        "name": "Stage 3: + touch",
        "commands": {"mkdir", "cd_child", "cd_up", "cd_abs", "ls", "pwd", "touch"},
        "error_rate": 0.0,
    },
    {
        "name": "Stage 4: + echo >",
        "commands": {"mkdir", "cd_child", "cd_up", "cd_abs", "ls", "pwd", "touch", "echo_write"},
        "error_rate": 0.0,
    },
    {
        "name": "Stage 5: + cat",
        "commands": {"mkdir", "cd_child", "cd_up", "cd_abs", "ls", "pwd", "touch", "echo_write", "cat"},
        "error_rate": 0.0,
    },
    {
        "name": "Stage 6: + echo >>",
        "commands": {"mkdir", "cd_child", "cd_up", "cd_abs", "ls", "pwd", "touch", "echo_write", "cat", "echo_append"},
        "error_rate": 0.0,
    },
    {
        "name": "Stage 7: + rm",
        "commands": {"mkdir", "cd_child", "cd_up", "cd_abs", "ls", "pwd", "touch", "echo_write", "cat", "echo_append", "rm"},
        "error_rate": 0.0,
    },
    {
        "name": "Stage 8: + errors",
        "commands": {"mkdir", "cd_child", "cd_up", "cd_abs", "ls", "pwd", "touch", "echo_write", "cat", "echo_append", "rm", "errors"},
        "error_rate": 0.05,
    },
]


# ---------------------------------------------------------------------------
# Validation set generation for each stage
# ---------------------------------------------------------------------------

def generate_stage_validation(stage_idx: int, num_sessions: int = 10,
                              min_ops: int = 200, seed: int = 99999) -> list[str]:
    """Generate validation transcripts for a curriculum stage."""
    stage = STAGES[stage_idx]
    commands = stage["commands"]
    error_rate = stage["error_rate"]
    transcripts = []

    for i in range(num_sessions):
        gen = SessionGenerator(
            min_ops=min_ops,
            target_ops=300,
            error_rate=error_rate,
            seed=seed + i,
            commands=commands,
        )
        transcript = gen.generate()
        transcripts.append(transcript)

    return transcripts


class StageValidationDataset(torch.utils.data.Dataset):
    """Validation dataset generated on-the-fly for a curriculum stage."""

    def __init__(self, stage_idx: int, seq_len: int = 65536,
                 num_sessions: int = 10, seed: int = 99999):
        self.seq_len = seq_len
        self.tokenizer = BashTokenizer()
        self.pad_id = self.tokenizer.pad_id

        transcripts = generate_stage_validation(
            stage_idx, num_sessions=num_sessions, seed=seed,
        )
        self.samples = []
        for t in transcripts:
            try:
                ids = self.tokenizer.encode(t)
                self.samples.append(ids)
            except ValueError:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = list(self.samples[idx])
        if len(ids) >= self.seq_len:
            ids = ids[:self.seq_len]

        labels = _build_labels(ids, self.tokenizer)

        pad_len = self.seq_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_dataset, device):
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
    ppl = math.exp(min(avg_loss, 20))
    model.train()
    return {"val_loss": avg_loss, "val_ppl": ppl, "val_tokens": total_tokens}


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Curriculum training
# ---------------------------------------------------------------------------

@dataclass
class CurriculumConfig:
    seq_len: int = 65536
    grad_accum: int = 4
    steps_per_stage: int = 2000       # max steps before forced gate check
    gate_threshold: float = 1.5       # val loss must be below this to advance
    gate_check_every: int = 100       # check gate this often
    warmup_steps: int = 100           # per stage
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    max_grad_norm: float = 1.0
    log_every: int = 10
    data_workers: int = 4
    buffer_size: int = 16
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    resume_stage: int = 0             # start from this stage
    resume_ckpt: str | None = None


def train_curriculum(cfg: CurriculumConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Model ---
    model_config = BashTransformerConfig(gradient_checkpointing=True)
    model = BashTransformer(model_config).to(device).to(torch.bfloat16)

    print("Compiling model...")
    model = torch.compile(model)

    # --- Optimizer (persists across stages) ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.max_lr,
        betas=(0.9, 0.95), weight_decay=0.1, fused=True,
    )

    # --- Resume ---
    if cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt):
        print(f"Resuming from {cfg.resume_ckpt}...")
        ckpt = torch.load(cfg.resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded checkpoint at global step {ckpt.get('global_step', '?')}")

    # --- Log ---
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "a")

    def log(entry):
        entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    global_step = 0
    total_tokens = 0
    t0_global = time.time()

    # --- Stage loop ---
    for stage_idx in range(cfg.resume_stage, len(STAGES)):
        stage = STAGES[stage_idx]
        print(f"\n{'='*70}")
        print(f"  {stage['name']}")
        print(f"  Commands: {sorted(stage['commands'])}")
        print(f"  Error rate: {stage['error_rate']}")
        print(f"{'='*70}\n")

        log({"type": "stage_start", "stage": stage_idx, "name": stage["name"],
             "global_step": global_step})

        # --- Stage validation set ---
        val_ds = StageValidationDataset(stage_idx, seq_len=cfg.seq_len)
        print(f"  Validation: {len(val_ds)} sessions")

        # --- Stage training data ---
        train_ds = BashSessionDataset(
            seq_len=cfg.seq_len,
            buffer_size=cfg.buffer_size,
            workers=cfg.data_workers,
            min_ops=300, target_ops=800,
            error_rate=stage["error_rate"],
            base_seed=cfg.seed + stage_idx * 1000,
            commands=stage["commands"],
        )
        train_iter = iter(train_ds)

        model.train()
        stage_step = 0
        running_loss = 0.0
        step_t0 = time.time()
        passed_gate = False

        while stage_step < cfg.steps_per_stage:
            # LR schedule (per-stage cosine)
            lr = get_lr(stage_step, cfg.warmup_steps, cfg.steps_per_stage,
                        cfg.max_lr, cfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation
            optimizer.zero_grad()
            accum_loss = 0.0

            for _ in range(cfg.grad_accum):
                token_ids, labels = next(train_iter)
                token_ids = token_ids.unsqueeze(0).to(device)
                labels = labels.unsqueeze(0).to(device)

                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(token_ids, labels=labels)
                    loss = out["loss"] / cfg.grad_accum

                loss.backward()
                accum_loss += out["loss"].item()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            avg_loss = accum_loss / cfg.grad_accum
            running_loss += avg_loss
            total_tokens += cfg.seq_len * cfg.grad_accum
            stage_step += 1
            global_step += 1

            # --- Logging ---
            if stage_step % cfg.log_every == 0:
                elapsed = time.time() - step_t0
                steps_per_sec = cfg.log_every / elapsed
                tokens_per_sec = (cfg.seq_len * cfg.grad_accum * cfg.log_every) / elapsed
                avg_running = running_loss / cfg.log_every
                ppl = math.exp(min(avg_running, 20))

                mem_used = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                mem_peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0

                entry = {
                    "stage": stage_idx, "stage_step": stage_step,
                    "step": global_step, "loss": round(avg_running, 4),
                    "ppl": round(ppl, 2), "lr": round(lr, 8),
                    "grad_norm": round(grad_norm.item(), 4),
                    "steps_per_sec": round(steps_per_sec, 3),
                    "tokens_per_sec": round(tokens_per_sec),
                    "total_tokens": total_tokens,
                    "gpu_mem_gb": round(mem_used, 2),
                    "gpu_peak_gb": round(mem_peak, 2),
                }
                log(entry)

                eta_stage = (cfg.steps_per_stage - stage_step) / max(steps_per_sec, 0.001)
                print(f"  S{stage_idx} {stage_step:>5d}/{cfg.steps_per_stage} "
                      f"(g:{global_step}) | loss {avg_running:.4f} | ppl {ppl:>7.2f} | "
                      f"lr {lr:.2e} | {steps_per_sec:.2f} s/s | {tokens_per_sec:,.0f} t/s | "
                      f"gpu {mem_peak:.1f}GB | eta {eta_stage/60:.0f}m")

                running_loss = 0.0
                step_t0 = time.time()

            # --- Gate check ---
            if stage_step % cfg.gate_check_every == 0:
                print(f"\n  Gate check at stage step {stage_step}...")
                val_metrics = evaluate(model, val_ds, device)
                log({"type": "gate_check", "stage": stage_idx,
                     "stage_step": stage_step, "step": global_step,
                     **val_metrics})
                print(f"  val_loss={val_metrics['val_loss']:.4f}  "
                      f"val_ppl={val_metrics['val_ppl']:.2f}  "
                      f"threshold={cfg.gate_threshold}")

                if val_metrics["val_loss"] < cfg.gate_threshold:
                    print(f"  GATE PASSED - advancing to next stage")
                    passed_gate = True

                    # Save stage checkpoint
                    ckpt_path = str(ckpt_dir / f"stage{stage_idx}_passed.pt")
                    torch.save({
                        "global_step": global_step,
                        "stage": stage_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": model_config,
                    }, ckpt_path)
                    print(f"  Saved: {ckpt_path}\n")
                    break
                else:
                    print(f"  Not yet — continuing training\n")

        train_ds.stop()

        if not passed_gate:
            print(f"\n  WARNING: Stage {stage_idx} did not pass gate after "
                  f"{cfg.steps_per_stage} steps (val_loss={val_metrics['val_loss']:.4f})")
            # Save and continue anyway
            ckpt_path = str(ckpt_dir / f"stage{stage_idx}_timeout.pt")
            torch.save({
                "global_step": global_step,
                "stage": stage_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # --- Done ---
    elapsed_total = time.time() - t0_global
    print(f"\n{'='*70}")
    print(f"  Curriculum complete!")
    print(f"  Total steps: {global_step}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total time: {elapsed_total/3600:.1f} hours")
    print(f"{'='*70}")

    final_path = str(ckpt_dir / "final.pt")
    torch.save({
        "global_step": global_step,
        "stage": len(STAGES) - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model_config,
    }, final_path)
    print(f"Saved: {final_path}")

    log_file.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Curriculum training for BashTransformer")
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--steps-per-stage", type=int, default=2000)
    parser.add_argument("--gate-threshold", type=float, default=1.5,
                        help="Val loss must be below this to advance")
    parser.add_argument("--gate-check-every", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-stage", type=int, default=0)
    parser.add_argument("--resume-ckpt", type=str, default=None)

    args = parser.parse_args()

    cfg = CurriculumConfig(
        seq_len=args.seq_len,
        grad_accum=args.grad_accum,
        steps_per_stage=args.steps_per_stage,
        gate_threshold=args.gate_threshold,
        gate_check_every=args.gate_check_every,
        warmup_steps=args.warmup_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        log_every=args.log_every,
        data_workers=args.data_workers,
        buffer_size=args.buffer_size,
        seed=args.seed,
        ckpt_dir=args.ckpt_dir,
        resume_stage=args.resume_stage,
        resume_ckpt=args.resume_ckpt,
    )

    train_curriculum(cfg)
