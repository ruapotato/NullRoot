"""
Curriculum training for BashTransformer (Phase 2 — memory-based).

Each command is processed in isolation. Filesystem state lives in the memory
bank. No token history crosses command boundaries. Memory is detached between
commands (truncated BPTT — gradients flow within a command, not across).

Sessions are self-contained: memory resets at session start, commands play
in order. Each command exchange is complete — the model always sees the full
<prompt>...<eoi> and generates <output>...<eor>.

Stages introduce commands progressively. Same curriculum and gate structure
as Phase 1.
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
from dataset import (BashSessionDataset, BashValidationDataset, RecallDataset,
                     generate_recall_session, _build_labels)
from generator import SessionGenerator


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES = [
    {
        "name": "Stage 0: memory warmup (echo recall)",
        "commands": None,
        "error_rate": 0.0,
        "recall": True,
    },
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
    {
        "name": "Stage 9: anneal (all commands)",
        "commands": {"mkdir", "cd_child", "cd_up", "cd_abs", "ls", "pwd", "touch", "echo_write", "cat", "echo_append", "rm", "errors"},
        "error_rate": 0.05,
        "anneal": True,
    },
]


# ---------------------------------------------------------------------------
# Validation set generation for each stage
# ---------------------------------------------------------------------------

def generate_stage_validation(stage_idx: int, num_sessions: int = 10,
                              min_ops: int = 200, seed: int = 99999) -> list[str]:
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
    """Validation dataset generated on-the-fly for a curriculum stage.

    Each sample is a complete session as a list of command exchange dicts.
    """

    def __init__(self, stage_idx: int, num_sessions: int = 10, seed: int = 99999):
        self.tokenizer = BashTokenizer()
        self.samples = []

        stage = STAGES[stage_idx]
        if stage.get("recall"):
            for i in range(num_sessions):
                cmds = generate_recall_session(num_cmds=5, seed=seed + i)
                if cmds:
                    self.samples.append(cmds)
        else:
            from dataset import split_session_into_commands
            transcripts = generate_stage_validation(
                stage_idx, num_sessions=num_sessions, seed=seed,
            )
            for t in transcripts:
                try:
                    cmds = split_session_into_commands(t, self.tokenizer)
                    if cmds:
                        self.samples.append(cmds)
                except ValueError:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Targeted gate tests — concrete scenarios per stage
# ---------------------------------------------------------------------------

def _build_gate_tests(stage_idx: int) -> list[list[tuple[str, str]]]:
    """Build targeted test scripts for a curriculum stage."""
    tests = []

    if stage_idx == 0:  # Stage 0: echo recall
        tests.append([
            ("gak", "<err>gak<eor>"),
            ("plim", "<err>gak plim<eor>"),
            ("tov", "<err>gak plim tov<eor>"),
        ])
        tests.append([
            ("brel", "<err>brel<eor>"),
            ("shuf", "<err>brel shuf<eor>"),
            ("dox", "<err>brel shuf dox<eor>"),
        ])
        return tests

    if stage_idx >= 1:  # Stage 1: mkdir + cd + ls
        tests.append([
            ("mkdir gal", "<output><eor>"),
            ("mkdir tov", "<output><eor>"),
            ("ls", "<output>gal  tov<eor>"),
        ])
        tests.append([
            ("mkdir plok", "<output><eor>"),
            ("cd plok", "<output><eor>"),
            ("ls", "<output><eor>"),
            ("cd ..", "<output><eor>"),
            ("ls", "<output>plok<eor>"),
        ])
        tests.append([
            ("mkdir dex", "<output><eor>"),
            ("cd dex", "<output><eor>"),
            ("mkdir sub", "<output><eor>"),
            ("mkdir rak", "<output><eor>"),
            ("ls", "<output>rak  sub<eor>"),
        ])

    if stage_idx >= 2:  # Stage 2: + pwd
        tests.append([
            ("pwd", "<output>/<eor>"),
            ("mkdir wun", "<output><eor>"),
            ("cd wun", "<output><eor>"),
            ("pwd", "<output>/wun<eor>"),
            ("mkdir deep", "<output><eor>"),
            ("cd deep", "<output><eor>"),
            ("pwd", "<output>/wun/deep<eor>"),
            ("cd ..", "<output><eor>"),
            ("pwd", "<output>/wun<eor>"),
        ])

    if stage_idx >= 3:  # Stage 3: + touch
        tests.append([
            ("mkdir tdir", "<output><eor>"),
            ("cd tdir", "<output><eor>"),
            ("touch fob.txt", "<output><eor>"),
            ("touch zel", "<output><eor>"),
            ("ls", "<output>fob.txt  zel<eor>"),
        ])

    if stage_idx >= 4:  # Stage 4: + echo >
        tests.append([
            ("echo hello world > greet.txt", "<output><eor>"),
            ("ls", "<output>greet.txt<eor>"),
        ])

    if stage_idx >= 5:  # Stage 5: + cat
        tests.append([
            ("echo test data here > myfile", "<output><eor>"),
            ("cat myfile", "<output>test data here<eor>"),
        ])
        tests.append([
            ("mkdir cdir", "<output><eor>"),
            ("cd cdir", "<output><eor>"),
            ("echo inside dir > inner.txt", "<output><eor>"),
            ("cat inner.txt", "<output>inside dir<eor>"),
        ])

    if stage_idx >= 6:  # Stage 6: + echo >>
        tests.append([
            ("echo first line > log.txt", "<output><eor>"),
            ("echo second line >> log.txt", "<output><eor>"),
            ("cat log.txt", "<output>first line\nsecond line<eor>"),
        ])

    if stage_idx >= 7:  # Stage 7: + rm
        tests.append([
            ("echo data > tmp.dat", "<output><eor>"),
            ("ls", "<output>tmp.dat<eor>"),
            ("rm tmp.dat", "<output><eor>"),
            ("ls", "<output><eor>"),
        ])

    if stage_idx >= 8:  # Stage 8: + errors
        tests.append([
            ("cd nofolder", "<err><eor>"),
            ("cat nofile.txt", "<err><eor>"),
            ("mkdir dup", "<output><eor>"),
            ("mkdir dup", "<err><eor>"),
        ])

    return tests


# ---------------------------------------------------------------------------
# Evaluation (Phase 2 — memory-based generation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_response(model, cmd_ids: list[int], memory: torch.Tensor,
                      tokenizer: BashTokenizer, device,
                      max_tokens: int = 512) -> tuple[list[int], torch.Tensor]:
    """Generate tokens for a single command using memory state.

    Returns (generated_ids, updated_memory).
    The model sees ONLY the current command tokens + memory.
    """
    ids = list(cmd_ids)
    for _ in range(max_tokens):
        input_t = torch.tensor([ids], dtype=torch.long, device=device)
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(input_t, memory=memory)
        next_id = out["logits"][0, -1, :].argmax().item()
        ids.append(next_id)
        # Update memory from the full sequence so far
        memory = out["memory"]
        if next_id in (tokenizer.eor_id, tokenizer.eos_id):
            break
    return ids[len(cmd_ids):], memory


@torch.no_grad()
def run_gate_tests(model, stage_idx: int, device, log_fn=None):
    """Run targeted gate tests for a stage. Returns (passed, results_dict).

    Each test script is a fresh session — memory resets at the start.
    Commands are fed one at a time, memory carries forward.
    """
    model.eval()
    tok = BashTokenizer()
    tests = _build_gate_tests(stage_idx)

    total = 0
    correct = 0
    all_samples = []

    for script_idx, script in enumerate(tests):
        # Fresh memory for each test script
        memory = model.reset_memory(1, device, next(model.parameters()).dtype)

        for cmd_str, expected_response in script:
            # Encode command: <prompt>cmd<eoi>
            cmd_text = f"<prompt>{cmd_str}<eoi>"
            cmd_ids = tok.encode(cmd_text)

            # Generate response using only current command + memory
            gen_ids, memory = generate_response(
                model, cmd_ids, memory, tok, device)
            gen_text = tok.decode(gen_ids)

            match = gen_text == expected_response

            is_meaningful = expected_response not in ("<output><eor>",)

            if is_meaningful:
                total += 1
                if match:
                    correct += 1

            sample = {
                "command": cmd_str,
                "expected": expected_response,
                "generated": gen_text,
                "match": match,
                "meaningful": is_meaningful,
            }
            all_samples.append(sample)

            tag = "GATE" if is_meaningful else "    "
            match_str = "OK" if match else "WRONG"
            print(f"    [{match_str}] [{tag}] {cmd_str}")
            print(f"      expected:  {expected_response!r}")
            print(f"      got:       {gen_text!r}")

            # For subsequent commands, feed the EXPECTED response into memory
            # so the test builds on correct state, not model errors
            expected_ids = tok.encode(expected_response)
            full_exchange = cmd_ids + expected_ids
            input_t = torch.tensor([full_exchange], dtype=torch.long, device=device)
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(input_t, memory=memory)
            memory = out["memory"]

    accuracy = correct / total if total > 0 else 0
    passed = correct == total

    total_all = len(all_samples)
    correct_all = sum(1 for s in all_samples if s["match"])

    print(f"\n    Gate tests (meaningful only): {correct}/{total} ({accuracy:.0%})")
    print(f"    All tests: {correct_all}/{total_all}")
    print(f"    {'PASSED' if passed else 'FAILED'}")

    if log_fn:
        log_fn({"type": "samples", "samples": all_samples,
                "gate_correct": correct, "gate_total": total,
                "all_correct": correct_all, "all_total": total_all})

    model.train()
    return passed, {
        "gate_correct": correct,
        "gate_total": total,
        "gate_accuracy": round(accuracy, 4),
        "gate_passed": passed,
        "all_correct": correct_all,
        "all_total": total_all,
    }


@torch.no_grad()
def evaluate_loss(model, val_dataset, device):
    """Run validation loss over sessions. Each session: reset memory, iterate commands."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(len(val_dataset)):
        session = val_dataset[i]
        memory = model.reset_memory(1, device, next(model.parameters()).dtype)

        for cmd in session:
            ids_t = torch.tensor([cmd["ids"]], dtype=torch.long, device=device)
            labels_t = torch.tensor([cmd["labels"]], dtype=torch.long, device=device)
            weights_t = torch.tensor([cmd["weights"]], dtype=torch.float32, device=device)

            with autocast("cuda", dtype=torch.bfloat16):
                out = model(ids_t, memory=memory, labels=labels_t, loss_weights=weights_t)

            memory = out["memory"].detach()

            n_tokens = (labels_t != -100).sum().item()
            if n_tokens > 0:
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
# Curriculum training (Phase 2 — memory-based)
# ---------------------------------------------------------------------------

@dataclass
class CurriculumConfig:
    grad_accum: int = 1
    steps_per_stage: int = 10000
    anneal_steps: int = 10000
    gate_check_every: int = 500
    warmup_steps: int = 100
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    anneal_max_lr: float = 5e-5
    anneal_min_lr: float = 1e-6
    max_grad_norm: float = 1.0
    log_every: int = 10
    data_workers: int = 4
    buffer_size: int = 16
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    resume_stage: int = 0
    resume_ckpt: str | None = None
    # Session size for training data
    min_ops: int = 30
    target_ops: int = 80


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

    # --- Optimizer ---
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

    global_step = ckpt.get("global_step", 0) if (cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt)) else 0
    total_tokens = 0
    t0_global = time.time()

    # --- Stage loop ---
    for stage_idx in range(cfg.resume_stage, len(STAGES)):
        stage = STAGES[stage_idx]
        print(f"\n{'='*70}")
        print(f"  {stage['name']}")
        print(f"  Commands: {sorted(stage['commands']) if stage.get('commands') else 'echo recall'}")
        print(f"  Error rate: {stage['error_rate']}")
        print(f"{'='*70}\n")

        log({"type": "stage_start", "stage": stage_idx, "name": stage["name"],
             "global_step": global_step})

        # --- Stage validation set ---
        val_ds = StageValidationDataset(stage_idx)
        print(f"  Validation: {len(val_ds)} sessions")

        # --- Stage training data ---
        if stage.get("recall"):
            train_ds = RecallDataset(
                buffer_size=cfg.buffer_size,
                workers=cfg.data_workers,
                min_cmds=3,
                max_cmds=8,
                base_seed=cfg.seed + stage_idx * 1000,
            )
        else:
            train_ds = BashSessionDataset(
                buffer_size=cfg.buffer_size,
                workers=cfg.data_workers,
                min_ops=cfg.min_ops,
                target_ops=cfg.target_ops,
                error_rate=stage["error_rate"],
                base_seed=cfg.seed + stage_idx * 1000,
                commands=stage["commands"],
            )
        train_iter = iter(train_ds)

        model.train()
        stage_step = 0
        running_loss = 0.0
        running_cmds = 0
        step_t0 = time.time()
        passed_gate = False

        is_anneal = stage.get("anneal", False)
        stage_max_steps = cfg.anneal_steps if is_anneal else cfg.steps_per_stage
        stage_max_lr = cfg.anneal_max_lr if is_anneal else cfg.max_lr
        stage_min_lr = cfg.anneal_min_lr if is_anneal else cfg.min_lr

        if is_anneal:
            print(f"  Annealing: {stage_max_steps} steps, LR {stage_max_lr} -> {stage_min_lr}")

        while not passed_gate and not (is_anneal and stage_step >= stage_max_steps):
            # LR schedule
            lr = get_lr(stage_step, cfg.warmup_steps, stage_max_steps,
                        stage_max_lr, stage_min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # One step = one complete session
            # Reset memory, iterate commands in order, accumulate loss
            optimizer.zero_grad()

            session = next(train_iter)

            # Initialize memory for this session
            memory = model.reset_memory(1, device, next(model.parameters()).dtype)

            session_loss = 0.0
            session_tokens = 0
            accumulated_loss = None

            for cmd in session:
                ids_t = torch.tensor([cmd["ids"]], dtype=torch.long, device=device)
                labels_t = torch.tensor([cmd["labels"]], dtype=torch.long, device=device)
                weights_t = torch.tensor([cmd["weights"]], dtype=torch.float32, device=device)

                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(ids_t, memory=memory, labels=labels_t, loss_weights=weights_t)

                n_tokens = (labels_t != -100).sum().item()
                if n_tokens > 0:
                    # Accumulate loss — don't backward yet so gradients
                    # flow through the full memory chain (full BPTT)
                    if accumulated_loss is None:
                        accumulated_loss = out["loss"] * n_tokens
                    else:
                        accumulated_loss = accumulated_loss + out["loss"] * n_tokens
                    session_loss += out["loss"].item() * n_tokens
                    session_tokens += n_tokens

                # No detach — full BPTT through entire session
                memory = out["memory"]

            # Single backward through the full computation graph
            if session_tokens > 0 and accumulated_loss is not None:
                (accumulated_loss / session_tokens).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                avg_loss = session_loss / session_tokens
                running_loss += avg_loss
                running_cmds += len(session)
                total_tokens += session_tokens
            else:
                grad_norm = torch.tensor(0.0)

            stage_step += 1
            global_step += 1

            # --- Logging ---
            if stage_step % cfg.log_every == 0:
                elapsed = time.time() - step_t0
                steps_per_sec = cfg.log_every / elapsed
                tokens_per_sec = total_tokens / max(time.time() - t0_global, 1)
                avg_running = running_loss / cfg.log_every
                ppl = math.exp(min(avg_running, 20))
                avg_cmds = running_cmds / cfg.log_every

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
                    "avg_cmds_per_session": round(avg_cmds, 1),
                    "gpu_mem_gb": round(mem_used, 2),
                    "gpu_peak_gb": round(mem_peak, 2),
                }
                log(entry)

                eta_stage = (stage_max_steps - stage_step) / max(steps_per_sec, 0.001)
                print(f"  S{stage_idx} {stage_step:>5d}/{stage_max_steps} "
                      f"(g:{global_step}) | loss {avg_running:.4f} | ppl {ppl:>7.2f} | "
                      f"lr {lr:.2e} | {steps_per_sec:.2f} s/s | {avg_cmds:.0f} cmd/sess | "
                      f"gpu {mem_peak:.1f}GB | eta {eta_stage/60:.0f}m")

                running_loss = 0.0
                running_cmds = 0
                step_t0 = time.time()

            # --- Gate check ---
            if stage_step % cfg.gate_check_every == 0:
                print(f"\n  Gate check at stage step {stage_step}...")

                loss_metrics = evaluate_loss(model, val_ds, device)

                gate_passed, gate_metrics = run_gate_tests(
                    model, stage_idx, device, log_fn=log)

                log({"type": "gate_check", "stage": stage_idx,
                     "stage_step": stage_step, "step": global_step,
                     **loss_metrics, **gate_metrics})
                print(f"  val_loss={loss_metrics['val_loss']:.4f}  "
                      f"val_ppl={loss_metrics['val_ppl']:.2f}  "
                      f"gate={gate_metrics['gate_correct']}/{gate_metrics['gate_total']}")

                # Always save a resumable checkpoint
                torch.save({
                    "global_step": global_step,
                    "stage": stage_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_config,
                }, str(ckpt_dir / "latest.pt"))

                if is_anneal:
                    if stage_step % 1000 == 0:
                        ckpt_path = str(ckpt_dir / f"anneal_step{global_step}.pt")
                        torch.save({
                            "global_step": global_step,
                            "stage": stage_idx,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": model_config,
                        }, ckpt_path)
                        print(f"  Saved: {ckpt_path}")
                    print()
                elif gate_passed:
                    print(f"  GATE PASSED - all tests correct, advancing")
                    passed_gate = True

                    ckpt_path = str(ckpt_dir / f"stage{stage_idx}_passed.pt")
                    torch.save({
                        "global_step": global_step,
                        "stage": stage_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "gate_metrics": gate_metrics,
                        "config": model_config,
                    }, ckpt_path)
                    print(f"  Saved: {ckpt_path}\n")
                    break
                else:
                    print(f"  Not yet — continuing training\n")

        train_ds.stop()

        # Stage only exits via gate pass (or anneal completion)

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

    parser = argparse.ArgumentParser(description="Curriculum training for BashTransformer (Phase 2)")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps-per-stage", type=int, default=10000)
    parser.add_argument("--gate-check-every", type=int, default=500)
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
    parser.add_argument("--min-ops", type=int, default=30)
    parser.add_argument("--target-ops", type=int, default=80)

    args = parser.parse_args()

    cfg = CurriculumConfig(
        grad_accum=args.grad_accum,
        steps_per_stage=args.steps_per_stage,
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
        min_ops=args.min_ops,
        target_ops=args.target_ops,
    )

    train_curriculum(cfg)
