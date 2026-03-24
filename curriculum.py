"""
Curriculum training for BashTransformer with state patches.

Each command is processed as: [state] <prompt>cmd<eoi> → <output>response<eor> <state>patch<eor>
The model reads state, processes command, produces response + state patch.
Patch is applied to produce state for next command.

Pure autoregressive — no memory modules. The state IS the memory,
expressed as tokens the model reads and writes.
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
from dataset import BashSessionDataset, build_session_samples
from generator import SessionGenerator, FileSystem


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

ALL_COMMANDS = [
    "mkdir", "cd_child", "cd_up", "cd_abs", "ls", "pwd",
    "touch", "echo_write", "cat", "echo_append", "rm",
    "cp", "mv", "head", "wc", "find", "grep",
]

STAGES = [
    {
        "name": "Stage 1: all commands",
        "commands": ALL_COMMANDS,
        "error_rate": 0.0,
    },
]


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------

def _build_gate_tests(stage_idx):
    tests = []

    # mkdir + cd + ls + pwd
    tests.append([
        ("mkdir gal", "<output><eor>"),
        ("mkdir tov", "<output><eor>"),
        ("ls", "<output>gal  tov<eor>"),
        ("cd gal", "<output><eor>"),
        ("pwd", "<output>/gal<eor>"),
        ("cd ..", "<output><eor>"),
        ("ls", "<output>gal  tov<eor>"),
    ])

    # echo > + cat
    tests.append([
        ("echo test data here > myfile", "<output><eor>"),
        ("cat myfile", "<output>test data here<eor>"),
    ])

    # echo >> + cat
    tests.append([
        ("echo first line > log.txt", "<output><eor>"),
        ("echo second line >> log.txt", "<output><eor>"),
        ("cat log.txt", "<output>first line\nsecond line<eor>"),
    ])

    # rm
    tests.append([
        ("echo data > tmp.dat", "<output><eor>"),
        ("rm tmp.dat", "<output><eor>"),
        ("ls", "<output><eor>"),
    ])

    # cp
    tests.append([
        ("echo original > src.txt", "<output><eor>"),
        ("cp src.txt dst.txt", "<output><eor>"),
        ("cat dst.txt", "<output>original<eor>"),
    ])

    # mv
    tests.append([
        ("echo moveme > old.txt", "<output><eor>"),
        ("mv old.txt new.txt", "<output><eor>"),
        ("cat new.txt", "<output>moveme<eor>"),
    ])

    # head
    tests.append([
        ("echo first line > multi.txt", "<output><eor>"),
        ("echo second line >> multi.txt", "<output><eor>"),
        ("head multi.txt", "<output>first line<eor>"),
    ])

    # wc
    tests.append([
        ("echo hello world > count.txt", "<output><eor>"),
        ("wc count.txt", "<output>1 2 11 count.txt<eor>"),
    ])

    # Variables
    tests.append([
        ("x=42", "<output><eor>"),
        ("echo $x", "<output>42<eor>"),
    ])

    # Math
    tests.append([
        ("expr 3 + 5", "<output>8<eor>"),
        ("expr 10 * 4", "<output>40<eor>"),
    ])

    # Script execution
    tests.append([
        ("echo echo hello > test.sh", "<output><eor>"),
        ("sh test.sh", "<output>hello<eor>"),
    ])

    return tests


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_response(model, context_ids, tokenizer, device, max_tokens=512):
    """Generate response + state patch (stop after 2nd <eor> or <nop>)."""
    ids = list(context_ids)
    generated = []
    eor_count = 0
    for _ in range(max_tokens):
        input_t = torch.tensor([ids], dtype=torch.long, device=device)
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(input_t)
        next_id = out["logits"][0, -1, :].argmax().item()
        ids.append(next_id)
        generated.append(next_id)
        if next_id == tokenizer.eor_id:
            eor_count += 1
            if eor_count >= 2:
                break
        if next_id in (tokenizer.nop_id, tokenizer.eos_id):
            break
    return generated


@torch.no_grad()
def run_gate_tests(model, stage_idx, device, log_fn=None):
    """Run gate tests with state tracking."""
    model.eval()
    tok = BashTokenizer()
    tests = _build_gate_tests(stage_idx)

    total = 0
    correct = 0
    all_samples = []

    for script in tests:
        state_str = ""
        fs = FileSystem()

        for cmd_str, expected_response in script:
            # Build input: [state]<prompt>cmd<eoi>
            input_text = ""
            if state_str:
                input_text += f"<state>{state_str}<eor>"
            input_text += f"<prompt>{cmd_str}<eoi>"
            context_ids = tok.encode(input_text)

            # Generate response (stop at <eor>)
            gen_ids = generate_response(model, context_ids, tok, device)
            gen_text = tok.decode(gen_ids)

            # Extract just the response part (before <state> or <nop>)
            response_text = gen_text
            if "<state>" in response_text:
                response_text = response_text[:response_text.index("<state>")]
            if "<nop>" in response_text:
                response_text = response_text[:response_text.index("<nop>")]

            match = response_text == expected_response
            is_meaningful = expected_response not in ("<output><eor>",)

            if is_meaningful:
                total += 1
                if match:
                    correct += 1

            sample = {
                "command": cmd_str,
                "expected": expected_response,
                "generated": response_text,
                "match": match,
                "meaningful": is_meaningful,
            }
            all_samples.append(sample)

            tag = "GATE" if is_meaningful else "    "
            match_str = "OK" if match else "WRONG"
            print(f"    [{match_str}] [{tag}] {cmd_str}")
            print(f"      expected:  {expected_response!r}")
            print(f"      got:       {response_text!r}")

            # Execute command on real filesystem for state tracking
            _execute_cmd_on_fs(fs, cmd_str)
            state_str = fs.serialize_state()

    accuracy = correct / total if total > 0 else 0
    passed = correct == total

    print(f"\n    Gate: {correct}/{total} ({accuracy:.0%}) {'PASSED' if passed else 'FAILED'}")

    if log_fn:
        log_fn({"type": "samples", "samples": all_samples,
                "gate_correct": correct, "gate_total": total,
                "all_correct": sum(1 for s in all_samples if s["match"]),
                "all_total": len(all_samples)})

    model.train()
    return passed, {"gate_correct": correct, "gate_total": total,
                     "gate_accuracy": round(accuracy, 4), "gate_passed": passed,
                     "all_correct": sum(1 for s in all_samples if s["match"]),
                     "all_total": len(all_samples)}


def _execute_cmd_on_fs(fs, cmd_str):
    """Execute a command on FileSystem for state tracking. Uses execute_command."""
    fs.execute_command(cmd_str)


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
# Training
# ---------------------------------------------------------------------------

@dataclass
class CurriculumConfig:
    warmup_steps: int = 100
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    max_grad_norm: float = 1.0
    log_every: int = 10
    gate_check_every: int = 500
    steps_per_stage: int = 50000  # for LR schedule only
    data_workers: int = 4
    buffer_size: int = 16
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    resume_stage: int = 0
    resume_ckpt: str | None = None
    min_ops: int = 10
    target_ops: int = 30


def train_curriculum(cfg: CurriculumConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    model_config = BashTransformerConfig(gradient_checkpointing=True)
    model = BashTransformer(model_config).to(device).to(torch.bfloat16)

    print("Compiling model...")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.max_lr,
        betas=(0.9, 0.95), weight_decay=0.1, fused=True,
    )

    # Resume
    if cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt):
        print(f"Resuming from {cfg.resume_ckpt}...")
        ckpt = torch.load(cfg.resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded checkpoint at global step {ckpt.get('global_step', '?')}")

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

    for stage_idx in range(cfg.resume_stage, len(STAGES)):
        stage = STAGES[stage_idx]
        print(f"\n{'='*70}")
        print(f"  {stage['name']}")
        print(f"  Commands: {sorted(stage['commands'])}")
        print(f"{'='*70}\n")

        log({"type": "stage_start", "stage": stage_idx, "name": stage["name"],
             "global_step": global_step})

        train_ds = BashSessionDataset(
            buffer_size=cfg.buffer_size,
            workers=cfg.data_workers,
            min_ops=cfg.min_ops,
            target_ops=cfg.target_ops,
            base_seed=cfg.seed + stage_idx * 1000,
        )
        train_iter = iter(train_ds)

        model.train()
        stage_step = 0
        running_loss = 0.0
        running_cmds = 0
        step_t0 = time.time()
        passed_gate = False

        while not passed_gate:
            lr = get_lr(stage_step, cfg.warmup_steps, cfg.steps_per_stage,
                        cfg.max_lr, cfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()

            session = next(train_iter)
            session_loss = 0.0
            session_tokens = 0

            # Each command in session is an independent autoregressive sample
            for cmd in session:
                ids_t = torch.tensor([cmd["ids"]], dtype=torch.long, device=device)
                labels_t = torch.tensor([cmd["labels"]], dtype=torch.long, device=device)
                weights_t = torch.tensor([cmd["weights"]], dtype=torch.float32, device=device)

                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(ids_t, labels=labels_t, loss_weights=weights_t)

                n_tokens = (labels_t != -100).sum().item()
                if n_tokens > 0:
                    out["loss"].backward()
                    session_loss += out["loss"].item() * n_tokens
                    session_tokens += n_tokens

            if session_tokens > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                running_loss += session_loss / session_tokens
                running_cmds += len(session)
                total_tokens += session_tokens
            else:
                grad_norm = torch.tensor(0.0)

            stage_step += 1
            global_step += 1

            if stage_step % cfg.log_every == 0:
                elapsed = time.time() - step_t0
                steps_per_sec = cfg.log_every / elapsed
                avg_running = running_loss / cfg.log_every
                ppl = math.exp(min(avg_running, 20))
                avg_cmds = running_cmds / cfg.log_every
                mem_peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0

                log({"stage": stage_idx, "stage_step": stage_step,
                     "step": global_step, "loss": round(avg_running, 4),
                     "ppl": round(ppl, 2), "lr": round(lr, 8),
                     "grad_norm": round(grad_norm.item(), 4),
                     "steps_per_sec": round(steps_per_sec, 3),
                     "total_tokens": total_tokens,
                     "avg_cmds_per_session": round(avg_cmds, 1),
                     "gpu_peak_gb": round(mem_peak, 2)})

                print(f"  S{stage_idx} {stage_step:>5d} (g:{global_step}) | "
                      f"loss {avg_running:.4f} | ppl {ppl:>7.2f} | "
                      f"lr {lr:.2e} | {steps_per_sec:.2f} s/s | "
                      f"{avg_cmds:.0f} cmd/sess | gpu {mem_peak:.1f}GB")

                running_loss = 0.0
                running_cmds = 0
                step_t0 = time.time()

            if stage_step % cfg.gate_check_every == 0:
                print(f"\n  Gate check at step {stage_step}...")
                gate_passed, gate_metrics = run_gate_tests(
                    model, stage_idx, device, log_fn=log)
                log({"type": "gate_check", "stage": stage_idx,
                     "stage_step": stage_step, "step": global_step,
                     **gate_metrics})

                # Save checkpoint
                torch.save({
                    "global_step": global_step,
                    "stage": stage_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_config,
                }, str(ckpt_dir / "latest.pt"))

                if gate_passed:
                    print(f"  GATE PASSED — advancing!")
                    passed_gate = True
                    torch.save({
                        "global_step": global_step,
                        "stage": stage_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": model_config,
                    }, str(ckpt_dir / f"stage{stage_idx}_passed.pt"))
                    break
                else:
                    print(f"  Not yet — continuing\n")

        train_ds.stop()

    print(f"\nCurriculum complete!")
    log_file.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-ops", type=int, default=10)
    parser.add_argument("--target-ops", type=int, default=30)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-stage", type=int, default=0)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--gate-check-every", type=int, default=500)
    args = parser.parse_args()

    cfg = CurriculumConfig(
        min_ops=args.min_ops,
        target_ops=args.target_ops,
        ckpt_dir=args.ckpt_dir,
        resume_stage=args.resume_stage,
        resume_ckpt=args.resume_ckpt,
        gate_check_every=args.gate_check_every,
    )
    train_curriculum(cfg)
