"""
Training for paged-context NullRoot.

Each command sees only the current directory's page. Tiny context, fast training.
Static prefix registers give the model a scratchpad for reasoning.
"""

import os, sys, time, math, json
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.amp import autocast

from tokenizer import BashTokenizer
from model import BashTransformer, BashTransformerConfig
from dataset import BashSessionDataset
from paged_fs import PagedFileSystem


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------

def _build_gate_tests():
    """Returns list of (setup_cmds, test_cmd, expected_response) tuples."""
    tests = []

    # mkdir + ls
    tests.append((["mkdir gal", "mkdir tov"], "ls", "gal  tov"))
    # cd + ls (page swap)
    tests.append((["mkdir plok", "cd plok"], "ls", ""))
    # cd back + ls
    tests.append((["mkdir plok", "cd plok", "cd .."], "ls", "plok"))
    # pwd
    tests.append((["mkdir wun", "cd wun"], "pwd", "/wun"))
    # echo + cat
    tests.append((["echo hello world > greet.txt"], "cat greet.txt", "hello world"))
    # append + cat
    tests.append((["echo line1 > f.txt", "echo line2 >> f.txt"], "cat f.txt", "line1\nline2"))
    # rm + ls
    tests.append((["echo x > tmp.dat", "rm tmp.dat"], "ls", ""))
    # cp + cat
    tests.append((["echo orig > a.txt", "cp a.txt b.txt"], "cat b.txt", "orig"))
    # mv + cat
    tests.append((["echo data > old.txt", "mv old.txt new.txt"], "cat new.txt", "data"))
    # head
    tests.append((["echo first > m.txt", "echo second >> m.txt"], "head m.txt", "first"))
    # wc
    tests.append((["echo hello world > c.txt"], "wc c.txt", "1 2 11 c.txt"))
    # variable
    tests.append((["x=42"], "echo $x", "42"))
    # math
    tests.append(([], "expr 3 + 5", "8"))
    tests.append(([], "expr 10 * 4", "40"))
    # cd deep + pwd
    tests.append((["mkdir a", "cd a", "mkdir b", "cd b"], "pwd", "/a/b"))

    return tests


@torch.no_grad()
def run_gate_tests(model, device, log_fn=None):
    """Run gate tests with paged filesystem."""
    model.eval()
    tok = BashTokenizer()
    tests = _build_gate_tests()

    total = 0
    correct = 0
    all_samples = []

    for setup_cmds, test_cmd, expected in tests:
        fs = PagedFileSystem()

        # Run setup commands
        for cmd in setup_cmds:
            response, patch = fs.execute(cmd)
            if patch:
                fs.apply_patch(patch)

        # Build model input
        full_input = fs.serialize_full_input(test_cmd)
        ids = tok.encode(full_input)

        # Generate
        gen_ids = []
        eor_count = 0
        ctx = list(ids)
        for _ in range(500):
            input_t = torch.tensor([ctx], dtype=torch.long, device=device)
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(input_t)
            next_id = out["logits"][0, -1, :].argmax().item()
            ctx.append(next_id)
            gen_ids.append(next_id)
            if next_id == tok.eor_id:
                eor_count += 1
                if eor_count >= 2:
                    break
            if next_id in (tok.nop_id, tok.eos_id):
                break

        gen_text = tok.decode(gen_ids)

        # Extract response
        resp = gen_text
        if "<state>" in resp:
            resp = resp[:resp.index("<state>")]
        elif "<nop>" in resp:
            resp = resp[:resp.index("<nop>")]
        resp = resp.replace("<output>", "").replace("<eor>", "").strip()

        match = resp == expected
        total += 1
        if match:
            correct += 1

        tag = "OK" if match else "WRONG"
        setup_str = " → ".join(setup_cmds) if setup_cmds else "(none)"
        print(f"    [{tag}] {setup_str} → {test_cmd}")
        print(f"      expected: {expected!r}")
        print(f"      got:      {resp!r}")

        all_samples.append({
            "command": test_cmd, "expected": f"<output>{expected}<eor>",
            "generated": resp, "match": match, "meaningful": True,
        })

    accuracy = correct / total if total > 0 else 0
    passed = correct == total
    print(f"\n    Gate: {correct}/{total} ({accuracy:.0%}) {'PASSED' if passed else 'FAILED'}")

    if log_fn:
        log_fn({"type": "samples", "samples": all_samples,
                "gate_correct": correct, "gate_total": total,
                "all_correct": correct, "all_total": total})

    model.train()
    return passed, {"gate_correct": correct, "gate_total": total,
                     "gate_accuracy": round(accuracy, 4), "gate_passed": passed,
                     "all_correct": correct, "all_total": total}


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    warmup_steps: int = 100
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    max_grad_norm: float = 1.0
    log_every: int = 10
    gate_check_every: int = 500
    steps_per_stage: int = 50000
    data_workers: int = 4
    buffer_size: int = 16
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    resume_ckpt: str | None = None
    min_ops: int = 20
    target_ops: int = 50


def train(cfg: TrainConfig):
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
        betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    if cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt):
        ckpt = torch.load(cfg.resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from step {ckpt.get('global_step', '?')}")

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "a")

    def log(entry):
        entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    global_step = ckpt.get("global_step", 0) if (cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt)) else 0

    log({"type": "stage_start", "stage": 0, "name": "paged context + registers",
         "global_step": global_step})

    train_ds = BashSessionDataset(
        buffer_size=cfg.buffer_size, workers=cfg.data_workers,
        min_ops=cfg.min_ops, target_ops=cfg.target_ops, base_seed=cfg.seed)
    train_iter = iter(train_ds)

    model.train()
    running_loss = 0.0
    running_cmds = 0
    step_t0 = time.time()
    total_tokens = 0
    passed = False

    while not passed:
        lr = get_lr(global_step, cfg.warmup_steps, cfg.steps_per_stage,
                    cfg.max_lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        session = next(train_iter)
        session_loss = 0.0
        session_tokens = 0

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

        global_step += 1

        if global_step % cfg.log_every == 0:
            elapsed = time.time() - step_t0
            sps = cfg.log_every / elapsed
            avg = running_loss / cfg.log_every
            ppl = math.exp(min(avg, 20))
            avg_cmds = running_cmds / cfg.log_every
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0

            log({"step": global_step, "loss": round(avg, 4), "ppl": round(ppl, 2),
                 "lr": round(lr, 8), "grad_norm": round(grad_norm.item(), 4),
                 "steps_per_sec": round(sps, 3), "total_tokens": total_tokens,
                 "avg_cmds": round(avg_cmds, 1), "gpu_peak_gb": round(mem, 2)})

            print(f"  {global_step:>6d} | loss {avg:.4f} | ppl {ppl:>7.2f} | "
                  f"lr {lr:.2e} | {sps:.1f} s/s | {avg_cmds:.0f} cmd/sess | gpu {mem:.1f}GB")

            running_loss = 0.0
            running_cmds = 0
            step_t0 = time.time()

        if global_step % cfg.gate_check_every == 0:
            print(f"\n  Gate check at step {global_step}...")
            gate_passed, gate_metrics = run_gate_tests(model, device, log_fn=log)
            log({"type": "gate_check", "step": global_step, **gate_metrics})

            torch.save({
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
            }, str(ckpt_dir / "latest.pt"))

            if gate_passed:
                print(f"  GATE PASSED!")
                passed = True
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                }, str(ckpt_dir / "nullroot_v4.pt"))
                break
            else:
                print(f"  Not yet — continuing\n")

    train_ds.stop()
    log_file.close()
    print(f"\nTraining complete at step {global_step}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-ops", type=int, default=20)
    parser.add_argument("--target-ops", type=int, default=50)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--gate-check-every", type=int, default=500)
    args = parser.parse_args()

    cfg = TrainConfig(
        min_ops=args.min_ops, target_ops=args.target_ops,
        ckpt_dir=args.ckpt_dir, resume_ckpt=args.resume_ckpt,
        gate_check_every=args.gate_check_every)
    train(cfg)
