"""
Interactive inference for NullRoot.

The model reads filesystem state + command, produces a response + state patch.
State patches are applied to maintain the filesystem between commands.

Usage:
    python sample.py checkpoints/nullroot_v2.pt
    python sample.py checkpoints/nullroot_v2.pt --demo unix
"""

import argparse
import sys

import torch
from torch.amp import autocast

from tokenizer import BashTokenizer
from model import BashTransformer, BashTransformerConfig
from generator import FileSystem


def load_model(ckpt_path: str, device: torch.device) -> BashTransformer:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", BashTransformerConfig())
    model = BashTransformer(config).to(device).to(torch.bfloat16)
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    step = ckpt.get("global_step", "?")
    print(f"Loaded checkpoint from step {step}")
    return model


@torch.no_grad()
def generate(model, context_ids, tokenizer, device, max_new=512):
    """Generate response + state patch.

    Output format: <output>...<eor><state>...<eor> or <output>...<eor><nop>
    So we need to keep going past the first <eor> until we hit a second
    <eor> or <nop> (which terminates the state patch).
    """
    ids = list(context_ids)
    generated = []
    eor_count = 0
    for _ in range(max_new):
        input_t = torch.tensor([ids], dtype=torch.long, device=device)
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(input_t)
        next_id = out["logits"][0, -1, :].argmax().item()
        ids.append(next_id)
        generated.append(next_id)
        if next_id == tokenizer.eor_id:
            eor_count += 1
            if eor_count >= 2:  # response <eor> + state <eor>
                break
        if next_id == tokenizer.nop_id:
            break
        if next_id == tokenizer.eos_id:
            break
    return generated


def run_command(model, state_str, cmd_str, tok, device, fs):
    """Run one command: build input, generate, parse output, apply patch."""
    # Build input
    input_text = ""
    if state_str:
        input_text += f"<state>{state_str}<eor>"
    input_text += f"<prompt>{cmd_str}<eoi>"
    context_ids = tok.encode(input_text)

    # Generate full output (response + state patch)
    gen_ids = generate(model, context_ids, tok, device)
    gen_text = tok.decode(gen_ids)

    # Parse response vs state patch
    response_text = gen_text
    patch_text = ""

    if "<state>" in gen_text:
        idx = gen_text.index("<state>")
        response_text = gen_text[:idx]
        rest = gen_text[idx + 7:]  # skip <state>
        if rest.endswith("<eor>"):
            rest = rest[:-5]
        patch_text = rest
    elif "<nop>" in gen_text:
        response_text = gen_text[:gen_text.index("<nop>")]

    # Apply patch to filesystem state
    if patch_text:
        state_str = FileSystem.apply_patch(state_str, patch_text)

    # Also execute on real fs for state tracking (fallback)
    _execute_on_fs(fs, cmd_str)

    # Clean response for display
    display = response_text
    for tag in ["<output>", "<err>", "<eor>", "<prompt>", "<eos>", "<eoi>"]:
        display = display.replace(tag, "")

    return display, state_str


def _execute_on_fs(fs, cmd_str):
    """Execute command on filesystem for state tracking."""
    parts = cmd_str.split()
    if not parts:
        return
    cmd = parts[0]
    if cmd == "mkdir" and len(parts) > 1:
        fs.mkdir(parts[1])
    elif cmd == "cd" and len(parts) > 1:
        fs.cd(parts[1])
    elif cmd == "touch" and len(parts) > 1:
        fs.touch(parts[1])
    elif cmd == "echo" and (">" in parts or ">>" in parts):
        if ">>" in parts:
            idx = parts.index(">>")
            content = " ".join(parts[1:idx])
            fs.append_file(parts[idx + 1], content)
        else:
            idx = parts.index(">")
            content = " ".join(parts[1:idx])
            fs.write_file(parts[idx + 1], content)
    elif cmd == "rm" and len(parts) > 1:
        fs.rm(parts[1])
    elif cmd == "cp" and len(parts) > 2:
        fs.cp(parts[1], parts[2])
    elif cmd == "mv" and len(parts) > 2:
        fs.mv(parts[1], parts[2])


# ---------------------------------------------------------------------------
# Demo: pre-built filesystem
# ---------------------------------------------------------------------------

DEMOS = {
    "unix": [
        "mkdir etc",
        "mkdir home",
        "mkdir tmp",
        "cd etc",
        "echo 127.0.0.1 localhost > hosts",
        "echo root 0 0 > passwd",
        "cd /home",
        "mkdir alice",
        "cd alice",
        "echo hello world > readme.txt",
        "mkdir docs",
        "cd docs",
        "echo todo fix bugs > todo.txt",
        "cd /",
    ],
    "project": [
        "mkdir src",
        "mkdir tests",
        "echo print hello > src/main.py",
        "echo assert true > tests/test.py",
        "echo my project > readme.md",
        "cd /",
    ],
}


def run_demo(model, demo_name, tok, device):
    """Build a filesystem with pre-set commands, then drop into interactive mode."""
    if demo_name not in DEMOS:
        print(f"Unknown demo: {demo_name}")
        print(f"Available: {', '.join(DEMOS.keys())}")
        return

    commands = DEMOS[demo_name]
    state_str = ""
    fs = FileSystem()

    print(f"\nBuilding '{demo_name}' filesystem ({len(commands)} commands)...")
    for cmd in commands:
        output, state_str = run_command(model, state_str, cmd, tok, device, fs)
        if output.strip():
            print(f"  $ {cmd}")
            print(f"  {output}")

    # Use the real fs state (more reliable than model patches for setup)
    state_str = fs.serialize_state()

    print(f"\nFilesystem ready. {len(fs.dirs)} dirs, {len(fs.files)} files.")
    print(f"CWD: {fs.cwd}")
    print(f"State: {len(state_str)} chars\n")

    return state_str, fs


def interactive(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BashTokenizer()
    model = load_model(args.checkpoint, device)

    state_str = ""
    fs = FileSystem()

    if args.demo:
        state_str, fs = run_demo(model, args.demo, tok, device)

    print(f"NullRoot interactive shell (state-patch architecture)")
    print(f"Commands: mkdir cd ls pwd touch echo cat rm cp mv head wc find grep")
    print(f"Type 'state' to see raw state. 'reset' to clear. Ctrl-C to quit.\n")

    while True:
        try:
            cwd = fs.cwd
            cmd = input(f"nullroot:{cwd}$ ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        cmd = cmd.strip()
        if not cmd:
            continue
        if cmd == "exit":
            break
        if cmd == "reset":
            state_str = ""
            fs = FileSystem()
            print("(reset)")
            continue
        if cmd == "state":
            print(f"  {state_str}")
            continue

        output, state_str = run_command(model, state_str, cmd, tok, device, fs)
        if output.strip():
            print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NullRoot interactive shell")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--demo", "-d", type=str, default=None,
                        choices=list(DEMOS.keys()),
                        help="Pre-build a filesystem before interactive mode")
    args = parser.parse_args()
    interactive(args)
