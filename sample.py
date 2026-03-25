"""
Interactive inference for NullRoot (paged context).

Each command sees only the current directory's page. cd swaps pages.
Python manages the page table. The model reads/writes single pages.

Usage:
    python sample.py
    python sample.py --demo unix
    python sample.py --checkpoint checkpoints/nullroot_v4.pt
"""

import argparse
import sys
import os

import torch
from torch.amp import autocast

from tokenizer import BashTokenizer
from model import BashTransformer, BashTransformerConfig
from paged_fs import PagedFileSystem


def load_model(ckpt_path: str, device: torch.device) -> BashTransformer:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", BashTransformerConfig())
    model = BashTransformer(config).to(device).to(torch.bfloat16)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    step = ckpt.get("global_step", "?")
    print(f"Loaded checkpoint from step {step}")
    return model


@torch.no_grad()
def generate(model, context_ids, tok, device, max_new=500):
    """Generate response + state patch (stop after 2nd <eor> or <nop>)."""
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
        if next_id == tok.eor_id:
            eor_count += 1
            if eor_count >= 2:
                break
        if next_id in (tok.nop_id, tok.eos_id):
            break
    return tok.decode(generated)


def run_command(model, fs, cmd, tok, device):
    """Run command: model generates response AND state patch. Both are used."""
    # Build input from current page
    full_input = fs.serialize_full_input(cmd)
    ids = tok.encode(full_input)

    # Model generates
    raw = generate(model, ids, tok, device)

    # Parse response + patch
    response = raw
    model_patch = ""
    if "<state>" in raw:
        response = raw[:raw.index("<state>")]
        model_patch = raw[raw.index("<state>") + 7:]
        if model_patch.endswith("<eor>"):
            model_patch = model_patch[:-5]
    elif "<nop>" in raw:
        response = raw[:raw.index("<nop>")]

    display = response.replace("<output>", "").replace("<err>", "").replace("<eor>", "").strip()

    # Apply model's patch to the page table
    if model_patch:
        # cd produces a full page swap — update cwd + load new page
        if cmd.strip().startswith("cd "):
            target = cmd.strip().split()[1]
            if target == "..":
                if fs.cwd != "/":
                    fs.cwd = "/".join(fs.cwd.rstrip("/").split("/")[:-1]) or "/"
            elif target.startswith("/"):
                fs.cwd = target
            else:
                child_path = fs._child_path(target)
                fs.cwd = child_path
            fs._ensure_page(fs.cwd)
        fs.apply_patch(model_patch)

    return display


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

DEMOS = {
    "unix": [
        "mkdir etc", "mkdir home", "mkdir tmp", "mkdir var",
        "cd etc",
        "echo 127.0.0.1 localhost > hosts",
        "echo root 0 0 > passwd",
        "echo nameserver 8.8.8.8 > resolv.conf",
        "cd /home",
        "mkdir alice", "mkdir bob",
        "cd alice",
        "mkdir documents", "mkdir projects",
        "echo hello world > readme.txt",
        "cd documents",
        "echo meeting notes > notes.txt",
        "echo todo fix bugs > todo.txt",
        "cd /home/bob",
        "mkdir scripts",
        "cd scripts",
        "echo echo backup started > backup.sh",
        "cd /var",
        "mkdir log",
        "cd log",
        "echo system started > syslog",
        "echo login alice >> syslog",
        "cd /",
    ],
    "project": [
        "mkdir src", "mkdir tests", "mkdir docs",
        "echo print hello > src/main.py",
        "echo assert true > tests/test.py",
        "echo my project > readme.md",
        "cd /",
    ],
    "scripting": [
        "x=10", "name=nullroot", "version=4",
        "mkdir data",
        "cd data",
        "echo alice 42 > users.txt",
        "echo bob 37 >> users.txt",
        "cd /",
    ],
}


def run_demo(fs, demo_name):
    if demo_name not in DEMOS:
        print(f"Unknown demo: {demo_name}")
        print(f"Available: {', '.join(DEMOS.keys())}")
        return

    print(f"Building '{demo_name}' filesystem ({len(DEMOS[demo_name])} commands)...")
    for cmd in DEMOS[demo_name]:
        response, patch = fs.execute(cmd)
        if patch:
            fs.apply_patch(patch)

    print(f"Ready. {len(fs.pages)} directories, CWD: {fs.cwd}\n")


def find_checkpoint():
    """Find the best available checkpoint."""
    candidates = [
        "checkpoints/nullroot_v4.pt",
        "checkpoints/latest.pt",
        "checkpoints/nullroot_v3.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def interactive(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BashTokenizer()

    ckpt_path = args.checkpoint or find_checkpoint()
    if not ckpt_path:
        print("No checkpoint found. Train first: python curriculum.py --min-ops 20 --target-ops 50")
        sys.exit(1)

    model = load_model(ckpt_path, device)
    fs = PagedFileSystem()

    if args.demo:
        run_demo(fs, args.demo)

    print(f"NullRoot interactive shell (paged context)")
    print(f"Commands: mkdir cd ls pwd touch echo cat rm cp mv head wc grep")
    print(f"          expr <math>, x=val, echo $x, export name=val")
    print(f"Type 'state' to see current page. 'pages' to see all pages.")
    print(f"     'reset' to clear. Ctrl-C or 'exit' to quit.\n")

    while True:
        try:
            cmd = input(f"nullroot:{fs.cwd}$ ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        cmd = cmd.strip()
        if not cmd:
            continue
        if cmd == "exit":
            break
        if cmd == "reset":
            fs = PagedFileSystem()
            print("(reset)")
            continue
        if cmd == "state":
            print(f"  page: {fs.serialize_page()}")
            continue
        if cmd == "pages":
            for path in sorted(fs.pages.keys()):
                n_children = len(fs.pages[path]["children"])
                n_files = len(fs.pages[path]["files"])
                marker = " ← " if path == fs.cwd else "   "
                print(f"  {marker}{path:30s} {n_children} children, {n_files} files")
            continue

        output = run_command(model, fs, cmd, tok, device)
        if output:
            print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NullRoot interactive shell")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to checkpoint (auto-detects if omitted)")
    parser.add_argument("--demo", "-d", type=str, default=None,
                        choices=list(DEMOS.keys()),
                        help="Pre-build a filesystem")
    args = parser.parse_args()
    interactive(args)
