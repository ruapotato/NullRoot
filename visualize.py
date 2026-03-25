"""
UMAP visualizations of NullRoot internals.

1. Token embedding space — how the model organizes its vocabulary
2. Hidden state trajectories — how representations evolve through layers during a session
3. Command clustering — how different commands are represented internally
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.amp import autocast
from umap import UMAP

from tokenizer import BashTokenizer
from model import BashTransformer, BashTransformerConfig
from generator import FileSystem


def load_model(path="checkpoints/nullroot_v3.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config", BashTransformerConfig())
    model = BashTransformer(config).to(device).to(torch.bfloat16)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    return model, device


def plot_token_embeddings(model, save="viz_token_embeddings.png"):
    """UMAP of the 96 token embeddings."""
    tok = BashTokenizer()
    embeddings = model.embed_tokens.weight.detach().float().cpu().numpy()

    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(embeddings)

    # Color by category
    categories = {}
    fs_cmds = {"ls", "cd", "pwd", "touch", "mkdir", "rm", "cat", "echo", "cp", "mv", "head", "wc", "find", "grep"}
    prog_cmds = {"sh", "export", "exit", "test", "expr"}
    control = {"if", "then", "else", "fi", "while", "do", "done", "for", "in", "true"}
    special = {"<prompt>", "<eoi>", "<output>", "<err>", "<eor>", "<pad>", "<eos>", "<state>", "<nop>"}

    colors = []
    labels = []
    for i in range(tok.vocab_size):
        t = tok.id_to_token[i]
        labels.append(t)
        if t in fs_cmds:
            colors.append("red")
        elif t in prog_cmds:
            colors.append("orange")
        elif t in control:
            colors.append("purple")
        elif t in special:
            colors.append("green")
        elif t.isalpha() and len(t) == 1:
            colors.append("steelblue")
        elif t.isdigit():
            colors.append("cyan")
        else:
            colors.append("gray")

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=80, alpha=0.7)

    for i, label in enumerate(labels):
        disp = label if len(label) <= 6 else label[:6]
        ax.annotate(disp, (coords[i, 0], coords[i, 1]),
                    fontsize=7, ha="center", va="bottom", alpha=0.8)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="red", label="Filesystem cmds"),
        Patch(color="orange", label="Programming cmds"),
        Patch(color="purple", label="Control flow"),
        Patch(color="green", label="Special tokens"),
        Patch(color="steelblue", label="Letters"),
        Patch(color="cyan", label="Digits"),
        Patch(color="gray", label="Punctuation"),
    ]
    ax.legend(handles=legend, loc="upper right")
    ax.set_title("NullRoot Token Embedding Space (UMAP)")
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f"Saved {save}")


def plot_hidden_states(model, device, save="viz_hidden_states.png"):
    """UMAP of hidden states at each layer during a session."""
    tok = BashTokenizer()
    fs = FileSystem()

    # Run a session and capture hidden states at each layer
    commands = [
        "mkdir etc", "mkdir home", "cd etc",
        "echo 127.0.0.1 localhost > hosts",
        "cat hosts", "cd /", "ls", "pwd",
        "x=42", "echo $x", "expr 5 + 3",
        "cd home", "mkdir alice", "cd alice",
        "echo hello > readme.txt", "cat readme.txt",
    ]

    all_hidden = []  # (layer, cmd_type, hidden_mean)
    cmd_labels = []
    layer_labels = []

    state = ""
    for cmd in commands:
        input_text = ""
        if state:
            input_text += f"<state>{state}<eor>"
        input_text += f"<prompt>{cmd}<eoi>"
        ids = tok.encode(input_text)
        input_t = torch.tensor([ids], dtype=torch.long, device=device)

        # Hook into each layer to capture hidden states
        layer_outputs = []

        def make_hook(storage):
            def hook(module, input, output):
                storage.append(output.detach().float().cpu())
            return hook

        hooks = []
        for layer in model.layers:
            storage = []
            layer_outputs.append(storage)
            h = layer.register_forward_hook(make_hook(storage))
            hooks.append(h)

        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                model(input_t)

        for h in hooks:
            h.remove()

        # Mean-pool hidden state per layer
        cmd_type = cmd.split()[0]
        if "=" in cmd and not cmd.startswith("echo"):
            cmd_type = "var_set"
        elif cmd.startswith("echo $"):
            cmd_type = "var_echo"

        for layer_idx, storage in enumerate(layer_outputs):
            if storage:
                hidden_mean = storage[0][0].mean(dim=0).numpy()
                all_hidden.append(hidden_mean)
                cmd_labels.append(cmd_type)
                layer_labels.append(layer_idx)

        fs.execute_command(cmd)
        state = fs.serialize_state()

    all_hidden = np.array(all_hidden)
    reducer = UMAP(n_neighbors=10, min_dist=0.2, random_state=42)
    coords = reducer.fit_transform(all_hidden)

    # Color by command type
    unique_cmds = sorted(set(cmd_labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_cmds))
    cmd_to_color = {c: cmap(i) for i, c in enumerate(unique_cmds)}

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: colored by command type
    ax = axes[0]
    for cmd_type in unique_cmds:
        mask = [c == cmd_type for c in cmd_labels]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=cmd_type,
                   color=cmd_to_color[cmd_type], s=60, alpha=0.7)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Hidden States by Command Type")

    # Plot 2: colored by layer depth
    ax = axes[1]
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=layer_labels, cmap="viridis", s=60, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="Layer")
    ax.set_title("Hidden States by Layer Depth")

    plt.suptitle("NullRoot Hidden State Trajectories (UMAP)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f"Saved {save}")


def plot_state_embeddings(model, device, save="viz_state_embeddings.png"):
    """UMAP of how the model represents different filesystem states."""
    tok = BashTokenizer()

    # Generate a variety of states
    states_and_labels = []

    # Empty states
    for _ in range(5):
        states_and_labels.append(("@/#$?=0#/:", "empty"))

    # Simple states
    for name in ["foo", "bar", "baz", "qux", "test"]:
        states_and_labels.append(
            (f"@/#$?=0#/:{name}#/{name}:", f"1 dir"))

    # States with files
    for name in ["hello", "world", "data", "test", "info"]:
        states_and_labels.append(
            (f"@/#$?=0#/:f.txt#/f.txt>{name}", f"1 file"))

    # Deep states
    states_and_labels.append(
        ("@/a/b#$?=0#/:a#/a:b#/a/b:c#/a/b/c:", "deep"))
    states_and_labels.append(
        ("@/x/y/z#$?=0#/:x#/x:y#/x/y:z#/x/y/z:", "deep"))

    # States with variables
    for v in range(5):
        states_and_labels.append(
            (f"@/#$?=0#$x={v}#/:", f"var x={v}"))

    # Complex states
    states_and_labels.append(
        ("@/home#$?=0#$x=1#/:etc home#/etc:hosts#/home:alice#/home/alice:readme.txt#/etc/hosts>localhost#/home/alice/readme.txt>hello",
         "complex"))

    # Different CWDs same filesystem
    base = "#$?=0#/:a b#/a:#/b:"
    for cwd in ["/", "/a", "/b"]:
        states_and_labels.append((f"@{cwd}{base}", f"cwd={cwd}"))

    # Encode each state and get the model's internal representation
    embeddings = []
    labels = []
    for state_str, label in states_and_labels:
        input_text = f"<state>{state_str}<eor><prompt>ls<eoi>"
        try:
            ids = tok.encode(input_text)
        except ValueError:
            continue
        input_t = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                hidden = model.embed_tokens(input_t)
                cos, sin = model.rotary_emb(hidden)
                for layer in model.layers:
                    hidden = layer(hidden, cos, sin)
                hidden = model.norm(hidden)

        # Mean pool
        emb = hidden[0].float().cpu().mean(dim=0).numpy()
        embeddings.append(emb)
        labels.append(label)

    embeddings = np.array(embeddings)
    reducer = UMAP(n_neighbors=8, min_dist=0.3, random_state=42)
    coords = reducer.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    label_to_color = {l: cmap(i) for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for label in unique_labels:
        mask = [l == label for l in labels]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=label,
                   color=label_to_color[label], s=80, alpha=0.7)
    ax.legend()
    ax.set_title("NullRoot State Embedding Space (UMAP)\nHow the model represents different filesystem states")
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f"Saved {save}")


if __name__ == "__main__":
    print("Loading model...")
    model, device = load_model()

    print("\n1. Token embeddings...")
    plot_token_embeddings(model)

    print("\n2. Hidden state trajectories...")
    plot_hidden_states(model, device)

    print("\n3. State embeddings...")
    plot_state_embeddings(model, device)

    print("\nDone! Check viz_*.png files.")
