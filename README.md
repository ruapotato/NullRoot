# NullRoot

A ~20M parameter transformer that learns to simulate a programmable bash filesystem from scratch. No real data — every training session is synthetically generated. The model tracks filesystem state across commands by reading and writing compact per-directory state pages as tokens.

## Quick start

```bash
git clone https://github.com/ruapotato/NullRoot
cd NullRoot
pip install torch                          # PyTorch with CUDA

# Try the reference simulator (no GPU needed, perfect execution)
python nullroot_sim.py --demo unix

# Try the neural network (requires CUDA GPU)
python sample.py --demo unix

# Verify the data pipeline
python verify_state.py

# Train from scratch (single stage, all commands at once)
python curriculum.py --min-ops 20 --target-ops 50
```

## How it works

**Paged context:** each directory is its own state page. The model only sees the current directory — never the whole filesystem. `cd` swaps pages. State size is bounded by a single directory, not total filesystem complexity.

**State patches:** the model outputs a response AND a minimal diff to update the current page. `mkdir foo` produces a patch adding `foo/` to children. `echo hello > file.txt` produces a patch with the file content. Read-only commands (`ls`, `cat`) produce `<nop>`.

**Static registers:** 32 learned vectors prepended to every input, processed through all layers alongside real tokens, then stripped at output. A fixed scratchpad for reasoning at 0.08% parameter cost.

```
Input:  <state>@/home/alice#$?=0#children:docs/ readme.txt#readme.txt>hello<eor>
        <prompt>cat readme.txt<eoi>

Output: <output>hello<eor><nop>
```

All commands trained simultaneously in a single stage — no curriculum. 15/15 gate tests passed at step 25K. 65/70 on extended tests (93%).

## Supported commands

### Filesystem

| Command | Example | What it does |
|---------|---------|-------------|
| `mkdir` | `mkdir foo` | Create directory |
| `cd` | `cd foo`, `cd ..`, `cd /path` | Change directory (page swap) |
| `ls` | `ls` | List current directory |
| `pwd` | `pwd` | Print working directory |
| `touch` | `touch file.txt` | Create empty file |
| `echo >` | `echo hello > file.txt` | Write to file |
| `echo >>` | `echo more >> file.txt` | Append to file |
| `cat` | `cat file.txt` | Read file contents |
| `rm` | `rm file.txt` | Remove file |
| `cp` | `cp src.txt dst.txt` | Copy file |
| `mv` | `mv old.txt new.txt` | Move/rename file |
| `head` | `head file.txt` | First line of file |
| `wc` | `wc file.txt` | Line/word/char count |
| `grep` | `grep pattern file.txt` | Search file contents |

### Programming

| Command | Example | What it does |
|---------|---------|-------------|
| `x=val` | `x=42` | Set variable |
| `echo $x` | `echo $x` | Read variable |
| `export` | `export name=val` | Set variable |
| `expr` | `expr 3 + 5` | Integer math (+, -, *) |

## Page format

Each directory is a separate page:

```
$?=0#$x=42#children:docs/ readme.txt#readme.txt>hello world
```

- `$key=value` = variables (global, prepended to every page)
- `children:NAME NAME` = directory contents (dirs end with `/`)
- `NAME>CONTENT` = file content (newlines escaped as `\n`)

Pages are cached by Python — `cd ..` restores the parent page, no model regeneration needed.

## Architecture

| | |
|---|---|
| Parameters | ~20.5M |
| Hidden dim | 512 |
| Layers | 6 |
| Heads | 8 |
| FFN | SwiGLU, 1536 intermediate |
| Registers | 32 static prefix (16K params) |
| Position encoding | RoPE (theta=500K), registers excluded |
| Attention | Flash (PyTorch SDPA) |
| Vocab | 96 tokens |
| Precision | bf16 |
| Hardware | Single RTX 3090 24GB |

## Training

All commands trained simultaneously in a single stage. No curriculum — the model learns filesystem ops, variables, and math all at once. Training data generated on the fly from a reference simulator.

```bash
python curriculum.py --min-ops 20 --target-ops 50
```

### Training parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--min-ops` | 20 | Minimum commands per training session |
| `--target-ops` | 50 | Target commands per session |
| `--gate-check-every` | 500 | Steps between gate evaluations |
| Learning rate | 3e-4 → 3e-5 | Cosine decay with 100-step warmup |
| Optimizer | AdamW | betas=(0.9, 0.95), weight_decay=0.1 |
| Grad clip | 1.0 | Max gradient norm |
| Data workers | 4 | Background generation threads |
| Seed | 42 | Base random seed |

### Reproducing

```bash
pip install torch

# Verify data pipeline (100 sessions, 8K+ commands, zero errors)
python verify_state.py

# Train from scratch
python curriculum.py --min-ops 20 --target-ops 50

# Resume after crash
python curriculum.py --min-ops 20 --target-ops 50 \
    --resume-ckpt checkpoints/latest.pt
```

### Gate tests

Training runs until all 15 gate tests pass: mkdir, cd, ls, pwd, echo, cat, append, rm, cp, mv, head, wc, variables, math (addition + multiplication), and deep navigation (3 levels). Checkpoint saved at every evaluation.

## Results

**v4 (paged context + registers): 15/15 gates at step 25K. 65/70 extended tests (93%).**

| Category | Status |
|----------|--------|
| Filesystem (mkdir, cd, ls, pwd, rm) | Perfect |
| Files (echo, cat, head, append) | Perfect |
| Copy, move | Perfect |
| Variables (set, expand) | Perfect |
| Math (expr +, -, *) | 4/5 |
| Deep navigation (3 levels, cd ..) | Perfect |
| Cross-directory isolation | Perfect |
| Grep | Working |
| Word count | Approximate |

## Files

| File | What it does |
|------|-------------|
| `tokenizer.py` | 96-token vocabulary |
| `paged_fs.py` | Paged filesystem with page table and state patches |
| `generator.py` | Core filesystem + shell simulator |
| `dataset.py` | Training data generation with paged context |
| `model.py` | Transformer with static prefix registers |
| `curriculum.py` | Training loop with gate tests |
| `sample.py` | Interactive shell (auto-detects checkpoint) |
| `nullroot_sim.py` | Reference simulator with demos |
| `verify_state.py` | Data pipeline verification |
| `visualize.py` | UMAP visualizations of model internals |
| `dashboard.py` | Live training dashboard |

## The state-patch pattern

The core idea generalizes beyond bash:

```
State:  [structured representation of current world]
Input:  [action or event]
Output: [response] + [minimal state diff]
```

The model learns to read structured state, process actions, and produce minimal updates. The state format and domain can be anything. Key properties:

- **State scales with complexity, not history** — mkdir+rm 100 times = small state
- **Paged context** — only load what's relevant, not the whole world
- **Patches are minimal** — only changed entries
- **Fully in token space** — no special memory modules, just a transformer
- **Python manages the page table** — model handles fuzzy reasoning, code handles deterministic state
