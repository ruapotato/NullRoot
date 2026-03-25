# NullRoot

A ~20M parameter transformer that learns to simulate a programmable bash filesystem from scratch. No real data — every training session is synthetically generated. The model tracks filesystem state (directories, files, contents, variables, cwd) across commands by reading and writing a compact state representation as tokens.

## Quick start

```bash
git clone https://github.com/ruapotato/NullRoot
cd NullRoot
pip install torch                          # PyTorch with CUDA

# Try the reference simulator (no GPU needed, perfect execution)
python nullroot_sim.py --demo unix

# Try the neural network (requires CUDA GPU)
python sample.py checkpoints/nullroot_v3.pt --demo unix

# Verify the data pipeline
python verify_state.py

# Train from scratch
python curriculum.py --min-ops 40 --target-ops 77
```

## How it works

Each command cycle:

```
Input:  [state tokens] <prompt>command<eoi>
Output: <output>response<eor> <state>patch<eor>
```

1. The model reads the current state (filesystem + variables, serialized as tokens)
2. Processes the command using standard causal self-attention
3. Produces the correct bash response
4. Produces a **state patch** — only the parts of the state that changed

The patch is applied to produce the state for the next command. The model never sees old commands — only the current state + current command. State grows with complexity, not command history.

## Try it

### Reference simulator (Python — perfect execution)

```bash
python nullroot_sim.py                    # bare shell
python nullroot_sim.py --demo unix        # pre-built Unix filesystem
python nullroot_sim.py --demo project     # project directory
python nullroot_sim.py --demo scripting   # variables, math, scripts
```

### Neural network version (requires trained checkpoint + CUDA GPU)

```bash
python sample.py checkpoints/nullroot_v3.pt
python sample.py checkpoints/nullroot_v3.pt --demo unix
```

## Supported commands

### Filesystem

| Command | Example | What it does |
|---------|---------|-------------|
| `mkdir` | `mkdir foo` | Create directory |
| `cd` | `cd foo`, `cd ..`, `cd /path` | Change directory |
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
| `find` | `find .` | List all files recursively |
| `grep` | `grep pattern file.txt` | Search file contents |

### Programming

| Command | Example | What it does |
|---------|---------|-------------|
| `x=val` | `x=42` | Set variable |
| `echo $x` | `echo $x` | Read variable |
| `export` | `export name=val` | Set variable |
| `expr` | `expr 3 + 5` | Integer math (+, -, *) |
| `sh` | `sh script.sh` | Execute script file |
| `test` | `test -f file.txt` | Test conditions (-f, -d, =, !=) |

## State format

The state is serialized compactly:

```
@/home/alice#$?=0#$x=42#/:etc home#/home:alice#/home/alice:notes.txt#/home/alice/notes.txt>hello world
```

- `@path` = current working directory
- `$key=value` = variables (including `$?` for exit status)
- `#` = entry separator
- `path:children` = directory with its contents
- `path>content` = file with its content (newlines escaped as `\n`)

After `mkdir foo`, the patch is just `/:foo#/foo:` — only the changed entries. After `cd foo`, just `@/foo`. After `x=5`, just `$x=5`. Read-only commands (`ls`, `cat`, `expr`) produce `<nop>`.

## Architecture

| | |
|---|---|
| Parameters | ~20.5M |
| Hidden dim | 512 |
| Layers | 6 |
| Heads | 8 |
| FFN | SwiGLU, 1536 intermediate |
| Position encoding | RoPE (theta=500K, supports 131K) |
| Attention | Flash (PyTorch SDPA) |
| Vocab | 96 tokens (19 commands + 10 control flow + a-z + 0-9 + punctuation + special) |
| Precision | bf16 |
| Hardware | Single RTX 3090 24GB |

## Training

All training data is generated on the fly — no datasets, no disk I/O. The simulator tracks full state and produces ground-truth responses and state patches.

```bash
# Phase 3: full programmable system (current)
python curriculum.py --min-ops 40 --target-ops 77

# Monitor training
python dashboard.py   # http://localhost:5000
```

### Training parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--min-ops` | 40 | Minimum commands per training session |
| `--target-ops` | 77 | Target commands per session (gaussian around this) |
| `--gate-check-every` | 500 | Steps between gate test evaluations |
| Learning rate | 3e-4 → 3e-5 | Cosine decay with 100-step warmup |
| Optimizer | AdamW | betas=(0.9, 0.95), weight_decay=0.1 |
| Grad clip | 1.0 | Max gradient norm |
| Data workers | 4 | Background session generation threads |
| Seed | 42 | Base random seed for reproducibility |

### Reproducing training

```bash
# Install dependencies
pip install torch  # tested with PyTorch 2.x, CUDA

# Verify data generation is correct
python verify_state.py

# Start training from scratch
python curriculum.py --min-ops 40 --target-ops 77

# Resume from checkpoint after crash
python curriculum.py --min-ops 40 --target-ops 77 \
    --resume-ckpt checkpoints/latest.pt --resume-stage 0
```

### Gate tests

Training runs until **all** gate tests pass (13 meaningful tests covering every command type). A `latest.pt` checkpoint is saved at every gate evaluation for crash recovery. Gate tests include:

- Directory creation, navigation, listing
- File creation, writing, reading, appending
- Copy, move, remove
- Head, word count
- Variable assignment and expansion
- Integer arithmetic
- Script execution

## Files

| File | What it does |
|------|-------------|
| `tokenizer.py` | 96-token vocabulary, encode/decode |
| `generator.py` | Filesystem + shell simulator with state serialization and patching |
| `dataset.py` | PyTorch dataset, background threads, state-patch training samples |
| `model.py` | Transformer (RoPE, SwiGLU, flash attention, gradient checkpointing) |
| `curriculum.py` | Training loop with gate tests and checkpointing |
| `sample.py` | Neural network interactive shell with demos |
| `nullroot_sim.py` | Reference simulator (Python) with demos |
| `verify_state.py` | Verifies state-patch system correctness (100 sessions, 8K+ commands) |
| `dashboard.py` | Live training dashboard with charts |
| `sweep.py` | Architecture sweep testing |

## Status

**Phase 3: 47/53 tests passing (89%).** 21 command types trained simultaneously at step 14K. Results:

| Category | Status | Details |
|----------|--------|---------|
| Filesystem | **Perfect** | mkdir, cd, ls, pwd, touch, rm, cp, mv — all correct |
| Files | **Perfect** | echo >, echo >>, cat, head — including multiline content |
| Variables | **Perfect** | `x=42` then `echo $x` → `42` |
| Scripts | **Working** | `echo echo hello > test.sh` then `sh test.sh` → `hello` |
| Absolute paths | **Working** | `cat /path/to/file`, `cd /deep/path` |
| Deep navigation | **Mostly** | 3+ levels deep occasionally garbles paths |
| Math | **Learning** | Outputs numbers but wrong — pattern error, needs more training |
| find | **Partial** | Complex recursive output still garbled |

**Phase 2 complete.** State-patch architecture proven on filesystem-only task (13/13, step 37K).

## The state-patch pattern

The core idea behind NullRoot generalizes beyond bash:

```
State:  [structured representation of current world]
Input:  [action or event]
Output: [response] + [minimal state diff]
```

The model learns to **read structured state, process actions, and produce minimal updates**. The state format and domain can be anything — filesystem, game world, database, API session. The key properties:

- **State scales with complexity, not history** — mkdir+rm 100 times = small state
- **Patches are minimal** — only changed entries, not full rewrites
- **Fully in token space** — no special memory modules, just a transformer reading and writing tokens
- **Verified deterministic** — reference simulator proves correctness before training
