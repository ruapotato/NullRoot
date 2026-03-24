# NullRoot

A ~20M parameter transformer that learns to simulate a bash filesystem from scratch. No real data — every training session is synthetically generated. The model tracks filesystem state (directories, files, contents, cwd) across commands by reading and writing a compact state representation as tokens.

## How it works

Each command cycle:

```
Input:  [state tokens] <prompt>command<eoi>
Output: <output>response<eor> <state>patch<eor>
```

1. The model reads the current filesystem state (serialized as tokens)
2. Processes the command using standard causal self-attention
3. Produces the correct bash response
4. Produces a **state patch** — only the parts of the filesystem that changed

The patch is applied to produce the state for the next command. The model never sees old commands — only the current state + current command. State grows with filesystem complexity, not command history.

## Try it

Interactive shell:

```bash
python sample.py checkpoints/nullroot_v2.pt
```

With a pre-built Unix filesystem:

```bash
python sample.py checkpoints/nullroot_v2.pt --demo unix
```

With a project directory:

```bash
python sample.py checkpoints/nullroot_v2.pt --demo project
```

## Supported commands

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

## State format

The filesystem state is serialized compactly:

```
@/home/alice#/:etc home tmp#/home:alice bob#/home/alice:notes.txt#/home/alice/notes.txt>hello world
```

- `@` = current working directory
- `#` = entry separator
- `path:children` = directory with its contents
- `path>content` = file with its content

After `mkdir foo`, the patch is just `/:foo#/foo:` — only the changed entries. After `cd foo`, just `@/foo`. Read-only commands (`ls`, `cat`) produce `<nop>`.

## Architecture

| | |
|---|---|
| Parameters | ~20.5M |
| Hidden dim | 512 |
| Layers | 6 |
| Heads | 8 |
| FFN | SwiGLU, 1536 intermediate |
| Position encoding | RoPE (theta=500K) |
| Attention | Flash (PyTorch SDPA) |
| Vocab | 71 tokens (14 commands + a-z + 0-9 + punctuation + special) |
| Precision | bf16 |

## Training

All training data is generated on the fly — no datasets, no disk I/O. The simulator tracks full filesystem state and produces ground-truth responses and state patches.

```bash
python curriculum.py --min-ops 10 --target-ops 30
```

Monitor training:

```bash
python dashboard.py   # http://localhost:5000
```

## Files

| File | What it does |
|------|-------------|
| `tokenizer.py` | 71-token vocabulary, encode/decode |
| `generator.py` | Filesystem simulator with state serialization and patching |
| `dataset.py` | PyTorch dataset, background threads, state-patch training samples |
| `model.py` | Transformer (RoPE, SwiGLU, flash attention) |
| `curriculum.py` | Training with gate tests |
| `sample.py` | Interactive shell with demos |
| `dashboard.py` | Live training dashboard |
| `sweep.py` | Architecture sweep testing |

## Status

**Phase 2 complete.** The model learns all 14 commands in a single training stage, achieving 13/13 on gate tests at step 37K. The state-patch architecture solves the horizon problem — state size scales with filesystem complexity, not command history length.
