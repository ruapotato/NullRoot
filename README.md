# NullRoot

An experiment to see if a small transformer (~20M params) trained from scratch on synthetic data can learn to simulate a bash terminal — tracking filesystem state, resolving paths, and producing correct command output across thousands of operations.

## The question

Can a tiny model learn the *rules* of a filesystem purely from examples? Not pattern matching on common paths, but actually maintaining coherent state: if you `mkdir foo && cd foo && touch bar`, then `ls` should output `bar`. Across 2000+ commands per session, with deeply nested hierarchies and random names it's never seen before.

## Approach

- **No real data.** All training data is generated on the fly by a simulator that tracks full filesystem state. Every session is unique — names are combinatorial gibberish so the model can't memorize them.
- **Tiny vocab.** 63 hand-defined tokens. No BPE, no subword. Commands, lowercase letters, digits, punctuation, and structural tokens.
- **Explicit boundaries.** `<eoi>` and `<eor>` tokens separate user input from model output. `\n` is never a structural delimiter — it's purely a content character for multiline file data.
- **Label masking.** The model only trains on predicting outputs, not commands. It sees the commands in attention (full context) but is only graded on the responses.
- **Curriculum training.** Commands are introduced in stages (mkdir+cd+ls first, then pwd, touch, echo, cat, etc). Each stage must pass a validation gate before advancing.
- **Session packing.** Short sessions are packed end-to-end to fill 65K context windows. No wasted padding.

## Tokenization

63 tokens total, all hand-defined. No BPE, no subword merging.

**Transcript format:**

```
<prompt>mkdir foo<eoi><output><eor><prompt>cat file<eoi><output>hello\nworld<eor><eos>
|---- user input ---|            |---- user input -----|                          |
                     |-- model --|                      |-------- model ----------|
```

**Structural tokens:**
- `<prompt>` — start of user command
- `<eoi>` — end of input (command boundary, system-provided, masked during training)
- `<output>` — model predicts this for successful commands
- `<err>` — model predicts this for invalid commands
- `<eor>` — end of response (model learns when to stop)
- `<eos>` — end of session
- `<pad>` — padding

**Label masking:**
```
<prompt>mkdir foo<eoi>  <output>  <eor>  <prompt>cat bar<eoi>  <err>  <eor>
  MASK   MASK MASK MASK   TRAIN   TRAIN    MASK  MASK MASK MASK TRAIN  TRAIN
```

The model sees everything in attention but only gets gradient on the response tokens. This teaches it: given the full command history, what should the output be?

**Content tokens:** 8 commands (`ls cd pwd touch mkdir rm cat echo`), a-z, 0-9, punctuation (`/ . _ - > >> " space \n`), shell chrome (`@ : #`).

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
| Training context | 65K tokens |
| Precision | bf16 |
| Hardware | Single RTX 3090 24GB |

## Curriculum stages

| Stage | Adds | Gate |
|-------|------|------|
| 1 | `mkdir` `cd` `ls` | val loss < 1.5 |
| 2 | `pwd` | val loss < 1.5 |
| 3 | `touch` | val loss < 1.5 |
| 4 | `echo >` | val loss < 1.5 |
| 5 | `cat` | val loss < 1.5 |
| 6 | `echo >>` | val loss < 1.5 |
| 7 | `rm` | val loss < 1.5 |
| 8 | `<err>` (intentional errors) | val loss < 1.5 |
| 9 | anneal (all commands, low LR decay) | runs to completion |

Once a command is introduced, it stays in the training mix forever.

## Running

Curriculum training (Phase 2, pre-configured for a 3090):

```bash
python curriculum.py --min-ops 30 --target-ops 80
```

Monitor training:

```bash
python dashboard.py   # http://localhost:5000
```

Interactive inference:

```bash
python sample.py checkpoints/final.pt
```

## Files

| File | What it does |
|------|-------------|
| `tokenizer.py` | 63-token vocabulary, encode/decode |
| `generator.py` | Live session generator with filesystem state tracker |
| `dataset.py` | PyTorch `IterableDataset`, background threads, session packing, label masking |
| `model.py` | Transformer (RoPE, SwiGLU, flash attention, gradient checkpointing) |
| `curriculum.py` | Staged training with validation gates |
| `train.py` | Flat training loop (alternative to curriculum) |
| `sample.py` | Interactive inference |
| `dashboard.py` | Flask training dashboard with live charts |
| `verify.py` | Replays transcripts against independent filesystem to prove correctness |
| `gen_validation.py` | Hand-crafted 369-command validation set building a full Unix filesystem |

## What "working" would look like

- Correct `ls` output after arbitrary sequences of `mkdir`, `touch`, `echo >`, `rm`
- Correct `pwd` after chains of `cd`, `cd ..`, `cd /absolute/path`
- Correct `cat` output including multi-line content from `>>` appends
- Appropriate `<err>` for invalid operations (cd to nonexistent dir, cat missing file)
- Coherence over 1000+ command sessions

## Status

**Phase 1: Complete.** The model successfully internalized bash filesystem semantics — `mkdir`, `cd`, `ls`, `pwd`, `touch`, `echo >`, `cat` — with correct state tracking across multi-step sessions. Scored 33/35 on the Stage 5 validation gate; the 2 mismatches were attributed to a trailing-space bug in the gate comparator, not model errors.

Phase 2 (in progress): Memory module with severed autoregressive loop — the model will process each command in isolation, maintaining filesystem state through an explicit learned memory bank instead of token history.
