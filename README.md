# NullRoot

An experiment to see if a small transformer (~20M params) trained from scratch on synthetic data can learn to simulate a bash terminal — tracking filesystem state, resolving paths, and producing correct command output across thousands of operations.

## The question

Can a tiny model learn the *rules* of a filesystem purely from examples? Not pattern matching on common paths, but actually maintaining coherent state: if you `mkdir foo && cd foo && touch bar`, then `ls` should output `bar`. Across 2000+ commands per session, with deeply nested hierarchies and random names it's never seen before.

## Approach

- **No real data.** All training data is generated on the fly by a simulator that tracks full filesystem state. Every session is unique — names are combinatorial gibberish so the model can't memorize them.
- **Tiny vocab.** 61 hand-defined tokens. No BPE, no subword. Commands, lowercase letters, digits, punctuation, and structural tokens.
- **Label masking.** The model only trains on predicting outputs, not commands. It sees the commands in attention (full context) but is only graded on the responses.
- **Session packing.** Short sessions are packed end-to-end to fill 65K context windows. No wasted padding.

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

## Running

Train with defaults (everything is pre-configured for a 3090):

```bash
python train.py
```

Logs go to `checkpoints/train_log.jsonl`. Validation runs every 250 steps.

Resume from checkpoint:

```bash
python train.py --resume checkpoints/step_005000.pt
```

Interactive inference:

```bash
python sample.py checkpoints/final.pt
```

## Files

| File | What it does |
|------|-------------|
| `tokenizer.py` | 61-token vocabulary, encode/decode |
| `generator.py` | Live session generator with filesystem state tracker |
| `dataset.py` | PyTorch `IterableDataset`, background threads, session packing, label masking |
| `model.py` | Transformer (RoPE, SwiGLU, flash attention, gradient checkpointing) |
| `train.py` | Training loop |
| `sample.py` | Interactive inference |
| `verify.py` | Replays transcripts against independent filesystem to prove correctness |
| `gen_validation.py` | Hand-crafted 369-command validation set building a full Unix filesystem |

## What "working" would look like

- Correct `ls` output after arbitrary sequences of `mkdir`, `touch`, `echo >`, `rm`
- Correct `pwd` after chains of `cd`, `cd ..`, `cd /absolute/path`
- Correct `cat` output including multi-line content from `>>` appends
- Appropriate `<err>` for invalid operations (cd to nonexistent dir, cat missing file)
- Coherence over 1000+ command sessions

## Status

Experimental. Training in progress.
