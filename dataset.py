"""
Live-generating PyTorch dataset for bash terminal sessions.

No disk I/O — sessions are generated on the fly in background threads and
buffered so the GPU never stalls. Every session is unique.

Short sessions are packed end-to-end (separated by <eos>) to fill the
context window and waste no padding.
"""

import threading
import queue
import json
import torch
from torch.utils.data import IterableDataset

from tokenizer import BashTokenizer
from generator import SessionGenerator


def _build_labels(ids: list[int], tokenizer: BashTokenizer) -> list[int]:
    """Build labels that only train on output/error responses, not input commands.

    Returns (labels, weights):
    - labels: list[int] — token ids or -100 for fully masked positions
    - weights: list[float] — per-token loss weight

    Masking strategy:
    - <prompt> through <eoi> (inclusive): masked (-100)
    - Empty responses (<output><eor>): low weight (0.1) — still learned
      but doesn't dominate gradient
    - <output> with actual content then <eor>: full weight (1.0)
    - <err><eor>: full weight (1.0) — error decisions are meaningful
    - <eos>: full weight (1.0)
    - <pad>: masked (-100)
    """
    prompt_id = tokenizer.prompt_id
    eoi_id = tokenizer.eoi_id
    output_id = tokenizer.output_id
    eor_id = tokenizer.eor_id
    pad_id = tokenizer.pad_id

    labels = list(ids)
    weights = [1.0] * len(ids)
    n = len(ids)
    in_prompt = False

    LOW_WEIGHT = 0.1  # weight for empty responses

    i = 0
    while i < n:
        tok_id = ids[i]

        if tok_id == prompt_id:
            labels[i] = -100
            weights[i] = 0.0
            in_prompt = True
        elif in_prompt:
            labels[i] = -100
            weights[i] = 0.0
            if tok_id == eoi_id:
                in_prompt = False
        elif tok_id == output_id:
            # Check if empty response: <output><eor>
            if i + 1 < n and ids[i + 1] == eor_id:
                weights[i] = LOW_WEIGHT
                weights[i + 1] = LOW_WEIGHT
                i += 1
            # else: content response — keep full weight
        elif tok_id == pad_id:
            labels[i] = -100
            weights[i] = 0.0

        i += 1

    return labels, weights


class BashSessionDataset(IterableDataset):
    """Infinite-length IterableDataset that generates sessions on the fly.

    Background threads continuously generate and tokenize sessions, packing
    multiple sessions into each context window when they're short enough.
    The main thread (dataloader) pulls from the queue, so the GPU never stalls.

    Each sample is a (token_ids, labels) pair of length seq_len.
    Labels are masked so the model only learns to predict output responses,
    not input commands. The model handles causal shift internally.
    """

    def __init__(
        self,
        seq_len: int = 65536,
        buffer_size: int = 32,
        workers: int = 4,
        min_ops: int = 300,
        target_ops: int = 800,
        error_rate: float = 0.05,
        base_seed: int = 42,
        commands: set[str] | None = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.workers = workers
        self.min_ops = min_ops
        self.target_ops = target_ops
        self.error_rate = error_rate
        self.base_seed = base_seed
        self.commands = commands

        self.tokenizer = BashTokenizer()
        self.pad_id = self.tokenizer.pad_id
        self.eos_id = self.tokenizer.eos_id

        self._queue: queue.Queue | None = None
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def _worker_loop(self, worker_id: int):
        """Background worker: generate complete sessions that fit in seq_len."""
        seed_counter = self.base_seed + worker_id * 10_000_000
        tok = BashTokenizer()
        pad_id = tok.pad_id

        while not self._stop_event.is_set():
            seed_counter += 1

            # Generate a session sized to fit in seq_len
            # Estimate: ~12 tokens per command, leave room for padding
            max_ops_for_window = max(10, (self.seq_len - 10) // 12)
            target = min(self.target_ops, max_ops_for_window)
            min_o = min(self.min_ops, target)

            gen = SessionGenerator(
                min_ops=min_o,
                target_ops=target,
                error_rate=self.error_rate,
                seed=seed_counter,
                commands=self.commands,
            )
            transcript = gen.generate()

            try:
                ids = tok.encode(transcript)
            except ValueError:
                continue

            # Truncate if still too long (shouldn't happen often)
            if len(ids) > self.seq_len:
                ids = ids[:self.seq_len]

            labels, weights = _build_labels(ids, tok)

            # Pad to seq_len
            pad_len = self.seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [pad_id] * pad_len
                labels = labels + [-100] * pad_len
                weights = weights + [0.0] * pad_len

            token_ids = torch.tensor(ids, dtype=torch.long)
            labels_t = torch.tensor(labels, dtype=torch.long)
            weights_t = torch.tensor(weights, dtype=torch.float32)

            try:
                self._queue.put((token_ids, labels_t, weights_t), timeout=1.0)
            except queue.Full:
                if self._stop_event.is_set():
                    return

    def _start_workers(self):
        if self._queue is not None:
            return
        self._stop_event.clear()
        self._queue = queue.Queue(maxsize=self.buffer_size)
        self._threads = []
        for i in range(self.workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads = []
        self._queue = None

    def __iter__(self):
        self._start_workers()
        while True:
            try:
                yield self._queue.get(timeout=10.0)
            except queue.Empty:
                continue

    def __del__(self):
        self.stop()


class BashValidationDataset(torch.utils.data.Dataset):
    """Fixed validation dataset loaded from a JSONL file."""

    def __init__(self, path: str, seq_len: int = 65536):
        self.seq_len = seq_len
        self.tokenizer = BashTokenizer()
        self.pad_id = self.tokenizer.pad_id
        self.samples = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ids = self.tokenizer.encode(record["transcript"])
                self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = list(self.samples[idx])

        if len(ids) >= self.seq_len:
            ids = ids[: self.seq_len]

        labels, weights = _build_labels(ids, self.tokenizer)

        # Pad
        pad_len = self.seq_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len
            weights = weights + [0.0] * pad_len

        token_ids = torch.tensor(ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        weights_t = torch.tensor(weights, dtype=torch.float32)

        return token_ids, labels_t, weights_t


if __name__ == "__main__":
    import time

    seq_len = 65536
    print(f"Testing BashSessionDataset (seq_len={seq_len}, packing enabled)...")
    print(f"  Label masking: only training on output responses, not input commands")
    ds = BashSessionDataset(seq_len=seq_len, buffer_size=8, workers=4)

    t0 = time.time()
    total_tokens = 0
    total_trained = 0
    for i, (token_ids, labels) in enumerate(ds):
        non_pad = (token_ids != ds.pad_id).sum().item()
        eos_count = (token_ids == ds.eos_id).sum().item()
        trained = (labels != -100).sum().item()
        total_tokens += non_pad
        total_trained += trained
        if i < 5:
            print(f"  Sample {i}: {non_pad:,} tokens, "
                  f"{eos_count} sessions packed, "
                  f"{trained:,} trained / {non_pad:,} total ({trained/max(non_pad,1)*100:.0f}%)")
        if i >= 19:
            break

    elapsed = time.time() - t0
    ds.stop()

    print(f"\n  20 samples in {elapsed:.2f}s ({20/elapsed:.1f} samples/s)")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Trained tokens: {total_trained:,} ({total_trained/total_tokens*100:.0f}%)")
    print(f"  Utilization: {total_tokens / (20 * seq_len) * 100:.1f}%")

    # Verify masking is correct on a small example
    print(f"\n  Verifying label masking on small example...")
    tok = BashTokenizer()
    test = "<prompt>mkdir foo\n<output>\n<prompt>cat bar\n<err>\n<eos>"
    ids = tok.encode(test)
    labels = _build_labels(ids, tok)
    print(f"  Input:  {test!r}")
    for j, (tid, lid) in enumerate(zip(ids, labels)):
        token_str = tok.decode([tid])
        trained = "TRAIN" if lid != -100 else "MASK "
        print(f"    [{j:2d}] {trained} {token_str!r}")

    # Test validation dataset
    import os
    val_path = os.path.join(os.path.dirname(__file__) or ".", "data", "validation.jsonl")
    if os.path.exists(val_path):
        print(f"\nTesting BashValidationDataset from {val_path}...")
        vds = BashValidationDataset(val_path, seq_len=seq_len)
        print(f"  {len(vds)} validation samples")
        tok_t, lab = vds[0]
        non_pad = (tok_t != vds.pad_id).sum().item()
        trained = (lab != -100).sum().item()
        masked = (lab == -100).sum().item()
        print(f"  Sample 0: {non_pad:,} real tokens, {trained:,} trained, {masked:,} masked")
