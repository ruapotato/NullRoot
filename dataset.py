"""
Live-generating PyTorch dataset for bash terminal sessions.

Phase 2: Commands are served one at a time, not packed into context windows.
Each session is a self-contained ordered list of command exchanges. Memory
carries state between commands — there is no token history across commands.

Each command exchange is complete: <prompt>cmd<eoi><output>response<eor>
The model sees the full command and must generate the response using only
its current tokens + memory from prior commands. No truncation, no splitting
exchanges across boundaries.
"""

import threading
import queue
import json
import torch
from torch.utils.data import IterableDataset

from tokenizer import BashTokenizer
from generator import SessionGenerator


def _build_labels(ids: list[int], tokenizer: BashTokenizer) -> tuple[list[int], list[float]]:
    """Build labels that only train on output/error responses, not input commands.

    Returns (labels, weights):
    - labels: list[int] — token ids or -100 for fully masked positions
    - weights: list[float] — per-token loss weight
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

    LOW_WEIGHT = 0.1

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
            if i + 1 < n and ids[i + 1] == eor_id:
                weights[i] = LOW_WEIGHT
                weights[i + 1] = LOW_WEIGHT
                i += 1
        elif tok_id == pad_id:
            labels[i] = -100
            weights[i] = 0.0

        i += 1

    return labels, weights


def split_session_into_commands(transcript: str, tokenizer: BashTokenizer) -> list[dict]:
    """Split a session transcript into individual command exchanges.

    Each command exchange is a complete unit: <prompt>...<eoi><output>...<eor>
    or <prompt>...<eoi><err><eor>. Returns a list of dicts, each with:
      - 'ids': token ids for this complete exchange
      - 'labels': training labels (-100 for masked positions)
      - 'weights': per-token loss weights

    The session must be self-contained. Commands are returned in order so
    memory can carry state forward during training.
    """
    # Remove trailing <eos> for splitting, we don't need it per-command
    text = transcript
    if text.endswith("<eos>"):
        text = text[:-5]

    # Split on <prompt> boundaries — each chunk is one command exchange
    parts = text.split("<prompt>")
    commands = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Re-add the <prompt> prefix
        exchange = "<prompt>" + part

        # Verify this is a complete exchange (has <eor>)
        if "<eor>" not in exchange:
            continue

        ids = tokenizer.encode(exchange)
        labels, weights = _build_labels(ids, tokenizer)

        commands.append({
            "ids": ids,
            "labels": labels,
            "weights": weights,
        })

    return commands


class BashSessionDataset(IterableDataset):
    """Infinite-length dataset yielding complete sessions as command lists.

    Each yielded item is a list of command exchanges (dicts with ids/labels/weights).
    Commands within a session are in order — memory must be carried forward
    sequentially. Sessions are self-contained: no state leaks between sessions.

    Background threads generate sessions continuously so the GPU never stalls.
    """

    def __init__(
        self,
        buffer_size: int = 32,
        workers: int = 4,
        min_ops: int = 300,
        target_ops: int = 800,
        error_rate: float = 0.05,
        base_seed: int = 42,
        commands: set[str] | None = None,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.workers = workers
        self.min_ops = min_ops
        self.target_ops = target_ops
        self.error_rate = error_rate
        self.base_seed = base_seed
        self.commands = commands

        self.tokenizer = BashTokenizer()

        self._queue: queue.Queue | None = None
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def _worker_loop(self, worker_id: int):
        """Background worker: generate complete sessions split into commands."""
        seed_counter = self.base_seed + worker_id * 10_000_000
        tok = BashTokenizer()

        while not self._stop_event.is_set():
            seed_counter += 1

            gen = SessionGenerator(
                min_ops=self.min_ops,
                target_ops=self.target_ops,
                error_rate=self.error_rate,
                seed=seed_counter,
                commands=self.commands,
            )
            transcript = gen.generate()

            try:
                cmds = split_session_into_commands(transcript, tok)
            except ValueError:
                continue

            if not cmds:
                continue

            try:
                self._queue.put(cmds, timeout=1.0)
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
        try:
            while True:
                try:
                    yield self._queue.get(timeout=10.0)
                except queue.Empty:
                    continue
        except GeneratorExit:
            return

    def __del__(self):
        self.stop()


class BashValidationDataset(torch.utils.data.Dataset):
    """Fixed validation dataset loaded from a JSONL file.

    Each sample is a complete session split into ordered command exchanges.
    """

    def __init__(self, path: str):
        self.tokenizer = BashTokenizer()
        self.samples = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cmds = split_session_into_commands(record["transcript"], self.tokenizer)
                if cmds:
                    self.samples.append(cmds)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    import time

    print("Testing Phase 2 BashSessionDataset (command-by-command)...")
    ds = BashSessionDataset(buffer_size=8, workers=4, min_ops=10, target_ops=30)

    t0 = time.time()
    total_cmds = 0
    total_tokens = 0

    for i, session in enumerate(ds):
        n_cmds = len(session)
        cmd_tokens = [len(cmd["ids"]) for cmd in session]
        total_cmds += n_cmds
        total_tokens += sum(cmd_tokens)

        if i < 3:
            print(f"\n  Session {i}: {n_cmds} commands")
            for j, cmd in enumerate(session[:5]):
                tok = BashTokenizer()
                text = tok.decode(cmd["ids"])
                trained = sum(1 for l in cmd["labels"] if l != -100)
                print(f"    Cmd {j}: {len(cmd['ids'])} tokens, {trained} trained | {text[:80]}")
            if n_cmds > 5:
                print(f"    ... ({n_cmds - 5} more)")

        if i >= 9:
            break

    elapsed = time.time() - t0
    ds.stop()

    print(f"\n  10 sessions in {elapsed:.2f}s")
    print(f"  Total commands: {total_cmds}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg commands/session: {total_cmds / 10:.0f}")
    print(f"  Avg tokens/command: {total_tokens / total_cmds:.1f}")

    # Verify self-containment: check a session's labels
    print(f"\n  Verifying command exchange completeness...")
    tok = BashTokenizer()
    test_session = [
        "<prompt>mkdir foo<eoi><output><eor>",
        "<prompt>cd foo<eoi><output><eor>",
        "<prompt>ls<eoi><output><eor>",
    ]
    test_transcript = "".join(test_session) + "<eos>"
    cmds = split_session_into_commands(test_transcript, tok)
    print(f"  Split into {len(cmds)} commands")
    for j, cmd in enumerate(cmds):
        text = tok.decode(cmd["ids"])
        trained = sum(1 for l in cmd["labels"] if l != -100)
        masked = sum(1 for l in cmd["labels"] if l == -100)
        print(f"    [{j}] {text!r}")
        print(f"         {len(cmd['ids'])} tokens: {trained} trained, {masked} masked")

    # Test validation dataset
    import os
    val_path = os.path.join(os.path.dirname(__file__) or ".", "data", "validation.jsonl")
    if os.path.exists(val_path):
        print(f"\nTesting BashValidationDataset from {val_path}...")
        vds = BashValidationDataset(val_path)
        print(f"  {len(vds)} validation sessions")
        session0 = vds[0]
        print(f"  Session 0: {len(session0)} commands")
        print(f"  First command: {tok.decode(session0[0]['ids'])!r}")
