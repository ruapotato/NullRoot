"""
Dataset for paged-context NullRoot.

Each training sample: @CWD#page_state + <prompt>cmd<eoi> → <output>response<eor> <state>patch<eor>
Page state is always the current directory only — tiny context.
cd produces a full page swap as its patch.
"""

import threading
import queue
import copy
import random
import torch
from torch.utils.data import IterableDataset

from tokenizer import BashTokenizer
from paged_fs import PagedFileSystem
from generator import _gen_syllable_name_rng, _random_filename_rng, _random_content


def build_session_samples(commands: list[dict], tok: BashTokenizer) -> list[dict]:
    """Build training samples from a paged session."""
    samples = []

    for cmd_info in commands:
        full_input = cmd_info["input"]
        response = cmd_info["response"]
        patch = cmd_info["patch"]

        # Build output
        if response:
            output_part = f"<output>{response}<eor>"
        else:
            output_part = "<output><eor>"

        if patch:
            state_part = f"<state>{patch}<eor>"
        else:
            state_part = "<nop>"

        full_text = full_input + output_part + state_part

        try:
            ids = tok.encode(full_text)
        except ValueError:
            continue

        # Labels: mask input, train on output + patch
        labels = list(ids)
        weights = [1.0] * len(ids)
        in_input = True
        for j, tid in enumerate(ids):
            if in_input:
                labels[j] = -100
                weights[j] = 0.0
                if tid == tok.eoi_id:
                    in_input = False

        samples.append({"ids": ids, "labels": labels, "weights": weights})

    return samples


class PagedSessionGenerator:
    """Generates training sessions with the paged filesystem."""

    COMMAND_WEIGHTS = {
        "mkdir": 12, "cd": 18, "ls": 15, "pwd": 5,
        "touch": 8, "echo_write": 10, "echo_append": 6,
        "cat": 12, "rm": 4, "cp": 4, "mv": 4,
        "head": 3, "wc": 3, "grep": 3,
        "var_set": 8, "var_echo": 6, "expr_math": 6,
        "export": 3,
    }

    def __init__(self, min_ops=20, target_ops=50, seed=None):
        self.min_ops = min_ops
        self.target_ops = target_ops
        self.rng = random.Random(seed)
        self.fs = PagedFileSystem()

    def generate(self) -> list[dict]:
        num_ops = max(self.min_ops, min(
            self.target_ops * 2,
            int(self.rng.gauss(self.target_ops, self.target_ops * 0.3))))

        results = []
        rng = self.rng
        ops = list(self.COMMAND_WEIGHTS.keys())
        wts = list(self.COMMAND_WEIGHTS.values())

        for _ in range(num_ops):
            op = rng.choices(ops, weights=wts, k=1)[0]
            cmd_info = self._exec_op(op, rng)
            if cmd_info:
                results.append(cmd_info)

        return results

    def _exec_op(self, op, rng) -> dict | None:
        fs = self.fs
        page = fs._current_page()
        cmd_str = None
        response = ""

        if op == "ls":
            cmd_str = "ls"

        elif op == "pwd":
            cmd_str = "pwd"

        elif op == "mkdir":
            name = _gen_syllable_name_rng(rng)
            if name not in page["children"]:
                cmd_str = f"mkdir {name}"

        elif op == "touch":
            name = _random_filename_rng(rng)
            if name not in page["files"]:
                cmd_str = f"touch {name}"

        elif op == "echo_write":
            name = _random_filename_rng(rng)
            content = _random_content(rng)
            cmd_str = f"echo {content} > {name}"

        elif op == "echo_append":
            files = list(page["files"].keys())
            if files and rng.random() < 0.7:
                name = rng.choice(files)
            else:
                name = _random_filename_rng(rng)
            content = _random_content(rng)
            cmd_str = f"echo {content} >> {name}"

        elif op == "cat":
            files = list(page["files"].keys())
            if not files:
                return None
            cmd_str = f"cat {rng.choice(files)}"

        elif op == "cd":
            style = rng.choices(["child", "up"], weights=[10, 6], k=1)[0]
            if style == "child":
                dirs = [n for n in page["children"]
                        if fs._child_path(n) in fs.pages]
                if not dirs:
                    return None
                cmd_str = f"cd {rng.choice(dirs)}"
            else:
                if fs.cwd == "/":
                    return None
                cmd_str = "cd .."

        elif op == "rm":
            files = list(page["files"].keys())
            if not files:
                return None
            cmd_str = f"rm {rng.choice(files)}"

        elif op == "cp":
            files = list(page["files"].keys())
            if not files:
                return None
            src = rng.choice(files)
            dst = _random_filename_rng(rng)
            cmd_str = f"cp {src} {dst}"

        elif op == "mv":
            files = list(page["files"].keys())
            if not files:
                return None
            src = rng.choice(files)
            dst = _random_filename_rng(rng)
            cmd_str = f"mv {src} {dst}"

        elif op == "head":
            files = [f for f in page["files"] if page["files"][f]]
            if not files:
                return None
            cmd_str = f"head {rng.choice(files)}"

        elif op == "wc":
            files = [f for f in page["files"] if page["files"][f]]
            if not files:
                return None
            cmd_str = f"wc {rng.choice(files)}"

        elif op == "grep":
            files = [f for f in page["files"] if page["files"][f]]
            if not files:
                return None
            name = rng.choice(files)
            words = page["files"][name].split()
            if not words:
                return None
            cmd_str = f"grep {rng.choice(words)} {name}"

        elif op == "var_set":
            name = _gen_syllable_name_rng(rng, 1)
            value = str(rng.randint(0, 99)) if rng.random() < 0.5 else _gen_syllable_name_rng(rng, 1)
            cmd_str = f"{name}={value}"

        elif op == "var_echo":
            if not fs.vars:
                return None
            var_name = rng.choice(list(fs.vars.keys()))
            cmd_str = f"echo ${var_name}"

        elif op == "expr_math":
            a = rng.randint(1, 50)
            b = rng.randint(1, 50)
            sym = rng.choice(["+", "-", "*"])
            cmd_str = f"expr {a} {sym} {b}"

        elif op == "export":
            name = _gen_syllable_name_rng(rng, 1)
            value = _gen_syllable_name_rng(rng, 1)
            cmd_str = f"export {name}={value}"

        if not cmd_str:
            return None

        # Build input BEFORE executing (model sees pre-command state)
        full_input = fs.serialize_full_input(cmd_str)

        # Execute and get response + patch
        response, patch = fs.execute(cmd_str)
        if patch:
            fs.apply_patch(patch)

        return {
            "input": full_input,
            "response": response,
            "patch": patch,
        }


class BashSessionDataset(IterableDataset):
    def __init__(self, buffer_size=32, workers=4, min_ops=20, target_ops=50,
                 base_seed=42):
        super().__init__()
        self.buffer_size = buffer_size
        self.workers = workers
        self.min_ops = min_ops
        self.target_ops = target_ops
        self.base_seed = base_seed
        self._queue = None
        self._threads = []
        self._stop_event = threading.Event()

    def _worker_loop(self, worker_id):
        seed_counter = self.base_seed + worker_id * 10_000_000
        tok = BashTokenizer()
        while not self._stop_event.is_set():
            seed_counter += 1
            gen = PagedSessionGenerator(
                min_ops=self.min_ops, target_ops=self.target_ops,
                seed=seed_counter)
            try:
                cmds = gen.generate()
                samples = build_session_samples(cmds, tok)
            except (ValueError, KeyError):
                continue
            if not samples:
                continue
            try:
                self._queue.put(samples, timeout=1.0)
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
        while self._queue is not None:
            try:
                item = self._queue.get(timeout=10.0)
            except queue.Empty:
                continue
            yield item

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    import time
    tok = BashTokenizer()

    print("Testing paged dataset...")
    gen = PagedSessionGenerator(min_ops=20, target_ops=40, seed=42)
    cmds = gen.generate()
    samples = build_session_samples(cmds, tok)

    print(f"\nSession: {len(samples)} commands")
    max_tokens = 0
    for i, s in enumerate(samples):
        text = tok.decode(s["ids"])
        max_tokens = max(max_tokens, len(s["ids"]))
        if i < 10:
            trained = sum(1 for l in s["labels"] if l != -100)
            print(f"  [{i:>2d}] ({len(s['ids']):>3d} tok, {trained:>3d} trained) {text[:100]}")

    print(f"\n  Max tokens per sample: {max_tokens}")
    print(f"  Pages created: {list(gen.fs.pages.keys())}")

    print(f"\n  Testing BashSessionDataset...")
    ds = BashSessionDataset(buffer_size=4, workers=2, min_ops=20, target_ops=40)
    t0 = time.time()
    for i, session in enumerate(ds):
        if i < 2:
            max_t = max(len(s["ids"]) for s in session)
            print(f"  Session {i}: {len(session)} cmds, max {max_t} tokens")
        if i >= 4:
            break
    ds.stop()
    print(f"  5 sessions in {time.time()-t0:.2f}s")
