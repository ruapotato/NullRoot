"""
Dataset for bash terminal with state patches.

Each training sample is one command exchange with state:
  [state_string] <prompt>cmd<eoi> <output>response<eor> <state>patch<eor>

The model sees the current state + command, produces the response + state patch.
The patch is applied to produce the state for the next command.

Labels mask everything before <eoi> (state + command are input context).
The model trains on predicting the response AND the state patch.
"""

import threading
import queue
import json
import torch
from torch.utils.data import IterableDataset

from tokenizer import BashTokenizer
from generator import SessionGenerator, FileSystem


def build_session_samples(transcript_commands: list[dict],
                          tok: BashTokenizer) -> list[dict]:
    """Build training samples from a session with state tracking.

    Each command gets: state input + command + response + state patch.

    Args:
        transcript_commands: list of dicts with 'cmd_str', 'response_str',
                            'is_error', and 'fs' (FileSystem snapshot after cmd)
        tok: tokenizer

    Returns:
        list of dicts with 'ids', 'labels', 'weights', 'state_after'
    """
    state_str = ""  # empty state at session start
    samples = []

    for cmd_info in transcript_commands:
        cmd_str = cmd_info["cmd_str"]
        response_str = cmd_info["response_str"]
        is_error = cmd_info["is_error"]
        fs = cmd_info["fs"]

        # Build the full sequence:
        # [state_string] <prompt>cmd<eoi> <output>response<eor> <state>patch<eor>
        # or for errors:
        # [state_string] <prompt>cmd<eoi> <err><eor> <state>patch<eor>

        new_state_str = fs.serialize_state()
        patch_str = fs.compute_patch(state_str) if state_str else new_state_str

        # Build input portion (masked)
        input_part = ""
        if state_str:
            input_part += f"<state>{state_str}<eor>"
        input_part += f"<prompt>{cmd_str}<eoi>"

        # Build output portion (trained)
        if is_error:
            output_part = "<err><eor>"
        elif response_str:
            output_part = f"<output>{response_str}<eor>"
        else:
            output_part = "<output><eor>"

        # Build state patch portion (trained)
        if patch_str:
            state_part = f"<state>{patch_str}<eor>"
        else:
            state_part = "<nop>"

        full_text = input_part + output_part + state_part

        try:
            ids = tok.encode(full_text)
        except ValueError:
            state_str = new_state_str
            continue

        # Labels: mask input portion, train on output + state patch
        labels = list(ids)
        weights = [1.0] * len(ids)

        # Find where the output starts (after <eoi>)
        in_input = True
        for j, tid in enumerate(ids):
            if in_input:
                labels[j] = -100
                weights[j] = 0.0
                if tid == tok.eoi_id:
                    in_input = False

        samples.append({
            "ids": ids,
            "labels": labels,
            "weights": weights,
        })

        state_str = new_state_str

    return samples


class BashSessionDataset(IterableDataset):
    """Infinite dataset yielding sessions with state patches.

    Each session is a list of command samples, each containing:
    - ids: [state + prompt + response + patch] token ids
    - labels: masked on input, trained on output + patch
    - weights: per-token loss weights
    """

    def __init__(
        self,
        buffer_size: int = 32,
        workers: int = 4,
        min_ops: int = 10,
        target_ops: int = 30,
        error_rate: float = 0.0,
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
        seed_counter = self.base_seed + worker_id * 10_000_000
        tok = BashTokenizer()

        while not self._stop_event.is_set():
            seed_counter += 1

            gen = StateTrackingGenerator(
                min_ops=self.min_ops,
                target_ops=self.target_ops,
                error_rate=self.error_rate,
                seed=seed_counter,
                commands=self.commands,
            )
            cmd_list = gen.generate_with_state()

            try:
                samples = build_session_samples(cmd_list, tok)
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


class StateTrackingGenerator:
    """Like SessionGenerator but captures filesystem state after each command."""

    ALL_COMMANDS = {"mkdir", "cd_child", "cd_up", "cd_abs", "ls", "pwd",
                    "touch", "echo_write", "cat", "echo_append", "rm"}

    def __init__(self, min_ops=10, target_ops=30, error_rate=0.0,
                 seed=None, commands=None):
        import random
        self.min_ops = min_ops
        self.target_ops = target_ops
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self.fs = FileSystem()
        self.commands = commands if commands is not None else self.ALL_COMMANDS

    def generate_with_state(self) -> list[dict]:
        """Generate session, returning command info with filesystem snapshots."""
        from generator import (_random_dirname_rng, _random_filename_rng,
                               _random_content)
        import copy

        num_ops = max(self.min_ops, min(
            self.target_ops * 2,
            int(self.rng.gauss(self.target_ops, self.target_ops * 0.3))))

        results = []
        rng = self.rng
        enabled = self.commands

        for _ in range(num_ops):
            # Pick operation (simplified from SessionGenerator)
            op = rng.choice([c for c in enabled if c != "errors"])

            cmd_str = ""
            response_str = ""
            is_error = False

            if op == "ls":
                entries = self.fs.list_dir(self.fs.cwd)
                cmd_str = "ls"
                response_str = "  ".join(entries) if entries else ""

            elif op == "pwd":
                cmd_str = "pwd"
                response_str = self.fs.cwd

            elif op == "mkdir":
                name = _random_dirname_rng(rng)
                for _ in range(10):
                    if not self.fs.exists(self.fs.resolve(name)):
                        break
                    name = _random_dirname_rng(rng)
                cmd_str = f"mkdir {name}"
                err = self.fs.mkdir(name)
                if err:
                    is_error = True

            elif op == "touch":
                name = _random_filename_rng(rng)
                cmd_str = f"touch {name}"
                err = self.fs.touch(name)
                if err:
                    is_error = True

            elif op == "echo_write":
                name = _random_filename_rng(rng)
                content = _random_content(rng)
                cmd_str = f"echo {content} > {name}"
                err = self.fs.write_file(name, content)
                if err:
                    is_error = True

            elif op == "echo_append":
                files = self.fs.get_child_files()
                if files and rng.random() < 0.7:
                    name = rng.choice(files)
                else:
                    name = _random_filename_rng(rng)
                content = _random_content(rng)
                cmd_str = f"echo {content} >> {name}"
                err = self.fs.append_file(name, content)
                if err:
                    is_error = True

            elif op == "cat":
                files = self.fs.get_child_files()
                if not files:
                    continue
                name = rng.choice(files)
                content, err = self.fs.cat(name)
                cmd_str = f"cat {name}"
                if err:
                    is_error = True
                else:
                    response_str = content if content else ""

            elif op == "cd_child":
                dirs = self.fs.get_child_dirs()
                if not dirs:
                    continue
                name = rng.choice(dirs)
                cmd_str = f"cd {name}"
                self.fs.cd(name)

            elif op == "cd_up":
                if self.fs.cwd == "/":
                    continue
                cmd_str = "cd .."
                self.fs.cd("..")

            elif op == "cd_abs":
                dirs = self.fs.get_all_dirs()
                if not dirs:
                    continue
                target = rng.choice(dirs)
                cmd_str = f"cd {target}"
                self.fs.cd(target)

            elif op == "rm":
                files = self.fs.get_child_files()
                if not files:
                    continue
                name = rng.choice(files)
                cmd_str = f"rm {name}"
                self.fs.rm(name)

            else:
                continue

            if not cmd_str:
                continue

            # Snapshot filesystem state AFTER this command
            results.append({
                "cmd_str": cmd_str,
                "response_str": response_str,
                "is_error": is_error,
                "fs": copy.deepcopy(self.fs),
            })

        return results


if __name__ == "__main__":
    import time

    tok = BashTokenizer()

    print("Testing state-patch dataset...")
    gen = StateTrackingGenerator(min_ops=5, target_ops=10, seed=42,
                                 commands={"mkdir", "cd_child", "cd_up", "cd_abs", "ls"})
    cmds = gen.generate_with_state()
    samples = build_session_samples(cmds, tok)

    print(f"\nSession: {len(samples)} commands")
    for i, s in enumerate(samples[:8]):
        text = tok.decode(s["ids"])
        trained = sum(1 for l in s["labels"] if l != -100)
        print(f"\n  Cmd {i}: ({len(s['ids'])} tokens, {trained} trained)")
        # Show truncated
        if len(text) > 120:
            print(f"    {text[:120]}...")
        else:
            print(f"    {text}")

    print(f"\n\nTesting BashSessionDataset...")
    ds = BashSessionDataset(buffer_size=4, workers=2, min_ops=5, target_ops=10,
                            commands={"mkdir", "cd_child", "cd_up", "cd_abs", "ls"})
    t0 = time.time()
    for i, session in enumerate(ds):
        if i < 2:
            print(f"\n  Session {i}: {len(session)} commands")
            for j, s in enumerate(session[:3]):
                text = tok.decode(s["ids"])
                print(f"    Cmd {j}: {len(s['ids'])} tok | {text[:100]}")
        if i >= 4:
            break
    ds.stop()
    print(f"\n  5 sessions in {time.time()-t0:.2f}s")
