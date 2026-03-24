"""
Dataset for NullRoot with state patches.

Each training sample: [state] <prompt>cmd<eoi> → <output>response<eor> <state>patch<eor>
Generates deep filesystems with all commands: filesystem ops, variables, math, scripts.
"""

import threading
import queue
import json
import copy
import random
import torch
from torch.utils.data import IterableDataset

from tokenizer import BashTokenizer
from generator import FileSystem, _gen_syllable_name_rng, _random_filename_rng, _random_content


def build_session_samples(transcript_commands: list[dict],
                          tok: BashTokenizer) -> list[dict]:
    """Build training samples from a session with state tracking."""
    state_str = ""
    samples = []

    for cmd_info in transcript_commands:
        cmd_str = cmd_info["cmd_str"]
        response_str = cmd_info["response_str"]
        is_error = cmd_info["is_error"]
        fs = cmd_info["fs"]

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

        # Build state patch (trained)
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

        samples.append({
            "ids": ids,
            "labels": labels,
            "weights": weights,
        })

        state_str = new_state_str

    return samples


# ---------------------------------------------------------------------------
# Session generator with full command coverage
# ---------------------------------------------------------------------------

class StateTrackingGenerator:
    """Generates training sessions with deep filesystems, variables, math, scripts."""

    # Weighted command pool — controls distribution
    COMMAND_WEIGHTS = {
        # Filesystem (heavily weighted — core skill)
        "mkdir": 12, "cd_child": 10, "cd_up": 6, "cd_abs": 5,
        "ls": 15, "pwd": 5,
        "touch": 8, "echo_write": 12, "echo_append": 6,
        "cat": 12, "rm": 4, "cp": 4, "mv": 4,
        "head": 3, "wc": 3, "find": 3, "grep": 3,
        # Variables + math
        "var_set": 8, "var_echo": 6, "expr_math": 6,
        "export": 3,
        # Scripts
        "write_script": 5, "run_script": 4,
    }

    def __init__(self, min_ops=50, target_ops=150, seed=None, commands=None):
        self.min_ops = min_ops
        self.target_ops = target_ops
        self.rng = random.Random(seed)
        self.fs = FileSystem()
        if commands:
            self.weights = {k: v for k, v in self.COMMAND_WEIGHTS.items()
                           if k in commands}
        else:
            self.weights = dict(self.COMMAND_WEIGHTS)

    def generate_with_state(self) -> list[dict]:
        num_ops = max(self.min_ops, min(
            self.target_ops * 2,
            int(self.rng.gauss(self.target_ops, self.target_ops * 0.3))))

        results = []
        rng = self.rng
        ops = list(self.weights.keys())
        wts = list(self.weights.values())

        for _ in range(num_ops):
            op = rng.choices(ops, weights=wts, k=1)[0]
            cmd_info = self._exec_op(op, rng)
            if cmd_info:
                results.append(cmd_info)

        return results

    def _exec_op(self, op, rng) -> dict | None:
        """Execute one operation, return command info or None if skipped."""
        fs = self.fs
        cmd_str = ""
        response_str = ""
        is_error = False

        # --- Filesystem ops ---
        if op == "ls":
            entries = fs.list_dir(fs.cwd)
            cmd_str = "ls"
            response_str = "  ".join(entries) if entries else ""

        elif op == "pwd":
            cmd_str = "pwd"
            response_str = fs.cwd

        elif op == "mkdir":
            name = _gen_syllable_name_rng(rng)
            for _ in range(10):
                if not fs.exists(fs.resolve(name)):
                    break
                name = _gen_syllable_name_rng(rng)
            cmd_str = f"mkdir {name}"
            err = fs.mkdir(name)
            if err:
                return None

        elif op == "touch":
            name = _random_filename_rng(rng)
            cmd_str = f"touch {name}"
            err = fs.touch(name)
            if err:
                return None

        elif op == "echo_write":
            name = _random_filename_rng(rng)
            content = _random_content(rng)
            cmd_str = f"echo {content} > {name}"
            err = fs.write_file(name, content)
            if err:
                return None

        elif op == "echo_append":
            files = fs.get_child_files()
            if files and rng.random() < 0.7:
                name = rng.choice(files)
            else:
                name = _random_filename_rng(rng)
            content = _random_content(rng)
            cmd_str = f"echo {content} >> {name}"
            err = fs.append_file(name, content)
            if err:
                return None

        elif op == "cat":
            files = fs.get_child_files()
            if not files:
                return None
            name = rng.choice(files)
            content, err = fs.cat(name)
            cmd_str = f"cat {name}"
            if err:
                return None
            response_str = content if content else ""

        elif op == "cd_child":
            dirs = fs.get_child_dirs()
            if not dirs:
                return None
            name = rng.choice(dirs)
            cmd_str = f"cd {name}"
            fs.cd(name)

        elif op == "cd_up":
            if fs.cwd == "/":
                return None
            cmd_str = "cd .."
            fs.cd("..")

        elif op == "cd_abs":
            dirs = fs.get_all_dirs()
            if not dirs:
                return None
            target = rng.choice(dirs)
            cmd_str = f"cd {target}"
            fs.cd(target)

        elif op == "rm":
            files = fs.get_child_files()
            if not files:
                return None
            name = rng.choice(files)
            cmd_str = f"rm {name}"
            fs.rm(name)

        elif op == "cp":
            files = fs.get_child_files()
            if not files:
                return None
            src = rng.choice(files)
            dst = _random_filename_rng(rng)
            cmd_str = f"cp {src} {dst}"
            err = fs.cp(src, dst)
            if err:
                return None

        elif op == "mv":
            files = fs.get_child_files()
            if not files:
                return None
            src = rng.choice(files)
            dst = _random_filename_rng(rng)
            cmd_str = f"mv {src} {dst}"
            err = fs.mv(src, dst)
            if err:
                return None

        elif op == "head":
            files = fs.get_child_files()
            if not files:
                return None
            name = rng.choice(files)
            content, err = fs.head(name, n=1)
            cmd_str = f"head {name}"
            if err:
                return None
            response_str = content if content else ""

        elif op == "wc":
            files = fs.get_child_files()
            if not files:
                return None
            name = rng.choice(files)
            result, err = fs.wc(name)
            cmd_str = f"wc {name}"
            if err:
                return None
            response_str = result

        elif op == "find":
            cmd_str = "find ."
            response_str = fs.find(".")

        elif op == "grep":
            files = fs.get_child_files()
            if not files:
                return None
            name = rng.choice(files)
            content = fs.files.get(fs.resolve(name), "")
            if not content:
                return None
            words = content.split()
            if not words:
                return None
            pattern = rng.choice(words)
            result, err = fs.grep(pattern, name)
            cmd_str = f"grep {pattern} {name}"
            if err:
                return None
            response_str = result if result else ""

        # --- Variable ops ---
        elif op == "var_set":
            name = _gen_syllable_name_rng(rng, 1)
            # Mix of string and numeric values
            if rng.random() < 0.5:
                value = str(rng.randint(0, 99))
            else:
                value = _gen_syllable_name_rng(rng, 1)
            cmd_str = f"{name}={value}"
            fs.set_var(name, value)

        elif op == "var_echo":
            if not fs.vars:
                return None
            var_name = rng.choice(list(fs.vars.keys()))
            cmd_str = f"echo ${var_name}"
            response_str = fs.get_var(var_name)

        elif op == "expr_math":
            a = rng.randint(1, 50)
            b = rng.randint(1, 50)
            op_sym = rng.choice(["+", "-", "*"])
            cmd_str = f"expr {a} {op_sym} {b}"
            result = fs.eval_math(f"{a} {op_sym} {b}")
            response_str = result if result else "0"

        elif op == "export":
            name = _gen_syllable_name_rng(rng, 1)
            value = _gen_syllable_name_rng(rng, 1)
            cmd_str = f"export {name}={value}"
            fs.set_var(name, value)

        # --- Script ops ---
        elif op == "write_script":
            # Write a small script: 2-4 commands
            name = _gen_syllable_name_rng(rng, 1) + ".sh"
            script_lines = []
            for _ in range(rng.randint(2, 4)):
                line = self._random_script_line(rng)
                if line:
                    script_lines.append(line)
            if not script_lines:
                return None
            content = "\\n".join(script_lines)
            cmd_str = f'echo "{content}" > {name}'
            # Store with actual newlines
            fs.write_file(name, "\n".join(script_lines))

        elif op == "run_script":
            files = fs.get_child_files()
            scripts = [f for f in files if f.endswith(".sh")]
            if not scripts:
                return None
            name = rng.choice(scripts)
            content, err = fs.cat(name)
            if err or not content:
                return None
            cmd_str = f"sh {name}"
            output, _ = fs.execute_script(content)
            response_str = output

        else:
            return None

        if not cmd_str:
            return None

        return {
            "cmd_str": cmd_str,
            "response_str": response_str,
            "is_error": is_error,
            "fs": copy.deepcopy(fs),
        }

    def _random_script_line(self, rng) -> str | None:
        """Generate a random script line."""
        kind = rng.choice(["echo", "mkdir", "touch", "var", "cd", "ls"])
        if kind == "echo":
            word = _gen_syllable_name_rng(rng, 1)
            return f"echo {word}"
        elif kind == "mkdir":
            name = _gen_syllable_name_rng(rng, 1)
            return f"mkdir {name}"
        elif kind == "touch":
            name = _gen_syllable_name_rng(rng, 1)
            return f"touch {name}"
        elif kind == "var":
            name = _gen_syllable_name_rng(rng, 1)
            val = str(rng.randint(0, 99))
            return f"{name}={val}"
        elif kind == "cd":
            dirs = self.fs.get_child_dirs()
            if dirs:
                return f"cd {rng.choice(dirs)}"
            return "pwd"
        elif kind == "ls":
            return "ls"
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BashSessionDataset(IterableDataset):
    def __init__(self, buffer_size=32, workers=4, min_ops=50, target_ops=150,
                 base_seed=42, commands=None):
        super().__init__()
        self.buffer_size = buffer_size
        self.workers = workers
        self.min_ops = min_ops
        self.target_ops = target_ops
        self.base_seed = base_seed
        self.commands = commands
        self.tokenizer = BashTokenizer()
        self._queue = None
        self._threads = []
        self._stop_event = threading.Event()

    def _worker_loop(self, worker_id):
        seed_counter = self.base_seed + worker_id * 10_000_000
        tok = BashTokenizer()
        while not self._stop_event.is_set():
            seed_counter += 1
            gen = StateTrackingGenerator(
                min_ops=self.min_ops, target_ops=self.target_ops,
                seed=seed_counter, commands=self.commands)
            try:
                cmds = gen.generate_with_state()
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
    print("Testing expanded dataset...")

    gen = StateTrackingGenerator(min_ops=50, target_ops=100, seed=42)
    cmds = gen.generate_with_state()
    samples = build_session_samples(cmds, tok)

    print(f"\nSession: {len(samples)} commands")

    # Count command types
    cmd_types = {}
    for c in cmds:
        cmd = c["cmd_str"].split()[0]
        if "=" in cmd:
            cmd = "var_set"
        cmd_types[cmd] = cmd_types.get(cmd, 0) + 1

    print(f"\nCommand distribution:")
    for k, v in sorted(cmd_types.items(), key=lambda x: -x[1]):
        print(f"  {k:>12s}: {v}")

    # Show some samples
    print(f"\nSample commands:")
    for i, s in enumerate(samples):
        text = tok.decode(s["ids"])
        if i < 5 or "expr" in text or "sh " in text or "$" in text:
            trained = sum(1 for l in s["labels"] if l != -100)
            print(f"  [{i:>3d}] ({len(s['ids']):>3d} tok, {trained:>3d} trained) {text[:120]}")

    # State size at end
    last_fs = cmds[-1]["fs"]
    state = last_fs.serialize_state()
    state_tokens = len(tok.encode(state))
    print(f"\nFinal state: {state_tokens} tokens, {len(last_fs.dirs)} dirs, "
          f"{len(last_fs.files)} files, {len(last_fs.vars)} vars")

    print(f"\nTesting BashSessionDataset...")
    ds = BashSessionDataset(buffer_size=4, workers=2, min_ops=50, target_ops=100)
    t0 = time.time()
    for i, session in enumerate(ds):
        if i < 2:
            print(f"  Session {i}: {len(session)} commands")
        if i >= 4:
            break
    ds.stop()
    print(f"  5 sessions in {time.time()-t0:.2f}s")
