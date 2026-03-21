"""
Synthetic bash session generator.
Generates TTY transcripts with full filesystem state tracking.
"""

import random
import string
import argparse
import json
from pathlib import PurePosixPath
from tokenizer import BashTokenizer


# ---------------------------------------------------------------------------
# Generative name system — combinatorial, not list-based.
#
# Names are assembled from syllable fragments so that every session can use
# entirely unique names. With ~400 syllables and 2-3 per name we get
# 400^2 + 400^3 = ~64 million distinct names, all short and pronounceable.
# A small fraction of names come from fixed pools (unix dirs, dotfiles,
# extensions) so the model also sees realistic structure.
# ---------------------------------------------------------------------------

# Syllable fragments for combinatorial name generation (CV, CVC, VC patterns)
_ONSETS = [
    "b", "bl", "br", "c", "ch", "cl", "cr", "d", "dr", "f", "fl", "fr",
    "g", "gl", "gr", "h", "j", "k", "kl", "kr", "l", "m", "n", "p", "pl",
    "pr", "qu", "r", "s", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st",
    "str", "sw", "t", "th", "tr", "tw", "v", "w", "wr", "x", "z",
]
_VOWELS = ["a", "e", "i", "o", "u", "ai", "au", "ei", "ou", "ea", "io"]
_CODAS = ["", "b", "ck", "d", "f", "g", "k", "l", "m", "n", "ng", "nk",
          "p", "r", "s", "sh", "sk", "st", "t", "th", "x", "z"]

# Pre-build a large syllable table once at import time
_SYLLABLES: list[str] = []
for _o in _ONSETS:
    for _v in _VOWELS:
        for _c in _CODAS[:8]:  # limit codas to keep names short
            _SYLLABLES.append(_o + _v + _c)
# Deduplicate and filter to reasonable length
_SYLLABLES = sorted(set(s for s in _SYLLABLES if 2 <= len(s) <= 5))

# Digit fragments for mixing in
_DIGIT_FRAGS = [str(i) for i in range(100)] + [
    "0x" + hex(i)[2:] for i in range(16)
]

# Small fixed pools for realistic structure (used sparingly)
_UNIX_DIRS = [
    "bin", "boot", "dev", "etc", "home", "lib", "mnt", "opt", "proc",
    "root", "run", "sbin", "srv", "sys", "tmp", "usr", "var",
]
_PROJECT_DIRS = [
    "src", "lib", "bin", "dist", "build", "out", "test", "tests", "docs",
    "scripts", "config", "data", "db", "api", "web", "pkg", "vendor",
    "deploy", "logs", "assets", "static",
]
_DOTNAMES = [
    ".env", ".config", ".cache", ".local", ".ssh", ".log", ".pid",
    ".lock", ".tmp", ".old", ".bak", ".git", ".npm", ".cargo",
]

# Extensions
_EXTENSIONS = [
    ".py", ".sh", ".c", ".go", ".rs", ".js", ".ts", ".rb", ".pl",
    ".cfg", ".conf", ".ini", ".env", ".toml", ".yml",
    ".csv", ".tsv", ".json", ".dat", ".bin", ".db",
    ".txt", ".md", ".rst", ".log", ".out",
    ".html", ".css", ".xml",
    ".o", ".so", ".a", ".lock",
    ".tar", ".gz", ".zip", ".bak",
]


_MAX_NAME_LEN = 15
_BAD_SUBSTRINGS = {"fuck", "shit", "cunt", "dick", "cock", "slut", "nigg", "fag"}


def _name_ok(name: str) -> bool:
    low = name.lower()
    return not any(b in low for b in _BAD_SUBSTRINGS)


def _gen_syllable_name(n_syllables: int = 0) -> str:
    """Generate a pronounceable name from random syllables."""
    if n_syllables == 0:
        n_syllables = random.choices([1, 2, 3], weights=[30, 45, 25], k=1)[0]
    parts = [random.choice(_SYLLABLES) for _ in range(n_syllables)]
    # Occasionally insert a separator
    if len(parts) > 1 and random.random() < 0.25:
        sep = random.choice(["_", "-", ""])
        name = sep.join(parts)
    else:
        name = "".join(parts)
    return name[:_MAX_NAME_LEN]


def _gen_mixed_name() -> str:
    """Generate a name mixing syllables with digits."""
    base = _gen_syllable_name(random.randint(1, 2))
    r = random.random()
    if r < 0.3:
        return base + str(random.randint(0, 99))
    if r < 0.5:
        return base + "_" + str(random.randint(0, 999))
    if r < 0.65:
        return str(random.randint(0, 9)) + base
    return base


def random_name() -> str:
    """Module-level convenience wrapper using global random state."""
    return _random_name_with_rng(random.Random(random.randint(0, 2**32)))


def random_extension() -> str:
    r = random.random()
    if r < 0.20:
        return ""
    return random.choice(_EXTENSIONS)


def random_filename() -> str:
    return _random_filename_rng(random.Random(random.randint(0, 2**32)))


def random_dirname() -> str:
    return _random_dirname_rng(random.Random(random.randint(0, 2**32)))


class FileSystem:
    """In-memory filesystem tracking dirs and files with content."""

    def __init__(self):
        # dirs: set of absolute paths (strings), always starts with "/"
        self.dirs: set[str] = {"/"}
        # files: dict of absolute path -> content string
        self.files: dict[str, str] = {}
        self.cwd = "/"

    def resolve(self, path: str) -> str:
        """Resolve a path relative to cwd into an absolute path."""
        if path.startswith("/"):
            p = PurePosixPath(path)
        else:
            p = PurePosixPath(self.cwd) / path
        # Normalize (resolve ..)
        parts = []
        for part in p.parts:
            if part == "/":
                continue
            elif part == "..":
                if parts:
                    parts.pop()
            else:
                parts.append(part)
        return "/" + "/".join(parts) if parts else "/"

    def parent(self, path: str) -> str:
        if path == "/":
            return "/"
        return str(PurePosixPath(path).parent)

    def exists_dir(self, path: str) -> bool:
        return path in self.dirs

    def exists_file(self, path: str) -> bool:
        return path in self.files

    def exists(self, path: str) -> bool:
        return self.exists_dir(path) or self.exists_file(path)

    def list_dir(self, path: str) -> list[str]:
        """List immediate children of a directory."""
        entries = []
        prefix = path.rstrip("/") + "/"
        for d in sorted(self.dirs):
            if d == path:
                continue
            if d.startswith(prefix):
                rest = d[len(prefix):]
                if "/" not in rest:
                    entries.append(rest)
        for f in sorted(self.files):
            if f.startswith(prefix):
                rest = f[len(prefix):]
                if "/" not in rest:
                    entries.append(rest)
        return sorted(set(entries))

    def mkdir(self, name: str) -> str | None:
        """Create directory under cwd. Returns None on success, error string on failure."""
        abspath = self.resolve(name)
        if self.exists(abspath):
            return f"mkdir: cannot create directory {name}: file exists"
        # Ensure parent exists
        parent = self.parent(abspath)
        if not self.exists_dir(parent):
            return f"mkdir: cannot create directory {name}: no such file or directory"
        self.dirs.add(abspath)
        return None

    def touch(self, name: str) -> str | None:
        abspath = self.resolve(name)
        parent = self.parent(abspath)
        if not self.exists_dir(parent):
            return f"touch: cannot touch {name}: no such file or directory"
        if abspath not in self.files:
            self.files[abspath] = ""
        return None

    def write_file(self, name: str, content: str) -> str | None:
        """echo content > file"""
        abspath = self.resolve(name)
        parent = self.parent(abspath)
        if not self.exists_dir(parent):
            return f"bash: {name}: no such file or directory"
        self.files[abspath] = content
        return None

    def append_file(self, name: str, content: str) -> str | None:
        """echo content >> file"""
        abspath = self.resolve(name)
        parent = self.parent(abspath)
        if not self.exists_dir(parent):
            return f"bash: {name}: no such file or directory"
        if abspath in self.files:
            self.files[abspath] += "\n" + content
        else:
            self.files[abspath] = content
        return None

    def cat(self, name: str) -> tuple[str | None, str | None]:
        """Returns (content, error). One will be None."""
        abspath = self.resolve(name)
        if abspath in self.files:
            return self.files[abspath], None
        return None, f"cat: {name}: no such file or directory"

    def rm(self, name: str) -> str | None:
        abspath = self.resolve(name)
        if abspath in self.files:
            del self.files[abspath]
            return None
        return f"rm: cannot remove '{name}': no such file or directory"

    def cd(self, path: str) -> str | None:
        if path == "..":
            if self.cwd == "/":
                # cd .. from root stays at root (not an error in real bash,
                # but we treat it as an error per spec for invalid ops)
                return None  # handled specially by generator for invalid ops
            new = self.parent(self.cwd)
            self.cwd = new
            return None
        abspath = self.resolve(path)
        if self.exists_dir(abspath):
            self.cwd = abspath
            return None
        return f"bash: cd: {path}: no such file or directory"

    def get_child_dirs(self) -> list[str]:
        """Get names of immediate child directories of cwd."""
        children = []
        prefix = self.cwd.rstrip("/") + "/"
        for d in self.dirs:
            if d == self.cwd:
                continue
            if d.startswith(prefix):
                rest = d[len(prefix):]
                if "/" not in rest:
                    children.append(rest)
        return children

    def get_child_files(self) -> list[str]:
        """Get names of immediate child files of cwd."""
        children = []
        prefix = self.cwd.rstrip("/") + "/"
        for f in self.files:
            if f.startswith(prefix):
                rest = f[len(prefix):]
                if "/" not in rest:
                    children.append(rest)
        return children

    def get_all_dirs(self) -> list[str]:
        """Get all directories except root."""
        return [d for d in sorted(self.dirs) if d != "/"]


def _random_content(rng: random.Random) -> str:
    """Generate random short file content using only valid token characters."""
    words = []
    for _ in range(rng.randint(1, 8)):
        wlen = rng.randint(1, 8)
        words.append("".join(rng.choice(string.ascii_lowercase) for _ in range(wlen)))
    return " ".join(words)


def _random_name_with_rng(rng: random.Random) -> str:
    """Thread-safe random_name using an explicit RNG.

    All names are generated combinatorially — no fixed pools, so the model
    learns "name" as a structural slot rather than memorizing specific words.
    """
    for _ in range(20):
        r = rng.random()
        if r < 0.05:
            # Dotname: generated, not from a list
            base = _gen_syllable_name_rng(rng, 1)
            name = "." + base
        elif r < 0.30:
            # Mixed: syllable + digits
            base = _gen_syllable_name_rng(rng, rng.randint(1, 2))
            r2 = rng.random()
            if r2 < 0.3:
                name = base + str(rng.randint(0, 99))
            elif r2 < 0.5:
                name = base + "_" + str(rng.randint(0, 999))
            elif r2 < 0.65:
                name = str(rng.randint(0, 9)) + base
            else:
                name = base
        else:
            # Pure syllable name
            name = _gen_syllable_name_rng(rng)
        if _name_ok(name):
            return name
    return _gen_syllable_name_rng(rng, 1)


def _gen_syllable_name_rng(rng: random.Random, n_syllables: int = 0) -> str:
    """Thread-safe syllable name generator."""
    if n_syllables == 0:
        n_syllables = rng.choices([1, 2, 3], weights=[30, 45, 25], k=1)[0]
    parts = [rng.choice(_SYLLABLES) for _ in range(n_syllables)]
    if len(parts) > 1 and rng.random() < 0.25:
        sep = rng.choice(["_", "-", ""])
        name = sep.join(parts)
    else:
        name = "".join(parts)
    return name[:_MAX_NAME_LEN]


def _random_filename_rng(rng: random.Random) -> str:
    name = _random_name_with_rng(rng)
    if name.startswith("."):
        return name[:_MAX_NAME_LEN]
    if "." in name:
        return name[:_MAX_NAME_LEN]
    ext = rng.choice(_EXTENSIONS) if rng.random() > 0.20 else ""
    max_stem = _MAX_NAME_LEN - len(ext)
    return name[:max_stem] + ext


def _random_dirname_rng(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.08:
        # Dot-directory
        return "." + _gen_syllable_name_rng(rng, 1)
    return _gen_syllable_name_rng(rng)


class SessionGenerator:
    """Generates a single bash session transcript.

    Each instance uses its own Random for thread safety.
    Builds complex directory hierarchies and populates them with files,
    randomly choosing among all available commands weighted by current
    filesystem state.
    """

    def __init__(self, min_ops: int = 300, target_ops: int = 800,
                 error_rate: float = 0.05, seed: int | None = None):
        self.min_ops = min_ops
        self.target_ops = target_ops
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self.fs = FileSystem()
        self.transcript_parts: list[str] = []

    def _emit(self, prompt_cmd: str, output: str | None = None, is_error: bool = False):
        part = f"<prompt>{prompt_cmd}\n"
        if is_error:
            part += "<err>\n"
        elif output:
            part += f"<output>{output}\n"
        else:
            part += "<output>\n"
        self.transcript_parts.append(part)

    def _choose_num_ops(self) -> int:
        # Log-normal: bulk at 500-800, right tail to 5000, floor at min_ops
        raw = self.rng.lognormvariate(mu=7.6, sigma=0.55)
        # mu=7.6 -> median ~2000, sigma=0.55 -> p95 ~4900, tail to 5000
        # ~44% of sessions fill a 65K context window
        clamped = max(self.min_ops, min(5000, int(raw)))
        return clamped

    def _depth(self) -> int:
        """Current directory depth (root=0)."""
        if self.fs.cwd == "/":
            return 0
        return self.fs.cwd.count("/")

    def _pick_op(self) -> tuple[str, bool]:
        """Pick next operation. Returns (op_name, is_error).

        Adapts weights to filesystem state to ensure natural behavior:
        - Early on: bias toward mkdir/touch/echo to build structure
        - Deep in tree: more cat/ls/cd to explore
        - With many files: more append/cat/rm
        - 5% chance of intentional error
        """
        if self.rng.random() < self.error_rate:
            return self._pick_invalid_op(), True
        return self._pick_valid_op(), False

    def _pick_valid_op(self) -> str:
        child_dirs = self.fs.get_child_dirs()
        child_files = self.fs.get_child_files()
        all_dirs = self.fs.get_all_dirs()
        all_files_count = len(self.fs.files)
        depth = self._depth()
        num_dirs = len(self.fs.dirs)

        pool = []

        # --- Creation ops: stronger early, weaker as fs grows ---
        # mkdir: high early, tapering as tree grows
        mkdir_w = max(3, 25 - num_dirs // 5)
        pool.append(("mkdir", mkdir_w))

        # touch: always useful
        pool.append(("touch", 12))

        # echo > : create files with content
        pool.append(("echo_write", 14))

        # echo >> : prefer when files exist nearby
        append_w = 12 if child_files else 4
        pool.append(("echo_append", append_w))

        # --- Navigation ops ---
        # cd into child dir
        if child_dirs:
            # Bias toward descending when shallow, less when deep
            cd_child_w = max(5, 18 - depth * 2)
            pool.append(("cd_child", cd_child_w))

        # cd ..
        if self.fs.cwd != "/":
            cd_up_w = max(3, 5 + depth * 2)  # more likely when deep
            pool.append(("cd_up", cd_up_w))

        # cd to random absolute path
        if all_dirs:
            pool.append(("cd_abs", 8))

        # --- Query ops ---
        pool.append(("ls", 12))
        pool.append(("pwd", 4))

        # cat: only if files here
        if child_files:
            pool.append(("cat", 14))

        # --- Destructive ---
        if child_files:
            # rm: low weight, just enough to exercise it
            pool.append(("rm", 4))

        ops, weights = zip(*pool)
        return self.rng.choices(ops, weights=weights, k=1)[0]

    def _pick_invalid_op(self) -> str:
        candidates = ["cd_nonexistent", "cat_nonexistent", "rm_nonexistent"]
        child_dirs = self.fs.get_child_dirs()
        child_files = self.fs.get_child_files()
        if child_dirs or child_files:
            candidates.append("mkdir_existing")
        if self.fs.cwd == "/":
            candidates.append("cd_up_from_root")
        return self.rng.choice(candidates)

    def _exec_op(self, op: str, is_error: bool):
        if is_error:
            self._exec_invalid(op)
        else:
            self._exec_valid(op)

    def _exec_valid(self, op: str):
        rng = self.rng

        if op == "ls":
            entries = self.fs.list_dir(self.fs.cwd)
            output = "  ".join(entries) if entries else ""
            self._emit("ls", output if entries else None)

        elif op == "pwd":
            self._emit("pwd", self.fs.cwd)

        elif op == "mkdir":
            name = _random_dirname_rng(rng)
            for _ in range(10):
                if not self.fs.exists(self.fs.resolve(name)):
                    break
                name = _random_dirname_rng(rng)
            err = self.fs.mkdir(name)
            if err:
                self._emit(f"mkdir {name}", is_error=True)
            else:
                self._emit(f"mkdir {name}")

        elif op == "touch":
            name = _random_filename_rng(rng)
            err = self.fs.touch(name)
            if err:
                self._emit(f"touch {name}", is_error=True)
            else:
                self._emit(f"touch {name}")

        elif op == "echo_write":
            name = _random_filename_rng(rng)
            content = _random_content(rng)
            err = self.fs.write_file(name, content)
            if err:
                self._emit(f"echo {content} > {name}", is_error=True)
            else:
                self._emit(f"echo {content} > {name}")

        elif op == "echo_append":
            files = self.fs.get_child_files()
            if files and rng.random() < 0.7:
                name = rng.choice(files)
            else:
                name = _random_filename_rng(rng)
            content = _random_content(rng)
            err = self.fs.append_file(name, content)
            if err:
                self._emit(f"echo {content} >> {name}", is_error=True)
            else:
                self._emit(f"echo {content} >> {name}")

        elif op == "cat":
            files = self.fs.get_child_files()
            if files:
                name = rng.choice(files)
                content, err = self.fs.cat(name)
                if err:
                    self._emit(f"cat {name}", is_error=True)
                else:
                    self._emit(f"cat {name}", content if content else None)

        elif op == "cd_child":
            dirs = self.fs.get_child_dirs()
            if dirs:
                name = rng.choice(dirs)
                self.fs.cd(name)
                self._emit(f"cd {name}")

        elif op == "cd_up":
            self.fs.cd("..")
            self._emit("cd ..")

        elif op == "cd_abs":
            dirs = self.fs.get_all_dirs()
            if dirs:
                target = rng.choice(dirs)
                self.fs.cd(target)
                self._emit(f"cd {target}")

        elif op == "rm":
            files = self.fs.get_child_files()
            if files:
                name = rng.choice(files)
                self.fs.rm(name)
                self._emit(f"rm {name}")

    def _exec_invalid(self, op: str):
        rng = self.rng

        if op == "cd_nonexistent":
            name = "no" + _random_dirname_rng(rng)
            self._emit(f"cd {name}", is_error=True)

        elif op == "cat_nonexistent":
            name = "no" + _random_filename_rng(rng)
            self._emit(f"cat {name}", is_error=True)

        elif op == "mkdir_existing":
            children = self.fs.get_child_dirs() + self.fs.get_child_files()
            if children:
                name = rng.choice(children)
                basename = name.split("/")[-1]
                self._emit(f"mkdir {basename}", is_error=True)
            else:
                self._exec_invalid("cd_nonexistent")

        elif op == "cd_up_from_root":
            self._emit("cd ..", is_error=True)

        elif op == "rm_nonexistent":
            name = "no" + _random_filename_rng(rng)
            self._emit(f"rm {name}", is_error=True)

    def generate(self) -> str:
        """Generate a complete session transcript."""
        num_ops = self._choose_num_ops()
        for _ in range(num_ops):
            op, is_error = self._pick_op()
            self._exec_op(op, is_error)
        return "".join(self.transcript_parts) + "<eos>"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_transcript(transcript: str, tokenizer: BashTokenizer) -> dict:
    """Validate that a transcript can be fully tokenized and round-tripped."""
    try:
        ids = tokenizer.encode(transcript)
        decoded = tokenizer.decode(ids)
        roundtrip_ok = decoded == transcript
    except ValueError as e:
        return {"valid": False, "error": str(e), "num_tokens": 0}

    num_prompts = transcript.count("<prompt>")
    num_outputs = transcript.count("<output>")
    num_errors = transcript.count("<err>")

    return {
        "valid": roundtrip_ok,
        "num_tokens": len(ids),
        "num_exchanges": num_prompts,
        "num_outputs": num_outputs,
        "num_errors": num_errors,
        "error_rate": num_errors / num_prompts if num_prompts > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Multithreaded dataset generation with integrated verification
# ---------------------------------------------------------------------------

def _generate_one_session(args: tuple) -> dict | None:
    """Worker function: generate + tokenize-validate + verify one session.

    Returns a record dict on success, None on failure.
    Uses its own RNG seeded from the provided seed for full reproducibility.
    """
    session_id, seed, min_ops, target_ops, error_rate = args

    gen = SessionGenerator(min_ops=min_ops, target_ops=target_ops,
                           error_rate=error_rate, seed=seed)
    transcript = gen.generate()

    # Tokenization round-trip check
    tokenizer = BashTokenizer()
    info = validate_transcript(transcript, tokenizer)
    if not info["valid"]:
        return None

    # Structural verification: replay against independent filesystem
    from verify import verify_transcript as structural_verify
    total, mismatches = structural_verify(transcript)
    if mismatches:
        return None

    return {
        "transcript": transcript,
        "num_tokens": info["num_tokens"],
        "num_exchanges": info["num_exchanges"],
        "num_errors": info["num_errors"],
        "num_dirs": len(gen.fs.dirs),
        "num_files": len(gen.fs.files),
    }


def generate_dataset(
    num_sessions: int,
    output_path: str,
    min_ops: int = 300,
    target_ops: int = 800,
    error_rate: float = 0.05,
    seed: int = 42,
    workers: int = 1,
):
    """Generate a dataset of verified sessions using multiple threads.

    Each session gets a unique seed derived from the base seed for
    reproducibility. Every session is verified before being written.
    """
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build work items: (session_id, per-session seed, min_ops, target_ops, error_rate)
    work = [(i, seed + i, min_ops, target_ops, error_rate) for i in range(num_sessions)]

    stats = {
        "sessions": 0, "failed": 0,
        "total_tokens": 0, "total_exchanges": 0, "total_errors": 0,
        "total_dirs": 0, "total_files": 0,
        "min_exchanges": float("inf"), "max_exchanges": 0,
    }

    t0 = time.time()

    with open(output_path, "w") as f:
        if workers <= 1:
            # Single-threaded: simpler, good for debugging
            for i, item in enumerate(work):
                result = _generate_one_session(item)
                if result is None:
                    stats["failed"] += 1
                    print(f"WARNING: Session {i} failed verification, regenerating...")
                    # Retry with a different seed
                    result = _generate_one_session((i, seed + num_sessions + i, min_ops, target_ops, error_rate))
                    if result is None:
                        stats["failed"] += 1
                        continue

                f.write(json.dumps(result) + "\n")
                stats["sessions"] += 1
                stats["total_tokens"] += result["num_tokens"]
                stats["total_exchanges"] += result["num_exchanges"]
                stats["total_errors"] += result["num_errors"]
                stats["total_dirs"] += result["num_dirs"]
                stats["total_files"] += result["num_files"]
                stats["min_exchanges"] = min(stats["min_exchanges"], result["num_exchanges"])
                stats["max_exchanges"] = max(stats["max_exchanges"], result["num_exchanges"])

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (num_sessions - i - 1) / rate
                    print(f"  [{i+1}/{num_sessions}] {rate:.1f} sess/s, ETA {eta:.0f}s")
        else:
            # Multithreaded via ProcessPoolExecutor
            completed = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_generate_one_session, item): item[0]
                           for item in work}
                for future in as_completed(futures):
                    session_id = futures[future]
                    result = future.result()
                    if result is None:
                        stats["failed"] += 1
                    else:
                        f.write(json.dumps(result) + "\n")
                        stats["sessions"] += 1
                        stats["total_tokens"] += result["num_tokens"]
                        stats["total_exchanges"] += result["num_exchanges"]
                        stats["total_errors"] += result["num_errors"]
                        stats["total_dirs"] += result["num_dirs"]
                        stats["total_files"] += result["num_files"]
                        stats["min_exchanges"] = min(stats["min_exchanges"], result["num_exchanges"])
                        stats["max_exchanges"] = max(stats["max_exchanges"], result["num_exchanges"])

                    completed += 1
                    if completed % 100 == 0:
                        elapsed = time.time() - t0
                        rate = completed / elapsed
                        eta = (num_sessions - completed) / rate
                        print(f"  [{completed}/{num_sessions}] {rate:.1f} sess/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0

    if stats["sessions"] == 0:
        print("ERROR: No sessions generated!")
        return

    avg_tokens = stats["total_tokens"] / stats["sessions"]
    avg_exchanges = stats["total_exchanges"] / stats["sessions"]
    avg_dirs = stats["total_dirs"] / stats["sessions"]
    avg_files = stats["total_files"] / stats["sessions"]

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {output_path}")
    print(f"  Sessions: {stats['sessions']}  (failed: {stats['failed']})")
    print(f"  Time: {elapsed:.1f}s  ({stats['sessions']/elapsed:.1f} sess/s)")
    print(f"  Workers: {workers}")
    print(f"{'=' * 60}")
    print(f"  Total tokens:     {stats['total_tokens']:>12,}")
    print(f"  Avg tokens/sess:  {avg_tokens:>12,.0f}")
    print(f"  Avg commands:     {avg_exchanges:>12,.0f}")
    print(f"  Command range:    {stats['min_exchanges']}-{stats['max_exchanges']}")
    print(f"  Avg dirs/sess:    {avg_dirs:>12,.0f}")
    print(f"  Avg files/sess:   {avg_files:>12,.0f}")
    print(f"  Error rate:       {stats['total_errors']/stats['total_exchanges']:>11.1%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic bash session data")
    parser.add_argument("--num-sessions", "-n", type=int, default=10,
                        help="Number of sessions to generate")
    parser.add_argument("--output", "-o", type=str, default="data/train.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--min-ops", type=int, default=300,
                        help="Minimum commands per session")
    parser.add_argument("--target-ops", type=int, default=800,
                        help="Target commands per session (median of distribution)")
    parser.add_argument("--error-rate", type=float, default=0.05,
                        help="Fraction of intentionally invalid ops")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="Number of parallel worker processes")
    parser.add_argument("--preview", action="store_true",
                        help="Print one session to stdout instead of writing dataset")
    args = parser.parse_args()

    if args.preview:
        gen = SessionGenerator(min_ops=args.min_ops, target_ops=args.target_ops,
                               error_rate=args.error_rate, seed=args.seed)
        transcript = gen.generate()

        tokenizer = BashTokenizer()
        info = validate_transcript(transcript, tokenizer)

        # Show a chunk
        print(transcript[:4000])
        if len(transcript) > 4000:
            print(f"\n--- TRUNCATED (showing 4000/{len(transcript)} chars) ---")
        print(f"\nStats: {info}")
        print(f"Dirs created: {len(gen.fs.dirs)}, Files created: {len(gen.fs.files)}")
        print(f"Max depth: {max(d.count('/') for d in gen.fs.dirs)}")
    else:
        generate_dataset(
            num_sessions=args.num_sessions,
            output_path=args.output,
            min_ops=args.min_ops,
            target_ops=args.target_ops,
            error_rate=args.error_rate,
            seed=args.seed,
            workers=args.workers,
        )
