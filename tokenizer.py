"""
Hand-defined tokenizer for NullRoot bash simulation.
Covers filesystem ops, variables, math, control flow, and scripting.
"""


class BashTokenizer:
    def __init__(self):
        self._build_vocab()

    def _build_vocab(self):
        # Filesystem commands (14)
        fs_commands = [
            "ls", "cd", "pwd", "touch", "mkdir", "rm", "cat", "echo",
            "cp", "mv", "head", "wc", "find", "grep",
        ]

        # Programming commands (5)
        prog_commands = ["sh", "export", "exit", "test", "expr"]

        # Control flow keywords (10)
        control = ["if", "then", "else", "fi", "while", "do", "done",
                    "for", "in", "true"]

        # Letters a-z (26)
        letters = [chr(i) for i in range(ord("a"), ord("z") + 1)]

        # Digits 0-9 (10)
        digits = [str(i) for i in range(10)]

        # Punctuation (17)
        punctuation = [
            "/", ".", "_", "-", ">", ">>",
            '"', " ", "\n", "\\",
            "$", "=", "+", "*", ";",
            "?", "!",
        ]

        # Logical operators (2)
        logical = ["&&", "||"]

        # Special tokens (9)
        special = [
            "<prompt>", "<eoi>", "<output>", "<err>", "<eor>",
            "<pad>", "<eos>", "<state>", "<nop>",
        ]

        # Shell chrome (3)
        chrome = ["@", ":", "#"]

        # Build ordered vocabulary
        all_tokens = (fs_commands + prog_commands + control + letters +
                      digits + punctuation + logical + special + chrome)

        self.token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)

        # Convenience IDs
        self.pad_id = self.token_to_id["<pad>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.prompt_id = self.token_to_id["<prompt>"]
        self.eoi_id = self.token_to_id["<eoi>"]
        self.output_id = self.token_to_id["<output>"]
        self.err_id = self.token_to_id["<err>"]
        self.eor_id = self.token_to_id["<eor>"]
        self.newline_id = self.token_to_id["\n"]
        self.space_id = self.token_to_id[" "]
        self.state_id = self.token_to_id["<state>"]
        self.nop_id = self.token_to_id["<nop>"]

        # Precompute multi-char tokens sorted longest-first for greedy matching
        self._multi_char_tokens = sorted(
            [t for t in self.token_to_id if len(t) > 1],
            key=len,
            reverse=True,
        )

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs using greedy longest-match."""
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for tok in self._multi_char_tokens:
                if text[i : i + len(tok)] == tok:
                    tokens.append(self.token_to_id[tok])
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ch = text[i]
                if ch in self.token_to_id:
                    tokens.append(self.token_to_id[ch])
                else:
                    raise ValueError(
                        f"Unknown character {ch!r} (ord={ord(ch)}) at position {i}"
                    )
                i += 1
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        parts = []
        for tid in ids:
            if tid in self.id_to_token:
                parts.append(self.id_to_token[tid])
            else:
                raise ValueError(f"Unknown token ID {tid}")
        return "".join(parts)

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"BashTokenizer(vocab_size={self.vocab_size})"


if __name__ == "__main__":
    tok = BashTokenizer()
    print(f"Vocab size: {tok.vocab_size}")

    # Round-trip tests
    tests = [
        "<prompt>mkdir test_dir<eoi><output><eor>",
        "<prompt>echo hello > test.txt<eoi><output><eor>",
        "<prompt>cat test.txt<eoi><output>hello<eor>",
        # Variables
        "<prompt>x=42<eoi><output><eor>",
        "<prompt>echo $x<eoi><output>42<eor>",
        # Math
        "<prompt>expr 3 + 5<eoi><output>8<eor>",
        # Control flow
        "<prompt>if test true; then echo yes; fi<eoi><output>yes<eor>",
        "<prompt>while true; do echo loop; done<eoi>",
        "<prompt>for x in a b c; do echo $x; done<eoi>",
        # Script
        "<prompt>sh script.sh<eoi><output>hello<eor>",
        # Logical
        "<prompt>true && echo ok<eoi><output>ok<eor>",
        # State
        "<state>@/#/:foo bar#/foo:$x=5$?=0<eor>",
    ]

    all_pass = True
    for t in tests:
        try:
            ids = tok.encode(t)
            decoded = tok.decode(ids)
            ok = decoded == t
            if not ok:
                all_pass = False
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {t[:70]}")
            if not ok:
                print(f"  Got: {decoded[:70]}")
        except ValueError as e:
            all_pass = False
            print(f"[ERR] {t[:70]}")
            print(f"  {e}")

    print(f"\nAll passed: {all_pass}")
    print(f"\nAll tokens ({tok.vocab_size}):")
    for i in range(tok.vocab_size):
        print(f"  {i:3d}: {tok.id_to_token[i]!r}")
