"""
Hand-defined tokenizer for bash terminal simulation.
~85 explicit tokens, no BPE/subword tokenization.
"""


class BashTokenizer:
    def __init__(self):
        self._build_vocab()

    def _build_vocab(self):
        # Commands (8)
        commands = ["ls", "cd", "pwd", "touch", "mkdir", "rm", "cat", "echo"]

        # Letters a-z (26)
        letters = [chr(i) for i in range(ord("a"), ord("z") + 1)]

        # Digits 0-9 (10)
        digits = [str(i) for i in range(10)]

        # Punctuation (9)
        punctuation = ["/", ".", "_", "-", ">", ">>", '"', " ", "\n"]

        # Special tokens (5)
        special = ["<prompt>", "<output>", "<err>", "<pad>", "<eos>"]

        # Shell chrome (3)
        chrome = ["@", ":", "#"]

        # Build ordered vocabulary
        all_tokens = commands + letters + digits + punctuation + special + chrome

        self.token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)

        # Convenience IDs
        self.pad_id = self.token_to_id["<pad>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.prompt_id = self.token_to_id["<prompt>"]
        self.output_id = self.token_to_id["<output>"]
        self.err_id = self.token_to_id["<err>"]
        self.newline_id = self.token_to_id["\n"]
        self.space_id = self.token_to_id[" "]

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
    print(f"Tokens: {list(tok.token_to_id.keys())}")
    print()

    # Round-trip tests
    tests = [
        "<prompt>ls\n<output>file1.txt\n",
        "<prompt>cd /home\n<output>\n",
        "<prompt>echo hello > test.txt\n<output>\n",
        '<prompt>echo "hello"\n<output>hello\n',
        "<prompt>cat nofile\n<err>cat: nofile: no such file\n",
        "<prompt>mkdir test_dir\n<output>\n",
        "<prompt>pwd\n<output>/home/user\n",
        "user@host:/home#",
    ]

    all_pass = True
    for t in tests:
        ids = tok.encode(t)
        decoded = tok.decode(ids)
        ok = decoded == t
        if not ok:
            all_pass = False
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {t!r}")
        if not ok:
            print(f"  Got: {decoded!r}")
        print(f"  IDs ({len(ids)}): {ids}")

    print(f"\nAll tests passed: {all_pass}")
